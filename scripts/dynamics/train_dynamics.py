from contextlib import nullcontext
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
from einops import rearrange
from models.dynamics import DynamicsModel
import copy
from datasets.data_utils import visualize_reconstruction, load_data_and_data_loaders, held_out_test_h5
from tqdm import tqdm
from einops import rearrange
from utils.wandb_utils import (
    init_wandb, log_training_metrics, log_learning_rate, log_system_metrics, finish_wandb, log_action_distribution
)
from utils.scheduler_utils import create_cosine_scheduler
from utils.utils import (
    readable_timestamp,
    save_training_state,
    load_videotokenizer_from_checkpoint,
    load_latent_actions_from_checkpoint,
    load_dynamics_from_checkpoint,
    maskgit_dynamics_kwargs,
    build_flow_dynamics,
    prepare_pipeline_run_root,
    prepare_stage_dirs,
)
from utils.config import DynamicsConfig, load_stage_config_merged
from models.frequency_loss import focal_log_frequency_loss, frequency_weight
from evaluation.motion import moving_token_mask, revealed_strip_tokens
from datasets.token_shift import random_token_shifts, shift_token_grid, shift_token_mask
import wandb
from dataclasses import asdict
from utils.distributed import init_distributed_from_env, prepare_model_for_distributed, unwrap_model, print_param_count_if_main, cleanup_distributed
from torch.distributed.fsdp import FSDPModule

def main():
    # dynamics config merged with training_config.yaml (training takes priority), plus CLI overrides
    args: DynamicsConfig = load_stage_config_merged(DynamicsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'dynamics.yaml'))
    # DDP setup
    dist_setup = init_distributed_from_env()

    # run save dir if it doesn't exist (running not from full train)
    is_main = dist_setup['is_main']
    run_root = os.environ.get('NG_RUN_ROOT_DIR')
    if not run_root:
        run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
    stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'dynamics')
    if is_main:
        print(f"Dynamics Training")
        print(f"Results will be saved in {stage_dir}")
        print(f"video_tokenizer_path: {args.video_tokenizer_path}")
        print(f"latent_actions_path:  {args.latent_actions_path}")

    # seed init, masking/dropout draws and (via the loader generator below) the batch order
    loader_generator = None
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        loader_generator = torch.Generator().manual_seed(args.seed)

    # load video tokenizer and latent action model
    if args.pixel_space:
        # no tokenizer: the dynamics model predicts patches of pixels (models/pixel_patches.py)
        from models.pixel_patches import PixelPatches
        video_tokenizer = PixelPatches(args.patch_size).to(args.device)
        args.video_tokenizer_path = None  # keep the saved config honest
        args.latent_dim = video_tokenizer.latent_dim
    elif os.path.isdir(args.video_tokenizer_path):
        video_tokenizer, vq_ckpt = load_videotokenizer_from_checkpoint(
            checkpoint_path=args.video_tokenizer_path, 
            device=args.device, 
            is_distributed=dist_setup['is_distributed'],
        )
        video_tokenizer.eval()
        for p in video_tokenizer.parameters():
            p.requires_grad = False
    else:
        raise FileNotFoundError(f"Video tokenizer checkpoint not found at {args.video_tokenizer_path}")
    # the dynamics latents are the tokenizer's: take their width and kind from the checkpoint, not the yaml
    args.continuous_latents = video_tokenizer.continuous
    if not args.pixel_space and args.latent_dim != video_tokenizer.decoder.latent_embed.in_features:
        print(f"latent_dim {args.latent_dim} -> {video_tokenizer.decoder.latent_embed.in_features} (from the tokenizer)")
        args.latent_dim = video_tokenizer.decoder.latent_embed.in_features
    assert args.action_source in ('lam', 'gt'), f"action_source must be 'lam' or 'gt', got {args.action_source}"
    use_gt_actions = args.action_source == 'gt'
    if not args.use_actions:
        # action-free dynamics (e.g. OBJ3D): no action model at all
        latent_action_model, conditioning_dim = None, 0
        args.latent_actions_path = None
    elif use_gt_actions:
        # ground-truth actions from the dataset (Push-T): no latent action model
        from datasets.data_utils import dataset_action_dim
        conditioning_dim = dataset_action_dim(args.dataset)
        assert conditioning_dim is not None, f"action_source=gt needs a dataset with actions, {args.dataset} has none"
        latent_action_model = None
        args.latent_actions_path = None  # keep the saved config honest
    elif args.latent_actions_path and os.path.isdir(args.latent_actions_path):
        latent_action_model, latent_action_ckpt = load_latent_actions_from_checkpoint(
            checkpoint_path=args.latent_actions_path,
            device=args.device,
            is_distributed=dist_setup['is_distributed'],
        )
        unwrap_model(latent_action_model).eval()
        for p in unwrap_model(latent_action_model).parameters():
            p.requires_grad = False
        conditioning_dim = unwrap_model(latent_action_model).action_dim
    else:
        raise FileNotFoundError(f"Latent Action Model checkpoint not found at {args.latent_actions_path}")

    # init dynamics model and optional ckpt load
    assert args.dynamics_type in ('maskgit', 'flow', 'biflow', 'edm', 'regression'), \
        f"dynamics_type must be 'maskgit', 'flow', 'biflow', 'edm' or 'regression', got {args.dynamics_type}"
    # continuous predictors: EMA, test curve, sampled previews (regression is one pass, but shares the flow interface)
    is_flow = args.dynamics_type in ('flow', 'biflow', 'edm', 'regression')
    assert args.dynamics_type != 'edm' or args.continuous_latents, 'edm samples are not rounded: needs continuous latents or pixel_space'
    assert args.dynamics_type != 'edm' or (args.fm_loss == 'v' and args.token_loss == 'mse' and not args.correction
                                           and not args.fm_random_context and args.action_dropout_prob == 0), \
        'edm: plain denoiser loss only (no fm_loss / token_loss / correction / fm_random_context / action dropout)'
    assert args.dynamics_type != 'regression' or (args.continuous_latents and args.fm_loss == 'v' and args.token_loss == 'mse'
                                                  and not args.correction and not args.fm_random_context
                                                  and args.action_dropout_prob == 0), \
        'regression: continuous latents, plain MSE only (no fm_loss / token_loss / correction / fm_random_context / action dropout)'
    assert is_flow or not args.continuous_latents, 'a continuous tokenizer needs dynamics_type flow (MaskGIT predicts codes)'
    assert args.fm_loss in ('v', 'f', 'fv'), f"fm_loss must be 'v', 'f' or 'fv', got {args.fm_loss}"
    assert not (args.correction and args.fm_loss != 'v'), 'correction is only wired for the velocity loss'
    assert args.fm_loss == 'v' or (args.dynamics_type == 'flow' and not args.fm_random_context and args.fm_source != 'bridge'), \
        'the frequency loss needs dynamics_type flow, no fm_random_context (x1_hat is the last frame of the full window), no bridge'
    assert args.token_loss == 'mse' or (args.dynamics_type == 'flow' and not args.fm_random_context), \
        'token_loss needs dynamics_type flow and no fm_random_context (the moving mask is for the last frame of the full window)'
    assert args.dynamics_type != 'biflow' or args.action_dropout_prob == 0, 'Bi-flow has no action dropout'
    assert not (args.token_shift or args.focal_exclude_scroll_strip) or (
        args.dynamics_type in ('flow', 'regression') and args.fm_source != 'bridge' and not args.fm_random_context), \
        'token_shift / focal_exclude_scroll_strip need dynamics_type flow or regression, no bridge, no fm_random_context (loss mask of the last frame)'
    assert not args.focal_exclude_scroll_strip or args.token_loss == 'focal', 'focal_exclude_scroll_strip needs token_loss focal'
    assert not (args.dropout or args.action_token) or args.dynamics_type in ('flow', 'regression'), \
        'dropout / action_token need dynamics_type flow or regression'
    if is_flow:
        dynamics_model = build_flow_dynamics(vars(args), conditioning_dim if args.use_actions else 0).to(args.device)
    else:
        dynamics_model = DynamicsModel(**maskgit_dynamics_kwargs(vars(args), conditioning_dim)).to(args.device)
    if args.init_weights:
        # fine-tune (STA-61): start from another run's weights (EMA if saved), fresh optimizer/scheduler/step
        from utils.utils import EMA_CHECKPOINT, MODEL_CHECKPOINT
        src = os.path.join(args.init_weights, EMA_CHECKPOINT)
        src = src if os.path.exists(src) else os.path.join(args.init_weights, MODEL_CHECKPOINT)
        sd = torch.load(src, map_location=args.device, weights_only=True)
        sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
        missing, unexpected = dynamics_model.load_state_dict(sd, strict=False)
        if args.correction and not any(k.startswith('corrector.') for k in sd):
            # correction fine-tune from a plain flow model: the corrector starts as a copy of it, the gate at 0
            missing = [k for k in missing if not k.startswith('corrector.') and k != 'correction_gate']
            dynamics_model.copy_dynamics_into_corrector()
        assert not unexpected and not missing, (missing, unexpected)
        print(f"Initialised dynamics weights from {src}")
    if args.checkpoint:
        dynamics_model, _ = load_dynamics_from_checkpoint(
            checkpoint_path=args.checkpoint, 
            device=args.device, 
            model=dynamics_model,
            is_distributed=dist_setup['is_distributed'],
        )

    # EMA of the flow model's weights (sampling uses it); kept outside compile/DDP, updated after every optimizer step
    raw_dynamics = dynamics_model
    ema_model = None
    if is_flow and args.ema_decay > 0:
        ema_model = copy.deepcopy(raw_dynamics).eval()
        for p in ema_model.parameters():
            p.requires_grad = False
        if args.checkpoint:
            from utils.utils import EMA_CHECKPOINT
            ema_ckpt = os.path.join(args.checkpoint, EMA_CHECKPOINT)
            if os.path.exists(ema_ckpt):
                ema_model.load_state_dict(torch.load(ema_ckpt, map_location=args.device, weights_only=True))

    # optional DDP, compile, param count, tf32
    print_param_count_if_main(dynamics_model, type(dynamics_model).__name__, is_main)
    if args.compile:
        # mode="default" rather than "reduce-overhead": CUDA-graph mode crashed the latent-actions stage on H100
        # (inductor: "storage data ptrs are not allocated in pool", torch 2.8); same model is compiled here
        if not args.pixel_space:  # PixelPatches is a reshape
            video_tokenizer = torch.compile(video_tokenizer, mode="default", fullgraph=False, dynamic=True)
        if latent_action_model is not None and not args.action_file:
            latent_action_model = torch.compile(latent_action_model, mode="default", fullgraph=False, dynamic=True)
        dynamics_model = torch.compile(dynamics_model, mode="default", fullgraph=False, dynamic=True)
        print("Compiled all models for training.")
    dynamics_model = prepare_model_for_distributed(
        dynamics_model, 
        args.distributed, 
        model_type=dynamics_model.model_type,
        device_mesh=dist_setup['device_mesh'],
    )
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # create optimizer(s) — AdamW or Muon+AdamW split
    from utils.optimizer_utils import create_optimizer
    optimizers = create_optimizer(dynamics_model, args)

    # cosine scheduler for lr warmup and AMP grad scaler
    schedulers = [create_cosine_scheduler(opt, args.n_updates) for opt in optimizers]

    # resume: restore optimizer/scheduler state and continue from the saved step (weights were loaded above)
    start_step = 0
    if args.checkpoint:
        from utils.utils import OPTIMIZER_CHECKPOINT, STATE
        import pathlib
        ckpt_dir = pathlib.Path(args.checkpoint)
        optimizers[0].load_state_dict(torch.load(ckpt_dir / OPTIMIZER_CHECKPOINT, map_location=args.device, weights_only=False))
        state = torch.load(ckpt_dir / STATE, map_location='cpu', weights_only=False)
        if state.get('scheduler_state_dict') is not None:
            schedulers[0].load_state_dict(state['scheduler_state_dict'])
        if len(optimizers) > 1 and (ckpt_dir / 'all_optimizers.pt').exists():
            all_opt = torch.load(ckpt_dir / 'all_optimizers.pt', map_location=args.device, weights_only=False)
            all_sch = torch.load(ckpt_dir / 'all_schedulers.pt', map_location='cpu', weights_only=False)
            for k, (o, sch) in enumerate(zip(optimizers, schedulers)):
                o.load_state_dict(all_opt[f'optimizer_{k}'])
                sch.load_state_dict(all_sch[f'scheduler_{k}'])
        start_step = int(state.get('step') or 0) + 1
        print(f"Resumed from {args.checkpoint}: continuing at step {start_step}/{args.n_updates}")
    train_ctx = torch.amp.autocast(args.device, enabled=True, dtype=torch.bfloat16) if args.amp and not args.distributed.use_fsdp else nullcontext()

    results = {
        'n_updates': 0,
        'dynamics_losses': [],
        'loss_vals': [],
    }

    # init wandb
    if args.use_wandb and is_main:
        run_name = f"dynamics_{readable_timestamp()}"
        init_wandb(args.wandb_project, asdict(args), run_name)

    unwrap_model(dynamics_model).train()

    # dataloader
    data_overrides = {}
    if hasattr(args, 'fps') and args.fps is not None:
        data_overrides['fps'] = args.fps
    if hasattr(args, 'preload_ratio') and args.preload_ratio is not None:
        data_overrides['preload_ratio'] = args.preload_ratio
    for k in ('num_workers', 'pin_memory'):  # dataloader throughput knobs; None = module defaults
        if getattr(args, k, None) is not None:
            data_overrides[k] = getattr(args, k)
    data_overrides['generator'] = loader_generator
    _, _, training_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size_per_gpu,
        num_frames=args.context_length,
        distributed=dist_setup['is_distributed'],
        rank=dist_setup['device_mesh'].get_rank() if dist_setup['device_mesh'] is not None else 0,
        world_size=dist_setup['world_size'],
        **data_overrides,
    )
    if args.action_file:
        # precomputed per-row actions (scripts/actions/como_actions.py); the .npy rows are .h5 frames, the dataset may skip the
        # first load_start_index of them (Zelda: 1000)
        assert not use_gt_actions and hasattr(unwrap_model(latent_action_model), 'standardize'), \
            'action_file needs a CoMo action dir as latent_actions_path'
        import numpy as np
        train_data = training_loader.dataset
        start = train_data.load_start_index
        acts = np.load(args.action_file)[start:start + len(train_data.data)]  # [N, A]
        assert len(acts) == len(train_data.data) and acts.shape[1] == conditioning_dim, (acts.shape, len(train_data.data), conditioning_dim)
        train_data.actions = acts
        if is_main:
            print(f"actions from {args.action_file}: {acts.shape}")
    train_iter = iter(training_loader)

    test_curve = None
    test_h5 = held_out_test_h5(args.dataset, args.test_eval_h5) if args.test_eval_interval > 0 else None
    if test_h5 and is_main:
        supported = is_flow and latent_action_model is not None and not args.distributed.use_fsdp
        assert supported or not args.test_eval_h5, \
            'test_eval_h5 needs a flow dynamics model with an action model, without FSDP (it runs on rank 0 only)'
        assert hasattr(training_loader.dataset, 'frame_skip'), 'test curve: block-split video datasets only (no episode .h5)'
        if supported:
            from evaluation.test_curve import TestCurve
            test_curve = TestCurve(test_h5, context=args.context_length - 1, frame_skip=training_loader.dataset.frame_skip,
                                   limit=args.test_eval_limit)
            print(f"test curve: {len(test_curve.windows)} windows of {test_h5} every {args.test_eval_interval} steps")
        else:
            print("test curve skipped: it needs a flow dynamics model with an action model, without FSDP")

    use_moe = getattr(args, 'use_moe', False)
    end_step = args.stop_step or args.n_updates  # stop_step: stop early on the lr schedule of n_updates
    assert end_step <= args.n_updates, f'stop_step {end_step} > n_updates {args.n_updates}'
    loss_terms = {'v_loss': [], 'f_loss': [], 'lambda': [], 'flow_loss': [], 'corr_loss': []}  # per-step scalars, averaged into train_log.jsonl

    def append_jsonl(name, record):
        # local log next to the checkpoints (read by experiments/focal_frequency/pick_switch.py)
        if is_main:
            with open(os.path.join(stage_dir, name), 'a') as f:
                f.write(json.dumps(record) + '\n')

    for i in tqdm(range(start_step, end_step), initial=start_step, total=end_step, disable=not is_main):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if isinstance(dynamics_model, FSDPModule):
            dynamics_model.set_requires_gradient_sync(False)
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        for micro_batch in range(args.gradient_accumulation_steps):
            try:
                x, gt_actions = next(train_iter)
            except StopIteration:
                train_iter = iter(training_loader)  # reset iterator when epoch ends
                x, gt_actions = next(train_iter)

            x = x.to(args.device, non_blocking=True)  # [batch_size, seq_len, channels, height, width]

            # get video tokens for batch
            if args.continuous_latents:
                video_tokens = None
                video_latents = video_tokenizer.encode(x)  # [B, T, P, L] unit-RMS continuous latents
            else:
                video_tokens = video_tokenizer.tokenize(x) # [B, T, P]
                video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_tokens, dim=-1) # [B, T, P, L]
            if args.use_actions and args.action_file:
                quantized_actions = latent_action_model.standardize(gt_actions.to(args.device, non_blocking=True))  # [B, T - 1, A]
            elif args.use_actions and use_gt_actions:
                quantized_actions = gt_actions.to(args.device, non_blocking=True).float()  # [B, T - 1, A]
            elif args.use_actions:
                quantized_actions = latent_action_model.encode(x)  # [B, T - 1, A]
            else:
                quantized_actions = None

            # token masks of the target frame, from the real frames: moving tokens for token_loss "motion" [B, P];
            # loss_mask [B, P] = tokens in the loss (token shift border and, optionally, the scroll-revealed strip left out)
            extra = {}
            video_latents_raw = video_latents  # [B, T, P, L] unshifted, for the sampled-prediction PNG and metric
            with torch.no_grad():
                if args.token_loss == 'motion' or args.focal_exclude_scroll_strip:
                    unit = (x[:, -2:].float() + 1) / 2  # [B, 2, C, H, W] last context frame and target in [0, 1]
                if args.token_loss == 'motion':
                    extra['token_mask'] = moving_token_mask(unit[:, 0], unit[:, 1], args.patch_size)
                if args.focal_exclude_scroll_strip:
                    extra['loss_mask'] = ~revealed_strip_tokens(unit[:, 0], unit[:, 1], args.patch_size)
                if args.token_shift > 0:
                    # move the whole clip (latents and masks) by the same random number of tokens per sample
                    grid = (args.frame_size // args.patch_size,) * 2  # (Hp, Wp)
                    shifts = random_token_shifts(x.shape[0], args.token_shift, x.device)  # [B, 2]
                    video_latents, inside = shift_token_grid(video_latents, shifts, grid)  # [B, T, P, L], [B, P]
                    for k in ('token_mask', 'loss_mask'):
                        if k in extra:
                            extra[k] = shift_token_mask(extra[k], shifts, grid)
                    extra['loss_mask'] = inside & extra.get('loss_mask', inside)

            # predict masked frame latents with dynamics model (masking in dynamics model)
            with train_ctx:
                predicted_next_logits, mask_positions, loss = dynamics_model(
                    video_latents, training=True, conditioning=quantized_actions, targets=video_tokens, **extra
                )
                if args.correction:
                    for k, v in mask_positions[2].items():
                        loss_terms[k].append(v.item())

                # frequency loss on the decoded frames: L = lambda * scale * f-loss + (1 - lambda) * v-loss
                if args.fm_loss != 'v':
                    lam = 1.0 if args.fm_loss == 'f' else frequency_weight(i, args.f_switch_step, args.f_switch_width)
                    v_loss = loss
                    loss_terms['v_loss'].append(v_loss.item()); loss_terms['lambda'].append(lam)
                    loss = (1 - lam) * v_loss
                    if lam >= 0.01:  # below, the f term is < 1% of the loss: skip the decoder
                        # for a flow model the second output is (x1_hat [B, P, L] predicted clean last frame, s [B] its noise level)
                        x1_hat, s, _ = mask_positions
                        pred_latents = torch.cat([video_latents[:, :-1].float(), x1_hat[:, None]], dim=1)  # [B, T, P, L]
                        # decoded in fp32 (bf16 rounding, ~1e-3 per pixel, is the size of late high-frequency residuals)
                        with torch.autocast(args.device, enabled=False):
                            pred_frame = video_tokenizer.decode(pred_latents)[:, -1]  # [B, C, H, W], gradient flows into x1_hat
                            with torch.no_grad():
                                target_frame = video_tokenizer.decode(video_latents.float())[:, -1]  # [B, C, H, W] tokenizer reconstruction
                        f_loss = focal_log_frequency_loss(pred_frame, target_frame, s)
                        loss_terms['f_loss'].append(f_loss.item())
                        loss = loss + lam * args.f_loss_scale * f_loss

                # add MoE load-balancing auxiliary loss
                moe_aux_scalar = 0.0
                if use_moe:
                    aux_loss = unwrap_model(dynamics_model).transformer.moe_aux_loss()
                    moe_aux_scalar = aux_loss.item()
                    loss = loss + aux_loss

                if isinstance(dynamics_model, FSDPModule):
                    if (micro_batch + 1) % args.gradient_accumulation_steps == 0:
                        dynamics_model.set_requires_gradient_sync(True)

                loss /= args.gradient_accumulation_steps
                loss.backward()

        results['n_updates'] = i
        results['dynamics_losses'].append(loss.detach().cpu())
        results['loss_vals'].append(loss.detach().cpu())

        torch.nn.utils.clip_grad_norm_(unwrap_model(dynamics_model).parameters(), max_norm=1.0)
        for opt in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()
        if ema_model is not None:
            with torch.no_grad():
                ema_p = list(ema_model.parameters())
                torch._foreach_lerp_(ema_p, [p.detach() for p in raw_dynamics.parameters()], 1 - args.ema_decay)

        # wandb logging
        if args.use_wandb and is_main:
            log_dict = {'train/loss': loss.item() * args.gradient_accumulation_steps}  # undo the accumulation scaling
            if args.fm_loss != 'v' or args.correction:
                log_dict.update({f'train/{k}': v[-1] for k, v in loss_terms.items() if v})
            if args.correction:
                gate = raw_dynamics.correction_gate.detach()  # [L]
                log_dict.update({'train/gate_mean': gate.mean().item(), 'train/gate_min': gate.min().item(),
                                 'train/gate_max': gate.max().item()})
            if use_moe:
                log_dict['train/moe_aux_loss'] = moe_aux_scalar
                # per-expert utilization across blocks
                expert_util = unwrap_model(dynamics_model).transformer.moe_expert_utilization()
                for block_idx, fracs in expert_util.items():
                    for expert_i, frac in enumerate(fracs.tolist()):
                        log_dict[f'moe/block{block_idx}_expert{expert_i}_frac'] = frac
            wandb.log(log_dict, step=i)
            log_system_metrics(i)
            log_learning_rate(optimizers[0], i)
            if args.use_actions and not use_gt_actions and getattr(latent_action_model, 'centroids', torch.ones(1)).numel() > 0:
                action_indices = latent_action_model.quantizer.get_indices_from_latents(quantized_actions)
                log_action_distribution(action_indices, i, args.n_actions)

        if test_curve is not None and (i % args.test_eval_interval == 0 or i == end_step - 1):
            test_metrics, test_image = test_curve(ema_model if ema_model is not None else raw_dynamics, video_tokenizer,
                                                  latent_action_model, args.device)
            print(f'\n Step {i} ' + '  '.join(f'{k} {v:.4f}' for k, v in test_metrics.items()))
            test_image.save(os.path.join(visualizations_dir, f'test_step_{i}.png'))
            if args.use_wandb:
                wandb.log({**test_metrics, 'test/image': wandb.Image(test_image)}, step=i)
            append_jsonl('test_curve.jsonl', {'step': i, **test_metrics})

        # save model and visualize results
        if i % args.log_interval == 0 or i == end_step - 1:  # also save the final step
            if args.fm_loss != 'v' or args.correction:  # interval means of the loss terms (the f-loss is absent where the decoder was skipped)
                peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
                append_jsonl('train_log.jsonl', {'step': i, **{k: sum(v) / len(v) for k, v in loss_terms.items() if v},
                                                 'peak_gpu_gb': round(peak_gb, 1)})
                loss_terms = {k: [] for k in loss_terms}
            if is_flow:
                # sample the last frame of 16 training windows from the model's source (10 Euler steps, one pass for regression; EMA weights if on) and decode
                sampler = ema_model if ema_model is not None else raw_dynamics
                sampler.eval()
                with torch.no_grad():
                    n_vis = min(16, video_latents_raw.shape[0])
                    cond = quantized_actions[:n_vis] if quantized_actions is not None else None
                    sampled = sampler.forward_inference(video_latents_raw[:n_vis, :-1], 1, 10, conditioning=cond)  # [n, T, P, L]
                    predicted_frames = video_tokenizer.decode(sampled)  # [n, T, C, H, W]
                    if args.continuous_latents:  # no codes: latent MSE of the sample instead of token accuracy
                        sample_metric = ('sample_latent_mse', F.mse_loss(sampled[:, -1].float(), video_latents_raw[:n_vis, -1].float()).item())
                    else:
                        sample_metric = ('sample_token_acc', (video_tokenizer.quantizer.get_indices_from_latents(sampled[:, -1])
                                                              == video_tokens[:n_vis, -1]).float().mean().item())
                raw_dynamics.train()
                masked_frames = x.clone()
                masked_frames[:, -1] = 0  # the generated frame
                print(f'\n Step {i} {sample_metric[0]} ({"1 pass" if args.dynamics_type == "regression" else "10 Euler steps"}, train windows): {sample_metric[1]:.3f}')
                if args.use_wandb and is_main:
                    wandb.log({f'train/{sample_metric[0]}': sample_metric[1]}, step=i)
            elif args.use_wandb:
                predicted_next_indices = torch.argmax(predicted_next_logits, dim=-1)
                predicted_next_latents = video_tokenizer.quantizer.get_latents_from_indices(predicted_next_indices, dim=-1)
                with torch.no_grad():
                    predicted_frames = video_tokenizer.decoder(predicted_next_latents[:16]) # [B, T, C, H, W]

                # convert mask_positions to patch-level mask for visualization
                B, T, P = mask_positions.shape
                patch_size = args.patch_size
                H, W = args.frame_size, args.frame_size
                pixel_mask = torch.zeros(B, T, H, W, device=mask_positions.device)
                # for each pixel patch, mask if equivalent token is masked
                for b in range(B):
                    for t in range(T):
                        for p in range(P):
                            if mask_positions[b, t, p]:
                                patch_row = (p // (W // patch_size)) * patch_size
                                patch_col = (p % (W // patch_size)) * patch_size
                                pixel_mask[b, t, patch_row:patch_row+patch_size, patch_col:patch_col+patch_size] = 1 # assigning 1 to the patch in the mask of dim [1, 1, Hp, Wp]
                pixel_mask_expanded = rearrange(pixel_mask, 'b t h w -> b t 1 h w')
                masked_frames = x * (1 - pixel_mask_expanded)
            
            hyperparameters = args.__dict__
            ckpt_path = save_training_state(dynamics_model, optimizers[0], schedulers[0], hyperparameters, checkpoints_dir, prefix='dynamics', step=i)
            if ema_model is not None:
                from utils.utils import EMA_CHECKPOINT
                torch.save(ema_model.state_dict(), os.path.join(ckpt_path, EMA_CHECKPOINT))
            # save secondary optimizer/scheduler state when using split optimizers (Muon+AdamW)
            if len(optimizers) > 1:
                import pathlib
                torch.save(
                    {f'optimizer_{i}': o.state_dict() for i, o in enumerate(optimizers)},
                    pathlib.Path(ckpt_path) / 'all_optimizers.pt',
                )
                torch.save(
                    {f'scheduler_{i}': s.state_dict() for i, s in enumerate(schedulers)},
                    pathlib.Path(ckpt_path) / 'all_schedulers.pt',
                )
            if is_main:
                save_path = os.path.join(visualizations_dir, f'dynamics_prediction_step_{i}.png')
                visualize_reconstruction(masked_frames[:16].cpu(), predicted_frames[:16].cpu(), save_path)

            print('\n Step', i, 'Loss:', torch.mean(torch.stack(results["loss_vals"][-args.log_interval:])).item())

    # finish wandb
    if args.use_wandb and is_main:
        finish_wandb()
    cleanup_distributed(dist_setup['is_distributed'])

if __name__ == "__main__":
    main()
