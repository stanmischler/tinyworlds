from contextlib import nullcontext
import torch
import os
from tqdm import tqdm
from einops import rearrange
from models.dynamics import DynamicsModel
from models.flow_dynamics import FlowDynamicsModel
import copy
from datasets.data_utils import visualize_reconstruction, load_data_and_data_loaders
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
    prepare_pipeline_run_root,
    prepare_stage_dirs,
)
from utils.config import DynamicsConfig, load_stage_config_merged
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
    if os.path.isdir(args.video_tokenizer_path):
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
    assert args.action_source in ('lam', 'gt'), f"action_source must be 'lam' or 'gt', got {args.action_source}"
    use_gt_actions = args.action_source == 'gt'
    if use_gt_actions:
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
    assert args.dynamics_type in ('maskgit', 'flow'), f"dynamics_type must be 'maskgit' or 'flow', got {args.dynamics_type}"
    is_flow = args.dynamics_type == 'flow'
    if is_flow:
        dynamics_model = FlowDynamicsModel(
            frame_size=(args.frame_size, args.frame_size),
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            latent_dim=args.latent_dim,
            num_bins=args.num_bins,
            conditioning_dim=conditioning_dim if args.use_actions else 0,
            fm_pred=args.fm_pred,
            fm_shift=args.fm_shift,
            qk_norm=args.qk_norm,
        ).to(args.device)
    else:
        dynamics_model = DynamicsModel(
            frame_size=(args.frame_size, args.frame_size),
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            conditioning_dim=conditioning_dim,
            latent_dim=args.latent_dim,
            num_bins=args.num_bins,
            use_moe=getattr(args, 'use_moe', False),
            num_experts=getattr(args, 'num_experts', 4),
            top_k_experts=getattr(args, 'top_k_experts', 2),
            moe_aux_loss_coeff=getattr(args, 'moe_aux_loss_coeff', 0.01),
            full_last_frame_mask_prob=getattr(args, 'full_last_frame_mask_prob', 0.0),
            action_dropout_prob=getattr(args, 'action_dropout_prob', 0.0),
            mask_mode=args.mask_mode,
            copy_prior=getattr(args, 'copy_prior', False),
        ).to(args.device)
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
    print_param_count_if_main(dynamics_model, "FlowDynamicsModel" if is_flow else "DynamicsModel", is_main)
    if args.compile:
        # mode="default" rather than "reduce-overhead": CUDA-graph mode crashed the latent-actions stage on H100
        # (inductor: "storage data ptrs are not allocated in pool", torch 2.8); same model is compiled here
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
        # precomputed per-row actions (scripts/como_actions.py); the .npy rows are .h5 frames, the dataset may skip the
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

    use_moe = getattr(args, 'use_moe', False)

    for i in tqdm(range(start_step, args.n_updates), initial=start_step, total=args.n_updates, disable=not is_main):
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

            # predict masked frame latents with dynamics model (masking in dynamics model)
            with train_ctx:
                predicted_next_logits, mask_positions, loss = dynamics_model(
                    video_latents, training=True, conditioning=quantized_actions, targets=video_tokens
                )

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
            log_dict = {'train/loss': loss.item()}
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
            if args.use_actions and not use_gt_actions:
                action_indices = latent_action_model.quantizer.get_indices_from_latents(quantized_actions)
                log_action_distribution(action_indices, i, args.n_actions)

        # save model and visualize results
        if i % args.log_interval == 0:
            if is_flow:
                # sample the last frame of 16 training windows from noise (10 Euler steps, EMA weights if on) and decode
                sampler = ema_model if ema_model is not None else raw_dynamics
                sampler.eval()
                with torch.no_grad():
                    n_vis = min(16, video_latents.shape[0])
                    cond = quantized_actions[:n_vis] if quantized_actions is not None else None
                    sampled = sampler.forward_inference(video_latents[:n_vis, :-1], 1, 10, conditioning=cond)  # [n, T, P, L]
                    predicted_frames = video_tokenizer.decoder(sampled)  # [n, T, C, H, W]
                    sample_acc = (video_tokenizer.quantizer.get_indices_from_latents(sampled[:, -1])
                                  == video_tokens[:n_vis, -1]).float().mean().item()
                raw_dynamics.train()
                masked_frames = x.clone()
                masked_frames[:, -1] = 0  # the generated frame
                print(f'\n Step {i} sample token acc (10 Euler steps, train windows): {sample_acc:.3f}')
                if args.use_wandb and is_main:
                    wandb.log({'train/sample_token_acc': sample_acc}, step=i)
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
