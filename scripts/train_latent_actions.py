from contextlib import nullcontext
import torch
import os
from models.latent_actions import LatentActionModel
from datasets.data_utils import load_data_and_data_loaders, visualize_reconstruction
from utils.scheduler_utils import create_cosine_scheduler
from tqdm import tqdm
import wandb
from utils.utils import readable_timestamp, save_training_state, prepare_stage_dirs, prepare_pipeline_run_root
from utils.config import LatentActionsConfig, load_stage_config_merged
from utils.utils import save_training_state, load_latent_actions_from_checkpoint
from utils.wandb_utils import init_wandb, log_system_metrics, finish_wandb, log_action_distribution, log_learning_rate
from dataclasses import asdict
from utils.distributed import init_distributed_from_env, prepare_model_for_distributed, unwrap_model, print_param_count_if_main, cleanup_distributed
from torch.distributed.fsdp import FSDPModule

def main():
    # latent actions config merged with training_config.yaml (training takes priority), plus CLI overrides
    args: LatentActionsConfig = load_stage_config_merged(LatentActionsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'latent_actions.yaml'))

    # DDP setup
    dist_setup = init_distributed_from_env()

    # run save dir if it doesn't exist (running not from full train)
    timestamp = readable_timestamp()
    run_root = os.environ.get('NG_RUN_ROOT_DIR')
    if not run_root:
        run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
    is_main = dist_setup['is_main']
    stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'latent_actions')
    if is_main:
        print(f"Latent Actions Training")
        print(f'Results will be saved in {stage_dir}')

    # dataloader
    data_overrides = {}
    if hasattr(args, 'fps') and args.fps is not None:
        data_overrides['fps'] = args.fps
    if hasattr(args, 'preload_ratio') and args.preload_ratio is not None:
        data_overrides['preload_ratio'] = args.preload_ratio
    for k in ('num_workers', 'pin_memory'):  # dataloader throughput knobs; None = module defaults
        if getattr(args, k, None) is not None:
            data_overrides[k] = getattr(args, k)
    training_data, validation_data, training_loader, validation_loader, x_train_var = load_data_and_data_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size_per_gpu,
        num_frames=args.context_length,
        distributed=dist_setup['is_distributed'],
        rank=dist_setup['device_mesh'].get_rank() if dist_setup['device_mesh'] is not None else 0,
        world_size=dist_setup['world_size'],
        **data_overrides,
    )

    if args.ot_plans:
        # OT-conditioned LAM: per-pair plans of the training .h5 (same frame indexing, same frame_skip)
        import numpy as np
        # the .npz rows are .h5 frames; the dataset may skip the first load_start_index of them (Zelda: 1000)
        start = training_data.load_start_index
        with np.load(args.ot_plans) as z:
            plans = {'sigma': z['sigma'][start:], 'created': z['created'][start:]}
            assert int(z['gap']) == training_data.frame_skip, (int(z['gap']), training_data.frame_skip)
        assert len(plans['sigma']) == len(training_data.data), (len(plans['sigma']), len(training_data.data))
        training_data.ot_plans = plans
    assert bool(args.ot_plans) == (args.ot_encoder or args.ot_decoder != 'none'), 'ot_plans needs ot_encoder / ot_decoder and back'
    if args.aux_labels:
        # pseudo-label head (STA-35 i5): per-pair labels of the training .h5, indexed like ot_plans
        import numpy as np
        assert not args.ot_plans, 'aux_labels and ot_plans share the second batch item'
        start = training_data.load_start_index
        with np.load(args.aux_labels) as z:
            labels = z[args.aux_label_key][start:].astype(np.int64)
            assert int(z['gap']) == training_data.frame_skip, (int(z['gap']), training_data.frame_skip)
        assert len(labels) >= len(training_data.data), (len(labels), len(training_data.data))  # preload_ratio < 1 loads a prefix
        training_data.aux_labels = labels[:len(training_data.data)]
        if is_main:
            print('aux labels:', args.aux_labels, args.aux_label_key, np.bincount(labels[labels >= 0]).tolist())
    assert bool(args.aux_labels) == (args.aux_label_weight > 0), 'aux_labels needs aux_label_weight > 0 and back'

    # init model and optional ckpt load
    model = LatentActionModel(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        n_actions=args.n_actions,
        decoder_keep_rate=args.decoder_keep_rate,
        decoder_residual=args.decoder_residual,
        entropy_loss_weight=args.entropy_loss_weight,
        entropy_sample_weight=args.entropy_sample_weight,
        encoder_pooling=args.encoder_pooling,
        recon_change_weight=args.recon_change_weight,
        continuous_actions=args.continuous_actions,
        action_kl_capacity=args.action_kl_capacity,
        action_kl_weight=args.action_kl_weight,
        action_fixed_noise=args.action_fixed_noise,
        encoder_input=args.encoder_input,
        decoder_hint=args.decoder_hint,
        decoder_warp=args.decoder_warp,
        decoder_warp_radius=args.decoder_warp_radius,
        decoder_warp_entropy_weight=args.decoder_warp_entropy_weight,
        decoder_warp_entropy_ramp=args.decoder_warp_entropy_ramp,
        decoder_warp_wta=args.decoder_warp_wta,
        wta_sinkhorn_eps=args.wta_sinkhorn_eps,
        wta_encoder_weight=args.wta_encoder_weight,
        decoder_warp_local=args.decoder_warp_local,
        decoder_warp_local_mask=args.decoder_warp_local_mask,
        wta_balance=args.wta_balance,
        decoder_warp_local_gate=args.decoder_warp_local_gate,
        decoder_warp_local_blur=args.decoder_warp_local_blur,
        wta_kernel_repulsion=args.wta_kernel_repulsion,
        ot_encoder=args.ot_encoder,
        ot_decoder=args.ot_decoder,
        aux_label_weight=args.aux_label_weight,
        aux_label_classes=args.aux_label_classes,
        aux_label_target=args.aux_label_target,
    ).to(args.device)
    if args.checkpoint:
        model, _ = load_latent_actions_from_checkpoint(
            args.checkpoint, 
            args.device,
            model,
            dist_setup['is_distributed'],
        )

    # optional DDP, compile, param count, tf32
    print_param_count_if_main(model, "LatentActionModel", is_main)
    if args.compile:
        # mode="default" rather than "reduce-overhead": CUDA-graph mode crashed this stage on H100 at step 1
        # (inductor: "storage data ptrs are not allocated in pool", torch 2.8)
        model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
    model = prepare_model_for_distributed(
        model, 
        args.distributed, 
        model_type=model.model_type, 
        device_mesh=dist_setup['device_mesh'],
    )
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # create optimizer(s) — AdamW or Muon+AdamW split
    from utils.optimizer_utils import create_optimizer
    optimizers = create_optimizer(model, args)

    # cosine scheduler for lr warmup and AMP
    schedulers = [create_cosine_scheduler(opt, args.n_updates) for opt in optimizers]
    train_ctx = torch.amp.autocast(args.device, enabled=True, dtype=torch.bfloat16) if args.amp and not args.distributed.use_fsdp else nullcontext()

    results = {
        'n_updates': 0,
        'loss_vals': [],
    }

    # init wandb
    if args.use_wandb and is_main:
        cfg = asdict(args)
        cfg.update({'timestamp': timestamp})
        run_name = f"latent_actions_{timestamp}"
        init_wandb(args.wandb_project, cfg, run_name)

    unwrap_model(model).train()

    train_iter = iter(training_loader)
    for i in tqdm(range(args.n_updates), disable=not is_main):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if isinstance(model, FSDPModule):
            model.set_requires_gradient_sync(False)
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        for micro_batch in range(args.gradient_accumulation_steps):
            try:
                (x, ot) = next(train_iter)
            except StopIteration:
                train_iter = iter(training_loader)
                (x, ot) = next(train_iter)

            x = x.to(args.device, non_blocking=True)
            side = ot.to(args.device, non_blocking=True) if (args.ot_plans or args.aux_labels) else None
            ot = side if args.ot_plans else None  # [B, T-1, 2, P]
            aux = side if args.aux_labels else None  # [B, T-1] pseudo-labels

            with train_ctx:
                loss, pred_frames = model(x, ot=ot, aux=aux)
                loss /= args.gradient_accumulation_steps
                if isinstance(model, FSDPModule):
                    if (micro_batch + 1) % args.gradient_accumulation_steps == 0:
                        model.set_requires_gradient_sync(True)
                loss.backward()

        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
        for opt in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()

        results['n_updates'] = i
        results['loss_vals'].append(loss.detach().cpu())

        if args.use_wandb and is_main:
            wandb.log({
                'train/loss': loss.item(),
            }, step=i)
            log_system_metrics(i)
            log_learning_rate(optimizers[0], i)
  
        # save model and visualize results
        if i % args.log_interval == 0:
            if args.use_wandb:
                with torch.no_grad():
                    actions = unwrap_model(model).pre_quant(x, ot)  # continuous actions: codes = sign pattern of the mean
                    actions_quantized = unwrap_model(model).quantizer(actions)
                    idx = unwrap_model(model).quantizer.get_indices_from_latents(actions_quantized)
                    codebook_usage = idx.unique().numel() / unwrap_model(model).quantizer.codebook_size
                    z_e_var = actions.var(dim=0, unbiased=False).mean().item()
                    counts = torch.bincount(idx.flatten(), minlength=unwrap_model(model).quantizer.codebook_size).float()
                    freq = counts / counts.sum()
                    code_entropy = -(freq[freq > 0] * freq[freq > 0].log()).sum().item()  # nats, hard codes
                    saturated = (torch.tanh(actions).abs() > 0.99).float().mean().item()
                    pred_frames_var = pred_frames.var(dim=0, unbiased=False).mean().item()

            if args.use_wandb and is_main:
                wandb.log({
                    "latent_actions/codebook_usage": codebook_usage,
                    "latent_actions/encoder_variance": z_e_var,
                    "latent_actions/decoder_variance": pred_frames_var,
                    "latent_actions/code_entropy": code_entropy,
                    "latent_actions/saturated_fraction": saturated,
                    **({"latent_actions/action_kl": unwrap_model(model).action_kl(x, ot).item()} if args.continuous_actions else {}),
                }, step=i)
                log_action_distribution(idx, i, args.n_actions)

            hyperparameters = vars(args)
            save_training_state(model, optimizers[0], None, hyperparameters, checkpoints_dir, prefix='latent_actions', step=i)
            if is_main:
                save_path = os.path.join(visualizations_dir, f'reconstructions_latent_actions_step_{i}.png')
                visualize_reconstruction(x, pred_frames, save_path)
            
                print('\n Step', i, 'Loss:', loss.item(), 'Codebook Usage:', codebook_usage, 'Encoder Variance:', z_e_var, 'Decoder Variance:', pred_frames_var, 'Code Entropy:', code_entropy, 'Saturated:', saturated,
                      *(('Aux acc:', unwrap_model(model).last_aux_acc.item()) if args.aux_labels else ()),
                      *(('WTA encoder agreement:', unwrap_model(model).last_wta_agree.item(), 'Kernel overlap:', unwrap_model(model).last_kernel_overlap.item()) if unwrap_model(model).warp_wta else ()))

    # finish wandb
    if args.use_wandb and is_main:
        finish_wandb()
    cleanup_distributed(dist_setup['is_distributed'])

if __name__ == "__main__":
    main()
