from pathlib import Path
import time
import glob
import subprocess
import os
import re
from typing import Optional

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP

MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIMIZER_CHECKPOINT = "optim_state_dict.pt"
STATE = "state.pt"
EMA_CHECKPOINT = "ema_state_dict.pt"  # flow dynamics (STA-28): EMA weights, used for sampling

def readable_timestamp():
    """Generate a sortable timestamp for filenames (no weekday)."""
    return time.strftime("%Y_%m_%d_%H_%M_%S")

def find_latest_checkpoint(base_dir, model_name, run_root_dir: Optional[str] = None, stage_name: Optional[str] = None):
    """Find latest checkpoint.
    If run_root_dir (and optional stage_name) are provided, search only under
    <run_root_dir>/<stage_name>/checkpoints (or <run_root_dir>/**/checkpoints if stage_name None).
    Otherwise, fall back to project-wide model type-based search.
    Newest run dir first, then highest step within it.
    If the newest run root has none for this model, keep searching older runs.
    """
    def collect_checkpoint_paths(roots, model_name):
        alias_map = {
            'video_tokenizer': ['video_tokenizer'],
            'latent_actions': ['latent_actions', 'lam', 'actions', 'action_tokenizer'],
            'dynamics': ['dynamics'],
        }
        aliases = alias_map.get(model_name, [model_name])
        candidates = []
        seen = set()

        def add_candidate(path: str) -> None:
            norm = os.path.normpath(path)
            if norm in seen:
                return
            if os.path.isdir(norm):
                # require at least state or model file
                state_file = os.path.join(norm, STATE)
                model_file = os.path.join(norm, MODEL_CHECKPOINT)
                if os.path.isfile(state_file) or os.path.isfile(model_file):
                    candidates.append(norm)
                    seen.add(norm)
            else:
                _, ext = os.path.splitext(norm)
                if ext in ('.pt', '.pth'):
                    candidates.append(norm)
                    seen.add(norm)

        patterns = []
        for alias in aliases:
            patterns.append(f"*{alias}_step_*")
            patterns.append(f"*{alias}_checkpoint_*")

        for root in roots:
            for pat in patterns:
                search_pattern = os.path.join(root, "**", pat)
                for match in glob.glob(search_pattern, recursive=True):
                    add_candidate(match)
        return candidates

    def run_dir_of(path: str) -> str:
        # Walk up until we reach the 'checkpoints' directory, then return its parent (the stage dir)
        d = path if os.path.isdir(path) else os.path.dirname(path)
        while d and os.path.basename(d) != 'checkpoints':
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        # If we found 'checkpoints', return its parent; otherwise, fallback to two-level up
        if d and os.path.basename(d) == 'checkpoints':
            return os.path.dirname(d)
        # Fallback: use the directory containing the checkpoint path (or its parent)
        candidate_dir = path if os.path.isdir(path) else os.path.dirname(path)
        return candidate_dir

    def project_wide_search():
        # Fallback to model_type/results search in repository
        # Allow multiple possible directory names per model
        model_type_dirs = {
            'video_tokenizer': ['video_tokenizer'],
            'latent_actions': ['latent_actions', 'lam', 'actions', 'action_tokenizer'],
            'dynamics': ['dynamics',],
        }
        dirs = model_type_dirs.get(model_name)
        if not dirs:
            roots = [os.path.join(base_dir, 'results', '**', 'checkpoints')]
        else:
            roots = [os.path.join(base_dir, 'results', '**', d, 'checkpoints') for d in dirs]
        files = collect_checkpoint_paths(roots, model_name)
        if not files:
            # Generic fallback: search all checkpoints regardless of stage dir name
            generic_roots = [os.path.join(base_dir, 'results', '**', 'checkpoints')]
            files = collect_checkpoint_paths(generic_roots, model_name)
        if not files:
            raise Exception(f"No checkpoint files found for {model_name}")
        run_dir_to_files = {}
        for p in files:
            rd = run_dir_of(p)
            run_dir_to_files.setdefault(rd, []).append(p)
        newest_run_dir = max(run_dir_to_files.keys(), key=lambda d: os.path.getctime(d))
        candidate_files = run_dir_to_files[newest_run_dir]
        return candidate_files

    if run_root_dir is not None:
        if stage_name:
            roots = [os.path.join(run_root_dir, stage_name, 'checkpoints')]
        else:
            roots = [os.path.join(run_root_dir, '**', 'checkpoints')]
        files = collect_checkpoint_paths(roots, model_name)
        if not files:
            # Fallback: search project-wide older runs until found
            candidate_files = project_wide_search()
        else:
            # Group by run dir within the provided root and choose newest run dir
            run_dir_to_files = {}
            for p in files:
                rd = run_dir_of(p)
                run_dir_to_files.setdefault(rd, []).append(p)
            newest_run_dir = max(run_dir_to_files.keys(), key=lambda d: os.path.getctime(d))
            candidate_files = run_dir_to_files[newest_run_dir]
    else:
        candidate_files = project_wide_search()

    def extract_step(path: str) -> int:
        fname = os.path.basename(path)
        m = re.search(r"_step_(\d+)", fname)
        return int(m.group(1)) if m else -1

    candidate_files.sort(key=lambda p: (extract_step(p), os.path.getctime(p)))
    return candidate_files[-1]

def run_command(cmd, description):
    # empirical max dataLoader throughput settings (I used 1-6 H100s)
    env = os.environ.copy()
    env.setdefault("NG_NUM_WORKERS", str(max(2, (os.cpu_count() or 4) - 2)))
    env.setdefault("NG_PREFETCH_FACTOR", "4")
    env.setdefault("NG_PIN_MEMORY", "1")
    env["NG_PERSISTENT_WORKERS"] = "0"
    env.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False
    except KeyboardInterrupt:
        return False

def save_training_state(model, optimizer, scheduler, config, checkpoints_dir, prefix, step):
    """Save a checkpoint with model/optimizer/scheduler and the exact config.
    The filename includes the global step and a timestamp for uniqueness.
    """
    import torch
    ts = readable_timestamp()
    if isinstance(model, (FSDPModule, DDP)):
        state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        optimizer_state_dict = get_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
    else:
        # Avoid saving the model with _orig_mod prefix if it's compiled
        state_dict = getattr(model, '_orig_mod', model).state_dict()
        optimizer_state_dict = optimizer.state_dict()
    state = {
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'config': config,
        'step': int(step) if step is not None else None,
        'timestamp': ts,
    }
    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, f"{prefix}_step_{int(step) if step is not None else 0}")
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(state_dict, Path(ckpt_path) / MODEL_CHECKPOINT)
    torch.save(optimizer_state_dict, Path(ckpt_path) / OPTIMIZER_CHECKPOINT)
    torch.save(state, Path(ckpt_path) / STATE)
    return ckpt_path


def _load_checkpoint_files(checkpoint_path):
    """-> (model state dict, state.pt dict, its config dict or {})."""
    import torch
    model_sd = torch.load(Path(checkpoint_path) / MODEL_CHECKPOINT, map_location='cpu', weights_only=True)
    state_cfg = torch.load(Path(checkpoint_path) / STATE, map_location='cpu', weights_only=False)
    return model_sd, state_cfg, state_cfg.get('config', {}) or {}


def _set_weights(model, model_sd, device, is_distributed):
    set_model_state_dict(
        model=model,
        model_state_dict=model_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=is_distributed,
        ),
    )
    return model.to(device)


# model constructor kwargs from a config dict: a checkpoint's saved config (defaults cover keys older runs lack)
# or vars(args) of the stage dataclass in the train scripts (every key present, so the defaults never apply)
def video_tokenizer_kwargs(cfg):
    frame_size = cfg.get('frame_size', 128)
    return {
        'frame_size': (frame_size, frame_size),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
        'latent_dim': cfg.get('latent_dim', 6),
        'num_bins': cfg.get('num_bins', 4),
        'per_frame': cfg.get('per_frame', False),
        'bottleneck': cfg.get('bottleneck', 'fsq'),
        'latent_noise': cfg.get('latent_noise', 0.8),
        'kl_weight': cfg.get('kl_weight', 1e-6),
    }


def latent_action_kwargs(cfg):
    frame_size = cfg.get('frame_size', 128)
    return {
        'frame_size': (frame_size, frame_size),
        'n_actions': cfg.get('n_actions', 8),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
        'decoder_keep_rate': cfg.get('decoder_keep_rate', 0.0),
        'decoder_residual': cfg.get('decoder_residual', False),
        'entropy_loss_weight': cfg.get('entropy_loss_weight', 0.0),
        'entropy_sample_weight': cfg.get('entropy_sample_weight', 0.1),
        'encoder_pooling': cfg.get('encoder_pooling', 'mean'),
        'recon_change_weight': cfg.get('recon_change_weight', 0.0),
        'continuous_actions': cfg.get('continuous_actions', False),
        'action_kl_capacity': cfg.get('action_kl_capacity', 2.0794),
        'action_kl_weight': cfg.get('action_kl_weight', 1.0),
        'action_fixed_noise': cfg.get('action_fixed_noise', False),
        'encoder_input': cfg.get('encoder_input', 'frames'),
        'decoder_hint': cfg.get('decoder_hint', 'none'),
        'decoder_warp': cfg.get('decoder_warp', 'none'),
        'decoder_warp_radius': cfg.get('decoder_warp_radius', 12),
        'decoder_warp_entropy_weight': cfg.get('decoder_warp_entropy_weight', 0.0),
        'decoder_warp_entropy_ramp': cfg.get('decoder_warp_entropy_ramp', 3000),
        'decoder_warp_wta': cfg.get('decoder_warp_wta', False),
        'wta_sinkhorn_eps': cfg.get('wta_sinkhorn_eps', 0.05),
        'wta_encoder_weight': cfg.get('wta_encoder_weight', 1.0),
        'decoder_warp_local': cfg.get('decoder_warp_local', 0),
        'decoder_warp_local_mask': cfg.get('decoder_warp_local_mask', False),
        'wta_balance': cfg.get('wta_balance', 1.0),
        'decoder_warp_local_gate': cfg.get('decoder_warp_local_gate', 0.0),
        'decoder_warp_local_blur': cfg.get('decoder_warp_local_blur', 1),
        'wta_kernel_repulsion': cfg.get('wta_kernel_repulsion', 0.0),
        'ot_encoder': cfg.get('ot_encoder', False),
        'ot_decoder': cfg.get('ot_decoder', 'none'),
        'aux_label_weight': cfg.get('aux_label_weight', 0.0),
        'aux_label_classes': cfg.get('aux_label_classes', 9),
        'aux_label_target': cfg.get('aux_label_target', 'latent'),
    }


def maskgit_dynamics_kwargs(cfg, conditioning_dim):
    frame_size = cfg.get('frame_size', 128)
    return {
        'frame_size': (frame_size, frame_size),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
        'conditioning_dim': conditioning_dim,
        'latent_dim': cfg.get('latent_dim', 6),
        'num_bins': cfg.get('num_bins', 4),
        'use_moe': cfg.get('use_moe', False),
        'num_experts': cfg.get('num_experts', 4),
        'top_k_experts': cfg.get('top_k_experts', 2),
        'moe_aux_loss_coeff': cfg.get('moe_aux_loss_coeff', 0.01),
        'full_last_frame_mask_prob': cfg.get('full_last_frame_mask_prob', 0.0),
        'action_dropout_prob': cfg.get('action_dropout_prob', 0.0),
        'mask_mode': cfg.get('mask_mode', 'maskgit'),
        'copy_prior': cfg.get('copy_prior', False),
    }


def flow_dynamics_kwargs(cfg, action_dim):
    # FlowDynamicsModel / BiFlowDynamicsModel kwargs (STA-28, STA-61); action_dim = FiLM action width (0 = no actions)
    frame_size = cfg.get('frame_size', 128)
    kwargs = {
        'frame_size': (frame_size, frame_size),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
        'latent_dim': cfg.get('latent_dim', 6),
        'num_bins': cfg.get('num_bins', 4),
        'conditioning_dim': action_dim,
        'qk_norm': cfg.get('qk_norm', True),
        'round_latents': not cfg.get('continuous_latents', False),
    }
    if cfg.get('dynamics_type') == 'biflow':
        kwargs.update(bf_alpha_max=cfg.get('bf_alpha_max', 1.0), bf_eps=cfg.get('bf_eps', 0.1))
    else:
        kwargs.update(fm_pred=cfg.get('fm_pred', 'v'), fm_shift=cfg.get('fm_shift', 1.0),
                      fm_source=cfg.get('fm_source', 'noise'), fm_source_noise=cfg.get('fm_source_noise', 0.0))
    return kwargs


def build_flow_dynamics(cfg, action_dim):
    from models.flow_dynamics import FlowDynamicsModel, BiFlowDynamicsModel
    cls = BiFlowDynamicsModel if cfg.get('dynamics_type') == 'biflow' else FlowDynamicsModel
    return cls(**flow_dynamics_kwargs(cfg, action_dim))


def load_videotokenizer_from_checkpoint(checkpoint_path, device, model = None, is_distributed = False):
    """Instantiate VideoTokenizer from a checkpoint's saved config and load weights."""
    from models.video_tokenizer import VideoTokenizer
    model_sd, state_cfg, cfg = _load_checkpoint_files(checkpoint_path)
    if model is None:
        model = VideoTokenizer(**video_tokenizer_kwargs(cfg))
    return _set_weights(model, model_sd, device, is_distributed), state_cfg


def load_latent_actions_from_checkpoint(checkpoint_path, device, model = None, is_distributed = False):
    """Instantiate LatentActionModel from a checkpoint's saved config and load weights."""
    from models.latent_actions import LatentActionModel
    if not (Path(checkpoint_path) / MODEL_CHECKPOINT).exists() and (Path(checkpoint_path) / STATE).exists():
        return load_como_actions(checkpoint_path, device)  # STA-42 action dir (scripts/actions/como_actions.py): no weights of its own
    model_sd, state_cfg, cfg = _load_checkpoint_files(checkpoint_path)
    if state_cfg.get('model_type') == 'laof':  # LAOF (models/laof.py, experiments/laof/train_laof.py): kwargs saved verbatim
        from models.laof import LAOF
        model = LAOF(**state_cfg['model_kwargs']) if model is None else model
        model.load_state_dict(model_sd)
        return model.to(device), state_cfg
    if cfg.get('model_type') == 'como':  # CoMo motion IDM (scripts/actions/train_como.py): eval adapter with the same encode()
        return load_como_from_checkpoint(checkpoint_path, device)
    if model is None:
        model = LatentActionModel(**latent_action_kwargs(cfg))
    return _set_weights(model, model_sd, device, is_distributed), state_cfg


COMO_ARCH_KEYS = ('frame_size', 'patch_size', 'idm_dim', 'idm_depth', 'idm_heads', 'idm_mlp', 'n_queries', 'latent_dim',
                  'dec_dim', 'dec_depth', 'dec_heads', 'dec_mlp', 'contrastive_weight', 'temperature', 'feat_dim', 'feat_tokens')


def load_como_from_checkpoint(checkpoint_path, device):
    """CoMo checkpoint (scripts/actions/train_como.py) -> CoMoLAM eval adapter (frozen MAE or tokenizer features + trained IDM;
    continuous actions, k-means into config n_actions clusters by the evals), and the saved state."""
    import torch
    from models.como import CoMo, CoMoLAM, TokenizerFeatures
    state_cfg = torch.load(Path(checkpoint_path) / STATE, map_location='cpu', weights_only=False)
    cfg = state_cfg['config']
    como = CoMo(**{k: cfg[k] for k in COMO_ARCH_KEYS if k in cfg})
    como.load_state_dict(torch.load(Path(checkpoint_path) / MODEL_CHECKPOINT, map_location='cpu', weights_only=True))
    features = None  # MAE
    if cfg.get('features', 'mae') == 'tokenizer':
        tok, _ = load_videotokenizer_from_checkpoint(cfg['tokenizer_path'], 'cpu')
        features = TokenizerFeatures(tok, cfg['tokenizer_feature'], cfg.get('merge', 2))
    return CoMoLAM(como, n_clusters=cfg.get('n_actions', 16), features=features, history=cfg.get('history', 0)).to(device), state_cfg


def load_como_actions(action_dir, device):
    """Action dir (scripts/actions/como_actions.py: state.pt {model_type 'como_actions', como_path, mode, mean, std, centroids})
    -> CoMoActions (CoMo IDM -> standardized full / snapped z), and the saved state."""
    import torch
    from models.como import CoMoActions
    st = torch.load(Path(action_dir) / STATE, map_location='cpu', weights_only=False)
    lam, _ = load_como_from_checkpoint(st['como_path'], device)
    return CoMoActions(lam, st['mean'], st['std'], st['centroids'], mode=st['mode']).to(device), st


def load_dynamics_from_checkpoint(checkpoint_path, device, model = None, is_distributed = False):
    """Instantiate DynamicsModel from a checkpoint's saved config and load weights."""
    import torch
    from models.dynamics import DynamicsModel
    model_sd, state_cfg, cfg = _load_checkpoint_files(checkpoint_path)
    # Infer conditioning_dim from checkpoint if missing
    conditioning_dim = cfg.get('conditioning_dim', None)
    if conditioning_dim is None:
        cond_inferred = None
        for k, v in model_sd.items():
            # Linear weight shape: [out_features, in_features]; in_features is conditioning dim
            if k.endswith('to_gamma_beta.1.weight'):
                cond_inferred = int(v.shape[1])
                break
        conditioning_dim = cond_inferred if cond_inferred is not None else 3
    kwargs = maskgit_dynamics_kwargs(cfg, conditioning_dim)
    if cfg.get('dynamics_type', 'maskgit') in ('flow', 'biflow'):
        # flow-matching dynamics (STA-28) or Bi-flow (STA-61); prefer the EMA weights when the checkpoint has them.
        # FiLM input = [action, 64-d time embedding (+ 64-d noise-level embedding for Bi-flow)]
        n_time = 2 if cfg.get('dynamics_type') == 'biflow' else 1
        action_dim = cfg['conditioning_dim'] if cfg.get('conditioning_dim') is not None else conditioning_dim - 64 * n_time
        ema_path = Path(checkpoint_path) / EMA_CHECKPOINT
        if model is None and ema_path.exists():
            model_sd = torch.load(ema_path, map_location='cpu', weights_only=True)
        if model is None:
            model = build_flow_dynamics(cfg, action_dim)
    if model is None:
        model = DynamicsModel(**kwargs)
    return _set_weights(model, model_sd, device, is_distributed), state_cfg

def prepare_pipeline_run_root(run_name: Optional[str] = None, base_cwd: Optional[str] = None):
    """Create a top-level run root directory results/<timestamp_or_name>"""
    cwd = base_cwd or os.getcwd()
    ts = readable_timestamp()
    name = run_name or ts
    run_root = os.path.join(cwd, 'results', name)
    os.makedirs(run_root, exist_ok=True)
    return run_root, name


def prepare_stage_dirs(run_root_dir: str, stage_name: str):
    """Create stage subdirectories under the given run root.

    Structure:
      <run_root_dir>/<stage_name>/checkpoints
      <run_root_dir>/<stage_name>/visualizations

    Returns (stage_dir, checkpoints_dir, visualizations_dir).
    """
    stage_dir = os.path.join(run_root_dir, stage_name)
    checkpoints_dir = os.path.join(stage_dir, 'checkpoints')
    visualizations_dir = os.path.join(stage_dir, 'visualizations')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    return stage_dir, checkpoints_dir, visualizations_dir
