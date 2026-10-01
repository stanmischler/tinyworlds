import sys
import os
from utils.utils import run_command, find_latest_checkpoint, prepare_pipeline_run_root
from utils.config import TrainingConfig, load_config

DEFAULT_TRAINING_CONFIG = os.path.join(os.getcwd(), 'configs', 'training.yaml')

def main():
    # load training config
    argv = sys.argv
    try:
        idx = argv.index('--config')
        training_cfg_path = argv[idx + 1] if idx + 1 < len(argv) else DEFAULT_TRAINING_CONFIG
    except ValueError:
        print(f"No training config path provided, using default: {DEFAULT_TRAINING_CONFIG}")
        training_cfg_path = DEFAULT_TRAINING_CONFIG

    train_config: TrainingConfig = load_config(TrainingConfig, default_config_path=training_cfg_path)

    # torchrun if distributed, else python
    if train_config.nproc_per_node > 1:
        launcher = [
            "torchrun",
            "--nproc_per_node", str(train_config.nproc_per_node),
        ]
        if train_config.standalone:
            launcher += ["--standalone"]
    else:
        launcher = [sys.executable]

    # top-level run root and export child processes can use
    run_root, run_name = prepare_pipeline_run_root(base_cwd=os.getcwd())
    os.environ['NG_RUN_ROOT_DIR'] = run_root

    if train_config.run_video_tokenizer:
        v_cmd = launcher + [
            "scripts/train_video_tokenizer.py",
            "--config", train_config.video_tokenizer_config,
            "--training_config", training_cfg_path,
        ]
        if not run_command(v_cmd, "Video Tokenizer Training"):
            return

    if train_config.run_latent_actions:
        latent_actions_cmd = launcher + [
            "scripts/train_latent_actions.py",
            "--config", train_config.latent_actions_config,
            "--training_config", training_cfg_path,
        ]
        if not run_command(latent_actions_cmd, "Latent Actions Training"):
            return

    # checkpoints handed to dynamics, in order of preference: an explicit path set in the dynamics yaml, the stage
    # that just ran in this pipeline (own run dir), and only then the newest run dir on disk (the legacy
    # cwd-wide search, which picks the wrong run when several runs share results/ or a volume)
    from omegaconf import OmegaConf
    dyn_yaml = OmegaConf.load(train_config.dynamics_config)

    def stage_checkpoint(stage, ran_here):
        explicit = dyn_yaml.get(f"{stage}_path")
        if explicit:
            return explicit
        if ran_here:
            return find_latest_checkpoint(".", stage, run_root_dir=run_root, stage_name=stage)
        return find_latest_checkpoint(".", stage)

    video_tokenizer_checkpoint = latent_actions_checkpoint = None
    if train_config.run_dynamics:
        video_tokenizer_checkpoint = stage_checkpoint("video_tokenizer", train_config.run_video_tokenizer)
        try:
            latent_actions_checkpoint = stage_checkpoint("latent_actions", train_config.run_latent_actions)
        except Exception:
            # none anywhere: fine for dynamics with action_source=gt (ground-truth actions, no LAM stage)
            latent_actions_checkpoint = None
        dyn_cmd = launcher + [
            "scripts/train_dynamics.py",
            "--config", train_config.dynamics_config,
            "--training_config", training_cfg_path,
            f"video_tokenizer_path={video_tokenizer_checkpoint}",
        ]
        if latent_actions_checkpoint is not None:
            dyn_cmd.append(f"latent_actions_path={latent_actions_checkpoint}")
        if not run_command(dyn_cmd, "Dynamics Model Training"):
            return

    dynamics_checkpoint = find_latest_checkpoint(".", "dynamics", run_root_dir=run_root, stage_name="dynamics") if train_config.run_dynamics else None
    print("\n📁 Results Summary:")
    print(f"Video Tokenizer: {video_tokenizer_checkpoint}")
    print(f"Latent Actions: {latent_actions_checkpoint}")
    print(f"Dynamics Model: {dynamics_checkpoint}")


if __name__ == "__main__":
    main() 