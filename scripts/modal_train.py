"""Run the TinyWorlds training pipeline on a Modal GPU.

One-time setup (from the repo root):
    pip install modal
    modal setup                                          # browser auth
    modal secret create wandb WANDB_API_KEY=<your key>    # or: TINYWORLDS_NO_WANDB=1 ... --no-wandb

Then:
    modal run scripts/modal_train.py::download --pattern "zelda_frames.h5"
    modal run --detach scripts/modal_train.py --dataset ZELDA   # returns immediately; --detach keeps the app alive
    modal app logs tinyworlds                                   # follow training; `modal app stop tinyworlds` cancels

Use a different shared config (e.g. a self-contained experiment under configs/<name>/):
    modal run --detach scripts/modal_train.py --dataset SONIC --training-config configs/sonic_short/training.yaml

Pick the GPU with an env var (defaults to one A100-80GB):
    TINYWORLDS_GPU=H100 modal run --detach scripts/modal_train.py --dataset ZELDA

Datasets and checkpoints live in Modal Volumes, so they survive between runs:
    modal volume ls  tinyworlds-data
    modal volume ls  tinyworlds-results
    modal volume get tinyworlds-results <run_dir> ./results   # pull checkpoints back
"""

import os
import threading

import modal

REPO_DIR = "/root/tinyworlds"
GPU = os.environ.get("TINYWORLDS_GPU", "A100-80GB")
COMMIT_EVERY_S = 10 * 60  # how often to persist results to the volume during training

# set TINYWORLDS_NO_WANDB=1 to run without creating a Modal secret (also pass --no-wandb)
SECRETS = [] if os.environ.get("TINYWORLDS_NO_WANDB") else [
    modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")  # opencv needs the GL/glib shared libs
    .pip_install_from_requirements("requirements.txt")
    .env({"PYTHONPATH": REPO_DIR})
    .add_local_dir(
        ".",
        remote_path=REPO_DIR,
        ignore=["**/.git", "**/.venv", "data/**", "results/**", "wandb/**", "inference_results/**"],
    )
)

app = modal.App("tinyworlds", image=image)

# persistent storage: datasets on one volume, run outputs/checkpoints on another
data_volume = modal.Volume.from_name("tinyworlds-data", create_if_missing=True)
results_volume = modal.Volume.from_name("tinyworlds-results", create_if_missing=True)
VOLUMES = {f"{REPO_DIR}/data": data_volume, f"{REPO_DIR}/results": results_volume}


@app.function(volumes=VOLUMES, timeout=60 * 60)
def download(pattern: str = "zelda_frames.h5"):
    """Fetch a dataset from HuggingFace straight into the data volume."""
    import subprocess

    subprocess.run(
        ["python", "scripts/download_assets.py", "datasets", "--pattern", pattern],
        cwd=REPO_DIR,
        check=True,
    )
    data_volume.commit()


@app.function(
    gpu=GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=24 * 60 * 60,  # Modal's per-call maximum
)
def train(dataset: str = "ZELDA", overrides: list[str] | None = None, training_config: str = "configs/training.yaml"):
    """Run the three-stage pipeline (video tokenizer -> latent actions -> dynamics)."""
    import subprocess

    from omegaconf import OmegaConf

    # full_train.py does not forward CLI overrides to the stage scripts (they re-read the
    # training config from disk), so bake the dataset and any overrides into a config copy.
    cfg = OmegaConf.load(f"{REPO_DIR}/{training_config}")
    cfg.dataset = dataset

    # the .h5 fixes the resolution (zelda/picodoom are 128, sonic/pong are 64) and the models
    # are built from frame_size, so read it off the data rather than trusting the config default
    h5_path = f"{REPO_DIR}/data/{dataset.lower()}_frames.h5"
    if os.path.exists(h5_path):
        import h5py

        with h5py.File(h5_path) as f:
            cfg.frame_size = int(f["frames"].shape[1])
        print(f"frame_size={cfg.frame_size} (from {h5_path})")
    else:
        raise FileNotFoundError(
            f"{h5_path} not found in the data volume. Run: "
            f'modal run scripts/modal_train.py::download --pattern "{dataset.lower()}_frames.h5"'
        )

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg_path = f"{REPO_DIR}/configs/training_modal.yaml"
    OmegaConf.save(cfg, cfg_path)
    print(OmegaConf.to_yaml(cfg))

    # volume writes only persist once committed; commit periodically so a timeout or a manual
    # `modal app stop` loses at most COMMIT_EVERY_S of checkpoints/visualizations
    stop = threading.Event()

    def commit_periodically():
        while not stop.wait(COMMIT_EVERY_S):
            results_volume.commit()

    threading.Thread(target=commit_periodically, daemon=True).start()
    try:
        subprocess.run(
            ["python", "scripts/full_train.py", "--config", cfg_path],
            cwd=REPO_DIR,
            check=True,
        )
    finally:
        stop.set()
        results_volume.commit()  # keep whatever checkpoints exist, even if a stage fails


@app.local_entrypoint()
def main(
    dataset: str = "ZELDA",
    no_wandb: bool = False,
    overrides: str = "",
    training_config: str = "configs/training.yaml",
):
    """modal run scripts/modal_train.py --dataset ZELDA --training-config configs/training.yaml --overrides "k=v,k=v" """
    extra = [o for o in overrides.split(",") if o]
    if no_wandb:
        extra.append("use_wandb=false")
    # spawn (not .remote) so this local process returns immediately: with `--detach` the app then
    # runs on its own, and a laptop going to sleep or losing wifi cannot take the training down.
    call = train.spawn(dataset=dataset, overrides=extra, training_config=training_config)
    print(f"training started (function call {call.object_id}); follow it with:  modal app logs tinyworlds")
