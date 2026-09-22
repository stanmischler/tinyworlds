"""Run the TinyWorlds training pipeline on a Modal GPU.

One-time setup (from the repo root):
    pip install modal
    modal setup                                          # browser auth
    modal secret create wandb WANDB_API_KEY=<your key>    # or: TINYWORLDS_NO_WANDB=1 ... --no-wandb

Then:
    modal run scripts/modal_train.py::download --pattern "zelda_frames.h5"
    modal run --detach scripts/modal_train.py --dataset ZELDA

Pick the GPU with an env var (defaults to one A100-80GB):
    TINYWORLDS_GPU=H100 modal run --detach scripts/modal_train.py --dataset ZELDA

Datasets and checkpoints live in Modal Volumes, so they survive between runs:
    modal volume ls  tinyworlds-data
    modal volume ls  tinyworlds-results
    modal volume get tinyworlds-results <run_dir> ./results   # pull checkpoints back
"""

import os

import modal

REPO_DIR = "/root/tinyworlds"
GPU = os.environ.get("TINYWORLDS_GPU", "A100-80GB")

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
def train(dataset: str = "ZELDA", overrides: list[str] | None = None):
    """Run the three-stage pipeline (video tokenizer -> latent actions -> dynamics)."""
    import subprocess

    from omegaconf import OmegaConf

    # full_train.py does not forward CLI overrides to the stage scripts (they re-read the
    # training config from disk), so bake the dataset and any overrides into a config copy.
    cfg = OmegaConf.load(f"{REPO_DIR}/configs/training.yaml")
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

    try:
        subprocess.run(
            ["python", "scripts/full_train.py", "--config", cfg_path],
            cwd=REPO_DIR,
            check=True,
        )
    finally:
        results_volume.commit()  # keep whatever checkpoints exist, even if a stage fails


@app.local_entrypoint()
def main(dataset: str = "ZELDA", no_wandb: bool = False, overrides: str = ""):
    """modal run scripts/modal_train.py --dataset ZELDA --overrides "video_tokenizer_config=..." """
    extra = [o for o in overrides.split(",") if o]
    if no_wandb:
        extra.append("use_wandb=false")
    train.remote(dataset=dataset, overrides=extra)
