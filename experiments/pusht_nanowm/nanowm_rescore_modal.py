"""Re-run NanoWM's published DINO-WM evaluation (Push-T, Point Maze or Wall) on a Modal GPU and dump its predictions.

Reproduces NanoWM-B/2 on DINO-WM <env> (paper Tab. 7 / their README, 256px: Push-T 33.19 / 0.982 / 0.016 / FID 13.63,
Point Maze 36.74 / 0.984 / 0.019 / 9.66, Wall 34.05 / 0.994 / 0.010 / 2.64) with NanoWM's own code
(github.com/simchowitzlabpublic/nano-world-model, pinned commit, deps installed from their uv.lock), and saves the
predicted frames so scripts/eval/eval_pusht.py / eval_nanowm.py can re-score them at our resolutions against our own
ground truth. The eval call, its deviations from their setup and the output files are documented in
experiments/pusht_nanowm/nanowm_dump.py (shared with nanowm_rescore_local.sh, the same thing without Modal).

Launch (from the repo root; Modal spend: one GPU, roughly 15-30 min incl. image build; --env defaults to pusht):
    TINYWORLDS_GPU=H100 ./.venv/bin/modal run --detach experiments/pusht_nanowm/nanowm_rescore_modal.py --env point_maze
    ./.venv/bin/modal app logs tinyworlds-nanowm        # follow; outputs land on the volume even if the client disconnects

Download the outputs:
    ./.venv/bin/modal volume get --force tinyworlds-results nanowm_point_maze_rescore ./results/   # -> ./results/nanowm_point_maze_rescore/ (a non-existent dest path makes modal write every file onto ONE file)

The raw DINO-WM data goes to the tinyworlds-data volume (/data/pusht_noise, /data/point_maze, /data/wall_single).
"""

import os

import modal

from experiments.pusht_nanowm.nanowm_dump import ENVS, NANOWM_COMMIT, NANOWM_REPO, ensure_data as _ensure_data, run_rescore

NANOWM_DIR = "/opt/nano-world-model"
NANOWM_VENV = "/opt/nanowm-venv"
DATA_DIR = "/data"  # tinyworlds-data volume; raw data at /data/<folder>
OUT_ROOT = "/results"  # tinyworlds-results volume
GPU = os.environ.get("TINYWORLDS_GPU", "H100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl")  # build-essential: triton (torch.compile) needs a C compiler
    .pip_install("uv==0.12.21", "numpy==1.26.4")
    .run_commands(
        f"git clone {NANOWM_REPO} {NANOWM_DIR}",
        f"cd {NANOWM_DIR} && git checkout {NANOWM_COMMIT}",
        # their locked environment (python 3.11.13, torch 2.7.1+cu128, lightning 1.9.5, piqa, lpips, pytorch-fid ...);
        # --frozen installs uv.lock as-is: at this commit `--locked` reports the lock as stale vs pyproject
        f"cd {NANOWM_DIR} && UV_PROJECT_ENVIRONMENT={NANOWM_VENV} uv sync --frozen --no-dev --managed-python",
        f"{NANOWM_VENV}/bin/python -c 'import torch, pytorch_lightning, piqa, lpips, pytorch_fid, decord; "
        "print(torch.__version__, pytorch_lightning.__version__)'",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "WANDB_MODE": "disabled"})
    .add_local_python_source("experiments")  # nanowm_dump (stdlib + numpy) inside the container
)

app = modal.App("tinyworlds-nanowm", image=image)
data_volume = modal.Volume.from_name("tinyworlds-data", create_if_missing=True)
results_volume = modal.Volume.from_name("tinyworlds-results", create_if_missing=True)


@app.function(volumes={DATA_DIR: data_volume}, cpu=4, memory=8192, timeout=2 * 60 * 60)
def ensure_data(env: str = "pusht"):
    """Download + unzip the env's DINO-WM zip into the data volume (/data/<folder>) if missing."""
    data_volume.reload()
    if _ensure_data(DATA_DIR, env, zip_dir="/tmp"):
        data_volume.commit()


@app.function(
    gpu=GPU,
    cpu=8,
    memory=65536,
    volumes={DATA_DIR: data_volume, OUT_ROOT: results_volume},
    timeout=3 * 60 * 60,
)
def evaluate(env: str = "pusht", probe: bool = True):
    data_volume.reload()
    out = f"{OUT_ROOT}/nanowm_{env}_rescore"
    payload = run_rescore(env, NANOWM_DIR, f"{NANOWM_VENV}/bin/python", DATA_DIR, "/tmp/nanowm", out, probe=probe,
                          on_log=results_volume.commit)
    results_volume.commit()
    return payload


@app.local_entrypoint()
def main(env: str = "pusht", probe: bool = True):
    assert env in ENVS, f"--env must be one of {sorted(ENVS)}"
    ensure_data.remote(env)
    print(evaluate.remote(env=env, probe=probe))
