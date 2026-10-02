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
    cpu=8,  # reserve cores for the dataloader workers (configs may set num_workers up to 8)
    memory=16384,  # MiB; preload_ratio 1.0 on zelda_train (65k x 128x128x3) is ~3.2 GB per dataset object, x2 (train + val)
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=24 * 60 * 60,  # Modal's per-call maximum
)
def train(dataset: str = "ZELDA", overrides: list[str] | None = None, training_config: str = "configs/training.yaml", run_name: str = ""):
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
            env={**os.environ, "NG_RUN_NAME": run_name},  # empty -> timestamped results/<run dir>
        )
    finally:
        stop.set()
        results_volume.commit()  # keep whatever checkpoints exist, even if a stage fails


@app.function(gpu="L4", cpu=4, memory=16384, volumes=VOLUMES, timeout=60 * 60)
def eval_lam(arms: str):
    """Score LAM checkpoints on the held-out Zelda judge set + eval_lam diagnostic, on a GPU (keeps the laptop free).

    arms: "<name>=<checkpoint dir relative to the results volume>,<name>=...". Outputs go to the results volume under
    evals/<name>/ (lam_judge score.json + code_<k>.png, eval_lam lam_diag_<name>.{json,png}); fetch them with
    `modal volume get tinyworlds-results evals/<name> eval_results/lam_judge/`.
        modal run scripts/modal_train.py::eval_lam --arms "i1_warp=lamloop_i1_warp/latent_actions/checkpoints/latent_actions_step_6000"
    """
    import shutil
    import subprocess

    for spec in [a for a in arms.split(",") if a]:
        name, ckpt = spec.split("=", 1)
        ckpt = f"{REPO_DIR}/results/{ckpt}"
        out = f"{REPO_DIR}/results/evals/{name}"
        os.makedirs(out, exist_ok=True)
        subprocess.run(["python", "scripts/eval/lam_judge.py", "score", "--device", "cuda", "--lam", f"{name}={ckpt}"], cwd=REPO_DIR, check=True)
        shutil.copytree(f"{REPO_DIR}/eval_results/lam_judge/{name}", out, dirs_exist_ok=True)
        subprocess.run(["python", "scripts/eval/eval_lam.py", "--device", "cuda", "--lam", f"{name}={ckpt}",
                        "--test-h5", "data/zelda_test_frames.h5", "--out-dir", out], cwd=REPO_DIR, check=True)
        results_volume.commit()
        print(f"EVAL DONE {name} -> evals/{name}")


@app.function(gpu="L4", cpu=4, memory=16384, volumes=VOLUMES, timeout=3 * 60 * 60)
def eval_dynamics(run_dir: str, name: str, extra_args: str = "", num_steps: str = "1,10"):
    """Held-out next-frame eval (scripts/eval/eval_next_frame.py) of a pipeline run dir on the results volume, on a GPU,
    once per MaskGIT step count in num_steps (both reported: 1 pass matches the full-last-frame training mode).
    Outputs <name>_s<steps>.{json,png} go to evals/next_frame/ on the results volume; fetch with
    `modal volume get tinyworlds-results evals/next_frame/<name>_s1.json eval_results/`.
        modal run scripts/modal_train.py::eval_dynamics --run-dir 2026_09_26_12_12_49 --name zelda_v3_ctx --extra-args "--test-h5 data/zelda_test_frames.h5"
    """
    import shlex
    import subprocess

    out = f"{REPO_DIR}/results/evals/next_frame"
    for steps in num_steps.split(","):
        subprocess.run(["python", "scripts/eval/eval_next_frame.py", "--device", "cuda", "--run-dir", f"{REPO_DIR}/results/{run_dir}",
                        "--name", f"{name}_s{steps}", "--num-steps", steps, "--out-dir", out, *shlex.split(extra_args)],
                       cwd=REPO_DIR, check=True)
        results_volume.commit()
        print(f"EVAL DONE {name}_s{steps} -> evals/next_frame/{name}_s{steps}.json")


@app.local_entrypoint()
def main(
    dataset: str = "ZELDA",
    no_wandb: bool = False,
    overrides: str = "",
    training_config: str = "configs/training.yaml",
    run_name: str = "",
):
    """modal run scripts/modal_train.py --dataset ZELDA --training-config configs/training.yaml --overrides "k=v,k=v" [--run-name <name>]"""
    extra = [o for o in overrides.split(",") if o]
    if no_wandb:
        extra.append("use_wandb=false")
    # spawn (not .remote) so this local process returns immediately: with `--detach` the app then
    # runs on its own, and a laptop going to sleep or losing wifi cannot take the training down.
    call = train.spawn(dataset=dataset, overrides=extra, training_config=training_config, run_name=run_name)
    print(f"training started (function call {call.object_id}); follow it with:  modal app logs tinyworlds")
