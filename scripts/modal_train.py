"""Run the TinyWorlds training pipeline on a Modal GPU.

One-time setup (from the repo root):
    pip install modal
    modal setup                                          # browser auth
    modal secret create wandb WANDB_API_KEY=<your key>    # or: TINYWORLDS_NO_WANDB=1 ... --no-wandb

Then:
    modal run scripts/modal_train.py::download --pattern "zelda_frames.h5"
    modal run scripts/modal_train.py::convert_pusht           # Push-T: download from OSF + convert in the volume
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


@app.function(volumes=VOLUMES, cpu=16, memory=32768, timeout=4 * 60 * 60)
def convert_pusht():
    """Download DINO-WM's pusht_noise into the data volume and convert it (scripts/convert_pusht.py).
    Leaves the raw dataset at data/pusht_noise/ (NanoWM's eval reads it) next to pusht_frames.h5 / pusht_val_frames.h5."""
    import subprocess

    subprocess.run(
        ["python", "scripts/convert_pusht.py", "--raw-dir", "data/pusht_noise", "--out-dir", "data", "--workers", "16"],
        cwd=REPO_DIR,
        check=True,
    )
    data_volume.commit()


@app.function(gpu=GPU, volumes=VOLUMES, memory=16384, timeout=2 * 60 * 60)
def eval_pusht(run_dir: str, extra_args: str = ""):
    """Run scripts/eval/eval_pusht.py on a run in the results volume (run_dir relative to results/), e.g.
    modal run scripts/modal_train.py::eval_pusht --run-dir 2026_10_01_17_00_00 --extra-args "--nanowm-npz results/nanowm_pusht_rescore/predictions_f16.npz"
    Writes results/eval_results/<name>.{json,png} in the volume."""
    import shlex
    import subprocess

    subprocess.run(
        ["python", "scripts/eval/eval_pusht.py", "--run-dir", f"results/{run_dir}", "--out-dir", "results/eval_results",
         "--batch-size", "64", *shlex.split(extra_args)],
        cwd=REPO_DIR,
        check=True,
    )
    results_volume.commit()


@app.function(
    gpu=GPU,
    cpu=8,  # reserve cores for the dataloader workers (configs may set num_workers up to 8)
    # MiB; preload_ratio 1.0 on zelda_train (65k x 128x128x3) is ~3.2 GB per dataset object, x2 (train + val).
    # Push-T's 467k x 128x128x3 train set is ~23 GB in RAM: launch with TINYWORLDS_MEMORY_MB=65536
    memory=int(os.environ.get("TINYWORLDS_MEMORY_MB", 16384)),
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


@app.function(gpu="L4", cpu=4, memory=16384, volumes=VOLUMES, timeout=60 * 60)
def eval_lam_set(game: str, arms: str = "", baselines: str = "random,camera"):
    """Score LAM checkpoints on a game's judge-labelled action set (scripts/eval/lam_eval.py; the set ships with the image
    from eval_results/lam_eval/<game>/set). arms: "<name>=<checkpoint dir relative to the results volume>,...".
    Outputs go to the results volume under lam_eval/<game>/<name>/score.json; fetch them with
    `modal volume get tinyworlds-results lam_eval/<game> eval_results/lam_eval/`.
        modal run scripts/modal_train.py::eval_lam_set --game zelda --arms "i4=lam_eval_ckpts/zelda_i4_bal16_seed2"
    """
    import shutil
    import subprocess

    cmd = ["python", "scripts/eval/lam_eval.py", "score", "--game", game, "--device", "cuda"]
    for spec in [a for a in arms.split(",") if a]:
        name, ckpt = spec.split("=", 1)
        cmd += ["--lam", f"{name}={REPO_DIR}/results/{ckpt}"]
    for b in [b for b in baselines.split(",") if b]:
        cmd += ["--baseline", b]
    subprocess.run(cmd, cwd=REPO_DIR, check=True)
    for d in os.listdir(f"{REPO_DIR}/eval_results/lam_eval/{game}"):
        if d not in ("set", "groups"):
            shutil.copytree(f"{REPO_DIR}/eval_results/lam_eval/{game}/{d}", f"{REPO_DIR}/results/lam_eval/{game}/{d}", dirs_exist_ok=True)
    results_volume.commit()
    print(f"EVAL DONE {game} -> lam_eval/{game}")


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
