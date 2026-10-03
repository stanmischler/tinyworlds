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
    .env({"PYTHONPATH": REPO_DIR, "HF_HOME": f"{REPO_DIR}/data/hf_cache"})  # HF weights (CoMo's MAE) cached in the data volume
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
def eval_lam(arms: str, extra: str = ""):
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
        x = extra.split()  # extra lam_judge score args, e.g. "--kmeans-seeds 10"
        subprocess.run(["python", "scripts/eval/lam_judge.py", "score", "--device", "cuda", "--lam", f"{name}={ckpt}", *x], cwd=REPO_DIR, check=True)
        shutil.copytree(f"{REPO_DIR}/eval_results/lam_judge/{name}", out, dirs_exist_ok=True)
        import torch
        if (torch.load(f"{ckpt}/state.pt", weights_only=False).get("config") or {}).get("model_type") == "como":
            # CoMo: continuous actions, no LAM decoder for eval_lam.py; also score 8 clusters
            subprocess.run(["python", "scripts/eval/lam_judge.py", "score", "--device", "cuda", "--k", "8", "--lam", f"{name}_k8={ckpt}", *x], cwd=REPO_DIR, check=True)
            shutil.copytree(f"{REPO_DIR}/eval_results/lam_judge/{name}_k8", f"{out}_k8", dirs_exist_ok=True)
            results_volume.commit()
            print(f"EVAL DONE {name} -> evals/{name}")
            continue
        subprocess.run(["python", "scripts/eval/eval_lam.py", "--device", "cuda", "--lam", f"{name}={ckpt}",
                        "--test-h5", "data/zelda_test_frames.h5", "--out-dir", out], cwd=REPO_DIR, check=True)
        results_volume.commit()
        print(f"EVAL DONE {name} -> evals/{name}")


@app.function(gpu=GPU, cpu=4, memory=32768, volumes=VOLUMES, timeout=2 * 60 * 60)
def como_features(splits: str = "zelda_train,zelda_test", limit: int = 0, suffix: str = ""):
    """Frozen MAE ViT-L features for CoMo (scripts/como_features.py): data/<split>_frames.h5 -> data/<split>_mae_large<suffix>.npy
    in the data volume (zelda_train ~26 GB). --limit N --suffix _smoke for a quick check.
        modal run scripts/modal_train.py::como_features"""
    import subprocess

    for sp in [s for s in splits.split(",") if s]:
        cmd = ["python", "scripts/como_features.py", "--h5", f"data/{sp}_frames.h5", "--out", f"data/{sp}_mae_large{suffix}.npy"]
        if limit:  # smoke: verify the fixed MAE patch order first
            subprocess.run(cmd + ["--check"], cwd=REPO_DIR, check=True)
        subprocess.run(cmd + (["--limit", str(limit)] if limit else []), cwd=REPO_DIR, check=True)
        data_volume.commit()


@app.function(gpu=GPU, cpu=8, memory=int(os.environ.get("TINYWORLDS_MEMORY_MB", 32768)), volumes=VOLUMES, secrets=SECRETS,
              timeout=24 * 60 * 60)
def train_como(config: str = "configs/como/zelda.yaml", overrides: str = ""):
    """CoMo motion IDM training (scripts/train_como.py; features from como_features), outputs in results/<run_name>/ on the
    results volume (committed every COMMIT_EVERY_S). Launch detached so the laptop can sleep:
        TINYWORLDS_GPU=H100 modal run --detach scripts/modal_train.py::train_como --overrides "run_name=como_zelda"
    """
    import subprocess

    stop = threading.Event()

    def commit_periodically():
        while not stop.wait(COMMIT_EVERY_S):
            results_volume.commit()

    threading.Thread(target=commit_periodically, daemon=True).start()
    try:
        subprocess.run(["python", "scripts/train_como.py", "--config", config, "--", *[o for o in overrides.split(",") if o]],
                       cwd=REPO_DIR, check=True)
    finally:
        stop.set()
        results_volume.commit()


@app.function(gpu="L4", cpu=4, memory=32768, volumes=VOLUMES, timeout=60 * 60)
def eval_como_pred(arms: str):
    """Next-frame PSNR/SSIM of CoMo's decoder (scripts/eval/eval_como_pred.py) on the held-out Zelda windows.
    arms: "<name>=<como ckpt dir relative to the results volume>,..." -> results volume evals/como_pred/<name>.json
        modal run scripts/modal_train.py::eval_como_pred --arms "v1_50k=como_zelda_v1/como/checkpoints/como_step_50000"
    """
    import subprocess

    for spec in [a for a in arms.split(",") if a]:
        name, ckpt = spec.split("=", 1)
        subprocess.run(["python", "scripts/eval/eval_como_pred.py", "--ckpt", f"results/{ckpt}", "--name", name,
                        "--out-dir", "results/evals/como_pred"], cwd=REPO_DIR, check=True)
        results_volume.commit()


@app.function(gpu=GPU, cpu=4, memory=32768, volumes=VOLUMES, timeout=6 * 60 * 60)
def laof_flow(split: str = "test", limit: int = 0, batch: int = 32):
    """LAOF flow targets (scripts/laof_flow.py) for data/zelda_<split>_frames.h5 -> data/zelda_<split>_flow_gap4.h5 in the
    data volume (+ a viz PNG in the results volume under laof/). limit > 0 writes a *_smoke file instead.
        modal run scripts/modal_train.py::laof_flow --split test --limit 512
    """
    import subprocess

    tag = "_smoke" if limit else ""
    os.makedirs(f"{REPO_DIR}/results/laof", exist_ok=True)
    subprocess.run(
        ["python", "scripts/laof_flow.py", "--h5", f"data/zelda_{split}_frames.h5", "--out", f"data/zelda_{split}_flow_gap4{tag}.h5",
         "--batch", str(batch), "--limit", str(limit), "--viz", f"results/laof/flow_{split}{tag}.png"],
        cwd=REPO_DIR,
        check=True,
    )
    data_volume.commit()
    results_volume.commit()


@app.function(gpu=GPU, cpu=4, memory=32768, volumes=VOLUMES, secrets=SECRETS, timeout=24 * 60 * 60)
def train_laof(config: str, overrides: str = "", run_name: str = ""):
    """LAOF latent action model (scripts/train_laof.py); checkpoints, viz and judge.jsonl in results/<run_name>/latent_actions/.
        TINYWORLDS_GPU=H100 modal run --detach scripts/modal_train.py::train_laof --config configs/laof/discrete.yaml --run-name laof_discrete
    """
    import subprocess

    stop = threading.Event()

    def commit_periodically():
        while not stop.wait(COMMIT_EVERY_S):
            results_volume.commit()

    threading.Thread(target=commit_periodically, daemon=True).start()
    try:
        subprocess.run(
            ["python", "scripts/train_laof.py", "--config", config, f"run_name={run_name}", *[o for o in overrides.split(",") if o]],
            cwd=REPO_DIR,
            check=True,
        )
    finally:
        stop.set()
        results_volume.commit()


ITC_TOKENIZER = "data/itc_i5/zelda_v3_tokenizer_step_29000"  # zelda_v3 tokenizer, uploaded to the data volume


@app.function(cpu=2, memory=4096, volumes=VOLUMES, timeout=3 * 60 * 60)
def itc_pseudo_chunk(lo: int, hi: int) -> bytes:
    """Teacher pseudo-labels (scripts/eval/itc_pseudo.py label) of train pairs (t, t+4), t in [lo, hi); returns the .npz bytes."""
    import subprocess
    import tempfile

    out = tempfile.mktemp(suffix=".npz")
    subprocess.run(["python", "scripts/eval/itc_pseudo.py", "label", "--lo", str(lo), "--hi", str(hi), "--out", out,
                    "--tokenizer", ITC_TOKENIZER, "--threads", "2"], cwd=REPO_DIR, check=True)
    return open(out, "rb").read()


@app.function(cpu=2, memory=8192, volumes=VOLUMES, timeout=6 * 60 * 60)
def itc_pseudo(lo: int = 0, hi: int = -1, n_chunks: int = 64, out: str = "data/zelda_train_itc_pseudo_gap4.npz"):
    """STA-35 itc_loop i5: label all gap-4 pairs of zelda_train_frames.h5 with the frozen i4 teacher, in parallel CPU
    containers (ITC = tokenizer encode on CPU + a 2048x2048 Hungarian per non-scroll pair), merged into `out` (data volume).
        modal run --detach scripts/modal_train.py::itc_pseudo --hi 200 --n-chunks 2 --out data/itc_i5/pseudo_smoke.npz
    """
    import subprocess

    import h5py

    n = len(h5py.File(f"{REPO_DIR}/data/zelda_train_frames.h5", "r")["source_index"])
    hi = n if hi < 0 else hi
    bounds = [lo + (hi - lo) * k // n_chunks for k in range(n_chunks + 1)]
    os.makedirs(f"{REPO_DIR}/data/itc_i5/parts", exist_ok=True)
    parts = []
    for (a, _), blob in zip(zip(bounds[:-1], bounds[1:]), itc_pseudo_chunk.starmap(zip(bounds[:-1], bounds[1:]))):
        parts.append(f"{REPO_DIR}/data/itc_i5/parts/{a}.npz")
        open(parts[-1], "wb").write(blob)
    subprocess.run(["python", "scripts/eval/itc_pseudo.py", "merge", "--parts", *parts, "--out", out], cwd=REPO_DIR, check=True)
    data_volume.commit()
    print(f"PSEUDO DONE -> {out}")


@app.function(cpu=2, memory=4096, volumes=VOLUMES, timeout=3 * 60 * 60)
def itc_relabel_chunk(lo: int, hi: int, base: str) -> bytes:
    """i6: add the itcpix teacher (scripts/eval/itc_pseudo.py relabel) on rows [lo, hi) where `base` ran ITC."""
    import subprocess
    import tempfile

    out = tempfile.mktemp(suffix=".npz")
    subprocess.run(["python", "scripts/eval/itc_pseudo.py", "relabel", "--base", base, "--lo", str(lo), "--hi", str(hi), "--out", out,
                    "--tokenizer", ITC_TOKENIZER, "--threads", "2"], cwd=REPO_DIR, check=True)
    return open(out, "rb").read()


@app.function(cpu=2, memory=8192, volumes=VOLUMES, timeout=6 * 60 * 60)
def itc_relabel(base: str = "data/zelda_train_itc_pseudo_gap4.npz", lo: int = 0, hi: int = -1, n_chunks: int = 64,
                out: str = "data/zelda_train_itc_pseudo_gap4_i6.npz"):
    """STA-35 itc_loop i6: relabel only the ITC pairs of the i5 label file with the itcpix teacher (ITC-kept AND pixel-change
    localiser); chunks are balanced by ITC-pair count. Output = base fields + code_itcpix/chg_itcpix/ctr_itcpix (data volume).
        modal run --detach scripts/modal_train.py::itc_relabel --hi 2000 --n-chunks 2 --out data/itc_i6/relabel_smoke.npz
    """
    import subprocess

    import numpy as np

    run = np.load(f"{REPO_DIR}/{base}")["itc_run"]
    hi = len(run) if hi < 0 else hi
    cum = np.cumsum(run[lo:hi] == 1)
    bounds = [lo] + [lo + int(np.searchsorted(cum, cum[-1] * k / n_chunks)) for k in range(1, n_chunks)] + [hi]
    os.makedirs(f"{REPO_DIR}/data/itc_i6/parts", exist_ok=True)
    parts = []
    args = [(a, b, base) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    for (a, _, _), blob in zip(args, itc_relabel_chunk.starmap(args)):
        parts.append(f"{REPO_DIR}/data/itc_i6/parts/{a}.npz")
        open(parts[-1], "wb").write(blob)
    subprocess.run(["python", "scripts/eval/itc_pseudo.py", "merge", "--base", base, "--parts", *parts, "--out", out], cwd=REPO_DIR, check=True)
    data_volume.commit()
    print(f"RELABEL DONE -> {out}")


@app.function(gpu=GPU, cpu=4, memory=16384, volumes=VOLUMES, timeout=2 * 60 * 60)
def train_encoder(run_name: str, extra_args: str = ""):
    """STA-35 itc_loop i5 arm A: frame-pair action encoder on teacher pseudo-labels (scripts/train_action_encoder.py).
    Writes results/<run_name>/action_encoder/{encoder.pt, train_log.json, codes_judge.json} in the results volume.
        TINYWORLDS_GPU=H100 modal run --detach scripts/modal_train.py::train_encoder --run-name itcloop_i5_enc_itc --extra-args "--label-key code_itc"
    """
    import shlex
    import subprocess

    try:
        subprocess.run(["python", "scripts/train_action_encoder.py", "train", "--out-dir", f"results/{run_name}/action_encoder",
                        *shlex.split(extra_args)], cwd=REPO_DIR, check=True)
    finally:
        results_volume.commit()


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
