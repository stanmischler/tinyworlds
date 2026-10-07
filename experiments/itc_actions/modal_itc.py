"""Modal functions of the ITC action experiment: teacher pseudo-labels in parallel CPU containers, then the frame-pair
action encoder trained on them. Shares the tinyworlds app, image and volumes of scripts/infra/modal_train.py:
    modal run --detach experiments/itc_actions/modal_itc.py::itc_pseudo --hi 200 --n-chunks 2 --out data/itc_i5/pseudo_smoke.npz"""

import os

from scripts.infra.modal_train import GPU, REPO_DIR, VOLUMES, app, data_volume, results_volume


ITC_TOKENIZER = "data/itc_i5/zelda_v3_tokenizer_step_29000"  # zelda_v3 tokenizer, uploaded to the data volume


@app.function(cpu=2, memory=4096, volumes=VOLUMES, timeout=3 * 60 * 60)
def itc_pseudo_chunk(lo: int, hi: int) -> bytes:
    """Teacher pseudo-labels (experiments/itc_actions/itc_pseudo.py label) of train pairs (t, t+4), t in [lo, hi); returns the .npz bytes."""
    import subprocess
    import tempfile

    out = tempfile.mktemp(suffix=".npz")
    subprocess.run(["python", "experiments/itc_actions/itc_pseudo.py", "label", "--lo", str(lo), "--hi", str(hi), "--out", out,
                    "--tokenizer", ITC_TOKENIZER, "--threads", "2"], cwd=REPO_DIR, check=True)
    return open(out, "rb").read()


@app.function(cpu=2, memory=8192, volumes=VOLUMES, timeout=6 * 60 * 60)
def itc_pseudo(lo: int = 0, hi: int = -1, n_chunks: int = 64, out: str = "data/zelda_train_itc_pseudo_gap4.npz"):
    """STA-35 itc_loop i5: label all gap-4 pairs of zelda_train_frames.h5 with the frozen i4 teacher, in parallel CPU
    containers (ITC = tokenizer encode on CPU + a 2048x2048 Hungarian per non-scroll pair), merged into `out` (data volume).
        modal run --detach experiments/itc_actions/modal_itc.py::itc_pseudo --hi 200 --n-chunks 2 --out data/itc_i5/pseudo_smoke.npz
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
    subprocess.run(["python", "experiments/itc_actions/itc_pseudo.py", "merge", "--parts", *parts, "--out", out], cwd=REPO_DIR, check=True)
    data_volume.commit()
    print(f"PSEUDO DONE -> {out}")


@app.function(cpu=2, memory=4096, volumes=VOLUMES, timeout=3 * 60 * 60)
def itc_relabel_chunk(lo: int, hi: int, base: str) -> bytes:
    """i6: add the itcpix teacher (experiments/itc_actions/itc_pseudo.py relabel) on rows [lo, hi) where `base` ran ITC."""
    import subprocess
    import tempfile

    out = tempfile.mktemp(suffix=".npz")
    subprocess.run(["python", "experiments/itc_actions/itc_pseudo.py", "relabel", "--base", base, "--lo", str(lo), "--hi", str(hi), "--out", out,
                    "--tokenizer", ITC_TOKENIZER, "--threads", "2"], cwd=REPO_DIR, check=True)
    return open(out, "rb").read()


@app.function(cpu=2, memory=8192, volumes=VOLUMES, timeout=6 * 60 * 60)
def itc_relabel(base: str = "data/zelda_train_itc_pseudo_gap4.npz", lo: int = 0, hi: int = -1, n_chunks: int = 64,
                out: str = "data/zelda_train_itc_pseudo_gap4_i6.npz"):
    """STA-35 itc_loop i6: relabel only the ITC pairs of the i5 label file with the itcpix teacher (ITC-kept AND pixel-change
    localiser); chunks are balanced by ITC-pair count. Output = base fields + code_itcpix/chg_itcpix/ctr_itcpix (data volume).
        modal run --detach experiments/itc_actions/modal_itc.py::itc_relabel --hi 2000 --n-chunks 2 --out data/itc_i6/relabel_smoke.npz
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
    subprocess.run(["python", "experiments/itc_actions/itc_pseudo.py", "merge", "--base", base, "--parts", *parts, "--out", out], cwd=REPO_DIR, check=True)
    data_volume.commit()
    print(f"RELABEL DONE -> {out}")


@app.function(gpu=GPU, cpu=4, memory=16384, volumes=VOLUMES, timeout=2 * 60 * 60)
def train_encoder(run_name: str, extra_args: str = ""):
    """STA-35 itc_loop i5 arm A: frame-pair action encoder on teacher pseudo-labels (experiments/itc_actions/train_action_encoder.py).
    Writes results/<run_name>/action_encoder/{encoder.pt, train_log.json, codes_judge.json} in the results volume.
        TINYWORLDS_GPU=H100 modal run --detach experiments/itc_actions/modal_itc.py::train_encoder --run-name itcloop_i5_enc_itc --extra-args "--label-key code_itc"
    """
    import shlex
    import subprocess

    try:
        subprocess.run(["python", "experiments/itc_actions/train_action_encoder.py", "train", "--out-dir", f"results/{run_name}/action_encoder",
                        *shlex.split(extra_args)], cwd=REPO_DIR, check=True)
    finally:
        results_volume.commit()
