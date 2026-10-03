"""Re-run NanoWM's published Push-T evaluation on a Modal GPU and dump its predictions.

Reproduces NanoWM-B/2 100k on DINO-WM Push-T (their README: PSNR 33.19 / SSIM 0.982 /
LPIPS 0.016 / FID 13.63) with NanoWM's own code (github.com/simchowitzlabpublic/nano-world-model,
pinned commit below, deps installed from their uv.lock), and saves the predicted frames so our
eval can re-score them at our resolutions against our own ground truth.

The eval call is exactly src/scripts/eval_single_model.sh (via src/scripts/eval/dino_wm_pusht.sh):
    src/main.py experiment=evaluate_only model=nanowm_b2 dataset=dino_wm/pusht
        dataset.loader.validation_size=null dataset.loader.validation_fixed_subset_size=256
        dataset.loader.validation_fixed_subset_seed=42 dataset.loader.validation_fixed_subset_path=<json>
        experiment.resume_from_checkpoint=<ckpt> wandb.enabled=false hydra.run.dir=<dir>
i.e. 256 fixed val clips (seed 42), 250 DDIM steps, sequential scheduling, seed 3407, eval batch
size = training.batch_size = 8, bf16, torch.compile on. Deviations from their setup:
  * the HF release is `model.safetensors`; it is converted (unchanged tensors) to a bare torch
    state_dict .pt, which their `_load_checkpoint` accepts (adds the `model.` prefix itself);
  * one GPU instead of their EVAL_GPUS=0,1,2,3 (DDP) -> same clips, different sampling noise;
  * `hydra/job_logging=default` so their `Epoch end val <metric>` lines reach the log.

Launch (from the repo root; Modal spend: one GPU, roughly 15-30 min incl. image build):
    TINYWORLDS_GPU=H100 ./.venv/bin/modal run --detach scripts/eval/nanowm_rescore_modal.py
    ./.venv/bin/modal app logs tinyworlds-nanowm        # follow; outputs land on the volume even if the client disconnects

Download the outputs:
    ./.venv/bin/modal volume get --force tinyworlds-results nanowm_pusht_rescore ./results/   # -> ./results/nanowm_pusht_rescore/ (a non-existent dest path makes modal write every file onto ONE file)

Outputs (results volume, nanowm_pusht_rescore/):
    metrics.json        their psnr/ssim/lpips/fid/mse (+ val loss if logged) and run metadata
    eval_log.txt        full stdout/stderr of their eval run
    fixed_subset.json   their generated subset file ({"dataset": ..., "slices": [{traj_idx, start_frame, end_frame}]})
    predictions.npz     pred, gt: uint8 [256, 4, 256, 256, 3] (all 4 frames, context first,
                        round(clamp(x, 0, 1) * 255) of their [0, 1] tensors); traj_idx, start_frame,
                        end_frame: int64 [256]; in subset order
    hydra_config.yaml   their composed config snapshot
    predictions_f16.npz (probe, default on; `--no-probe` to skip) same layout, float16 [0,1] pred/gt
    probe.json          their Evaluator re-run in the same pass on float vs uint8-quantised pred/gt,
                        plus autocast state / dtype / value range of evaluate_all's inputs per batch
"""

import os

import modal

NANOWM_REPO = "https://github.com/simchowitzlabpublic/nano-world-model"
NANOWM_COMMIT = "cc1f151a9d8c7d7afd0cdc5de6a9ae220a7c90c0"
NANOWM_DIR = "/opt/nano-world-model"
NANOWM_VENV = "/opt/nanowm-venv"
HF_MODEL = "knightnemo/nanowm-b2-dino-wm-pusht-100k"
HF_REVISION = "613919a559470a3b551160461627a3c1ac5cc17f"
PUSHT_URL = "https://osf.io/download/k2d8w/"  # DINO-WM pusht_noise.zip (2.79 GB)

DATA_DIR = "/data"  # tinyworlds-data volume; raw data at /data/pusht_noise/{train,val}
OUT_ROOT = "/results"  # tinyworlds-results volume
OUT_NAME = "nanowm_pusht_rescore"
GPU = os.environ.get("TINYWORLDS_GPU", "H100")

NUM_EVAL_SAMPLES = 256
EVAL_SEED = 42
SUBSET_NAME = "dino_wm_pusht"

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
)

app = modal.App("tinyworlds-nanowm", image=image)
data_volume = modal.Volume.from_name("tinyworlds-data", create_if_missing=True)
results_volume = modal.Volume.from_name("tinyworlds-results", create_if_missing=True)


def _have_pusht() -> bool:
    return all(
        os.path.exists(f"{DATA_DIR}/pusht_noise/{split}/{name}")
        for split in ("train", "val")
        for name in ("seq_lengths.pkl", "states.pth", "rel_actions.pth", "velocities.pth", "obses")
    )


@app.function(volumes={DATA_DIR: data_volume}, cpu=4, memory=8192, timeout=2 * 60 * 60)
def ensure_data():
    """Download + unzip DINO-WM pusht_noise into the data volume (pusht_noise/ at its root) if missing."""
    import shutil
    import subprocess
    import zipfile

    data_volume.reload()
    if _have_pusht():
        print("pusht_noise already on the data volume, skipping download")
        return
    zip_path = "/tmp/pusht_noise.zip"
    print(f"downloading {PUSHT_URL} ...", flush=True)
    subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-o", zip_path, PUSHT_URL], check=True)
    print("extracting ...", flush=True)
    tmp_root = f"{DATA_DIR}/.pusht_noise_extract"
    shutil.rmtree(tmp_root, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp_root)
    if os.path.exists(f"{DATA_DIR}/pusht_noise"):  # partial earlier copy
        shutil.rmtree(f"{DATA_DIR}/pusht_noise")
    os.rename(f"{tmp_root}/pusht_noise", f"{DATA_DIR}/pusht_noise")
    shutil.rmtree(tmp_root, ignore_errors=True)
    data_volume.commit()
    assert _have_pusht()
    print("pusht_noise ready")


def _parse_metrics(log: str) -> dict:
    import re

    metrics = {}
    for key, val in re.findall(r"Epoch end val (\w+): ([-+0-9.eEnaif]+)", log):
        metrics[key] = float(val)
    # fallback: Lightning's "Validate metric" table (val_eval/<name> │ value)
    for key, val in re.findall(r"val_eval/(\w+)\s*[│|]\s*([-+0-9.eE]+)", log):
        metrics.setdefault(key, float(val))
    return metrics


# Prepended to a copy of their src/main.py (src/main_probe.py, so Hydra's config_path still resolves).
# Wraps utils.metrics.Evaluator.evaluate_all: the reported numbers are untouched (original call, same
# args, same order); afterwards it re-runs a second, save-less Evaluator on variants of the SAME tensors
# and records the autocast state / dtypes / value ranges of the inputs. Dumps PROBE_DIR/probe.json and
# PROBE_DIR/f16.npz (float16 [0,1] pred/gt, [N,T,H,W,C], arrival order) at validation-epoch end.
PROBE_SRC = r'''
import json as _pj, os as _pos
_pos.environ["PL_DISABLE_SUBPROCESS_LOGGING"] = "1"  # their main.py sets this before importing lightning
import numpy as _pnp
import torch as _pt
import utils.metrics as _pm
import callbacks as _pc
from pytorch_fid.fid_score import calculate_frechet_distance as _pfd

_P_ORIG = _pm.Evaluator.evaluate_all
_P = {"info": [], "acc": {}, "f16_pred": [], "f16_gt": [], "names": []}
_P_EVAL = []

def _p_q(x):  # uint8 round trip in [-1,1] space
    return _pt.round(((x.float() + 1) / 2).clamp(0, 1) * 255) / 255 * 2 - 1

def _p_acc(name, res):
    a = _P["acc"].setdefault(name, {})
    for k, v in res.items():
        a.setdefault(k, []).append(_pnp.asarray(v))

def _p_eval_all(self, x_pred, x_gt, raw, path_dict=None, evaluate=True, compute_fvd=True, n_context_frames=0):
    try:
        ac_dtype = str(_pt.get_autocast_dtype("cuda"))
    except Exception:
        ac_dtype = str(_pt.get_autocast_gpu_dtype())
    ac = bool(_pt.is_autocast_enabled())
    _P["info"].append(dict(
        autocast_cuda=ac, autocast_dtype=ac_dtype, grad_enabled=_pt.is_grad_enabled(),
        pred_dtype=str(x_pred.dtype), gt_dtype=str(x_gt.dtype), shape=list(x_pred.shape),
        pred_min=float(x_pred.min()), pred_max=float(x_pred.max()),
        gt_min=float(x_gt.min()), gt_max=float(x_gt.max()),
    ))
    res = _P_ORIG(self, x_pred, x_gt, raw, path_dict, evaluate, compute_fvd, n_context_frames)
    if res is None:
        return res
    _p_acc("a_reported", res)
    if not _P_EVAL:
      with _pt.random.fork_rng(devices=[_pt.cuda.current_device()]):  # keep sampling RNG streams untouched
        _P_EVAL.append(_pm.Evaluator(i3d_model_path=self.i3d_model_path, max_batchsize=self.max_batchsize,
                                     device=self.device, env=self.env, save_dir=None))
    ev = _P_EVAL[0]
    p32, g32 = x_pred.float(), x_gt.float()
    pq, gq = _p_q(x_pred), _p_q(x_gt)
    variants = {
        "a_float_fp32_noautocast": (p32, g32, False),
        "b_uint8_both_same_ctx": (pq, gq, None),
        "b_uint8_both_fp32_noautocast": (pq, gq, False),
        "c_uint8_gt_only_noautocast": (p32, gq, False),
        "c_uint8_pred_only_noautocast": (pq, g32, False),
    }
    for name, (p, g, ctx) in variants.items():
        if ctx is None:
            r = _P_ORIG(ev, p.clone(), g.clone(), True, None, True, compute_fvd, n_context_frames)
        else:
            with _pt.autocast("cuda", enabled=False):
                r = _P_ORIG(ev, p.clone(), g.clone(), True, None, True, compute_fvd, n_context_frames)
        _p_acc(name, r)
    to01 = lambda x: ((x.float() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 4, 1).cpu().numpy().astype(_pnp.float16)
    _P["f16_pred"].append(to01(x_pred)); _P["f16_gt"].append(to01(x_gt))
    _P["names"].extend(list(path_dict) if path_dict is not None else [None] * x_pred.shape[0])
    return res

_pm.Evaluator.evaluate_all = _p_eval_all

_P_ORIG_END = _pc.MetricsLogger.on_validation_epoch_end

def _p_epoch_end(self, trainer, pl_module):
    _P_ORIG_END(self, trainer, pl_module)
    out = {}
    for name, a in _P["acc"].items():
        m = {k: float(_pnp.mean(_pnp.stack(v))) for k, v in a.items() if k not in ("real_stats", "fake_stats")}
        if "real_stats" in a:
            rs, fs = _pnp.concatenate(a["real_stats"]), _pnp.concatenate(a["fake_stats"])
            m["fid"] = float(_pfd(rs.mean(0), _pnp.cov(rs, rowvar=False), fs.mean(0), _pnp.cov(fs, rowvar=False)))
            m["fid_frames"] = int(rs.shape[0])
        out[name] = m
    info = _P["info"]
    summary = {k: sorted({str(i[k]) for i in info}) for k in ("autocast_cuda", "autocast_dtype", "grad_enabled", "pred_dtype", "gt_dtype")}
    summary.update(pred_min=min(i["pred_min"] for i in info), pred_max=max(i["pred_max"] for i in info),
                   gt_min=min(i["gt_min"] for i in info), gt_max=max(i["gt_max"] for i in info), n_batches=len(info))
    d = _pos.environ["PROBE_DIR"]
    with open(f"{d}/probe.json", "w") as f:
        _pj.dump({"variants": out, "input_summary": summary, "per_batch_inputs": info}, f, indent=2)
    _pnp.savez(f"{d}/f16.npz", pred=_pnp.concatenate(_P["f16_pred"]), gt=_pnp.concatenate(_P["f16_gt"]),
               names=_pnp.array(_P["names"]))
    print("[probe]", _pj.dumps({"variants": out, "input_summary": summary}), flush=True)

_pc.MetricsLogger.on_validation_epoch_end = _p_epoch_end
# ---------------- their src/main.py follows ----------------
'''


@app.function(
    gpu=GPU,
    cpu=8,
    memory=65536,
    volumes={DATA_DIR: data_volume, OUT_ROOT: results_volume},
    timeout=3 * 60 * 60,
)
def evaluate(probe: bool = True):
    import glob
    import json
    import shutil
    import subprocess
    import time

    import numpy as np

    data_volume.reload()
    assert _have_pusht(), "pusht_noise missing on the data volume; run ensure_data first"

    py = f"{NANOWM_VENV}/bin/python"
    work = "/tmp/nanowm"
    results_dir = f"{work}/results"
    eval_dir = f"{results_dir}/eval"
    subset_path = f"{eval_dir}/fixed_subsets/{SUBSET_NAME}_val{NUM_EVAL_SAMPLES}.json"
    run_dir = f"{eval_dir}/{SUBSET_NAME}"
    os.makedirs(f"{eval_dir}/fixed_subsets", exist_ok=True)

    # 1. checkpoint: HF safetensors -> bare torch state_dict (tensors unchanged)
    ckpt_pt = f"{work}/ckpt/nanowm-b2-dino-wm-pusht-100k.pt"
    os.makedirs(os.path.dirname(ckpt_pt), exist_ok=True)
    subprocess.run(
        [py, "-c", (
            "import torch, sys\n"
            "from huggingface_hub import snapshot_download\n"
            "from safetensors.torch import load_file\n"
            f"d = snapshot_download('{HF_MODEL}', revision='{HF_REVISION}', allow_patterns=['config.yaml', 'model.safetensors'])\n"
            "sd = load_file(d + '/model.safetensors')\n"
            f"torch.save(sd, '{ckpt_pt}')\n"
            "print('converted', len(sd), 'tensors,', sum(v.numel() for v in sd.values()), 'params; first keys:', list(sd)[:3])\n"
        )],
        check=True,
    )

    # 2. their eval, argument-for-argument as src/scripts/eval_single_model.sh
    #    (probe: same script with PROBE_SRC prepended; reported numbers come from the untouched original call)
    probe_dir = f"{work}/probe"
    entry = "src/main.py"
    if probe:
        os.makedirs(probe_dir, exist_ok=True)
        with open(f"{NANOWM_DIR}/src/main.py") as f:
            main_src = f.read()
        with open(f"{NANOWM_DIR}/src/main_probe.py", "w") as f:
            f.write(PROBE_SRC + main_src)
        entry = "src/main_probe.py"
    cmd = [
        py, entry,
        "experiment=evaluate_only",
        "model=nanowm_b2",
        "dataset=dino_wm/pusht",
        "dataset.loader.validation_size=null",
        f"dataset.loader.validation_fixed_subset_size={NUM_EVAL_SAMPLES}",
        f"dataset.loader.validation_fixed_subset_seed={EVAL_SEED}",
        f"dataset.loader.validation_fixed_subset_path={subset_path}",
        f"experiment.resume_from_checkpoint={ckpt_pt}",
        "wandb.enabled=false",
        f"hydra.run.dir={run_dir}",
        "hydra/job_logging=default",  # their config disables logging; we need the metric lines
    ]
    env = dict(
        os.environ,
        REPO=NANOWM_DIR,
        DATASET_DIR=DATA_DIR,
        RESULTS_DIR=results_dir,
        PRETRAINED_MODELS_DIR=f"{work}/pretrained_models",
        CUDA_VISIBLE_DEVICES="0",
        WANDB_MODE="disabled",
        PYTHONUNBUFFERED="1",
        PROBE_DIR=probe_dir,
    )
    print("running:", " ".join(cmd), flush=True)
    t0 = time.time()
    lines = []
    proc = subprocess.Popen(cmd, cwd=NANOWM_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()
    wall = time.time() - t0
    log = "".join(lines)

    out = f"{OUT_ROOT}/{OUT_NAME}"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/eval_log.txt", "w") as f:
        f.write(log)
    results_volume.commit()
    if rc != 0:
        raise RuntimeError(f"NanoWM eval exited with {rc}; log at {OUT_NAME}/eval_log.txt")

    # 3. outputs
    with open(subset_path) as f:
        subset = json.load(f)
    shutil.copy(subset_path, f"{out}/fixed_subset.json")
    hydra_cfg = f"{run_dir}/.hydra/config.yaml"
    if os.path.exists(hydra_cfg):
        shutil.copy(hydra_cfg, f"{out}/hydra_config.yaml")

    slices = subset["slices"]
    step_dirs = sorted(glob.glob(f"{run_dir}/eval_videos/step_*"))
    assert len(step_dirs) == 1, step_dirs
    n, T, H, W = len(slices), 4, 256, 256
    pred = np.zeros((n, T, H, W, 3), np.uint8)
    gt = np.zeros((n, T, H, W, 3), np.uint8)
    to_u8 = lambda x: np.round(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)  # noqa: E731
    for k, s in enumerate(slices):
        raw = np.load(f"{step_dirs[0]}/traj_{s['traj_idx']:04d}_start_{s['start_frame']:04d}/raw_data.npz")
        pred[k] = to_u8(raw["pred"].transpose(1, 2, 3, 0))  # [C,T,H,W] -> [T,H,W,C]
        gt[k] = to_u8(raw["gt"].transpose(1, 2, 3, 0))
    np.savez_compressed(
        f"{out}/predictions.npz",
        pred=pred,
        gt=gt,
        traj_idx=np.array([s["traj_idx"] for s in slices], np.int64),
        start_frame=np.array([s["start_frame"] for s in slices], np.int64),
        end_frame=np.array([s["end_frame"] for s in slices], np.int64),
    )

    probe_res = None
    if probe:
        # float16 [0,1] pred/gt in subset order (same layout as predictions.npz)
        f16 = np.load(f"{probe_dir}/f16.npz")
        idx = {str(nm): i for i, nm in enumerate(f16["names"])}
        order = [idx[f"traj_{s['traj_idx']:04d}_start_{s['start_frame']:04d}"] for s in slices]
        np.savez(
            f"{out}/predictions_f16.npz",
            pred=f16["pred"][order],
            gt=f16["gt"][order],
            traj_idx=np.array([s["traj_idx"] for s in slices], np.int64),
            start_frame=np.array([s["start_frame"] for s in slices], np.int64),
            end_frame=np.array([s["end_frame"] for s in slices], np.int64),
        )
        shutil.copy(f"{probe_dir}/probe.json", f"{out}/probe.json")
        with open(f"{probe_dir}/probe.json") as f:
            probe_res = json.load(f)
            probe_res.pop("per_batch_inputs", None)

    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    metrics = _parse_metrics(log)
    payload = {
        "metrics": metrics,
        "probe": probe_res,
        "published": {"psnr": 33.19, "ssim": 0.982, "lpips": 0.016, "fid": 13.63},
        "num_clips": n,
        "gpu": gpu_name,
        "eval_wall_time_s": round(wall, 1),
        "nanowm_commit": NANOWM_COMMIT,
        "hf_model": HF_MODEL,
        "hf_revision": HF_REVISION,
        "command": " ".join(cmd),
        "notes": "1 GPU (theirs: 4-GPU DDP); safetensors converted to bare state_dict .pt; "
                 "metrics on predicted frames only (frames 1..3), FID pooled over 768 frames.",
    }
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    results_volume.commit()
    print(json.dumps(payload, indent=2))
    return payload


@app.local_entrypoint()
def main(probe: bool = True):
    ensure_data.remote()
    print(evaluate.remote(probe=probe))
