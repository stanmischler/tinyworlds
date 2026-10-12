"""Shared driver for re-running NanoWM's published DINO-WM evaluation and dumping its predictions in our format, used by
nanowm_rescore_modal.py (Modal GPU) and nanowm_rescore_local.sh (plain Linux GPU box).

Standard library + numpy only: on the local box it runs with NanoWM's own venv python, from the repo root:
    <nanowm-venv>/bin/python -m experiments.pusht_nanowm.nanowm_dump --env point_maze --nanowm-dir ... --py ... \
        --dataset-dir ... --work ... --out ...

The eval call is exactly their src/scripts/eval_single_model.sh (via src/scripts/eval/dino_wm_<env>.sh):
    src/main.py experiment=evaluate_only model=nanowm_b2 dataset=dino_wm/<env>
        dataset.loader.validation_size=null dataset.loader.validation_fixed_subset_size=256
        dataset.loader.validation_fixed_subset_seed=42 dataset.loader.validation_fixed_subset_path=<json>
        experiment.resume_from_checkpoint=<ckpt> wandb.enabled=false hydra.run.dir=<dir>
i.e. 256 fixed val clips (seed 42), 250 DDIM steps, sequential scheduling, seed 3407, eval batch size 8, bf16,
torch.compile on. Deviations from their setup:
  * the HF release is `model.safetensors`; it is converted (unchanged tensors) to a bare torch state_dict .pt, which
    their `_load_checkpoint` accepts (adds the `model.` prefix itself);
  * one GPU instead of their EVAL_GPUS=0,1,2,3 (DDP) -> same clips, different sampling noise;
  * `hydra/job_logging=default` so their `Epoch end val <metric>` lines reach the log.
The subset json is always regenerated (deleted first): when it already exists their eval takes a fast path
(create_eval_only_dataset) that computes the Point Maze action stats over ALL trajectories instead of the train split.

Outputs (in --out):
    metrics.json        their psnr/ssim/lpips/fid/mse and run metadata
    eval_log.txt        full stdout/stderr of their eval run
    fixed_subset.json   their generated subset file ({"dataset": ..., "slices": [{traj_idx, start_frame, end_frame}]})
    predictions.npz     pred, gt: uint8 [256, 4, 256, 256, 3] (all 4 frames, context first, round(clamp(x, 0, 1) * 255)
                        of their [0, 1] tensors); traj_idx, start_frame, end_frame: int64 [256]; in subset order
    hydra_config.yaml   their composed config snapshot
    predictions_f16.npz (probe, default on) same layout, float16 [0, 1] pred/gt
    probe.json          their Evaluator re-run in the same pass on float vs uint8-quantised pred/gt, plus autocast
                        state / dtype / value range of evaluate_all's inputs per batch
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np

NANOWM_REPO = "https://github.com/simchowitzlabpublic/nano-world-model"
NANOWM_COMMIT = "cc1f151a9d8c7d7afd0cdc5de6a9ae220a7c90c0"
NUM_EVAL_SAMPLES = 256
EVAL_SEED = 42

# per env: their dataset config, the raw DINO-WM folder under DATASET_DIR (+ files that must exist), its OSF zip, the
# released checkpoint and the published numbers (256px, 1 -> 3 frames)
ENVS = {
    "pusht": dict(
        dataset="dino_wm/pusht", subset="dino_wm_pusht", folder="pusht_noise", url="https://osf.io/download/k2d8w/",
        required=[f"{s}/{n}" for s in ("train", "val") for n in ("seq_lengths.pkl", "states.pth", "rel_actions.pth", "velocities.pth", "obses")],
        hf_model="knightnemo/nanowm-b2-dino-wm-pusht-100k", hf_revision="613919a559470a3b551160461627a3c1ac5cc17f",
        published={"psnr": 33.19, "ssim": 0.982, "lpips": 0.016, "fid": 13.63}),
    "point_maze": dict(
        dataset="dino_wm/point_maze", subset="dino_wm_point_maze", folder="point_maze", url="https://osf.io/download/vr5gy/",
        required=["states.pth", "actions.pth", "seq_lengths.pth", "obses"],
        hf_model="knightnemo/nanowm-b2-dino-wm-point-maze-30k", hf_revision="b84303b7151cbd1836841c3a87bf095e1015b150",
        published={"psnr": 36.74, "ssim": 0.984, "lpips": 0.019, "fid": 9.66}),
    "wall": dict(
        dataset="dino_wm/wall", subset="dino_wm_wall", folder="wall_single", url="https://osf.io/download/49rnx/",
        required=["states.pth", "actions.pth", "obses"],
        hf_model="knightnemo/nanowm-b2-dino-wm-wall-15k", hf_revision="ffa198f1efa8942b6026cd185cd0c2a817ca9a43",
        published={"psnr": 34.05, "ssim": 0.994, "lpips": 0.010, "fid": 2.64}),
}


def have_data(dataset_dir, env):
    return all(os.path.exists(f"{dataset_dir}/{ENVS[env]['folder']}/{n}") for n in ENVS[env]["required"])


def ensure_data(dataset_dir, env, zip_dir=None):
    """Download (into zip_dir, default dataset_dir) + unzip the env's DINO-WM zip into dataset_dir/<folder> unless it is
    already there. -> True if fetched."""
    import zipfile

    if have_data(dataset_dir, env):
        print(f"{ENVS[env]['folder']} already in {dataset_dir}, skipping download")
        return False
    spec = ENVS[env]
    os.makedirs(dataset_dir, exist_ok=True)
    zip_path = f"{zip_dir or dataset_dir}/.{spec['folder']}.zip"
    print(f"downloading {spec['url']} ...", flush=True)
    subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", zip_path, spec["url"]], check=True)
    print("extracting ...", flush=True)
    tmp_root = f"{dataset_dir}/.{spec['folder']}_extract"
    shutil.rmtree(tmp_root, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp_root)
    shutil.rmtree(f"{dataset_dir}/{spec['folder']}", ignore_errors=True)  # partial earlier copy
    os.rename(f"{tmp_root}/{spec['folder']}", f"{dataset_dir}/{spec['folder']}")
    shutil.rmtree(tmp_root, ignore_errors=True)
    os.remove(zip_path)
    assert have_data(dataset_dir, env)
    print(f"{spec['folder']} ready")
    return True


def parse_metrics(log):
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


def convert_checkpoint(py, env, ckpt_pt):
    """HF safetensors -> bare torch state_dict .pt (tensors unchanged), with their venv's python."""
    spec = ENVS[env]
    os.makedirs(os.path.dirname(ckpt_pt), exist_ok=True)
    subprocess.run(
        [py, "-c", (
            "import torch, sys\n"
            "from huggingface_hub import snapshot_download\n"
            "from safetensors.torch import load_file\n"
            f"d = snapshot_download('{spec['hf_model']}', revision='{spec['hf_revision']}', allow_patterns=['config.yaml', 'model.safetensors'])\n"
            "sd = load_file(d + '/model.safetensors')\n"
            f"torch.save(sd, '{ckpt_pt}')\n"
            "print('converted', len(sd), 'tensors,', sum(v.numel() for v in sd.values()), 'params; first keys:', list(sd)[:3])\n"
        )],
        check=True,
    )


def run_rescore(env, nanowm_dir, py, dataset_dir, work, out, probe=True, on_log=None):
    """Convert the checkpoint, run their eval argument-for-argument as eval_single_model.sh (probe: same script with
    PROBE_SRC prepended; reported numbers come from the untouched original call), dump everything into `out`.
    on_log(): called once eval_log.txt is written (Modal commits the volume there). -> metrics.json payload."""
    spec = ENVS[env]
    assert have_data(dataset_dir, env), f"{spec['folder']} missing in {dataset_dir}"
    results_dir = f"{work}/results"
    eval_dir = f"{results_dir}/eval"
    subset_path = f"{eval_dir}/fixed_subsets/{spec['subset']}_val{NUM_EVAL_SAMPLES}.json"
    run_dir = f"{eval_dir}/{spec['subset']}"
    os.makedirs(f"{eval_dir}/fixed_subsets", exist_ok=True)
    if os.path.exists(subset_path):  # force their slow path (train-split action stats), see the module docstring
        os.remove(subset_path)
    shutil.rmtree(f"{run_dir}/eval_videos", ignore_errors=True)

    ckpt_pt = f"{work}/ckpt/{spec['hf_model'].split('/')[-1]}.pt"
    convert_checkpoint(py, env, ckpt_pt)

    probe_dir = f"{work}/probe"
    entry = "src/main.py"
    if probe:
        os.makedirs(probe_dir, exist_ok=True)
        with open(f"{nanowm_dir}/src/main.py") as f:
            main_src = f.read()
        with open(f"{nanowm_dir}/src/main_probe.py", "w") as f:
            f.write(PROBE_SRC + main_src)
        entry = "src/main_probe.py"
    cmd = [
        py, entry,
        "experiment=evaluate_only",
        "model=nanowm_b2",
        f"dataset={spec['dataset']}",
        "dataset.loader.validation_size=null",
        f"dataset.loader.validation_fixed_subset_size={NUM_EVAL_SAMPLES}",
        f"dataset.loader.validation_fixed_subset_seed={EVAL_SEED}",
        f"dataset.loader.validation_fixed_subset_path={subset_path}",
        f"experiment.resume_from_checkpoint={ckpt_pt}",
        "wandb.enabled=false",
        f"hydra.run.dir={run_dir}",
        "hydra/job_logging=default",  # their config disables logging; we need the metric lines
    ]
    env_vars = dict(
        {k: v for k, v in os.environ.items() if k != "PYTHONPATH"},  # our repo's datasets/ would shadow HF datasets
        REPO=nanowm_dir,
        DATASET_DIR=dataset_dir,
        RESULTS_DIR=results_dir,
        PRETRAINED_MODELS_DIR=f"{work}/pretrained_models",
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        WANDB_MODE="disabled",
        PYTHONUNBUFFERED="1",
        PROBE_DIR=probe_dir,
    )
    print("running:", " ".join(cmd), flush=True)
    t0 = time.time()
    lines = []
    proc = subprocess.Popen(cmd, cwd=nanowm_dir, env=env_vars, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()
    wall = time.time() - t0
    log = "".join(lines)

    os.makedirs(out, exist_ok=True)
    with open(f"{out}/eval_log.txt", "w") as f:
        f.write(log)
    if on_log is not None:
        on_log()
    if rc != 0:
        raise RuntimeError(f"NanoWM eval exited with {rc}; log at {out}/eval_log.txt")

    n, probe_res = dump_outputs(out, subset_path, run_dir, probe_dir if probe else None)
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    payload = {
        "env": env,
        "metrics": parse_metrics(log),
        "probe": probe_res,
        "published": spec["published"],
        "num_clips": n,
        "gpu": gpu_name,
        "eval_wall_time_s": round(wall, 1),
        "nanowm_commit": NANOWM_COMMIT,
        "hf_model": spec["hf_model"],
        "hf_revision": spec["hf_revision"],
        "command": " ".join(cmd),
        "notes": "1 GPU (theirs: 4-GPU DDP); safetensors converted to bare state_dict .pt; "
                 "metrics on predicted frames only (frames 1..3), FID pooled over 768 frames.",
    }
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    return payload


def dump_outputs(out, subset_path, run_dir, probe_dir=None):
    """Their per-clip raw_data.npz + probe dump -> predictions.npz / predictions_f16.npz in subset order. -> (n, probe)."""
    with open(subset_path) as f:
        subset = json.load(f)
    shutil.copy(subset_path, f"{out}/fixed_subset.json")
    hydra_cfg = f"{run_dir}/.hydra/config.yaml"
    if os.path.exists(hydra_cfg):
        shutil.copy(hydra_cfg, f"{out}/hydra_config.yaml")

    slices = subset["slices"]
    ids = dict(traj_idx=np.array([s["traj_idx"] for s in slices], np.int64),
               start_frame=np.array([s["start_frame"] for s in slices], np.int64),
               end_frame=np.array([s["end_frame"] for s in slices], np.int64))
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
    np.savez_compressed(f"{out}/predictions.npz", pred=pred, gt=gt, **ids)

    probe_res = None
    if probe_dir is not None:
        # float16 [0,1] pred/gt in subset order (same layout as predictions.npz)
        f16 = np.load(f"{probe_dir}/f16.npz")
        idx = {str(nm): i for i, nm in enumerate(f16["names"])}
        order = [idx[f"traj_{s['traj_idx']:04d}_start_{s['start_frame']:04d}"] for s in slices]
        np.savez(f"{out}/predictions_f16.npz", pred=f16["pred"][order], gt=f16["gt"][order], **ids)
        shutil.copy(f"{probe_dir}/probe.json", f"{out}/probe.json")
        with open(f"{probe_dir}/probe.json") as f:
            probe_res = json.load(f)
            probe_res.pop("per_batch_inputs", None)
    return n, probe_res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=sorted(ENVS))
    p.add_argument("--nanowm-dir", required=True, help="their repo checked out at NANOWM_COMMIT")
    p.add_argument("--py", required=True, help="their venv's python")
    p.add_argument("--dataset-dir", required=True, help="DATASET_DIR: holds <folder>/ (downloaded if missing)")
    p.add_argument("--work", required=True, help="scratch dir (checkpoint, their results dir, probe dump)")
    p.add_argument("--out", required=True)
    p.add_argument("--no-probe", action="store_true")
    args = p.parse_args()
    ensure_data(args.dataset_dir, args.env)
    run_rescore(args.env, os.path.abspath(args.nanowm_dir), os.path.abspath(args.py), os.path.abspath(args.dataset_dir),
                os.path.abspath(args.work), os.path.abspath(args.out), probe=not args.no_probe)


if __name__ == "__main__":
    main()
