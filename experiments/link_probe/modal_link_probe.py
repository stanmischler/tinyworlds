"""Train small probes that read Link's motion from each stage of the action model (frames -> frozen MAE -> IDM -> z), on Modal.

    ./.venv/bin/modal run experiments/link_probe/modal_link_probe.py::probe_smoke     # small subset, few steps
    ./.venv/bin/modal run experiments/link_probe/modal_link_probe.py::run_all      # prep, then every probe in parallel
    ./.venv/bin/modal volume get tinyworlds-results link_probe/ <dir>/          # then experiments/link_probe/report.py

prep writes the pairs, their frames, MAE features and z to results/link_probe/ once; every probe task saves its out-of-fold
predictions to results/link_probe/tasks/ and is skipped if already there, so a relaunch resumes."""

import os
import time

import modal
import numpy as np

from scripts.infra.modal_train import REPO_DIR, VOLUMES, app, committing, results_volume

PROBE_GPU = "T4"        # the nets are tiny; the cheapest GPU fits
ROOT = f"{REPO_DIR}/results/link_probe"
SHORT_CKPT = "results/como_zelda_v1/como/checkpoints/como_step_50000"            # 20 min of play
LONG_CKPT = "results/zelda_more_data_como/como/checkpoints/como_step_100000"     # 5 h of play

GRIDS = ['raw pixels', 'raw pixels around Link', 'MAE patches', 'MAE patches around Link']
GRIDS_GAP = ['raw pixels, last 4 frames only', 'MAE patches, last 4 frames only']
VECS = ['MAE averaged', 'z 20 min | two steps', 'z 5 h | two steps', 'IDM output 20 min | two steps', 'IDM output 5 h | two steps']
VECS_GAP = ['z 20 min | last step', 'z 5 h | last step', 'z 20 min | direct 8 frames', 'z 5 h | direct 8 frames', 'z 20 min | two steps summed']
SETS = ['link, camera still', 'link, scroll', 'link, all windows', 'camera, scroll', 'camera, scroll, same count']
SETS_GAP = ['link, camera still', 'link, scroll']


def safe(s):
    return s.replace(' | ', '_').replace(', ', '_').replace(' ', '_')


@app.function(gpu=PROBE_GPU, cpu=4, memory=16384, volumes=VOLUMES, timeout=30 * 60)
def probe_prep(root: str = ROOT, limit: int = 0):
    """Pairs + labels + frames + MAE features of the pairs + z and IDM output of both CoMo checkpoints."""
    import h5py
    import torch

    from evaluation.lam_vis.data import read_rows
    from experiments.link_probe.labels import H5, build_pairs
    from models.como import CoMo
    from utils.utils import COMO_ARCH_KEYS, MODEL_CHECKPOINT, STATE

    os.chdir(REPO_DIR)
    os.makedirs(root, exist_ok=True)
    if os.path.exists(f'{root}/mae.npy'):
        return
    t0 = time.time()
    p = build_pairs(limit=limit or None)
    rows = p['rows']
    print(f"prep: {len(rows)} pairs of {p['n_consecutive_pairs']}, still {p['still'].sum()}, scroll {p['scroll'].sum()}, labels {time.time() - t0:.0f} s", flush=True)
    with h5py.File(H5, 'r') as f:
        frames = np.stack([read_rows(f['frames'], rows[:, k]) for k in range(3)], 1)  # uint8 [N, 3, H, W, C]
    mm = np.load('data/zelda_test_mae_large.npy', mmap_mode='r')
    assert mm.shape[1:] == (197, 1024), f'expected per-patch MAE features, got {mm.shape}'
    assert mm.shape[0] == 7240, mm.shape
    mae = np.stack([np.asarray(mm[rows[:, k]]) for k in range(3)], 1)  # float16 [N, 3, S, 1024]
    print(f'prep: MAE {mae.shape} read, {time.time() - t0:.0f} s', flush=True)
    device = 'cuda'
    out = {}
    for short, ck in (('short', SHORT_CKPT), ('long', LONG_CKPT)):
        cfg = torch.load(f'{ck}/{STATE}', map_location='cpu', weights_only=False)['config']
        assert cfg['gap'] == 4, cfg['gap']
        como = CoMo(**{k: cfg[k] for k in COMO_ARCH_KEYS if k in cfg})
        como.load_state_dict(torch.load(f'{ck}/{MODEL_CHECKPOINT}', map_location='cpu', weights_only=True))
        idm = como.idm.to(device).eval()

        @torch.no_grad()
        def run(a, b, bs=32):
            zs, hs = [], []
            for i in range(0, len(a), bs):
                fa, fb = (torch.from_numpy(x[i:i + bs]).float().to(device) for x in (a, b))
                h = idm.tokens(fa, fb - fa)  # [B, Q, E], before the final projection
                zs.append(idm.down(h).flatten(1).cpu().numpy()), hs.append(h.flatten(1).cpu().numpy())
            return np.concatenate(zs), np.concatenate(hs)
        for part, (i, j) in {'first': (0, 1), 'second': (1, 2), 'direct': (0, 2)}.items():
            z, h = run(mae[:, i], mae[:, j])
            out[f'z_{short}_{part}'], out[f'idm_{short}_{part}'] = z.astype(np.float32), h.astype(np.float16)
        print(f'prep: {ck} done, {time.time() - t0:.0f} s', flush=True)
        del idm, como
    np.savez(f'{root}/pairs.npz', frames=frames, **p, **out)
    np.save(f'{root}/mae.npy', mae)
    results_volume.commit()
    print(f'prep done in {time.time() - t0:.0f} s', flush=True)


_DATA = {}


def data(root):
    if root not in _DATA:
        results_volume.reload()
        d = dict(np.load(f'{root}/pairs.npz'))
        d['mae'] = np.load(f'{root}/mae.npy')
        _DATA[root] = d
    return _DATA[root]


def run_task(root, input_name, set_name, method, is_shuffled, device, seeds, steps):
    import torch

    from experiments.link_probe import probes

    tag = f"{safe(input_name)}__{safe(set_name)}__{method}__{'shuffled' if is_shuffled else 'real'}"
    path = f'{root}/tasks/{tag}.npz'
    results_volume.reload()
    if os.path.exists(path):
        return tag, 'skipped', 0.0
    t0 = time.time()
    torch.set_num_threads(os.cpu_count() or 4)
    d = data(root)
    kind, x = probes.load_input(input_name, d)
    kernels = probes.linear_kernels(kind, x) if method == 'ridge' else None
    runs = {'idx': [], 'y': [], 'dir': [], 'ok': [], 'pred': [], 'dir_pred': [], 'train_r2': []}
    for s in range(seeds):
        idx, y, dirs, ok = probes.task_targets(set_name, d, s)
        if is_shuffled:
            y, dirs, ok = probes.shuffled(y, dirs, ok, s)
        if method == 'ridge':
            pred, dir_pred = probes.ridge_oof(kernels, idx, d['block'], y, dirs, ok)
            tr2 = np.nan
        else:
            pred, dir_pred, tr2 = probes.net_oof(kind, x, idx, d['block'], y, dirs, ok, s, device, steps)
        for k, v in zip(runs, (idx, y, dirs, ok, pred, dir_pred, tr2)):
            runs[k].append(v)
    os.makedirs(f'{root}/tasks', exist_ok=True)
    np.savez(path, input=input_name, set=set_name, method=method, shuffled=is_shuffled, seconds=time.time() - t0,
             **{k: np.stack(v) for k, v in runs.items()})
    results_volume.commit()
    print(f'{tag}: {time.time() - t0:.0f} s', flush=True)
    return tag, 'done', time.time() - t0


@app.function(gpu=PROBE_GPU, cpu=4, memory=16384, volumes=VOLUMES, timeout=2 * 60 * 60, max_containers=6, scaledown_window=60, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def probe_gpu(root: str, input_name: str, set_name: str, method: str, is_shuffled: bool, seeds: int = 3, steps: int = 0):
    return run_task(root, input_name, set_name, method, is_shuffled, 'cuda', seeds, steps or None)


@app.function(cpu=8, memory=24576, volumes=VOLUMES, timeout=2 * 60 * 60, max_containers=10, scaledown_window=60, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def probe_cpu(root: str, input_name: str, set_name: str, method: str, is_shuffled: bool, seeds: int = 3, steps: int = 0):
    return run_task(root, input_name, set_name, method, is_shuffled, 'cpu', seeds, steps or None)


def plan():
    # (function, input, set, method, seeds): nets on the GPU; ridge and MLP on the CPU
    from experiments.link_probe.probes import NET_SEEDS, RIDGE_SHUFFLES
    gpu, cpu = [], []
    for inp in GRIDS:
        gpu += [(inp, st, 'net', NET_SEEDS) for st in SETS]
        cpu += [(inp, st, 'ridge', RIDGE_SHUFFLES) for st in SETS]
    for inp in GRIDS_GAP:
        gpu += [(inp, st, 'net', NET_SEEDS) for st in SETS_GAP]
        cpu += [(inp, st, 'ridge', RIDGE_SHUFFLES) for st in SETS_GAP]
    for inp in VECS:
        cpu += [(inp, st, m, s) for st in SETS for m, s in (('ridge', RIDGE_SHUFFLES), ('mlp', NET_SEEDS))]
    for inp in VECS_GAP:
        cpu += [(inp, st, m, s) for st in SETS_GAP for m, s in (('ridge', RIDGE_SHUFFLES), ('mlp', NET_SEEDS))]
    return gpu, cpu


def fan_out(root, gpu, cpu, steps=0):
    # a ridge fit on real labels is deterministic: more than one run only when the windows are drawn or the labels shuffled
    runs_ridge = lambda st, m, sh, s: 1 if m == 'ridge' and not sh and 'same count' not in st else s
    # nets: the shuffled control gets 2 seeds (GPU budget); a shuffled label still gets the full fold set
    runs = lambda st, m, sh, s: 2 if m == 'net' and sh and s > 2 else runs_ridge(st, m, sh, s)
    jobs = lambda items: [(root, i, st, m, sh, runs(st, m, sh, s), steps) for i, st, m, s in items for sh in (False, True)]
    print(f'{len(jobs(gpu))} GPU tasks, {len(jobs(cpu))} CPU tasks', flush=True)
    t0 = time.time()
    calls = [probe_gpu.spawn(*j) for j in jobs(gpu)] + [probe_cpu.spawn(*j) for j in jobs(cpu)]
    done = 0
    failed = 0
    for c in calls:  # one lost task must not stop the others: its output is missing, so a relaunch runs it again
        done += 1
        try:
            tag, state, sec = c.get()
            print(f'[{done}/{len(calls)}] {time.time() - t0:.0f} s  {state} {sec:.0f} s  {tag}', flush=True)
        except Exception as e:
            failed += 1
            print(f'[{done}/{len(calls)}] FAILED {c.object_id}: {type(e).__name__} {e}', flush=True)
    print(f'{failed} failed', flush=True)


@app.local_entrypoint()
def probe_smoke():
    root = f'{ROOT}/smoke'
    probe_prep.remote(root, 120)
    fan_out(root, [(GRIDS[1], 'link, camera still', 'net', 1), (GRIDS[2], 'camera, scroll', 'net', 1)],
            [(GRIDS[2], 'link, camera still', 'ridge', 1), (VECS[1], 'camera, scroll', 'ridge', 1), (VECS[1], 'link, camera still', 'mlp', 1),
             (VECS[3], 'link, all windows', 'ridge', 1)], steps=60)


@app.local_entrypoint()
def run_all():
    gpu, cpu = plan()
    probe_prep.remote(ROOT, 0)
    fan_out(ROOT, gpu, cpu)
