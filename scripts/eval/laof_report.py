"""STA-39 final LAM report on the lam_judge set: seed spread of the k-means codes + direction vs position diagnostic.

For each checkpoint: encodes every held-out judge window once, then
  - judge moves / all NMI_adj (lam_judge.judged_metrics) for `--seeds` k-means seeds (continuous actions; discrete
    codes are deterministic, one row), reported as mean +- std
  - on all held-out transitions with flow-oracle pseudo-labels (camera-masked flow, laof_flow_oracle): NMI(code,
    direction | moving) vs NMI(code, Link's 4x4 screen cell | moving), and 5-fold ridge-probe accuracy latent ->
    direction / cell (does the latent carry direction, and does it also carry position?)

    python scripts/eval/laof_report.py --lam disc=<ckpt> --lam cont=<ckpt> [--seeds 5]   # -> eval_results/laof_report.json
"""

import argparse
import json
import math
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import lam_judge  # noqa: E402
from eval_lam import nmi, kmeans  # noqa: E402
from eval_next_frame import load_window_batch, to_model_range  # noqa: E402
from models.laof import camera_mask  # noqa: E402
from utils.utils import load_latent_actions_from_checkpoint  # noqa: E402


def pseudo_labels(wins, flow_h5):
    # flow-oracle direction (0 still, 1-8 sectors) and 4x4 screen cell of the moving region (-1 still) per transition
    D, P = [], []
    with h5py.File(lam_judge.H5, 'r') as f, h5py.File(flow_h5, 'r') as g:
        for _, s in wins:
            for t in range(lam_judge.SEQ - 1):
                a = s + t * lam_judge.SKIP
                fl = g['flow'][a].astype(np.float32)
                fr = torch.from_numpy(np.stack([f['frames'][a], f['frames'][a + lam_judge.SKIP]])).permute(0, 3, 1, 2).float() / 127.5 - 1
                m = camera_mask(fr[:1], fr[1:], torch.from_numpy(g['shift'][a][None]))[0].numpy() & (np.linalg.norm(fl, axis=0) > 1)
                if m.sum() < 20:
                    D.append(0)
                    P.append(-1)
                    continue
                D.append(1 + int(round(math.atan2(fl[1][m].mean(), fl[0][m].mean()) / (math.pi / 4))) % 8)
                ys, xs = np.nonzero(m)
                P.append(int(ys.mean() // 32) * 4 + int(xs.mean() // 32))
    return np.array(D), np.array(P)


def table(a, b, na, nb):
    t = np.zeros((na, nb))
    np.add.at(t, (a, b), 1)
    return t


def ridge_acc(X, y, n_cls, lam=10.0):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = np.c_[X, np.ones(len(X))]
    idx = np.random.default_rng(0).permutation(len(X))
    acc = []
    for k in range(5):
        te = idx[k::5]
        tr = np.setdiff1d(idx, te)
        W = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ np.eye(n_cls)[y[tr]])
        acc.append(float((np.argmax(X[te] @ W, 1) == y[te]).mean()))
    return round(float(np.mean(acc)), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lam', action='append', required=True, help='name=<checkpoint dir>')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--flow', default='data/zelda_test_flow_gap4.h5')
    p.add_argument('--out', default='eval_results/laof_report.json')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    meta = json.load(open(f'{lam_judge.SET}/transitions.json'))
    labels = json.load(open(f'{lam_judge.SET}/labels.json'))
    wins = lam_judge.test_windows(lam_judge.H5, lam_judge.SEQ - 1, lam_judge.SKIP, lam_judge.STRIDE)
    D, P = pseudo_labels(wins, a.flow)
    mv = D > 0
    report = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for spec in a.lam:
        name, ckpt = spec.split('=', 1)
        lam, _ = load_latent_actions_from_checkpoint(ckpt, a.device)
        lam.eval()
        with h5py.File(lam_judge.H5, 'r') as h5, torch.no_grad():
            z = torch.cat([lam.encode(to_model_range(load_window_batch(h5['frames'], wins[i:i + 32], lam_judge.SEQ - 1, lam_judge.SKIP), a.device))
                           for i in range(0, len(wins), 32)]).float()  # [N, T-1, A]
        flat = z.reshape(-1, z.shape[-1])
        cont = getattr(lam, 'continuous_actions', False)
        k = lam.quantizer.codebook_size
        rows = []
        for seed in range(a.seeds if cont else 1):
            if cont:
                codes = torch.cdist(flat, kmeans(flat, k, seed=seed)).argmin(1).numpy()
            else:
                codes = lam.quantizer.get_indices_from_latents(flat).cpu().numpy()
            r = lam_judge.judged_metrics(meta, labels, k, codes_all=codes.reshape(z.shape[:2]))
            rows.append({'moves': r['moves_only']['nmi_adj'], 'all': r['all']['nmi_adj'], 'codes': codes})
        mv_ = np.array([x['moves'] for x in rows])
        al_ = np.array([x['all'] for x in rows])
        c0 = rows[0]['codes']
        res = {'ckpt': ckpt, 'continuous': bool(cont), 'n_seeds': len(rows),
               'moves_nmi_adj_mean': round(float(mv_.mean()), 4), 'moves_nmi_adj_std': round(float(mv_.std()), 4),
               'all_nmi_adj_mean': round(float(al_.mean()), 4), 'all_nmi_adj_std': round(float(al_.std()), 4),
               'per_seed_moves': mv_.tolist(), 'per_seed_all': al_.tolist(),
               'nmi_code_direction_moving': round(nmi(table(c0[mv], D[mv] - 1, k, 8)), 4),
               'nmi_code_cell_moving': round(nmi(table(c0[mv], P[mv], k, 16)), 4),
               'probe_direction_acc': ridge_acc(flat.numpy()[mv], D[mv] - 1, 8),
               'probe_cell_acc': ridge_acc(flat.numpy()[mv], P[mv], 16),
               'direction_majority': round(float(np.bincount(D[mv] - 1).max() / mv.sum()), 4),
               'cell_majority': round(float(np.bincount(P[mv]).max() / mv.sum()), 4)}
        report[name] = res
        print(name, {key: v for key, v in res.items() if key not in ('ckpt', 'per_seed_moves', 'per_seed_all')}, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
