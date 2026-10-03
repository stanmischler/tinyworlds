"""CoMo actions for the dynamics model (STA-42): run a trained CoMo IDM over every gap-n frame pair of the train/test .h5
from the precomputed MAE features (scripts/como_features.py), so dynamics training never runs the ViT-L.

Outputs
  - data/<split>_como_z<suffix>.npy  float32 [N, A=Q*L]: row i = raw z(frame i, frame i+gap); the last `gap` rows are 0
    (no pair; no training clip reaches them). Dynamics training reads it through `action_file` (datasets.py).
  - <out-dir>_{full,k<K>}/state.pt  action dirs for `latent_actions_path` (utils.load_como_actions -> models.como.CoMoActions):
    como_path, mode, mean/std of train z (per dim), K k-means centroids of train z (seed 0). Both dirs hold the same stats;
    mode decides whether z is snapped to its centroid before standardizing.
  - <out-dir>_full/summary.json  code histogram on train/test, and the --check agreement numbers.

    python scripts/como_actions.py --como-ckpt results/como_zelda_v1/como/checkpoints/como_step_50000 --out-dir results/como_actions_v1
    python scripts/como_actions.py ... --check   # online encode (MAE on pixels) vs the npy rows on the first test pairs
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eval'))
from eval_lam import kmeans  # noqa: E402
from utils.utils import load_como_from_checkpoint, STATE  # noqa: E402


@torch.no_grad()
def pair_z(idm, feats, gap, batch, device, limit=0):
    # feats: memmap fp16 [N, S, 1024] -> raw z [N, A] float32 (rows >= N - gap stay 0)
    n = min(len(feats), limit) if limit else len(feats)
    z = np.zeros((n, idm.n_queries * idm.down[-1].out_features), np.float32)
    t0 = time.time()
    for i in range(0, n - gap, batch):
        j = min(i + batch, n - gap)
        fa = torch.from_numpy(np.asarray(feats[i:j])).to(device).float()  # [b, S, 1024]
        fb = torch.from_numpy(np.asarray(feats[i + gap:j + gap])).to(device).float()
        z[i:j] = idm(fa, fb).flatten(1).cpu().numpy()
        if (i // batch) % 50 == 0:
            print(f'  {j}/{n - gap} pairs, {time.time() - t0:.0f} s', flush=True)
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--como-ckpt', required=True)
    p.add_argument('--out-dir', required=True, help='action dirs are <out-dir>_full and <out-dir>_k<K>')
    p.add_argument('--splits', default='zelda_train,zelda_test')
    p.add_argument('--fit-split', default='zelda_train', help='split whose z gives mean/std and the k-means centroids')
    p.add_argument('--start-index', type=int, default=1000, help='fit rows from here on (ZeldaDataset load_start_index)')
    p.add_argument('--gap', type=int, default=4)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--limit', type=int, default=0, help='only the first N rows of each split (smoke)')
    p.add_argument('--suffix', default='', help='output npy suffix (smoke)')
    p.add_argument('--feature-suffix', default='', help='MAE npy suffix, e.g. _smoke')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--check', type=int, default=0, help='>0: also compare online encode with the npy on N test pairs')
    a = p.parse_args()

    lam, _ = load_como_from_checkpoint(a.como_ckpt, a.device)  # CoMoLAM (loads the MAE too, used by --check)
    idm = lam.como.idm.eval()
    zs = {}
    for sp in [s for s in a.splits.split(',') if s]:
        feats = np.load(f'data/{sp}_mae_large{a.feature_suffix}.npy', mmap_mode='r')
        print(f'{sp}: {len(feats)} feature rows')
        zs[sp] = pair_z(idm, feats, a.gap, a.batch, a.device, a.limit)
        np.save(f'data/{sp}_como_z{a.suffix}.npy', zs[sp])
        print(f'wrote data/{sp}_como_z{a.suffix}.npy {zs[sp].shape}')

    fit = torch.from_numpy(zs[a.fit_split][a.start_index:len(zs[a.fit_split]) - a.gap])  # [M, A] valid pairs only
    mean, std = fit.mean(0), fit.std(0).clamp_min(1e-6)
    centroids = kmeans(fit, a.k, seed=0)  # [K, A]
    summary = {'como_ckpt': a.como_ckpt, 'gap': a.gap, 'k': a.k, 'fit_split': a.fit_split, 'fit_rows': len(fit),
               'z_std_mean': float(std.mean()), 'z_std_min': float(std.min()), 'z_std_max': float(std.max())}
    for sp, z in zs.items():
        zz = torch.from_numpy(z[:len(z) - a.gap])
        codes = torch.cdist(zz, centroids).argmin(1)
        summary[f'{sp}_code_hist'] = torch.bincount(codes, minlength=a.k).tolist()

    if a.check:
        from models.como import CoMoActions
        act = CoMoActions(lam, mean, std, centroids, mode='full').to(a.device)
        sp = [s for s in zs if s != a.fit_split][0] if len(zs) > 1 else a.fit_split
        with h5py.File(f'data/{sp}_frames.h5', 'r') as h5:
            rows = np.arange(min(a.check, len(zs[sp]) - a.gap))
            fr = h5['frames'][:rows[-1] + a.gap + 1]
        x = torch.from_numpy(np.stack([fr[rows], fr[rows + a.gap]], 1)).to(a.device).permute(0, 1, 4, 2, 3).float() / 127.5 - 1
        online = torch.cat([lam.encode(x[i:i + 64]) for i in range(0, len(x), 64)])[:, 0].float().cpu()  # [n, A] raw z
        stored = torch.from_numpy(zs[sp][rows])
        same = (act.codes(online.to(a.device)) == act.codes(stored.to(a.device))).float().mean().item()
        rel = ((online - stored).norm(dim=1) / stored.norm(dim=1)).mean().item()
        summary['check'] = {'split': sp, 'pairs': len(rows), 'code_agreement': same, 'rel_l2': rel}
        print(f'check on {len(rows)} {sp} pairs: k{a.k} code agreement {same:.4f}, mean relative L2 {rel:.4f}')

    for mode in ('full', f'k{a.k}'):
        d = f'{a.out_dir}_{mode}'
        os.makedirs(d, exist_ok=True)
        torch.save({'model_type': 'como_actions', 'como_path': a.como_ckpt, 'mode': mode, 'mean': mean, 'std': std,
                    'centroids': centroids, 'gap': a.gap, 'config': {'model_type': 'como_actions', 'n_actions': a.k}},
                   os.path.join(d, STATE))
        print(f'wrote {d}/{STATE}')
    json.dump(summary, open(f'{a.out_dir}_full/summary.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if 'hist' not in k}))
    for k, v in summary.items():
        if 'hist' in k:
            print(k, v)


if __name__ == '__main__':
    main()
