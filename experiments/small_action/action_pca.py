"""The PCA picture of an action space on the dev pairs (never the confirm half): `still_walk_picture`, drawn at each checkpoint of training, and
a dump of the learned action a of each model for pictures made elsewhere.

Per pair: id, a, the measured shift s that D receives, camera group (still / moving), judged Link label, video block. One .npz per
model, plus CoMo's 128-number action on the same pairs (the vectors full_action.py and assess.py read). Labels are stored for colouring
only; nothing here fits on them.

    python experiments/small_action/action_pca.py --model name=<ckpt> --model ... --out results/action_pca
"""

import argparse
import json
import os

import h5py
import numpy as np

from evaluation.action_eval import camera_groups
from experiments.small_action.assess import COMO_VECTORS
from experiments.small_action.eval import GAP_ROW, SET, TEST_H5, TEST_SHIFTS, actions_and_shifts, load_model, subset_rows

SUBSET = 'dev'
WALK_COLOURS = {'U': 'tab:blue', 'D': 'tab:orange', 'L': 'tab:green', 'R': 'tab:red'}
N_COMPONENTS = 3


def still_walk_picture(actions, subset, path, title=''):
    # actions [N, A] of the subset's pairs, in pairs.json order -> the picture at `path` (principal components fit on every pair; the still-camera
    # walks coloured by the judged direction over the other pairs in grey) and one number: how often a still walk's direction is the one of the
    # nearest direction centre in the first N_COMPONENTS components, centres from the other video blocks, averaged over the 4 directions
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = subset_rows(subset)[0]
    kept = json.load(open(f'{SET}/labels.json'))['kept']
    labels = np.array([kept.get(str(r['id']), '') for r in rows])
    blocks = np.array([r['block'] for r in rows])
    still = camera_groups(np.array([r['shift'] for r in rows], dtype=np.float64)) == 'still'
    centred = actions.astype(np.float64) - actions.mean(0)
    _, singular, axes = np.linalg.svd(centred, full_matrices=False)
    points = centred @ axes[:N_COMPONENTS].T  # [N, 3]
    explained = singular[:N_COMPONENTS] ** 2 / (singular ** 2).sum()
    walk = still & np.isin(labels, list(WALK_COLOURS))
    one_hot = np.stack([labels[walk] == d for d in WALK_COLOURS], 1).astype(np.float64)  # [M, 4]
    distance = np.zeros_like(one_hot)  # [M, 4]: to each direction's centre, the centre taken from the other blocks
    for b in np.unique(blocks[walk]):
        inside, outside = blocks[walk] == b, blocks[walk] != b
        for k in range(len(WALK_COLOURS)):
            centre = points[walk][outside & (one_hot[:, k] == 1)].mean(0) if (outside & (one_hot[:, k] == 1)).any() else np.full(N_COMPONENTS, np.inf)
            distance[inside, k] = np.linalg.norm(points[walk][inside] - centre, axis=1)
    guess, truth = distance.argmin(1), one_hot.argmax(1)
    accuracy = float(np.mean([(guess[truth == k] == k).mean() for k in range(len(WALK_COLOURS)) if (truth == k).any()]))
    fig, panels = plt.subplots(1, 2, figsize=(11, 5))
    for panel, (i, j) in zip(panels, [(0, 1), (1, 2)]):
        panel.scatter(points[~walk, i], points[~walk, j], s=4, c='lightgrey', label='other pairs')
        for d, colour in WALK_COLOURS.items():
            m = walk & (labels == d)
            panel.scatter(points[m, i], points[m, j], s=28, c=colour, edgecolors='k', linewidths=.4, label=f'still walk {d} ({m.sum()})')
        panel.set_xlabel(f'component {i + 1} ({explained[i]:.0%})')
        panel.set_ylabel(f'component {j + 1} ({explained[j]:.0%})')
    panels[0].legend(fontsize=8)
    fig.suptitle(f'{title}  still-walk direction from the nearest centre: {accuracy:.2f} (chance .25)')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', action='append', default=[], help='name=<checkpoint dir>')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    rows, keep = subset_rows(SUBSET)
    labels = json.load(open(f'{SET}/labels.json'))['kept']
    s = np.load(TEST_SHIFTS)['shifts'][:, GAP_ROW].astype(np.float64)[[r['a'] for r in rows]]  # [N, 2], as in full_action.py
    common = {'id': np.array([r['id'] for r in rows]), 'block': np.array([r['block'] for r in rows]), 's': s,
              'pair_shift': np.array([r['shift'] for r in rows], dtype=np.float64),
              'camera': camera_groups(np.array([r['shift'] for r in rows], dtype=np.float64)),
              'label': np.array([labels.get(str(r['id']), '') for r in rows])}
    os.makedirs(a.out, exist_ok=True)
    np.savez(f'{a.out}/como.npz', a=np.load(COMO_VECTORS)[keep].astype(np.float32), **common)
    with h5py.File(TEST_H5, 'r') as f:
        frames_a = np.stack([f['frames'][r['a']] for r in rows])
        frames_b = np.stack([f['frames'][r['b']] for r in rows])
    for spec in a.model:
        name, ckpt = spec.split('=', 1)
        model, _ = load_model(ckpt, a.device)
        actions = actions_and_shifts(model, frames_a, frames_b, s.astype(np.float32), a.device)[0]
        np.savez(f'{a.out}/{name}.npz', a=actions.astype(np.float32), ckpt=ckpt, **common)
        print(f'ACTION_PCA {name} {actions.shape} std {actions.std(0).round(3).tolist()}', flush=True)


if __name__ == '__main__':
    main()
