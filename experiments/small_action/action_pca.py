"""Dump the learned action a of each model on the dev pairs (never the confirm half), for a PCA picture of the action space.

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
