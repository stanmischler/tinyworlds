"""The action eval on the full action of the architecture, measured shift s + a, on the dev half (--subset confirm: the confirm half, once per finalist). For information: the criteria score a alone.

The training evals and assess.py give k-means a alone. Here the same eval (k = 16, fit on the dev pairs, same groups and labels) runs on a,
on s alone and on s concatenated with a. Each block is centred and divided by one number so that the block's variances sum to 1, so s and a
weigh the same in the distance whatever their sizes, and a's dimensions keep their relative scale. Also, as a check of that choice, every
dimension scaled to unit variance, where a's 8 numbers outweigh s's 2. Seeds 0 (the eval's) to 4; k-means seed 0 is known to flatter a.

    python experiments/small_action/full_action.py --model name=<ckpt> --model ... --out results/full_action
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch

from evaluation.action_eval import camera_groups, codes_of, link_scores
from experiments.small_action.assess import COMO_VECTORS
from experiments.small_action.eval import GAP_ROW, SET, TEST_H5, TEST_SHIFTS, actions_and_shifts, load_model, subset_rows

SEEDS = range(5)
MIN_PAIRS = 5


def unit_block(x):
    # [N, D] -> centred, the variances of its D dimensions summing to 1
    x = x - x.mean(0)
    return x / np.sqrt(x.var(0).sum())


def unit_dims(x):
    # [N, D] -> every dimension centred, unit variance
    return (x - x.mean(0)) / x.std(0).clip(min=1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', action='append', default=[], help='name=<checkpoint dir>')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--subset', default='dev', choices=['dev', 'confirm'])
    a = p.parse_args()
    rows, keep = subset_rows(a.subset)
    judged = json.load(open(f'{SET}/labels.json'))
    labels = np.array([judged['kept'].get(str(r['id']), '') for r in rows], dtype=object)
    group = camera_groups(np.array([r['shift'] for r in rows], dtype=np.float64))
    blocks = np.array([r['block'] for r in rows])
    s = np.load(TEST_SHIFTS)['shifts'][:, GAP_ROW].astype(np.float64)[[r['a'] for r in rows]]  # the measured shift D receives, [N, 2]
    assert np.isfinite(s).all()
    with h5py.File(TEST_H5, 'r') as f:
        frames_a = np.stack([f['frames'][r['a']] for r in rows])
        frames_b = np.stack([f['frames'][r['b']] for r in rows])

    def scored(vectors):
        # vectors [N, D] -> eval per k-means seed and the median over seeds
        out = []
        for seed in SEEDS:
            link = link_scores(codes_of(torch.from_numpy(vectors).float(), seed=seed), labels, group, blocks, MIN_PAIRS)
            out.append({'still': link['still'].get('nmi_adj'), 'moving': link['moving'].get('nmi_adj'), 'headline': link['headline']})
        return {'seed_0': out[0], 'median': {k: round(float(np.median([o[k] for o in out])), 4) for k in out[0]}}

    results = {'subset': a.subset, 'n': len(rows), 's alone': scored(unit_block(s)), 'CoMo 128 numbers, shift inside': scored(np.load(COMO_VECTORS)[keep].astype(np.float64))}
    for spec in a.model:
        name, ckpt = spec.split('=', 1)
        model, _ = load_model(ckpt, a.device)
        actions = actions_and_shifts(model, frames_a, frames_b, s.astype(np.float32), a.device)[0].astype(np.float64)
        results[name] = {'a alone': scored(actions), 's + a, unit block': scored(np.concatenate([unit_block(s), unit_block(actions)], 1)),
                         's + a, unit dimension': scored(np.concatenate([unit_dims(s), unit_dims(actions)], 1))}
    os.makedirs(a.out, exist_ok=True)
    json.dump(results, open(f'{a.out}/scores.json', 'w'), indent=1)
    print('FULL_ACTION ' + json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
