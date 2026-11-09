"""What does an action encode? Ridge read-outs from action vectors on the judged pairs, always tested on held-out blocks.

Targets: Link's measured position in frame t+4 (where to draw him), his on-screen move, his move in the world (on-screen move
minus the shift), the shift, and the judged direction (classification by the largest ridge output).

    python experiments/small_action/probe.py --ckpt <dir>/step_20000 --device cuda
    python experiments/small_action/probe.py --vectors eval_results/action_eval/zelda/como_20min/vectors.npy
"""

import argparse
import json

import numpy as np

from experiments.small_action.eval import SET, subset_rows

DIRECTIONS = ['U', 'D', 'L', 'R', 'STANDING']
MIN_CONFIDENCE = 0.5
RIDGE = 1.0


def fit_predict(train_x, train_y, test_x):
    # standardised inputs, ridge, intercept from the training mean
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    a, b = (train_x - mean) / std, (test_x - mean) / std
    y_mean = train_y.mean(0)
    w = np.linalg.solve(a.T @ a + RIDGE * np.eye(a.shape[1]), a.T @ (train_y - y_mean))
    return b @ w + y_mean


def cross_blocks(x, y, blocks):
    # predictions of y [N, K] from x [N, A], each block predicted by a model fit on the others
    out = np.zeros_like(y, dtype=np.float64)
    for b in np.unique(blocks):
        out[blocks == b] = fit_predict(x[blocks != b], y[blocks != b], x[blocks == b])
    return out


def r2(y, pred):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def probe(vectors, subset=None):
    # vectors: [N, A], one per pair of the subset, in pairs.json order
    rows = subset_rows(subset)[0]
    kept = json.load(open(f'{SET}/labels.json'))['kept']
    found = np.load(f'{SET}/link_positions.npz')
    yx, conf = found['yx'].astype(np.float64), found['conf']
    a, b = np.array([r['a'] for r in rows]), np.array([r['b'] for r in rows])
    blocks, shift = np.array([r['block'] for r in rows]), np.array([r['shift'] for r in rows], dtype=np.float64)
    labels = np.array([kept.get(str(r['id']), '') for r in rows], dtype=object)
    still = np.abs(shift).max(1) <= 1
    seen = (conf[a] > MIN_CONFIDENCE) & (conf[b] > MIN_CONFIDENCE)
    screen_move, world_move = yx[b] - yx[a], yx[b] - yx[a] - shift
    out = {'n_with_link': int(seen.sum()), 'shift_all': r2(shift, cross_blocks(vectors, shift, blocks))}
    for name, target in [('position_next', yx[b]), ('screen_move', screen_move), ('world_move', world_move)]:
        pred = cross_blocks(vectors[seen], target[seen], blocks[seen])
        out[name] = r2(target[seen], pred)
        for group, mask in [('still', still[seen]), ('moving', ~still[seen])]:
            out[f'{name}_{group}'] = r2(target[seen][mask], pred[mask])
    known = np.isin(labels, DIRECTIONS)
    one_hot = np.stack([labels[known] == d for d in DIRECTIONS], 1).astype(np.float64)
    guess = cross_blocks(vectors[known], one_hot, blocks[known]).argmax(1)
    truth = one_hot.argmax(1)
    walks = truth < 4  # the 4 walking directions, balanced
    for group, mask in [('still', still[known] & walks), ('moving', ~still[known] & walks)]:
        per = [float((guess[mask & (truth == k)] == k).mean()) for k in range(4) if (mask & (truth == k)).any()]
        out[f'direction_balanced_acc_{group}'] = float(np.mean(per))
        out[f'n_walks_{group}'] = int(mask.sum())
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt')
    p.add_argument('--vectors')
    p.add_argument('--device', default='cuda')
    p.add_argument('--subset', default='all', choices=['all', 'dev', 'confirm'])
    a = p.parse_args()
    headline = {}
    if a.vectors:
        vectors = np.load(a.vectors).astype(np.float64)[subset_rows(a.subset)[1]]
    else:
        from experiments.small_action.eval import action_scores, load_model
        model, _ = load_model(a.ckpt, a.device)
        scores = action_scores(model, a.device, subset=a.subset)
        vectors = scores['actions'].astype(np.float64)
        headline = {'headline': scores['link']['headline'], 'headline_still': scores['link']['still'].get('nmi_adj'), 'headline_moving': scores['link']['moving'].get('nmi_adj')}
    print('PROBE ' + json.dumps({'source': a.ckpt or a.vectors, **headline, **probe(vectors, a.subset)}), flush=True)


if __name__ == '__main__':
    main()
