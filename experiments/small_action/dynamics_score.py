"""Score dynamics evals (scripts/eval/eval_next_frame.py jsons) on one half of the held-out windows, and pick the sweep's winner.

The split is the judged pairs' dev/confirm split (eval_results/action_eval/zelda/set/split.json, experiments/small_action/split.py):
contiguous stretches of the 10 test blocks, chosen from group counts only, 16-frame guards at each boundary. A window belongs to a half
when its frame t (start + 8) lies in it: dev 432 of the 890 windows, confirm 430, 28 in guards. The same half the action model was chosen on;
that action model (late_weight_focus 200k) was already scored on confirm (2026-11-07), so confirm is unseen by the dynamics model only.
Still-walk windows: frame t is the first frame of a judged pair labelled walking with the camera still and Link detected in both frames
(experiments/small_action/assess.py judged_groups); their judge-box LPIPS is the selection metric.

Selection rule (fixed before any sweep result): lowest dev still-walk judge-box LPIPS at 32 Euler steps; arms within .005 of the lowest
are tied, and among them the highest dev PSNR wins. Printed beside it (not part of the rule): each arm's 95% interval of that LPIPS and
of arm minus winner on the same windows, resampling BOOT_CHUNK-frame chunks of the held-out video as assess.py does.

    python experiments/small_action/dynamics_score.py --split dev --eval name=<eval json> --eval ...
    python experiments/small_action/dynamics_score.py --counts
"""

import argparse
import json

import h5py
import numpy as np

from experiments.small_action.assess import BOOT_CHUNK, bootstrap_95, judged_groups
from experiments.small_action.eval import SPLIT, TEST_H5
from evaluation.windows import split_windows, test_windows

CONTEXT, SKIP, STRIDE = 3, 4, 8  # eval_next_frame defaults: frame t = start + 8, target = start + 12
TIE = 0.005  # LPIPS
GROUP = 'still_walking'  # the selection group


def windows_of(split):
    return split_windows(test_windows(TEST_H5, CONTEXT, SKIP, STRIDE), SPLIT, split, CONTEXT, SKIP)


def frame_t(windows):
    return np.array([s + (CONTEXT - 1) * SKIP for _, s in windows])


def group_masks(windows, split):
    # -> {group: bool [N]}: windows whose frame t is the first frame of a judged pair of that group
    groups, _ = judged_groups(split)
    t = frame_t(windows)
    return {g: np.isin(t, [r['a'] for r in rows]) for g, rows in groups.items()}


def score(path, split):
    # one eval json, run with --split-file SPLIT --split <split> (or on all windows): the numbers of that half
    result = json.load(open(path))
    assert result['config']['num_steps'] == 32, f"{path}: {result['config']['num_steps']} Euler steps, the rule is scored at 32"
    windows = windows_of(split)[:result['config'].get('limit') or None]  # --limit: the first N windows of the half (smoke)
    w = result['windows']
    with h5py.File(TEST_H5, 'r') as f:
        source = f['source_index'][:]
    if result['n_windows'] != len(windows):  # an eval on all 890 windows: keep this half's rows
        every = test_windows(TEST_H5, CONTEXT, SKIP, STRIDE)
        assert result['n_windows'] == len(every), (result['n_windows'], len(every))
        keep = np.isin([s for _, s in every], [s for _, s in windows])
        w = {k: np.asarray(v, dtype=object)[keep].tolist() for k, v in w.items()}
    assert w['source_start'] == [int(source[s]) for _, s in windows], 'eval windows differ from this split'
    masks = group_masks(windows, split)
    lpips = np.array(w['judge_box_lpips'], dtype=np.float64)  # NaN where no judge box
    copy_lpips = np.array(w['copy_judge_box_lpips'], dtype=np.float64)
    out = {'eval': path, 'split': split, 'windows': len(windows), 'psnr': float(np.mean(np.array(w['psnr'], dtype=np.float64))),
           'copy_psnr': float(np.mean(np.array(w['copy_psnr'], dtype=np.float64))), 'judge_box_lpips': float(np.nanmean(lpips)),
           'judge_box_fid': result['summary']['model'].get('judge_box_fid') if result['n_windows'] == len(windows) else None}
    for g, m in masks.items():
        boxed = m & ~np.isnan(lpips)
        out[f'{g}_n'] = int(boxed.sum())
        out[f'{g}_judge_box_lpips'] = float(lpips[boxed].mean()) if boxed.any() else None
        out[f'{g}_copy_judge_box_lpips'] = float(copy_lpips[boxed].mean()) if boxed.any() else None
    still = masks[GROUP] & ~np.isnan(lpips)  # per-window values, for the intervals
    out[f'{GROUP}_t'], out[f'{GROUP}_values'] = frame_t(windows)[still].tolist(), lpips[still].tolist()
    return out


def pick(scores, split='dev'):
    # scores: {name: score dict} -> the winner by the rule above
    if split != 'dev':
        raise ValueError(f'the sweep is picked on dev only, not {split}: confirm scores the winner once')
    lpips = {n: s[f'{GROUP}_judge_box_lpips'] if s[f'{GROUP}_judge_box_lpips'] is not None else float('inf') for n, s in scores.items()}
    best = min(lpips.values())
    tied = [n for n in scores if lpips[n] <= best + TIE]
    return max(tied, key=lambda n: scores[n]['psnr']), tied


def intervals(scores, winner):
    # -> {name: {lpips_95, minus_winner_95}}: still-walk judge-box LPIPS, and arm minus winner paired on the same windows
    ref = dict(zip(scores[winner][f'{GROUP}_t'], scores[winner][f'{GROUP}_values']))
    out = {}
    for n, s in scores.items():
        t, v = np.array(s[f'{GROUP}_t']), np.array(s[f'{GROUP}_values'])
        if not len(t):
            continue
        assert set(t.tolist()) == set(ref), f'{n}: still-walk windows differ from {winner}'
        chunks = t // BOOT_CHUNK
        out[n] = {'lpips_95': bootstrap_95(v, chunks), 'minus_winner_95': bootstrap_95(v - np.array([ref[x] for x in t]), chunks)}
        print(f"INTERVAL {n}: still-walk LPIPS {v.mean():.4f} {out[n]['lpips_95']}, minus {winner} {out[n]['minus_winner_95']}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval', action='append', default=[], help='name=<eval_next_frame json>')
    p.add_argument('--split', default='dev', choices=['dev', 'confirm'])
    p.add_argument('--counts', action='store_true', help='windows per half and group, no eval needed')
    p.add_argument('--out')
    a = p.parse_args()
    if a.counts:
        for split in ('dev', 'confirm'):
            windows = windows_of(split)
            print(split, len(windows), {g: int(m.sum()) for g, m in group_masks(windows, split).items()})
        return
    scores = {}
    for spec in a.eval:
        name, path = spec.split('=', 1)
        scores[name] = score(path, a.split)
        print('SCORE', name, json.dumps(scores[name]), flush=True)
    if len(scores) > 1 and a.split == 'dev':
        winner, tied = pick(scores, a.split)
        print(f'WINNER {winner} (tied within {TIE}: {tied})', flush=True)
        scores['_winner'] = {'name': winner, 'tied': tied, 'intervals': intervals(scores, winner)}
    if a.out:
        json.dump(scores, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
