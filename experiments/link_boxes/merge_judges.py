"""Merge the blind judges' Link annotations (make_sheets.py sheets) into evaluation/zelda_link_boxes.json.

Judges A and B annotate every window, judge C only their disagreements. Per window:
  - A and B both see Link within 8 px (max of |dx|, |dy|) -> their mean
  - A and B both say absent -> null
  - otherwise (one absent, or > 8 px apart) -> judge C: absent -> null; present -> mean of C and the closer of A / B if
    within 8 px, else C alone
Prints the agreement statistics and the count of windows per rule.

    python experiments/link_boxes/merge_judges.py --dir eval_results/link_boxes --out evaluation/zelda_link_boxes.json
"""

import argparse
import glob
import json

import numpy as np

from evaluation.windows import test_windows

H5 = 'data/zelda_test_frames.h5'
AGREE_PX = 8


def load(pattern):
    rows = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows[r['window']] = r
    return rows


def seen(r):
    return r is not None and not r['absent'] and r['x'] is not None


def dist(a, b):
    return max(abs(a['x'] - b['x']), abs(a['y'] - b['y']))


def mean_xy(*rs):
    return float(np.mean([r['x'] for r in rs])), float(np.mean([r['y'] for r in rs]))


def merge(a, b, c):
    # -> (x, y) or None, and the rule used
    if seen(a) and seen(b) and dist(a, b) <= AGREE_PX:
        return mean_xy(a, b), 'A+B'
    if not seen(a) and not seen(b):
        return None, 'A+B absent'
    assert c is not None, f"window {a['window']}: A and B disagree but judge C has no answer"
    if not seen(c):
        return None, 'C absent'
    close = [r for r in (a, b) if seen(r) and dist(r, c) <= AGREE_PX]
    if close:
        return mean_xy(c, min(close, key=lambda r: dist(r, c))), 'C+closest'
    return (float(c['x']), float(c['y'])), 'C'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='eval_results/link_boxes')
    p.add_argument('--out', default='evaluation/zelda_link_boxes.json')
    args = p.parse_args()
    A, B, C = load(f'{args.dir}/judge_A_part*.jsonl'), load(f'{args.dir}/judge_B_part*.jsonl'), load(f'{args.dir}/judge_C.jsonl')
    wins = test_windows(H5, 3, 4, 8)
    assert set(A) == set(B) == set(range(len(wins))), 'judges A and B must cover every window'

    both = [dist(A[i], B[i]) for i in A if seen(A[i]) and seen(B[i])]
    print(f'A and B both see Link in {len(both)} windows: |d| median {np.median(both):.0f} px, 90% {np.percentile(both, 90):.0f}, '
          f'99% {np.percentile(both, 99):.0f}, > {AGREE_PX} px {sum(d > AGREE_PX for d in both)}')
    out, rules = [], {}
    for i, (block, start) in enumerate(wins):
        xy, rule = merge(A[i], B[i], C.get(i))
        rules[rule] = rules.get(rule, 0) + 1
        out.append({'block': block, 'start': start, 'x': None if xy is None else round(xy[0], 1),
                    'y': None if xy is None else round(xy[1], 1), 'dialog': bool(A[i]['dialog'] or B[i]['dialog'])})
    print('merge rules:', rules, f"-> Link located in {sum(r['x'] is not None for r in out)} / {len(out)} windows")
    with open(args.out, 'w') as f:
        json.dump({'about': 'Link centre (x = column, y = row, 128x128 px) on the target frame of each held-out Zelda eval '
                            'window (eval_next_frame.py: context 3, frame skip 4, stride 8), placed by two blind judges '
                            '(third judge on disagreements); null = Link not visible. Built by experiments/link_boxes/.',
                   'windows': out}, f, indent=0)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
