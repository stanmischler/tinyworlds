"""Split the judged eval pairs into dev (tune here) and confirm (score once per finalist), by contiguous stretches of video.

Each test block is cut in two, giving 20 stretches. The partition into 10 + 10 is chosen from counts of pairs per group only (no
model score), to balance camera-still walks, camera-moving walks and standing, then to spread still walks over many stretches.
Pairs within GUARD frames of a dev/confirm boundary belong to neither, so no frame is shared. Dev is the side holding stretch 0.

    python experiments/small_action/split.py
"""

import itertools
import json

import numpy as np

from experiments.small_action.assess import judged_groups
from experiments.small_action.eval import SET, SPLIT

GUARD = 16  # frames; a pair and its context reach 12 frames back and 4 forward


def stretches(rows):
    # -> [(first, last)] frame of `a`, two per block, in video order
    out = []
    for block in sorted({r['block'] for r in rows}):
        a = sorted(r['a'] for r in rows if r['block'] == block)
        out += [(a[0], a[len(a) // 2 - 1]), (a[len(a) // 2], a[-1])]
    return out


def choose(counts):
    # counts: {group: pairs per stretch [S]} -> the indices of the dev side
    n, best, best_key = len(next(iter(counts.values()))), None, None
    still = counts['still_walking'] > 0
    for side in itertools.combinations(range(1, n), n // 2 - 1):
        dev = np.zeros(n, bool)
        dev[[0, *side]] = True
        imbalance = max(abs(c[dev].sum() / c.sum() - 0.5) for c in counts.values())
        spread = min(still[dev].sum(), still[~dev].sum())
        boundaries = int((dev[1:] != dev[:-1]).sum())
        key = (round(imbalance / 0.02), -spread, boundaries)  # imbalance within 2 points counts as equal
        if best_key is None or key < best_key:
            best, best_key = dev, key
    return best


def main():
    rows = json.load(open(f'{SET}/pairs.json'))
    parts = stretches(rows)
    groups = judged_groups()[0]
    per = lambda g: np.array([sum(lo <= r['a'] <= hi for r in groups[g]) for lo, hi in parts])  # noqa: E731
    counts = {g: per(g) for g in groups}
    dev = choose(counts)
    out = {'dev': [], 'confirm': []}
    for k, (lo, hi) in enumerate(parts):
        before = k > 0 and dev[k - 1] != dev[k]
        after = k + 1 < len(parts) and dev[k + 1] != dev[k]
        out['dev' if dev[k] else 'confirm'].append([int(lo + GUARD * before), int(hi - GUARD * after)])
    json.dump(out, open(SPLIT, 'w'), indent=1)
    for side, ranges in out.items():
        inside = lambda r: any(lo <= r['a'] <= hi for lo, hi in ranges)  # noqa: E731
        print(side, ranges, {g: sum(map(inside, v)) for g, v in groups.items()}, 'pairs', sum(map(inside, rows)))


if __name__ == '__main__':
    main()
