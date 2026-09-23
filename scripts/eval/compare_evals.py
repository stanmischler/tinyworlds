"""Paired comparison of two (or more pairs of) eval_next_frame.py result files.

Both files must come from the same windows (same test .h5, context, stride). For each metric the per-window
difference A - B is summarised by:
  - mean difference
  - 95% cluster-bootstrap CI over the test blocks (the decision interval: adjacent windows overlap and share a
    scene, so a per-window bootstrap is about half as wide as it should be)
  - 95% naive per-window bootstrap CI (secondary)
  - number of blocks whose mean difference favours A
  - the same, restricted to windows whose target transition carries a LAM code other than the dominant code
    (most of the action information lives there; the pooled mean dilutes it)

Usage (from the repo root):
    python scripts/eval/compare_evals.py eval_results/lam_20000_lam.json eval_results/none_20000_none.json
    python scripts/eval/compare_evals.py A.json B.json C.json D.json      # several pairs
"""

import argparse
import json
import sys

import numpy as np

METRICS = ('token_acc', 'psnr', 'ssim')


def load(path):
    r = json.load(open(path))
    w = r['windows']
    key = list(zip(w['block'], w['source_start']))
    codes = np.array(w.get('lam_code', w['action']))  # older files only have the fed action (equal to lam_code in lam mode)
    return r['name'], key, np.array(w['block']), codes, {m: np.array(w[m], dtype=np.float64) for m in METRICS if m in w}


def bootstrap_ci(diff, blocks, n_boot, rng, cluster):
    if cluster:
        ids = np.unique(blocks)
        groups = [diff[blocks == b] for b in ids]
        means = np.array([np.concatenate([groups[i] for i in rng.integers(0, len(ids), len(ids))]).mean() for _ in range(n_boot)])
    else:
        means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return np.percentile(means, [2.5, 97.5])


def compare(a_path, b_path, n_boot, seed):
    name_a, key_a, blocks, codes, ma = load(a_path)
    name_b, key_b, _, _, mb = load(b_path)
    assert key_a == key_b, 'the two files were not evaluated on the same windows'
    dominant = np.bincount(codes[codes >= 0]).argmax()
    print(f'\n{name_a}  minus  {name_b}   ({len(key_a)} windows, {len(set(blocks))} blocks; '
          f'{(codes != dominant).sum()} windows with LAM code != {dominant})')
    print(f'{"metric":26s} {"mean diff":>10s}  {"block 95% CI":>20s}  {"window 95% CI":>20s}  blocks favouring A')
    for m in METRICS:
        if m not in ma or m not in mb:
            print(f'{m:26s} (not in both files)')
            continue
        for label, sel in (('all', np.ones(len(blocks), bool)), (f'code!={dominant}', codes != dominant)):
            rng = np.random.default_rng(seed)
            d, bl = ma[m][sel] - mb[m][sel], blocks[sel]
            if len(d) == 0:
                continue
            lo, hi = bootstrap_ci(d, bl, n_boot, rng, cluster=True)
            nlo, nhi = bootstrap_ci(d, bl, n_boot, rng, cluster=False)
            fav = sum(d[bl == b].mean() > 0 for b in np.unique(bl))
            star = ' *' if (lo > 0 or hi < 0) else ''
            print(f'{m + " [" + label + "]":26s} {d.mean():+10.4f}  [{lo:+8.4f}, {hi:+8.4f}]  [{nlo:+8.4f}, {nhi:+8.4f}]  {fav}/{len(np.unique(bl))}{star}')
    print('  * = block CI excludes zero')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='+', help='pairs of result JSONs: A1 B1 [A2 B2 ...]; each pair reports A - B')
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if len(args.files) % 2:
        sys.exit('give an even number of files (pairs)')
    for a, b in zip(args.files[::2], args.files[1::2]):
        compare(a, b, args.n_boot, args.seed)


if __name__ == '__main__':
    main()
