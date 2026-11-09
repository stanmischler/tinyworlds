"""Score the probe tasks of modal_link_probe.py: out-of-fold predictions -> R2, balanced direction accuracy, bootstrap intervals
(by window and by test block), paired differences between inputs, a results JSON and one figure.

    ./.venv/bin/python experiments/link_probe/report.py --tasks <dir>/tasks --pairs <dir>/pairs.npz --out <dir>
"""

import argparse
import glob
import json
import os

import numpy as np

from experiments.link_probe.labels import MOVED_PX

N_BOOT = 2000
CLEAR_PX = 8        # a stricter "moved", far above the judges' jitter of about 1.5 px
DIRS = 4


def r2(y, p):
    # pooled over both axes, against the mean of the scored windows
    sst = ((y - y.mean(0)) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / sst) if sst > 0 else float('nan')


def balanced(d, dp):
    per = [(dp[d == k] == k).mean() for k in range(DIRS) if (d == k).any()]
    return float(np.mean(per)) if per else float('nan')


def score(y, p, d, dp):
    return {'r2': r2(y, p), 'bal_acc': balanced(d, dp), 'acc': float((d == dp).mean()) if len(d) else float('nan')}


def resamples(block, rng):
    # index sets: over windows, and over test blocks (whole blocks kept together, as windows inside a clip are correlated)
    n = len(block)
    windows = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    groups = {b: np.flatnonzero(block == b) for b in np.unique(block)}
    keys = list(groups)
    blocks = [np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(N_BOOT)]
    return windows, blocks


def ci(vals):
    v = np.array(vals, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def summarise(runs, mask_fn, block, key, seed=0):
    """runs: list of (y, pred, dir, dir_pred) over the same windows; mask_fn -> scored windows. Mean + spread over runs, bootstrap
    on the mean over runs, so the interval covers the windows, not only the seeds."""
    m = mask_fn
    out = {'n': int(m.sum())}
    if m.sum() < 5:
        return out
    per = [score(y[m], p[m], d[m], dp[m])[key] for y, p, d, dp in runs]
    out.update(mean=float(np.mean(per)), spread=float(np.std(per)), runs=len(per))
    rng = np.random.default_rng(seed)
    b = block[m]
    wins, blks = resamples(b, rng)
    arrs = [(y[m], p[m], d[m], dp[m]) for y, p, d, dp in runs]
    f = lambda i: np.mean([score(y[i], p[i], d[i], dp[i])[key] for y, p, d, dp in arrs])
    out['ci_window'], out['ci_block'] = ci([f(i) for i in wins]), ci([f(i) for i in blks])
    return out


def paired(a_runs, b_runs, m, block, key, seed=0):
    # difference of means, same windows and same resamples on both sides
    rng = np.random.default_rng(seed)
    wins, blks = resamples(block[m], rng)
    A = [tuple(x[m] for x in r) for r in a_runs]
    B = [tuple(x[m] for x in r) for r in b_runs]
    g = lambda R, i: np.mean([score(y[i], p[i], d[i], dp[i])[key] for y, p, d, dp in R])
    full = np.arange(int(m.sum()))
    return {'difference': float(g(A, full) - g(B, full)), 'ci_window': ci([g(A, i) - g(B, i) for i in wins]),
            'ci_block': ci([g(A, i) - g(B, i) for i in blks])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--figure-only', action='store_true')
    a = ap.parse_args()
    if a.figure_only:
        figure(json.load(open(f'{a.out}/results.json')), a.out)
        return
    d = np.load(a.pairs)
    block, still, scroll = d['block'], d['still'], d['scroll']
    res = {'n': {'pairs': int(len(block)), 'consecutive_pairs_before_filter': int(d['n_consecutive_pairs']), 'camera_still': int(still.sum()),
                 'scroll': int(scroll.sum()), 'neither': int((~still & ~scroll).sum()),
                 'camera_still_link_moved': int((still & (np.abs(d['disp']).max(1) >= MOVED_PX)).sum()),
                 'camera_still_link_moved_clearly': int((still & (np.abs(d['disp']).max(1) >= CLEAR_PX)).sum()),
                 'scroll_link_moved': int((scroll & (np.abs(d['disp']).max(1) >= MOVED_PX)).sum()),
                 'blocks_with_still_moved': np.bincount(block[still & (np.abs(d['disp']).max(1) >= MOVED_PX)], minlength=10).tolist()},
           'scores': {}}
    tasks = {}
    for path in sorted(glob.glob(f'{a.tasks}/*.npz')):
        t = np.load(path)
        tasks[(str(t['input']), str(t['set']), str(t['method']), bool(t['shuffled']))] = t
    res['seconds'] = {'gpu_net_seconds': float(sum(float(t['seconds']) for k, t in tasks.items() if k[2] == 'net')),
                      'cpu_seconds': float(sum(float(t['seconds']) for k, t in tasks.items() if k[2] != 'net')), 'tasks': len(tasks)}
    full_runs = {}
    for key, t in tasks.items():
        inp, st, method, sh = key
        runs_all = [(t['idx'][i], t['y'][i], t['pred'][i], t['dir'][i], t['dir_pred'][i], t['ok'][i]) for i in range(len(t['idx']))]
        row = {'train_r2': float(np.nanmean(t['train_r2']))}
        if st == 'camera, scroll, same count':  # a new draw of windows per run: spread and window interval over the draws
            rs, ba = [], []
            for idx, y, p, dd, dp, ok in runs_all:
                rs.append(r2(y, p)), ba.append(balanced(dd, dp))
            row.update(n=int(len(runs_all[0][0])), r2=float(np.mean(rs)), r2_spread=float(np.std(rs)), bal_acc=float(np.mean(ba)), bal_acc_spread=float(np.std(ba)), draws=len(rs))
        else:
            idx = runs_all[0][0]
            b = block[idx]
            groups = {'link, camera still': {'still': np.ones(len(idx), bool)}, 'link, scroll': {'scroll': np.ones(len(idx), bool)},
                      'link, all windows': {'still': still[idx], 'scroll': scroll[idx]}, 'camera, scroll': {'scroll': np.ones(len(idx), bool)}}[st]
            for g, gm in groups.items():
                ys = np.stack([r[1] for r in runs_all])
                moved = runs_all[0][5] if not sh else np.abs(d['disp'][idx]).max(1) >= MOVED_PX
                rr = [(r[1], r[2], r[3], r[4]) for r in runs_all]
                if st.startswith('camera'):
                    row[g] = {'r2': summarise(rr, gm, b, 'r2'), 'bal_acc': summarise(rr, gm, b, 'bal_acc')}
                else:
                    # shuffled: each run scored on its own permuted "moved" windows, with the permuted direction
                    rows = {}
                    rows['r2_all'] = summarise(rr, gm, b, 'r2')
                    if sh:
                        per_r2, per_ba = [], []
                        for r in runs_all:
                            mk = gm & r[5]
                            per_r2.append(r2(r[1][mk], r[2][mk])), per_ba.append(balanced(r[3][mk], r[4][mk]))
                        rows['r2_moved'] = {'mean': float(np.mean(per_r2)), 'spread': float(np.std(per_r2)), 'runs': len(per_r2), 'n': int((gm & runs_all[0][5]).sum())}
                        rows['bal_acc'] = {'mean': float(np.mean(per_ba)), 'spread': float(np.std(per_ba)), 'runs': len(per_ba), 'n': int((gm & runs_all[0][5]).sum())}
                    else:
                        rows['r2_moved'] = summarise(rr, gm & moved, b, 'r2')
                        rows['bal_acc'] = summarise(rr, gm & moved, b, 'bal_acc')
                        clear = np.abs(runs_all[0][1]).max(1) >= CLEAR_PX
                        rows['r2_clear'] = summarise(rr, gm & clear, b, 'r2')
                        rows['bal_acc_clear'] = summarise(rr, gm & clear, b, 'bal_acc')
                        dirs = runs_all[0][3][gm & moved]
                        rows['majority'] = float(np.bincount(dirs, minlength=DIRS).max() / max(len(dirs), 1))
                        rows['acc'] = summarise(rr, gm & moved, b, 'acc')
                        rows['dir_from_displacement'] = float(np.mean([balanced(r[3][gm & moved], _dir(r[2][gm & moved])) for r in runs_all]))
                    row[g] = rows
                if not sh:
                    full_runs[(inp, st, method, g)] = (rr, gm & (np.abs(runs_all[0][1]).max(1) >= MOVED_PX if st.startswith('link') else gm), b, idx)
        res['scores'][f"{inp} / {st} / {method} / {'shuffled' if sh else 'real'}"] = row
    pix = track(d['frames'], d['link_t'])
    res['tracker_reference'] = {}
    for g, gm in (('still', still), ('scroll', scroll)):
        m = gm & (np.abs(d['disp']).max(1) >= MOVED_PX) & ~np.isnan(pix).any(1)
        res['tracker_reference'][g] = {'n': int(m.sum()), 'r2': r2(d['disp'][m], pix[m]), 'bal_acc': balanced(_dir(d['disp'][m]), _dir(pix[m])),
                                       'median_err_px': float(np.median(np.abs(pix[m] - d['disp'][m]).max(1)))}
    # paired differences on the headline windows
    res['paired'] = {}
    z = {'z 20 min': ('z 20 min | two steps', 'ridge'), 'z 5 h': ('z 5 h | two steps', 'ridge'), 'z 20 min MLP': ('z 20 min | two steps', 'mlp'), 'z 5 h MLP': ('z 5 h | two steps', 'mlp')}
    mae = {'MAE patches': ('MAE patches', 'net'), 'MAE around Link': ('MAE patches around Link', 'net'), 'raw pixels around Link': ('raw pixels around Link', 'net')}
    for st, g in (('link, camera still', 'still'), ('link, scroll', 'scroll')):
        for mn, mk in mae.items():
            for zn, zk in z.items():
                x, y = (mk[0], st, mk[1], g), (zk[0], st, zk[1], g)
                if x in full_runs and y in full_runs:
                    ra, m, b, _ = full_runs[x]
                    rb = full_runs[y][0]
                    res['paired'][f'{mn} minus {zn}, {g}'] = {k: paired(ra, rb, m, b, k) for k in ('r2', 'bal_acc')}
    x = ('z 5 h | two steps', 'link, camera still', 'ridge', 'still'), ('z 20 min | two steps', 'link, camera still', 'ridge', 'still')
    if all(k in full_runs for k in x):
        ra, m, b, _ = full_runs[x[0]]
        res['paired']['z 5 h minus z 20 min, still'] = {k: paired(ra, full_runs[x[1]][0], m, b, k) for k in ('r2', 'bal_acc')}
    with open(f'{a.out}/results.json', 'w') as f:
        json.dump(res, f, indent=1)
    figure(res, a.out)
    print(f"{a.out}/results.json: {len(tasks)} tasks; n {res['n']}")


def track(fr, link_t, half=10, reach=24):
    # untrained reference: Link's box at t found again in t+8 by least squared error within +-reach px -> (dy, dx) [N, 2], NaN at edges
    out = np.full((len(fr), 2), np.nan)
    g = fr.astype(np.float32).mean(-1)  # [N, 3, H, W]
    for i, (y, x) in enumerate(np.round(link_t).astype(int)):
        if y < half or x < half or y + half > 128 or x + half > 128:
            continue
        tpl = g[i, 0, y - half:y + half, x - half:x + half]
        best = (np.inf, 0, 0)
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                yy, xx = y + dy, x + dx
                if yy < half or xx < half or yy + half > 128 or xx + half > 128:
                    continue
                e = ((g[i, 2, yy - half:yy + half, xx - half:xx + half] - tpl) ** 2).mean()
                if e < best[0]:
                    best = (e, dy, dx)
        out[i] = best[1:]
    return out


def _dir(p):
    from experiments.link_probe.labels import direction
    return direction(p)


def figure(res, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [('raw pixels around Link', 'net', 'Raw pixels, around Link'), ('raw pixels', 'net', 'Raw pixels, full frame'), ('MAE patches around Link', 'net', 'MAE patches, around Link'),
            ('MAE patches', 'net', 'MAE patches, full frame'), ('MAE averaged', 'mlp', 'MAE averaged'), ('IDM output 20 min | two steps', 'mlp', 'IDM output, 20 min'),
            ('IDM output 5 h | two steps', 'mlp', 'IDM output, 5 h'), ('z 20 min | two steps', 'mlp', 'z, 20 min'), ('z 5 h | two steps', 'mlp', 'z, 5 h')]
    series = [('Link, camera still', 'link, camera still', 'still', 'bal_acc', '#2a78d6', 'o'), ('Link, scroll', 'link, scroll', 'scroll', 'bal_acc', '#e0702e', 's'),
              ('Camera shift, scroll', 'camera, scroll', 'scroll', 'bal_acc', '#52514e', 'D')]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    Y = np.arange(len(rows))[::-1]
    for j, (label, st, g, key, c, mk) in enumerate(series):
        xs, lo, hi, ys = [], [], [], []
        for (inp, method, _), y in zip(rows, Y):
            s = res['scores'].get(f'{inp} / {st} / {method} / real', {}).get(g, {}).get(key)
            if s and 'mean' in s:
                xs.append(100 * s['mean']), lo.append(100 * s['ci_block'][0]), hi.append(100 * s['ci_block'][1]), ys.append(y + 0.22 - 0.22 * j)
        xs, lo, hi = map(np.array, (xs, lo, hi))
        ax.errorbar(xs, ys, xerr=[np.maximum(xs - lo, 0), np.maximum(hi - xs, 0)], fmt=mk, color=c, ms=5, lw=1.2, capsize=0, label=label)
    ax.axvline(25, color='#9a9994', lw=1, ls='--')
    ax.text(25.6, len(rows) - 0.45, 'chance 25%', color='#52514e', fontsize=8, va='top')
    ax.set_yticks(Y, [r[2] for r in rows], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Direction, balanced accuracy on held-out blocks (%); bars: 95% block bootstrap', fontsize=9, color='#52514e')
    ax.tick_params(axis='x', labelsize=8, colors='#52514e', length=0)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#e8e7e3', lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.suptitle('Which stage still shows where Link went', fontsize=11, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(f'{out}/link_probe.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
