"""STA-35 itc_loop i6 step 1: contact sheet of train pairs where the i5 ITC and pixel teachers disagree.

Each row: frame t and t+gap (red box = ITC crop, cyan box = pixel crop), then the two 24x24 crops of t / t+gap enlarged 3x,
plus both codes and crop changes. Used to count by eye how often each teacher's crop misses Link.

  python scripts/eval/itc_i6_diag.py --labels eval_results/itc_loop/i6/pseudo_i5.npz --n 40 --out-dir eval_results/itc_loop/i6
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from itc_actions import DIRS  # noqa: E402
from itc_facing import CROP, crop  # noqa: E402

NAMES = DIRS + ['STILL']


def box(d, cy, cx, s, color, H):
    if np.isnan(cy):
        return
    y0 = int(np.clip(round(cy) - CROP // 2, 0, H - CROP)); x0 = int(np.clip(round(cx) - CROP // 2, 0, H - CROP))
    d.rectangle([x0 * s, y0 * s, (x0 + CROP) * s - 1, (y0 + CROP) * s - 1], outline=color, width=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', default='eval_results/itc_loop/i6/pseudo_i5.npz')
    p.add_argument('--train-frames', default='data/zelda_train_frames.h5')
    p.add_argument('--n', type=int, default=40)
    p.add_argument('--per-sheet', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default='eval_results/itc_loop/i6')
    a = p.parse_args()
    z = np.load(a.labels)
    gap = int(z['gap'])
    ci, cp = z['code_itc'], z['code_pix']
    rows = np.nonzero((ci >= 0) & (ci != cp))[0]
    pick = np.sort(np.random.default_rng(a.seed).choice(rows, a.n, replace=False))
    X = h5py.File(a.train_frames, 'r')['frames']
    H = X.shape[1]
    s, cs = 2, 3  # frame scale, crop scale
    rh = H * s + 4
    meta = []
    for sh in range(0, a.n, a.per_sheet):
        sub = pick[sh:sh + a.per_sheet]
        W = 2 * H * s + 4 * CROP * cs + 6 * 6 + 260
        im = Image.new('RGB', (W, rh * len(sub)), 'white')
        d = ImageDraw.Draw(im)
        for r, t in enumerate(sub):
            fa, fb = X[t], X[t + gap]
            y = r * rh
            x = 0
            for f in (fa, fb):
                fi = Image.fromarray(f).resize((H * s, H * s), Image.NEAREST)
                dd = ImageDraw.Draw(fi)
                box(dd, *z['ctr_itc'][t], s, (255, 0, 0), H); box(dd, *z['ctr_pix'][t], s, (0, 220, 255), H)
                im.paste(fi, (x, y)); x += H * s + 6
            for key in ('ctr_itc', 'ctr_pix'):
                cy, cx = z[key][t]
                for f in (fa, fb):
                    if not np.isnan(cy):
                        im.paste(Image.fromarray(np.ascontiguousarray(crop(f, cy, cx))).resize((CROP * cs, CROP * cs), Image.NEAREST), (x, y))
                    x += CROP * cs + 6
            k = sh + r
            txt = (f'#{k} t={t}\nITC(red): {NAMES[ci[t]]} chg {z["chg_itc"][t]}\nPIX(cyan): {NAMES[cp[t]]} chg {z["chg_pix"][t]}\n'
                   f'frame chg {z["fchg"][t]}')
            d.multiline_text((x + 4, y + 4), txt, fill='black')
            meta.append(dict(k=k, t=int(t), itc=NAMES[ci[t]], pix=NAMES[cp[t]], chg_itc=int(z['chg_itc'][t]), chg_pix=int(z['chg_pix'][t]),
                             fchg=int(z['fchg'][t]), ctr_dist=float(np.hypot(*(z['ctr_itc'][t] - z['ctr_pix'][t])))))
        im.save(os.path.join(a.out_dir, f'disagree_sheet_{sh // a.per_sheet}.png'))
    json.dump(meta, open(os.path.join(a.out_dir, 'disagree_sample.json'), 'w'), indent=0)
    print('saved', a.n, 'rows')


if __name__ == '__main__':
    main()
