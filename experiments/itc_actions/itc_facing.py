"""ITC as the localiser, Link's facing as the action (STA-35 itc_loop iteration 3).

Link faces the way he walks, so on still-camera pairs the action can be read from his sprite in frame t+1. ITC
localises him (largest cluster of t+1 candidates the ITC plan keeps instead of reusing); a 24x24 crop there is
classified against facing prototypes mined without labels from scrolling train pairs (label = flipped scroll direction).

  mine:   python experiments/itc_actions/itc_facing.py mine --n-pairs 400   (train pairs, ITC on scroll-compensated frames, cached)
  codes:  python experiments/itc_actions/itc_facing.py codes                (judge transitions; needs eval_results/itc_loop/i1/features.npz)
  sheet:  python experiments/itc_actions/itc_facing.py sheet                (contact sheet of still-camera moves with crop centres)
  then:   python scripts/eval/lam_judge.py score --codes itc_facing=eval_results/itc_loop/i3/codes_itc_facing.json

Codes: 0..7 = R, DR, D, DL, L, UL, U, UR, 8 = STILL. Mining keeps 4-way labels (R, D, L, U) only.
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
from scipy import ndimage

from experiments.itc_actions.itc_actions import STILL, dir8, global_shift, is_scroll, itc_plan
from experiments.itc_actions.itc_correspondence import DATASETS
from experiments.itc_actions.itc_shift import translate

PATCH, CROP = 4, 24
FOUR = [0, 2, 4, 6]  # R, D, L, U in the 8-way code


# ----------------------------------------------------------------------------- localisers
def densest(cells, Hp, Wp, win=6, prior=None):
    # cells [L] bool -> (n cells in the densest win x win window, centroid (y, x) px of those cells, n in the window bbox)
    # prior: max distance (px) of the window centre from the screen centre (None = anywhere)
    c2 = cells.reshape(Hp, Wp).astype(float)
    if c2.sum() == 0:
        return None
    dens = ndimage.uniform_filter(c2, size=win, mode='constant')  # [Hp, Wp]
    if prior is not None:
        yy, xx = np.mgrid[:Hp, :Wp]
        dens[np.hypot((yy + 0.5) * PATCH - Hp * PATCH / 2, (xx + 0.5) * PATCH - Wp * PATCH / 2) > prior] = -1
    cy, cx = np.unravel_index(dens.argmax(), dens.shape)
    y0, x0 = max(cy - win // 2, 0), max(cx - win // 2, 0)
    w = np.zeros_like(c2, bool); w[y0:y0 + win, x0:x0 + win] = True
    ys, xs = np.nonzero(w & (c2 > 0))
    if len(ys) == 0:
        return None
    return len(ys), ((ys.mean() + 0.5) * PATCH, (xs.mean() + 0.5) * PATCH), int(c2.sum())


def pix_cells(mask, Hp, Wp):
    # [H, W] bool pixel mask -> [L] bool cells (any changed pixel in the cell)
    return mask.reshape(Hp, PATCH, Wp, PATCH).any((1, 3)).reshape(-1)


def crop(img, cy, cx):
    # img [H, W, C] uint8 -> [CROP, CROP, C] centred at (cy, cx), clamped inside the frame
    H, W = img.shape[:2]
    y0 = int(np.clip(round(cy) - CROP // 2, 0, H - CROP)); x0 = int(np.clip(round(cx) - CROP // 2, 0, W - CROP))
    return img[y0:y0 + CROP, x0:x0 + CROP]


def feat(c, kind):
    # crop [CROP, CROP, C] uint8 -> normalised feature vector
    g = c.astype(np.float32) / 255.0
    if kind == 'hog':  # 4x4-cell histograms of 8 gradient orientations on the gray crop
        y = g.mean(-1)
        gy, gx = np.gradient(y)
        mag, ang = np.hypot(gy, gx), (np.arctan2(gy, gx) % (2 * np.pi))
        b = (ang / (2 * np.pi) * 8).astype(int) % 8
        h = np.zeros((CROP // 6, CROP // 6, 8))
        for i in range(CROP // 6):
            for j in range(CROP // 6):
                h[i, j] = np.bincount(b[i * 6:(i + 1) * 6, j * 6:(j + 1) * 6].ravel(),
                                      mag[i * 6:(i + 1) * 6, j * 6:(j + 1) * 6].ravel(), minlength=8)
        v = h.ravel()
    else:
        v = (g - g.mean((0, 1))).ravel()
    return v / (np.linalg.norm(v) + 1e-8)


# ----------------------------------------------------------------------------- mining (train split)
def mine(args):
    torch.set_num_threads(4)
    from utils.utils import load_videotokenizer_from_checkpoint
    h = h5py.File(args.train_frames, 'r')
    X, src = h['frames'], h['source_index'][:]
    rng = np.random.default_rng(args.seed)
    gap = args.gap
    cand = np.nonzero(src[gap:] == src[:-gap] + gap)[0] if args.contiguous else np.arange(len(src) - gap)
    idx = np.sort(rng.choice(cand, args.n_pairs, replace=False))
    device = torch.device('cpu')
    tok, _ = load_videotokenizer_from_checkpoint(args.tokenizer, device)
    tok.eval()
    H = X.shape[1]; Hp = Wp = H // PATCH
    out = {k: [] for k in ['frame', 'shift', 'label', 'itc_c', 'itc_n', 'pix_c', 'pix_n', 'crop_itc', 'crop_pix', 'crop_ctr',
                           'chg_itc', 'chg_pix', 'chg_frame']}
    t0, n_sc = time.time(), 0
    for n, i in enumerate(idx):
        fa, fb = X[i], X[i + gap]
        g = np.stack([fa, fb]).astype(np.float32).mean(-1) / 255.0
        (dy, dx), eb, e0 = global_shift(g[0], g[1])
        if not is_scroll((dy, dx), eb, e0, args.scroll_ratio):
            continue
        n_sc += 1
        lab = dir8(-dy, -dx)
        fs = translate(fa, dy, dx)  # frame t moved with the camera -> only what does not follow it changes
        with torch.no_grad():
            _, kept, _, _ = itc_plan(tok, np.stack([fs, fb]), device, args.temp, args.c_d, args.c_w, Wp)
        border = np.zeros((Hp, Wp), bool); m = (abs(dy) + PATCH - 1) // PATCH + 1, (abs(dx) + PATCH - 1) // PATCH + 1
        border[m[0]:Hp - m[0], m[1]:Wp - m[1]] = True  # drop the edge strip revealed by the scroll
        ci = densest(kept & border.reshape(-1), Hp, Wp, prior=args.prior)
        pm = np.abs(fs.astype(np.int16) - fb.astype(np.int16)).max(-1) > args.pix_thresh
        cp = densest(pix_cells(pm, Hp, Wp) & border.reshape(-1), Hp, Wp, prior=args.prior)
        if ci is None or cp is None:
            continue
        out['frame'].append(int(i)); out['shift'].append((dy, dx)); out['label'].append(lab)
        out['itc_c'].append(ci[1]); out['itc_n'].append(ci[0]); out['pix_c'].append(cp[1]); out['pix_n'].append(cp[0])
        out['crop_itc'].append(crop(fb, *ci[1])); out['crop_pix'].append(crop(fb, *cp[1]))
        out['crop_ctr'].append(crop(fb, H / 2, H / 2))  # fixed screen-centre crop (no localiser)
        # gate calibration (label-free): changed pixels in the crop / interior of the frame on these walking pairs
        out['chg_itc'].append(int(crop(pm[..., None], *ci[1]).sum())); out['chg_pix'].append(int(crop(pm[..., None], *cp[1]).sum()))
        bpx = np.kron(border, np.ones((PATCH, PATCH), bool)).astype(bool)
        out['chg_frame'].append(int((pm & bpx).sum()))
        print(f'{n + 1}/{len(idx)} frame {i}: shift {(dy, dx)} lab {lab} itc {ci[0]} cells @ {np.round(ci[1])} '
              f'pix {cp[0]} @ {np.round(cp[1])} ({time.time() - t0:.0f}s, {n_sc} scrolling)', flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, 'mined.npz'), **{k: np.array(v) for k, v in out.items()})
    print('saved', len(out['frame']), 'mined crops from', n_sc, 'scrolling of', len(idx), 'pairs')


# ----------------------------------------------------------------------------- judge-set localisation
def judge_loc(args):
    # per judge transition: scroll flag/code, ITC cluster, pixel-change cluster, pix_block location (i1 features)
    F = np.load(os.path.join(args.i1_dir, 'features.npz'))
    tr = {str(t['id']): t for t in json.load(open(args.transitions))['transitions']}
    X = h5py.File(args.frames, 'r')['frames']
    H = X.shape[1]; Hp = Wp = H // PATCH
    rows = {}
    for n, i in enumerate(F['id']):
        t = tr[str(i)]
        fa, fb = X[t['frame_a']], X[t['frame_b']]
        sh, eb, e0 = F['shift'][n], F['err_best'][n], F['err0'][n]
        scroll = is_scroll(sh, eb, e0, args.scroll_ratio)
        ci = densest(F['kept'][n], Hp, Wp)
        pm = F['pixmask'][n]
        cp = densest(pix_cells(pm, Hp, Wp), Hp, Wp)
        # oracle-ish crop: the i1 pix_block template (bbox of the change mask) at its best match in t+1
        ys, xs = np.nonzero(pm)
        bs = F['block_shift'][n]
        co = None if len(ys) == 0 else ((ys.min() + ys.max() + 1) / 2 + bs[0], (xs.min() + xs.max() + 1) / 2 + bs[1])
        rows[str(i)] = dict(fa=fa, fb=fb, scroll=scroll, scode=dir8(-sh[0], -sh[1]) if scroll else None, itc=ci,
                            pix=cp, orc=co, pm=pm, block=tuple(int(v) for v in bs))
    return rows


# ----------------------------------------------------------------------------- facing classifier + codes
def bank(args, which):
    # mined crops of one localiser -> (features [N, D], 4-way labels [N]); pure-axis scrolls only, optional L<->R mirror
    M = np.load(os.path.join(args.out_dir, 'mined.npz'))
    keep = np.isin(M['label'], FOUR)
    C, y = M[f'crop_{which}'][keep], M['label'][keep]
    if args.mirror:  # Link's L and R sprites are mirror images
        lr = np.isin(y, [0, 4])
        C, y = np.concatenate([C, C[lr][:, :, ::-1]]), np.concatenate([y, 4 - y[lr]])
    return np.stack([feat(c, args.feat) for c in C]), y


def knn(Fb, yb, f, k, exclude=None):
    sim = Fb @ f  # [N] cosine similarity
    if exclude is not None:
        sim[exclude] = -np.inf
    top = np.argsort(-sim)[:k]
    votes = np.bincount(yb[top], weights=sim[top] - sim[top].min() + 1e-3, minlength=8)
    return int(votes.argmax())


def loo(Fb, yb, k, n_orig):
    # leave-one-out accuracy on the original (non-mirrored) mined crops
    pred = [knn(Fb, yb, Fb[i], k, exclude=[i]) for i in range(n_orig)]
    return float(np.mean(np.array(pred) == yb[:n_orig]))


def codes(args):
    rows = judge_loc(args)
    banks = {}
    for which in ['itc', 'pix', 'ctr']:
        Fb, yb = bank(args, which)
        n_orig = int(np.isin(np.load(os.path.join(args.out_dir, 'mined.npz'))['label'], FOUR).sum())
        banks[which] = (Fb, yb)
        print(f'bank {which}: {n_orig} crops (+mirror {len(yb) - n_orig}), per class {np.bincount(yb, minlength=8)[FOUR]}, '
              f'LOO {args.k}-NN acc {loo(Fb, yb, args.k, n_orig):.3f}', flush=True)
    arms = {'itc_facing': ('itc', 'itc'), 'pix_facing': ('pix', 'pix'), 'facing_oracle_crop': ('orc', 'itc')}
    out = {a: {} for a in arms}
    meta = {}
    for i, r in rows.items():
        meta[i] = {'scroll': bool(r['scroll'])}
        for a, (loc, bk) in arms.items():
            if r['scroll']:
                out[a][i] = r['scode']; continue
            c = r[loc]
            if c is None:
                out[a][i] = STILL; continue
            cy, cx = c if loc == 'orc' else c[1]
            y0 = int(np.clip(round(cy) - CROP // 2, 0, 128 - CROP)); x0 = int(np.clip(round(cx) - CROP // 2, 0, 128 - CROP))
            changed = int(r['pm'][y0:y0 + CROP, x0:x0 + CROP].sum())
            if changed < args.min_pix:
                out[a][i] = STILL
            else:
                out[a][i] = knn(*banks[bk], feat(crop(r['fb'], cy, cx), args.feat), args.k)
            meta[i][a] = {'centre': [round(float(cy)), round(float(cx))], 'changed_px': changed, 'code': out[a][i]}
    tag = args.tag
    for a, cj in out.items():
        json.dump(cj, open(os.path.join(args.out_dir, f'codes_{a}{tag}.json'), 'w'))
        print(a + tag, np.bincount(list(cj.values()), minlength=9))
    json.dump(meta, open(os.path.join(args.out_dir, f'meta{tag}.json'), 'w'), indent=0)


def report(args):
    # per arm: all / moves / still-camera moves / scrolling moves (i1 scroll split) + STILL/NONCONTROL/ATTACK code tables
    from evaluation.action_metrics import label_metrics as metrics
    from evaluation.zelda_judge import LABELS, MOVES
    labels = json.load(open(args.labels))
    rows = judge_loc(args)
    for spec in args.arms:
        name, path = spec.split('=', 1)
        cj = json.load(open(path))
        nc = int(max(cj.values())) + 1
        ids = [i for i in labels if i in cj]
        sub = {'all': ids, 'moves': [i for i in ids if labels[i] in MOVES]}
        sub['still'] = [i for i in sub['moves'] if not rows[i]['scroll']]
        sub['scroll'] = [i for i in sub['moves'] if rows[i]['scroll']]
        msg = name + ':'
        for k, v in sub.items():
            m = metrics([cj[i] for i in v], [labels[i] for i in v], nc, LABELS if k == 'all' else MOVES)
            msg += f" {k} n={m['n']} nmi_adj {m['nmi_adj']} pur {m['purity']}/{m['majority_baseline']} |"
        exact = sum(cj[i] == dir8(*{'R': (0, 1), 'D': (1, 0), 'L': (0, -1), 'U': (-1, 0), 'DR': (1, 1), 'DL': (1, -1),
                                    'UL': (-1, -1), 'UR': (-1, 1)}[labels[i]]) for i in sub['still'])
        print(msg + f' still exact {exact}/{len(sub["still"])}')
        for lab in ['STILL', 'NONCONTROL', 'ATTACK']:
            v = [cj[i] for i in ids if labels[i] == lab]
            print(f'   {lab}: n={len(v)} non-STILL {sum(c != STILL for c in v)} codes {np.bincount(v, minlength=9).tolist()}')
        print('   still-cam moves (id label->code):', ' '.join(f'{i}:{labels[i]}->{cj[i]}' for i in sub['still']))


def sheet(args):
    from PIL import Image, ImageDraw
    rows = judge_loc(args)
    labels = json.load(open(args.labels))
    ids = [i for i, r in rows.items() if not r['scroll'] and labels.get(i) in args.sheet_labels]
    tiles = []
    for i in ids:
        r = rows[i]
        im = Image.fromarray(np.concatenate([r['fa'], r['fb']], 1)).resize((512, 256), Image.NEAREST)
        d = ImageDraw.Draw(im)
        for key, col in [('itc', (255, 0, 0)), ('pix', (0, 255, 255))]:
            if r[key] is not None:
                cy, cx = r[key][1]
                for off in (0, 256):
                    d.rectangle([off + 2 * (cx - 12), 2 * (cy - 12), off + 2 * (cx + 12), 2 * (cy + 12)], outline=col)
        d.text((4, 4), f'id {i} {labels[i]} itc n={None if r["itc"] is None else r["itc"][0]}', fill=(255, 255, 0))
        tiles.append(np.array(im))
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, f'sheet_{"_".join(args.sheet_labels) if len(args.sheet_labels) < 4 else "moves"}.png')
    Image.fromarray(np.concatenate(tiles, 0)).save(p)
    print('saved', p, len(ids), 'transitions:', ids)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['mine', 'codes', 'sheet', 'report'])
    p.add_argument('--arms', nargs='+', default=[], help='report: name=codes.json')
    p.add_argument('--feat', choices=['pix', 'hog'], default='pix')
    p.add_argument('--k', type=int, default=7)
    p.add_argument('--mirror', type=int, default=1)
    p.add_argument('--min-pix', type=int, default=20, help='min changed pixels inside the crop for a facing code')
    p.add_argument('--tag', default='')
    p.add_argument('--transitions', default='eval_results/lam_judge/set/transitions.json')
    p.add_argument('--labels', default='eval_results/lam_judge/set/labels.json')
    p.add_argument('--frames', default='data/zelda_test_frames.h5')
    p.add_argument('--train-frames', default='data/zelda_train_frames.h5')
    p.add_argument('--tokenizer', default=DATASETS['zelda'][0])
    p.add_argument('--i1-dir', default='eval_results/itc_loop/i1')
    p.add_argument('--out-dir', default='eval_results/itc_loop/i3')
    p.add_argument('--n-pairs', type=int, default=400)
    p.add_argument('--gap', type=int, default=4)
    p.add_argument('--contiguous', type=int, default=1, help='only pairs whose source_index differs by exactly gap')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--temp', type=float, default=0.1)
    p.add_argument('--c-d', type=float, default=0.1)
    p.add_argument('--c-w', type=float, default=0.5)
    p.add_argument('--prior', type=float, default=40, help='mining: max px from the screen centre for the Link window')
    p.add_argument('--pix-thresh', type=int, default=40)
    p.add_argument('--scroll-ratio', type=float, default=0.5)
    p.add_argument('--sheet-labels', nargs='+', default=['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR'])
    args = p.parse_args()
    {'mine': mine, 'sheet': sheet, 'codes': codes, 'report': report}[args.cmd](args)


if __name__ == '__main__':
    main()
