"""Training-free action codes from ITC (STA-35 itc_loop iteration 1).

For every judge transition (eval_results/lam_judge/set/transitions.json) run ITC (ratio affinity, binarized plan =
Hungarian on A) between frame_a and frame_b, cache per-transition features, then derive action codes for several arms.

  features: python experiments/itc_actions/itc_actions.py features --device cpu      (~1-3 s / transition, cached to features.npz)
  codes:    python experiments/itc_actions/itc_actions.py codes                       (writes codes_<arm>.json, fast)
  then:     python scripts/eval/lam_judge.py score --codes itc_centroid=eval_results/itc_loop/i1/codes_itc_centroid.json

Codes: 0..7 = 8-way direction of Link's motion (R, DR, D, DL, L, UL, U, UR), 8 = STILL.
Rows/cols of A: rows < L frame-t tokens, rows >= L generation slots; cols < L frame-t+1 candidates.
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from experiments.itc_actions.itc_correspondence import DATASETS, affinities, code_digits, encode, prob_of_codes
from utils.utils import load_videotokenizer_from_checkpoint

DIRS = ['R', 'DR', 'D', 'DL', 'L', 'UL', 'U', 'UR']
STILL = 8


# ----------------------------------------------------------------------------- features
def global_shift(a, b, max_s=16, margin=20):
    # a, b float [H, W] gray. Exhaustive SAD over integer shifts: content at (y, x) in a is at (y + dy, x + dx) in b.
    # -> (dy, dx), err(best), err(0)
    H, W = a.shape
    ca = a[margin:H - margin, margin:W - margin]  # [H', W']
    best, err0, errs = None, None, {}
    for dy in range(-max_s, max_s + 1):
        for dx in range(-max_s, max_s + 1):
            cb = b[margin + dy:H - margin + dy, margin + dx:W - margin + dx]
            e = float(np.abs(ca - cb).mean())
            errs[(dy, dx)] = e
            if best is None or e < best[1]:
                best = ((dy, dx), e)
    return best[0], best[1], errs[(0, 0)]


def block_match(a, b, mask, max_s=8):
    # local template match of the changed region of a (bbox of mask, gray) into b -> (dy, dx) of the best match
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return (0, 0), 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    H, W = a.shape
    tpl = a[y0:y1, x0:x1]
    best = None
    for dy in range(-max_s, max_s + 1):
        for dx in range(-max_s, max_s + 1):
            if y0 + dy < 0 or x0 + dx < 0 or y1 + dy > H or x1 + dx > W:
                continue
            e = float(np.abs(tpl - b[y0 + dy:y1 + dy, x0 + dx:x1 + dx]).mean())
            if best is None or e < best[1]:
                best = ((dy, dx), e)
    return best if best is not None else ((0, 0), 0.0)


def itc_plan(tok, pair, device, temp, c_d, c_w, Wp):
    # -> before [L] bool (frame-t tokens not reused), kept [L] bool (t+1 candidates kept), src [L] int (row per column)
    codes, digit_prob, q = encode(tok, pair, device, temp)  # [2, L], [2, L, Ld, nb]
    p_prev = prob_of_codes(digit_prob[1], code_digits(codes[0], q))  # [L(i), L(j)]
    p_max = digit_prob[1].max(-1).values.prod(-1)  # [L]
    p_prev, p_max = p_prev / p_max[None, :], torch.ones_like(p_max)  # ratio variant
    A, _ = affinities(p_prev, p_max, Wp, c_d, c_w)  # [2L, 2L]
    A = A.numpy()
    L = A.shape[0] // 2
    r, c = linear_sum_assignment(np.where(np.isinf(A), -1e9, A), maximize=True)
    src = np.empty(2 * L, int); src[c] = r  # row assigned to each column
    col = np.empty(2 * L, int); col[r] = c  # column assigned to each row
    kept = src[:L] >= L  # [L]
    before = col[:L] >= L  # [L] frame-t token matched to a dummy column (not reused)
    return before, kept, src[:L], int((codes[0] != codes[1]).sum())


def features(args):
    torch.set_num_threads(4)
    tr = json.load(open(args.transitions))['transitions']
    if args.limit:
        tr = tr[:args.limit]
    X = h5py.File(args.frames, 'r')['frames']
    device = torch.device(args.device)
    tok, _ = load_videotokenizer_from_checkpoint(args.tokenizer, device)
    tok.eval()
    patch = 4
    H = X.shape[1]
    Hp = Wp = H // patch
    L = Hp * Wp
    out = {k: [] for k in ['id', 'before', 'kept', 'src', 'codes_changed', 'shift', 'err_best', 'err0', 'pixmask',
                           'block_shift']}
    t0 = time.time()
    for n, t in enumerate(tr):
        pair = np.stack([X[t['frame_a']], X[t['frame_b']]])  # uint8 [2, H, W, C]
        before, kept, src, changed = itc_plan(tok, pair, device, args.temp, args.c_d, args.c_w, Wp)
        g = pair.astype(np.float32).mean(-1) / 255.0  # [2, H, W]
        (dy, dx), eb, e0 = global_shift(g[0], g[1])
        pm = np.abs(pair[0].astype(np.int16) - pair[1].astype(np.int16)).max(-1) > args.pix_thresh  # [H, W]
        bs, _ = block_match(g[0], g[1], pm)
        out['id'].append(str(t['id'])); out['before'].append(before); out['kept'].append(kept); out['src'].append(src)
        out['codes_changed'].append(changed); out['shift'].append((dy, dx)); out['err_best'].append(eb)
        out['err0'].append(e0); out['pixmask'].append(pm); out['block_shift'].append(bs)
        print(f'{n + 1}/{len(tr)} id {t["id"]}: before {before.sum()} kept {kept.sum()} changed {changed} '
              f'shift {(dy, dx)} err {eb:.4f}/{e0:.4f} pix {pm.sum()} block {bs} ({time.time() - t0:.0f}s)', flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, 'features.npz'), **{k: np.array(v) for k, v in out.items()},
                        Wp=Wp, L=L, patch=patch)
    print('saved', os.path.join(args.out_dir, 'features.npz'))


# ----------------------------------------------------------------------------- codes
def dir8(dy, dx):
    # image coords (dy down) -> 0..7 = R, DR, D, DL, L, UL, U, UR
    return int(np.round(np.arctan2(dy, dx) / (np.pi / 4))) % 8


def is_scroll(shift, eb, e0, ratio):
    return tuple(shift) != (0, 0) and eb < ratio * e0


def windowed_centroids(before, kept, Wp, win):
    # densest cluster of before | kept (box-blurred count map), both sets restricted to a (2 win + 1)^2 cell window
    Hp = len(before) // Wp
    b2, k2 = before.reshape(Hp, Wp), kept.reshape(Hp, Wp)
    dens = ndimage.uniform_filter((b2 | k2).astype(float), size=2 * win + 1, mode='constant')
    cy, cx = np.unravel_index(dens.argmax(), dens.shape)
    yy, xx = np.mgrid[:Hp, :Wp]
    w = (np.abs(yy - cy) <= win) & (np.abs(xx - cx) <= win)
    bb, kk = b2 & w, k2 & w
    if bb.sum() == 0 or kk.sum() == 0:
        return None, int(bb.sum()), int(kk.sum()), (bb | kk).reshape(-1)
    cb = np.array([yy[bb].mean(), xx[bb].mean()]); ck = np.array([yy[kk].mean(), xx[kk].mean()])
    return ck - cb, int(bb.sum()), int(kk.sum()), (bb | kk).reshape(-1)


def window_px_mask(cells, Wp, patch):
    # [L] bool cells -> [H, W] bool pixel mask (each cell = patch x patch pixels)
    return np.kron(cells.reshape(-1, Wp), np.ones((patch, patch), bool)).astype(bool)


def codes(args):
    F = np.load(os.path.join(args.out_dir, 'features.npz'))
    Wp = int(F['Wp'])
    arms = {a: {} for a in ['itc_centroid', 'pix_centroid', 'pix_block', 'itc_block', 'scroll_only']}
    tr = {str(t['id']): t for t in json.load(open(args.transitions))['transitions']}
    X = h5py.File(args.frames, 'r')['frames']
    meta = {}
    for n, i in enumerate(F['id']):
        sh, eb, e0 = F['shift'][n], F['err_best'][n], F['err0'][n]
        scroll = is_scroll(sh, eb, e0, args.scroll_ratio)
        scode = dir8(-sh[0], -sh[1]) if scroll else None  # camera follows Link: content moves opposite to Link
        d, nb, nk, wcells = windowed_centroids(F['before'][n], F['kept'][n], Wp, args.win)
        if scroll:
            c_itc = scode
        elif d is not None and min(nb, nk) >= args.min_tokens and np.hypot(*d) >= args.min_disp:
            c_itc = dir8(d[0], d[1])
        else:
            c_itc = STILL
        # ITC ablation: before = after = pixel change mask -> zero displacement by construction
        pm = F['pixmask'][n]
        c_pix = scode if scroll else STILL
        bs = F['block_shift'][n]
        if scroll:
            c_blk = scode
        elif pm.sum() >= args.min_pix and np.hypot(*bs) >= args.min_disp * 4:
            c_blk = dir8(bs[0], bs[1])
        else:
            c_blk = STILL
        # ITC as a localiser: template match of the ITC window (windowed not-reused | kept cells) instead of the
        # pixel change-mask bbox
        if scroll:
            c_ib = scode
        elif min(nb, nk) >= args.min_tokens:
            t = tr[str(i)]
            g = np.stack([X[t['frame_a']], X[t['frame_b']]]).astype(np.float32).mean(-1) / 255.0  # [2, H, W]
            ib, _ = block_match(g[0], g[1], window_px_mask(wcells, Wp, int(F['patch'])))
            c_ib = dir8(ib[0], ib[1]) if np.hypot(*ib) >= args.min_disp * 4 else STILL
        else:
            c_ib = STILL
        arms['itc_block'][str(i)] = c_ib
        arms['itc_centroid'][str(i)] = c_itc; arms['pix_centroid'][str(i)] = c_pix
        arms['pix_block'][str(i)] = c_blk; arms['scroll_only'][str(i)] = scode if scroll else STILL
        meta[str(i)] = {'scroll': bool(scroll), 'shift': [int(sh[0]), int(sh[1])],
                        'itc_disp_cells': None if d is None else [round(float(d[0]), 2), round(float(d[1]), 2)],
                        'n_before': nb, 'n_kept': nk}
    for a, cj in arms.items():
        json.dump(cj, open(os.path.join(args.out_dir, f'codes_{a}.json'), 'w'))
    json.dump(meta, open(os.path.join(args.out_dir, 'meta.json'), 'w'), indent=0)
    print('scrolling transitions:', sum(m['scroll'] for m in meta.values()), '/', len(meta))
    for a, cj in arms.items():
        print(a, np.bincount(list(cj.values()), minlength=9))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['features', 'codes'])
    p.add_argument('--transitions', default='eval_results/lam_judge/set/transitions.json')
    p.add_argument('--frames', default='data/zelda_test_frames.h5')
    p.add_argument('--tokenizer', default=DATASETS['zelda'][0])
    p.add_argument('--out-dir', default='eval_results/itc_loop/i1')
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--temp', type=float, default=0.1)
    p.add_argument('--c-d', type=float, default=0.1)
    p.add_argument('--c-w', type=float, default=0.5)
    p.add_argument('--pix-thresh', type=int, default=40, help='max-channel abs diff for the pixel change mask')
    p.add_argument('--scroll-ratio', type=float, default=0.5, help='scroll if SAD(best shift) < ratio * SAD(0)')
    p.add_argument('--win', type=int, default=5, help='half-size (cells) of the ITC localisation window')
    p.add_argument('--min-tokens', type=int, default=2)
    p.add_argument('--min-disp', type=float, default=0.5, help='min centroid displacement in cells (4 px)')
    p.add_argument('--min-pix', type=int, default=10)
    args = p.parse_args()
    features(args) if args.cmd == 'features' else codes(args)


if __name__ == '__main__':
    main()
