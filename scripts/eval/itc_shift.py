"""Motion hypotheses scored by ITC reuse (STA-35 itc_loop iteration 2).

For each judge transition and each candidate shift s, build a hypothesis frame t_s from frame t (pixel shift, then
re-encode the frame alone), run ITC (ratio affinity, binarized plan) of t_s against t+1 on a column set C, and score s
by the number of t+1 candidates in C that the plan reuses instead of generating. The argmax shift is the action.
  global: t_s = frame t translated by s (edge-replicated), C = interior cells -> camera scroll (Link = -s)
  local:  t_s = frame t with only the window W (largest cluster of i1 ITC-kept cells, dilated) translated, C = W -> Link = s
Ablations on the same hypotheses: pixel SAD over C's pixels, plain same-position token equality over C (no ITC).

  scores: python scripts/eval/itc_shift.py scores --device cpu   (needs eval_results/itc_loop/i1/features.npz)
  codes:  python scripts/eval/itc_shift.py codes                  (writes codes_<arm>.json)

Restricted ITC: with the ratio affinity p <= 1, reusing candidate j by token i (value p_ij - c_d d_ij) only beats
"generate j, drop i" (value p_max_j - c_w = 1 - c_w) when w_ij = p_ij - c_d d_ij - (1 - c_w) > 0, so the binarized plan
on the [2L, 2L] matrix equals a max-weight bipartite matching on the positive part of w (exchange argument); we solve
that rectangular problem on rows R (C dilated) x columns C with linear_sum_assignment.
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from itc_actions import STILL, dir8  # noqa: E402
from itc_correspondence import DATASETS, code_digits, encode, prob_of_codes  # noqa: E402
from utils.utils import load_videotokenizer_from_checkpoint  # noqa: E402


def shift_grid(vals):
    return [(dy, dx) for dy in vals for dx in vals]


def translate(img, dy, dx):
    # img [H, W, C]; content at (y, x) moves to (y + dy, x + dx), edges replicated
    H, W = img.shape[:2]
    ys = np.clip(np.arange(H) - dy, 0, H - 1)
    xs = np.clip(np.arange(W) - dx, 0, W - 1)
    return img[ys][:, xs]


def local_window(kept, Hp, Wp, dil, tight=False):
    # kept [L] bool -> W [L] bool: largest (by kept count) 8-connected cluster of kept dilated by dil cells
    # (tight: only the kept cells inside that cluster)
    k2 = kept.reshape(Hp, Wp)
    if k2.sum() == 0:
        return None
    d = ndimage.binary_dilation(k2, structure=np.ones((2 * dil + 1, 2 * dil + 1), bool))
    lab, n = ndimage.label(d, structure=np.ones((3, 3), bool))
    counts = ndimage.sum(k2, lab, index=np.arange(1, n + 1))
    W = lab == 1 + int(np.argmax(counts))
    return (W & k2 if tight else W).reshape(-1)


def dilate_cells(cells, Hp, Wp, r):
    return ndimage.binary_dilation(cells.reshape(Hp, Wp), structure=np.ones((2 * r + 1, 2 * r + 1), bool)).reshape(-1)


def itc_reuse(digits_s, dp_b, p_max_b, R, C, pos, c_d, c_w):
    # digits_s [L, Ld] (hypothesis tokens), dp_b [L, Ld, nb] (t+1 candidate digit probs), p_max_b [L]
    # R, C index arrays -> (#C candidates reused, total positive reuse gain)
    p = prob_of_codes(dp_b[C], digits_s[R]).numpy() / p_max_b[C][None, :]  # [|R|, |C|] ratio affinity
    dist = np.sqrt(((pos[R][:, None, :] - pos[C][None, :, :]) ** 2).sum(-1))  # [|R|, |C|]
    w = np.maximum(p - c_d * dist - (1 - c_w), 0)  # [|R|, |C|] gain of reuse over generation
    r, c = linear_sum_assignment(w, maximize=True)
    g = w[r, c]
    return int((g > 0).sum()), float(g.sum())


def scores(args):
    torch.set_num_threads(4)
    tr = json.load(open(args.transitions))['transitions']
    F = np.load(os.path.join(args.i1_dir, 'features.npz'))
    fidx = {str(i): n for n, i in enumerate(F['id'])}
    if args.limit:
        tr = tr[:args.limit]
    X = h5py.File(args.frames, 'r')['frames']
    device = torch.device(args.device)
    tok, _ = load_videotokenizer_from_checkpoint(args.tokenizer, device)
    tok.eval()
    patch, H = 4, X.shape[1]
    Hp = Wp = H // patch
    L = Hp * Wp
    pos = np.stack([np.arange(L) // Wp, np.arange(L) % Wp], -1).astype(np.float64)  # [L, 2]
    S = shift_grid(args.shifts)
    b = args.border
    interior = np.zeros((Hp, Wp), bool); interior[b:Hp - b, b:Wp - b] = True
    interior = interior.reshape(-1)
    keys = ['itc', 'gain', 'tok', 'sad']
    out = {f'{v}_{k}': [] for v in ['g', 'l'] for k in keys}
    out.update({'id': [], 'win': []})
    t0 = time.time()
    for n, t in enumerate(tr):
        fa, fb = X[t['frame_a']], X[t['frame_b']]  # uint8 [H, W, C]
        M = local_window(F['kept'][fidx[str(t['id'])]], Hp, Wp, args.dil, args.local_hyp != 'window')
        W = M if M is None or args.local_hyp != 'paste' else dilate_cells(M, Hp, Wp, args.paste_cdil)  # columns
        Mpx = None if M is None else np.kron(M.reshape(Hp, Wp), np.ones((patch, patch), bool)).astype(bool)  # [H, W]
        hyp = [translate(fa, dy, dx) for dy, dx in S] if 'g' in args.variants else []  # global hypotheses
        if M is not None and 'l' in args.variants:
            for dy, dx in S:
                h = fa.copy()
                if args.local_hyp != 'paste':  # window content translated, cut to the window
                    h[Mpx] = translate(fa, dy, dx)[Mpx]
                else:  # paste: the changed cells' pixels moved by s, everything else (incl. vacated pixels) kept
                    dst = translate(Mpx[..., None].astype(np.uint8), dy, dx)[..., 0].astype(bool)
                    h[dst] = translate(fa, dy, dx)[dst]
                hyp.append(h)
        frames = np.stack(hyp + [fb])  # [N, H, W, C]
        with torch.no_grad():
            cs, dps = [], []
            for i in range(0, len(frames), args.batch):
                c, dp, q = encode(tok, frames[i:i + args.batch], device, args.temp)  # [n, L], [n, L, Ld, nb]
                cs.append(c); dps.append(dp)
            codes, dp = torch.cat(cs), torch.cat(dps)
        dp_b, codes_b = dp[-1], codes[-1].numpy()  # [L, Ld, nb], [L]
        p_max_b = dp_b.max(-1).values.prod(-1).numpy()  # [L]
        gb = fb.astype(np.float32).mean(-1) / 255.0  # [H, W]
        off_l = len(S) if 'g' in args.variants else 0
        for v, (C, off) in {'g': (interior, 0), 'l': (W, off_l)}.items():
            if C is None or v not in args.variants:
                for k in keys:
                    out[f'{v}_{k}'].append(np.full(len(S), np.nan))
                continue
            Ci = np.nonzero(C)[0]
            Ri = np.nonzero(dilate_cells(C, Hp, Wp, args.row_dil))[0]
            Cpx = np.kron(C.reshape(Hp, Wp), np.ones((patch, patch), bool)).astype(bool)
            res = {k: [] for k in keys}
            for si in range(len(S)):
                digits = code_digits(codes[off + si], q)  # [L, Ld]
                nr, g = itc_reuse(digits, dp_b, p_max_b, Ri, Ci, pos, args.c_d, args.c_w)
                res['itc'].append(nr); res['gain'].append(g)
                res['tok'].append(int((codes[off + si].numpy()[Ci] == codes_b[Ci]).sum()))
                gs = frames[off + si].astype(np.float32).mean(-1) / 255.0
                res['sad'].append(float(np.abs(gs - gb)[Cpx].mean()))
            for k in keys:
                out[f'{v}_{k}'].append(np.array(res[k], float))
        out['id'].append(str(t['id'])); out['win'].append(np.zeros(L, bool) if W is None else W)
        msg = f'{n + 1}/{len(tr)} id {t["id"]}: |W| {0 if W is None else W.sum()}'
        for v in args.variants:
            sc = out[f'{v}_itc'][-1]
            if not np.isnan(sc).any():
                msg += f' {v} best {S[int(np.argmax(sc))]} ({sc.max():.0f} vs0 {sc[S.index((0, 0))]:.0f})'
        print(msg + f' ({time.time() - t0:.0f}s)', flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, f'scores{args.tag}.npz'), **{k: np.array(v) for k, v in out.items()},
                        shifts=np.array(S), n_interior=int(interior.sum()))
    print('saved', os.path.join(args.out_dir, f'scores{args.tag}.npz'))


# ----------------------------------------------------------------------------- codes
def best_shift(sc, S, higher_better=True):
    # -> (best s, score gain of best over s=0); ties -> smaller |s|
    s0 = S.index((0, 0))
    v = sc if higher_better else -sc
    order = sorted(range(len(S)), key=lambda i: (-v[i], np.hypot(*S[i])))
    i = order[0]
    return S[i], float(v[i] - v[s0])


def codes(args):
    Z = dict(np.load(os.path.join(args.out_dir, f'scores{args.global_tag}.npz')))
    Zl = np.load(os.path.join(args.out_dir, f'scores{args.local_tag}.npz'))  # local variant (may be another run)
    assert list(Zl['id']) == list(Z['id'])
    Z.update({k: Zl[k] for k in Zl.files if k.startswith('l_')})
    S_l = [tuple(int(v) for v in s) for s in Zl['shifts']]  # local shift grid (may differ from the global one)
    F = np.load(os.path.join(args.i1_dir, 'features.npz'))
    fidx = {str(i): n for n, i in enumerate(F['id'])}
    S = [tuple(int(v) for v in s) for s in Z['shifts']]
    n_int = int(Z['n_interior'])
    def rel_gain(sc, k, best, gain, n_cols, S=S):
        # gain of the best shift relative to what s=0 leaves unexplained: for reuse counts the fraction of the
        # candidates not reused at s=0 that the shift makes reusable, for SAD the relative SAD drop
        s0 = sc[S.index((0, 0))]
        return gain / max(1e-9, s0 if k == 'sad' else n_cols - s0) if best != (0, 0) else 0.0
    rules = {'itc': ('itc', True, args.g_rel, args.l_rel), 'tok': ('tok', True, args.g_rel, args.l_rel),
             'sad': ('sad', False, args.g_sad, args.l_sad)}  # (score key, higher is better, global / local min rel gain)
    arms = {}
    for n, i in enumerate(Z['id']):
        fi = fidx[str(i)]
        sh, eb, e0 = F['shift'][fi], F['err_best'][fi], F['err0'][fi]
        sad_scroll = tuple(sh) != (0, 0) and eb < args.scroll_ratio * e0
        n_loc = int(Zl['win'][n].sum())
        for name, (k, hb, gmin, lmin) in rules.items():
            gsc = Z[f'g_{k}'][n]
            gs, gg = best_shift(gsc, S, hb)
            g_code = dir8(-gs[0], -gs[1]) if rel_gain(gsc, k, gs, gg, n_int) >= gmin else None
            l_code, lsc = None, Z[f'l_{k}'][n]
            if not np.isnan(lsc).any():
                ls, lg = best_shift(lsc, S_l, hb)
                l_code = dir8(ls[0], ls[1]) if lg >= (args.l_min if k != 'sad' else 0) and \
                    rel_gain(lsc, k, ls, lg, n_loc, S_l) >= lmin else None
            scode = dir8(-sh[0], -sh[1]) if sad_scroll else None
            arms.setdefault(f'{name}_shift', {})[str(i)] = g_code if g_code is not None else (
                l_code if l_code is not None else STILL)
            arms.setdefault(f'{name}_global', {})[str(i)] = g_code if g_code is not None else STILL
            arms.setdefault(f'scroll_{name}_local', {})[str(i)] = scode if scode is not None else (
                l_code if l_code is not None else STILL)
    for a, cj in arms.items():
        a = a + (args.local_tag if 'local' in a or a.endswith('_shift') else '') + (args.global_tag if 'global' in a or a.endswith('_shift') else '')
        json.dump(cj, open(os.path.join(args.out_dir, f'codes_{a}.json'), 'w'))
        print(a, np.bincount(list(cj.values()), minlength=9))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['scores', 'codes'])
    p.add_argument('--transitions', default='eval_results/lam_judge/set/transitions.json')
    p.add_argument('--frames', default='data/zelda_test_frames.h5')
    p.add_argument('--tokenizer', default=DATASETS['zelda'][0])
    p.add_argument('--i1-dir', default='eval_results/itc_loop/i1')
    p.add_argument('--out-dir', default='eval_results/itc_loop/i2')
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--batch', type=int, default=26)
    p.add_argument('--shifts', type=int, nargs='+', default=[-8, -4, 0, 4, 8], help='per-axis pixel shifts')
    p.add_argument('--temp', type=float, default=0.1)
    p.add_argument('--c-d', type=float, default=0.1)
    p.add_argument('--c-w', type=float, default=0.5)
    p.add_argument('--border', type=int, default=3, help='cells excluded from the global column set')
    p.add_argument('--dil', type=int, default=2, help='dilation (cells) of the ITC-kept cells for the local window')
    p.add_argument('--local-hyp', choices=['window', 'tight', 'paste'], default='window',
                   help='window: W (dilated cluster) translated; tight: same with W = kept cells of the cluster; paste: only the kept cells of the cluster moved, '
                        'columns = those cells dilated by --paste-cdil')
    p.add_argument('--paste-cdil', type=int, default=2)
    p.add_argument('--variants', nargs='+', default=['g', 'l'], choices=['g', 'l'])
    p.add_argument('--tag', default='', help='scores<tag>.npz; codes reads local scores from --local-tag')
    p.add_argument('--local-tag', default='')
    p.add_argument('--global-tag', default='', help='codes reads global scores from scores<global-tag>.npz')
    p.add_argument('--row-dil', type=int, default=2, help='rows R = columns C dilated by this many cells')
    p.add_argument('--g-rel', type=float, default=0.02,
                   help='global itc/tok: min relative gain (calibrated label-free vs the SAD scroll flag)')
    p.add_argument('--l-rel', type=float, default=0.1, help='local itc/tok: min relative gain')
    p.add_argument('--l-min', type=float, default=2, help='local itc/tok: min reuse gain in cells')
    p.add_argument('--g-sad', type=float, default=0.05, help='sad ablation: min global relative SAD drop')
    p.add_argument('--l-sad', type=float, default=0.05, help='sad ablation: min local relative SAD drop')
    p.add_argument('--scroll-ratio', type=float, default=0.5)
    args = p.parse_args()
    scores(args) if args.cmd == 'scores' else codes(args)


if __name__ == '__main__':
    main()
