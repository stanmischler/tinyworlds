"""Pseudo-labels from the frozen i4 teacher for every gap-4 train pair (STA-35 itc_loop iteration 5).

Teacher = i4 gate_stat_itc (no further tuning): flipped SAD scroll direction if the camera scrolls, else the facing k-NN
(i3 mined bank) of a crop of frame t+1 at the densest ITC-kept cluster, emitted only if the crop changed >= t_chg px and
the frame changed <= t_frame px (two-level Otsu + cap, calibrated on train pairs in i4), else STILL. The ITC ablation
teacher gate_stat_pix (crop at the densest pixel-change cluster, its own Otsu threshold) is computed in the same pass.

  prep:   python scripts/eval/itc_pseudo.py prep          (freeze thresholds + facing banks -> eval_results/itc_loop/i5/teacher.npz)
  judge:  python scripts/eval/itc_pseudo.py judge         (relabel the 200 judge transitions; must equal the i4 codes)
  label:  python scripts/eval/itc_pseudo.py label --lo 0 --hi 200 --out x.npz   (train pairs (t, t+gap), t in [lo, hi))
  merge:  python scripts/eval/itc_pseudo.py merge --parts a.npz b.npz --out data/zelda_train_itc_pseudo_gap4.npz

Output rows are .h5 frame indices t (the pair (t, t + gap)); -1 = no label (non-contiguous pair or not computed).
Codes: 0..7 = R, DR, D, DL, L, UL, U, UR, 8 = STILL.
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from itc_actions import STILL, dir8, global_shift, is_scroll, itc_plan  # noqa: E402
from itc_correspondence import DATASETS  # noqa: E402
from itc_facing import PATCH, bank, crop, densest, feat, knn, pix_cells  # noqa: E402

TEACHERS = ['itc', 'pix']  # code_itc = gate_stat_itc, code_pix = gate_stat_pix


def prep(args):
    from itc_gate import calibrate
    cal_args = SimpleNamespace(i3_dir=args.i3_dir, out_dir=args.i4_dir, train_frames=args.train_frames, gap=args.gap,
                               frame_margin=2.0)
    out = {}
    for which in TEACHERS:
        cal, _ = calibrate(cal_args, which)
        Fb, yb = bank(SimpleNamespace(out_dir=args.i3_dir, mirror=1, feat='pix'), which)
        out.update({f'{which}_t_chg': cal['t_chg_px'], f'{which}_t_frame': cal['t_frame_px'], f'{which}_Fb': Fb.astype(np.float32),
                    f'{which}_yb': yb})
        print(which, 't_chg', round(cal['t_chg_px'], 2), 't_frame', round(cal['t_frame_px'], 1), 'bank', Fb.shape)
    os.makedirs(os.path.dirname(args.teacher), exist_ok=True)
    np.savez_compressed(args.teacher, **out)
    print('saved', args.teacher)


def load_teacher(path):
    z = np.load(path)
    return {w: dict(t_chg=float(z[f'{w}_t_chg']), t_frame=float(z[f'{w}_t_frame']), Fb=z[f'{w}_Fb'].astype(np.float64),
                    yb=z[f'{w}_yb']) for w in TEACHERS}


def gated_code(T, fa, fb, c, fchg, k=7, min_pix=20, thresh=40):
    # densest-cluster result c of one localiser -> teacher code (itc_gate.codes, gate 'stat')
    if c is None:
        return STILL, 0, (np.nan, np.nan)
    cy, cx = c[1]
    ca, cb = crop(fa, cy, cx), crop(fb, cy, cx)
    chg = int((np.abs(ca.astype(np.int16) - cb.astype(np.int16)).max(-1) > thresh).sum())  # = pair_feats()['chg']
    if chg < min_pix or not (chg >= T['t_chg'] and fchg <= T['t_frame']):
        return STILL, chg, (cy, cx)
    return knn(T['Fb'], T['yb'], feat(cb, 'pix'), k), chg, (cy, cx)


def label_pair(fa, fb, tok, T, device, args):
    # fa, fb [H, W, C] uint8 -> dict(code_itc, code_pix, scroll, shift, fchg, chg_*, ctr_*, itc_run)
    H = fa.shape[0]; Hp = Wp = H // PATCH
    g = np.stack([fa, fb]).astype(np.float32).mean(-1) / 255.0  # [2, H, W]
    sh, eb, e0 = global_shift(g[0], g[1])
    pm = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(-1) > args.pix_thresh  # [H, W]
    fchg = int(pm.sum())
    r = dict(scroll=False, shift=tuple(int(v) for v in sh), fchg=fchg, itc_run=False)
    if is_scroll(sh, eb, e0, args.scroll_ratio):
        c = dir8(-sh[0], -sh[1])  # camera follows Link: content moves opposite to him
        r.update(scroll=True, **{f'code_{w}': c for w in TEACHERS}, **{f'chg_{w}': -1 for w in TEACHERS},
                 **{f'ctr_{w}': (np.nan, np.nan) for w in TEACHERS})
        return r
    # pixel-crop ablation teacher (no ITC)
    r['code_pix'], r['chg_pix'], r['ctr_pix'] = gated_code(T['pix'], fa, fb, densest(pix_cells(pm, Hp, Wp), Hp, Wp), fchg, args.k)
    # ITC teacher: the crop change is <= the frame change, so the gate cannot pass outside [t_chg, t_frame]: skip ITC there
    Ti = T['itc']
    if fchg < max(Ti['t_chg'], 20) or fchg > Ti['t_frame']:
        r.update(code_itc=STILL, chg_itc=-1, ctr_itc=(np.nan, np.nan))
        return r
    with torch.no_grad():
        _, kept, _, _ = itc_plan(tok, np.stack([fa, fb]), device, args.temp, args.c_d, args.c_w, Wp)
    r['itc_run'] = True
    r['code_itc'], r['chg_itc'], r['ctr_itc'] = gated_code(Ti, fa, fb, densest(kept, Hp, Wp), fchg, args.k)
    return r


def load_tok(args):
    from utils.utils import load_videotokenizer_from_checkpoint
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    tok, _ = load_videotokenizer_from_checkpoint(args.tokenizer, device)
    tok.eval()
    return tok, device


def judge(args):
    # the 200 judge transitions -> codes; must reproduce eval_results/itc_loop/i4/codes_gate_stat_{itc,pix}.json
    tok, device = load_tok(args)
    T = load_teacher(args.teacher)
    X = h5py.File(args.frames, 'r')['frames']
    tr = json.load(open(args.transitions))['transitions']
    out = {w: {} for w in TEACHERS}
    t0 = time.time()
    for t in tr:
        r = label_pair(X[t['frame_a']], X[t['frame_b']], tok, T, device, args)
        for w in TEACHERS:
            out[w][str(t['id'])] = int(r[f'code_{w}'])
    print(f'{len(tr)} judge transitions in {time.time() - t0:.0f}s')
    os.makedirs(args.out_dir, exist_ok=True)
    for w in TEACHERS:
        json.dump(out[w], open(os.path.join(args.out_dir, f'codes_teacher_{w}.json'), 'w'))
        ref = json.load(open(os.path.join(args.i4_dir, f'codes_gate_stat_{w}.json')))
        same = sum(out[w][i] == ref[i] for i in ref)
        print(f'teacher {w}: agreement with i4 gate_stat_{w} {same}/{len(ref)}',
              [(i, ref[i], out[w][i]) for i in ref if out[w][i] != ref[i]][:10])


FIELDS = {'code_itc': -1, 'code_pix': -1, 'scroll': -1, 'fchg': -1, 'chg_itc': -1, 'chg_pix': -1, 'itc_run': -1}


def label(args):
    tok, device = load_tok(args)
    T = load_teacher(args.teacher)
    h = h5py.File(args.train_frames, 'r')
    src = h['source_index'][:]
    gap = args.gap
    N = len(src)
    hi = min(args.hi, N - gap)
    X = h['frames'][args.lo:hi + gap]  # [hi - lo + gap, H, W, C] uint8, one read
    out = {k: np.full(hi - args.lo, v, np.int64) for k, v in FIELDS.items()}
    out['shift'] = np.zeros((hi - args.lo, 2), np.int64)
    out['ctr_itc'] = np.full((hi - args.lo, 2), np.nan, np.float32); out['ctr_pix'] = out['ctr_itc'].copy()
    t0 = time.time()
    for n, t in enumerate(range(args.lo, hi)):
        if src[t + gap] != src[t] + gap:
            continue
        r = label_pair(X[n], X[n + gap], tok, T, device, args)
        for k in FIELDS:
            out[k][n] = int(r[k])
        out['shift'][n] = r['shift']; out['ctr_itc'][n] = r['ctr_itc']; out['ctr_pix'][n] = r['ctr_pix']
        if (n + 1) % 200 == 0:
            print(f'[{args.lo}, {hi}) {n + 1}/{hi - args.lo} ({time.time() - t0:.0f}s, ITC on {int((out["itc_run"] == 1).sum())})',
                  flush=True)
    np.savez_compressed(args.out, lo=args.lo, hi=hi, gap=gap, **out)
    print(f'saved {args.out}: [{args.lo}, {hi}) in {time.time() - t0:.0f}s')


def merge(args):
    N = len(h5py.File(args.train_frames, 'r')['source_index'])
    full = None
    for p in args.parts:
        z = np.load(p)
        if full is None:
            full = {k: np.full((N,) + z[k].shape[1:], -1 if z[k].dtype.kind == 'i' else np.nan, z[k].dtype)
                    for k in z.files if k not in ('lo', 'hi', 'gap')}
        lo, hi = int(z['lo']), int(z['hi'])
        for k in full:
            full[k][lo:hi] = z[k]
    np.savez_compressed(args.out, gap=args.gap, **full)
    lab = full['code_itc'] >= 0
    print(f'saved {args.out}: {lab.sum()} labelled pairs of {N}; scroll {(full["scroll"] == 1).sum()}, ITC run {(full["itc_run"] == 1).sum()}')
    for w in TEACHERS:
        print(f'code_{w} counts (R DR D DL L UL U UR STILL):', np.bincount(full[f'code_{w}'][lab], minlength=9).tolist())
    print('itc/pix agreement', float((full['code_itc'][lab] == full['code_pix'][lab]).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['prep', 'judge', 'label', 'merge'])
    p.add_argument('--lo', type=int, default=0)
    p.add_argument('--hi', type=int, default=200)
    p.add_argument('--out', default='eval_results/itc_loop/i5/pseudo_part.npz')
    p.add_argument('--parts', nargs='+', default=[])
    p.add_argument('--teacher', default='eval_results/itc_loop/i5/teacher.npz')
    p.add_argument('--k', type=int, default=7)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--transitions', default='eval_results/lam_judge/set/transitions.json')
    p.add_argument('--frames', default='data/zelda_test_frames.h5')
    p.add_argument('--train-frames', default='data/zelda_train_frames.h5')
    p.add_argument('--tokenizer', default=DATASETS['zelda'][0])
    p.add_argument('--i3-dir', default='eval_results/itc_loop/i3')
    p.add_argument('--i4-dir', default='eval_results/itc_loop/i4')
    p.add_argument('--out-dir', default='eval_results/itc_loop/i5')
    p.add_argument('--gap', type=int, default=4)
    p.add_argument('--temp', type=float, default=0.1)
    p.add_argument('--c-d', type=float, default=0.1)
    p.add_argument('--c-w', type=float, default=0.5)
    p.add_argument('--pix-thresh', type=int, default=40)
    p.add_argument('--scroll-ratio', type=float, default=0.5)
    args = p.parse_args()
    {'prep': prep, 'judge': judge, 'label': label, 'merge': merge}[args.cmd](args)


if __name__ == '__main__':
    main()
