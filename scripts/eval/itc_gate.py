"""Movement gate + ITC-crop facing (STA-35 itc_loop iteration 4).

i3 (itc_facing.py) reads the direction of a still-camera walk from Link's facing in a crop at the ITC-kept cluster, but has
no "did Link move?" test, so it gives a direction to most still-screen non-moves. Here the direction is unchanged and a
gate decides whether to emit it (else STILL). Gate thresholds come from train pairs only, never from judge labels:
walking = i3 mined scrolling train pairs (scroll-compensated), still-screen = train pairs mined here (`mine`).

  mine:   python scripts/eval/itc_gate.py mine --n-keep 250   (ITC on still-screen gap-4 train pairs, cached)
  codes:  python scripts/eval/itc_gate.py codes               (writes eval_results/itc_loop/i4/codes_<arm>.json)
  then:   python scripts/eval/lam_judge.py score --codes i4_gate_itc=eval_results/itc_loop/i4/codes_gate_itc.json

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
from itc_actions import STILL, block_match, global_shift, is_scroll, itc_plan  # noqa: E402
from itc_correspondence import DATASETS  # noqa: E402
from itc_facing import CROP, PATCH, bank, crop, densest, feat, judge_loc, knn, pix_cells  # noqa: E402
from itc_shift import translate  # noqa: E402

# learned gate: shape of the change only (no magnitude, which the mined labels are defined from)
FEATS = ['bbox_frac', 'shift_mag', 'resid', 'log_frame', 'share']


# ----------------------------------------------------------------------------- crop-pair features
def pair_feats(ca, cb, frame_chg, thresh=40, max_s=4):
    # ca, cb [CROP, CROP, C] uint8 crops of frames t and t+1 at the same place -> dict of gate features
    d = np.abs(ca.astype(np.int16) - cb.astype(np.int16)).max(-1)  # [CROP, CROP]
    m = d > thresh
    chg = int(m.sum())
    ys, xs = np.nonzero(m)
    bbox = 0.0 if chg == 0 else (np.ptp(ys) + 1) * (np.ptp(xs) + 1) / CROP ** 2
    ga, gb = ca.astype(np.float32).mean(-1) / 255.0, cb.astype(np.float32).mean(-1) / 255.0
    c = ga[max_s:CROP - max_s, max_s:CROP - max_s]  # [CROP-2s, CROP-2s] centre of t, searched in t+1
    errs = {(dy, dx): float(np.abs(c - gb[max_s + dy:CROP - max_s + dy, max_s + dx:CROP - max_s + dx]).mean())
            for dy in range(-max_s, max_s + 1) for dx in range(-max_s, max_s + 1)}
    best = min(errs, key=errs.get)
    return dict(chg=chg, log_chg=np.log1p(chg), mad=float(d.mean()) / 255.0, bbox_frac=bbox,
                shift_mag=float(np.hypot(*best)), resid=errs[best] / (errs[(0, 0)] + 1e-6),
                log_frame=np.log1p(frame_chg), share=chg / max(frame_chg, 1), frame=int(frame_chg))


# ----------------------------------------------------------------------------- mining still-screen train pairs
def mine(args):
    torch.set_num_threads(4)
    from utils.utils import load_videotokenizer_from_checkpoint
    h = h5py.File(args.train_frames, 'r')
    X, src = h['frames'], h['source_index'][:]
    gap = args.gap
    cand = np.nonzero(src[gap:] == src[:-gap] + gap)[0]
    order = np.random.default_rng(args.seed).permutation(cand)
    device = torch.device('cpu')
    tok, _ = load_videotokenizer_from_checkpoint(args.tokenizer, device)
    tok.eval()
    H = X.shape[1]; Hp = Wp = H // PATCH
    out = {k: [] for k in ['frame', 'kept_n', 'itc_c', 'itc_n', 'pix_c', 'pix_n', 'frame_chg', 'block_shift',
                           'crop_itc_a', 'crop_itc_b', 'crop_pix_a', 'crop_pix_b']}
    t0, seen = time.time(), 0
    for i in order:
        if len(out['frame']) >= args.n_keep:
            break
        seen += 1
        fa, fb = X[i], X[i + gap]
        g = np.stack([fa, fb]).astype(np.float32).mean(-1) / 255.0  # [2, H, W]
        sh, eb, e0 = global_shift(g[0], g[1])
        if is_scroll(sh, eb, e0, args.scroll_ratio):
            continue
        with torch.no_grad():
            _, kept, _, _ = itc_plan(tok, np.stack([fa, fb]), device, args.temp, args.c_d, args.c_w, Wp)
        pm = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(-1) > args.pix_thresh  # [H, W], as in i1 features
        ci, cp = densest(kept, Hp, Wp), densest(pix_cells(pm, Hp, Wp), Hp, Wp)
        nan2 = (np.nan, np.nan)
        out['frame'].append(int(i)); out['kept_n'].append(int(kept.sum())); out['frame_chg'].append(int(pm.sum()))
        out['itc_c'].append(nan2 if ci is None else ci[1]); out['itc_n'].append(0 if ci is None else ci[0])
        out['pix_c'].append(nan2 if cp is None else cp[1]); out['pix_n'].append(0 if cp is None else cp[0])
        out['block_shift'].append(block_match(g[0], g[1], pm)[0])
        for which, c in [('itc', ci), ('pix', cp)]:
            cy, cx = (H / 2, H / 2) if c is None else c[1]
            out[f'crop_{which}_a'].append(crop(fa, cy, cx)); out[f'crop_{which}_b'].append(crop(fb, cy, cx))
        if len(out['frame']) % 25 == 0:
            print(f'{len(out["frame"])}/{args.n_keep} still-screen ({seen} drawn, {time.time() - t0:.0f}s)', flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, 'mined_still.npz'), **{k: np.array(v) for k, v in out.items()})
    print('saved', len(out['frame']), 'still-screen train pairs of', seen, 'drawn')


# ----------------------------------------------------------------------------- train-side gate data
def walking_feats(args, which):
    # i3 mined scrolling train pairs: frame t translated with the camera, crops of t and t+1 at the i3 cluster centre
    M = np.load(os.path.join(args.i3_dir, 'mined.npz'))
    X = h5py.File(args.train_frames, 'r')['frames']
    rows = []
    for n, i in enumerate(M['frame']):
        dy, dx = M['shift'][n]
        fs, fb = translate(X[i], dy, dx), X[i + args.gap]
        cy, cx = M[f'{which}_c'][n]
        rows.append(pair_feats(crop(fs, cy, cx), crop(fb, cy, cx), int(M['chg_frame'][n])))
    return rows


def still_feats(args, which):
    S = np.load(os.path.join(args.out_dir, 'mined_still.npz'))
    return [pair_feats(S[f'crop_{which}_a'][n], S[f'crop_{which}_b'][n], int(S['frame_chg'][n]))
            for n in range(len(S['frame']))], S


def otsu(v, bins=64):
    # 1-D Otsu threshold of v (maximise between-class variance)
    h, e = np.histogram(v, bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def calibrate(args, which):
    # label-free gate thresholds and the learned gate, from train pairs only
    W, (S, Sraw) = walking_feats(args, which), still_feats(args, which)
    wl, sl = np.array([r['log_chg'] for r in W]), np.array([r['log_chg'] for r in S])
    pooled = np.concatenate([wl, sl])
    t1 = otsu(pooled)  # level 1: no change vs some change (both pools have a zero mode)
    t_chg = otsu(pooled[pooled >= t1])  # level 2: within the changed crops, walking-sized vs small (ambient) change
    wf = np.array([r['log_frame'] for r in W])
    t_frame = float(np.percentile(wf, 99.5)) + np.log(args.frame_margin)  # walking frame change never exceeds this
    cal = dict(t1_px=float(np.expm1(t1)), t_chg_px=float(np.expm1(t_chg)), t_frame_px=float(np.expm1(t_frame)),
               walk_chg_pct=np.percentile(np.expm1(wl), [10, 25, 50, 75]).round(0).tolist(),
               still_chg_pct=np.percentile(np.expm1(sl), [10, 25, 50, 75, 90]).round(0).tolist(),
               walk_above=float((wl >= t_chg).mean()), still_above=float((sl >= t_chg).mean()))
    # learned gate: walking = mined scrolling pairs with walking-sized crop change; standing = still-screen train pairs
    # whose crop is unchanged or only has small (ambient) change, or whose change is screen-wide
    pos = [r for r in W if r['log_chg'] >= t_chg and r['log_frame'] <= t_frame]
    neg = [r for r in S if r['log_chg'] < t_chg or r['log_frame'] > t_frame]
    Xtr = np.array([[r[k] for k in FEATS] for r in pos + neg]); ytr = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    w = fit_logreg((Xtr - mu) / sd, ytr)
    cal.update(n_pos=len(pos), n_neg=len(neg), logreg_w=w.round(3).tolist())
    # how the learned gate splits the (unlabelled) still-screen train pairs
    ps = np.array([prob(w, (np.array([r[k] for k in FEATS]) - mu) / sd) for r in S])
    cal['still_learned_fire'] = float((ps > 0.5).mean())
    return cal, (w, mu, sd)


def fit_logreg(X, y, l2=1e-2, iters=2000, lr=0.5):
    Xb = np.c_[X, np.ones(len(X))]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        w -= lr * (Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
    return w


def prob(w, x):
    return float(1 / (1 + np.exp(-(np.r_[x, 1] @ w))))


# ----------------------------------------------------------------------------- judge codes
def codes(args):
    rows = judge_loc(SimpleNamespace(i1_dir=args.i1_dir, transitions=args.transitions, frames=args.frames,
                                     scroll_ratio=args.scroll_ratio))
    bargs = SimpleNamespace(out_dir=args.i3_dir, mirror=1, feat='pix')
    out, cal_all = {}, {}
    for which in ['itc', 'pix']:
        cal, lr = calibrate(args, which)
        cal_all[which] = cal
        print(which, json.dumps(cal), flush=True)
        Fb, yb = bank(bargs, which)
        arms = {f'gate_none_{which}': {}, f'gate_pixblock_{which}': {}, f'gate_stat_{which}': {}, f'gate_learned_{which}': {}}
        for i, r in rows.items():
            if r['scroll']:
                for a in arms:
                    arms[a][i] = r['scode']
                continue
            c = r[which]
            if c is None:
                for a in arms:
                    arms[a][i] = STILL
                continue
            cy, cx = c[1]
            f = pair_feats(crop(r['fa'], cy, cx), crop(r['fb'], cy, cx), int(r['pm'].sum()))
            d = knn(Fb, yb, feat(crop(r['fb'], cy, cx), 'pix'), args.k) if f['chg'] >= args.min_pix else STILL
            g = {'none': True,
                 'pixblock': np.hypot(*r['block']) >= args.min_block,
                 'stat': f['chg'] >= cal['t_chg_px'] and f['frame'] <= cal['t_frame_px'],
                 'learned': prob(lr[0], (np.array([f[k] for k in FEATS]) - lr[1]) / lr[2]) > 0.5}
            for gname, ok in g.items():
                arms[f'gate_{gname}_{which}'][i] = d if ok else STILL
        out.update(arms)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(cal_all, open(os.path.join(args.out_dir, 'calibration.json'), 'w'), indent=1)
    labels = json.load(open(args.labels))
    so = json.load(open(os.path.join(args.i1_dir, 'codes_scroll_only.json')))
    for a, cj in out.items():
        json.dump(cj, open(os.path.join(args.out_dir, f'codes_{a}.json'), 'w'))
        # diagnostics computed exactly like the orchestrator (judge labels used only for reporting)
        still_cam = [i for i in cj if labels.get(i) in DIR and so[i] == STILL]
        exact = sum(cj[i] == DIR[labels[i]] for i in still_cam)
        non = [i for i in cj if labels.get(i) in ('STILL', 'NONCONTROL', 'ATTACK') and so[i] == STILL]
        fires = {lab: sum(cj[i] != STILL for i in non if labels[i] == lab) for lab in ('STILL', 'NONCONTROL', 'ATTACK')}
        print(f'{a}: still-cam exact {exact}/{len(still_cam)}  fires {sum(fires.values())}/{len(non)} {fires}')


DIR = {'R': 0, 'DR': 1, 'D': 2, 'DL': 3, 'L': 4, 'UL': 5, 'U': 6, 'UR': 7}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['mine', 'codes'])
    p.add_argument('--k', type=int, default=7)
    p.add_argument('--min-pix', type=int, default=20, help='i3 rule: min changed px in the crop for a facing code')
    p.add_argument('--min-block', type=float, default=2.0, help='gate_pixblock: min |pix_block shift| (px)')
    p.add_argument('--stand-px', type=int, default=5, help='learned gate negatives: crop change below this (px)')
    p.add_argument('--frame-margin', type=float, default=2.0, help='frame-change cap = margin x walking p99.5')
    p.add_argument('--transitions', default='eval_results/lam_judge/set/transitions.json')
    p.add_argument('--labels', default='eval_results/lam_judge/set/labels.json')
    p.add_argument('--frames', default='data/zelda_test_frames.h5')
    p.add_argument('--train-frames', default='data/zelda_train_frames.h5')
    p.add_argument('--tokenizer', default=DATASETS['zelda'][0])
    p.add_argument('--i1-dir', default='eval_results/itc_loop/i1')
    p.add_argument('--i3-dir', default='eval_results/itc_loop/i3')
    p.add_argument('--out-dir', default='eval_results/itc_loop/i4')
    p.add_argument('--n-keep', type=int, default=250)
    p.add_argument('--gap', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--temp', type=float, default=0.1)
    p.add_argument('--c-d', type=float, default=0.1)
    p.add_argument('--c-w', type=float, default=0.5)
    p.add_argument('--pix-thresh', type=int, default=40)
    p.add_argument('--scroll-ratio', type=float, default=0.5)
    args = p.parse_args()
    {'mine': mine, 'codes': codes}[args.cmd](args)


if __name__ == '__main__':
    main()
