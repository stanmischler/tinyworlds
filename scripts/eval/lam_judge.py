"""LLM-as-judge evaluation of a latent action model on held-out Zelda: do transitions that share a code share an action?

Protocol (fixed, deterministic, LAM-independent labels so every LAM is scored on the same ground truth):
  - set:     `n` transitions (frame t -> t+1 of a 4-frame LAM window, frame_skip 4) drawn (seed 0) from all
             held-out windows of data/zelda_test_frames.h5 (eval_next_frame.test_windows, stride 8)
  - sheets:  `build` writes them as numbered sheets (4 transitions per PNG: frame t | frame t+1 | |change| x3, 2x
             nearest upscale) for blind labelling by LLM judges (no LAM involved); each judge writes a JSON
             {"<id>": "<LABEL>"} with LABEL in LABELS; `consensus` merges two judges (agreement only) into labels.json
  - score:   `score` encodes every window with the LAM, reads the code of each judged transition, and reports against
             the consensus labels: NMI(code, label), chance-corrected NMI (minus the mean NMI of 200 code
             permutations), purity (share of transitions whose label is their code's majority label), the same on the
             movement-only subset (the 8 direction labels),
             and the code x label table. Also writes per-code grids (6 random held-out transitions per code, any
             window, seed 0) for the qualitative per-iteration judge.

Usage (repo root, PYTHONPATH=$PWD):
    python scripts/eval/lam_judge.py build                      # once: eval_results/lam_judge/set/{transitions.json,sheet_*.png}
    python scripts/eval/lam_judge.py consensus labels_A.json labels_B.json
    python scripts/eval/lam_judge.py score --lam Z4=<ckpt dir>  # -> eval_results/lam_judge/<label>/{score.json,code_*.png}
    python scripts/eval/lam_judge.py score --codes my=codes.json  # a LAM evaluated elsewhere: {"<transition id>": code}
"""

import argparse
import json
import math
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_next_frame import test_windows, load_window_batch, to_model_range  # noqa: E402
from eval_lam import nmi, kmeans  # noqa: E402

LABELS = ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR', 'STILL', 'ATTACK', 'NONCONTROL', 'UNSURE']
MOVES = ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR']
H5 = 'data/zelda_test_frames.h5'
SEQ, SKIP, STRIDE = 4, 4, 8  # LAM window: 4 frames, 4 stored frames apart; window starts every 8 stored frames
OUT = 'eval_results/lam_judge'
SET = f'{OUT}/set'


def font_draw(img):
    return ImageDraw.Draw(img)


def transition_panel(fr_a, fr_b, scale=2, tag=''):
    # fr_a, fr_b: uint8 [H, W, C] -> PIL image: frame t | frame t+1 | amplified |change| (magma-free grayscale-hot)
    d = np.clip(np.abs(fr_b.astype(np.int16) - fr_a.astype(np.int16)).max(-1) * 3, 0, 255).astype(np.uint8)  # [H, W]
    heat = np.stack([d, (d.astype(np.float32) * 0.6).astype(np.uint8), np.zeros_like(d)], -1)  # orange on black
    H, W = d.shape
    gap, top = 6, 16
    img = Image.new('RGB', (3 * W * scale + 2 * gap, H * scale + top), 'white')
    for k, a in enumerate([fr_a, fr_b, heat]):
        img.paste(Image.fromarray(a).resize((W * scale, H * scale), Image.NEAREST), (k * (W * scale + gap), top))
    font_draw(img).text((2, 2), tag, fill='black')
    return img


def stack_rows(rows, title):
    W = max(r.width for r in rows)
    img = Image.new('RGB', (W, sum(r.height + 8 for r in rows) + 20), 'white')
    font_draw(img).text((4, 4), title, fill='black')
    y = 20
    for r in rows:
        img.paste(r, (0, y))
        y += r.height + 8
    return img


def build(args):
    os.makedirs(SET, exist_ok=True)
    wins = test_windows(H5, SEQ - 1, SKIP, STRIDE)
    pairs = [(w, t) for w in range(len(wins)) for t in range(SEQ - 1)]
    rng = np.random.default_rng(0)
    pick = [pairs[i] for i in rng.choice(len(pairs), args.n, replace=False)]
    trans = []
    with h5py.File(H5, 'r') as h5:
        fr = h5['frames']
        rows = []
        for tid, (w, t) in enumerate(pick):
            blk, s = wins[w]
            a, b = fr[s + t * SKIP], fr[s + (t + 1) * SKIP]
            trans.append({'id': tid, 'window': w, 't': t, 'block': blk, 'frame_a': s + t * SKIP, 'frame_b': s + (t + 1) * SKIP})
            rows.append(transition_panel(a, b, tag=f'#{tid}   (frame t | frame t+1 | |change| x3)'))
        for k in range(0, len(rows), 4):
            stack_rows(rows[k:k + 4], f'sheet {k // 4:02d}: transitions #{k}-#{min(k + 3, len(rows) - 1)}').save(f'{SET}/sheet_{k // 4:02d}.png')
    json.dump({'h5': H5, 'seq_len': SEQ, 'frame_skip': SKIP, 'sample_stride': STRIDE, 'n_windows': len(wins),
               'labels': LABELS, 'transitions': trans}, open(f'{SET}/transitions.json', 'w'), indent=1)
    print(f'{len(trans)} transitions, {math.ceil(len(trans) / 4)} sheets in {SET}')


def consensus(args):
    js = [json.load(open(p)) for p in args.files]
    ids = sorted(set.intersection(*[set(j) for j in js]), key=int)
    out, agree = {}, 0
    for i in ids:
        ls = [j[i] for j in js]
        if all(l == ls[0] for l in ls):
            agree += 1
            if ls[0] != 'UNSURE':
                out[i] = ls[0]
    json.dump(out, open(f'{SET}/labels.json', 'w'), indent=1)
    counts = {l: sum(v == l for v in out.values()) for l in LABELS}
    print(f'agreement {agree}/{len(ids)} = {agree / max(len(ids), 1):.2f}; {len(out)} consensus labels; {counts}')


def lam_codes(ckpt, wins, device, batch=32):
    from utils.utils import load_latent_actions_from_checkpoint
    lam, _ = load_latent_actions_from_checkpoint(ckpt, device)
    lam.eval()
    return codes_from_lam(lam, wins, device, batch)


def codes_from_lam(lam, wins, device, batch=32):
    # lam in eval mode -> codes [N, T-1] for every held-out window, n_codes
    q, A = lam.quantizer, lam.action_dim
    with h5py.File(H5, 'r') as h5, torch.no_grad():
        zq = torch.cat([lam.encode(to_model_range(load_window_batch(h5['frames'], wins[i:i + batch], SEQ - 1, SKIP), device))
                        for i in range(0, len(wins), batch)])  # [N, T-1, A]
    if getattr(lam, 'continuous_actions', False):
        c = kmeans(zq.reshape(-1, A).float(), q.codebook_size)
        return torch.cdist(zq.reshape(-1, A).float(), c).argmin(1).reshape(zq.shape[:-1]).cpu().numpy(), q.codebook_size  # [N, T-1]
    return q.get_indices_from_latents(zq).cpu().numpy(), q.codebook_size


def metrics(codes, labels, n_codes, classes):
    tab = np.zeros((n_codes, len(classes)))
    for c, l in zip(codes, labels):
        tab[c, classes.index(l)] += 1
    rng = np.random.default_rng(0)
    perm = [nmi(_table(rng.permutation(codes), labels, n_codes, classes)) for _ in range(200)]
    v = nmi(tab)
    return {'n': int(len(codes)), 'nmi': round(v, 4), 'nmi_chance': round(float(np.mean(perm)), 4),
            'nmi_adj': round((v - np.mean(perm)) / (1 - np.mean(perm)), 4),
            'purity': round(float(tab.max(1).sum() / max(tab.sum(), 1)), 4),
            'majority_baseline': round(float(tab.sum(0).max() / max(tab.sum(), 1)), 4),
            'table': {f'code {k}': {cl: int(tab[k, j]) for j, cl in enumerate(classes) if tab[k, j]} for k in range(n_codes) if tab[k].sum()}}


def _table(codes, labels, n_codes, classes):
    tab = np.zeros((n_codes, len(classes)))
    for c, l in zip(codes, labels):
        tab[c, classes.index(l)] += 1
    return tab


def judged_metrics(meta, labels, n_codes, codes_all=None, code_of_id=None):
    # codes_all [N windows, T-1] (or code_of_id {"<transition id>": code}) -> metrics on the judged set
    ids, codes, labs = [], [], []
    for tr in meta['transitions']:
        i = str(tr['id'])
        if i not in labels:
            continue
        ids.append(i)
        codes.append(int(codes_all[tr['window'], tr['t']]) if codes_all is not None else int(code_of_id[i]))
        labs.append(labels[i])
    return {'n_codes': n_codes, 'all': metrics(codes, labs, n_codes, LABELS),
            'moves_only': metrics([c for c, l in zip(codes, labs) if l in MOVES], [l for l in labs if l in MOVES], n_codes, MOVES),
            'per_transition': dict(zip(ids, codes))}


def code_grids(name, codes_all, wins, n_codes, out_dir, per_code=6):
    # codes_all: [N, T-1] -> one PNG per code with up to `per_code` random held-out transitions mapped to it
    rng = np.random.default_rng(0)
    flat = [(w, t, int(codes_all[w, t])) for w in range(len(wins)) for t in range(codes_all.shape[1])]
    usage = np.bincount([c for _, _, c in flat], minlength=n_codes) / len(flat)
    paths = []
    with h5py.File(H5, 'r') as h5:
        fr = h5['frames']
        for k in range(n_codes):
            cand = [(w, t) for w, t, c in flat if c == k]
            if not cand:
                continue
            sel = [cand[i] for i in rng.choice(len(cand), min(per_code, len(cand)), replace=False)]
            rows = []
            for j, (w, t) in enumerate(sel):
                blk, s = wins[w]
                rows.append(transition_panel(fr[s + t * SKIP], fr[s + (t + 1) * SKIP], tag=f'code {k} example {j} (block {blk})'))
            p = f'{out_dir}/code_{k}.png'
            stack_rows(rows, f'{name}: code {k}, {usage[k]:.1%} of held-out transitions (n={len(cand)}); frame t | frame t+1 | |change| x3').save(p)
            paths.append(p)
    return usage, paths


def score(args):
    meta = json.load(open(f'{SET}/transitions.json'))
    labels = json.load(open(args.labels))
    wins = test_windows(H5, SEQ - 1, SKIP, STRIDE)
    specs = [(s, 'lam') for s in args.lam] + [(s, 'codes') for s in args.codes]
    for spec, kind in specs:
        name, path = spec.split('=', 1)
        out_dir = f'{OUT}/{name}'
        os.makedirs(out_dir, exist_ok=True)
        if kind == 'lam':
            codes_all, n_codes = lam_codes(path, wins, args.device)
        else:  # {"<transition id>": code}: only the judged set, no grids
            cj = json.load(open(path))
            codes_all, n_codes = None, int(max(cj.values())) + 1
        res = {'name': name, 'source': path, 'labels_file': args.labels,
               **judged_metrics(meta, labels, n_codes, codes_all=codes_all, code_of_id=None if codes_all is not None else cj)}
        if codes_all is not None:
            usage, paths = code_grids(name, codes_all, wins, n_codes, out_dir)
            p = usage[usage > 0]
            res['usage'] = [round(float(u), 4) for u in usage]
            res['entropy_nats'] = round(float(-(p * np.log(p)).sum()), 4)
            res['grids'] = paths
        json.dump(res, open(f'{out_dir}/score.json', 'w'), indent=1)
        a, m = res['all'], res['moves_only']
        print(f"{name}: all n={a['n']} NMI {a['nmi']} (adj {a['nmi_adj']}) purity {a['purity']} (majority {a['majority_baseline']}) | "
              f"moves n={m['n']} NMI {m['nmi']} (adj {m['nmi_adj']}) purity {m['purity']} (majority {m['majority_baseline']}) "
              f"| entropy {res.get('entropy_nats')} -> {out_dir}/score.json")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--n', type=int, default=200)
    c = sub.add_parser('consensus')
    c.add_argument('files', nargs='+')
    s = sub.add_parser('score')
    s.add_argument('--lam', action='append', default=[], help='name=<latent_actions checkpoint dir>; repeatable')
    s.add_argument('--codes', action='append', default=[], help='name=<codes.json {transition id: code}>; repeatable')
    s.add_argument('--labels', default=f'{SET}/labels.json')
    s.add_argument('--device', default='cpu', help='cpu: ~40 s per LAM; mps hung in Metal once (2026-10-02)')
    a = p.parse_args()
    {'build': build, 'consensus': consensus, 'score': score}[a.cmd](a)


if __name__ == '__main__':
    main()
