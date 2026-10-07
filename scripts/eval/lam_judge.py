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
    python scripts/eval/lam_judge.py score --lam como=<como ckpt> --k 8  # continuous LAM: k-means clusters + linear probe
"""

import argparse
import json
import math
import os

import h5py
import numpy as np
import torch

from evaluation.action_metrics import nmi, kmeans, label_metrics
from evaluation.windows import test_windows
from evaluation.zelda_judge import (LABELS, MOVES, H5, SEQ, SKIP, STRIDE, OUT, SET, transition_panel, stack_rows, lam_codes,
                                    probe, judged_metrics, code_grids)



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



def score(args):
    meta = json.load(open(f'{SET}/transitions.json'))
    labels = json.load(open(args.labels))
    wins = test_windows(H5, SEQ - 1, SKIP, STRIDE)
    specs = [(s, 'lam') for s in args.lam] + [(s, 'codes') for s in args.codes]
    for spec, kind in specs:
        name, path = spec.split('=', 1)
        out_dir = f'{OUT}/{name}'
        os.makedirs(out_dir, exist_ok=True)
        z_all = None
        if kind == 'lam':
            codes_all, n_codes, z_all = lam_codes(path, wins, args.device, k=args.k)
        else:  # {"<transition id>": code}: only the judged set, no grids
            cj = json.load(open(path))
            codes_all, n_codes = None, int(max(cj.values())) + 1
        res = {'name': name, 'source': path, 'labels_file': args.labels,
               **judged_metrics(meta, labels, n_codes, codes_all=codes_all, code_of_id=None if codes_all is not None else cj)}
        ids = list(res['per_transition'])
        labs = [labels[i] for i in ids]
        if z_all is not None:  # continuous actions: linear probe on the raw action, independent of the clustering
            trs = {str(tr['id']): tr for tr in meta['transitions']}
            zf = z_all.reshape(-1, z_all.shape[-1])
            pcs = torch.linalg.svd(zf - zf.mean(0), full_matrices=False).Vh[:args.probe_pcs].T  # [A, k], unsupervised, all held-out actions
            zs = (torch.stack([z_all[trs[i]['window'], trs[i]['t']] for i in ids]) - zf.mean(0)) @ pcs
            mv = [j for j, l in enumerate(labs) if l in MOVES]
            pa, pm = probe(zs, labs), probe(zs[mv], [labs[j] for j in mv])
            if args.kmeans_seeds > 1:  # clustering noise: moves / all NMI_adj over k-means seeds 0..n-1 (seed 0 = headline)
                sc = {'moves': [], 'all': []}
                for sd in range(args.kmeans_seeds):
                    cs = torch.cdist(zf, kmeans(zf, n_codes, seed=sd)).argmin(1).reshape(z_all.shape[:-1]).numpy()
                    cc = [int(cs[trs[i]['window'], trs[i]['t']]) for i in ids]
                    sc['all'].append(label_metrics(cc, labs, n_codes, LABELS)['nmi_adj'])
                    sc['moves'].append(label_metrics([c for c, l in zip(cc, labs) if l in MOVES], [l for l in labs if l in MOVES], n_codes, MOVES)['nmi_adj'])
                res['kmeans_seeds'] = {f'{k}_nmi_adj_{f}': round(float(fn(v)), 4) for k, v in sc.items() for f, fn in (('mean', np.mean), ('sd', np.std))}
                res['kmeans_seeds']['n'] = args.kmeans_seeds
            res['probe'] = {'pcs': args.probe_pcs, 'all_acc': pa['acc'], 'all_majority': pa['majority'], 'moves_acc': pm['acc'], 'moves_majority': pm['majority']}
        if codes_all is not None:  # appearance leakage: how much the code tells which held-out block (scene) it is from
            blk = np.repeat([b for b, _ in wins], codes_all.shape[1])
            ct = np.zeros((n_codes, int(blk.max()) + 1))
            np.add.at(ct, (codes_all.reshape(-1), blk), 1)
            res['nmi_code_block'] = round(nmi(ct), 4)
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
              f"| entropy {res.get('entropy_nats')} | probe {res.get('probe')} | seeds {res.get('kmeans_seeds')} -> {out_dir}/score.json")


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
    s.add_argument('--kmeans-seeds', type=int, default=1, help='continuous LAMs: also report NMI_adj mean/sd over this many k-means seeds')
    s.add_argument('--probe-pcs', type=int, default=16, help='continuous LAMs: principal components the linear probe sees')
    s.add_argument('--k', type=int, default=0, help='continuous LAMs: number of k-means clusters (default: its n_actions)')
    s.add_argument('--device', default='cpu', help='cpu: ~40 s per LAM; mps hung in Metal once (2026-10-02)')
    a = p.parse_args()
    {'build': build, 'consensus': consensus, 'score': score}[a.cmd](a)


if __name__ == '__main__':
    main()
