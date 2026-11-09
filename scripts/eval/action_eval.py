"""Movement-vs-camera action eval on Zelda: see evaluation/action_eval.py for the protocol.

Usage (repo root, PYTHONPATH=$PWD):
    python scripts/eval/action_eval.py sample --link-positions positions.npz    # set/pairs.json + 2 judges' sheets
    python scripts/eval/action_eval.py consensus labels_a.json labels_b.json    # set/labels.json
    python scripts/eval/action_eval.py score --baseline random --baseline constant --baseline shift --baseline world
    python scripts/eval/action_eval.py score --lam como=<como checkpoint dir>   # vectors saved, later re-scored without a model
    python scripts/eval/action_eval.py score --vectors mine=vectors.npy         # [N, A], one row per entry of set/pairs.json
    python scripts/eval/action_eval.py score --vectors mine=vectors.npy --own-shift mine=mine_shift.npy     # scored directly, no read-out; (dy, dx) of the scene: t+1[y, x] ~ t[y - dy, x - dx]
"""

import argparse
import json
import os
import shutil

import h5py
import numpy as np
import torch

from evaluation.action_eval import (N_CODES, block_interval, camera_groups, codes_of, consensus as judge_consensus, judge_panel,
                                    judge_sheet, link_scores, linear_shift_readout, shift_scores)
from evaluation.motion import best_shift
from evaluation.windows import load_history_batch, test_windows, to_model_range

H5, SKIP, SEQ = 'data/zelda_test_frames.h5', 4, 4
PER_SHEET, PER_PART, N_PLANTED, CONF = 4, 30, 20, 0.5


def sample(args):
    os.makedirs(f'{args.root}/set', exist_ok=True)
    shutil.copy(args.link_positions, f'{args.root}/set/link_positions.npz')
    wins = test_windows(H5, SEQ - 1, SKIP, SKIP)  # one judged pair (last transition) every SKIP frames
    with h5py.File(H5) as f:
        frames = f['frames'][:]
    rows = [dict(id=k, block=int(b), start=int(s), a=int(s + (SEQ - 2) * SKIP), b=int(s + (SEQ - 1) * SKIP)) for k, (b, s) in enumerate(wins)]
    for c in range(0, len(rows), args.batch):
        part = rows[c:c + args.batch]
        a, b = (torch.from_numpy(frames[[r[k] for r in part]]).to(args.device) for k in 'ab')
        for r, s in zip(part, best_shift(a, b).tolist()):
            r['shift'] = s
    json.dump(rows, open(f'{args.root}/set/pairs.json', 'w'))
    write_sheets(args, rows, frames)


def write_sheets(args, rows, frames):
    z = np.load(f'{args.root}/set/link_positions.npz')
    rng = np.random.default_rng(0)
    planted = []  # deliberately wrong alignments: check that judges flag a bad alignment, never scored
    fake_ids = rng.choice(np.arange(len(rows), 2 * len(rows)), N_PLANTED, replace=False)  # ids not in the real 0..N-1 range, so a judge could spot them: read the catch rate as optimistic
    for k, r in enumerate(rng.choice(rows, N_PLANTED, replace=False)):
        wrong = [int(v) for v in np.array(r['shift']) + rng.choice([-1, 1], 2) * rng.integers(8, 20, 2)]
        planted.append({**r, 'id': int(fake_ids[k]), 'shift': wrong})
    json.dump(planted, open(f'{args.root}/set/planted.json', 'w'))
    for judge in 'ab':
        out = f'{args.root}/set/judge/sheets_{judge}'
        os.makedirs(out, exist_ok=True)
        order = rng.permutation(len(rows) + len(planted))
        items = [(rows + planted)[i] for i in order]
        index = []
        for s in range(0, len(items), PER_SHEET):
            part = items[s:s + PER_SHEET]
            judge_sheet([judge_panel(frames[r['a']], frames[r['b']], r['shift'], z['yx'][r['a']], z['yx'][r['b']], f"#{r['id']}") for r in part],
                        f'{out}/sheet_{s // PER_SHEET:03d}.png')
            index.append([r['id'] for r in part])
        json.dump(index, open(f'{out}/index.json', 'w'))
        print(f'{judge}: {len(index)} sheets, {-(-len(index) // PER_PART)} parts of {PER_PART}')


def consensus(args):
    a, b = (dict(p for part in f.split(',') for p in json.load(open(part)).items()) for f in args.files)
    planted = {str(r['id']) for r in json.load(open(f'{args.root}/set/planted.json'))} if os.path.exists(f'{args.root}/set/planted.json') else set()
    caught = sum(not (a[i]['alignment_ok'] and b[i]['alignment_ok']) for i in planted if i in a and i in b)
    real_a, real_b = ({i: v for i, v in j.items() if i not in planted} for j in (a, b))
    kept, unclear, no_shift, stats = judge_consensus(real_a, real_b)
    stats['planted_bad_alignments_flagged'] = f'{caught}/{len(planted)}'
    json.dump({'kept': kept, 'unclear': unclear, 'no_shift': no_shift, 'stats': stats}, open(f'{args.root}/set/labels.json', 'w'), indent=1)
    print(stats)


def world_motion(rows, shift, root):
    # measured move of Link in the world: his position change minus the shift; zero where he is not found
    z = np.load(f'{root}/set/link_positions.npz')
    a, b = [r['a'] for r in rows], [r['b'] for r in rows]
    found = (z['conf'][a] > CONF) & (z['conf'][b] > CONF)
    return np.where(found[:, None], z['yx'][b] - z['yx'][a] - shift, 0.0)


def baseline_vectors(kind, rows, shift, root):
    if kind == 'random':
        return np.random.default_rng(0).normal(size=(len(rows), 128))
    if kind == 'constant':  # a vector that says nothing: its shift read-out is the mean shift
        return np.zeros((len(rows), 1))
    if kind == 'shift':
        return shift.astype(np.float64)
    assert kind == 'world', kind
    return world_motion(rows, shift, root)


@torch.no_grad()
def lam_vectors(ckpt, rows, device, batch):
    # action vector of the judged transition: last of the model's window, the 2 earlier frames being its context
    from utils.utils import load_latent_actions_from_checkpoint
    lam, _ = load_latent_actions_from_checkpoint(ckpt, device)
    lam.eval()
    wins, history, out = [(r['block'], r['start']) for r in rows], getattr(lam, 'history', 0), []
    with h5py.File(H5) as f:
        for i in range(0, len(wins), batch):
            x = to_model_range(load_history_batch(f['frames'], wins[i:i + batch], SEQ - 1, SKIP, history, H5), device)
            out.append(lam.encode(x)[:, -1].float().cpu())  # [B, A]
            print(f'{i + batch}/{len(wins)}', flush=True)
    return torch.cat(out).numpy()


def score_one(name, vectors, own_shift, rows, labels, no_shift, args):
    shift = np.array([r['shift'] for r in rows], dtype=np.float64)
    group, blocks = camera_groups(shift), np.array([r['block'] for r in rows])
    codes = codes_of(torch.from_numpy(vectors))
    res = {'name': name, 'n_codes': N_CODES, 'link': link_scores(codes, labels, group, blocks, args.min_pairs)}
    res['link']['block_95'] = block_interval(codes, labels, group, blocks, args.min_pairs, args.n_boot)
    res['link']['headline_kmeans_seeds'] = [link_scores(codes_of(torch.from_numpy(vectors), seed=s), labels, group, blocks, args.min_pairs)['headline'] for s in range(args.kmeans_seeds)]
    valid = np.array([str(r['id']) not in no_shift for r in rows])
    pred = own_shift if own_shift is not None else linear_shift_readout(vectors, shift, blocks)
    res['shift'] = shift_scores(pred, shift, group, valid)
    res['shift']['source'] = 'own' if own_shift is not None else 'linear read-out'
    os.makedirs(f'{args.root}/{name}', exist_ok=True)
    json.dump(res, open(f'{args.root}/{name}/score.json', 'w'), indent=1)
    l, s = res['link'], res['shift']
    print(f"{name}: Link still {l['still'].get('nmi_adj')} (n={l['still']['n']}) moving {l['moving'].get('nmi_adj')} (n={l['moving']['n']}) "
          f"headline {l['headline']} block 95% {l['block_95'].get('headline')} | shift still {s['still']['median']} moving {s['moving']['median']}")


def score(args):
    rows = json.load(open(f'{args.root}/set/pairs.json'))
    j = json.load(open(f'{args.root}/set/labels.json'))
    labels = np.array([j['kept'].get(str(r['id']), '') for r in rows], dtype=object)
    no_shift = set(j['no_shift'])
    shift = np.array([r['shift'] for r in rows], dtype=np.float64)
    own = dict(s.split('=', 1) for s in args.own_shift)
    for b in args.baseline:
        score_one(f'baseline_{b}', baseline_vectors(b, rows, shift, args.root), shift if b == 'shift' else None, rows, labels, no_shift, args)
    for spec in args.lam:
        name, ckpt = spec.split('=', 1)
        vectors = lam_vectors(ckpt, rows, args.device, args.batch)
        os.makedirs(f'{args.root}/{name}', exist_ok=True)
        np.save(f'{args.root}/{name}/vectors.npy', vectors)
        score_one(name, vectors, None, rows, labels, no_shift, args)
    for spec in args.vectors:
        name, path = spec.split('=', 1)
        score_one(name, np.load(path), np.load(own[name]) if name in own else None, rows, labels, no_shift, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='eval_results/action_eval/zelda')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--batch', type=int, default=32)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('sample')
    p.add_argument('--link-positions', required=True, help='npz with yx [frames, 2] and conf [frames] of Link in the test h5')
    p = sub.add_parser('consensus')
    p.add_argument('files', nargs=2, help='one labels json per judge; a judge split in parts: a0.json,a1.json')
    p = sub.add_parser('score')
    p.add_argument('--baseline', action='append', default=[], choices=['random', 'constant', 'shift', 'world'])
    p.add_argument('--lam', action='append', default=[])
    p.add_argument('--vectors', action='append', default=[])
    p.add_argument('--own-shift', action='append', default=[], help='name=shift.npy: [N, 2] predicted shift of a --vectors model')
    p.add_argument('--min-pairs', type=int, default=5)
    p.add_argument('--n-boot', type=int, default=200)
    p.add_argument('--kmeans-seeds', type=int, default=5)
    args = ap.parse_args()
    {'sample': sample, 'consensus': consensus, 'score': score}[args.cmd](args)


if __name__ == '__main__':
    main()
