"""Judge-labelled action eval for a latent action model, on any game: do frame pairs that share an action share a LAM code?

Protocol (fixed, deterministic, LAM-independent, so every LAM of a game is scored on the same ground truth):
  - pool:      `sample` draws `n` (default 400, seed 0) held-out windows of 4 frames (`skip` stored frames apart, windows
               share no frame, or half-overlap if the split is too short) from the game's test h5 (eval_next_frame.test_windows). The judged pair is the LAST transition
               of each window (frame t -> t+1); the 2 frames before it are the LAM's context, the judges never see them.
               Writes numbered sheets (5 pairs per PNG: frame t | frame t+1 | |change| x3) and vocab sheets (the first 40).
  - groups:    one VLM judge looks at the vocab sheets and writes the game's action vocabulary set/vocab.json
               {"groups": {"<NAME>": "<one-line definition>"}} (always incl. STILL and NONCONTROL); then 2 blind judges label
               every pair: {"<id>": {"label": "<NAME>|UNSURE", "confidence": "clear|unsure"}}.
  - consensus: same label from both judges and >= 1 of them clear -> main set (at most `cap` pairs per group, seeded); same
               label, both unsure -> uncertain (bonus) set; disagreements and UNSURE are dropped. Writes the frozen set/labels.json
               {"main": {id: group}, "uncertain": {...}}, set/groups.json {group: [ids]} and groups/<GROUP>.png (one large
               example pair + up to 4 more).
  - score:     encodes each window with the LAM and reads the code of its last transition; reports on main and uncertain:
               NMI(code, group), chance-corrected NMI_adj (minus the mean NMI of 200 code permutations; headline),
               purity (homogeneity of codes), completeness (share of a group in its majority code), majority baseline and the
               code x group table. Baselines: `--baseline random` (uniform codes, 16) and `--baseline camera` (phase-correlation
               global shift binned into 9 directions; what reading the scroll alone gets).

Usage (repo root, PYTHONPATH=$PWD):
    python scripts/eval/lam_eval.py sample --game zelda                 # eval_results/lam_eval/zelda/set/{pool.json,sheets/}
    python scripts/eval/lam_eval.py sample --game mygame --h5 data/mygame_test_frames.h5 --skip 2
    python scripts/eval/lam_eval.py consensus --game zelda labels_A.json labels_B.json
    python scripts/eval/lam_eval.py score --game zelda --lam i4=<ckpt dir> --baseline random --baseline camera
    python scripts/eval/lam_eval.py score --game zelda --codes my=codes.json     # {"<id>": code}, a LAM run elsewhere
    python scripts/eval/lam_eval.py score --game zelda --lam como=<como ckpt> --kmeans-seeds 10   # continuous: k-means (k = n_actions)
    python scripts/eval/lam_eval.py score --game zelda --pair-encoder enc_pix=<encoder.pt>         # frame-pair classifier
    python scripts/eval/lam_eval.py table --game zelda                          # markdown table of every <name>/score.json
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_next_frame import test_windows, load_window_batch, to_model_range  # noqa: E402
from eval_lam import nmi, kmeans, global_shift, motion_class, window_ot  # noqa: E402
from lam_judge import transition_panel, stack_rows  # noqa: E402

# frame_skip = 60 // fps of the dataset class (datasets/datasets.py); every LAM so far uses context_length 4
GAMES = {'zelda': ('data/zelda_test_frames.h5', 4), 'sonic': ('data/sonic_test_frames.h5', 4), 'pong': ('data/pong_test_frames.h5', 2)}
SEQ = 4
PER_SHEET = 5
OUT = 'eval_results/lam_eval'


def game_cfg(args):
    h5, skip = GAMES.get(args.game, (None, None))
    h5, skip = args.h5 or h5, args.skip or skip
    assert h5 and skip, f'unknown game {args.game}: pass --h5 and --skip'
    return h5, skip, f'{OUT}/{args.game}'


def windows_of(pool):
    return [(w['block'], w['start']) for w in pool['windows']]


def pair_of(fr, w, skip):
    # last transition of the window: frames start + 2*skip -> start + 3*skip
    return fr[w['start'] + (SEQ - 2) * skip], fr[w['start'] + (SEQ - 1) * skip]


def sample(args):
    h5, skip, root = game_cfg(args)
    os.makedirs(f'{root}/set/sheets', exist_ok=True)
    os.makedirs(f'{root}/set/vocab_sheets', exist_ok=True)
    stride = SEQ * skip  # window span -> windows share no frame
    wins = test_windows(h5, SEQ - 1, skip, stride)
    if len(wins) < args.n:  # short test split (sonic): windows overlap by half, judged pairs stay distinct
        stride = 2 * skip
        wins = test_windows(h5, SEQ - 1, skip, stride)
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(wins), min(args.n, len(wins)), replace=False))
    pool = {'h5': h5, 'frame_skip': skip, 'seq_len': SEQ, 'n_candidate_windows': len(wins), 'window_stride': stride, 'seed': 0,
            'pair': 'last transition of the window', 'windows': []}
    with h5py.File(h5, 'r') as f:
        fr = f['frames']
        rows = []
        for i, w in enumerate(pick):
            blk, s = wins[w]
            rec = {'id': i, 'block': int(blk), 'start': int(s), 'frame_a': int(s + (SEQ - 2) * skip), 'frame_b': int(s + (SEQ - 1) * skip)}
            pool['windows'].append(rec)
            a, b = pair_of(fr, rec, skip)
            scale = max(1, 256 // a.shape[0])
            rows.append(transition_panel(a, b, scale=scale, tag=f'#{i}   (frame t | frame t+1 | |change| x3)'))
        for k in range(0, len(rows), PER_SHEET):
            img = stack_rows(rows[k:k + PER_SHEET], f'{args.game} sheet {k // PER_SHEET:02d}: pairs #{k}-#{min(k + PER_SHEET, len(rows)) - 1}')
            img.save(f'{root}/set/sheets/sheet_{k // PER_SHEET:02d}.png')
            if k < 40:
                img.save(f'{root}/set/vocab_sheets/sheet_{k // PER_SHEET:02d}.png')
    json.dump(pool, open(f'{root}/set/pool.json', 'w'), indent=1)
    print(f'{len(pick)} pairs of {len(wins)} candidate windows, {-(-len(pick) // PER_SHEET)} sheets in {root}/set/sheets')


def group_image(name, ids, pool, skip, path, n_more=4):
    # one large example pair (first id) + up to n_more smaller ones
    with h5py.File(pool['h5'], 'r') as f:
        fr = f['frames']
        byid = {w['id']: w for w in pool['windows']}
        H = fr.shape[1]
        rows = [transition_panel(*pair_of(fr, byid[ids[0]], skip), scale=max(2, 384 // H), tag=f'{name}: example pair #{ids[0]}')]
        rows += [transition_panel(*pair_of(fr, byid[i], skip), scale=max(1, 256 // H), tag=f'#{i}') for i in ids[1:1 + n_more]]
    stack_rows(rows, f'group {name}: {len(ids)} pairs in the main set (frame t | frame t+1 | |change| x3)').save(path)


def consensus(args):
    h5, skip, root = game_cfg(args)
    pool = json.load(open(f'{root}/set/pool.json'))
    vocab = json.load(open(f'{root}/set/vocab.json'))['groups']
    js = [{k: v for part in f.split(',') for k, v in json.load(open(part)).items()} for f in args.files]  # a judge = comma-joined parts
    main, unsure, agree = {}, {}, 0
    for w in pool['windows']:
        i = str(w['id'])
        ls = [j.get(i) for j in js]
        if any(l is None for l in ls) or len({l['label'] for l in ls}) > 1:
            continue
        agree += 1
        lab = ls[0]['label']
        if lab == 'UNSURE':
            continue
        assert lab in vocab, f'label {lab} of pair {i} not in vocab {list(vocab)}'
        (main if any(l['confidence'] == 'clear' for l in ls) else unsure)[i] = lab
    rng = np.random.default_rng(0)
    for g in vocab:  # per-group cap, so frequent groups (STILL, NONCONTROL) cannot dominate the main set
        ids = sorted([i for i, l in main.items() if l == g], key=int)
        if len(ids) > args.cap:
            for i in set(ids) - set(rng.choice(ids, args.cap, replace=False)):
                del main[i]
    groups = {g: sorted([i for i, l in main.items() if l == g], key=int) for g in vocab}
    json.dump({'judges': args.files, 'agreement': round(agree / len(pool['windows']), 4), 'main': main, 'uncertain': unsure},
              open(f'{root}/set/labels.json', 'w'), indent=1)
    json.dump(groups, open(f'{root}/set/groups.json', 'w'), indent=1)
    os.makedirs(f'{root}/groups', exist_ok=True)
    for g, ids in groups.items():
        if ids:
            group_image(g, [int(i) for i in ids], pool, skip, f'{root}/groups/{g}.png')
    print(f'agreement {agree}/{len(pool["windows"])}; main {len(main)} {dict((g, len(v)) for g, v in groups.items())}; '
          f'uncertain {len(unsure)} -> {root}/set/labels.json, {root}/groups/')


def lam_codes(ckpt, pool, device, ot_plans=None, kmeans_seeds=1, batch=32):
    # -> codes of the last transition of every pool window, one [N] array per k-means seed (1 for a discrete LAM), codebook size
    from utils.utils import load_latent_actions_from_checkpoint
    lam, _ = load_latent_actions_from_checkpoint(ckpt, device)
    lam.eval()
    q, A = lam.quantizer, lam.action_dim
    wins, skip = windows_of(pool), pool['frame_skip']
    plans = None
    if getattr(lam, 'uses_ot', False):  # OT-conditioned LAM (STA-35): needs the OT plans of the test h5
        assert ot_plans, f'{ckpt} is OT-conditioned: pass --ot-plans'
        with np.load(ot_plans) as z:
            plans = {'sigma': z['sigma'], 'created': z['created']}
    with h5py.File(pool['h5'], 'r') as f, torch.no_grad():
        x = lambda ws: to_model_range(load_window_batch(f['frames'], ws, SEQ - 1, skip), device)
        enc = (lambda ws: lam.encode(x(ws), window_ot(plans, ws, SEQ - 1, skip, device))) if plans is not None else (lambda ws: lam.encode(x(ws)))
        zq = torch.cat([enc(wins[i:i + batch]) for i in range(0, len(wins), batch)])  # [N, T-1, A]
    if getattr(lam, 'continuous_actions', False):  # k-means over all 3 transitions of each pool window, codes of the last
        z = zq.reshape(-1, A).float()
        return [torch.cdist(z, kmeans(z, q.codebook_size, seed=sd)).argmin(1).reshape(zq.shape[:2])[:, -1].cpu().numpy()
                for sd in range(kmeans_seeds)], q.codebook_size
    return [q.get_indices_from_latents(zq[:, -1]).cpu().numpy()], q.codebook_size


def pair_encoder_codes(ckpt, pool, device, batch=256):
    # frame-pair classifier (scripts/train_action_encoder.py PairEncoder, e.g. STA-35 enc_pix) -> argmax class of each pair
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from train_action_encoder import PairEncoder, N_CLASSES, to_float
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = PairEncoder(ck['args']['width']).to(device).eval()
    model.load_state_dict(ck['model'])
    out = []
    with h5py.File(pool['h5'], 'r') as f, torch.no_grad():
        for i in range(0, len(pool['windows']), batch):
            ab = [pair_of(f['frames'], w, pool['frame_skip']) for w in pool['windows'][i:i + batch]]
            fa, fb = (torch.from_numpy(np.stack([p[k] for p in ab])).to(device) for k in (0, 1))
            out.append(model(to_float(fa), to_float(fb)).argmax(-1).cpu())
    return [torch.cat(out).numpy()], N_CLASSES


def baseline_codes(kind, pool):
    if kind == 'random':
        return np.random.default_rng(0).integers(0, 16, len(pool['windows'])), 16
    assert kind == 'camera', kind
    with h5py.File(pool['h5'], 'r') as f:
        fr = f['frames']
        ab = np.stack([np.stack(pair_of(fr, w, pool['frame_skip'])) for w in pool['windows']])  # [N, 2, H, W, C]
    g = torch.from_numpy(ab).float().mean(-1) / 255  # [N, 2, H, W]
    return motion_class(global_shift(g[:, 0], g[:, 1])).numpy(), 9


def table_of(codes, labels, n_codes, classes):
    tab = np.zeros((n_codes, len(classes)))
    for c, l in zip(codes, labels):
        tab[c, classes.index(l)] += 1
    return tab


def metrics(codes, labels, n_codes, classes):
    if not len(codes):
        return {'n': 0}
    tab = table_of(codes, labels, n_codes, classes)
    rng = np.random.default_rng(0)
    perm = np.mean([nmi(table_of(rng.permutation(codes), labels, n_codes, classes)) for _ in range(200)])
    v, n = nmi(tab), tab.sum()
    return {'n': int(n), 'nmi': round(v, 4), 'nmi_chance': round(float(perm), 4), 'nmi_adj': round((v - perm) / (1 - perm), 4),
            'purity': round(float(tab.max(1).sum() / n), 4), 'completeness': round(float(tab.max(0).sum() / n), 4),
            'majority_baseline': round(float(tab.sum(0).max() / n), 4), 'codes_used': int((tab.sum(1) > 0).sum()),
            'table': {f'code {k}': {cl: int(tab[k, j]) for j, cl in enumerate(classes) if tab[k, j]} for k in range(n_codes) if tab[k].sum()}}


def score(args):
    _, _, root = game_cfg(args)
    pool = json.load(open(f'{root}/set/pool.json'))
    labs = json.load(open(f'{root}/set/labels.json'))
    classes = list(json.load(open(f'{root}/set/vocab.json'))['groups'])
    specs = [(s.split('=', 1), 'lam') for s in args.lam] + [(s.split('=', 1), 'codes') for s in args.codes] + \
            [(s.split('=', 1), 'pair') for s in args.pair_encoder] + \
            [((f'baseline_{b}', b), 'baseline') for b in args.baseline]
    for (name, src), kind in specs:
        if kind == 'lam':
            runs, n_codes = lam_codes(src, pool, args.device, args.ot_plans, args.kmeans_seeds)
        elif kind == 'pair':
            runs, n_codes = pair_encoder_codes(src, pool, args.device)
        elif kind == 'baseline':
            codes, n_codes = baseline_codes(src, pool)
            runs = [codes]
        else:
            cj = json.load(open(src))
            runs, n_codes = [np.array([int(cj.get(str(w['id']), 0)) for w in pool['windows']])], int(max(cj.values())) + 1
        codes = runs[0]  # headline = k-means seed 0 for a continuous LAM
        res = {'name': name, 'source': src, 'game': args.game, 'n_codes': n_codes}
        for split in ['main', 'uncertain']:
            ids = sorted(labs[split], key=int)
            res[split] = metrics([int(codes[int(i)]) for i in ids], [labs[split][i] for i in ids], n_codes, classes)
            if len(runs) > 1:  # continuous LAM: clustering noise over k-means seeds
                v = [metrics([int(c[int(i)]) for i in ids], [labs[split][i] for i in ids], n_codes, classes).get('nmi_adj', 0) for c in runs]
                res[split]['nmi_adj_kmeans'] = {'mean': round(float(np.mean(v)), 4), 'sd': round(float(np.std(v)), 4), 'n_seeds': len(runs)}
        u = np.bincount(codes, minlength=n_codes) / len(codes)
        p = u[u > 0]
        res['pool_usage'] = [round(float(x), 4) for x in u]
        res['pool_entropy_nats'] = round(float(-(p * np.log(p)).sum()), 4)
        res['per_pair'] = {str(w['id']): int(c) for w, c in zip(pool['windows'], codes)}
        os.makedirs(f'{root}/{name}', exist_ok=True)
        json.dump(res, open(f'{root}/{name}/score.json', 'w'), indent=1)
        m, un = res['main'], res['uncertain']
        print(f"{args.game}/{name}: main n={m['n']} NMI_adj {m.get('nmi_adj')} purity {m.get('purity')} completeness {m.get('completeness')} "
              f"(majority {m.get('majority_baseline')}) | uncertain n={un['n']} NMI_adj {un.get('nmi_adj')} | pool entropy {res['pool_entropy_nats']}"
              + (f" | k-means seeds main {m['nmi_adj_kmeans']}" if 'nmi_adj_kmeans' in m else ''))


def table(args):
    _, _, root = game_cfg(args)
    rows = []
    for p in sorted(glob.glob(f'{root}/*/score.json')):
        r = json.load(open(p))
        m, u = r['main'], r['uncertain']
        rows.append((m.get('nmi_adj', 0), f"| {r['name']} | {r['n_codes']} | {m.get('codes_used')} | {m.get('nmi_adj')} | {m.get('purity')} | "
                     f"{m.get('completeness')} | {u.get('nmi_adj')} | {r['pool_entropy_nats']} |"))
    print(f'### {args.game} (main n={m["n"]}, majority {m.get("majority_baseline")})\n')
    print('| LAM | codes | codes used (main) | NMI_adj | purity | completeness | uncertain NMI_adj | pool entropy |\n|---|---|---|---|---|---|---|---|')
    print('\n'.join(r for _, r in sorted(rows, key=lambda x: -x[0])))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    cmds = {}
    for c in ['sample', 'consensus', 'score', 'table']:
        cmds[c] = s = sub.add_parser(c)
        s.add_argument('--game', required=True, help=f'one of {list(GAMES)} or any name with --h5/--skip')
        s.add_argument('--h5', default=None, help='test h5 with a test_blocks_local attr (scripts/eval/split_dataset.py)')
        s.add_argument('--skip', type=int, default=None, help='stored frames between LAM frames (60 // dataset fps)')
    cmds['sample'].add_argument('--n', type=int, default=400)
    cmds['consensus'].add_argument('files', nargs='+', help='one labels json per judge; a judge split in parts: a0.json,a1.json')
    cmds['consensus'].add_argument('--cap', type=int, default=40, help='max pairs per group in the main set')
    cmds['score'].add_argument('--lam', action='append', default=[], help='name=<latent_actions checkpoint dir>; repeatable')
    cmds['score'].add_argument('--codes', action='append', default=[], help='name=<codes.json {pair id: code}>; repeatable')
    cmds['score'].add_argument('--baseline', action='append', default=[], choices=['random', 'camera'])
    cmds['score'].add_argument('--pair-encoder', action='append', default=[], help='name=<train_action_encoder.py encoder.pt>; repeatable')
    cmds['score'].add_argument('--kmeans-seeds', type=int, default=1, help='continuous LAMs (CoMo): also report main NMI_adj mean/sd over seeds')
    cmds['score'].add_argument('--device', default='cpu')
    cmds['score'].add_argument('--ot-plans', default=None, help='.npz of OT plans aligned with the test h5 (OT-conditioned LAMs only)')
    a = p.parse_args()
    {'sample': sample, 'consensus': consensus, 'score': score, 'table': table}[a.cmd](a)


if __name__ == '__main__':
    main()
