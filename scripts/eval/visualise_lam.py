"""Visualise what a frozen CoMo action model learned on Zelda (continuous z, no k-means). Library: evaluation/lam_vis/.

Outputs, in <out-dir>/<name>/:
  scores.json         cycle and leakage scores on the held-out test windows (evaluation/lam_vis/cycle.py): PSNR, SSIM,
                      LPIPS of direct Dec(A_t, z), cycle Dec(A_t, z'), other_block_action Dec(A_t, z_B), copy A_t, all vs
                      A_t+4; leak_to_C and copy_to_C vs C; centred cos(z, z') and cos(z, z_B) (unrelated pairs)
  cycle_strips.png    A_t | A_t+4 | Dec(A_t, z) | B_t | Dec(B_t, z) | Dec(A_t, z') for labelled walk pairs
  same_action.png     the action of one real pair per label applied to the same start frames
  repeated_action.png that action applied again and again from one start frame
  seed_windows.png    Dec(frame t, z) on the 8 PNG windows of eval_next_frame (first window of test blocks 0-7), where
                      the flow dynamics models lose Link
  cosine.png          mean cosine of z between judge labels
  pca.png             3 static 2D views of the 3D PCA
  page/               interactive 3D PCA (page.html + thumbs/), published as an Artifact

Pairs: labelled = judge set of scripts/eval/lam_eval.py (both judges agree); windows = the 890 eval_next_frame windows
(pair = last context frame, target). B (transfer start) and C (leakage "future") come from another test block, seeded.
For the figures, B is a STILL pair from another block (Link is visible and standing, so a transferred walk shows).
The per-label example pair is the label's most typical pair: its z is the closest to the label's mean z (no cherry-picking
by score). Cosines use z centred on the train mean (the PCA origin).

    PYTHONPATH=$PWD python scripts/eval/visualise_lam.py --ckpt results/como_zelda_v1/como/checkpoints/como_step_50000 \
        --name como_v1_50k --device mps
"""

import argparse
import json
import os

import numpy as np
import torch

from evaluation.image_grid import render_grid
from evaluation.image_metrics import try_lpips
from evaluation.lam_vis import figures
from evaluation.lam_vis.actions import FrozenCoMo
from evaluation.lam_vis.cycle import cycle, image_scores, leakage
from evaluation.lam_vis.data import labelled_pairs, other_block_partner, train_pairs, window_pairs
from evaluation.lam_vis.embedding import centred_cosine, fit_pca, group_cosine, most_typical, project
from evaluation.lam_vis.page import write_page

WALKS = ['WALK_U', 'WALK_D', 'WALK_L', 'WALK_R']
COSINE_GROUPS = WALKS + ['STILL', 'NONCONTROL']


def to_model(frames_u8):
    # uint8 [N, H, W, C] -> float [N, C, H, W] in [-1, 1] (CPU; FrozenCoMo moves batches to the device)
    return torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float() / 127.5 - 1


def mean_scores(per_pair):
    return {k: round(float(v.float().mean()), 4) for k, v in per_pair.items()}


def score_windows(model, win, mean, lpips_fn):
    """Cycle + leakage on the held-out windows -> summary dict (means over windows)."""
    a_t, a_next = to_model(win['frame_t']), to_model(win['frame_next'])
    b = other_block_partner(win['block'], seed=0)
    c = other_block_partner(win['block'], seed=1)
    out = cycle(model, a_t, a_next, a_t[b], a_next[b])
    leak = leakage(model, a_t, a_next[c])
    s = {'direct': image_scores(out['direct'], a_next, lpips_fn),
         'cycle': image_scores(out['a_hat'], a_next, lpips_fn),
         'other_block_action': image_scores(out['other_block_action'], a_next, lpips_fn),
         'copy': image_scores(a_t, a_next, lpips_fn),
         'leak_to_C': image_scores(leak, a_next[c], lpips_fn),
         'copy_to_C': image_scores(a_t, a_next[c], lpips_fn)}
    summary = {k: mean_scores(v) for k, v in s.items()}
    summary['cos_z_zback'] = round(float(centred_cosine(out['z'].numpy(), out['z_back'].numpy(), mean).mean()), 4)
    summary['cos_z_zB'] = round(float(centred_cosine(out['z'].numpy(), out['z_b'].numpy(), mean).mean()), 4)  # unrelated pairs
    if lpips_fn is not None:
        summary['lpips_ratio_cycle_over_direct'] = round(summary['cycle']['lpips'] / summary['direct']['lpips'], 3)
    return summary


def figure_cycle_strips(model, lab, z_lab, mean, per_label=2):
    """Strips for the per_label most typical pairs of each walk label, B = a STILL pair from another block."""
    rng = np.random.default_rng(0)
    still = np.flatnonzero(lab['label'] == 'STILL')
    rows = []
    for label in WALKS:
        for i in most_typical(z_lab, lab['label'] == label, per_label):
            j = rng.choice(still[lab['block'][still] != lab['block'][i]])
            a_t, a_next = to_model(lab['frame_t'][[i]]), to_model(lab['frame_next'][[i]])
            b_t, b_next = to_model(lab['frame_t'][[j]]), to_model(lab['frame_next'][[j]])
            out = cycle(model, a_t, a_next, b_t, b_next)
            p_direct = image_scores(out['direct'], a_next)['psnr'][0]
            p_cycle = image_scores(out['a_hat'], a_next)['psnr'][0]
            rows.append({'label': label, 'a_t': a_t[0], 'a_next': a_next[0], 'direct': out['direct'][0], 'b_t': b_t[0],
                         'b_hat': out['b_hat'][0], 'a_hat': out['a_hat'][0], 'psnr_direct': float(p_direct),
                         'psnr_cycle': float(p_cycle),
                         'cos': float(centred_cosine(out['z'].numpy(), out['z_back'].numpy(), mean)[0])})
    return figures.cycle_strips(rows)


def figure_seed_windows(model, h5_path, n_blocks=8):
    """The decoder with each pair's own action on eval_next_frame's PNG windows (first window of each test block)."""
    win = window_pairs(h5_path)
    _, first = np.unique(win['block'], return_index=True)  # windows are in block order
    first = first[:n_blocks]
    a_t, a_next = to_model(win['frame_t'][first]), to_model(win['frame_next'][first])
    pred = model.apply(a_t, model.infer(a_t, a_next))
    p_pred, p_copy = image_scores(pred, a_next)['psnr'], image_scores(a_t, a_next)['psnr']
    rows = [[figures.unit(a_t[i]), figures.unit(a_next[i]), figures.unit(pred[i])] for i in range(len(first))]
    captions = [['', f'copy {p_copy[i]:.1f} dB', f'{p_pred[i]:.1f} dB'] for i in range(len(first))]
    return render_grid(rows, ['last context frame', 'target (real)', "decoder + the pair's own action"],
                       [f'block {b}' for b in win['block'][first]], captions)


def start_frames(lab, n):
    """frame t of n STILL pairs (Link visible and standing): the first of each test block in pool order, then the
    next ones in pool order (STILL pairs cover only 7 of the 10 test blocks)."""
    still = np.flatnonzero(lab['label'] == 'STILL')
    _, first = np.unique(lab['block'][still], return_index=True)
    order = list(still[np.sort(first)]) + [i for i in still if i not in set(still[first])]
    assert len(order) >= n, f'only {len(order)} STILL pairs'
    return to_model(lab['frame_t'][order[:n]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='results/como_zelda_v1/como/checkpoints/como_step_50000')
    p.add_argument('--name', default='como_v1_50k')
    p.add_argument('--set-dir', default='eval_results/lam_eval/zelda/set')
    p.add_argument('--test-h5', default='data/zelda_test_frames.h5')
    p.add_argument('--train-h5', default='data/zelda_train_frames.h5')
    p.add_argument('--train-z', default='data/zelda_train_como_z.npy', help='precomputed raw z of train pairs (i, i+4)')
    p.add_argument('--n-train', type=int, default=5000, help='train pairs drawn in the grey cloud')
    p.add_argument('--n-windows', type=int, default=None, help='N test windows evenly spaced over the test set (default: all 890)')
    p.add_argument('--n-starts', type=int, default=8)
    p.add_argument('--n-repeats', type=int, default=6)
    p.add_argument('--device', default='mps')
    p.add_argument('--out-dir', default='eval_results/lam_vis')
    a = p.parse_args()
    out = os.path.join(a.out_dir, a.name)
    os.makedirs(out, exist_ok=True)

    model = FrozenCoMo(a.ckpt, a.device)
    lab = labelled_pairs(a.set_dir, a.test_h5)
    train = train_pairs(a.train_h5, a.train_z, a.n_train)
    z_lab = model.infer(to_model(lab['frame_t']), to_model(lab['frame_next'])).numpy()  # [N_lab, A]

    # the precomputed train z must be what this checkpoint infers, or the grey cloud is another model's (the npy was built
    # from fp16 MAE features: per-pair cosine ~0.9999, relative error ~1%, vs ~140% between different pairs)
    z_check = model.infer(to_model(train['frame_t'][:16]), to_model(train['frame_next'][:16])).numpy()
    cos_check = float(centred_cosine(z_check, train['z'][:16], np.zeros_like(z_check[0])).min())
    print(f'train z check: min cosine {cos_check:.4f}')
    assert cos_check > 0.99, 'precomputed train z do not match this checkpoint'

    # 3D PCA, fit on all valid train z
    mean, comps, var = fit_pca(train['z_all'])
    train_xyz, lab_xyz = project(train['z'], mean, comps), project(z_lab, mean, comps)
    figures.pca_views(train_xyz, lab_xyz, lab['label'], var, os.path.join(out, 'pca.png'))
    write_page(os.path.join(out, 'page'), train, train_xyz, lab, lab_xyz, var,
               {'ckpt_name': a.name, 'n_train': a.n_train, 'n_labelled': len(z_lab)})

    # cosine between labels, centred on the train mean
    cos = group_cosine(z_lab, lab['label'], COSINE_GROUPS, mean)
    figures.cosine_heatmap(cos, COSINE_GROUPS, [int((lab['label'] == g).sum()) for g in COSINE_GROUPS], os.path.join(out, 'cosine.png'))

    # cycle + leakage scores on the held-out windows
    lpips_fn = try_lpips(a.device)
    win = window_pairs(a.test_h5, a.n_windows)
    scores = score_windows(model, win, mean, lpips_fn)
    scores.update({'name': a.name, 'ckpt': a.ckpt, 'n_windows': len(win['row']), 'pca_var': [float(v) for v in var],
                   'group_cosine': {g: dict(zip(COSINE_GROUPS, map(float, row))) for g, row in zip(COSINE_GROUPS, cos)}})
    json.dump(scores, open(os.path.join(out, 'scores.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in scores.items() if k != 'group_cosine'}, indent=1))

    # figures on labelled pairs
    figure_seed_windows(model, a.test_h5).save(os.path.join(out, 'seed_windows.png'))
    figure_cycle_strips(model, lab, z_lab, mean).save(os.path.join(out, 'cycle_strips.png'))
    starts = start_frames(lab, a.n_starts)  # [S, C, H, W]
    sources, preds, rollouts = [], [], []
    for label in WALKS + ['STILL']:
        i = most_typical(z_lab, lab['label'] == label)[0]
        z = torch.from_numpy(z_lab[[i]])  # [1, A]
        sources.append((label, lab['frame_t'][i], lab['frame_next'][i]))
        preds.append(list(model.apply(starts, z.expand(len(starts), -1))))
        if label in WALKS:
            frames = [starts[0]]
            for _ in range(a.n_repeats):
                frames.append(model.apply(frames[-1][None], z)[0])
            rollouts.append((label, frames))
    figures.same_action_grid(sources, list(starts), preds).save(os.path.join(out, 'same_action.png'))
    figures.rollout_grid(rollouts, a.n_repeats).save(os.path.join(out, 'repeated_action.png'))
    print('wrote', out)


if __name__ == '__main__':
    main()
