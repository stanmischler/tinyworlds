"""STA-35: example transitions of the OT-conditioned LAMs on the held-out Zelda split.

Per pair (t, t + gap): frame t, the calibrated OT plan drawn on frame t, the ground-truth frame t + gap, and frame t + gap
rebuilt by each LAM from frame t + its own inferred action (+ the plan for the OT arms), with PSNR and the action code.
Copying frame t is the baseline PSNR. The decoder runs unmasked (eval mode), two frames only (the eval_lam two-frame regime).

Usage (from the repo root, PYTHONPATH=$PWD):
    python experiments/patch_similarity/lam_ot_examples.py --lam O1=results/lam_zelda_ot/O1/latent_actions_step_6000 --lam Z4=... \
        --pairs 78,1050,2539,4000,6222,2979 --n-random 6
Outputs: <out_dir>/example_t<t>.png per pair and <out_dir>/overview.png.
"""

import argparse
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.image_metrics import psnr
from evaluation.windows import to_model_range, to_unit
from experiments.patch_similarity.patch_similarity import show_frame, test_blocks
from utils.utils import load_latent_actions_from_checkpoint


def draw_plan(ax, img, sigma, created, Hp, Wp):
    # orange arrow = token transported to another cell, red x = destroyed, white o = created (in frame t + gap)
    show_frame(ax, img, Hp, Wp, grid=False)
    i = np.arange(Hp * Wp)
    moved = (sigma >= 0) & (sigma != i)
    src = i[moved]
    if len(src):
        ax.quiver(src % Wp + 0.5, src // Wp + 0.5, sigma[src] % Wp - src % Wp, sigma[src] // Wp - src // Wp,
                  angles='xy', scale_units='xy', scale=1, color='orange', width=0.006, headwidth=3.5, headlength=4)
    dst = i[sigma < 0]
    ax.plot(dst % Wp + 0.5, dst // Wp + 0.5, 'x', color='red', ms=4, mew=1.2)
    crt = i[created]
    ax.plot(crt % Wp + 0.5, crt // Wp + 0.5, 'o', mfc='none', mec='white', ms=4, mew=1)
    return int(moved.sum()), len(dst), len(crt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lam', action='append', required=True, help='label=<latent_actions checkpoint dir>; repeatable')
    p.add_argument('--test-h5', default='data/zelda_test_frames.h5')
    p.add_argument('--ot-plans', default='data/zelda_test_uot_gap4.npz')
    p.add_argument('--pairs', default='78,1050,2539,4000,6222,2979', help='local indices t in the test .h5')
    p.add_argument('--n-random', type=int, default=6, help='extra pairs drawn uniformly over the test blocks (seed 0)')
    p.add_argument('--gap', type=int, default=4)
    p.add_argument('--patch', type=int, default=4)
    p.add_argument('--out-dir', default='eval_results/sta35_lam_ot/examples')
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    h5 = h5py.File(args.test_h5, 'r')
    X = h5['frames']
    blocks = test_blocks(h5)
    plans = np.load(args.ot_plans)
    sig_all, cr_all = plans['sigma'], plans['created']
    Hp, Wp = X.shape[1] // args.patch, X.shape[2] // args.patch

    ts = [int(v) for v in args.pairs.split(',') if v]
    rng = np.random.default_rng(0)
    valid = np.concatenate([np.arange(s, e - args.gap) for s, e in blocks])
    ts += sorted(int(v) for v in rng.choice(valid, args.n_random, replace=False))

    lams = []
    for spec in args.lam:
        label, ckpt = spec.split('=', 1)
        lam, _ = load_latent_actions_from_checkpoint(ckpt, device)
        lams.append((label, lam.eval()))

    rows = []
    with torch.no_grad():
        for t in ts:
            pair = np.stack([X[t], X[t + args.gap]])  # uint8 [2, H, W, C]
            x = to_model_range(pair[None], device)  # [1, 2, C, H, W]
            ot = torch.from_numpy(np.stack([sig_all[t], cr_all[t]])[None, None].astype(np.int64)).to(device)  # [1, 1, 2, P]
            gt = to_unit(x[:, 1])  # [1, C, H, W]
            recs = []
            for label, lam in lams:
                o = ot if getattr(lam, 'uses_ot', False) else None
                a = lam.encode(x, o)  # [1, 1, A]
                code = int(lam.quantizer.get_indices_from_latents(a).item())
                rec = to_unit(lam.decoder(x, a, training=False, ot=o)[:, 0])  # [1, C, H, W]
                recs.append((label, rec[0].permute(1, 2, 0).cpu().numpy(), float(psnr(rec, gt)[0]), code))
            rows.append((t, pair, sig_all[t].astype(int), cr_all[t], float(psnr(to_unit(x[:, 0]), gt)[0]), recs))

    def draw_row(axes, row):
        t, pair, sigma, created, p_copy, recs = row
        axes[0].imshow(pair[0]); axes[0].set_title(f't={t}: frame t\n(copy = {p_copy:.1f} dB)', fontsize=8)
        n_mov, n_des, n_cr = draw_plan(axes[1], pair[0], sigma, created, Hp, Wp)
        axes[1].set_title(f'OT plan on frame t\nmoved {n_mov} / destroyed {n_des} / created {n_cr}', fontsize=8)
        axes[2].imshow(pair[1]); axes[2].set_title(f'ground truth t+{args.gap}', fontsize=8)
        for ax, (label, img, p_rec, code) in zip(axes[3:], recs):
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f'{label} rebuilt, action {code}\n{p_rec:.1f} dB ({p_rec - p_copy:+.1f} vs copy)', fontsize=8)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])

    n_col = 3 + len(lams)
    legend = 'orange arrow = token moved, red x = destroyed, white o = created (calibrated unbalanced OT, hidden tokens, lam 0.3, tau 0.5)'
    for row in rows:
        fig, axes = plt.subplots(1, n_col, figsize=(n_col * 2.6, 3.2))
        draw_row(axes, row)
        fig.suptitle(legend, fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(args.out_dir, f'example_t{row[0]}.png'), dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(len(rows), n_col, figsize=(n_col * 2.4, len(rows) * 2.8), squeeze=False)
    for r, row in enumerate(rows):
        draw_row(axes[r], row)
    fig.suptitle(legend, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(os.path.join(args.out_dir, 'overview.png'), dpi=90)
    plt.close(fig)
    for t, _, _, _, p_copy, recs in rows:
        print(f't={t:5d} copy {p_copy:5.1f} | ' + ' | '.join(f'{l} {p:5.1f} (a={c})' for l, _, p, c in recs))


if __name__ == '__main__':
    main()
