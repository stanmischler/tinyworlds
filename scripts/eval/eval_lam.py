"""Latent action model (LAM) diagnostic on the held-out test split: is the action code collapsed, and does it carry information?

Protocol (fixed, deterministic): windows of `seq_len` frames, `frame_skip` stored frames apart (the LAM training
sequence), starting every `sample_stride` frames inside each test block. Every transition of every window is scored.
  - usage:      code histogram, entropy (nats, max ln n_actions), perplexity, codes used (>= 1% of transitions)
  - bits:       per FSQ dim, P(bit = 1), mean |z| (pre-tanh) and the fraction saturated (|tanh z| > 0.99, where the
                straight-through gradient is ~0)
  - decoder:    reconstruction error (smooth L1 and PSNR) of frames 1..T-1 with the true actions vs the actions of a random
                other window (seeded global permutation; shuffle gap = relative loss increase; ~0 means the decoder ignores actions), in two regimes:
                `masked` = the training regime (only frame 0 visible), `full` = every context frame visible
  - spread:     mean per-pixel std (in [0, 1]) of the decoded last frame across all n_actions codes for the last
                transition; ~0 means every action decodes to the same frame
  - motion:     global shift (dx, dy) between consecutive frames by phase correlation, binned by sign into 9 classes
                (|shift| <= 1 px counts as 0); normalized mutual information NMI(code, motion class) in [0, 1]

Outputs per model: <out_dir>/<name>.json and <out_dir>/<name>.png (code histogram, code-vs-motion table, and for
4 windows spread over the split: frame 0, true last frame, decoded last frame under each action, masked regime).

Usage (from the repo root, PYTHONPATH=$PWD):
    python scripts/eval/eval_lam.py --lam v4=results/sonic_v4_2026_10_01/latent_actions/checkpoints/latent_actions_step_9000 \
        --lam v3=../test/results/sonic_v3_2026_09_23/latent_actions/checkpoints/latent_actions_step_2250
    python scripts/eval/eval_lam.py --lam v4=... --limit 64   # smoke
"""

import argparse
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_next_frame import psnr, test_windows, load_window_batch, to_model_range, to_unit  # noqa: E402
from utils.utils import load_latent_actions_from_checkpoint  # noqa: E402

MOTION_LABELS = [f'{v}{h}' for v in ('U', '0', 'D') for h in ('L', '0', 'R')]  # 9 classes, '00' = still


# ----------------------------------------------------------------------------- helpers
def global_shift(a, b):
    # a, b: [N, H, W] grayscale -> integer (dy, dx) [N, 2] such that b ~ roll(a, (dy, dx)), by phase correlation
    Fa, Fb = torch.fft.fft2(a), torch.fft.fft2(b)
    r = Fb * Fa.conj()
    r = r / r.abs().clamp_min(1e-8)
    corr = torch.fft.ifft2(r).real  # [N, H, W]
    N, H, W = corr.shape
    flat = corr.flatten(1).argmax(1)
    dy, dx = flat // W, flat % W
    dy = torch.where(dy > H // 2, dy - H, dy)
    dx = torch.where(dx > W // 2, dx - W, dx)
    return torch.stack([dy, dx], 1)  # [N, 2]


def motion_class(shift, tol=1):
    # shift: [N, 2] (dy, dx) -> class index [N] in 0..8 (row = vertical U/0/D, col = horizontal L/0/R)
    v = torch.where(shift[:, 0] < -tol, 0, torch.where(shift[:, 0] > tol, 2, 1))
    h = torch.where(shift[:, 1] < -tol, 0, torch.where(shift[:, 1] > tol, 2, 1))
    return v * 3 + h


def entropy_nats(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def nmi(table):
    # table: [n_codes, n_classes] joint counts -> I(X;Y) / sqrt(H(X) H(Y))
    pxy = table / table.sum()
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    hx, hy = entropy_nats(table.sum(1)), entropy_nats(table.sum(0))
    return mi / math.sqrt(hx * hy) if hx > 0 and hy > 0 else 0.0


def decode(lam, x, actions, masked):
    # x: [B, T, C, H, W], actions: [B, T-1, A] -> predicted frames 1..T-1 [B, T-1, C, H, W]
    # the decoder masks frames 1.. only when in train mode (no other layer of the LAM depends on the mode)
    lam.decoder.train(masked)
    out = lam.decoder(x, actions, training=True)
    lam.decoder.eval()
    return out


def per_sample_loss(pred, target):
    # [B, T-1, C, H, W] -> smooth L1 [B], PSNR [B] (mean over frames, in [0, 1] space)
    l1 = F.smooth_l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    p = torch.stack([psnr(to_unit(pred[:, t]), to_unit(target[:, t])) for t in range(pred.shape[1])], 1).mean(1)
    return l1, p


# ----------------------------------------------------------------------------- per model
def evaluate(label, ckpt, args, windows, frames_dset, device):
    lam, _ = load_latent_actions_from_checkpoint(ckpt, device)
    lam.eval()
    q = lam.quantizer
    n_actions, A = q.codebook_size, lam.action_dim
    all_codes = q.get_latents_from_indices(torch.arange(n_actions, device=device))  # [n_actions, A]

    codes, bits, absz, sat, motion = [], [], [], [], []
    acc = {f'{r}_{k}': [] for r in ('masked', 'full') for k in ('l1_true', 'l1_shuf', 'psnr_true', 'psnr_shuf')}
    spread = {'masked': [], 'full': []}
    seeds = []
    seed_ids = set(np.linspace(0, len(windows) - 1, 4).round().astype(int).tolist())  # 4 windows spread over the split
    with torch.no_grad():
        # pass 1: quantized actions of every window, so the shuffled actions come from anywhere in the split
        # (neighbouring windows of a block have near-identical actions)
        zq_all = torch.cat([q(lam.encoder(to_model_range(load_window_batch(
            frames_dset, windows[i:i + args.batch_size], args.seq_len - 1, args.frame_skip), device)))
            for i in range(0, len(windows), args.batch_size)])  # [N, T-1, A]
        perm = torch.randperm(len(windows), generator=torch.Generator().manual_seed(0)).to(device)  # [N]

        for i in range(0, len(windows), args.batch_size):
            batch = windows[i:i + args.batch_size]
            x = to_model_range(load_window_batch(frames_dset, batch, args.seq_len - 1, args.frame_skip), device)  # [B, T, C, H, W]
            B, T = x.shape[:2]
            z = lam.encoder(x)  # [B, T-1, A]
            zq = q(z)  # [B, T-1, A]
            idx = q.get_indices_from_latents(zq)  # [B, T-1]
            codes.append(idx.flatten().cpu())
            bits.append((z > 0).reshape(-1, A).float().cpu())
            absz.append(z.abs().reshape(-1, A).cpu())
            sat.append((torch.tanh(z).abs() > 0.99).reshape(-1, A).float().cpu())

            gray = to_unit(x).mean(2)  # [B, T, H, W]
            shift = global_shift(gray[:, :-1].reshape(-1, *gray.shape[2:]), gray[:, 1:].reshape(-1, *gray.shape[2:]))  # [B*(T-1), 2]
            motion.append(motion_class(shift).cpu())

            target = x[:, 1:]  # [B, T-1, C, H, W]
            zq_shuf = zq_all[perm[i:i + B]]  # [B, T-1, A] a random other window's actions
            for regime in ('masked', 'full'):
                masked = regime == 'masked'
                l1_t, p_t = per_sample_loss(decode(lam, x, zq, masked), target)
                l1_s, p_s = per_sample_loss(decode(lam, x, zq_shuf, masked), target)
                acc[f'{regime}_l1_true'].append(l1_t.cpu()); acc[f'{regime}_l1_shuf'].append(l1_s.cpu())
                acc[f'{regime}_psnr_true'].append(p_t.cpu()); acc[f'{regime}_psnr_shuf'].append(p_s.cpu())

                # decoded last frame under every action for the last transition
                outs = []
                for k in range(n_actions):
                    a = zq.clone()
                    a[:, -1] = all_codes[k]
                    outs.append(to_unit(decode(lam, x, a, masked)[:, -1]))  # [B, C, H, W]
                outs = torch.stack(outs, 1)  # [B, n_actions, C, H, W]
                spread[regime].append(outs.std(1).flatten(1).mean(1).cpu())
                if masked:
                    for j in [w - i for w in sorted(seed_ids) if i <= w < i + B]:
                        seeds.append((to_unit(x[j, 0]).cpu(), to_unit(x[j, -1]).cpu(), outs[j].cpu(), idx[j, -1].item()))

    codes, motion = torch.cat(codes).numpy(), torch.cat(motion).numpy()
    bits, absz, sat = torch.cat(bits), torch.cat(absz), torch.cat(sat)
    counts = np.bincount(codes, minlength=n_actions).astype(float)
    table = np.zeros((n_actions, 9))
    np.add.at(table, (codes, motion), 1)
    H = entropy_nats(counts)
    res = {
        'label': label, 'checkpoint': ckpt, 'n_actions': n_actions, 'windows': len(windows), 'transitions': int(len(codes)),
        'entropy_nats': H, 'max_entropy_nats': math.log(n_actions), 'perplexity': math.exp(H),
        'codes_used_1pct': int((counts / counts.sum() >= 0.01).sum()), 'code_freq': (counts / counts.sum()).tolist(),
        'bit_p1': bits.mean(0).tolist(), 'bit_abs_z': absz.mean(0).tolist(), 'bit_saturated': sat.mean(0).tolist(),
        'motion_freq': (table.sum(0) / table.sum()).tolist(), 'motion_labels': MOTION_LABELS,
        'code_motion_counts': table.astype(int).tolist(), 'nmi_code_motion': nmi(table),
    }
    for regime in ('masked', 'full'):
        l1_t, l1_s = torch.cat(acc[f'{regime}_l1_true']).mean().item(), torch.cat(acc[f'{regime}_l1_shuf']).mean().item()
        res[regime] = {
            'l1_true': l1_t, 'l1_shuffled': l1_s, 'shuffle_gap_rel': (l1_s - l1_t) / l1_t,
            'psnr_true': torch.cat(acc[f'{regime}_psnr_true']).mean().item(),
            'psnr_shuffled': torch.cat(acc[f'{regime}_psnr_shuf']).mean().item(),
            'action_spread': torch.cat(spread[regime]).mean().item(),
        }
    return res, seeds


def plot(res, seeds, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = res['n_actions']
    fig = plt.figure(figsize=(2 + 1.3 * (n + 2), 4 + 1.4 * len(seeds)))
    gs = fig.add_gridspec(1 + len(seeds), n + 2, height_ratios=[2.2] + [1] * len(seeds))
    ax = fig.add_subplot(gs[0, :(n + 2) // 2])
    ax.bar(range(n), res['code_freq'])
    ax.set_xticks(range(n)); ax.set_xlabel('action code'); ax.set_ylabel('freq')
    ax.set_title(f"{res['label']}: H={res['entropy_nats']:.2f}/{res['max_entropy_nats']:.2f} nats, "
                 f"gap(masked)={res['masked']['shuffle_gap_rel']:.3f}, NMI={res['nmi_code_motion']:.3f}", fontsize=9)
    ax = fig.add_subplot(gs[0, (n + 2) // 2:])
    t = np.array(res['code_motion_counts'], dtype=float)
    ax.imshow(t / t.sum(1, keepdims=True).clip(1), aspect='auto', cmap='viridis')
    ax.set_xticks(range(9)); ax.set_xticklabels(MOTION_LABELS, fontsize=7); ax.set_yticks(range(n))
    ax.set_xlabel('image content shift (vertical, horizontal)'); ax.set_ylabel('code'); ax.set_title('P(motion | code)', fontsize=9)
    for r, (f0, last, outs, code) in enumerate(seeds):
        imgs = [('frame 0', f0), ('true last', last)] + [(f'a={k}' + (' *' if k == code else ''), outs[k]) for k in range(n)]
        for c, (title, im) in enumerate(imgs):
            a = fig.add_subplot(gs[1 + r, c])
            a.imshow(im.permute(1, 2, 0).numpy()); a.axis('off')
            if r == 0 or c >= 2:
                a.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lam', action='append', required=True, help='label=<latent_actions checkpoint dir>; repeatable')
    p.add_argument('--test-h5', default='data/sonic_test_frames.h5')
    p.add_argument('--seq-len', type=int, default=4, help='frames per window (the LAM training context_length)')
    p.add_argument('--frame-skip', type=int, default=4, help='stored frames between sequence frames (60 // fps in the loader)')
    p.add_argument('--sample-stride', type=int, default=8, help='stored frames between window starts inside a block')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    p.add_argument('--limit', type=int, help='evaluate only the first N windows (smoke test)')
    p.add_argument('--out-dir', default='eval_results')
    p.add_argument('--prefix', default='lam_diag_', help='output file stem prefix; the stem is <prefix><label>')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    windows = test_windows(args.test_h5, args.seq_len - 1, args.frame_skip, args.sample_stride)
    if args.limit:
        windows = windows[:args.limit]
    h5 = h5py.File(args.test_h5, 'r')
    print(f'{len(windows)} windows of {args.seq_len} frames from {args.test_h5}')

    rows = []
    for spec in args.lam:
        label, ckpt = spec.split('=', 1)
        assert os.path.isdir(ckpt), f'checkpoint dir missing: {ckpt}'
        t0 = time.time()
        res, seeds = evaluate(label, ckpt, args, windows, h5['frames'], device)
        stem = os.path.join(args.out_dir, f'{args.prefix}{label}')
        with open(stem + '.json', 'w') as f:
            json.dump(res, f, indent=1)
        plot(res, seeds, stem + '.png')
        print(f'== {label} ({time.time() - t0:.0f}s) -> {stem}.json/.png')
        rows.append(res)

    print(f"\n{'model':<8}{'H nats':>8}{'used':>6}{'top code':>10}{'sat bits':>18}{'gap mask':>10}{'gap full':>10}"
          f"{'spread m':>10}{'NMI':>7}")
    for r in rows:
        print(f"{r['label']:<8}{r['entropy_nats']:>8.3f}{r['codes_used_1pct']:>6}{max(r['code_freq']):>10.3f}"
              f"{' '.join(f'{s:.2f}' for s in r['bit_saturated']):>18}{r['masked']['shuffle_gap_rel']:>10.4f}"
              f"{r['full']['shuffle_gap_rel']:>10.4f}{r['masked']['action_spread']:>10.4f}{r['nmi_code_motion']:>7.3f}")


if __name__ == '__main__':
    main()
