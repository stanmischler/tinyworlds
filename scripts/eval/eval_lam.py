"""Latent action model (LAM) diagnostic on the held-out test split: is the action code collapsed, and does it carry information?

Protocol (fixed, deterministic): windows of `seq_len` frames, `frame_skip` stored frames apart (the LAM training
sequence), starting every `sample_stride` frames inside each test block. Every transition of every window is scored.
  - usage:      code histogram, entropy (nats, max ln n_actions), perplexity, codes used (>= 1% of transitions)
  - bits:       per FSQ dim, P(bit = 1), mean |z| (pre-tanh) and the fraction saturated (|tanh z| > 0.99, where the
                straight-through gradient is ~0)
  - decoder:    reconstruction error (smooth L1 and PSNR) of frames 1..T-1 with the true actions vs the actions of a random
                other window (seeded global permutation; shuffle gap = relative loss increase; ~0 means the decoder ignores actions), in two regimes:
                `masked` = the model's own training regime (its decoder_keep_rate, seeded; frame 0 only for the original
                LAM), `full` = every context frame visible
  - two-frame:  frame 1 rebuilt from frame 0 + the inferred action a_0 (the first transition of a window), PSNR vs
                copying frame 0, with the true and the shuffled action
  - spread:     mean per-pixel std (in [0, 1]) of the decoded last frame across all n_actions codes for the last
                transition; ~0 means every action decodes to the same frame
  - continuous: for a LAM with continuous actions the 'codes' are the k-means clusters (n_actions, seeded) of the
                mean action over the evaluated windows, the per-code decodes use the cluster centres, the actions fed to
                the decoder are the means themselves (no quantization), and the bit columns describe the sign of the mean
  - motion:     global shift (dx, dy) between consecutive frames by phase correlation, binned by sign into 9 classes
                (|shift| <= 1 px counts as 0); normalized mutual information NMI(code, motion class) in [0, 1]
  - player:     a proxy of the player's move per transition (no sprite tracker): if the screen scrolls, minus the
                global shift; otherwise the dominant local shift (block matching, 8 px blocks, +-4 px) among the blocks
                that changed, or still. Reported as NMI(code, player class) and as the held-out R^2 of a linear fit
                from the action (pre-quantization latent; one-hot code) to the (dy, dx) move, fit on the first half of
                the windows and scored on the second

Outputs per model: <out_dir>/<name>.json, <out_dir>/<name>.png (code histogram, code-vs-motion table, and for
4 windows spread over the split: frame 0, true last frame, decoded last frame under each action, masked regime) and
<out_dir>/<name>_twoframe.png (3 windows: frame 0, frame 1, frame 1 rebuilt from frame 0 + action).

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


def local_shift(a, b, max_shift=4, block=8, thr=0.08, min_frac=0.05):
    # a, b: [N, H, W] grayscale in [0, 1] -> dominant integer (dy, dx) [N, 2] of the blocks that changed (0 if none)
    pool = lambda t: F.avg_pool2d(t[:, None], block)[:, 0]  # [N, h, w]
    changed = pool(((a - b).abs() > thr).float()) > min_frac  # [N, h, w]
    shifts = [(dy, dx) for dy in range(-max_shift, max_shift + 1) for dx in range(-max_shift, max_shift + 1)]
    err = torch.stack([pool((torch.roll(a, (dy, dx), (1, 2)) - b).abs()) for dy, dx in shifts])  # [S, N, h, w]
    best = torch.tensor(shifts, device=a.device)[err.argmin(0)]  # [N, h, w, 2]
    moving = changed & (best.abs().sum(-1) > 0)  # [N, h, w]
    out = torch.zeros(a.shape[0], 2, dtype=torch.long, device=a.device)
    for n in torch.nonzero(moving.flatten(1).any(1)).flatten().tolist():
        v, c = best[n][moving[n]].unique(dim=0, return_counts=True)
        out[n] = v[c.argmax()]
    return out  # [N, 2]


def player_move(a, b):
    # a, b: [N, H, W] -> (dy, dx) [N, 2]: minus the camera scroll if there is one, else the dominant local move
    cam = global_shift(a, b)  # content shift
    scrolling = (cam.abs() > 1).any(1, keepdim=True)
    return torch.where(scrolling, -cam, local_shift(a, b))


def heldout_r2(feats, target):
    # feats: [N, D], target: [N, 2] -> R^2 of least squares (with bias) fit on the first half, scored on the second
    X = torch.cat([feats.double(), torch.ones(len(feats), 1, dtype=torch.double)], 1)
    y = target.double()
    h = len(X) // 2
    w = torch.linalg.lstsq(X[:h], y[:h]).solution
    resid = y[h:] - X[h:] @ w
    return float(1 - resid.pow(2).sum() / (y[h:] - y[h:].mean(0)).pow(2).sum().clamp_min(1e-8))


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


def kmeans(z, k, iters=50, seed=0):
    # z: [N, A] -> centres [k, A] (Lloyd, k-means++ init, seeded)
    g = torch.Generator().manual_seed(seed)
    zc = z.cpu().float()
    centres = [zc[torch.randint(len(zc), (1,), generator=g)].squeeze(0)]
    for _ in range(1, k):
        d = torch.cdist(zc, torch.stack(centres)).min(1).values.pow(2)
        centres.append(zc[torch.multinomial(d / d.sum(), 1, generator=g)].squeeze(0))
    c = torch.stack(centres)
    for _ in range(iters):
        a = torch.cdist(zc, c).argmin(1)
        c = torch.stack([zc[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
    return c.to(z.device)


def decode(lam, x, actions, masked):
    # x: [B, T, C, H, W], actions: [B, T-1, A] -> predicted frames 1..T-1 [B, T-1, C, H, W]
    # the decoder masks frames 1.. only when in train mode (no other layer of the LAM depends on the mode);
    # reseeded so the true and shuffled decodes see the same mask
    torch.manual_seed(0)
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
    continuous = getattr(lam, 'continuous_actions', False)

    codes, bits, absz, sat, motion, player, pre = [], [], [], [], [], [], []
    acc = {f'{r}_{k}': [] for r in ('masked', 'full') for k in ('l1_true', 'l1_shuf', 'psnr_true', 'psnr_shuf')}
    two = {k: [] for k in ('true', 'shuf', 'copy')}
    two_ids = set(np.linspace(0, len(windows) - 1, 3).round().astype(int).tolist())
    two_rows = []
    spread = {'masked': [], 'full': []}
    seeds = []
    seed_ids = set(np.linspace(0, len(windows) - 1, 4).round().astype(int).tolist())  # 4 windows spread over the split
    with torch.no_grad():
        # pass 1: quantized actions of every window, so the shuffled actions come from anywhere in the split
        # (neighbouring windows of a block have near-identical actions)
        zq_all = torch.cat([lam.encode(to_model_range(load_window_batch(
            frames_dset, windows[i:i + args.batch_size], args.seq_len - 1, args.frame_skip), device))
            for i in range(0, len(windows), args.batch_size)])  # [N, T-1, A]
        if continuous:
            all_codes = kmeans(zq_all.reshape(-1, A), n_actions)  # [n_actions, A] cluster centres act as the codes
            to_idx = lambda zq: torch.cdist(zq.reshape(-1, A).float(), all_codes).argmin(1).reshape(zq.shape[:-1])
        else:
            all_codes = q.get_latents_from_indices(torch.arange(n_actions, device=device))  # [n_actions, A]
            to_idx = q.get_indices_from_latents
        perm = torch.randperm(len(windows), generator=torch.Generator().manual_seed(0)).to(device)  # [N]

        for i in range(0, len(windows), args.batch_size):
            batch = windows[i:i + args.batch_size]
            x = to_model_range(load_window_batch(frames_dset, batch, args.seq_len - 1, args.frame_skip), device)  # [B, T, C, H, W]
            B, T = x.shape[:2]
            z = lam.pre_quant(x)  # [B, T-1, A]
            zq = z if continuous else q(z)  # [B, T-1, A]
            idx = to_idx(zq)  # [B, T-1]
            codes.append(idx.flatten().cpu())
            bits.append((z > 0).reshape(-1, A).float().cpu())
            absz.append(z.abs().reshape(-1, A).cpu())
            sat.append((torch.tanh(z).abs() > 0.99).reshape(-1, A).float().cpu())

            gray = to_unit(x).mean(2)  # [B, T, H, W]
            shift = global_shift(gray[:, :-1].reshape(-1, *gray.shape[2:]), gray[:, 1:].reshape(-1, *gray.shape[2:]))  # [B*(T-1), 2]
            motion.append(motion_class(shift).cpu())
            mv = player_move(gray[:, :-1].reshape(-1, *gray.shape[2:]), gray[:, 1:].reshape(-1, *gray.shape[2:]))  # [B*(T-1), 2]
            player.append(mv.cpu())
            pre.append(z.reshape(-1, A).float().cpu())

            target = x[:, 1:]  # [B, T-1, C, H, W]
            zq_shuf = zq_all[perm[i:i + B]]  # [B, T-1, A] a random other window's actions

            # two-frame: frame 1 from frame 0 + a_0 (causal model, so identical to the first transition of the window)
            f1 = to_unit(x[:, 1])  # [B, C, H, W]
            rec = to_unit(decode(lam, x[:, :2], zq[:, :1], False)[:, 0])  # [B, C, H, W]
            two['true'].append(psnr(rec, f1).cpu())
            two['shuf'].append(psnr(to_unit(decode(lam, x[:, :2], zq_shuf[:, :1], False)[:, 0]), f1).cpu())
            two['copy'].append(psnr(to_unit(x[:, 0]), f1).cpu())
            for j in [w - i for w in sorted(two_ids) if i <= w < i + B]:
                two_rows.append((to_unit(x[j, 0]).cpu(), f1[j].cpu(), rec[j].cpu(), idx[j, 0].item(),
                                 two['true'][-1][j].item(), two['copy'][-1][j].item()))
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
    player, pre = torch.cat(player), torch.cat(pre)  # [N, 2], [N, A]
    pclass = motion_class(player, tol=0).numpy()
    ptable = np.zeros((n_actions, 9))
    np.add.at(ptable, (codes, pclass), 1)
    bits, absz, sat = torch.cat(bits), torch.cat(absz), torch.cat(sat)
    counts = np.bincount(codes, minlength=n_actions).astype(float)
    table = np.zeros((n_actions, 9))
    np.add.at(table, (codes, motion), 1)
    H = entropy_nats(counts)
    res = {
        'label': label, 'checkpoint': ckpt, 'continuous': continuous, 'n_actions': n_actions, 'windows': len(windows), 'transitions': int(len(codes)),
        'entropy_nats': H, 'max_entropy_nats': math.log(n_actions), 'perplexity': math.exp(H),
        'codes_used_1pct': int((counts / counts.sum() >= 0.01).sum()), 'code_freq': (counts / counts.sum()).tolist(),
        'bit_p1': bits.mean(0).tolist(), 'bit_abs_z': absz.mean(0).tolist(), 'bit_saturated': sat.mean(0).tolist(),
        'motion_freq': (table.sum(0) / table.sum()).tolist(), 'motion_labels': MOTION_LABELS,
        'code_motion_counts': table.astype(int).tolist(), 'nmi_code_motion': nmi(table),
        'player_freq': (ptable.sum(0) / ptable.sum()).tolist(), 'code_player_counts': ptable.astype(int).tolist(),
        'nmi_code_player': nmi(ptable),
        'r2_player_latent': heldout_r2(pre, player.float()),
        'r2_player_code': heldout_r2(F.one_hot(torch.from_numpy(codes), n_actions)[:, 1:], player.float()),
    }
    for regime in ('masked', 'full'):
        l1_t, l1_s = torch.cat(acc[f'{regime}_l1_true']).mean().item(), torch.cat(acc[f'{regime}_l1_shuf']).mean().item()
        res[regime] = {
            'l1_true': l1_t, 'l1_shuffled': l1_s, 'shuffle_gap_rel': (l1_s - l1_t) / l1_t,
            'psnr_true': torch.cat(acc[f'{regime}_psnr_true']).mean().item(),
            'psnr_shuffled': torch.cat(acc[f'{regime}_psnr_shuf']).mean().item(),
            'action_spread': torch.cat(spread[regime]).mean().item(),
        }
    res['two_frame'] = {k: torch.cat(v).mean().item() for k, v in two.items()}
    res['two_frame']['gain_over_copy_db'] = res['two_frame']['true'] - res['two_frame']['copy']
    res['two_frame']['shuffle_loss_db'] = res['two_frame']['true'] - res['two_frame']['shuf']
    return res, seeds, two_rows


def plot_two_frame(label, rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.7 * len(rows)), squeeze=False)
    for r, (f0, f1, rec, code, p_rec, p_copy) in enumerate(rows):
        for a, im, t in zip(axes[r], [f0, f1, rec], [f'frame 0 (copy {p_copy:.1f} dB)', 'frame 1 (target)',
                                                    f'rebuilt, action {code}: {p_rec:.1f} dB']):
            a.imshow(im.permute(1, 2, 0).numpy(), interpolation='nearest'); a.set_title(t, fontsize=8); a.axis('off')
    fig.suptitle(f'{label}: frame 1 rebuilt from frame 0 + action', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


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
        res, seeds, two_rows = evaluate(label, ckpt, args, windows, h5['frames'], device)
        stem = os.path.join(args.out_dir, f'{args.prefix}{label}')
        with open(stem + '.json', 'w') as f:
            json.dump(res, f, indent=1)
        plot(res, seeds, stem + '.png')
        plot_two_frame(label, two_rows, stem + '_twoframe.png')
        print(f'== {label} ({time.time() - t0:.0f}s) -> {stem}.json/.png')
        rows.append(res)

    print(f"\n{'model':<8}{'H nats':>8}{'used':>6}{'top code':>10}{'sat bits':>18}{'gap mask':>10}{'gap full':>10}"
          f"{'spread m':>10}{'NMI':>7}{'NMI pl':>8}{'R2 lat':>8}{'2f dB':>8}{'vs copy':>9}{'shuf -dB':>9}")
    for r in rows:
        print(f"{r['label']:<8}{r['entropy_nats']:>8.3f}{r['codes_used_1pct']:>6}{max(r['code_freq']):>10.3f}"
              f"{' '.join(f'{s:.2f}' for s in r['bit_saturated']):>18}{r['masked']['shuffle_gap_rel']:>10.4f}"
              f"{r['full']['shuffle_gap_rel']:>10.4f}{r['masked']['action_spread']:>10.4f}{r['nmi_code_motion']:>7.3f}{r['nmi_code_player']:>8.3f}{r['r2_player_latent']:>8.3f}"
              f"{r['two_frame']['true']:>8.2f}{r['two_frame']['gain_over_copy_db']:>+9.2f}{r['two_frame']['shuffle_loss_db']:>9.2f}")


if __name__ == '__main__':
    main()
