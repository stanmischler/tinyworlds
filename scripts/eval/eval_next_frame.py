"""One-step next-frame evaluation on the held-out test split.

Protocol (fixed, deterministic):
  - windows of `context` real frames + 1 target, `frame_skip` stored frames apart (as in training),
    starting every `sample_stride` frames inside each test block of the test .h5
  - the dynamics model predicts the fully masked target frame from the real context (teacher forced),
    greedy decoding (temperature 0) over `num_steps` MaskGIT iterations
  - actions: `lam`   = latent action model run on context + target, so the true transition is given
             `random`= LAM actions for the context transitions, a seeded random action for the target
             `none`  = the dynamics model's learned null action on every transition (zeros if it has none);
                       for models trained with action_dropout_prob > 0
    the LAM code of the target transition is recorded per window in every mode (`lam_code`)
  - reported next to two references: copy-last-context-frame baseline, and the tokenizer's own
    reconstruction of the target (the ceiling any token prediction can reach)
  - metrics per window, then mean / per-block mean: PSNR, SSIM, token accuracy, LPIPS if the
    `lpips` package is installed

Outputs: <out_dir>/<name>.json (all numbers + the source indices of every window) and
<out_dir>/<name>.png (8 seed windows: context, target, prediction, copy baseline).

Usage (from the repo root, PYTHONPATH=$PWD):
    python scripts/eval/eval_next_frame.py --run-dir results/sonic_short_2026_09_22 --action-mode lam
    python scripts/eval/eval_next_frame.py --run-dir results/... --action-mode random --limit 64   # smoke
    # another game: point --test-h5 at its split (same protocol; frame_skip 4 = 60 // 15 fps for zelda as for sonic)
    python scripts/eval/eval_next_frame.py --run-dir results/zelda_v3_... --test-h5 data/zelda_test_frames.h5 --action-mode lam
"""

import argparse
import json
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from utils.inference_utils import load_models
from utils.utils import find_latest_checkpoint


# ----------------------------------------------------------------------------- metrics
def psnr(pred, target):
    # pred, target: [B, C, H, W] in [0, 1] -> [B]
    mse = ((pred - target) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


def _gaussian_window(size=11, sigma=1.5, device='cpu'):
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])  # [size, size]


def ssim(pred, target, window_size=11):
    # standard SSIM (Wang et al.), gaussian window, per channel, mean over image -> [B]
    B, C, H, W = pred.shape
    w = _gaussian_window(window_size, device=pred.device).expand(C, 1, window_size, window_size)
    pad = window_size // 2
    mu_p = F.conv2d(pred, w, padding=pad, groups=C)
    mu_t = F.conv2d(target, w, padding=pad, groups=C)
    sigma_p = F.conv2d(pred * pred, w, padding=pad, groups=C) - mu_p ** 2
    sigma_t = F.conv2d(target * target, w, padding=pad, groups=C) - mu_t ** 2
    sigma_pt = F.conv2d(pred * target, w, padding=pad, groups=C) - mu_p * mu_t
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)) / ((mu_p ** 2 + mu_t ** 2 + c1) * (sigma_p + sigma_t + c2))
    return s.flatten(1).mean(1)


def try_lpips(device):
    try:
        import lpips  # optional dependency
        return lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    except Exception:
        return None


# ----------------------------------------------------------------------------- data
def test_windows(h5_path, context, frame_skip, sample_stride):
    """Deterministic list of (block_id, local_start) windows; the target is local_start + context*frame_skip."""
    with h5py.File(h5_path, 'r') as f:
        blocks = json.loads(f.attrs['test_blocks_local'])
    span = context * frame_skip  # index offset of the target from the window start
    windows = []
    for block_id, (start, end) in enumerate(blocks):
        for s in range(start, end - span, sample_stride):
            windows.append((block_id, s))
    return windows


def load_window_batch(frames_dset, windows, context, frame_skip):
    # -> uint8 [B, T=context+1, H, W, C]
    idx = np.array([[s + k * frame_skip for k in range(context + 1)] for _, s in windows])
    out = np.stack([frames_dset[list(row)] for row in idx])
    return out


def to_model_range(frames_u8, device):
    # uint8 [B, T, H, W, C] -> float [-1, 1] [B, T, C, H, W], same as the training transform
    x = torch.from_numpy(frames_u8).to(device).permute(0, 1, 4, 2, 3).float() / 255.0
    return x * 2 - 1


def to_unit(x):
    return ((x + 1) / 2).clamp(0, 1)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--test-h5', default='data/sonic_test_frames.h5')
    p.add_argument('--run-dir', help='results dir holding the three stage checkpoints (latest step of each is used)')
    p.add_argument('--video-tokenizer-path'); p.add_argument('--latent-actions-path'); p.add_argument('--dynamics-path')
    p.add_argument('--action-mode', choices=['lam', 'random', 'none'], default='lam')
    p.add_argument('--context', type=int, default=3, help='real context frames; training used sequences of context+1')
    p.add_argument('--frame-skip', type=int, default=4, help='stored frames between sequence frames (60 // fps in the loader)')
    p.add_argument('--sample-stride', type=int, default=8, help='stored frames between window starts inside a block')
    p.add_argument('--num-steps', type=int, default=10, help='MaskGIT unmasking iterations, or Euler steps for a flow dynamics model')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--decode', choices=['context', 'alone'], default='context',
                   help='context: tokenize/detokenize the target with its context frames, as in training (the tokenizer is '
                        'temporal); alone: the target frame on its own (protocol before 2026-10-02)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    p.add_argument('--seed', type=int, default=0, help='seed for the random actions')
    p.add_argument('--n-seeds', type=int, default=8, help='windows shown in the PNG: first window of the first n blocks')
    p.add_argument('--limit', type=int, help='evaluate only the first N windows (smoke test)')
    p.add_argument('--out-dir', default='eval_results')
    p.add_argument('--name', help='output file stem; default <run-dir basename>_<action-mode>')
    args = p.parse_args()

    if args.run_dir:  # explicit paths win (a dynamics-only run dir holds no tokenizer / LAM)
        for stage in ('video_tokenizer', 'latent_actions', 'dynamics'):
            if not getattr(args, f'{stage}_path'):
                setattr(args, f'{stage}_path', find_latest_checkpoint('.', stage, run_root_dir=args.run_dir))
    for k in ('video_tokenizer_path', 'latent_actions_path', 'dynamics_path'):
        assert getattr(args, k) and os.path.exists(getattr(args, k)), f'{k} missing: {getattr(args, k)}'
    name = args.name or f"{os.path.basename(os.path.normpath(args.run_dir or os.path.dirname(args.dynamics_path)))}_{args.action_mode}"
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tok, lam, dyn = load_models(args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path, device, use_actions=True)
    lpips_fn = try_lpips(device)
    n_actions = lam.quantizer.codebook_size
    rng = torch.Generator(device='cpu').manual_seed(args.seed)

    windows = test_windows(args.test_h5, args.context, args.frame_skip, args.sample_stride)
    if args.limit:
        windows = windows[:args.limit]
    h5 = h5py.File(args.test_h5, 'r')
    frames_dset, source_index = h5['frames'], h5['source_index'][:]
    n_blocks = len(json.loads(h5.attrs['test_blocks_local']))
    print(f'{len(windows)} windows from {n_blocks} blocks of {args.test_h5}; checkpoints:')
    for k in ('video_tokenizer_path', 'latent_actions_path', 'dynamics_path'):
        print(f'  {getattr(args, k)}')

    def idx_to_latents(idx):
        return tok.quantizer.get_latents_from_indices(idx, dim=-1)

    per = {k: [] for k in ('block', 'source_start', 'psnr', 'ssim', 'lpips', 'token_acc',
                           'copy_psnr', 'copy_ssim', 'copy_lpips', 'copy_token_acc',
                           'recon_psnr', 'recon_ssim', 'recon_lpips', 'action', 'lam_code')}
    seed_rows = []  # (context frames, target, prediction, copy) for the PNG
    seed_starts = {}
    for block_id, s in windows:
        seed_starts.setdefault(block_id, s)
    seed_set = {(b, s) for b, s in seed_starts.items() if b < args.n_seeds}

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(windows), args.batch_size):
            batch = windows[i:i + args.batch_size]
            x = to_model_range(load_window_batch(frames_dset, batch, args.context, args.frame_skip), device)  # [B, T, C, H, W]
            context, target = x[:, :args.context], x[:, args.context:]  # [B, Tc, C, H, W], [B, 1, C, H, W]
            B = x.shape[0]

            if args.decode == 'context':
                full_idx = tok.tokenize(x)  # [B, T, P] causal encoder: context codes are the same as tokenizing them alone
                ctx_idx, target_idx = full_idx[:, :args.context], full_idx[:, args.context:]  # [B, Tc, P], [B, 1, P]
            else:
                ctx_idx = tok.tokenize(context)  # [B, Tc, P]
                target_idx = tok.tokenize(target)  # [B, 1, P]
            ctx_lat = idx_to_latents(ctx_idx)  # [B, Tc, P, L]

            lam_cond = lam.encode(x)  # [B, T-1, A]: every transition incl. the one into the target
            lam_code = lam.quantizer.get_indices_from_latents(lam_cond[:, -1])  # [B] true code of the target transition
            if args.action_mode == 'lam':
                cond, last_action = lam_cond, lam_code
            elif args.action_mode == 'none':
                # the model's learned null action on every transition, as dropped samples saw in training
                null = dyn.null_action if getattr(dyn, 'null_action', None) is not None else torch.zeros(1, 1, lam_cond.shape[-1], device=device)
                cond = null.to(lam_cond.dtype).expand(B, lam_cond.shape[1], -1)  # [B, T-1, A]
                last_action = torch.full((B,), -1, device=device)  # [B] no action fed
            else:
                ctx_actions = lam.encode(context)  # [B, Tc-1, A]
                last_action = torch.randint(0, n_actions, (B,), generator=rng).to(device)  # [B]
                rand_lat = lam.quantizer.get_latents_from_indices(last_action)[:, None]  # [B, 1, A]
                cond = torch.cat([ctx_actions, rand_lat], dim=1)  # [B, T-1, A]

            pred_lat = dyn.forward_inference(ctx_lat, prediction_horizon=1, num_steps=args.num_steps,
                                             index_to_latents_fn=idx_to_latents, conditioning=cond,
                                             temperature=args.temperature)  # [B, T, P, L]
            pred_idx = tok.quantizer.get_indices_from_latents(pred_lat[:, -1:], dim=-1)  # [B, 1, P]
            if args.decode == 'context':  # decode context + target together, keep the target
                pred = to_unit(tok.detokenize(pred_lat)[:, -1])  # [B, C, H, W]
                recon = to_unit(tok.detokenize(idx_to_latents(full_idx))[:, -1])  # tokenizer ceiling
            else:
                pred = to_unit(tok.detokenize(pred_lat[:, -1:])[:, 0])  # [B, C, H, W]
                recon = to_unit(tok.detokenize(idx_to_latents(target_idx))[:, 0])  # tokenizer ceiling
            tgt = to_unit(target[:, 0])
            copy = to_unit(context[:, -1])

            def lp(a, b):
                return lpips_fn(a * 2 - 1, b * 2 - 1).flatten() if lpips_fn is not None else torch.full((B,), float('nan'))

            per['block'] += [b for b, _ in batch]
            per['source_start'] += [int(source_index[s]) for _, s in batch]
            per['action'] += last_action.tolist(); per['lam_code'] += lam_code.tolist()
            per['psnr'] += psnr(pred, tgt).tolist(); per['ssim'] += ssim(pred, tgt).tolist(); per['lpips'] += lp(pred, tgt).tolist()
            per['token_acc'] += (pred_idx == target_idx).float().flatten(1).mean(1).tolist()
            per['copy_psnr'] += psnr(copy, tgt).tolist(); per['copy_ssim'] += ssim(copy, tgt).tolist(); per['copy_lpips'] += lp(copy, tgt).tolist()
            per['copy_token_acc'] += (ctx_idx[:, -1:] == target_idx).float().flatten(1).mean(1).tolist()
            per['recon_psnr'] += psnr(recon, tgt).tolist(); per['recon_ssim'] += ssim(recon, tgt).tolist(); per['recon_lpips'] += lp(recon, tgt).tolist()

            for j, w in enumerate(batch):
                if w in seed_set:
                    seed_rows.append((w, to_unit(context[j]).cpu(), tgt[j].cpu(), pred[j].cpu(), copy[j].cpu(), int(last_action[j])))
            print(f'  {min(i + B, len(windows))}/{len(windows)}  psnr {np.mean(per["psnr"]):.2f}  copy {np.mean(per["copy_psnr"]):.2f}  '
                  f'token_acc {np.mean(per["token_acc"]):.3f}  ({time.time() - t0:.0f}s)', flush=True)

    # ------------------------------------------------------------------ aggregate
    def mean(key, mask=None):
        v = np.array(per[key], dtype=np.float64)
        if mask is not None:
            v = v[mask]
        return None if len(v) == 0 or np.all(np.isnan(v)) else float(np.nanmean(v))

    groups = {'model': '', 'copy_baseline': 'copy_', 'tokenizer_recon': 'recon_'}
    metric_names = {'': ['psnr', 'ssim', 'lpips', 'token_acc'], 'copy_': ['psnr', 'ssim', 'lpips', 'token_acc'], 'recon_': ['psnr', 'ssim', 'lpips']}
    blocks_arr = np.array(per['block'])
    summary = {g: {m: mean(pre + m) for m in metric_names[pre]} for g, pre in groups.items()}
    per_block = {int(b): {g: {m: mean(pre + m, blocks_arr == b) for m in metric_names[pre]} for g, pre in groups.items()}
                 for b in sorted(set(per['block']))}
    result = {
        'name': name, 'n_windows': len(windows), 'test_h5': args.test_h5,
        'config': {k: v for k, v in vars(args).items() if k not in ('out_dir', 'name')},
        'summary': summary, 'per_block': per_block,
        'windows': {k: per[k] for k in ('block', 'source_start', 'action', 'lam_code', 'psnr', 'ssim', 'token_acc',
                                        'copy_psnr', 'copy_ssim', 'copy_token_acc')},
    }
    json_path = os.path.join(args.out_dir, f'{name}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=1)

    # ------------------------------------------------------------------ PNG of the seed windows
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seed_rows.sort(key=lambda r: r[0])
    if seed_rows:
        cols = args.context + 3
        fig, axes = plt.subplots(len(seed_rows), cols, figsize=(1.6 * cols, 1.7 * len(seed_rows)))
        axes = np.atleast_2d(axes)
        for r, (w, ctx, tgt_i, pred_i, copy_i, act) in enumerate(seed_rows):
            imgs = [ctx[k] for k in range(args.context)] + [tgt_i, pred_i, copy_i]
            titles = [f'ctx {k + 1}' for k in range(args.context)] + ['target', f'pred (a={"null" if act < 0 else act})', 'copy-last']
            for c, (img, title) in enumerate(zip(imgs, titles)):
                ax = axes[r, c]
                ax.imshow(img.permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=8)
                if c == 0:
                    ax.set_ylabel(f'block {w[0]}\nsrc {source_index[w[1]]}', fontsize=7)
        s = summary
        fig.suptitle(f"{name}: PSNR {s['model']['psnr']:.2f} (copy {s['copy_baseline']['psnr']:.2f}, recon {s['tokenizer_recon']['psnr']:.2f})  "
                     f"token acc {s['model']['token_acc']:.3f} (copy {s['copy_baseline']['token_acc']:.3f})", fontsize=9)
        plt.tight_layout()
        png_path = os.path.join(args.out_dir, f'{name}.png')
        plt.savefig(png_path, dpi=130); plt.close(fig)
    else:
        png_path = None

    print('\nsummary:')
    for g, vals in summary.items():
        print(f'  {g:16s} ' + '  '.join(f'{m} {v:.4f}' for m, v in vals.items() if v is not None))
    print(f'saved {json_path}' + (f' and {png_path}' if png_path else ''))


if __name__ == '__main__':
    main()
