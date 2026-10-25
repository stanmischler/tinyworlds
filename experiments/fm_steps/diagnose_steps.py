"""Why does a noise-start flow-matching dynamics model get its best next-frame PSNR at 1 Euler step?

Hypothesis: 1 step returns x0 + v(x0, t=0) = E[x1 | context] (the conditional mean, PSNR-optimal), many steps return a
sample of p(x1 | context), whose squared error is that of the mean plus the spread of the samples (up to 2x).

On a few held-out windows, K noise seeds per window, the same start noise for every solver / step count:
  - Euler at N = 1, 2, 4, 8, 16, 32 steps and Heun (2 calls per step, Euler on the last)
  - per setting: mean PSNR of single samples, PSNR of the pixel average of the K samples, best-of-K PSNR, the
    error split  E_k||x_k - y||^2 = ||mean_k x_k - y||^2 + E_k||x_k - mean_k x_k||^2  (exact over the K samples),
    and the PSNR of the average to the 1-step output
  - per window: copy-last PSNR (low = Link / the screen moves) next to the 1-step vs N-step gap

Outputs <out-dir>/<name>.json and <name>.png (one row per window: target, 1 step, 3 samples at 32 steps, their average).

    python experiments/fm_steps/diagnose_steps.py --dynamics-path results/zelda_longer_fm/dynamics/checkpoints/dynamics_step_99999 \
      --video-tokenizer-path results/zelda_cont_final_tp/video_tokenizer/checkpoints/video_tokenizer_step_29000 \
      --latent-actions-path results/como_actions_v1_full --test-h5 data/zelda_test_frames.h5 --name longer_fm
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

from evaluation.windows import test_windows, load_window_batch, to_model_range, to_unit
from models.flow_dynamics import shift_noise_level
from utils.inference_utils import load_models


def noise_grid(num_steps, shift, device):
    # noise levels s_0 = 1 > ... > s_N = 0 on the shifted grid used by FlowDynamicsModel.sample_next  [N+1]
    return shift_noise_level(torch.linspace(1, 0, num_steps + 1, device=device), shift)


def velocity(dyn, ctx_lat, x, cond, s):
    # ctx_lat: [B, Tc, P, L] clean context; x: [B, P, L] noisy target at noise level s (float) -> v [B, P, L]
    B, Tc = ctx_lat.shape[:2]
    t = torch.ones(B, Tc + 1, device=x.device)  # context is clean (t = 1)
    t[:, -1] = 1 - s  # flow time of the target
    inp = torch.cat([ctx_lat.float(), x[:, None]], dim=1)  # [B, Tc+1, P, L]
    return dyn.velocity(inp, cond, t).float()


@torch.no_grad()
def sample(dyn, ctx_lat, cond, x0, num_steps, solver):
    # Euler or Heun (trapezoid) integration from the start noise x0 [B, P, L] to the clean target latents [B, P, L]
    s_grid = noise_grid(num_steps, dyn.fm_shift, x0.device)
    x = x0.clone()
    for k in range(num_steps):
        s, s_next = float(s_grid[k]), float(s_grid[k + 1])
        dt = s - s_next  # step in flow time t = 1 - s
        v = velocity(dyn, ctx_lat, x, cond, s)
        if solver == "heun" and s_next > 0:  # last step stays Euler (as in EDM): at s = 0 the learned v is x - E[x0] = x, not the path's slope
            x_euler = x + dt * v  # predictor
            v_next = velocity(dyn, ctx_lat, x_euler, cond, s_next)
            v = (v + v_next) / 2  # corrector: average slope
        x = x + dt * v
    return x


def mse_per_window(a, b):
    # a, b: [..., C, H, W] in [0, 1] -> mean squared error over C, H, W  [...]
    return ((a - b) ** 2).flatten(-3).mean(-1)


def to_psnr(mse):
    return 10 * np.log10(1.0 / np.maximum(mse, 1e-10))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--test-h5', default='data/zelda_test_frames.h5')
    p.add_argument('--video-tokenizer-path', required=True); p.add_argument('--latent-actions-path', required=True)
    p.add_argument('--dynamics-path', required=True)
    p.add_argument('--context', type=int, default=3); p.add_argument('--frame-skip', type=int, default=4)
    p.add_argument('--sample-stride', type=int, default=8)
    p.add_argument('--n-windows', type=int, default=32, help='windows spread evenly over the eval protocol windows')
    p.add_argument('--n-seeds', type=int, default=8, help='K samples per window')
    p.add_argument('--euler-steps', default='1,2,4,8,16,32')
    p.add_argument('--heun-steps', default='2,3,5,9,17', help='Heun steps (2 network calls each, the last one 1)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps')
    p.add_argument('--out-dir', default='eval_results/fm_steps'); p.add_argument('--name', required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)
    tok, lam, dyn = load_models(args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path, dev, use_actions=True)
    assert dyn.fm_source == "noise" and not dyn.correction, 'meant for a plain noise-start flow model'

    all_windows = test_windows(args.test_h5, args.context, args.frame_skip, args.sample_stride)
    pick = np.linspace(0, len(all_windows) - 1, args.n_windows).round().astype(int)
    windows = [all_windows[i] for i in pick]
    with h5py.File(args.test_h5, 'r') as h5:
        x = to_model_range(load_window_batch(h5['frames'], windows, args.context, args.frame_skip), dev)  # [W, T, C, H, W]
    W = x.shape[0]

    with torch.no_grad():
        full_lat = tok.encode(x)  # [W, T, P, L]
        ctx_lat = full_lat[:, :args.context]  # [W, Tc, P, L]
        cond = lam.encode(x)  # [W, T-1, A] true actions, incl. the transition into the target
        target = to_unit(x[:, -1])  # [W, C, H, W]
        copy = to_unit(x[:, args.context - 1])  # [W, C, H, W]
        recon = to_unit(tok.decode(full_lat)[:, -1])  # [W, C, H, W] tokenizer ceiling

        def decode(pred_lat):  # [W, P, L] -> [W, C, H, W], decoded with the context as in eval_next_frame
            return to_unit(tok.decode(torch.cat([ctx_lat, pred_lat[:, None].to(ctx_lat.dtype)], dim=1))[:, -1])

        gens = [torch.Generator(device=dev).manual_seed(k) for k in range(args.n_seeds)]
        x0s = [torch.randn(ctx_lat[:, 0].shape, device=dev, generator=g) for g in gens]  # K x [W, P, L]

        settings = [('euler', int(n)) for n in args.euler_steps.split(',')] + [('heun', int(n)) for n in args.heun_steps.split(',')]
        frames, t0 = {}, time.time()
        for solver, n in settings:
            frames[(solver, n)] = torch.stack([decode(sample(dyn, ctx_lat, cond, x0, n, solver)) for x0 in x0s]).cpu()  # [K, W, C, H, W]
            print(f'{solver} {n}: done ({time.time() - t0:.0f}s)', flush=True)

    target, copy, recon = target.cpu(), copy.cpu(), recon.cpu()
    one_step = frames[('euler', 1)]  # [K, W, C, H, W]
    one_step_mean = one_step.mean(0)  # [W, C, H, W]
    rows = []
    for (solver, n), f in frames.items():
        avg = f.mean(0)  # [W, C, H, W] pixel average of the K samples
        sample_mse = mse_per_window(f, target[None]).numpy()  # [K, W]
        avg_mse = mse_per_window(avg, target).numpy()  # [W]
        spread = mse_per_window(f, avg[None]).mean(0).numpy()  # [W]   sample_mse.mean(0) = avg_mse + spread (exact)
        rows.append({
            'solver': solver, 'steps': n, 'nfe': 2 * n - 1 if solver == 'heun' else n,
            'sample_psnr': float(to_psnr(sample_mse).mean()),  # mean over K and windows of single-sample PSNR (as the eval)
            'avg_psnr': float(to_psnr(avg_mse).mean()),
            'best_of_k_psnr': float(to_psnr(sample_mse.min(0)).mean()),
            'sample_mse': float(sample_mse.mean()), 'avg_mse': float(avg_mse.mean()), 'spread': float(spread.mean()),
            'avg_vs_1step_psnr': float(to_psnr(mse_per_window(avg, one_step_mean).numpy()).mean()),
            'per_window_sample_psnr': to_psnr(sample_mse).mean(0).tolist(), 'per_window_spread': spread.tolist(),
        })
    result = {
        'name': args.name, 'dynamics_path': args.dynamics_path, 'fm_shift': dyn.fm_shift, 'n_windows': W, 'n_seeds': args.n_seeds,
        'windows': [[int(b), int(s)] for b, s in windows],
        'copy_psnr': to_psnr(mse_per_window(copy, target).numpy()).tolist(),
        'recon_psnr': float(to_psnr(mse_per_window(recon, target).numpy()).mean()),
        'rows': rows,
    }
    with open(os.path.join(args.out_dir, f'{args.name}.json'), 'w') as fh:
        json.dump(result, fh, indent=1)

    print(f"\n{args.name}: {W} windows x {args.n_seeds} seeds, recon {result['recon_psnr']:.2f}, "
          f"copy {np.mean(result['copy_psnr']):.2f}")
    print(f"{'solver':6s} {'N':>3s} {'NFE':>3s} | {'sample':>6s} {'avg':>6s} {'best':>6s} | {'mse':>8s} = {'avg_mse':>8s} + {'spread':>8s} | avg~1step")
    for r in rows:
        print(f"{r['solver']:6s} {r['steps']:3d} {r['nfe']:3d} | {r['sample_psnr']:6.2f} {r['avg_psnr']:6.2f} {r['best_of_k_psnr']:6.2f} | "
              f"{r['sample_mse']:.2e} = {r['avg_mse']:.2e} + {r['spread']:.2e} | {r['avg_vs_1step_psnr']:.1f}")

    # one row per window (first 8): target, 1 step, 3 samples at the most Euler steps, their average
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_max = max(n for s, n in frames if s == 'euler')
    many = frames[('euler', n_max)]
    order = np.argsort(result['copy_psnr'])[:8]  # the 8 windows with the most motion
    cols = [('target', lambda w: target[w]), ('1 step', lambda w: one_step[0, w])] + \
           [(f'{n_max} steps, seed {k}', lambda w, k=k: many[k, w]) for k in range(min(3, args.n_seeds))] + \
           [(f'avg of {args.n_seeds} @ {n_max}', lambda w: many[:, w].mean(0))]
    fig, axes = plt.subplots(len(order), len(cols), figsize=(1.7 * len(cols), 1.8 * len(order)))
    for r, w in enumerate(order):
        for c, (title, get) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(get(w).permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
            if c == 0:
                ax.set_ylabel(f'copy {result["copy_psnr"][w]:.1f} dB', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f'{args.name}.png'), dpi=120); plt.close(fig)
    print(f'saved {args.out_dir}/{args.name}.json and .png')


if __name__ == '__main__':
    main()
