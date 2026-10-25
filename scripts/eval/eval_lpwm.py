"""OBJ3D video prediction with LPWM's protocol (Latent Particle World Models, Tab. 8; details and their file:line in
evaluation/lpwm.py): one clip per test episode (frames 0..49 of each of the 200), 6 real context frames -> 44 predicted,
autoregressive, no future information; PSNR / SSIM / LPIPS-VGG (piqa) on [0, 1], mean over all 8800 predicted frames.
Their numbers: LPWM 31.45 / 0.926 / 0.081, DDLP 31.29 / 0.923 / 0.088, G-SWM 31.7 / 0.924 / 0.118, SlotFormer/OCVP
31.2 / 0.925 / 0.135 (PSNR / SSIM / LPIPS).

Our model: evaluation/rollout.py, one frame at a time, sliding context window of --max-context (6, LPWM's cond_steps)
frames, greedy (temperature 0, our analogue of their deterministic default). Context frames are not scored.
References: copy_last (frame 5 repeated) and tokenizer_recon (the real frames encoded + decoded: the ceiling).

Actions hook (--latent-actions-path, off by default): a CoMo action dir / latent-action checkpoint whose
encode(frames [B, T, C, H, W] in [-1, 1]) -> [B, T - 1, A] is run on the REAL 50 frames, future ones included, and
the actions are fed to the rollout. This is the "learned actions from ground truth" arm: it uses future information,
so it is NOT comparable to LPWM's action-free Tab. 8 row (closer to their --ctx mode, see the open questions).

Open questions (not settled by LPWM's code; do not read these numbers as exactly their protocol on these points):
  - Tab. 8 deterministic or --sample? eval_gen_metrics.py defaults to deterministic (deterministic = not args.sample,
    :453) but the README example command passes --sample (README.md:302, for sketchy). Not stated for OBJ3D.
  - --ctx (use_all_ctx: latent context inferred from the whole GT clip, future included) for Tab. 8? The README example
    uses it; configs/obj3d128.json has ctx_for_eval false (only used for the in-training val eval, train_lpwm.py:580).
  - whether the baseline rows (G-SWM, SlotFormer/OCVP, DDLP) were re-scored with this script or copied from the DDLP
    paper (whose protocol may differ, e.g. horizon / metric library).

Outputs <out-dir>/<name>.json (means, stds, per-step curves) and <name>.png (GT vs prediction at several steps for a
few clips, plus PSNR vs step).

Usage (repo root, PYTHONPATH=$PWD):
    python scripts/eval/eval_lpwm.py --copy-only                       # references without a model
    python scripts/eval/eval_lpwm.py --run-dir results/<run>           # tokenizer + dynamics, latest step of each
    python scripts/eval/eval_lpwm.py --run-dir results/<run> --latent-actions-path <action dir>   # GT-actions arm
    python scripts/eval/eval_lpwm.py --run-dir results/<run> --limit 4  # smoke
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from evaluation.lpwm import HORIZON, N_CONTEXT, LPWMMetrics, load_clips, lpwm_clips, to_unit
from evaluation.rollout import rollout
from utils.utils import (find_latest_checkpoint, load_dynamics_from_checkpoint, load_latent_actions_from_checkpoint,
                         load_videotokenizer_from_checkpoint)
from utils.inference_utils import default_num_steps

SHOW_STEPS = (6, 10, 20, 30, 40, 49)  # frame indices in the PNG (6 = first predicted)


@torch.no_grad()
def gt_actions(lam, x, batch_size):
    # x: float [B, T, C, H, W] in [-1, 1] (real frames, future included) -> [B, T - 1, A]
    return torch.cat([lam.encode(x[i:i + batch_size]).float() for i in range(0, len(x), batch_size)])


def save_png(path, gt, rows, clips, curves, title, n_show=4):
    """gt: [B, T, C, H, W] in [0, 1]; rows: {label: [B, T, C, H, W]}; curves: {label: per-step psnr list}."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = min(n_show, len(gt))
    n_rows = n * (1 + len(rows))
    fig = plt.figure(figsize=(1.5 * len(SHOW_STEPS), 1.5 * n_rows + 3))
    gs = fig.add_gridspec(n_rows + 2, len(SHOW_STEPS))
    r = 0
    for b in range(n):
        for label, x in [('GT', gt)] + list(rows.items()):
            for c, t in enumerate(SHOW_STEPS):
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(x[b, t].permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f'frame {t}', fontsize=8)
                if c == 0:
                    ax.set_ylabel(f'ep {clips[b][0]}\n{label}', fontsize=7)
            r += 1
    ax = fig.add_subplot(gs[n_rows:, :])
    for label, ys in curves.items():
        ax.plot(range(N_CONTEXT, N_CONTEXT + len(ys)), ys, label=label)
    ax.set_xlabel('frame index'); ax.set_ylabel('PSNR (dB)'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--test-h5', default='data/obj3d_test_frames.h5')
    p.add_argument('--copy-only', action='store_true', help='references only (copy_last), no model')
    p.add_argument('--run-dir', help='results dir holding the tokenizer + dynamics checkpoints (latest step of each)')
    p.add_argument('--video-tokenizer-path'); p.add_argument('--dynamics-path')
    p.add_argument('--latent-actions-path', help='actions hook: per-transition actions encoded from the real frames (uses the future)')
    p.add_argument('--n-context', type=int, default=N_CONTEXT)
    p.add_argument('--horizon', type=int, default=HORIZON)
    p.add_argument('--max-context', type=int, default=N_CONTEXT, help='dynamics context window during the rollout')
    p.add_argument('--num-steps', type=int, help='Euler steps (flow, default 32) or unmasking iterations (MaskGIT, default 10) per frame')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    p.add_argument('--limit', type=int, help='first N clips only (smoke test)')
    p.add_argument('--out-dir', default='eval_results')
    p.add_argument('--name')
    args = p.parse_args()

    if args.run_dir:  # explicit paths win
        for stage in ('video_tokenizer', 'dynamics'):
            if not getattr(args, f'{stage}_path'):
                setattr(args, f'{stage}_path', find_latest_checkpoint('.', stage, run_root_dir=args.run_dir))
    has_model = not args.copy_only
    if has_model:
        for k in ('video_tokenizer_path', 'dynamics_path'):
            assert getattr(args, k) and os.path.exists(getattr(args, k)), f'{k} missing: {getattr(args, k)} (or pass --copy-only)'
    stem = 'references' if not has_model else os.path.basename(os.path.normpath(args.run_dir or os.path.dirname(args.dynamics_path)))
    name = args.name or f"{stem}{'_gtactions' if args.latent_actions_path else ''}_obj3d_lpwm"
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()

    clips = lpwm_clips(args.test_h5, args.horizon)
    if args.limit:
        clips = clips[:args.limit]
    gt = to_unit(load_clips(args.test_h5, clips, args.horizon))  # [B, T, C, H, W] in [0, 1]
    print(f'{len(clips)} clips x {args.horizon} frames from {args.test_h5}, context {args.n_context}')
    metrics = LPWMMetrics(device)
    target = gt[:, args.n_context:]  # [B, S, C, H, W]
    tables = {'copy_last': metrics.summarise(gt[:, args.n_context - 1:args.n_context].expand_as(target), target)}

    pred = None
    if has_model:
        print(f'checkpoints:\n  {args.video_tokenizer_path}\n  {args.dynamics_path}')
        tok, _ = load_videotokenizer_from_checkpoint(args.video_tokenizer_path, device)
        dyn, _ = load_dynamics_from_checkpoint(args.dynamics_path, device)  # flow: EMA weights when saved
        args.num_steps = default_num_steps(dyn) if args.num_steps is None else args.num_steps
        tok.eval(); dyn.eval()
        x = gt.to(device) * 2 - 1  # [B, T, C, H, W] in [-1, 1], the training transform
        actions = None
        if args.latent_actions_path:  # ---- actions hook: learned actions from the GT clip (future frames included)
            lam, _ = load_latent_actions_from_checkpoint(args.latent_actions_path, device)
            lam.eval()
            actions = gt_actions(lam, x, args.batch_size)  # [B, T - 1, A]
            print(f'actions from {args.latent_actions_path}: {tuple(actions.shape)}')
        pred, recon = rollout(tok, dyn, x, args.n_context, actions=actions, max_context=args.max_context,
                              num_steps=args.num_steps, temperature=args.temperature, batch_size=args.batch_size)
        tables['model'] = metrics.summarise(pred[:, args.n_context:], target)
        tables['tokenizer_recon'] = metrics.summarise(recon[:, args.n_context:], target)

    result = {'name': name, 'protocol': 'LPWM Tab. 8 (OBJ3D test, 1 clip per episode from frame 0, 6 -> 44, piqa)',
              'n_clips': len(clips), 'n_predicted_frames': int(target.shape[0] * target.shape[1]),
              'config': {k: v for k, v in vars(args).items() if k not in ('out_dir', 'name')},
              'tables': tables, 'seconds': round(time.time() - t0, 1)}
    json_path = os.path.join(args.out_dir, f'{name}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=1)

    rows = {'model': pred} if pred is not None else {'copy last': gt[:, args.n_context - 1:args.n_context].expand_as(gt)}
    head = tables.get('model', tables['copy_last'])
    save_png(os.path.join(args.out_dir, f'{name}.png'), gt, rows, clips,
             {k: v['per_step']['psnr'] for k, v in tables.items()},
             f"{name}: PSNR {head['psnr']:.2f}  SSIM {head['ssim']:.3f}  LPIPS {head['lpips']:.3f}  (LPWM 31.45 / 0.926 / 0.081)")

    print()
    for g, v in tables.items():
        ps = v['per_step']['psnr']
        steps = '  '.join(f"t{args.n_context + i}: {ps[i]:.2f}" for i in (0, 4, 14, 24, 34, len(ps) - 1) if i < len(ps))
        print(f"  {g:16s} PSNR {v['psnr']:.2f}  SSIM {v['ssim']:.4f}  LPIPS {v['lpips']:.4f}   ({steps})")
    print(f'saved {json_path} ({result["seconds"]}s)')


if __name__ == '__main__':
    main()
