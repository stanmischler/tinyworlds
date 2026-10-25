"""NanoWM-protocol evaluation (arXiv 2605.23993, Tab. 7) on DINO-WM Point Maze / Wall (and Push-T), for a tinyworlds model
trained with ground-truth actions on scripts/data/convert_dinowm.py's <env>_frames.h5 (convert_pusht.py for pusht).

Protocol (evaluation/nanowm.py has the clip rule and the metric stack):
  - clips: 256 fixed val clips of NanoWM's val split (seed 42), span 20 env frames, frames s + {0, 5, 10, 15}
  - 1 real context frame -> 3 predicted frames, ground-truth 10-D actions (normalised as NanoWM for that env: Point Maze
    and Push-T z-scored with the train stats, Wall raw), autoregressive with evaluation/rollout.py
  - PSNR / SSIM / LPIPS-VGG / FID on the 3 predicted frames (768 frames), plus per predicted step
Scored at two resolutions against the native frame resized two ways:
  256: vs the native frame bilinear-stretched to 256 (NanoWM's ground truth, = the convert_dinowm.py training frames)
  128: vs the native frame INTER_AREA-resized to 128 (as convert_pusht.py's training set)
A model's prediction is scored as is at its own resolution, bicubic-upsampled above it and INTER_AREA-downsampled
below it (a 256 model: 256 direct and 128 downsampled; a 128 model: 128 direct and 256 upsampled, as eval_pusht.py).
References: copy_last (frame 0 repeated), tokenizer_recon (the 4 real frames encoded + decoded), resize_ceiling (256
only: the 128 ground truth bicubic-upsampled, the cap of a 128px model at 256), nanowm (if --nanowm-npz: NanoWM's own predictions from
experiments/pusht_nanowm/nanowm_rescore_{modal.py,local.sh}; use predictions_f16.npz).

Outputs <out_dir>/<name>.json and, with a model, <out_dir>/<name>.png (8 clips: context, targets, predictions).

Usage (repo root, PYTHONPATH=$PWD):
    python scripts/eval/eval_nanowm.py --env point_maze --copy-only                      # copy-last + resize ceiling
    python scripts/eval/eval_nanowm.py --env wall --run-dir results/<run> --nanowm-npz results/nanowm_wall_rescore/predictions_f16.npz
    python scripts/eval/eval_nanowm.py --env point_maze --run-dir results/<smoke> --limit 8 --no-fid   # smoke
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

from evaluation.nanowm import (ENVS, NUM_FRAMES, RESOLUTIONS, Metrics, area_resize, ground_truth, load_clips, nanowm_clips,
                               nanowm_val_order, print_tables, reference_tables, resize_bt, score_nanowm_npz, summarise,
                               to_chw01, upsample)
from evaluation.rollout import rollout
from utils.utils import find_latest_checkpoint, load_videotokenizer_from_checkpoint, load_dynamics_from_checkpoint
from utils.inference_utils import default_num_steps


def val_clips(h5, num_clips, seed):
    """-> clips [(traj_id, start)] in NanoWM order, n val slices, row_offset {traj_id: first row}, action key, mean, std."""
    ep = h5['episode_index'][:]  # [N]
    file_ids, first = np.unique(ep, return_index=True)
    file_ids = file_ids[np.argsort(first)]  # episodes in file order
    row_offset = dict(zip(file_ids.tolist(), np.sort(first).tolist()))
    length = dict(zip(file_ids.tolist(), [int(L) for L in h5['seq_lengths'][:]]))
    if 'val_traj_ids' in h5.attrs:  # convert_dinowm.py: val episodes already in NanoWM order (maybe truncated)
        order = [int(t) for t in h5.attrs['val_traj_ids'] if int(t) in length]
        mean, std = (h5.attrs['action_mean'], h5.attrs['action_std']) if h5.attrs['normalize_action'] else (None, None)
        key = 'actions'
    else:  # convert_pusht.py: the shipped val folder, split_ratio 0 (all of it, still shuffled)
        order = nanowm_val_order(len(file_ids), 0.0, seed).tolist()
        mean, std, key = h5.attrs['action_mean'], h5.attrs['action_std'], 'rel_actions'
    clips, n_slices = nanowm_clips(order, [length[t] for t in order], num_clips, seed)
    return clips, n_slices, row_offset, key, mean, std


def rescale(x, size):
    # [B, T, C, S, S] in [0, 1] -> [B, T, C, size, size]: bicubic up (evaluation.nanowm.upsample), INTER_AREA down
    if x.shape[-1] <= size:
        return upsample(x, size)
    return to_chw01(area_resize(x.permute(0, 1, 3, 4, 2).numpy(), size), 'cpu')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', required=True, choices=sorted(ENVS))
    p.add_argument('--val-h5', help='default data/<env>_val_frames.h5')
    p.add_argument('--run-dir', help='results dir holding the tokenizer + dynamics checkpoints (latest step of each)')
    p.add_argument('--video-tokenizer-path'); p.add_argument('--dynamics-path')
    p.add_argument('--copy-only', action='store_true', help='no model: copy-last + resize ceiling (+ --nanowm-npz) only')
    p.add_argument('--nanowm-npz', help="NanoWM's predictions (experiments/pusht_nanowm/nanowm_rescore_*) to score alongside")
    p.add_argument('--num-clips', type=int, default=256)
    p.add_argument('--seed', type=int, default=42, help="NanoWM's random_seed and validation_fixed_subset_seed")
    p.add_argument('--num-steps', type=int, help='Euler steps (flow, default 32) or unmasking iterations (MaskGIT, default 10) per frame')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    p.add_argument('--limit', type=int, help='first N clips only (smoke test; FID needs many)')
    p.add_argument('--no-fid', action='store_true')
    p.add_argument('--out-dir', default='eval_results')
    p.add_argument('--name')
    args = p.parse_args()

    spec = ENVS[args.env]
    args.val_h5 = args.val_h5 or spec['val_h5']
    has_model = not args.copy_only and bool(args.run_dir or args.dynamics_path)
    if has_model and args.run_dir:
        args.video_tokenizer_path = args.video_tokenizer_path or find_latest_checkpoint('.', 'video_tokenizer', run_root_dir=args.run_dir)
        args.dynamics_path = args.dynamics_path or find_latest_checkpoint('.', 'dynamics', run_root_dir=args.run_dir)
    tag = (os.path.basename(os.path.normpath(args.run_dir)) if has_model and args.run_dir else
           'model' if has_model else 'nanowm' if args.nanowm_npz else 'references')
    name = args.name or f'{tag}_{args.env}'
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    # ---- clips and ground truth
    h5 = h5py.File(args.val_h5, 'r')
    clips, n_slices, row_offset, key, mean, std = val_clips(h5, args.num_clips, args.seed)
    if args.limit:
        clips = clips[:args.limit]
    print(f'{args.env}: {len(clips)} clips (of {n_slices} val slices, seed {args.seed}) from {args.val_h5}')
    frames, actions = load_clips(h5, row_offset, clips, mean, std, action_key=key)  # [B, T, H, W, 3] native, [B, T-1, 10]
    gt = ground_truth(frames)  # {128: INTER_AREA, 256: NanoWM's float bilinear stretch}, each [B, T, C, S, S]
    with_fid = not args.no_fid
    metrics = Metrics(device, fid=with_fid)
    tables = reference_tables(metrics, gt, with_fid)

    result = {'name': name, 'env': args.env, 'n_clips': len(clips), 'n_val_slices': n_slices, 'val_h5': args.val_h5,
              'native_resolution': list(frames.shape[2:4]), 'published_nanowm_256': spec['published'],
              'clips': [[int(e), int(s)] for e, s in clips],
              'config': {k: v for k, v in vars(args).items() if k not in ('out_dir', 'name')}}

    # ---- our model
    pred = None
    if has_model:
        print(f'checkpoints:\n  {args.video_tokenizer_path}\n  {args.dynamics_path}')
        tok, tok_state = load_videotokenizer_from_checkpoint(args.video_tokenizer_path, device)
        dyn, _ = load_dynamics_from_checkpoint(args.dynamics_path, device)  # flow: EMA weights when saved
        args.num_steps = default_num_steps(dyn) if args.num_steps is None else args.num_steps
        tok.eval(); dyn.eval()
        size = int((tok_state.get('config') or {}).get('frame_size', 128))  # training resolution
        # model input = the training frames: 256 -> NanoWM's bilinear stretch (convert_dinowm.py default, uint8-rounded
        # as stored), 128 -> INTER_AREA (convert_pusht.py / --size 128 --resize area), else bilinear from the 256
        x = {256: (gt[256] * 255).round() / 255, 128: gt[128]}.get(size)
        x = resize_bt(gt[256], size, 'bilinear') if x is None else x  # [B, T, C, S, S] in [0, 1]
        pred, recon = rollout(tok, dyn, x.to(device) * 2 - 1, n_context=1, actions=torch.from_numpy(actions).to(device),
                              num_steps=args.num_steps, temperature=args.temperature, batch_size=args.batch_size)
        # [B, T, C, S, S] in [0, 1]; scored as is at its own resolution, bicubic-upsampled above it, INTER_AREA below it
        for res in RESOLUTIONS:
            tables[res]['model'] = summarise(metrics, rescale(pred, res)[:, 1:], gt[res][:, 1:], with_fid)
            tables[res]['tokenizer_recon'] = summarise(metrics, rescale(recon, res)[:, 1:], gt[res][:, 1:], with_fid)
        pred = rescale(pred, 128)  # for the PNG
        result['model_resolution'] = size

    # ---- NanoWM's own predictions, same clips, same metric code
    if args.nanowm_npz:
        result.update(score_nanowm_npz(metrics, args.nanowm_npz, clips, gt, tables, with_fid))

    result['tables'] = {str(r): t for r, t in tables.items()}
    result['seconds'] = round(time.time() - t0, 1)
    json_path = os.path.join(args.out_dir, f'{name}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=1)

    # ---- PNG: first 8 clips, context + 3 targets + 3 predictions at 128
    if pred is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = min(8, len(clips))
        fig, axes = plt.subplots(n, 7, figsize=(11, 1.65 * n))
        axes = np.atleast_2d(axes)
        for r in range(n):
            imgs = [gt[128][r, 0]] + [gt[128][r, t] for t in range(1, NUM_FRAMES)] + [pred[r, t] for t in range(1, NUM_FRAMES)]
            titles = ['context'] + [f'target +{5 * t}' for t in range(1, NUM_FRAMES)] + [f'pred +{5 * t}' for t in range(1, NUM_FRAMES)]
            for c, (img, title) in enumerate(zip(imgs, titles)):
                ax = axes[r, c]
                ax.imshow(img.permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=8)
                if c == 0:
                    ax.set_ylabel(f'ep {clips[r][0]} s {clips[r][1]}', fontsize=7)
        m = tables[128]['model']
        fig.suptitle(f"{name} @128: PSNR {m['psnr']:.2f}  SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}"
                     + (f"  FID {m['fid']:.1f}" if m['fid'] is not None else ''), fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'{name}.png'), dpi=130); plt.close(fig)

    print()
    print_tables(tables)
    print(f"published NanoWM @256: {spec['published']}")
    print(f'saved {json_path} ({result["seconds"]}s)')


if __name__ == '__main__':
    main()
