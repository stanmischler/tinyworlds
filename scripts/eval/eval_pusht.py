"""Push-T evaluation with NanoWM's protocol, for a tinyworlds model trained with action_source=gt.

Protocol (NanoWM, src/scripts/eval_single_model.sh + src/wm_datasets/world_model_dataset.py):
  - clips: the 21 shipped val episodes of DINO-WM's pusht_noise shuffled by np.random.seed(42), every start s in range(0, L - 20 + 1) in that episode
    order (stride 1), then np.random.RandomState(42).choice(n_slices, 256, replace=False), in that order
  - a clip is env frames s + {0, 5, 10, 15}; actions[i] = the 5 rel_actions/100 of steps s+5i .. s+5i+4, z-scored
    with the train stats and flattened to 10-D
  - 1 real context frame -> 3 predicted frames, ground-truth actions, autoregressive (one frame at a time),
    greedy MaskGIT (temperature 0); the 4 frames' tokens are decoded together, as in training
  - metrics on the 3 predicted frames only, mean over the 768 frames: PSNR (piqa, value_range 1), SSIM (piqa,
    11x11 gaussian, sigma 1.5), LPIPS-VGG (lpips 0.1.4, inputs in [-1, 1]), FID (pytorch-fid InceptionV3 pool3,
    bilinear resize to 299, inputs in [0, 1]); plus per predicted step (no FID)
Scored at two resolutions:
  128: prediction as is vs the 224px frame INTER_AREA-resized to 128 (as the training set)   <- headline
  256: prediction bicubic-upsampled vs the 224px frame bilinear-resized to 256 (NanoWM's ground truth exactly)
References in every table: copy-last (frame 0 repeated), tokenizer_recon (the 4 real frames tokenized and decoded:
the ceiling of any token prediction), resize_ceiling (256 only: the 128 ground truth bicubic-upsampled, the best any
128px model can score at 256), and nanowm (if --nanowm-npz: NanoWM's own predictions from
experiments/pusht_nanowm/nanowm_rescore_modal.py, scored with the same code; use predictions_f16.npz: rounding their float
predictions to uint8 alone raises LPIPS by ~0.003 and FID by ~3; at 128 they are INTER_AREA-downsampled).

Outputs <out_dir>/<name>.json and <out_dir>/<name>.png (8 clips: context, targets, predictions).

Usage (repo root, PYTHONPATH=$PWD):
    python scripts/eval/eval_pusht.py --run-dir results/pusht_v1_<date> --nanowm-npz results/nanowm_pusht_rescore/predictions_f16.npz
    python scripts/eval/eval_pusht.py --run-dir results/<smoke> --limit 8 --no-fid          # smoke
    python scripts/eval/eval_pusht.py --nanowm-npz results/nanowm_pusht_rescore/predictions_f16.npz   # NanoWM only
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

from evaluation.nanowm import (NUM_FRAMES, RESOLUTIONS, Metrics, ground_truth, load_clips, nanowm_clips, nanowm_val_order,
                               print_tables, reference_tables, score_nanowm_npz, summarise, upsample)
from utils.utils import find_latest_checkpoint, load_videotokenizer_from_checkpoint, load_dynamics_from_checkpoint


# ----------------------------------------------------------------------------- model
@torch.no_grad()
def rollout(tok, dyn, x, actions, num_steps, temperature, batch_size):
    """x: float [B, T, C, H, W] in [-1, 1] (only frame 0 is used as context); actions [B, T - 1, A].
    -> pred, recon float [B, T, C, H, W] in [0, 1]; pred_idx, gt_idx [B, T, P]."""
    preds, recons, pidx, gidx = [], [], [], []

    def idx_to_latents(idx):
        return tok.quantizer.get_latents_from_indices(idx, dim=-1)

    for i in range(0, len(x), batch_size):
        xb, ab = x[i:i + batch_size], actions[i:i + batch_size]
        gt_idx = tok.tokenize(xb)  # [b, T, P]; causal tokenizer: frame 0's codes do not depend on later frames
        lat = idx_to_latents(gt_idx[:, :1])  # [b, 1, P, L]
        for t in range(1, xb.shape[1]):
            lat = dyn.forward_inference(lat, prediction_horizon=1, num_steps=num_steps, index_to_latents_fn=idx_to_latents,
                                        conditioning=ab[:, :t], temperature=temperature)  # [b, t + 1, P, L]
        preds.append(((tok.detokenize(lat) + 1) / 2).clamp(0, 1).cpu())  # decode the 4 frames together, as trained
        recons.append(((tok.detokenize(idx_to_latents(gt_idx)) + 1) / 2).clamp(0, 1).cpu())
        pidx.append(tok.quantizer.get_indices_from_latents(lat, dim=-1).cpu())
        gidx.append(gt_idx.cpu())
        print(f'  rollout {min(i + batch_size, len(x))}/{len(x)}', flush=True)
    return torch.cat(preds), torch.cat(recons), torch.cat(pidx), torch.cat(gidx)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--val-h5', default='data/pusht_val_frames.h5')
    p.add_argument('--run-dir', help='results dir holding the tokenizer + dynamics checkpoints (latest step of each)')
    p.add_argument('--video-tokenizer-path'); p.add_argument('--dynamics-path')
    p.add_argument('--nanowm-npz', help='NanoWM predictions (experiments/pusht_nanowm/nanowm_rescore_modal.py) to score alongside')
    p.add_argument('--num-clips', type=int, default=256)
    p.add_argument('--seed', type=int, default=42, help="NanoWM's validation_fixed_subset_seed")
    p.add_argument('--num-steps', type=int, default=10, help='MaskGIT unmasking iterations per frame')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    p.add_argument('--limit', type=int, help='first N clips only (smoke test; FID needs many)')
    p.add_argument('--no-fid', action='store_true')
    p.add_argument('--out-dir', default='eval_results')
    p.add_argument('--name')
    args = p.parse_args()

    has_model = bool(args.run_dir or args.dynamics_path)
    if args.run_dir:
        args.video_tokenizer_path = find_latest_checkpoint('.', 'video_tokenizer', run_root_dir=args.run_dir)
        args.dynamics_path = find_latest_checkpoint('.', 'dynamics', run_root_dir=args.run_dir)
    # with neither a model nor --nanowm-npz only the references are computed (copy-last, resize ceiling)
    name = args.name or (os.path.basename(os.path.normpath(args.run_dir)) if args.run_dir else ('nanowm' if args.nanowm_npz else 'references')) + '_pusht'
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    # ---- clips and ground truth
    h5 = h5py.File(args.val_h5, 'r')
    seq_lengths = [int(L) for L in h5['seq_lengths'][:]]
    ep_offset = np.concatenate([[0], np.cumsum(seq_lengths)[:-1]])
    mean, std = h5.attrs['action_mean'], h5.attrs['action_std']
    # their val split (split_ratio 0) still shuffles the episodes first; verified against their fixed_subset.json by
    # experiments/pusht_nanowm/nanowm_rescore_modal.py: identical list and order
    order = nanowm_val_order(len(seq_lengths), 0.0, args.seed)
    clips, n_slices = nanowm_clips(order, [seq_lengths[e] for e in order], args.num_clips, args.seed)
    if args.limit:
        clips = clips[:args.limit]
    print(f'{len(clips)} clips (of {n_slices} val slices, seed {args.seed}) from {args.val_h5}')
    frames224, actions = load_clips(h5, ep_offset, clips, mean, std)  # [B, T, 224, 224, 3], [B, T-1, 10]
    gt = ground_truth(frames224)  # {128: INTER_AREA, 256: NanoWM's float bilinear 224 -> 256}, each [B, T, C, S, S]
    with_fid = not args.no_fid
    metrics = Metrics(device, fid=with_fid)
    tables = reference_tables(metrics, gt, with_fid)

    result = {'name': name, 'n_clips': len(clips), 'n_val_slices': n_slices, 'val_h5': args.val_h5,
              'clips': [[int(e), int(s)] for e, s in clips],
              'config': {k: v for k, v in vars(args).items() if k not in ('out_dir', 'name')}}

    # ---- our model
    pred = None
    if has_model:
        print(f'checkpoints:\n  {args.video_tokenizer_path}\n  {args.dynamics_path}')
        tok, _ = load_videotokenizer_from_checkpoint(args.video_tokenizer_path, device)
        dyn, _ = load_dynamics_from_checkpoint(args.dynamics_path, device)
        tok.eval(); dyn.eval()
        x = gt[128].to(device) * 2 - 1  # [B, T, C, 128, 128] in [-1, 1], the training transform
        pred, recon, pred_idx, gt_idx = rollout(tok, dyn, x, torch.from_numpy(actions).to(device),
                                                args.num_steps, args.temperature, args.batch_size)
        for res in RESOLUTIONS:
            tables[res]['model'] = summarise(metrics, upsample(pred, res)[:, 1:], gt[res][:, 1:], with_fid)
            tables[res]['tokenizer_recon'] = summarise(metrics, upsample(recon, res)[:, 1:], gt[res][:, 1:], with_fid)
        acc = (pred_idx[:, 1:] == gt_idx[:, 1:]).float().mean((0, 2))  # [3]
        copy_acc = (gt_idx[:, :1] == gt_idx[:, 1:]).float().mean((0, 2))
        result['token_acc'] = {'model': float(acc.mean()), 'copy_last': float(copy_acc.mean()),
                               'model_per_step': {str(t + 1): float(a) for t, a in enumerate(acc)}}

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
            imgs = [gt[128][r, 0]] + [gt[128][r, t] for t in range(1, 4)] + [pred[r, t] for t in range(1, 4)]
            titles = ['context'] + [f'target +{5 * t}' for t in range(1, 4)] + [f'pred +{5 * t}' for t in range(1, 4)]
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
    if 'token_acc' in result:
        print(f"token acc {result['token_acc']['model']:.3f} (copy {result['token_acc']['copy_last']:.3f})")
    print(f'saved {json_path} ({result["seconds"]}s)')


if __name__ == '__main__':
    main()
