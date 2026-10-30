"""Memorisation check for one dynamics checkpoint: the same 1-step prediction on TRAIN windows and on held-out windows
of the same scenes and situations, so a train vs held-out gap means memorising, not harder frames.

Windows (from the Link-blur diagnosis): heldout_windows.json = the 97 judged held-out windows (camera still, Link
moving); train_windows.json = 150 train windows matched to them by scene (nearest judged window by pooled-frame RMSE).
Each window: context start, +4, +8, target +12; tokenizer + CoMo actions re-encoded from the window's pixels
(evaluation.predictors.DynamicsPredictor, as scripts/eval/eval_next_frame.py), EMA dynamics weights, 1 Euler step,
generator seed = seed * n_candidates + index (held-out) or the window's index (train), as in the diagnosis.
Metrics (pixels in [0, 1]): full-frame PSNR / SSIM / LPIPS-alex, Link-box PSNR (the 32x32 box with most moving pixels)
and box sharpness (gradient energy of the prediction / of the target). Box crops are saved for blind judging.

Outputs in --out-dir: metrics.json (per window + per-split medians/means), box_crops.npz (uint8 [N, 3, C, 32, 32]: last
context, target, prediction).

    python experiments/regularisation/matched_windows.py --dynamics-path results/<run>/dynamics/checkpoints/dynamics_step_N \
        --out-dir results/evals/<run>/step_N
"""

import argparse
import json
import math
import os
import time

import h5py
import numpy as np
import torch

from evaluation.image_metrics import psnr, ssim, try_lpips
from evaluation.predictors import DynamicsPredictor
from evaluation.windows import load_window_batch, to_model_range, to_unit

BOX = 32
HERE = os.path.dirname(os.path.abspath(__file__))


def crop(img, box):
    # img: [..., H, W] -> [..., BOX, BOX] at (top, left) = box
    t, l = box
    return img[..., t:t + BOX, l:l + BOX]


def grad_energy(img):
    # img: [C, h, w] -> mean squared finite-difference gradient
    return float((img[:, 1:, :] - img[:, :-1, :]).pow(2).mean() + (img[:, :, 1:] - img[:, :, :-1]).pow(2).mean())


def summary(rows):
    # per split: n, mean and median of every metric
    out = {}
    for split in ('train', 'heldout'):
        r = [x for x in rows if x['split'] == split]
        out[split] = {'n': len(r)}
        for k in ('psnr', 'ssim', 'lpips', 'box_psnr', 'box_sharpness'):
            v = np.array([x[k] for x in r], dtype=float)
            out[split][k] = {'mean': float(np.nanmean(v)), 'median': float(np.nanmedian(v))}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dynamics-path', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--train-windows', default=os.path.join(HERE, 'train_windows.json'))
    p.add_argument('--heldout-windows', default=os.path.join(HERE, 'heldout_windows.json'))
    p.add_argument('--train-h5', default='data/zelda_train_frames.h5')
    p.add_argument('--test-h5', default='data/zelda_test_frames.h5')
    p.add_argument('--video-tokenizer-path', default='results/zelda_cont_final_tp/video_tokenizer/checkpoints/video_tokenizer_step_29000')
    p.add_argument('--latent-actions-path', default='results/como_actions_v1_full')
    p.add_argument('--limit', type=int, help='first N windows of each split (smoke)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    device = torch.device(a.device)
    dyn = DynamicsPredictor(a.video_tokenizer_path, a.latent_actions_path, a.dynamics_path, device, num_steps=1)
    lpips_fn = try_lpips(device)
    assert lpips_fn is not None, 'the lpips package (and its cached AlexNet weights) is needed'

    tr = json.load(open(a.train_windows))['windows'][:a.limit]
    ho_spec = json.load(open(a.heldout_windows))
    ho = ho_spec['windows'][:a.limit]
    sets = [('train', a.train_h5, tr, lambda i, w: i),
            ('heldout', a.test_h5, ho, lambda i, w: ho_spec['seed'] * ho_spec['n_windows_eval'] + w['index'])]

    rows, crops = [], []
    with torch.no_grad():
        for split, h5_path, wins, seed_of in sets:
            with h5py.File(h5_path, 'r') as h5:
                src = h5['source_index'][:]
                x_all = load_window_batch(h5['frames'], [(0, w['start']) for w in wins], 3, 4)  # uint8 [N, 4, H, W, C]
            for i, w in enumerate(wins):
                assert int(src[w['start']]) == w['source_start'], w
                x = to_model_range(x_all[i:i + 1], device)  # [1, T, C, H, W]
                last, target = to_unit(x[0, -2]), to_unit(x[0, -1])  # [C, H, W]
                seed = int(seed_of(i, w))
                pred = dyn(x, torch.Generator(device=device).manual_seed(seed))  # [C, H, W]
                box = tuple(w['box'])
                rows.append({
                    'split': split, 'id': w['id'], 'start': w['start'], 'scene': w.get('scene', f"block{w.get('block')}"),
                    'seed': seed, 'box_yx': list(box), 'heldout_label': w.get('label'),
                    'psnr': float(psnr(pred[None], target[None])), 'ssim': float(ssim(pred[None], target[None])),
                    'lpips': float(lpips_fn(pred[None] * 2 - 1, target[None] * 2 - 1)),
                    'box_psnr': float(psnr(crop(pred, box)[None], crop(target, box)[None])),
                    'box_sharpness': grad_energy(crop(pred, box)) / max(grad_energy(crop(target, box)), 1e-8)})
                crops.append(torch.stack([crop(f, box) for f in (last, target, pred)]).cpu())
    to_u8 = lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    np.savez_compressed(os.path.join(a.out_dir, 'box_crops.npz'), crops=to_u8(torch.stack(crops)))
    out = {'protocol': __doc__, 'config': vars(a), 'summary': summary(rows), 'windows': rows,
           'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(a.out_dir, 'metrics.json'), 'w'), indent=1)
    s = out['summary']
    print(f"train psnr {s['train']['psnr']['mean']:.2f} box {s['train']['box_psnr']['mean']:.2f} | heldout psnr "
          f"{s['heldout']['psnr']['mean']:.2f} box {s['heldout']['box_psnr']['mean']:.2f} ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
