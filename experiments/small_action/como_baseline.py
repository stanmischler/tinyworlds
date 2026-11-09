"""CoMo's decoder on the same next-frame scores as the small action model (eval.py), so the two compare on the same pixels.

    python experiments/small_action/como_baseline.py --ckpt <dir>/como_step_50000 --out eval_results/small_action/como_20min
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch

from evaluation.image_metrics import try_lpips
from evaluation.link_box import box_lpips, judge_boxes, overlay_windows
from evaluation.windows import test_windows
from experiments.small_action.eval import GAP_ROW, LINK_BOXES, TEST_H5, TEST_SHIFTS, psnr_of_mse, to_pixels
from experiments.small_action.shift import known_mse
from utils.utils import load_latent_actions_from_checkpoint


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    lam, _ = load_latent_actions_from_checkpoint(a.ckpt, a.device)
    windows = test_windows(TEST_H5, 3, 4, 8)
    table = np.load(TEST_SHIFTS)['shifts'][:, GAP_ROW].astype(np.float32)
    with h5py.File(TEST_H5, 'r') as f:
        frames_a = np.stack([f['frames'][s + 8] for _, s in windows])
        frames_b = np.stack([f['frames'][s + 12] for _, s in windows])
    shifts = table[[s + 8 for _, s in windows]]
    full, known, predictions = [], [], []
    for i in range(0, len(windows), 32):
        x_a, x_b = to_pixels(frames_a[i:i + 32], a.device), to_pixels(frames_b[i:i + 32], a.device)
        s = torch.from_numpy(shifts[i:i + 32]).to(a.device)
        z = lam.encode(torch.stack([x_a, x_b], 1))  # [B, 1, A]
        pred = lam.como.decoder(x_a, z.reshape(z.shape[0], lam.como.idm.n_queries, -1)).clamp(-1, 1)
        full.append(psnr_of_mse((pred - x_b).pow(2).mean((1, 2, 3))).cpu())
        known.append(psnr_of_mse(known_mse(pred, x_b, s)).cpu())
        predictions.append(((pred + 1) / 2).cpu())
    predictions = torch.cat(predictions)
    target = torch.from_numpy(frames_b).permute(0, 3, 1, 2).float() / 255
    boxes = torch.as_tensor(judge_boxes(LINK_BOXES, windows, frames_a.shape[1]))
    boxes[torch.tensor(overlay_windows(TEST_H5, windows, 3, 4))] = -1
    lpips_fn = try_lpips(a.device)
    lpips_values = torch.cat([box_lpips(lpips_fn, predictions[i:i + 64].to(a.device), target[i:i + 64].to(a.device), boxes[i:i + 64]) for i in range(0, len(target), 64)])
    res = {'val_psnr_full': torch.cat(full).mean().item(), 'val_psnr_known': torch.cat(known).mean().item(),
           'val_judge_box_lpips': lpips_values.nanmean().item(), 'n_boxes': int((boxes[:, 0] >= 0).sum())}
    print(json.dumps(res, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f'{a.out}/score.json', 'w'), indent=1)
    torch.save(predictions.mul(255).round().byte(), f'{a.out}/predictions.pt')


if __name__ == '__main__':
    main()
