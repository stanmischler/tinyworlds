"""The measured camera shift between two frames, and the operations that use it.

A shift s = (dy, dx), in pixels, says that the content of frame t appears s pixels further along in the later frame: the
output pixel (r, c) of shift_frames(x, s) is x(r - dy, c - dx). Shifts are found once per pair, in two passes:
  1. whole pixels: smallest MSE over +-SEARCH px, on a central crop that stays inside the frame for every candidate,
  2. quarter pixels: the same crop, candidates within +-0.75 px of the first answer, bicubic.

    python experiments/small_action/shift.py --h5 data/zelda_train_frames.h5 --out data/zelda_train_shifts.npz --device mps
"""

import argparse
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

SEARCH = 24  # px; the central crop leaves this margin on each side, so every candidate is compared on the same pixels
SIZE = 128
GAPS = (3, 4, 5)  # stored frames between the two frames of a pair; 3 and 5 are the InfoNCE positives of gap 4
QUARTERS = [q / 4 for q in range(-3, 4)]


def shift_frames(x, s, bounds=(-1, 1)):
    # x: [B, C, H, W], s: float (dy, dx) [B, 2] -> x moved by s [B, C, H, W]; pixels from outside the frame repeat its border;
    # bounds: the pixel range, which bicubic overshoot is clamped to
    H, W = x.shape[-2:]
    rows, cols = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype), torch.arange(W, device=x.device, dtype=x.dtype), indexing='ij')
    from_rows = rows[None] - s[:, 0, None, None]  # [B, H, W]
    from_cols = cols[None] - s[:, 1, None, None]  # [B, H, W]
    grid = torch.stack([from_cols / (W - 1) * 2 - 1, from_rows / (H - 1) * 2 - 1], -1)  # [B, H, W, 2], x then y in [-1, 1]
    return F.grid_sample(x, grid, mode='bicubic', padding_mode='border', align_corners=True).clamp(*bounds)


def known_pixels(s, size=SIZE):
    # s: (dy, dx) [B, 2] -> bool [B, H, W], False on the strip the shift reveals (its content is not in frame t)
    rows = torch.arange(size, device=s.device)[None]  # [1, H]
    cols = torch.arange(size, device=s.device)[None]  # [1, W]
    top, left = s[:, 0:1].clamp(min=0).ceil(), s[:, 1:2].clamp(min=0).ceil()  # scene moved down / right: strip at the top / left
    bottom, right = (-s[:, 0:1]).clamp(min=0).ceil(), (-s[:, 1:2]).clamp(min=0).ceil()
    row_ok = (rows >= top) & (rows < size - bottom)  # [B, H]
    col_ok = (cols >= left) & (cols < size - right)  # [B, W]
    return row_ok[:, :, None] & col_ok[:, None, :]


def crop(x):
    return x[..., SEARCH:-SEARCH, SEARCH:-SEARCH]


@torch.no_grad()
def best_shift(a, b):
    # a, b: float [B, C, H, W] in [0, 1] -> (dy, dx) [B, 2], smallest MSE on the central crop
    B = a.shape[0]
    target = crop(b)
    best, whole = torch.full((B,), float('inf'), device=a.device), torch.zeros(B, 2, device=a.device)
    grey_a, grey_target = a.mean(1), target.mean(1)
    H, W = a.shape[-2:]
    for dy in range(-SEARCH, SEARCH + 1):
        for dx in range(-SEARCH, SEARCH + 1):
            moved = grey_a[:, SEARCH - dy:H - SEARCH - dy, SEARCH - dx:W - SEARCH - dx]  # [B, 80, 80]
            error = (moved - grey_target).pow(2).mean((1, 2)) + (1e-7 if dy or dx else 0.0)  # no shift wins ties
            better = error < best
            best[better] = error[better]
            whole[better] = torch.tensor([dy, dx], device=a.device, dtype=whole.dtype)
    best = (crop(shift_frames(a, whole, (0, 1))) - target).pow(2).mean((1, 2, 3))  # the same score as the quarter-pixel pass
    found = whole.clone()
    for fy in QUARTERS:
        for fx in QUARTERS:
            if fy or fx:
                candidate = whole + torch.tensor([fy, fx], device=a.device)
                error = (crop(shift_frames(a, candidate, (0, 1))) - target).pow(2).mean((1, 2, 3))
                better = error < best
                best[better] = error[better]
                found[better] = candidate[better]
    return found


def known_mse(pred, target, s):
    # per-pair MSE over the pixels the shift does not reveal: pred, target [B, C, H, W], s [B, 2] -> [B]
    known = known_pixels(s, pred.shape[-1]).unsqueeze(1)  # [B, 1, H, W]
    return ((pred - target).pow(2) * known).sum((1, 2, 3)) / (known.sum((1, 2, 3)) * pred.shape[1]).clamp(min=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    with h5py.File(a.h5, 'r') as f:
        frames = f['frames'][:a.limit or None]  # uint8 [N, H, W, C]
    N = len(frames)
    shifts = np.full((N, len(GAPS), 2), np.nan, dtype=np.float16)  # (dy, dx) from frame t to frame t + gap
    errors = np.full((N, len(GAPS)), np.nan, dtype=np.float32)  # MSE in [0, 1] pixels after the shift, revealed strip left out
    to_float = lambda i: torch.from_numpy(frames[i]).to(a.device).permute(0, 3, 1, 2).float() / 255  # noqa: E731
    t0 = time.time()
    for g, gap in enumerate(GAPS):
        for start in range(0, N - gap, a.batch):
            idx = np.arange(start, min(start + a.batch, N - gap))
            x, y = to_float(idx), to_float(idx + gap)
            s = best_shift(x, y)
            shifts[idx, g] = s.cpu().numpy()
            errors[idx, g] = known_mse(shift_frames(x, s, (0, 1)), y, s).cpu().numpy()
            if (start // a.batch) % 50 == 0:
                print(f'gap {gap}: {start + len(idx)}/{N - gap} in {(time.time() - t0) / 60:.1f} min', flush=True)
    np.savez(a.out, shifts=shifts, errors=errors, gaps=np.array(GAPS))
    print(f'saved {a.out} in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
