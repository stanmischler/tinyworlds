"""Pixel-motion proxies between consecutive frames (camera scroll by phase correlation, local block matching),
used to check what latent actions encode."""

import torch
import torch.nn.functional as F


MOTION_LABELS = [f'{v}{h}' for v in ('U', '0', 'D') for h in ('L', '0', 'R')]  # 9 classes, '00' = still


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
