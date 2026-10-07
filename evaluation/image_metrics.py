"""Per-image quality metrics on [B, C, H, W] tensors in [0, 1] (the held-out eval protocol of scripts/eval/eval_next_frame.py)."""

import torch
import torch.nn.functional as F


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
