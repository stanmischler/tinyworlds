"""Focal log-frequency loss (f-loss) and the frequency-to-pixel switch of Degeorge et al. 2026, "Balancing Frequencies and
Pixels in Flow Matching" (arXiv 2609.02748), Eq. 3-4, on decoded frames.

    L_f = 1 / max(s, s_min)^2 * mean_{c,u,v} [ sg(e / max e) * log(1 + e) ],   e = |F(pred) - F(target)|

F is the orthonormal 2D DFT of each RGB channel over the whole frame; the max is per sample, over all channels and
frequencies; s = 1 - t is the noise level of the target (the paper's 1 / (1 - t)^2 timestep weight, clipped at 0.05).
The paper sums over frequencies and leaves the DFT normalisation unstated; we take the mean and the orthonormal DFT."""

import math

import torch


def focal_log_frequency_loss(pred, target, s, s_min=0.05):
    # pred, target: [B, C, H, W] frames (pred carries the gradient); s: [B] noise level -> scalar
    f_pred = torch.fft.fft2(pred.float(), norm='ortho')  # [B, C, H, W] complex
    f_target = torch.fft.fft2(target.float(), norm='ortho')  # [B, C, H, W] complex
    e = (f_pred - f_target).abs()  # [B, C, H, W] per-frequency residual
    focal = (e / e.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)).detach()  # [B, C, H, W] in [0, 1], no gradient
    per_sample = (focal * torch.log1p(e)).mean(dim=(1, 2, 3))  # [B]
    time_weight = 1.0 / s.float().clamp_min(s_min) ** 2  # [B]
    return (time_weight * per_sample).mean()


def frequency_weight(step, switch_step, width):
    # lambda(step): sigmoid from 1 (f-loss) to 0 (v-loss), 0.5 at switch_step
    return 1.0 / (1.0 + math.exp(min((step - switch_step) / width, 50.0)))
