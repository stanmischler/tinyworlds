"""Radially averaged power spectrum of frames and the high-frequency power deficit of predictions: how much of the
fine detail (edges, textures) a model's frames lose against a reference, per frequency band (cf. Degeorge et al. 2026,
Fig. 2a)."""

import torch

# bands in cycles per pixel (Nyquist = 0.5)
BANDS = {'low': (0.0, 0.0625), 'mid': (0.0625, 0.25), 'high': (0.25, 0.5)}


def radial_power_sum(frames):
    # frames: [N, C, H, W] (H = W) -> power summed over N and channels per integer radius 1 .. H/2 - 1 [H/2 - 1] (float64)
    x = frames.detach().cpu().double()  # on the CPU: float64 (MPS has none); the DC bin (radius 0) is not returned
    power = torch.fft.fft2(x, norm='ortho').abs().pow(2).sum(dim=(0, 1))  # [H, W]
    H, W = power.shape
    fy, fx = torch.fft.fftfreq(H)[:, None], torch.fft.fftfreq(W)[None]
    radius = (torch.sqrt(fy ** 2 + fx ** 2) * H).round().long()  # [H, W] in frequency bins
    return torch.stack([power[radius == r].mean() for r in range(1, H // 2)])  # [H/2 - 1]


def band_deficits(pred_power, ref_power, size):
    # pred_power, ref_power: [size/2 - 1] radial power sums -> {band: mean over the band's radii of the % power of pred vs
    # ref}, negative = detail lost (a mean of per-radius ratios, so the band's low radii, which hold most power, do not dominate)
    freqs = torch.arange(1, size // 2, dtype=torch.float64) / size  # [size/2 - 1] cycles / pixel
    out = {}
    for band, (lo, hi) in BANDS.items():
        keep = (freqs >= lo) & (freqs < hi)
        out[band] = float(100 * (pred_power[keep] / ref_power[keep] - 1).mean())
    return out
