"""Statistics that say whether new video frames are in the same frame of reference as our training frames.

Every function takes uint8 frames [N, H, W, C] (or windows [N, T, H, W, C]) at 15 fps and returns a number per channel
or a scalar, so the same call runs on our frames and on a new video's.
"""

import cv2
import numpy as np
import torch

from evaluation.image_metrics import psnr
from evaluation.windows import to_model_range, to_unit


def channel_means(frames):
    """[N,H,W,C] -> [C] mean intensity per channel (0-255)."""
    return frames.reshape(-1, frames.shape[-1]).mean(0)


def chunk_channel_means(frames, chunk):
    """[N,H,W,C] -> [N // chunk, C]: channel means of consecutive chunks (the spread of colour inside one video)."""
    n = len(frames) // chunk
    return np.stack([channel_means(frames[i * chunk:(i + 1) * chunk]) for i in range(n)])


def sharpness(frames):
    """[N,H,W,C] -> mean variance of the Laplacian of the grey frame (high = crisp edges, low = blur/compression)."""
    return float(np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames]))


def motion(windows):
    """[N,T,H,W,C] -> mean absolute intensity change between consecutive frames of each window (0-255)."""
    w = windows.astype(np.float32)
    return float(np.abs(w[:, 1:] - w[:, :-1]).mean())


@torch.no_grad()
def recon_psnr(tokenizer, windows, device, batch_size=32):
    """[N,T,H,W,C] -> mean PSNR (dB) of the frozen tokenizer's noise-free reconstruction of each window's last frame."""
    tokenizer.eval()
    scores = []
    for i in range(0, len(windows), batch_size):
        x = to_model_range(windows[i:i + batch_size], device).contiguous()    # [B,T,C,H,W] in [-1,1]
        _, x_hat = tokenizer(x)                                               # [B,T,C,H,W]
        scores += psnr(to_unit(x_hat[:, -1]), to_unit(x[:, -1])).tolist()
    return float(np.mean(scores))
