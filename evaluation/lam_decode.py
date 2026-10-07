"""Run a latent action model's decoder on held-out windows (true vs shuffled actions, masked vs full context)."""

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.image_metrics import psnr
from evaluation.windows import to_unit


def window_ot(plans, windows, n_trans, frame_skip, device):
    # OT plans of the first n_trans transitions of each window -> [B, n_trans, 2, P] long, or None without plans
    if plans is None:
        return None
    idx = np.array([[s + k * frame_skip for k in range(n_trans)] for _, s in windows])  # [B, n_trans]
    return torch.from_numpy(np.stack([plans['sigma'][idx], plans['created'][idx]], 2).astype(np.int64)).to(device)


def decode(lam, x, actions, masked, ot=None):
    # x: [B, T, C, H, W], actions: [B, T-1, A], ot: [B, T-1, 2, P] or None -> predicted frames 1..T-1 [B, T-1, C, H, W]
    # the decoder masks frames 1.. only when in train mode (no other layer of the LAM depends on the mode);
    # reseeded so the true and shuffled decodes see the same mask
    torch.manual_seed(0)
    lam.decoder.train(masked)
    out = lam.decoder(x, actions, training=True, ot=ot)
    lam.decoder.eval()
    return out


def per_sample_loss(pred, target):
    # [B, T-1, C, H, W] -> smooth L1 [B], PSNR [B] (mean over frames, in [0, 1] space)
    l1 = F.smooth_l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    p = torch.stack([psnr(to_unit(pred[:, t]), to_unit(target[:, t])) for t in range(pred.shape[1])], 1).mean(1)
    return l1, p
