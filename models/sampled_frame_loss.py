"""Sampled-frame loss (train_dynamics.py, sampled_loss_weight > 0): the flow dynamics model samples the next latent from
noise like the eval, the frozen tokenizer decodes it with the real context, and Link's 32x32 motion box is compared with
the same box of the real frame, LPIPS-alex + L1, scored as the eval's box LPIPS (evaluation/link_box.py)."""

import torch
import torch.nn.functional as F

from evaluation.link_box import BOX, LPIPS_SIZE, crop
from evaluation.motion import motion_box


def frozen_lpips(device):
    # LPIPS-alex as the eval's (evaluation/image_metrics.py try_lpips), with its linear heads frozen too
    import lpips
    fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    for p in fn.parameters():
        p.requires_grad = False
    return fn


@torch.no_grad()
def boxed_windows(last, target, max_windows):
    # last, target: [B, C, H, W] real frames in [0, 1] -> (rows [n] of the first <= max_windows windows with a motion box,
    # their boxes [n, 2] top-left on the CPU, as link_box.crop wants)
    boxes, _ = motion_box(last.float(), target.float(), size=BOX)  # [B, 2], -1 where there is no box
    rows = torch.nonzero(boxes[:, 0] >= 0).flatten()[:max_windows]  # [n]
    return rows, boxes[rows].cpu()


def box_lpips_l1(lpips_fn, pred, target, boxes):
    # pred: [n, C, H, W] decoded sample in [0, 1] (not clamped: a clamp would cut the gradient of out-of-range pixels);
    # target: [n, C, H, W] real frame in [0, 1]; boxes: [n, 2] -> scalar mean over the n boxes of LPIPS + L1
    p, _ = crop(pred, boxes)  # [n, C, BOX, BOX]
    t, _ = crop(target, boxes)
    up = lambda x: F.interpolate(x, size=LPIPS_SIZE, mode='bilinear', align_corners=False)  # the eval's upsampling
    lpips = lpips_fn(up(p) * 2 - 1, up(t) * 2 - 1).mean()  # LPIPS takes [-1, 1]
    l1 = (p - t).abs().mean()
    return lpips, l1
