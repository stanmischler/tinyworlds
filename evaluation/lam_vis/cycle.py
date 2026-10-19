"""Action transfer, cycle and leakage tests of Garrido et al. 2026 ("Learning latent action world models in the wild",
arXiv 2601.05230, §6, Fig. 5, Tables 1-2), with CoMo's decoder as the world model and one step (t -> t+4).

Cycle, for a pair A = (A_t, A_t+4) and a start frame B_t of another pair:
    z      = IDM(A_t, A_t+4)          action of A
    direct = Dec(A_t, z)              A predicted with its own action (reference)
    B_hat  = Dec(B_t, z)              1st transfer: A's action applied to B
    z_back = IDM(B_t, B_hat)          the action re-inferred from the transfer
    A_hat  = Dec(A_t, z_back)         2nd transfer: back on A
  score: error(A_hat, A_t+4) vs error(direct, A_t+4), and cos(z, z_back). If the transfer kept the motion, A_hat is as
  good as direct; if it lost it, A_hat falls to the `other_block_action` floor Dec(A_t, z_B) (B's own action on A).
Leakage, with C a frame of an unrelated pair:
    z_cut  = IDM(A_t, C)              a "scene cut" as the future
    leak   = Dec(A_t, z_cut)
  score: error(leak, C) vs copy error(A_t, C). If leak is close to C, z carries the whole next frame (not an action),
  and a good cycle score would mean nothing."""

import torch

from evaluation.image_metrics import psnr, ssim


def cycle(model, a_t, a_next, b_t, b_next):
    # all frames [N, C, H, W] in [-1, 1] -> dict of tensors
    z = model.infer(a_t, a_next)  # [N, A]
    b_hat = model.apply(b_t, z)  # 1st transfer
    z_back = model.infer(b_t, b_hat)  # [N, A]
    z_b = model.infer(b_t, b_next)  # [N, A] B's own action
    return {'z': z, 'z_back': z_back, 'z_b': z_b, 'b_hat': b_hat,
            'direct': model.apply(a_t, z),
            'a_hat': model.apply(a_t, z_back),  # 2nd transfer
            'other_block_action': model.apply(a_t, z_b)}  # B's own action on A: the "action lost" floor


def leakage(model, a_t, c):
    # a_t, c: [N, C, H, W] -> Dec(A_t, IDM(A_t, C)) [N, C, H, W]
    return model.apply(a_t, model.infer(a_t, c))


def image_scores(pred, target, lpips_fn=None):
    # pred, target: [N, C, H, W] in [-1, 1] -> per-pair PSNR, SSIM (pixels in [0, 1]) and LPIPS, each [N]
    p, t = (pred + 1) / 2, (target + 1) / 2
    out = {'psnr': psnr(p, t), 'ssim': ssim(p, t)}
    if lpips_fn is not None:
        dev = next(lpips_fn.parameters()).device
        with torch.no_grad():
            out['lpips'] = torch.cat([lpips_fn(pred[i:i + 32].to(dev), target[i:i + 32].to(dev)).flatten().cpu()
                                      for i in range(0, len(pred), 32)])
    return out
