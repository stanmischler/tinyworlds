"""Long rollouts of a flow dynamics model against reality (still-camera games such as Bigfish).

From each held-out clip's 3 real context frames, the model dreams the next `horizon` frames with the clip's true
actions, feeding back its own frames. Per frame index we compare the dream with the real frame:
  - psnr: pixel PSNR (dream vs real), averaged over clips;
  - object_ratio: share of the dream's pixels that are not background / the same share in the real frame, averaged
    over clips. The background is the per-clip, per-pixel median of its real frames (fixed camera, so moving fish drop
    out of the median); a pixel is foreground when it differs from it by more than `fg_thresh` (0-255, max over RGB).
    1 = as many objects as reality; above 1 = objects appear that are not there (e.g. fish multiplying).
"""
import numpy as np
import torch


@torch.no_grad()
def rollout(tok, dyn, frames, actions, num_steps, context=3, batch=32, seed=0, device='cuda'):
    # frames: [S, 3 + H, Hpx, Wpx, C] uint8 real clips; actions: [S, 2 + H, A] standardized (actions[:, t] drives
    # t -> t+1) -> dreamed frames [S, H, Hpx, Wpx, C] uint8
    S, L = frames.shape[:2]
    H = L - context
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i in range(0, S, batch):
        fr = torch.from_numpy(frames[i:i + batch, :context]).to(device)  # [b, 3, Hpx, Wpx, C]
        x = fr.permute(0, 1, 4, 2, 3).float() / 127.5 - 1  # [b, 3, C, Hpx, Wpx]
        lat = tok.encode(x)  # [b, 3, P, L]
        act = torch.from_numpy(actions[i:i + batch]).to(device).float()  # [b, L-1, A]
        dream = []
        for t in range(H):
            cond = act[:, t:t + context]  # [b, 3, A]; the last drives the new frame
            new = dyn.sample_next(lat[:, -context:], cond, num_steps, generator=g)[:, None].to(lat.dtype)  # [b, 1, P, L]
            img = tok.decode(new)[:, 0]  # [b, C, Hpx, Wpx] in [-1, 1]
            dream.append(((img.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
            lat = torch.cat([lat[:, 1:], new], 1)
        out.append(np.stack(dream, 1))  # [b, H, Hpx, Wpx, C]
    return np.concatenate(out)


def rollout_metrics(dream, real, fg_thresh=40):
    # dream, real: [S, H, Hpx, Wpx, C] uint8 (real = the clips' frames after the context) -> dict of [H] arrays
    d, r = dream.astype(np.float32), real.astype(np.float32)
    mse = ((d - r) ** 2).mean((2, 3, 4))  # [S, H]
    psnr = 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-10))  # [S, H]
    bg = np.median(r, axis=1, keepdims=True)  # [S, 1, Hpx, Wpx, C] per-clip background
    fg_d = (np.abs(d - bg).max(-1) > fg_thresh).mean((2, 3))  # [S, H] foreground share of the dream
    fg_r = (np.abs(r - bg).max(-1) > fg_thresh).mean((2, 3))  # [S, H] ... of reality
    ratio = fg_d / np.maximum(fg_r, 1e-3)  # [S, H]
    return {'psnr': psnr.mean(0), 'object_ratio': ratio.mean(0), 'fg_dream': fg_d.mean(0), 'fg_real': fg_r.mean(0)}
