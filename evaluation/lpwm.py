"""LPWM's video-prediction protocol on OBJ3D (Latent Particle World Models, arXiv 2603.04553, Tab. 8), reimplemented on the
.h5 files of scripts/data/convert_obj3d.py. Paths below refer to LPWM's repo (github.com/taldatech/lpwm).

Verified in their code:
  - clips: datasets/obj3d_ds.py Obj3D.__getitem__ in a non-train mode returns the WHOLE episode (all 100 frames) and
    __len__ = number of episodes, so one clip per test episode, starting at frame 0; eval/eval_gen_metrics.py:91 keeps
    x[:, :timestep_horizon] = frames 0..49 (--horizon default 50, :438). The eval CLI uses the 'test' split (:490).
  - context: -c 6 in the README command (README.md:302); the CLI default would be config timestep_horizon = 20 (:489).
  - frames: PIL resize to 128 (datasets/get_dataset.py:27-29), ToTensor()[:3] -> [0, 1], no normalisation.
  - metrics (eval_gen_metrics.py:35-52): piqa 1.3.2 (requirements.txt:8) SSIM(reduction='none') (11x11 gaussian, sigma
    1.5, value range 1), PSNR(reduction='none') (epsilon 1e-8, value range 1), LPIPS(network='vgg', reduction='none')
    (inputs in [0, 1], ImageNet-normalised inside piqa). Prediction clamped to [0, 1] (:109); scored on frames
    cond_steps.. only (:111-112), every frame of every clip pooled, mean and unbiased std over all of them (:121-138).
    Per-frame metrics do not depend on the batch, so batching is free.
Not settled by their code (see the open questions in scripts/eval/eval_lpwm.py): deterministic vs --sample and --ctx for
the OBJ3D Tab. 8 numbers."""

import h5py
import numpy as np
import torch

N_CONTEXT = 6
HORIZON = 50


def lpwm_clips(h5_path, horizon=HORIZON):
    """-> [(episode, first row)] in LPWM's order: one clip per episode, starting at its frame 0."""
    with h5py.File(h5_path, 'r') as h5:
        lengths = h5['seq_lengths'][:]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    assert lengths.min() >= horizon, f'episode shorter than the horizon {horizon}: {lengths.min()}'
    return [(int(e), int(s)) for e, s in enumerate(starts)]


def load_clips(h5_path, clips, horizon=HORIZON):
    """-> uint8 [B, T, H, W, C], T = horizon."""
    with h5py.File(h5_path, 'r') as h5:
        return np.stack([h5['frames'][s:s + horizon] for _, s in clips])


class LPWMMetrics:
    """eval/eval_gen_metrics.py ImageMetrics; inputs float [N, C, H, W] in [0, 1]."""

    def __init__(self, device):
        import piqa
        self.device = device
        self.ssim = piqa.SSIM(reduction='none').to(device)
        self.psnr = piqa.PSNR(reduction='none').to(device)
        self.lpips = piqa.LPIPS(network='vgg', reduction='none').to(device)

    @torch.no_grad()
    def per_frame(self, pred, gt, bs=64):
        out = {'psnr': [], 'ssim': [], 'lpips': []}
        for i in range(0, len(pred), bs):
            p, g = pred[i:i + bs].to(self.device).float().clamp(0, 1), gt[i:i + bs].to(self.device).float()
            out['ssim'].append(self.ssim(g, p).cpu())  # argument order as LPWM: evaluator(gt, generated)
            out['psnr'].append(self.psnr(g, p).cpu())
            out['lpips'].append(self.lpips(g, p).cpu())
        return {k: torch.cat(v) for k, v in out.items()}  # each [N]

    def summarise(self, pred, gt):
        """pred, gt: float [B, S, C, H, W] in [0, 1] (the predicted steps only) -> {psnr, ssim, lpips, *_std, per_step}."""
        B, S = pred.shape[:2]
        pf = self.per_frame(pred.flatten(0, 1), gt.flatten(0, 1))  # each [B * S], step-minor
        out = {}
        for k, v in pf.items():
            out[k] = float(v.mean())
            out[f'{k}_std'] = float(v.std())  # torch default: unbiased, as LPWM
        out['per_step'] = {k: v.reshape(B, S).mean(0).tolist() for k, v in pf.items()}  # each [S]
        return out


def to_unit(frames):
    # uint8 [B, T, H, W, C] -> float [B, T, C, H, W] in [0, 1] (LPWM's ToTensor)
    return torch.from_numpy(frames).permute(0, 1, 4, 2, 3).float() / 255.0
