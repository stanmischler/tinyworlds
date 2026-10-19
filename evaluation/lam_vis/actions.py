"""A frozen CoMo checkpoint as the two operations every visualisation uses:
  - infer(frame t, frame t+4) -> z [B, A=128]   (frozen MAE ViT-L features -> motion IDM)
  - apply(frame t, z)         -> frame t+4      (CoMo's forward-dynamics decoder)
Frames are [B, C, H, W] in [-1, 1]. apply clamps its output to [-1, 1], so a predicted frame can be fed back to infer
(the MAE preprocessing does not clamp) or to apply (repeated actions)."""

import torch

from utils.utils import load_como_from_checkpoint


class FrozenCoMo:

    def __init__(self, ckpt, device, batch_size=32):
        self.lam, _ = load_como_from_checkpoint(ckpt, device)  # CoMoLAM: frozen MAE + IDM + decoder
        self.lam.eval()
        # infer() feeds exactly two frames: no history frames (temporal-tokenizer CoMo) and not the drawing model
        assert self.lam.history == 0 and not hasattr(self.lam.como, 'draw'), f'{ckpt}: only plain CoMo checkpoints'
        idm = self.lam.como.idm
        self.z_shape = (idm.n_queries, idm.down[-1].out_features)  # (Q=8, L=16)
        self.device, self.batch_size = device, batch_size

    @torch.no_grad()
    def infer(self, frame_t, frame_next):
        # [N, C, H, W] x2 -> z [N, A]
        return self._batched(lambda a, b: self.lam.encode(torch.stack([a, b], 1))[:, 0].float(), frame_t, frame_next)

    @torch.no_grad()
    def apply(self, frame_t, z):
        # frame [N, C, H, W], z [N, A] -> predicted frame t+4 [N, C, H, W] in [-1, 1]
        return self._batched(lambda f, a: self.lam.como.decoder(f, a.view(-1, *self.z_shape)).clamp(-1, 1), frame_t, z)

    def _batched(self, fn, x, y):
        # run fn on device in chunks of batch_size; inputs and output stay where the caller keeps them
        outs = [fn(x[i:i + self.batch_size].to(self.device), y[i:i + self.batch_size].to(self.device)).to(x.device)
                for i in range(0, len(x), self.batch_size)]
        return torch.cat(outs)
