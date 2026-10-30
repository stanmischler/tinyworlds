import torch.nn as nn
from einops import rearrange


class PixelPatches(nn.Module):
    """Stand-in for the video tokenizer when the dynamics model works in pixel space (DynamicsConfig.pixel_space): a frame
    is cut into non-overlapping p x p patches, each flattened to a token of L = C * p * p pixel values in [-1, 1].
    No weights; encode/decode are exact inverses, so the training loop, test curve and eval run unchanged."""

    continuous = True  # the dynamics model's samples are used as they are (no grid rounding)
    is_pixels = True

    def __init__(self, patch_size, channels=3):
        super().__init__()
        self.patch_size, self.channels = patch_size, channels
        self.latent_dim = channels * patch_size ** 2

    def encode(self, x):
        # x: [B, T, C, H, W] -> [B, T, P, L], P = (H/p)(W/p), L = C p p
        p = self.patch_size
        return rearrange(x, 'b t c (hp p1) (wp p2) -> b t (hp wp) (c p1 p2)', p1=p, p2=p)

    def decode(self, z):
        # z: [B, T, P, L] -> [B, T, C, H, W] (square frames: Hp = Wp = sqrt(P))
        p = self.patch_size
        hp = int(round(z.shape[2] ** 0.5))
        return rearrange(z, 'b t (hp wp) (c p1 p2) -> b t c (hp p1) (wp p2)', hp=hp, c=self.channels, p1=p, p2=p)
