"""Token-shift augmentation for the dynamics model: the whole clip (context frames + target) is translated by a random
whole number of tokens on the latent grid, the same shift for every frame of a sample. One token = patch_size pixels,
so the shift never resamples a token. The border the shift reveals is filled with zeros and returned as invalid, to
be left out of the loss. No flips: they would swap the meaning of left and right in the action."""

import torch


def random_token_shifts(batch_size, max_shift, device):
    # -> [B, 2] integer (dy, dx), each uniform in [-max_shift, max_shift]
    return torch.randint(-max_shift, max_shift + 1, (batch_size, 2), device=device)


def shift_token_grid(x, shifts, grid_size):
    # x: [B, T, P, L] tokens of row-major (Hp, Wp) grids; shifts: [B, 2] (dy, dx) in tokens; grid_size: (Hp, Wp)
    # -> (shifted [B, T, P, L] with out[r, c] = x[r - dy, c - dx] and zeros where that is off the grid, valid [B, P] bool)
    B, T, P, L = x.shape
    Hp, Wp = grid_size
    src_r = torch.arange(Hp, device=x.device)[None] - shifts[:, :1]  # [B, Hp] source row of each output row
    src_c = torch.arange(Wp, device=x.device)[None] - shifts[:, 1:]  # [B, Wp]
    valid = ((src_r >= 0) & (src_r < Hp))[:, :, None] & ((src_c >= 0) & (src_c < Wp))[:, None, :]  # [B, Hp, Wp]
    src = src_r.clamp(0, Hp - 1)[:, :, None] * Wp + src_c.clamp(0, Wp - 1)[:, None, :]  # [B, Hp, Wp] flat source index
    src, valid = src.flatten(1), valid.flatten(1)  # [B, P]
    shifted = x.gather(2, src[:, None, :, None].expand(B, T, P, L))  # [B, T, P, L]
    return shifted * valid[:, None, :, None].to(x.dtype), valid


def shift_token_mask(mask, shifts, grid_size):
    # mask: [B, P] bool per-token mask of the unshifted target -> [B, P] the same mask moved with the tokens (False on the border)
    shifted, _ = shift_token_grid(mask[:, None, :, None].float(), shifts, grid_size)  # [B, 1, P, 1]
    return shifted[:, 0, :, 0] > 0
