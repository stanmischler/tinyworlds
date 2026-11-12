"""Small action model: a measured camera shift s plus a learned action a of a few numbers.

    s  = best shift from frame t to frame t+n (experiments/small_action/shift.py, no weights)
    a  = ActionEncoder(frame t moved by s, frame t+n)
    x^ = frame t moved by s + Decoder(frame t moved by s, a)
    s^ = ShiftPredictor(frame t, a with no gradient)

The decoder is a training aid. Sizes in plain numbers: frames 128 px, encoder tokens 32 x 32, decoder tokens 16 x 16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.link_box import LPIPS_SIZE
from models.como import _encoder, _init, info_nce
from models.norms import AdaptiveNormalizer
from experiments.small_action.shift import known_pixels, shift_frames

HIDDEN_MARGIN = 12  # px; a fixed width, so the action encoder cannot read the shift from the width of the revealed strip
ERROR_FLOOR = 0.01  # mean |error| of a pair, pixels in [-1, 1], below which its error weights are not amplified
WINDOW = 96  # px; side of the window some training pairs are cut to (window_fraction of the trainer)
WHOLE = 999  # jitter value marking a pair that stays whole


def hide_margin(x, margin=HIDDEN_MARGIN):
    # x: [B, C, H, W] in [-1, 1] -> the same with its outer `margin` px set to 0 (mid grey)
    mask = torch.zeros_like(x[:, :1])
    mask[..., margin:-margin, margin:-margin] = 1
    return x * mask


def window_corner(s, jitter, size=WINDOW, frame=128):
    # s: the shift of a pair (dy, dx) [B, 2], jitter: [B, 2] long, WHOLE = the pair stays whole -> the corner (row, col) of its window
    # [B, 2] long, -1 = whole. The window follows the shift: corner = centred + round(s) - jitter, so it frames the same patch of ground
    # in the aligned frame t whatever the shift, as a camera that stays still would. Where the camera follows Link, he then starts near the
    # centre of the window and ends displaced by his step: the same picture as a pair filmed by a still camera, and nothing in it tells the
    # direction of the step. The jitter is the same in every direction, so it carries no direction either
    corner = (frame - size) // 2 + s.round().long() - jitter
    return torch.where(jitter[:, :1] == WHOLE, torch.full_like(corner, -1), corner)


def window_canvas(x, corner, size=WINDOW):
    # x: [B, 3, H, W] in [-1, 1], corner: (row, col) of each pair's window [B, 2], -1 = the pair stays whole -> [B, 3, H, W]: on the cut
    # pairs, the size x size window copied pixel for pixel to the centre of a mid-grey frame. No resize, so nothing changes scale
    cut = corner[:, 0] >= 0
    if not cut.any():
        return x
    margin = (x.shape[-1] - size) // 2
    steps = torch.arange(size, device=x.device)
    rows = (corner[cut, 0, None] + steps)[:, :, None]  # [C, size, 1]
    cols = (corner[cut, 1, None] + steps)[:, None, :]  # [C, 1, size]
    pair = torch.arange(len(rows), device=x.device)[:, None, None]
    window = x[cut][pair, :, rows, cols].permute(0, 3, 1, 2)  # [C, 3, size, size]
    canvas = torch.zeros_like(window[:, :, :1, :1]).expand(-1, -1, *x.shape[-2:]).clone()  # [C, 3, H, W], 0 = mid grey
    canvas[:, :, margin:margin + size, margin:margin + size] = window
    out = x.clone()
    out[cut] = canvas
    return out


def scored_pixels(s, corner, size=WINDOW):
    # -> float [B, 1, H, W], the pixels the reconstruction loss counts: a whole pair's known pixels (not the strip its shift reveals);
    # a cut pair's window, which the trainer's jitter keeps inside the known pixels
    known = known_pixels(s).unsqueeze(1).float()
    if corner is None:
        return known
    margin = (known.shape[-1] - size) // 2
    centre = torch.zeros_like(known[:1])
    centre[..., margin:margin + size, margin:margin + size] = 1
    return torch.where((corner[:, 0] >= 0)[:, None, None, None], centre, known)


def pixel_weights(aligned, target, known, strength):
    # aligned, target: [B, 3, H, W], known: [B, 1, H, W] float -> [B, 1, H, W]: 1 + strength * error / the pair's mean error, 0 on
    # unknown pixels. Normalised per pair, so scrolling pairs (large error everywhere) cannot take the weight from a still pair, and
    # the floor of 1 keeps pixels without error in the loss
    error = (aligned - target).abs().mean(1, keepdim=True).detach()  # [B, 1, H, W], |error after the shift|
    mean = (error * known).sum((2, 3), keepdim=True) / known.sum((2, 3), keepdim=True).clamp(min=1)  # [B, 1, 1, 1]
    return known * (1 + strength * error / mean.clamp(min=ERROR_FLOOR))


def residual_crops(pred, target, aligned, known, size):
    # the size x size window holding the most error after the shift, per pair; found from pixels alone -> crops [B, 3, size, size] x 2
    error = (aligned - target).abs().mean(1, keepdim=True).detach() * known  # [B, 1, H, W]
    pooled = F.avg_pool2d(error, size, stride=1)  # [B, 1, H-size+1, W-size+1]
    best = pooled.flatten(1).argmax(1)  # [B]
    rows, cols = (best // pooled.shape[-1]).tolist(), (best % pooled.shape[-1]).tolist()
    take = lambda x: torch.stack([x[i, :, r:r + size, c:c + size] for i, (r, c) in enumerate(zip(rows, cols))])  # noqa: E731
    return take(pred), take(target)


class ConvStem(nn.Module):
    """Frame [B, 3, 128, 128] -> features [B, 1024, D], one per 4 px. Each token sees 21 px, wider than a 9 px walk step."""

    def __init__(self, dim=128):
        super().__init__()
        layers, width = [], 3
        for out, stride in [(dim // 2, 1), (dim // 2, 2), (dim, 1), (dim, 2), (dim, 1)]:
            layers += [nn.Conv2d(width, out, 3, stride, 1), nn.GroupNorm(8, out), nn.GELU()]
            width = out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).flatten(2).transpose(1, 2)  # [B, 1024, D]


class ActionEncoder(nn.Module):
    """(aligned frame t, frame t+n) [B, 3, 128, 128] x 2 -> a [B, A]. The same stem reads both frames, and the tokens carry
    both feature maps and their difference. Q learned queries go through the transformer with the tokens, then a learned
    weight per query merges them into one action."""

    def __init__(self, action_dim, stem_dim=128, dim=256, depth=4, heads=8, mlp=1024, n_queries=8, n_tokens=1024, noisy=False):
        super().__init__()
        self.n_queries, self.action_dim = n_queries, action_dim
        self.stem = ConvStem(stem_dim)
        self.proj = nn.Linear(3 * stem_dim, dim)
        self.queries = nn.Parameter(torch.zeros(1, n_queries, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_queries + n_tokens, dim))
        self.transformer = _encoder(dim, heads, mlp, depth)
        self.norm = nn.LayerNorm(dim, eps=1e-12)
        self.query_weight = nn.Linear(dim, 1)
        self.to_action = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, action_dim * (2 if noisy else 1)))  # noisy: mean and log variance
        _init(self)
        nn.init.trunc_normal_(self.queries, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def tokens(self, aligned, later):
        # -> [B, Q + 1024, E]: queries first, position embedding added
        f_a, f_b = self.stem(hide_margin(aligned)), self.stem(hide_margin(later))  # [B, 1024, D] x 2
        tokens = self.proj(torch.cat([f_a, f_b, f_b - f_a], -1))  # [B, 1024, E]
        queries = self.queries.expand(tokens.shape[0], -1, -1)  # [B, Q, E]
        return torch.cat([queries, tokens], 1) + self.pos

    def query_outputs(self, aligned, later):
        return self.norm(self.transformer(self.tokens(aligned, later)))[:, :self.n_queries]  # [B, Q, E]

    def head(self, q):
        # q: [B, Q, E] -> [B, A] or, noisy, [B, 2 A]
        weights = self.query_weight(q).softmax(1)  # [B, Q, 1]
        return self.to_action((weights * q).sum(1))

    def merge(self, q):
        # q: [B, Q, E] -> a [B, A], without noise
        return self.head(q)[:, :self.action_dim]

    def sample(self, q):
        # q: [B, Q, E] -> (a = mean + noise [B, A], information it carries in nats per pair, KL to a unit Gaussian)
        mean, log_var = self.head(q).chunk(2, dim=1)
        a = mean + (0.5 * log_var).exp() * torch.randn_like(mean)
        return a, 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1).sum(1).mean()

    def forward(self, aligned, later):
        return self.merge(self.query_outputs(aligned, later))

    @torch.no_grad()
    def attention(self, aligned, later):
        # for viewing: last layer's attention of each query -> (maps [B, Q, 32, 32], query weights [B, Q])
        x = self.tokens(aligned, later)
        layers = self.transformer.layers
        for layer in layers[:-1]:
            x = layer(x)
        last = layers[-1]
        n = last.norm1(x)
        _, att = last.self_attn(n, n, n, need_weights=True, average_attn_weights=True)  # [B, Q + 1024, Q + 1024]
        maps = att[:, :self.n_queries, self.n_queries:]  # [B, Q, 1024]
        side = int(maps.shape[-1] ** 0.5)
        weights = self.query_weight(self.query_outputs(aligned, later)).softmax(1)[..., 0]
        return maps.view(*maps.shape[:2], side, side), weights


class FiLMBlock(nn.Module):
    # pre-norm transformer block whose two norms scale and shift the tokens from the action
    def __init__(self, dim, heads, mlp, action_dim):
        super().__init__()
        self.norm_attn = AdaptiveNormalizer(dim, action_dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_mlp = AdaptiveNormalizer(dim, action_dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))

    def forward(self, x, a):
        # x: [B, P, E], a: [B, A]; AdaptiveNormalizer expects a time axis, here of length 1
        n = self.norm_attn(x[:, None], a[:, None])[:, 0]
        x = x + self.attn(n, n, n, need_weights=False)[0]
        n = self.norm_mlp(x[:, None], a[:, None])[:, 0]
        return x + self.mlp(n)


class Decoder(nn.Module):
    """(aligned frame t [B, 3, 128, 128], a [B, A]) -> correction [B, 3, 128, 128], zero before training."""

    def __init__(self, action_dim, frame_size=128, patch_size=8, dim=384, depth=8, heads=6, mlp=1536):
        super().__init__()
        self.patch_size, self.side = patch_size, frame_size // patch_size
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.side ** 2, dim))
        self.blocks = nn.ModuleList(FiLMBlock(dim, heads, mlp, action_dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-12)
        self.head = nn.Conv2d(dim, patch_size ** 2 * 3, 1)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, aligned, a):
        x = self.patch_embed(aligned).flatten(2).transpose(1, 2) + self.pos  # [B, 256, E]
        for block in self.blocks:
            x = block(x, a)
        x = self.norm(x).transpose(1, 2).reshape(x.shape[0], -1, self.side, self.side)  # [B, E, 16, 16]
        return F.pixel_shuffle(self.head(x), self.patch_size)  # [B, 3, 128, 128]


class ShiftPredictor(nn.Module):
    """(frame t [B, 3, 128, 128], a [B, A]) -> predicted shift (dy, dx) [B, 2], in pixels."""

    def __init__(self, action_dim, widths=(32, 64, 128, 128, 128), hidden=256):
        super().__init__()
        layers, width = [], 3
        for out in widths:  # each layer halves the frame: 128 -> 4 px
            layers += [nn.Conv2d(width, out, 3, 2, 1), nn.GroupNorm(8, out), nn.GELU()]
            width = out
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(width * 4 * 4 + action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, frame, a):
        return self.head(torch.cat([self.net(frame).flatten(1), a], 1))


def relative_shift_error(pred, s):
    # |pred - s| / max(|s|, 1 px), per pair: pred, s [B, 2] -> [B]
    return (pred - s).norm(dim=1) / s.norm(dim=1).clamp(min=1)


class SmallActionModel(nn.Module):
    """Action encoder + decoder + shift predictor. The shift predictor gets no gradient path into the action."""

    def __init__(self, action_dim=4, dec_dim=384, dec_depth=8, dec_heads=6, dec_mlp=1536, contrastive_weight=0.004, temperature=0.1, kl_weight=0.0,
                 error_weight=0.0, lpips_weight=0.0, lpips_crop=0, pixel_loss='mse', edge_weight=0.0, edge_crop=32):
        super().__init__()
        self.kl_weight = kl_weight
        self.encoder = ActionEncoder(action_dim, noisy=kl_weight > 0)
        self.decoder = Decoder(action_dim, dim=dec_dim, depth=dec_depth, heads=dec_heads, mlp=dec_mlp)
        self.shift_predictor = ShiftPredictor(action_dim)
        self.contrastive_weight, self.temperature = contrastive_weight, temperature
        self.error_weight, self.lpips_weight, self.lpips_crop = error_weight, lpips_weight, lpips_crop
        assert pixel_loss in ('mse', 'l1'), pixel_loss
        self.pixel_loss, self.edge_weight, self.edge_crop = pixel_loss, edge_weight, edge_crop
        self.__dict__['lpips_fn'] = None  # set by training when lpips_weight > 0; kept out of the state dict

    def lpips_term(self, pred, aligned, target, known):
        # LPIPS on known pixels (unknown ones copied from the target): whole frame, or lpips_crop > 0: the crop holding the most error
        pred = torch.where(known.bool(), pred, target).clamp(-1, 1)
        if self.lpips_crop:
            pred, target = residual_crops(pred, target, aligned, known, self.lpips_crop)
            pred, target = (F.interpolate(x, size=LPIPS_SIZE, mode='bilinear', align_corners=False) for x in (pred, target))
        return self.lpips_fn(pred, target).mean()

    def edge_term(self, pred, aligned, target, known):
        # mean |edges of the prediction - edges of the target| on the edge_crop px window holding the most error (0: the whole frame);
        # an edge = the difference between two neighbouring pixels, down and right. A smeared patch has weak edges where the target's
        # are sharp, which the pixel loss alone barely penalises. No network
        pred = torch.where(known.bool(), pred, target)
        if self.edge_crop:
            pred, target = residual_crops(pred, target, aligned, known, self.edge_crop)
        down = lambda x: x[..., 1:, :] - x[..., :-1, :]  # noqa: E731
        right = lambda x: x[..., :, 1:] - x[..., :, :-1]  # noqa: E731
        return ((down(pred) - down(target)).abs().mean() + (right(pred) - right(target)).abs().mean()) / 2

    def predict(self, x_t, x_later, s):
        # -> (prediction of frame t+n, a)
        aligned = shift_frames(x_t, s)
        a = self.encoder(aligned, x_later)
        return aligned + self.decoder(aligned, a), a

    def reconstruction_loss(self, pred, aligned, target, known):
        # pixel loss over the scored pixels `known` [B, 1, H, W], optionally weighted by each pixel's error after the shift, plus an optional
        # LPIPS term:  sum_p w_p |pred_p - target_p|^k / (3 sum_p w_p) + lpips_weight * LPIPS(crop),  k = 2 (pixel_loss mse) or 1 (l1)
        weight = pixel_weights(aligned, target, known, self.error_weight) if self.error_weight > 0 else known
        per_pixel = (pred - target).abs() if self.pixel_loss == 'l1' else (pred - target).pow(2)  # [B, 3, H, W]
        loss = (per_pixel * weight).sum() / (weight.sum() * 3).clamp(min=1)
        return loss + self.lpips_weight * self.lpips_term(pred, aligned, target, known) if self.lpips_weight > 0 else loss

    def contrastive_loss(self, a, x_t, x_b, x_j, s_b, s_j, cut):
        # InfoNCE on a: a(t, t+n+delta) is the positive, the reversed a(t+n, t) the negative. As in CoMo, the two extra
        # branches run the transformer without gradient and only train the merge and action heads. cut(frame, shift of its pair): the window
        with torch.no_grad():
            q_pos = self.encoder.query_outputs(cut(shift_frames(x_t, s_j), s_j), cut(x_j, s_j))
            q_neg = self.encoder.query_outputs(cut(shift_frames(x_b, -s_b), -s_b), cut(x_t, -s_b))
        return info_nce(a, self.encoder.merge(q_pos), self.encoder.merge(q_neg), self.temperature)

    def forward(self, x_t, x_b, x_j, s_b, s_j, window=None):
        # x_t, x_b, x_j: frames t, t+n, t+n+delta [B, 3, 128, 128] in [-1, 1]; s_b, s_j: their measured shifts from t [B, 2]
        # window: None, or the jitter of each pair's window [B, 2] long (WHOLE = whole pair; window_corner). The shift comes first, on the
        # whole frames; the encoder and the decoder then see the window of the aligned pair; the shift predictor still reads the whole frame t
        cut = (lambda x, s: window_canvas(x, window_corner(s, window))) if window is not None else (lambda x, s: x)
        aligned, target = cut(shift_frames(x_t, s_b), s_b), cut(x_b, s_b)
        q = self.encoder.query_outputs(aligned, target)  # [B, Q, E]
        a, kl = self.encoder.sample(q) if self.kl_weight > 0 else (self.encoder.merge(q), torch.zeros(()))  # [B, A]
        pred = aligned + self.decoder(aligned, a)
        known = scored_pixels(s_b, window_corner(s_b, window) if window is not None else None)
        recon = self.reconstruction_loss(pred, aligned, target, known)
        edge = self.edge_term(pred, aligned, target, known) if self.edge_weight > 0 else torch.zeros(())
        nce = self.contrastive_loss(a, x_t, x_b, x_j, s_b, s_j, cut) if self.contrastive_weight > 0 else torch.zeros(())
        predicted_shift = self.shift_predictor(x_t, a.detach())  # no gradient into the action
        shift_error = relative_shift_error(predicted_shift, s_b).mean()
        return {'loss': recon + self.edge_weight * edge + self.contrastive_weight * nce + self.kl_weight * kl,
                'recon': recon, 'edge': edge, 'nce': nce, 'kl': kl, 'shift_error': shift_error,
                'a': a, 'pred': pred, 'aligned': aligned}
