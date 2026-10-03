"""LAOF latent action model (STA-39): "robust Latent Action learning with Optical Flow constraints", arXiv 2511.16407.

Port of the released PROCGEN code (github.com/XizoB/LAOF @ 29c8d97, agent/models.py), not of the paper's DINOv2
variant (unreleased):
  - IDM: IMPALA CNN on the stacked pair [o_t, o_{t+1}] -> z [B, A] (A = 128), then either an EMA vector quantizer
    (discrete; VQEmbeddingEMA, models.py:1696) or mu / log-var with reparameterisation (continuous; IDM_Continue,
    models.py:2065, KL weight 0 in their scripts)
  - FDM: U-Net (o_t, z) -> o_{t+1}, z tiled at the input and injected at the 1x1 bottleneck (WorldModel, models.py:192)
  - flow decoder: the same U-Net fed only z (FlowDecoder_WOX1, models.py:627) -> the RGB-encoded optical flow of the
    transition. It must not see o_t: with the frame it reads the motion off the image and bypasses z (paper ablation).
  loss = MSE(FDM, o_{t+1}) + MSE(flow decoder, rgb(flow)) + VQ commitment (or kl_weight * KL); pixels in [-0.5, 0.5].
  flow_target 'global' (STA-39 deviation): an MLP on z predicts a position-free summary of the flow instead of the
  image (global_flow_target): to draw the flow image from z alone, z must also encode WHERE the mover is, and on Zelda
  the k-means codes split as much by Link's screen cell as by his direction (NMI 0.253 vs 0.258, step 15k).
Generalised to any power-of-two frame size (the U-Net gets log2(H) - 1 down levels, channels capped at 32 * base).

Exposes the interface the LAM evals use (scripts/eval/lam_judge.py): encode([B, T, C, H, W] in [-1, 1]) ->
[B, T-1, A], action_dim, continuous_actions, quantizer.codebook_size / get_indices_from_latents.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_to_rgb(flow, sigma):
    """LAOF's "paper" flow image (sample_datasets_with_opticalflow_sam_masknums.py:91): hue = direction,
    saturation = value = min(1, |flow| / (sigma * image diagonal)).
    flow: [B, 2, H, W] (dx, dy) in px -> [B, 3, H, W] RGB in [-0.5, 0.5] (the models' pixel range)."""
    u, v = flow[:, 0], flow[:, 1]  # [B, H, W]
    H, W = flow.shape[-2:]
    hue = (torch.atan2(v, u) + math.pi) / (2 * math.pi)  # [B, H, W] in [0, 1]
    m = (torch.sqrt(u ** 2 + v ** 2) / (sigma * math.sqrt(H ** 2 + W ** 2))).clamp(max=1.0)  # [B, H, W]
    # HSV -> RGB with S = V = m: channel n gets V - V * S * clamp(min(k, 4 - k), 0, 1), k = (n + 6 * hue) mod 6
    k = (torch.tensor([5.0, 3.0, 1.0], device=flow.device)[None, :, None, None] + 6 * hue[:, None]) % 6  # [B, 3, H, W]
    rgb = m[:, None] - m[:, None] ** 2 * torch.minimum(k, 4 - k).clamp(0, 1)  # [B, 3, H, W] in [0, 1]
    return rgb - 0.5


def global_flow_target(flow, px=1.0, scale=4.0, max_pixels=2000):
    """Position-free summary of a (camera-masked) flow field: mean flow of the moving pixels (|flow| > px) in units of
    `scale` px, clamped to [-2, 2], and log(1 + n moving) / log(1 + max_pixels) (how much moves; 0 = still).
    flow: [B, 2, H, W] -> [B, 3]."""
    moving = (flow.norm(dim=1) > px).float()  # [B, H, W]
    n = moving.flatten(1).sum(1)  # [B]
    mean = (flow * moving[:, None]).flatten(2).sum(2) / n.clamp(min=1)[:, None]  # [B, 2]
    amount = torch.log1p(n) / math.log1p(max_pixels)
    return torch.cat([(mean / scale).clamp(-2, 2), amount.clamp(max=1)[:, None]], 1)


def camera_mask(fa, fb, shift, tol=0.1, dilate=2):
    """Pixels the camera shift does NOT explain (STA-39 refinement of the camera compensation): RAFT leaves large
    spurious residuals on flat / repetitive pixel-art backgrounds while the camera pans; a pixel whose frame-t+gap
    colour matches frame t sampled at x - shift (max channel error, 3x3-eroded, <= tol) is background, its flow is
    zeroed. The rest (Link, enemies, NPCs) is dilated by `dilate` px; the strip entering the view is dropped.
    fa, fb: [B, C, H, W] in [-1, 1]; shift: [B, 2] px (dx, dy) -> bool [B, H, W]."""
    B, _, H, W = fa.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=fa.device, dtype=fa.dtype), torch.arange(W, device=fa.device, dtype=fa.dtype), indexing='ij')
    sx, sy = xs[None] - shift[:, 0, None, None], ys[None] - shift[:, 1, None, None]  # [B, H, W] source coordinates
    grid = torch.stack([sx / (W - 1) * 2 - 1, sy / (H - 1) * 2 - 1], -1)  # [B, H, W, 2]
    warped = F.grid_sample(fa, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # [B, C, H, W]
    err = ((warped - fb).abs().amax(1, keepdim=True)) / 2  # [B, 1, H, W] in [0, 1] colour units
    err = -F.max_pool2d(-err, 3, 1, 1)  # erode: ignore 1-px resampling edges
    m = err > tol
    if dilate:
        m = F.max_pool2d(m.float(), 2 * dilate + 1, 1, dilate) > 0
    inside = (sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1)
    return m[:, 0] & inside


class ResidualLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(nn.ReLU(), nn.Conv2d(dim, hidden_dim, 3, 1, 1), nn.ReLU(), nn.Conv2d(hidden_dim, dim, 3, 1, 1))

    def forward(self, x):
        return x + self.block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_depth, out_depth, 3, 1, 1), nn.BatchNorm2d(out_depth),
                                 ResidualLayer(out_depth, out_depth // 2), nn.MaxPool2d(2, 2), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(in_depth, out_depth, 2, 2), nn.BatchNorm2d(out_depth),
                                 ResidualLayer(out_depth, out_depth // 2), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class ActionUNet(nn.Module):
    """LAOF's U-Net world model (WorldModel) and, with in_depth 0, its z-only flow decoder (FlowDecoder_WOX1)."""

    def __init__(self, action_dim, in_depth, out_depth, frame_size, base=24):
        super().__init__()
        n_down = int(math.log2(frame_size)) - 1  # down to 2x2, then a 2x2 conv to the 1x1 bottleneck
        assert 2 ** (n_down + 1) == frame_size, 'frame_size must be a power of two'
        self.in_depth = in_depth
        widths = [min(base * 2 ** i, 32 * base) for i in range(n_down + 1)]  # 64 px: 24 ... 768 as in LAOF
        down = [in_depth + action_dim] + widths
        self.down = nn.ModuleList([DownsampleBlock(down[i], down[i + 1]) for i in range(n_down)]
                                  + [nn.Conv2d(down[-2], down[-1], 2, 1)])
        up = widths[::-1] + [base]
        self.up = nn.ModuleList([UpsampleBlock(up[i] + (action_dim if i == 0 else down[-i - 1]), up[i + 1])
                                 for i in range(n_down + 1)])
        self.final = nn.Sequential(nn.Conv2d(base + in_depth, base, 3, 1, 1), ResidualLayer(base, base // 2), nn.ReLU(),
                                   nn.Conv2d(base, out_depth, 1, 1))
        self.frame_size = frame_size

    def forward(self, state, action):
        # state: [B, C, H, W] in [-0.5, 0.5] or None (flow decoder), action: [B, A] -> [B, out, H, W] in [-0.5, 0.5]
        a = action[:, :, None, None]  # [B, A, 1, 1]
        tiled = a.expand(-1, -1, self.frame_size, self.frame_size)  # [B, A, H, W]
        x = tiled if state is None else torch.cat([state, tiled], 1)
        xs = []
        for layer in self.down:
            x = layer(x)
            xs.append(x)
        xs[-1] = a  # the action replaces the bottleneck skip
        for i, layer in enumerate(self.up):
            x = layer(torch.cat([x, xs[-i - 1]], 1))
        if state is not None:
            x = torch.cat([x, state], 1)
        return torch.tanh(self.final(x)) / 2


class ImpalaBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv0, self.conv1 = nn.Conv2d(ch, ch, 3, padding=1), nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.conv1(F.relu(self.conv0(F.relu(x))))


class ImpalaIDM(nn.Module):
    """IMPALA CNN (scale x (16, 32, 32), 3 conv sequences) -> FC 256 -> head (get_impala, models.py:1365)."""

    def __init__(self, frame_size, out_dim, scale=4, channels=(16, 32, 32), features=256, in_ch=6):
        super().__init__()
        layers, c, s = [], in_ch, frame_size
        for ch in channels:
            layers += [nn.Conv2d(c, scale * ch, 3, padding=1), nn.MaxPool2d(3, 2, 1), ImpalaBlock(scale * ch), ImpalaBlock(scale * ch)]
            c, s = scale * ch, (s + 1) // 2
        self.conv = nn.Sequential(*layers, nn.Flatten(), nn.ReLU())
        self.fc = nn.Linear(c * s * s, features)
        self.head = nn.Linear(features, out_dim)

    def forward(self, pair):
        # pair: [B, 2, C, H, W] in [-0.5, 0.5] -> [B, out_dim]
        return self.head(F.relu(self.fc(self.conv(pair.flatten(1, 2)))))


class VQEmbeddingEMA(nn.Module):
    """LAOF's EMA vector quantizer: z [B, A] is split into num_codebooks x num_latents chunks of emb_dim; each chunk
    snaps to the nearest of num_embs entries of its codebook (EMA updates, no codebook gradient).
    Deviation from LAOF (restart_threshold > 0): the codebook is initialised from the first batch and an entry whose EMA
    usage drops below restart_threshold hits per batch is reset to a random current latent. Without it, an entry that
    is never picked gets ema_count ~ 0, its embedding ema_weight / ema_count explodes, and it is unreachable for good
    (with 16 codes everything collapsed onto one code at step 1 in the CPU check)."""

    def __init__(self, num_codebooks=1, num_latents=1, emb_dim=128, num_embs=16, commitment=0.05, decay=0.999, eps=1e-5,
                 restart_threshold=0.0):
        super().__init__()
        self.N, self.L, self.D, self.M = num_codebooks, num_latents, emb_dim, num_embs
        self.commitment, self.decay, self.eps, self.restart_threshold = commitment, decay, eps, restart_threshold
        self.register_buffer('initialised', torch.tensor(restart_threshold <= 0))
        emb = torch.empty(num_codebooks, num_embs, emb_dim).uniform_(-5 / num_embs, 5 / num_embs)
        self.register_buffer('embedding', emb)
        self.register_buffer('ema_count', torch.zeros(num_codebooks, num_embs))
        self.register_buffer('ema_weight', emb.clone())

    @property
    def codebook_size(self):
        return self.M ** (self.N * self.L)

    def _split(self, z):
        # z: [..., A] -> [N, B', D] chunks (codebook-major), A = N * L * D
        return z.reshape(-1, self.N, self.L, self.D).permute(1, 0, 2, 3).reshape(self.N, -1, self.D)

    def _indices(self, flat):
        # flat: [N, B', D] -> [N, B'] nearest entries
        return torch.cdist(flat.float(), self.embedding.float()).argmin(-1)

    def forward(self, z):
        # z: [B, A] -> z_q [B, A] (straight-through), commitment loss, perplexity, indices [B, N, L]
        B = z.shape[0]
        flat = self._split(z.detach())  # [N, B*L, D]
        if self.training and not self.initialised:
            pick = torch.randint(flat.shape[1], (self.N, self.M), device=z.device)
            self.embedding.copy_(torch.gather(flat, 1, pick[..., None].expand(-1, -1, self.D)))
            self.ema_weight.copy_(self.embedding)
            self.ema_count.fill_(1.0)
            self.initialised.fill_(True)
        idx = self._indices(flat)  # [N, B*L]
        enc = F.one_hot(idx, self.M).float()  # [N, B*L, M]
        zq = torch.gather(self.embedding, 1, idx[..., None].expand(-1, -1, self.D))  # [N, B*L, D]
        if self.training:
            self.ema_count.mul_(self.decay).add_((1 - self.decay) * enc.sum(1))
            n = self.ema_count.sum(-1, keepdim=True)
            self.ema_count.copy_((self.ema_count + self.eps) / (n + self.M * self.eps) * n)
            self.ema_weight.mul_(self.decay).add_((1 - self.decay) * torch.bmm(enc.transpose(1, 2), flat.float()))
            self.embedding.copy_(self.ema_weight / self.ema_count[..., None])
            if self.restart_threshold > 0:
                dead = self.ema_count < self.restart_threshold  # [N, M]
                if dead.any():
                    pick = torch.randint(flat.shape[1], (self.N, self.M), device=z.device)
                    fresh = torch.gather(flat, 1, pick[..., None].expand(-1, -1, self.D))  # [N, M, D]
                    self.embedding.copy_(torch.where(dead[..., None], fresh, self.embedding))
                    self.ema_weight.copy_(torch.where(dead[..., None], fresh, self.ema_weight))
                    self.ema_count.copy_(torch.where(dead, torch.ones_like(self.ema_count), self.ema_count))
        zq = zq.reshape(self.N, B, self.L, self.D).permute(1, 0, 2, 3).reshape(B, -1)  # [B, A]
        loss = self.commitment * F.mse_loss(z, zq.detach())
        zq = zq.detach() + (z - z.detach())
        probs = enc.mean(1)  # [N, M]
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum(-1)).sum()
        return zq, loss, perplexity, idx.reshape(self.N, B, self.L).permute(1, 0, 2)

    def get_indices_from_latents(self, z):
        # z: [..., A] -> flat code index [...] (mixed radix over the N * L chunks)
        lead = z.shape[:-1]
        idx = self._indices(self._split(z)).reshape(self.N, -1, self.L).permute(1, 0, 2).reshape(-1, self.N * self.L)  # [B', N*L]
        radix = self.M ** torch.arange(self.N * self.L - 1, -1, -1, device=z.device)
        return (idx * radix).sum(-1).reshape(lead)


class KMeansQuantizer(nn.Module):
    """Stand-in for continuous LAOF: the LAM evals k-means the latents into codebook_size clusters."""

    def __init__(self, n_clusters):
        super().__init__()
        self.codebook_size = n_clusters


class LAOF(nn.Module):
    def __init__(self, frame_size=128, action_dim=128, continuous=False, num_codebooks=1, num_latents=1, num_embs=16,
                 commitment=0.05, decay=0.999, restart_threshold=0.0, kl_weight=0.0, flow_weight=1.0, flow_sigma=0.02, impala_scale=4,
                 unet_base=24, eval_clusters=16, flow_target='image'):
        super().__init__()
        self.action_dim, self.continuous_actions = action_dim, continuous
        self.kl_weight, self.flow_weight, self.flow_sigma = kl_weight, flow_weight, flow_sigma
        self.idm = ImpalaIDM(frame_size, 2 * action_dim if continuous else action_dim, impala_scale)
        if continuous:
            self.quantizer = KMeansQuantizer(eval_clusters)
        else:
            assert action_dim == num_codebooks * num_latents * (action_dim // (num_codebooks * num_latents))
            self.quantizer = VQEmbeddingEMA(num_codebooks, num_latents, action_dim // (num_codebooks * num_latents),
                                            num_embs, commitment, decay, restart_threshold=restart_threshold)
        self.fdm = ActionUNet(action_dim, 3, 3, frame_size, unet_base)
        self.flow_target = flow_target
        if flow_weight == 0:
            self.flow_decoder = None
        elif flow_target == 'global':
            self.flow_decoder = nn.Sequential(nn.Linear(action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3))
        else:
            self.flow_decoder = ActionUNet(action_dim, 0, 3, frame_size, unet_base)

    def infer_action(self, pair):
        # pair: [B, 2, C, H, W] in [-0.5, 0.5] -> action [B, A], regulariser, stats
        z = self.idm(pair)
        if self.continuous_actions:
            mu, logvar = z.chunk(2, -1)
            a = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1).mean()
            return a, self.kl_weight * kl, {'kl': kl.detach(), 'action_std': torch.exp(0.5 * logvar).mean().detach()}
        zq, vq_loss, perp, _ = self.quantizer(z)
        return zq, vq_loss, {'vq_loss': vq_loss.detach(), 'perplexity': perp.detach()}

    def forward(self, frames, flow):
        # frames: [B, 2, C, H, W] in [-1, 1]; flow: [B, 2, H, W] camera-compensated px -> loss, logs, predictions
        x = frames / 2
        a, reg, logs = self.infer_action(x)
        pred = self.fdm(x[:, 0], a)  # [B, C, H, W]
        recon = F.mse_loss(pred, x[:, 1])
        loss = recon + reg
        logs['recon_loss'] = recon.detach()
        flow_pred = flow_target = None
        if self.flow_decoder is not None:
            if self.flow_target == 'global':
                flow_target = global_flow_target(flow)  # [B, 3]
                flow_pred = self.flow_decoder(a)
            else:
                flow_target = flow_to_rgb(flow, self.flow_sigma)  # [B, 3, H, W]
                flow_pred = self.flow_decoder(None, a)
            flow_loss = F.mse_loss(flow_pred, flow_target)
            loss = loss + self.flow_weight * flow_loss
            logs['flow_loss'] = flow_loss.detach()
        logs['loss'] = loss.detach()
        return loss, logs, {'action': a, 'pred': pred, 'flow_pred': flow_pred, 'flow_target': flow_target}

    @torch.no_grad()
    def encode(self, x):
        # x: [B, T, C, H, W] in [-1, 1] -> [B, T-1, A] (quantized latents, or the mean if continuous)
        B, T = x.shape[:2]
        pairs = torch.stack([x[:, :-1], x[:, 1:]], 2).flatten(0, 1) / 2  # [B*(T-1), 2, C, H, W]
        z = self.idm(pairs)
        if self.continuous_actions:
            z = z.chunk(2, -1)[0]
        else:
            idx = self.quantizer._indices(self.quantizer._split(z))  # [N, B*(T-1)*L]
            z = torch.gather(self.quantizer.embedding, 1, idx[..., None].expand(-1, -1, self.quantizer.D))
            q = self.quantizer
            z = z.reshape(q.N, -1, q.L, q.D).permute(1, 0, 2, 3).reshape(-1, self.action_dim)
        return z.reshape(B, T - 1, -1)
