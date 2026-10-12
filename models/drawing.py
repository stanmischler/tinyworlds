"""STA-76 drawing model: encoder, CoMo action model and ONE decoder trained jointly from scratch; the decoder is both the
frame decoder (NULL action) and the dynamics model (inferred action).

  - encoder  : frame [N, C, H, W] -> per-frame ST-transformer (as J1's tokenizer encoder: patch 4, 256-d, 8 blocks)
               -> LayerNorm -> 2x2 space-to-depth -> z [N, S=256, D=1024] (continuous, no bottleneck)
  - IDM      : CoMo motion IDM on (z(t), z(t+n) - z(t)) -> action a [B, Q=8, L=16] (+ InfoNCE, as models/como.py CoMo)
  - decoder  : z(t) [B, S, D] + action (a, or a learned NULL action) -> frame [B, C, H, W]; ST-transformer (spatial only,
               QK-norm), action pooled CoMo-style and added to every token, 8 px output patches.
               decoder_type 'plain': one pass, pixel MSE.
               decoder_type 'fm'   : rectified flow in pixel space (STA-28/62 recipe): x_s = (1 - s) x1 + s x0, x0 ~ N(0, I),
               x1 = frame / pixel_scale (unit RMS), the noisy frame enters as 8 px patches added to the tokens, the flow time
               through per-layer FiLM; v-pred, timestep shift; sampled with Euler steps from noise.
Every step decodes 3 targets in one batched pass: (z(t), NULL) -> frame t, (z(t+n), NULL) -> frame t+n (redraw), and
(z(t), a) -> frame t+n (predict). loss = predict + redraw_weight * mean(redraw) + contrastive_weight * InfoNCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.como import MotionIDM, _init, info_nce, merge_tokens
from models.flow_dynamics import shift_noise_level, timestep_embedding
from models.patch_embed import PatchEmbedding
from models.st_transformer import STTransformer


class DrawingEncoder(nn.Module):
    """Per-frame encoder: frames [N, Th=1, C, H, W] (CoMoLAM features interface) or [N, C, H, W] -> z [N, S, D]."""

    def __init__(self, frame_size=128, patch_size=4, embed_dim=256, num_heads=8, hidden_dim=512, num_blocks=8, merge=2):
        super().__init__()
        self.patch_embed = PatchEmbedding((frame_size, frame_size), patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True, temporal=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.hp, self.merge = frame_size // patch_size, merge
        self.dim = embed_dim * merge * merge
        self.n_tokens = (self.hp // merge) ** 2

    def forward(self, x):
        if x.dim() == 5:  # [N, 1, C, H, W]
            x = x[:, -1]
        h = self.norm(self.transformer(self.patch_embed(x[:, None]))[:, 0])  # [N, P, E]
        return merge_tokens(h, self.hp, self.merge)  # [N, S, D]


class DrawingDecoder(nn.Module):
    """z [B, S, D] + action [B, Q, L] (+ noisy frame [B, C, H, W] and flow time [B] for FM) -> frame (plain) or velocity
    (FM) [B, C, H, W]."""

    def __init__(self, frame_size=128, z_dim=1024, n_tokens=256, dim=768, depth=12, heads=12, mlp=3072, n_queries=8,
                 latent_dim=16, channels=3, fm=False, time_dim=64):
        super().__init__()
        self.hp = int(n_tokens ** 0.5)
        self.patch = frame_size // self.hp
        self.fm, self.time_dim = fm, time_dim
        self.z_proj = nn.Linear(z_dim, dim)
        self.up = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh(), nn.Linear(latent_dim, dim))  # as MotionDecoder
        self.query_pool = nn.Linear(n_queries * dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        if fm:
            self.noisy_embed = nn.Conv2d(channels, dim, self.patch, self.patch)
            self.time_mlp = nn.Sequential(nn.Linear(time_dim, 2 * time_dim), nn.SiLU(), nn.Linear(2 * time_dim, time_dim))
        self.transformer = STTransformer(dim, heads, mlp, depth, causal=False, conditioning_dim=time_dim if fm else None,
                                         temporal=False, qk_norm=True)
        self.head = nn.Linear(dim, self.patch * self.patch * channels)
        _init(self)
        nn.init.trunc_normal_(self.pos, std=0.02)
        if fm:  # zero-init output: the flow starts by predicting v = 0 (as FlowDynamicsModel)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, z, action, noisy=None, t=None):
        B = z.shape[0]
        x = self.z_proj(z) + self.query_pool(self.up(action).reshape(B, 1, -1)) + self.pos  # [B, S, E]
        cond = None
        if self.fm:
            x = x + self.noisy_embed(noisy).flatten(2).transpose(1, 2)  # [B, S, E]
            cond = self.time_mlp(timestep_embedding(t, self.time_dim))[:, None]  # [B, 1, Dt]
        x = self.transformer(x[:, None], conditioning=cond)[:, 0]  # [B, S, E]
        x = self.head(x).transpose(1, 2).reshape(B, -1, self.hp, self.hp)  # [B, C*p*p, Hp, Wp]
        return F.pixel_shuffle(x, self.patch)  # [B, C, H, W]


class DrawingModel(nn.Module):
    """Encoder + IDM + one decoder; forward(x_a, x_b, x_j) on pixels [B, C, H, W] of frames t, t+n, t+n+delta -> losses."""

    def __init__(self, frame_size=128, enc_patch_size=4, enc_dim=256, enc_heads=8, enc_mlp=512, enc_depth=8, merge=2,
                 idm_dim=768, idm_depth=4, idm_heads=12, idm_mlp=3072, n_queries=8, latent_dim=16,
                 dec_dim=768, dec_depth=12, dec_heads=12, dec_mlp=3072, decoder_type='plain', redraw_weight=0.5,
                 contrastive_weight=0.004, temperature=0.1, fm_shift=3.0, time_dim=64):
        super().__init__()
        assert decoder_type in ('plain', 'fm'), decoder_type
        self.encoder = DrawingEncoder(frame_size, enc_patch_size, enc_dim, enc_heads, enc_mlp, enc_depth, merge)
        self.idm = MotionIDM(self.encoder.dim, self.encoder.n_tokens, idm_dim, n_queries, idm_depth, idm_heads, idm_mlp, latent_dim)
        self.decoder = DrawingDecoder(frame_size, self.encoder.dim, self.encoder.n_tokens, dec_dim, dec_depth, dec_heads,
                                      dec_mlp, n_queries, latent_dim, fm=decoder_type == 'fm', time_dim=time_dim)
        self.null_action = nn.Parameter(torch.zeros(1, n_queries, latent_dim))
        nn.init.trunc_normal_(self.null_action, std=0.02)
        self.register_buffer('pixel_scale', torch.ones(()))  # FM: RMS of the training frames in [-1, 1] (set by the trainer)
        self.decoder_type, self.redraw_weight = decoder_type, redraw_weight
        self.contrastive_weight, self.temperature, self.fm_shift = contrastive_weight, temperature, float(fm_shift)

    @property
    def fm(self):
        return self.decoder_type == 'fm'

    def null(self, B):
        return self.null_action.expand(B, -1, -1)  # [B, Q, L]

    def forward(self, x_a, x_b, x_j):
        B = x_a.shape[0]
        f = self.encoder(torch.cat([x_a, x_b]))  # [2B, S, D]; per frame, so t and t+n never attend to each other
        f_a, f_b = f[:B], f[B:]
        z = self.idm(f_a, f_b)  # [B, Q, L]
        out = {'z': z}
        loss = 0.0
        if self.contrastive_weight > 0:
            with torch.no_grad():  # as models/como.py CoMo: the extra branches train only the down-resampler
                f_j = self.encoder(x_j)
                t_neg = self.idm.tokens(f_b, f_a - f_b)
                t_pos = self.idm.tokens(f_a, f_j - f_a)
            out['nce'] = info_nce(z.reshape(B, -1), self.idm.down(t_pos).reshape(B, -1), self.idm.down(t_neg).reshape(B, -1),
                                  self.temperature)
            loss = self.contrastive_weight * out['nce']
        # one batched decoder pass: [redraw t | redraw t+n | predict t+n]
        cond = torch.cat([f_a, f_b, f_a])  # [3B, S, D]
        act = torch.cat([self.null(B), self.null(B), z])  # [3B, Q, L]
        target = torch.cat([x_a, x_b, x_b])  # [3B, C, H, W]
        if self.fm:
            x1 = target / self.pixel_scale
            x0 = torch.randn_like(x1)
            s = shift_noise_level(torch.rand(3 * B, device=x1.device), self.fm_shift)  # [3B] noise level
            xs = (1 - s)[:, None, None, None] * x1 + s[:, None, None, None] * x0
            pred = self.decoder(cond, act, xs, 1 - s)  # velocity
            err = (pred.float() - (x1 - x0)).pow(2).flatten(1).mean(1)  # [3B]
        else:
            pred = self.decoder(cond, act)
            err = (pred.float() - target).pow(2).flatten(1).mean(1)  # [3B]
            out['pred'] = pred[2 * B:]
        out['redraw'], out['predict'] = err[:2 * B].mean(), err[2 * B:].mean()
        out['loss'] = loss + out['predict'] + self.redraw_weight * out['redraw']
        return out

    @torch.no_grad()
    def draw(self, f, action, steps=1, generator=None):
        # z [B, S, D] + action [B, Q, L] -> frame [B, C, H, W] in [-1, 1] scale (plain: one pass; FM: Euler from noise)
        if not self.fm:
            return self.decoder(f, action)
        B, H = f.shape[0], self.decoder.hp * self.decoder.patch
        x = torch.randn((B, 3, H, H), device=f.device, generator=generator)
        s_grid = shift_noise_level(torch.linspace(1, 0, steps + 1, device=f.device), self.fm_shift)  # 1 -> 0
        for k in range(steps):
            v = self.decoder(f, action, x, (1 - s_grid[k]).expand(B)).float()
            x = x + (s_grid[k] - s_grid[k + 1]) * v
        return x * self.pixel_scale

    def predict(self, x_a, action, steps=1, generator=None):
        # pixels of frame t [B, C, H, W] + action [B, Q, L] -> predicted frame t+n
        return self.draw(self.encoder(x_a), action, steps, generator)

    def redraw(self, x, steps=1, generator=None):
        # frame [B, C, H, W] -> its reconstruction through z and the NULL action
        return self.draw(self.encoder(x), self.null(x.shape[0]), steps, generator)
