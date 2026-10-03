"""CoMo Motion IDM (Yang et al. 2025, arXiv 2505.17006; port of github.com/MCG-NJU/CoMo @62bd06d, idm/CoMo/
latent_motion_tokenizer): continuous latent motion between two frames, learned by forward prediction.

  - frozen MAE ViT-L/16 (facebook/vit-mae-large) features F of frame t and frame t+n: [B, S=197, 1024]
  - motion IDM: [8 learned queries, Lin(F_t), SEP, Lin(F_{t+n} - F_t)] -> 4-layer transformer -> first 8 outputs
    -> Lin-Tanh-Lin -> z [B, Q=8, L=16] (continuous, no quantizer; 128 numbers per transition)
  - forward dynamics decoder: frame t pixels (native 128 px, patch 8) + Lin(flatten(up(z))) added to every patch
    -> 12-layer ViT -> frame t+n pixels, MSE
  - InfoNCE (weight 0.004, temperature 0.1): z(t, t+n) vs the jittered z(t, t+n+delta) as positive, the reversed
    z(t+n, t) as negative; the Q-former runs without gradient on the two extra branches (as in CoMo's code)

Deviations from CoMo's released code, on purpose: MAE patch order is fixed (HF ViTMAE shuffles patches even at
mask_ratio 0, so CoMo's F_{t+n} - F_t subtracts unaligned patches), the difference tokens get their own token type
(CoMo: both type 0, flagged as a bug in their code), the decoder works on native 128 px Zelda frames with 8 px patches
(CoMo: 224 px, 16 px patches), LPIPS is dropped (CoMo computes it under no_grad, so it never trains anything).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

MAE_NAME = 'facebook/vit-mae-large'
MAE_DIM, MAE_TOKENS, MAE_SIZE = 1024, 197, 224
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _encoder(dim, heads, mlp, depth):
    # pre-LN ViT blocks (= HF ViTLayer: LN -> attn -> residual, LN -> GELU MLP -> residual)
    layer = nn.TransformerEncoderLayer(dim, heads, mlp, dropout=0.0, activation='gelu', batch_first=True, norm_first=True,
                                       layer_norm_eps=1e-12)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


def _init(module, std=0.02):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class MAEFeatures(nn.Module):
    """Frozen MAE ViT-L/16 encoder, patch order fixed: frames in [-1, 1] [N, C, H, W] -> last_hidden_state [N, S, 1024]."""

    def __init__(self, name=MAE_NAME):
        super().__init__()
        from transformers import ViTMAEModel
        self.vit = ViTMAEModel.from_pretrained(name, mask_ratio=0.0).eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        return super().train(False)  # always eval

    @torch.no_grad()
    def forward(self, x):
        # x: [N, C, H, W] in [-1, 1] -> [N, S, 1024]
        x = F.interpolate((x + 1) / 2, size=(MAE_SIZE, MAE_SIZE), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        n_patch = (MAE_SIZE // 16) ** 2
        noise = torch.arange(n_patch, device=x.device, dtype=torch.float32).expand(x.shape[0], -1)  # identity shuffle
        return self.vit(pixel_values=x, noise=noise).last_hidden_state


class MotionIDM(nn.Module):
    """Q-former over [queries, cond tokens, SEP, difference tokens]: (F_t, F_{t+n} - F_t) [B, S, 1024] -> z [B, Q, L]."""

    def __init__(self, feat_dim=MAE_DIM, n_tokens=MAE_TOKENS, dim=768, n_queries=8, depth=4, heads=12, mlp=3072, latent_dim=16):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.zeros(1, n_queries, dim))
        self.sep = nn.Parameter(torch.zeros(1, 1, dim))
        self.proj = nn.Linear(feat_dim, dim)  # shared by cond and difference tokens (CoMo: one projection)
        self.pos = nn.Parameter(torch.zeros(1, n_queries + 2 * n_tokens + 1, dim))
        self.token_type = nn.Parameter(torch.zeros(2, dim))
        self.transformer = _encoder(dim, heads, mlp, depth)
        self.norm = nn.LayerNorm(dim, eps=1e-12)
        self.down = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, latent_dim))
        _init(self)
        for p in (self.queries, self.sep, self.pos, self.token_type):
            nn.init.trunc_normal_(p, std=0.02)

    def tokens(self, cond, diff):
        # cond, diff: [B, S, 1024] -> Q-former outputs at the query slots [B, Q, E]
        B, S = cond.shape[:2]
        x = torch.cat([self.queries.expand(B, -1, -1), self.proj(cond), self.sep.expand(B, -1, -1), self.proj(diff)], 1)
        n_cond = self.n_queries + S + 1
        x = x + self.pos + torch.cat([self.token_type[0].expand(n_cond, -1), self.token_type[1].expand(S, -1)], 0)
        return self.norm(self.transformer(x))[:, :self.n_queries]

    def forward(self, f_a, f_b):
        # f_a, f_b: [B, S, 1024] features of frame t and frame t+n -> z [B, Q, L]
        return self.down(self.tokens(f_a, f_b - f_a))


class MotionDecoder(nn.Module):
    """Forward dynamics model: frame t [B, C, H, W] + z [B, Q, L] -> frame t+n [B, C, H, W] (pixels in [-1, 1] scale)."""

    def __init__(self, frame_size=128, patch_size=8, dim=768, depth=12, heads=12, mlp=3072, n_queries=8, latent_dim=16, channels=3):
        super().__init__()
        self.patch_size, self.hp = patch_size, frame_size // patch_size
        self.up = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh(), nn.Linear(latent_dim, dim))
        self.patch_embed = nn.Conv2d(channels, dim, patch_size, patch_size)
        self.query_pool = nn.Linear(n_queries * dim, dim)  # CoMo's "pooling": one vector added to every patch
        self.pos = nn.Parameter(torch.zeros(1, self.hp * self.hp, dim))
        self.transformer = _encoder(dim, heads, mlp, depth)
        self.norm = nn.LayerNorm(dim, eps=1e-12)
        self.head = nn.Conv2d(dim, patch_size * patch_size * channels, 1)
        _init(self)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, frame, z):
        B = frame.shape[0]
        x = self.patch_embed(frame).flatten(2).transpose(1, 2)  # [B, P, E]
        x = x + self.query_pool(self.up(z).reshape(B, 1, -1)) + self.pos
        x = self.norm(self.transformer(x))  # [B, P, E]
        x = x.transpose(1, 2).reshape(B, -1, self.hp, self.hp)
        return F.pixel_shuffle(self.head(x), self.patch_size)  # [B, C, H, W]


def info_nce(z, z_pos, z_neg, temperature=0.1):
    # CoMo compute_cross_contrastive_loss: z, z_pos, z_neg [B, D]; positive (z, z_pos), negatives (z_neg, z) and (z_neg, z_pos)
    z, z_pos, z_neg = (F.normalize(t, dim=-1) for t in (z, z_pos, z_neg))
    logits = torch.stack([(z * z_pos).sum(-1), (z_neg * z).sum(-1), (z_neg * z_pos).sum(-1)], 1) / temperature
    return -logits.log_softmax(1)[:, 0].mean()


class CoMo(nn.Module):
    """Trainable part (IDM + decoder); the frozen MAE is not a submodule, so checkpoints hold only trained weights."""

    def __init__(self, frame_size=128, patch_size=8, idm_dim=768, idm_depth=4, idm_heads=12, idm_mlp=3072, n_queries=8,
                 latent_dim=16, dec_dim=768, dec_depth=12, dec_heads=12, dec_mlp=3072, contrastive_weight=0.004,
                 temperature=0.1):
        super().__init__()
        self.idm = MotionIDM(MAE_DIM, MAE_TOKENS, idm_dim, n_queries, idm_depth, idm_heads, idm_mlp, latent_dim)
        self.decoder = MotionDecoder(frame_size, patch_size, dec_dim, dec_depth, dec_heads, dec_mlp, n_queries, latent_dim)
        self.contrastive_weight, self.temperature = contrastive_weight, temperature

    def forward(self, f_a, f_b, f_j, x_a, x_b):
        # f_a, f_b, f_j: MAE features [B, S, 1024] of frames t, t+n, t+n+delta; x_a, x_b: pixels [B, C, H, W] of t, t+n
        z = self.idm(f_a, f_b)  # [B, Q, L]
        pred = self.decoder(x_a, z)
        recon = F.mse_loss(pred, x_b)
        out = {'recon': recon, 'z': z, 'pred': pred}
        loss = recon
        if self.contrastive_weight > 0:
            with torch.no_grad():  # as in CoMo: the extra branches train only the down-resampler
                t_neg = self.idm.tokens(f_b, f_a - f_b)
                t_pos = self.idm.tokens(f_a, f_j - f_a)
            B = z.shape[0]
            nce = info_nce(z.reshape(B, -1), self.idm.down(t_pos).reshape(B, -1), self.idm.down(t_neg).reshape(B, -1), self.temperature)
            out['nce'] = nce
            loss = loss + self.contrastive_weight * nce
        out['loss'] = loss
        return out


class CoMoLAM(nn.Module):
    """Eval adapter with the LatentActionModel interface used by scripts/eval (lam_judge, eval_lam's kmeans path):
    encode(frames [B, T, C, H, W] in [-1, 1]) -> continuous actions [B, T-1, A=Q*L], one per consecutive pair."""

    continuous_actions = True

    def __init__(self, como, n_clusters=16, mae=None):
        super().__init__()
        self.como = como
        self.mae = mae if mae is not None else MAEFeatures()
        self.action_dim = como.idm.n_queries * como.idm.down[-1].out_features
        self.quantizer = SimpleNamespace(codebook_size=n_clusters)  # k-means clusters used to turn z into codes

    @torch.no_grad()
    def encode(self, frames):
        B, T = frames.shape[:2]
        f = self.mae(frames.flatten(0, 1)).float().view(B, T, MAE_TOKENS, MAE_DIM)  # [B, T, S, 1024]
        z = self.como.idm(f[:, :-1].flatten(0, 1), f[:, 1:].flatten(0, 1))  # [B*(T-1), Q, L]
        return z.reshape(B, T - 1, -1)


class CoMoActions(nn.Module):
    """Dynamics conditioning from a CoMo IDM (STA-42), with the LatentActionModel interface used by train_dynamics and
    eval_next_frame: encode(frames [B, T, C, H, W]) -> actions [B, T-1, A=Q*L], z standardized per dim with train stats;
    mode 'k16' (any 'k<N>') first snaps z to its nearest k-means centroid (fit on train z), so the dynamics model sees
    one of N vectors. quantizer: nearest-centroid codes (logging, eval lam_code) and centroid vectors (eval random)."""

    continuous_actions = True

    def __init__(self, lam, mean, std, centroids, mode='full'):
        super().__init__()
        self.lam, self.mode = lam, mode  # CoMoLAM (frozen MAE + IDM)
        self.action_dim = lam.action_dim
        self.register_buffer('mean', mean.float())  # [A]
        self.register_buffer('std', std.float())  # [A]
        self.register_buffer('centroids', centroids.float())  # [K, A] raw z space
        self.quantizer = _CentroidQuantizer(self)

    def codes(self, z):
        # raw z [..., A] -> nearest centroid [...]
        return torch.cdist(z.reshape(-1, z.shape[-1]).float(), self.centroids).argmin(1).view(z.shape[:-1])

    def standardize(self, z):
        # raw z [..., A] -> conditioning [..., A] (snapped first in k-modes)
        if self.mode != 'full':
            z = self.centroids[self.codes(z)]
        return (z.float() - self.mean) / self.std

    @torch.no_grad()
    def encode(self, frames):
        return self.standardize(self.lam.encode(frames))  # [B, T-1, A]


class _CentroidQuantizer:
    # not an nn.Module (would register a cycle); latents here are standardized conditioning vectors
    def __init__(self, owner):
        self.owner = owner
        self.codebook_size = owner.centroids.shape[0]

    def get_indices_from_latents(self, latents, dim=-1):
        o = self.owner
        return o.codes(latents.float() * o.std + o.mean)  # back to raw z, nearest centroid

    def get_latents_from_indices(self, indices, dim=-1):
        o = self.owner
        return (o.centroids[indices] - o.mean) / o.std
