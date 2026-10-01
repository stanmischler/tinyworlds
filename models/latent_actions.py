from models.utils import ModelType
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from einops import rearrange, repeat, reduce
from models.st_transformer import STTransformer, PatchEmbedding
from models.fsq import FiniteScalarQuantizer

NUM_LATENT_ACTIONS_BINS = 2

class LatentActionsEncoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8, 
                 hidden_dim=256, num_blocks=4, action_dim=3, pooling='mean', out_dim=None):
        super().__init__()
        # pooling: 'mean' = mean over patches, then concat(frame t, frame t+1) (original);
        #   'attention' = per patch concat(frame t, frame t+1), then a learned softmax over patches, so the action can
        #   come from the few patches that changed (a small sprite) instead of being averaged away
        # out_dim: head width (action_dim, or 2 * action_dim for a mean and a log-variance)
        assert pooling in ('mean', 'attention'), pooling
        self.pooling = pooling
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        if pooling == 'attention':
            self.pool_score = nn.Sequential(nn.LayerNorm(embed_dim * 2), nn.Linear(embed_dim * 2, 1))

        # embeddings to discrete latent bottleneck actions
        out_dim = out_dim or action_dim
        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 4 * out_dim),
            nn.GELU(),
            nn.Linear(4 * out_dim, out_dim)
        )

    def forward(self, frames):
        # frames: [B, T, C, H, W]
        batch_size, seq_len, C, H, W = frames.shape

        embeddings = self.patch_embed(frames)  # [B, T, P, E]
        transformed = self.transformer(embeddings)

        if self.pooling == 'attention':
            pairs = torch.cat([transformed[:, :-1], transformed[:, 1:]], dim=-1)  # [B, T-1, P, E*2]
            weights = self.pool_score(pairs).softmax(dim=2)  # [B, T-1, P, 1]
            combined = (weights * pairs).sum(dim=2)  # [B, T-1, E*2]
        else:
            # mean pool over patches (since one action per frame), then concat current and next frame features
            pooled = transformed.mean(dim=2)  # [B, T, E]
            combined = torch.cat([pooled[:, :-1], pooled[:, 1:]], dim=-1)  # [B, T-1, E*2]
        actions = self.action_head(combined)  # [B, T-1, A] (or [B, T-1, 2A])

        return actions

class LatentActionsDecoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, conditioning_dim=3, keep_rate=0.0, residual=False):
        super().__init__()
        # keep_rate: fraction of patches of frames 1..T-2 left visible in training (frame 0 always is).
        #   0.0 = original (decoder sees frame 0 only), 1.0 = Genie (decoder sees every past frame)
        # residual: predict frame t+1 as the most recent visible frame + a correction instead of from scratch,
        #   so copying is free and the action only has to explain what changed
        self.keep_rate = float(keep_rate)
        self.residual = bool(residual)
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True, conditioning_dim=conditioning_dim)

        # embeddings to mixed frame output patches
        self.frame_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3 * patch_size * patch_size),
            nn.Tanh()
        )
        if self.residual:
            # start as an exact copy of the last visible frame (correction 0)
            nn.init.zeros_(self.frame_head[1].weight)
            nn.init.zeros_(self.frame_head[1].bias)

        self.frame_size = frame_size
        self.patch_size = patch_size
        self.num_patches = (frame_size[0] // patch_size) * (frame_size[1] // patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

    def forward(self, frames, actions, training=True):
        # frames: [B, T, C, H, W]
        # actions: [B, T - 1, A]
        B, T, C, H, W = frames.shape
        frames = frames[:, :-1] # [B, T-1, C, H, W]
        video_embeddings = self.patch_embed(frames)  # [B, T-1, P, E]
        _, _, P, E = video_embeddings.shape

        # mask certain tokens from all frames except first frame
        # this strongly forces actions to contain most useful info (I recommend to keep based on experiments)
        keep = torch.ones(B, T-1, P, 1, dtype=torch.bool, device=frames.device)  # [B, T-1, P, 1]
        if training and self.training:
            keep = (torch.rand(B, T-1, P, 1, device=frames.device) < self.keep_rate)
            keep[:, 0] = 1  # never mask first frame tokens (anchor) TODO: try rid of ablation
            video_embeddings = torch.where(
                keep, video_embeddings,
                self.mask_token.to(video_embeddings.dtype).expand_as(video_embeddings)
            )

        transformed = self.transformer(video_embeddings, conditioning=actions)  # [B, T-1, P, E]
        patches = self.frame_head(transformed)  # [B, T-1, P, 3 * S * S]
        patches = rearrange(
            patches, 'b t p (c p1 p2) -> b t c p p1 p2', c=3, p1=self.patch_size, p2=self.patch_size
        ) # [B, T-1, C, P, S, S]
        pred_frames = rearrange(
            patches, 'b t c (h w) p1 p2 -> b t c (h p1) (w p2)', h=H//self.patch_size, w=W//self.patch_size
        ) # [B, T-1, C, H, W]
        if self.residual:
            # per patch, the most recent visible frame (frame 0 always is), plus a correction in [-2, 2]
            keep_px = repeat(
                keep[..., 0], 'b t (h w) -> b t 1 (h p1) (w p2)', h=H//self.patch_size, p1=self.patch_size, p2=self.patch_size
            )  # [B, T-1, 1, H, W]
            base = [frames[:, 0]]  # each [B, C, H, W]
            for t in range(1, T-1):
                base.append(torch.where(keep_px[:, t], frames[:, t], base[-1]))
            pred_frames = torch.stack(base, dim=1) + 2 * pred_frames  # [B, T-1, C, H, W]
        return pred_frames  # [B, T-1, C, H, W]

class LatentActionModel(nn.Module):
    def __init__(self, frame_size=(128, 128), n_actions=8, patch_size=8, embed_dim=128, 
                 num_heads=8, hidden_dim=256, num_blocks=4,
                 decoder_keep_rate=0.0, decoder_residual=False, entropy_loss_weight=0.0, entropy_sample_weight=0.1,
                 encoder_pooling='mean', recon_change_weight=0.0, continuous_actions=False,
                 action_kl_capacity=math.log(8), action_kl_weight=1.0):
        super().__init__()
        assert math.log(n_actions, NUM_LATENT_ACTIONS_BINS).is_integer(), f"n_actions must be a power of {NUM_LATENT_ACTIONS_BINS}"
        self.action_dim=int(math.log(n_actions, NUM_LATENT_ACTIONS_BINS))
        # continuous_actions: no FSQ; the action is a Gaussian sample mean + std * noise in training (its mean otherwise),
        #   and the KL to N(0, I) is held at action_kl_capacity nats (ln 8 = the information of 8 discrete codes)
        #   by action_kl_weight * |KL - capacity|. The quantizer is kept for the code diagnostics (sign pattern of the mean).
        self.continuous_actions = bool(continuous_actions)
        self.action_kl_capacity = float(action_kl_capacity)
        self.action_kl_weight = float(action_kl_weight)
        # recon_change_weight: pixels that change between frame t and t+1 weigh 1 + this in the reconstruction loss
        self.recon_change_weight = float(recon_change_weight)
        self.encoder = LatentActionsEncoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, action_dim=self.action_dim,
                                            pooling=encoder_pooling, out_dim=2 * self.action_dim if self.continuous_actions else None)
        self.quantizer = FiniteScalarQuantizer(latent_dim=self.action_dim, num_bins=NUM_LATENT_ACTIONS_BINS)
        self.decoder = LatentActionsDecoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, conditioning_dim=self.action_dim,
                                            keep_rate=decoder_keep_rate, residual=decoder_residual)
        self.var_target = 0.01
        self.var_lambda = 100.0
        # code-usage loss (0.0 = original: variance penalty only). Replaces the variance penalty when on.
        self.entropy_loss_weight = float(entropy_loss_weight)
        self.entropy_sample_weight = float(entropy_sample_weight)
        # all codes as {-1, 1} bit patterns, for the soft code distribution
        codes = self.quantizer.get_latents_from_indices(torch.arange(self.quantizer.codebook_size))  # [n_actions, A]
        self.register_buffer('code_bits', (codes > 0).float(), persistent=False)

    def code_entropies(self, action_latents):
        # action_latents: [B, T-1, A] pre-tanh -> (mean per-sample entropy, entropy of the batch-mean distribution), nats
        # soft bit prob P(bit = 1) = (tanh z + 1) / 2, so the gradient flows through the same tanh as the quantizer
        p1 = ((torch.tanh(action_latents.float()) + 1) / 2).reshape(-1, 1, self.action_dim)  # [N, 1, A]
        bits = self.code_bits.to(p1.device)  # [n_actions, A]
        probs = (bits * p1 + (1 - bits) * (1 - p1)).prod(-1)  # [N, n_actions] joint code distribution per sample
        h_sample = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()
        mean_probs = probs.mean(0)  # [n_actions]
        h_batch = -(mean_probs * mean_probs.clamp_min(1e-8).log()).sum()
        return h_sample, h_batch

    @staticmethod
    def gaussian_kl(mu, logvar):
        # mu, logvar: [B, T-1, A] -> KL(N(mu, exp(logvar)) || N(0, I)) in nats, mean over transitions
        return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1).mean()

    def action_kl(self, frames):
        # frames: [B, T, C, H, W] -> KL of the continuous action posterior (nats per transition), for logging
        mu, logvar = self.encoder(frames).float().chunk(2, dim=-1)
        return self.gaussian_kl(mu, logvar.clamp(-10, 10))

    def pre_quant(self, frames):
        # frames: [B, T, C, H, W] -> pre-quantization action latents [B, T-1, A] (the mean for continuous actions)
        return self.encoder(frames)[..., :self.action_dim]

    def forward(self, frames):
        # frames: [B, T, C, H, W]

        # get (quantized or sampled) action latents
        action_latents = self.encoder(frames) # [B, T - 1, A] (continuous: [B, T - 1, 2A])
        if self.continuous_actions:
            mu, logvar = action_latents.float().chunk(2, dim=-1)  # [B, T - 1, A] each
            logvar = logvar.clamp(-10, 10)
            action_latents_quantized = mu + torch.randn_like(mu) * (0.5 * logvar).exp()  # [B, T - 1, A]
            kl = self.gaussian_kl(mu, logvar)
        else:
            action_latents_quantized = self.quantizer(action_latents) # [B, T - 1, A]

        # decode to get predicted frames
        pred_frames = self.decoder(frames, action_latents_quantized, training=True)  # [B, T - 1, C, H, W]

        # reconstruction loss
        target_frames = frames[:, 1:]  # All frames except first [B, T - 1, C, H, W]
        if self.recon_change_weight > 0:
            # pixels that changed (any channel by > 0.1 in [-1, 1]) count 1 + w times; normalized so the scale stays a mean
            changed = ((target_frames - frames[:, :-1]).abs().amax(dim=2, keepdim=True) > 0.1).float()  # [B, T - 1, 1, H, W]
            weights = 1 + self.recon_change_weight * changed
            per_px = F.smooth_l1_loss(pred_frames, target_frames, reduction='none')  # [B, T - 1, C, H, W]
            recon_loss = (per_px * weights).sum() / (weights.sum() * per_px.shape[2])
        else:
            recon_loss = F.smooth_l1_loss(pred_frames, target_frames)

        if self.continuous_actions:
            total_loss = recon_loss + self.action_kl_weight * (kl - self.action_kl_capacity).abs()
        elif self.entropy_loss_weight > 0:
            # confident per sample, uniform over the batch (LFQ / MAGVIT-v2 style), on the joint code distribution
            h_sample, h_batch = self.code_entropies(action_latents)
            total_loss = recon_loss + self.entropy_loss_weight * (self.entropy_sample_weight * h_sample - h_batch)
        else:
            # variance loss across batch dim for pre-quant encoder outputs (helps prevent action collapse)
            z_var = action_latents.var(dim=0, unbiased=False).mean()
            var_penalty = F.relu(self.var_target - z_var)
            total_loss = recon_loss + self.var_lambda * var_penalty

        return total_loss, pred_frames

    def encode(self, frames):
        action_latents = self.pre_quant(frames)  # [B, T, A]
        if self.continuous_actions:
            return action_latents  # the mean, no noise
        action_latents_quantized = self.quantizer(action_latents) # [B, T, A]
        return action_latents_quantized
    
    @property
    def model_type(self) -> str:
        return ModelType.LatentActionModel