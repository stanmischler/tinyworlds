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
                 hidden_dim=256, num_blocks=4, action_dim=3, pooling='mean', out_dim=None, input_mode='frames'):
        super().__init__()
        # input_mode: 'frames' = the raw frames (original); 'diff' = only the differences x_{t+1} - x_t, so the action of
        #   transition t is read from diff t alone (pooled over its own patches) and cannot carry frame appearance
        assert input_mode in ('frames', 'diff'), input_mode
        self.input_mode = input_mode
        # pooling: 'mean' = mean over patches, then concat(frame t, frame t+1) (original);
        #   'attention' = per patch concat(frame t, frame t+1), then a learned softmax over patches, so the action can
        #   come from the few patches that changed (a small sprite) instead of being averaged away
        # out_dim: head width (action_dim, or 2 * action_dim for a mean and a log-variance)
        assert pooling in ('mean', 'attention'), pooling
        self.pooling = pooling
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        feat_dim = embed_dim if input_mode == 'diff' else embed_dim * 2
        if pooling == 'attention':
            self.pool_score = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))

        # embeddings to discrete latent bottleneck actions
        out_dim = out_dim or action_dim
        self.action_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 4 * out_dim),
            nn.GELU(),
            nn.Linear(4 * out_dim, out_dim)
        )

    def forward(self, frames):
        # frames: [B, T, C, H, W]
        batch_size, seq_len, C, H, W = frames.shape

        if self.input_mode == 'diff':
            diffs = frames[:, 1:] - frames[:, :-1]  # [B, T-1, C, H, W] in [-2, 2]
            transformed = self.transformer(self.patch_embed(diffs))  # [B, T-1, P, E]
            if self.pooling == 'attention':
                weights = self.pool_score(transformed).softmax(dim=2)  # [B, T-1, P, 1]
                combined = (weights * transformed).sum(dim=2)  # [B, T-1, E]
            else:
                combined = transformed.mean(dim=2)  # [B, T-1, E]
            return self.action_head(combined)  # [B, T-1, A] (or [B, T-1, 2A])

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
                 hidden_dim=256, num_blocks=4, conditioning_dim=3, keep_rate=0.0, residual=False, hint='none',
                 warp='none', warp_radius=12):
        super().__init__()
        # warp: an action-conditioned global warp of frame t is the base of the prediction (CDNA, Finn et al. 2016:
        #   the action predicts transformation kernels applied to the previous frame). The action alone (not the hint,
        #   not the history) gives a separable kernel = softmax over 2 * warp_radius + 1 vertical and horizontal taps,
        #   so a full-screen scroll (Zelda walking = ~9 px per transition at 128px) is a few-parameter function of the code.
        #   'none' = original; 'global' = base is the warped (true, unmasked) frame t, plus the residual correction of the
        #   transformer; 'global_only' = the warp alone (no transformer: the code can only explain a global shift)
        assert warp in ('none', 'global', 'global_only'), warp
        self.warp = warp
        self.warp_radius = int(warp_radius)
        if self.warp != 'none':
            K = 2 * self.warp_radius + 1
            self.warp_head = nn.Sequential(nn.Linear(conditioning_dim, 64), nn.GELU(), nn.Linear(64, 2 * K))
            nn.init.zeros_(self.warp_head[2].weight)
            with torch.no_grad():  # start near identity: the centre tap holds ~45% of each 1D kernel
                self.warp_head[2].bias.zero_()
                self.warp_head[2].bias[self.warp_radius] = 3.0
                self.warp_head[2].bias[K + self.warp_radius] = 3.0
        # hint: also tell the decoder where the player is in each transition t -> t+1 (one patch per transition):
        #   its (y, x) centre in [-1, 1] and a has-change flag are appended to the action for FiLM, and a learned marker is
        #   added to that patch's token, so the action only has to say how the player moves, not where it is.
        #   'none' = original; 'max_diff' = the patch with the largest mean |x_{t+1} - x_t| (hits Link 0/35 on hand-labelled
        #   Zelda frames: scrolling edges, water and text win); 'player' = see player_location (Link 25/35, 4/5 off-centre).
        #   Both read frame t+1, so the hint is extra (not bottlenecked) information about the target.
        assert hint in ('none', 'max_diff', 'player'), hint
        self.hint = hint
        if self.hint != 'none':
            conditioning_dim = conditioning_dim + 3
            self.hint_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
            nn.init.normal_(self.hint_token, std=0.02)
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
        if self.residual or self.warp != 'none':
            # start as an exact copy of the last visible frame (correction 0)
            nn.init.zeros_(self.frame_head[1].weight)
            nn.init.zeros_(self.frame_head[1].bias)

        self.frame_size = frame_size
        self.patch_size = patch_size
        self.num_patches = (frame_size[0] // patch_size) * (frame_size[1] // patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

    @torch.compiler.disable
    @torch.no_grad()
    def player_location(self, a, b, blur=9, sigma=0.125, ratio=2.0):
        # a, b: [N, C, H, W] frames t and t+1 -> change map [N, H, W] whose peak patch is the player, and whether to trust it [N].
        # Tuned on 35 hand-labelled held-out Zelda transitions (the camera follows Link, so he sits near (64, 62)):
        #   1. global camera shift by phase correlation; while scrolling (|shift| > 1 px), the player is what changes only once
        #      the shift is removed (fixed on screen while the world moves): relu(|b - shift(a)| - |b - a|), border band zeroed;
        #      when the camera is still, the plain |b - a|
        #   2. 9x9 box blur (sprite scale), times a Gaussian prior at the screen centre (sigma = 1/8 of the frame)
        #   3. the peak counts only if it beats the value at the centre by 2x, else the centre is used (no hint if nothing changed)
        a, b = a.float(), b.float()
        N, C, H, W = a.shape
        R = torch.fft.fft2(b.mean(1)) * torch.fft.fft2(a.mean(1)).conj()  # [N, H, W]
        r = torch.fft.ifft2(R / R.abs().clamp_min(1e-8)).real.flatten(1).argmax(1)  # [N]
        dy, dx = r // W, r % W
        dy = torch.where(dy > H // 2, dy - H, dy)
        dx = torch.where(dx > W // 2, dx - W, dx)
        ys = torch.arange(H, device=a.device)[None, :, None]  # [1, H, 1]
        xs = torch.arange(W, device=a.device)[None, None, :]  # [1, 1, W]
        src_y = (ys - dy[:, None, None]) % H  # [N, H, 1]
        src_x = (xs - dx[:, None, None]) % W  # [N, 1, W]
        shifted = a[torch.arange(N, device=a.device)[:, None, None], :, src_y, src_x].permute(0, 3, 1, 2)  # [N, C, H, W]
        raw = (b - a).abs().mean(1)  # [N, H, W]
        stab = (b - shifted).abs().mean(1)  # [N, H, W]
        border = ((ys < dy.clamp(min=0)[:, None, None]) | (ys >= H + dy.clamp(max=0)[:, None, None]) |
                  (xs < dx.clamp(min=0)[:, None, None]) | (xs >= W + dx.clamp(max=0)[:, None, None]))  # [N, H, W]
        stab = stab.masked_fill(border, 0)
        scroll = ((dy.abs() + dx.abs()) > 1)[:, None, None]  # [N, 1, 1]
        m = torch.where(scroll, (stab - raw).clamp(min=0), raw)  # [N, H, W]
        m = F.avg_pool2d(m[:, None], blur, 1, blur // 2, count_include_pad=False)[:, 0]
        yy = torch.arange(H, device=a.device).float()[:, None]
        xx = torch.arange(W, device=a.device).float()[None, :]
        cy, cx = H * 62 / 128, W / 2
        m = m * torch.exp(-((xx - cx) / (sigma * W)) ** 2 / 2 - ((yy - cy) / (sigma * H)) ** 2 / 2)  # [N, H, W]
        return m, cy, cx

    @torch.no_grad()
    def change_location(self, frames, thr=0.02):
        # frames: [B, T, C, H, W] -> hint patch index per transition [B, T-1],
        #   FiLM hint [B, T-1, 3] = (y, x) of that patch centre in [-1, 1] and 1 if anything changed (else all 0)
        S = self.patch_size
        B, T, C, H, W = frames.shape
        Hp, Wp = H // S, W // S
        change = (frames[:, 1:] - frames[:, :-1]).abs().float().mean(dim=2)  # [B, T-1, H, W]
        changed = reduce(change, 'b t (h p1) (w p2) -> b t (h w)', 'mean', p1=S, p2=S).amax(-1) > thr  # [B, T-1]
        if self.hint == 'max_diff':
            score = change
        else:
            m, cy, cx = self.player_location(rearrange(frames[:, :-1], 'b t c h w -> (b t) c h w'),
                                             rearrange(frames[:, 1:], 'b t c h w -> (b t) c h w'))
            score = rearrange(m, '(b t) h w -> b t h w', b=B)  # [B, T-1, H, W]
        per_patch = reduce(score, 'b t (h p1) (w p2) -> b t (h w)', 'mean', p1=S, p2=S)  # [B, T-1, P]
        peak, idx = per_patch.max(dim=-1)  # [B, T-1] each
        if self.hint == 'player':
            centre = int(cy) // S * Wp + int(cx) // S
            idx = torch.where(peak > 2.0 * per_patch[..., centre].clamp_min(1e-6), idx, torch.full_like(idx, centre))
        y = ((idx // Wp).float() + 0.5) / Hp * 2 - 1  # [B, T-1]
        x = ((idx % Wp).float() + 0.5) / Wp * 2 - 1  # [B, T-1]
        valid = changed.float()  # [B, T-1]
        hint = torch.stack([y * valid, x * valid, valid], dim=-1)  # [B, T-1, 3]
        return idx, hint

    def warp_frames(self, frames, actions):
        # frames: [B, T-1, C, H, W] (frames t), actions: [B, T-1, A] -> frames t warped by the action's kernel [B, T-1, C, H, W]
        B, T1, C, H, W = frames.shape
        r, K = self.warp_radius, 2 * self.warp_radius + 1
        with torch.autocast(device_type=frames.device.type, enabled=False):  # fp32: the taps must sum to 1 exactly
            logits = self.warp_head(actions.float())  # [B, T-1, 2K]
            ky, kx = logits[..., :K].softmax(-1), logits[..., K:].softmax(-1)  # [B, T-1, K] each
            x = rearrange(frames.float(), 'b t c h w -> 1 (b t c) h w')  # [1, B*(T-1)*C, H, W]
            wy = repeat(ky, 'b t k -> (b t c) 1 k 1', c=C)  # [B*(T-1)*C, 1, K, 1]
            wx = repeat(kx, 'b t k -> (b t c) 1 1 k', c=C)  # [B*(T-1)*C, 1, 1, K]
            x = F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), wy, groups=x.shape[1])
            x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), wx, groups=x.shape[1])
        return rearrange(x, '1 (b t c) h w -> b t c h w', b=B, t=T1)  # [B, T-1, C, H, W]

    def forward(self, frames, actions, training=True):
        # frames: [B, T, C, H, W]
        # actions: [B, T - 1, A]
        B, T, C, H, W = frames.shape
        frames_full = frames  # [B, T, C, H, W]
        frames = frames[:, :-1] # [B, T-1, C, H, W]
        if self.warp != 'none':
            warped = self.warp_frames(frames, actions)  # [B, T-1, C, H, W]
            if self.warp == 'global_only':
                return warped.to(frames.dtype)
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

        if self.hint != 'none':
            idx, hint = self.change_location(frames_full)  # [B, T-1], [B, T-1, 3]
            marker = F.one_hot(idx, P).to(video_embeddings.dtype)[..., None] * hint[..., 2:, None].to(video_embeddings.dtype)  # [B, T-1, P, 1]
            video_embeddings = video_embeddings + marker * self.hint_token.to(video_embeddings.dtype)
            actions = torch.cat([actions, hint.to(actions.dtype)], dim=-1)  # [B, T-1, A+3]

        transformed = self.transformer(video_embeddings, conditioning=actions)  # [B, T-1, P, E]
        patches = self.frame_head(transformed)  # [B, T-1, P, 3 * S * S]
        patches = rearrange(
            patches, 'b t p (c p1 p2) -> b t c p p1 p2', c=3, p1=self.patch_size, p2=self.patch_size
        ) # [B, T-1, C, P, S, S]
        pred_frames = rearrange(
            patches, 'b t c (h w) p1 p2 -> b t c (h p1) (w p2)', h=H//self.patch_size, w=W//self.patch_size
        ) # [B, T-1, C, H, W]
        if self.warp != 'none':
            pred_frames = warped.to(pred_frames.dtype) + 2 * pred_frames  # [B, T-1, C, H, W]
        elif self.residual:
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
                 action_kl_capacity=math.log(8), action_kl_weight=1.0, action_fixed_noise=False,
                 encoder_input='frames', decoder_hint='none', decoder_warp='none', decoder_warp_radius=12):
        super().__init__()
        assert math.log(n_actions, NUM_LATENT_ACTIONS_BINS).is_integer(), f"n_actions must be a power of {NUM_LATENT_ACTIONS_BINS}"
        self.action_dim=int(math.log(n_actions, NUM_LATENT_ACTIONS_BINS))
        # continuous_actions: no FSQ; the action is a Gaussian sample mean + std * noise in training (its mean otherwise),
        #   and the KL to N(0, I) is held at action_kl_capacity nats (ln 8 = the information of 8 discrete codes)
        #   by action_kl_weight * |KL - capacity|. The quantizer is kept for the code diagnostics (sign pattern of the mean).
        # action_fixed_noise: unit noise and KL = 0.5 E||mean - batch mean||^2, which bounds the information the action
        #   carries; the learned-variance KL to N(0, I) can instead be met by a constant offset carrying none
        self.action_fixed_noise = bool(action_fixed_noise)
        self.continuous_actions = bool(continuous_actions)
        self.action_kl_capacity = float(action_kl_capacity)
        self.action_kl_weight = float(action_kl_weight)
        # recon_change_weight: pixels that change between frame t and t+1 weigh 1 + this in the reconstruction loss
        self.recon_change_weight = float(recon_change_weight)
        self.encoder = LatentActionsEncoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, action_dim=self.action_dim,
                                            pooling=encoder_pooling, input_mode=encoder_input,
                                            out_dim=2 * self.action_dim if self.continuous_actions and not self.action_fixed_noise else None)
        self.quantizer = FiniteScalarQuantizer(latent_dim=self.action_dim, num_bins=NUM_LATENT_ACTIONS_BINS)
        self.decoder = LatentActionsDecoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, conditioning_dim=self.action_dim,
                                            keep_rate=decoder_keep_rate, residual=decoder_residual, hint=decoder_hint,
                                            warp=decoder_warp, warp_radius=decoder_warp_radius)
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

    @staticmethod
    def spread_kl(mu):
        # mu: [B, T-1, A] -> 0.5 E||mu - batch mean||^2, nats per transition (unit-noise information bound)
        return 0.5 * (mu - mu.mean(dim=(0, 1), keepdim=True)).pow(2).sum(-1).mean()

    def action_kl(self, frames):
        # frames: [B, T, C, H, W] -> KL of the continuous action posterior (nats per transition), for logging
        if self.action_fixed_noise:
            return self.spread_kl(self.encoder(frames).float())
        mu, logvar = self.encoder(frames).float().chunk(2, dim=-1)
        return self.gaussian_kl(mu, logvar.clamp(-10, 10))

    def pre_quant(self, frames):
        # frames: [B, T, C, H, W] -> pre-quantization action latents [B, T-1, A] (the mean for continuous actions)
        return self.encoder(frames)[..., :self.action_dim]

    def forward(self, frames):
        # frames: [B, T, C, H, W]

        # get (quantized or sampled) action latents
        action_latents = self.encoder(frames) # [B, T - 1, A] (continuous: [B, T - 1, 2A])
        if self.continuous_actions and self.action_fixed_noise:
            mu = action_latents.float()  # [B, T - 1, A]
            action_latents_quantized = mu + torch.randn_like(mu)  # [B, T - 1, A]
            kl = self.spread_kl(mu)
        elif self.continuous_actions:
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