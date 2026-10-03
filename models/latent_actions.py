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

def ot_features(ot, Wp):
    # ot: [B, T-1, 2, P] long, per transition t -> t+1 the unbalanced OT plan between the tokenizer tokens of the two frames
    #   (scripts/eval/patch_similarity.py ot-plans): [:, :, 0] = destination of token i of frame t (-1 = destroyed),
    #   [:, :, 1] = 1 if token j of frame t+1 is created (fed by nothing)
    # -> src [B, T-1, P, 4] at frame t positions: (moved, destroyed, drow / 8, dcol / 8) (rows / cols in token cells),
    #    created [B, T-1, P, 1] at frame t+1 positions, changed [B, T-1, P, 1]: any token leaving, arriving, destroyed or
    #    created at that position (where the plan says the frame changes, not where things go)
    sigma, created = ot[:, :, 0], ot[:, :, 1].float()  # [B, T-1, P] each
    P = sigma.shape[-1]
    i = torch.arange(P, device=sigma.device)  # [P]
    destroyed = sigma < 0  # [B, T-1, P]
    dst = sigma.clamp_min(0)
    moved = ~destroyed & (dst != i)
    dr = torch.where(moved, dst // Wp - i // Wp, 0).float() / 8
    dc = torch.where(moved, dst % Wp - i % Wp, 0).float() / 8
    src = torch.stack([moved.float(), destroyed.float(), dr, dc], dim=-1)  # [B, T-1, P, 4]
    arrived = torch.zeros_like(created).scatter_add_(-1, dst, moved.float()) > 0  # [B, T-1, P] destinations of moves
    changed = (moved | destroyed | arrived | (created > 0)).float()[..., None]  # [B, T-1, P, 1]
    return src, created[..., None], changed


class LatentActionsEncoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8, 
                 hidden_dim=256, num_blocks=4, action_dim=3, pooling='mean', out_dim=None, input_mode='frames', use_ot=False):
        super().__init__()
        # use_ot: add the OT plan to each frame's patch embeddings (ot_features): frame k gets the plan of transition k
        #   (where its tokens go) and the created tokens of transition k-1 (what is new in it)
        self.use_ot = bool(use_ot)
        if self.use_ot:
            assert input_mode == 'frames', 'the OT plan is given on the frames, not the diffs'
            self.ot_embed = nn.Linear(5, embed_dim)
        self.Wp = frame_size[1] // patch_size
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

    def forward(self, frames, ot=None):
        # frames: [B, T, C, H, W], ot: [B, T-1, 2, P] (see ot_features; required iff use_ot)
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
        if self.use_ot:
            src, created, _ = ot_features(ot, self.Wp)  # [B, T-1, P, 4], [B, T-1, P, 1]
            pad = torch.zeros_like(src[:, :1])  # [B, 1, P, 4]
            feats = torch.cat([torch.cat([src, pad], 1), torch.cat([pad[..., :1], created], 1)], -1)  # [B, T, P, 5]
            embeddings = embeddings + self.ot_embed(feats.to(embeddings.dtype))
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
                 warp='none', warp_radius=12, warp_init_std=0.0, warp_local=0, warp_local_mask=False,
                 warp_local_gate=0.0, warp_local_blur=1, ot='none'):
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
            # warp_init_std > 0: random last layer so each code starts with a slightly different kernel (winner-takes-all
            #   training needs the symmetry broken; with zeros every code is the same identity kernel)
            nn.init.normal_(self.warp_head[2].weight, std=warp_init_std) if warp_init_std > 0 else nn.init.zeros_(self.warp_head[2].weight)
            with torch.no_grad():  # start near identity: the centre tap holds ~45% of each 1D kernel
                self.warp_head[2].bias.zero_()
                self.warp_head[2].bias[self.warp_radius] = 3.0
                self.warp_head[2].bias[K + self.warp_radius] = 3.0
        # warp_local: window radius in px (0 = off, original). When the camera does not move between t and t+1 (phase
        #   correlation |shift| <= 1 px: Link walks on a still screen, stands, or a fade/text plays) the base is frame t with
        #   only a window around the largest change moved, by the code's kernel FLIPPED: the content scrolls by -v when the
        #   camera follows a walk by v, the player sprite moves by +v on a still screen, so one code = one walking direction
        #   in both cases. Needs warp != 'none'. Like hint, it reads frame t+1 (camera moved or not + where the change is),
        #   never the direction, which only the code gives.
        self.warp_local = int(warp_local)
        self.warp_local_mask = bool(warp_local_mask)  # see local_window
        # warp_local_gate: > 0 = the local mode fires only where a sprite-sized blob translates: on warp_local_blur-blurred
        #   grayscale frames, the best of a fixed set of 3-10 px shifts of the window must explain the window better than
        #   gate x the copy error (judge set: fires on 16/200, 2/40 STILL; the i3 rule fired on 88% of STILL).
        #   0 = the i3 rule (any >= 12 changed px on a still screen). warp_local_blur > 1 also scores the local WTA cost on
        #   blurred frames (Link's walk animation changes the sprite, a rigid shift only fits it at sprite scale)
        self.warp_local_gate = float(warp_local_gate)
        self.warp_local_blur = int(warp_local_blur)
        assert not self.warp_local or self.warp != 'none', 'warp_local needs a warp decoder'
        # ot: give the decoder the OT plan of transition t -> t+1 on the tokens of frame t (never masked).
        #   'none' = original; 'plan' = the whole plan (ot_features src + created: where each token goes, what is new),
        #   which with frame t nearly determines frame t+1, so the action may carry nothing;
        #   'where' = only the changed-token mask (which positions change, not how), so the action has to say how
        assert ot in ('none', 'where', 'plan'), ot
        self.ot = ot
        if self.ot != 'none':
            self.ot_embed = nn.Linear(5 if ot == 'plan' else 1, embed_dim)
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

    @torch.compiler.disable
    @torch.no_grad()
    def local_window(self, frames, thr=0.1, min_px=12):
        # frames: [B, T, C, H, W] -> (local [B, T-1] bool, move weight [B, T-1, 1, H, W], window [B, T-1, 1, H, W]) for warp_local
        #   local = camera still (phase correlation |shift| <= 1 px, or the shift does not explain frame t+1) AND >= min_px pixels inside the window changed by > thr;
        #   elsewhere the global warp is used (a scroll, or nothing changed: then the identity code wins clearly, as before)
        #   window = flat-topped disc of radius warp_local px at the peak of the blurred change (mild centre prior);
        #   move weight = window (warp_local_mask False) or window x dilated change mask (True: unchanged pixels stay frame t,
        #   so a textured background inside the window is not dragged along with the sprite)
        B, T, C, H, W = frames.shape
        a = rearrange(frames[:, :-1], 'b t c h w -> (b t) c h w').float().mean(1)  # [N, H, W]
        b = rearrange(frames[:, 1:], 'b t c h w -> (b t) c h w').float().mean(1)  # [N, H, W]
        R = torch.fft.fft2(b) * torch.fft.fft2(a).conj()  # [N, H, W]
        r = torch.fft.ifft2(R / R.abs().clamp_min(1e-8)).real.flatten(1).argmax(1)  # [N]
        dy, dx = r // W, r % W
        dy = torch.where(dy > H // 2, dy - H, dy)
        dx = torch.where(dx > W // 2, dx - W, dx)
        diff = (b - a).abs()  # [N, H, W]
        # a scroll only if the global shift explains frame t+1 much better than a copy does (inside a 12 px margin): a sprite
        #   walking on a smooth, still screen can win the phase correlation, but shifting the whole frame then does not help
        n = torch.arange(a.shape[0], device=a.device)[:, None, None]
        src_y = (torch.arange(H, device=a.device)[None, :, None] - dy[:, None, None]) % H  # [N, H, 1]
        src_x = (torch.arange(W, device=a.device)[None, None, :] - dx[:, None, None]) % W  # [N, 1, W]
        e_shift = (b - a[n, src_y, src_x]).abs()[:, 12:-12, 12:-12].mean(dim=(1, 2))  # [N]
        e_copy = diff[:, 12:-12, 12:-12].mean(dim=(1, 2))  # [N]
        still = ((dy.abs() + dx.abs()) <= 1) | (e_shift > 0.5 * e_copy)  # [N]
        m = F.avg_pool2d(diff[:, None], 9, 1, 4, count_include_pad=False)[:, 0]  # [N, H, W]
        yy = torch.arange(H, device=a.device).float()[:, None]  # [H, 1]
        xx = torch.arange(W, device=a.device).float()[None, :]  # [1, W]
        m = m * torch.exp(-(((yy - H / 2) / (0.35 * H)) ** 2 + ((xx - W / 2) / (0.35 * W)) ** 2) / 2) + 1e-6
        idx = m.flatten(1).argmax(1)  # [N]
        cy, cx = (idx // W).float()[:, None, None], (idx % W).float()[:, None, None]  # [N, 1, 1]
        d2 = ((yy[None] - cy) ** 2 + (xx[None] - cx) ** 2) / self.warp_local ** 2  # [N, H, W]
        g = torch.exp(-0.5 * d2 ** 2)  # flat-topped window: ~1 within 0.7 radius, 0.6 at the radius
        changed = (diff > thr).float()  # [N, H, W]
        local = still & ((changed * (g > 0.5)).sum(dim=(1, 2)) >= min_px)  # [N]
        if self.warp_local_gate > 0:
            k = self.warp_local_blur
            blur = lambda t: F.avg_pool2d(t[:, None], k, 1, k // 2, count_include_pad=False)[:, 0] if k > 1 else t
            ab, bb = blur(a), blur(b)  # [N, H, W]
            g_sum = g.sum(dim=(1, 2)).clamp_min(1e-6)  # [N]
            e_copy_w = ((bb - ab).abs() * g).sum(dim=(1, 2)) / g_sum  # [N]
            best = torch.full_like(e_copy_w, float('inf'))  # [N]
            for s in range(3, 11):
                for u, v in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    moved = ab + g * (torch.roll(ab, (s * u, s * v), (-2, -1)) - ab)  # [N, H, W]
                    best = torch.minimum(best, ((moved - bb).abs() * g).sum(dim=(1, 2)) / g_sum)
            local = local & (best < self.warp_local_gate * e_copy_w.clamp_min(1e-6))  # [N]
        w = g * F.max_pool2d(changed[:, None], 7, 1, 3)[:, 0] if self.warp_local_mask else g  # [N, H, W]
        to5 = lambda t: rearrange(t, '(b t) h w -> b t 1 h w', b=B)
        return local.reshape(B, T - 1), to5(w), to5(g)

    def local_base(self, frames_t, warped, flipped, local):
        # frames_t, warped (global warp), flipped (flipped-kernel warp): [B, T-1, C, H, W]; local = local_window output
        #   -> base [B, T-1, C, H, W]: the global warp, except frame t with the window moved where local
        is_local, w, _ = local
        moved = frames_t.float() + w * (flipped - frames_t.float())  # [B, T-1, C, H, W]
        return torch.where(is_local[:, :, None, None, None], moved, warped)

    @torch.compiler.disable  # inductor stalled >8 min at step 0 on the grouped conv with dynamic shapes (H100, 2026-10-02)
    def warp_frames(self, frames, actions, flip=False):
        # frames: [B, T-1, C, H, W] (frames t), actions: [B, T-1, A] -> frames t warped by the action's kernel [B, T-1, C, H, W]
        B, T1, C, H, W = frames.shape
        r, K = self.warp_radius, 2 * self.warp_radius + 1
        with torch.autocast(device_type=frames.device.type, enabled=False):  # fp32: the taps must sum to 1 exactly
            logits = self.warp_head(actions.float())  # [B, T-1, 2K]
            ky, kx = logits[..., :K].softmax(-1), logits[..., K:].softmax(-1)  # [B, T-1, K] each
            if flip:  # the opposite shift (warp_local: the sprite moves by -(content scroll))
                ky, kx = ky.flip(-1), kx.flip(-1)
            # mean entropy of the 1D kernels (nats), for the optional sharpness penalty (decoder_warp_entropy_weight)
            self.last_warp_entropy = -(ky * ky.clamp_min(1e-8).log()).sum(-1).mean() - (kx * kx.clamp_min(1e-8).log()).sum(-1).mean()
            x = rearrange(frames.float(), 'b t c h w -> 1 (b t c) h w')  # [1, B*(T-1)*C, H, W]
            wy = repeat(ky, 'b t k -> (b t c) 1 k 1', c=C)  # [B*(T-1)*C, 1, K, 1]
            wx = repeat(kx, 'b t k -> (b t c) 1 1 k', c=C)  # [B*(T-1)*C, 1, 1, K]
            x = F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), wy, groups=x.shape[1])
            x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), wx, groups=x.shape[1])
        return rearrange(x, '1 (b t c) h w -> b t c h w', b=B, t=T1)  # [B, T-1, C, H, W]

    def forward(self, frames, actions, training=True, local=None, ot=None):
        # frames: [B, T, C, H, W]
        # actions: [B, T - 1, A]
        # local: precomputed local_window(frames) (warp_local only; None = compute it)
        # ot: [B, T - 1, 2, P] (see ot_features; required iff self.ot != 'none')
        B, T, C, H, W = frames.shape
        frames_full = frames  # [B, T, C, H, W]
        frames = frames[:, :-1] # [B, T-1, C, H, W]
        if self.warp != 'none':
            warped = self.warp_frames(frames, actions)  # [B, T-1, C, H, W]
            if self.warp_local:
                local = local if local is not None else self.local_window(frames_full)
                warped = self.local_base(frames, warped, self.warp_frames(frames, actions, flip=True), local)
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

        if self.ot != 'none':
            src, created, changed = ot_features(ot, W // self.patch_size)
            feats = torch.cat([src, created], -1) if self.ot == 'plan' else changed  # [B, T-1, P, 5 or 1]
            video_embeddings = video_embeddings + self.ot_embed(feats.to(video_embeddings.dtype))

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
                 encoder_input='frames', decoder_hint='none', decoder_warp='none', decoder_warp_radius=12,
                 decoder_warp_entropy_weight=0.0, decoder_warp_entropy_ramp=3000,
                 decoder_warp_wta=False, wta_sinkhorn_eps=0.05, wta_encoder_weight=1.0, decoder_warp_local=0, decoder_warp_local_mask=False, wta_balance=1.0,
                 decoder_warp_local_gate=0.0, decoder_warp_local_blur=1, wta_kernel_repulsion=0.0,
                 ot_encoder=False, ot_decoder='none', aux_label_weight=0.0, aux_label_classes=9, aux_label_target='latent'):
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
                                            pooling=encoder_pooling, input_mode=encoder_input, use_ot=ot_encoder,
                                            out_dim=2 * self.action_dim if self.continuous_actions and not self.action_fixed_noise else None)
        self.quantizer = FiniteScalarQuantizer(latent_dim=self.action_dim, num_bins=NUM_LATENT_ACTIONS_BINS)
        self.decoder = LatentActionsDecoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, conditioning_dim=self.action_dim,
                                            keep_rate=decoder_keep_rate, residual=decoder_residual, hint=decoder_hint,
                                            warp=decoder_warp, warp_radius=decoder_warp_radius,
                                            warp_init_std=0.3 if decoder_warp_wta else 0.0, warp_local=decoder_warp_local, warp_local_mask=decoder_warp_local_mask,
                                            warp_local_gate=decoder_warp_local_gate, warp_local_blur=decoder_warp_local_blur, ot=ot_decoder)
        # decoder_warp_wta: winner-takes-all (multiple-choice / best-of-K) training of the warp: frame t is warped under EVERY
        #   code, the per-transition losses are turned into a balanced soft assignment (Sinkhorn-Knopp, SwAV-style equal
        #   partition, so no code dies and none takes everything), each code's kernel is trained only on its assigned
        #   transitions (a code cannot half-fit two opposite shifts: the other shift has its own code), the transformer
        #   correction is conditioned on the winner code (teacher forcing), and the encoder learns to predict the assignment
        #   (cross-entropy on its soft code distribution, weight wta_encoder_weight). At inference the encoder's code is used.
        #   Needs decoder_warp != 'none'. False = original (the encoder's straight-through code drives the decoder)
        self.warp_wta = bool(decoder_warp_wta)
        assert not self.warp_wta or (decoder_warp != 'none' and not self.continuous_actions), 'wta needs a warp decoder and FSQ codes'
        self.wta_sinkhorn_eps = float(wta_sinkhorn_eps)
        self.wta_encoder_weight = float(wta_encoder_weight)
        self.wta_balance = float(wta_balance)  # exponent of the Sinkhorn column normalisation (1 = equal partition, original)
        # wta_kernel_repulsion: weight of sum over code pairs of the overlap of their 2D warp kernels (ky_i . ky_j)(kx_i . kx_j)
        #   (1 = same sharp shift, 0 = disjoint). Two codes with the same shift tie in the WTA cost, so the split between them
        #   is arbitrary and the encoder learns it from appearance (duplicate U / L / R codes); repelled, the spare kernel
        #   moves to another shift (a diagonal, another speed) or dies. 0 = off (original)
        self.wta_kernel_repulsion = float(wta_kernel_repulsion)
        self.last_kernel_overlap = torch.zeros(())
        # decoder_warp_entropy_weight: penalty on the entropy of the warp kernels so each code commits to ONE shift
        #   (a bimodal kernel = two shifted copies serves up- and down-scrolls with one code); ramped in linearly over
        #   decoder_warp_entropy_ramp training steps so the kernels first move away from the identity. 0 = off
        self.warp_entropy_weight = float(decoder_warp_entropy_weight)
        self.warp_entropy_ramp = max(int(decoder_warp_entropy_ramp), 1)
        self.register_buffer('train_steps', torch.zeros((), dtype=torch.long), persistent=False)
        # OT-conditioned LAM (STA-35): the calibrated token transport plan of each transition as an input
        self.uses_ot = bool(ot_encoder) or ot_decoder != 'none'
        # aux_label_weight (STA-35 itc_loop i5): linear head on the squashed action latent tanh(z) -> aux_label_classes
        #   pseudo-labels (e.g. the ITC teacher's 8 directions + STILL), CE weight aux_label_weight, so the codes align with
        #   the teacher's classes while the decoder objective is kept. 0 = off (original, no extra parameters)
        self.aux_label_weight = float(aux_label_weight)
        # aux_label_target (i6 B2): 'latent' = the head above (i5); 'codes' = class probs code_probs @ softmax(aux_map), a learned
        #   [n_actions, classes] code -> class map (zero init = uniform), so the CE moves the quantised sign pattern itself
        assert aux_label_target in ('latent', 'codes'), aux_label_target
        self.aux_label_target = aux_label_target
        if self.aux_label_weight > 0 and aux_label_target == 'latent':
            self.aux_head = nn.Linear(self.action_dim, int(aux_label_classes))
        if self.aux_label_weight > 0 and aux_label_target == 'codes':
            self.aux_map = nn.Parameter(torch.zeros(n_actions, int(aux_label_classes)))  # [n_actions, classes] logits
        self.last_aux_acc = torch.zeros(())
        self.var_target = 0.01
        self.var_lambda = 100.0
        # code-usage loss (0.0 = original: variance penalty only). Replaces the variance penalty when on.
        self.entropy_loss_weight = float(entropy_loss_weight)
        self.entropy_sample_weight = float(entropy_sample_weight)
        # all codes as {-1, 1} bit patterns, for the soft code distribution
        codes = self.quantizer.get_latents_from_indices(torch.arange(self.quantizer.codebook_size))  # [n_actions, A]
        self.register_buffer('code_bits', (codes > 0).float(), persistent=False)
        self.register_buffer('code_latents', codes.float(), persistent=False)  # [n_actions, A] in {-1, 1}

    def code_probs(self, action_latents):
        # action_latents: [B, T-1, A] pre-tanh -> soft joint code distribution [N, n_actions], N = B * (T-1)
        # soft bit prob P(bit = 1) = (tanh z + 1) / 2, so the gradient flows through the same tanh as the quantizer
        p1 = ((torch.tanh(action_latents.float()) + 1) / 2).reshape(-1, 1, self.action_dim)  # [N, 1, A]
        bits = self.code_bits.to(p1.device)  # [n_actions, A]
        return (bits * p1 + (1 - bits) * (1 - p1)).prod(-1)  # [N, n_actions]

    def code_entropies(self, action_latents):
        # action_latents: [B, T-1, A] pre-tanh -> (mean per-sample entropy, entropy of the batch-mean distribution), nats
        probs = self.code_probs(action_latents)  # [N, n_actions] joint code distribution per sample
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

    def action_kl(self, frames, ot=None):
        # frames: [B, T, C, H, W] -> KL of the continuous action posterior (nats per transition), for logging
        if self.action_fixed_noise:
            return self.spread_kl(self.encoder(frames, ot).float())
        mu, logvar = self.encoder(frames, ot).float().chunk(2, dim=-1)
        return self.gaussian_kl(mu, logvar.clamp(-10, 10))

    def pre_quant(self, frames, ot=None):
        # frames: [B, T, C, H, W] -> pre-quantization action latents [B, T-1, A] (the mean for continuous actions)
        return self.encoder(frames, ot)[..., :self.action_dim]

    @torch.compiler.disable
    @torch.no_grad()
    def sinkhorn(self, costs, iters=3):
        # costs: [N, K] per-transition loss under each code -> balanced soft assignment [N, K] (rows sum to 1, columns ~N/K)
        c = costs.float() / costs.float().mean().clamp_min(1e-8)  # dimensionless: a wrong 9 px shift costs O(1)
        q = torch.exp(-(c - c.min(dim=1, keepdim=True).values) / self.wta_sinkhorn_eps)  # [N, K], row-shifted for range
        N, K = q.shape
        q = q / q.sum()
        for _ in range(iters):
            # each code gets 1/K of the mass (wta_balance 1); < 1 only partly evens out the columns, so a frequent class
            #   (STILL, walking R) need not be split over several codes
            q = q / (q.sum(dim=0, keepdim=True).clamp_min(1e-30) * K) ** self.wta_balance
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-30) / N  # each transition is assigned once
        return q * N

    def kernel_overlap(self, codes):
        # codes: [K, A] -> sum over code pairs i < j of the overlap of their separable warp kernels (scalar)
        Kt = 2 * self.decoder.warp_radius + 1
        with torch.autocast(device_type=codes.device.type, enabled=False):
            logits = self.decoder.warp_head(codes.float())  # [K, 2 Kt]
            ky, kx = logits[:, :Kt].softmax(-1), logits[:, Kt:].softmax(-1)  # [K, Kt] each
            o = (ky @ ky.T) * (kx @ kx.T)  # [K, K]
            pair = o.triu(diagonal=1).sum()
        self.last_kernel_overlap = pair.detach()
        return pair

    def forward_wta(self, frames, action_latents):
        # frames: [B, T, C, H, W], action_latents: [B, T-1, A] pre-quant encoder outputs -> (total loss, pred frames [B, T-1, C, H, W])
        B, T, C, H, W = frames.shape
        K = self.code_latents.shape[0]
        frames_t, target = frames[:, :-1], frames[:, 1:]  # [B, T-1, C, H, W] each
        codes = self.code_latents.to(frames.device)  # [K, A]
        rep_t = repeat(frames_t, 'b t c h w -> (k b) t c h w', k=K)  # [K*B, T-1, C, H, W]
        rep_codes = repeat(codes, 'k a -> (k b) t a', b=B, t=T-1)  # [K*B, T-1, A]
        warped = rearrange(self.decoder.warp_frames(rep_t, rep_codes), '(k b) t c h w -> b t k c h w', k=K)  # [B, T-1, K, C, H, W] fp32
        tgt = target.float()[:, :, None]  # [B, T-1, 1, C, H, W]
        losses = F.smooth_l1_loss(warped, tgt.expand_as(warped), reduction='none').mean(dim=(3, 4, 5))  # [B, T-1, K]
        local = None
        if self.decoder.warp_local:
            # camera still: frame t with a window around the change moved by the flipped kernel, scored inside the window
            local = self.decoder.local_window(frames)  # [B, T-1] bool, [B, T-1, 1, H, W] x 2
            is_local, w, g = local
            flipped = rearrange(self.decoder.warp_frames(rep_t, rep_codes, flip=True), '(k b) t c h w -> b t k c h w', k=K)
            ft = frames_t.float()[:, :, None]  # [B, T-1, 1, C, H, W]
            moved = ft + w[:, :, None] * (flipped - ft)  # [B, T-1, K, C, H, W]
            if self.decoder.warp_local_blur > 1:  # score at sprite scale (see warp_local_gate)
                kb = self.decoder.warp_local_blur
                blur = lambda t: F.avg_pool2d(t.flatten(0, -3), kb, 1, kb // 2, count_include_pad=False).reshape(t.shape)
                per_px = F.smooth_l1_loss(blur(moved), blur(tgt).expand_as(moved), reduction='none').mean(dim=3)  # [B, T-1, K, H, W]
            else:
                per_px = F.smooth_l1_loss(moved, tgt.expand_as(moved), reduction='none').mean(dim=3)  # [B, T-1, K, H, W]
            in_window = (per_px * g).sum(dim=(3, 4)) / g.sum(dim=(2, 3, 4))[..., None].clamp_min(1e-6)  # [B, T-1, K]
            losses = torch.where(is_local[:, :, None], in_window, losses)
            if self.decoder.warp == 'global_only':
                warped = torch.where(is_local[:, :, None, None, None, None], moved, warped)
        q = self.sinkhorn(losses.detach().reshape(-1, K))  # [N, K]
        warp_loss = (q * losses.reshape(-1, K)).sum(-1).mean()
        winner = q.argmax(-1).reshape(B, T-1)  # [B, T-1]
        enc_probs = self.code_probs(action_latents)  # [N, K]
        ce = -(q * enc_probs.clamp_min(1e-8).log()).sum(-1).mean()
        self.last_wta_agree = (enc_probs.argmax(-1) == q.argmax(-1)).float().mean().detach()
        if self.decoder.warp == 'global_only':
            pred_frames = warped.gather(2, winner[:, :, None, None, None, None].expand(B, T-1, 1, C, H, W))[:, :, 0]  # [B, T-1, C, H, W]
            recon_loss = warp_loss.new_zeros(())
        else:
            pred_frames = self.decoder(frames, codes[winner], training=True, local=local)  # [B, T-1, C, H, W] winner warp + correction
            recon_loss = F.smooth_l1_loss(pred_frames, target)
        total_loss = warp_loss + recon_loss + self.wta_encoder_weight * ce
        overlap = self.kernel_overlap(codes)  # also logged when off
        if self.wta_kernel_repulsion > 0:
            total_loss = total_loss + self.wta_kernel_repulsion * overlap
        if self.entropy_loss_weight > 0:
            _, h_batch = self.code_entropies(action_latents)
            total_loss = total_loss - self.entropy_loss_weight * h_batch
        return total_loss, pred_frames.to(frames.dtype)

    def aux_loss(self, action_latents, aux):
        # action_latents: [B, T-1, A] pre-tanh, aux: [B, T-1] long pseudo-labels (-1 = none) -> weighted CE (scalar)
        if self.aux_label_weight <= 0 or aux is None:
            return action_latents.new_zeros(()).float()
        valid = aux >= 0  # [B, T-1]
        if self.aux_label_target == 'codes':
            probs = self.code_probs(action_latents) @ self.aux_map.float().softmax(-1)  # [N, n_actions] @ [n_actions, K] -> [N, K]
            logp = probs.clamp_min(1e-8).log().reshape(*aux.shape, -1)  # [B, T-1, K]
            if not valid.any():
                return logp.sum() * 0.0
            self.last_aux_acc = (logp.argmax(-1)[valid] == aux[valid]).float().mean().detach()
            return self.aux_label_weight * F.nll_loss(logp[valid], aux[valid])
        logits = self.aux_head(torch.tanh(action_latents.float()))  # [B, T-1, K]
        if not valid.any():
            return logits.sum() * 0.0
        self.last_aux_acc = (logits.argmax(-1)[valid] == aux[valid]).float().mean().detach()
        return self.aux_label_weight * F.cross_entropy(logits[valid], aux[valid])

    def forward(self, frames, ot=None, aux=None):
        # frames: [B, T, C, H, W], ot: [B, T - 1, 2, P] OT plans (only for an OT-conditioned LAM),
        # aux: [B, T - 1] long pseudo-labels (only with aux_label_weight > 0)
        if self.warp_wta:
            assert ot is None, 'wta + OT not wired'
            action_latents = self.encoder(frames)  # [B, T-1, A]
            total_loss, pred_frames = self.forward_wta(frames, action_latents)
            return total_loss + self.aux_loss(action_latents, aux), pred_frames

        # get (quantized or sampled) action latents
        action_latents = self.encoder(frames, ot) # [B, T - 1, A] (continuous: [B, T - 1, 2A])
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
        pred_frames = self.decoder(frames, action_latents_quantized, training=True, ot=ot)  # [B, T - 1, C, H, W]

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

        if self.warp_entropy_weight > 0:
            ramp = (self.train_steps.float() / self.warp_entropy_ramp).clamp(max=1.0)
            total_loss = total_loss + self.warp_entropy_weight * ramp * self.decoder.last_warp_entropy
            if self.training:
                self.train_steps += 1

        total_loss = total_loss + self.aux_loss(action_latents[..., :self.action_dim], aux)
        return total_loss, pred_frames

    def encode(self, frames, ot=None):
        action_latents = self.pre_quant(frames, ot)  # [B, T, A]
        if self.continuous_actions:
            return action_latents  # the mean, no noise
        action_latents_quantized = self.quantizer(action_latents) # [B, T, A]
        return action_latents_quantized
    
    @property
    def model_type(self) -> str:
        return ModelType.LatentActionModel