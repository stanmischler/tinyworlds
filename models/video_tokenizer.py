from models.utils import ModelType
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.st_transformer import STTransformer
from models.fsq import FiniteScalarQuantizer
from models.patch_embed import PatchEmbedding
from models.positional_encoding import build_spatial_only_pe

class VideoTokenizerEncoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8, 
                 hidden_dim=256, num_blocks=4, latent_dim=5, per_frame=False):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True, temporal=not per_frame)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim)
        )

    def forward(self, frames):
        # frames: [B, T, C, H, W]
        # frames to patch embeddings, pass through transformer, project to latent dim
        embeddings = self.patch_embed(frames)  # [B, T, P, E]
        transformed = self.transformer(embeddings) # [B, T, P, E]
        predicted_latents = self.latent_head(transformed) # [B, T, P, L]
        return predicted_latents


class PixelShuffleFrameHead(nn.Module):
    # conv2D embeddings to pixels head
    def __init__(self, embed_dim, patch_size=8, channels=3, H=128, W=128):
        super().__init__()
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.to_pixels = nn.Conv2d(embed_dim, channels * (patch_size ** 2), kernel_size=1)

    def forward(self, tokens):  # [B, T, P, E]
        B, T, P, E = tokens.shape
        x = rearrange(tokens, 'b t (hp wp) e -> (b t) e hp wp', hp=self.Hp, wp=self.Wp) # [(B*T), E, Hp, Wp]
        x = self.to_pixels(x)                  # [(B*T), C*p^2, Hp, Wp]
        x = rearrange(x, '(b t) (c p1 p2) hp wp -> b t c (hp p1) (wp p2)', p1=self.patch_size, p2=self.patch_size, b=B, t=T) # [B, T, C, H, W]
        return x


class VideoTokenizerDecoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, latent_dim=5, per_frame=False):
        super().__init__()
        H, W = frame_size
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.num_patches = self.Hp * self.Wp
        
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True, temporal=not per_frame)
        self.frame_head = PixelShuffleFrameHead(embed_dim, patch_size=patch_size, channels=3, H=H, W=W)

        # first 2/3 spatial PE (temporal is last 1/3)
        pe_spatial_dec = build_spatial_only_pe((H, W), self.patch_size, embed_dim, device='cpu', dtype=torch.float32)  # [1,P,E]
        self.register_buffer("pos_spatial_dec", pe_spatial_dec, persistent=False)

    def forward(self, latents):
        # latents: [B, T, P, L]
        # embed latents and add spatial PE
        embedding = self.latent_embed(latents)  # [B, T, P, E]
        embedding = embedding + self.pos_spatial_dec.to(dtype=embedding.dtype, device=embedding.device)

        # apply transformer (temporal PE added inside)
        embedding = self.transformer(embedding)  # [B, T, P, E]

        # reconstruct frames using patch-wise head
        frames_out = self.frame_head(embedding)  # [B, T, C, H, W]

        return frames_out


class VideoTokenizer(nn.Module):
    """bottleneck 'fsq' (default): discrete FSQ codes. Continuous bottlenecks (STA-62) keep the latents unquantized:
    'tanh' bounds them with tanh and trains the decoder on noised latents (RAE: per-sample std ~ |N(0, latent_noise^2)| in
    units of the latent RMS); 'kl' is a KL-VAE (encoder emits mean and log-variance, KL weight kl_weight). encode()/decode()
    are the latent API for the dynamics model: FSQ grid values, or continuous latents divided by latent_scale (running RMS
    of the training latents) so they have unit RMS."""
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, latent_dim=3, num_bins=4, per_frame=False,
                 bottleneck='fsq', latent_noise=0.8, kl_weight=1e-6):
        super().__init__()
        assert bottleneck in ('fsq', 'tanh', 'kl'), bottleneck
        self.bottleneck, self.latent_noise, self.kl_weight = bottleneck, float(latent_noise), float(kl_weight)
        # per_frame=True: encoder and decoder have no temporal attention, so a frame's tokens depend on that frame only
        enc_dim = 2 * latent_dim if bottleneck == 'kl' else latent_dim  # kl: [mean, log-variance]
        self.encoder = VideoTokenizerEncoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, enc_dim, per_frame)
        self.decoder = VideoTokenizerDecoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim, per_frame)
        self.quantizer = FiniteScalarQuantizer(latent_dim, num_bins)
        self.codebook_size = num_bins**latent_dim
        if self.continuous:
            self.register_buffer('latent_scale', torch.ones(()))  # running RMS of the continuous latents (saved)

    @property
    def continuous(self):
        return self.bottleneck != 'fsq'

    def _bottleneck(self, h):
        # h: [B, T, P, L or 2L] encoder output -> (latents [B, T, P, L], deterministic latents [B, T, P, L], kl or None);
        # kl samples only in training, and its deterministic latents (what encode() returns) are the mean
        if self.bottleneck == 'fsq':
            z = self.quantizer(h)
            return z, z, None
        if self.bottleneck == 'tanh':
            z = torch.tanh(h)
            return z, z, None
        mean, logvar = h.float().chunk(2, dim=-1)  # [B, T, P, L] each
        logvar = logvar.clamp(-30, 20)
        kl = 0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar).mean()
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if self.training else mean
        return z, mean, kl

    def forward(self, frames):
        # encode frames to latent representations, quantize (or bottleneck), and decode back to frames
        embeddings = self.encoder(frames)  # [B, T, P, L]
        z, z_det, kl = self._bottleneck(embeddings)  # [B, T, P, L]
        if self.continuous and self.training:
            with torch.no_grad():  # running RMS of encode()'s latents -> unit-RMS latents for the dynamics model
                self.latent_scale.lerp_(z_det.detach().float().pow(2).mean().sqrt(), 0.01)
            if self.bottleneck == 'tanh' and self.latent_noise > 0:
                std = (torch.randn(z.shape[0], 1, 1, 1, device=z.device).abs() * self.latent_noise) * self.latent_scale  # [B, 1, 1, 1]
                z = z + std * torch.randn_like(z)
        x_hat = self.decoder(z)  # [B, T, C, H, W]
        recon_loss = F.smooth_l1_loss(x_hat, frames)
        if kl is not None:
            recon_loss = recon_loss + self.kl_weight * kl
        return recon_loss, x_hat

    def tokenize(self, frames):
        # encode frames to latent representations, quantize, and return indices
        assert not self.continuous, 'continuous tokenizer has no indices: use encode()'
        embeddings = self.encoder(frames)  # [B, T, P, L]
        quantized_z = self.quantizer(embeddings)
        indices = self.quantizer.get_indices_from_latents(quantized_z, dim=-1)
        return indices

    def detokenize(self, quantized_z):
        # decode quantized latents back to frames
        x_hat = self.decoder(quantized_z)  # [B, T, C, H, W]
        return x_hat

    def encode(self, frames):
        # frames [B, T, C, H, W] -> dynamics latents [B, T, P, L]: FSQ grid values (exactly as indices -> latents), or
        # deterministic continuous latents / latent_scale
        if not self.continuous:
            return self.quantizer.get_latents_from_indices(self.tokenize(frames), dim=-1)
        _, z, _ = self._bottleneck(self.encoder(frames))
        return z / self.latent_scale

    def decode(self, latents):
        # dynamics latents [B, T, P, L] -> frames [B, T, C, H, W]
        return self.decoder(latents * self.latent_scale if self.continuous else latents)

    @property
    def model_type(self) -> str:
        return ModelType.VideoTokenizer