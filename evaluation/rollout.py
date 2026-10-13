"""Autoregressive rollouts of a tokenizer + dynamics pair, shared by the benchmark evals (NanoWM suite, LPWM
protocol). Works for both tokenizer kinds (FSQ / continuous, via tok.encode / tok.decode) and both dynamics kinds
(MaskGIT / flow, via forward_inference)."""

import torch


@torch.no_grad()
def rollout(tok, dyn, frames, n_context, actions=None, max_context=None, num_steps=1, temperature=0.0, batch_size=16,
            generator=None, progress=True):
    """Predict frames[:, n_context:] from frames[:, :n_context], one frame at a time, each new frame fed back as context.

    frames: float [B, T, C, H, W] in [-1, 1] (only the first n_context frames are read)
    actions: float [B, T - 1, A] (actions[:, t] drives frame t -> t + 1) or None
    max_context: keep at most this many frames as dynamics context (sliding window, e.g. 6 for LPWM's 6 -> 44); None = all
    num_steps: Euler steps (flow) or unmasking iterations (MaskGIT) per frame
    -> pred float [B, T, C, H, W] in [0, 1] (context frames are the tokenizer reconstructions), recon float [B, T, C, H, W]
       in [0, 1] (all T real frames encoded and decoded: the tokenizer ceiling)."""
    B, T = frames.shape[:2]
    preds, recons = [], []
    idx_to_lat = None if tok.continuous else (lambda idx: tok.quantizer.get_latents_from_indices(idx, dim=-1))  # MaskGIT only
    for i in range(0, B, batch_size):
        xb = frames[i:i + batch_size]
        ab = None if actions is None else actions[i:i + batch_size]
        lat = tok.encode(xb[:, :n_context])  # [b, Tc, P, L]; per-frame / causal tokenizer: same as encoding the whole clip
        for t in range(n_context, T):  # predict frame t
            ctx = lat if max_context is None else lat[:, -max_context:]  # [b, <=max_context, P, L]
            first = t - ctx.shape[1]  # global index of the first context frame
            cond = None if ab is None else ab[:, first:t]  # [b, Tc, A]; the last one drives frame t - 1 -> t
            kw = {} if generator is None else {'generator': generator}
            out = dyn.forward_inference(ctx, prediction_horizon=1, num_steps=num_steps, conditioning=cond,
                                        index_to_latents_fn=idx_to_lat, temperature=temperature, **kw)  # [b, Tc + 1, P, L]
            lat = torch.cat([lat, out[:, -1:].to(lat.dtype)], dim=1)
        preds.append(((decode_frames(tok, lat) + 1) / 2).clamp(0, 1).cpu())
        recons.append(((decode_frames(tok, tok.encode(xb)) + 1) / 2).clamp(0, 1).cpu())
        if progress:
            print(f'  rollout {min(i + batch_size, B)}/{B}', flush=True)
    return torch.cat(preds), torch.cat(recons)


def decode_frames(tok, lat, chunk=16):
    # [B, T, P, L] -> [B, T, C, H, W] in [-1, 1]. A per-frame tokenizer decodes frames independently, so long rollouts are
    # decoded in chunks of frames; a temporal one needs the whole clip at once
    if not getattr(tok.decoder.transformer, 'temporal', True):
        return torch.cat([tok.decode(lat[:, s:s + chunk]) for s in range(0, lat.shape[1], chunk)], dim=1)
    return tok.decode(lat)
