"""Precompute frozen video-tokenizer features (models/como.py TokenizerFeatures) for every frame of a frames .h5, so CoMo
training (scripts/actions/train_como.py) can read them in place of the MAE features (STA-43).

Output: <out>.npy, float16 [N, S, D], row i = frame i of the .h5. With patch 4 at 128 px and merge 2: S = 256,
D = 4 * 6 = 24 (quant) or 4 * 256 = 1024 (hidden; ~512 KB per frame, zelda_train ~34 GB).
--history h (temporal tokenizer): frame t is encoded as the last frame of [t - h*skip, ..., t - skip, t] (skip = the
tokenizer's training frame spacing, 4 for Zelda), indices clamped to the start of t's contiguous segment (train split
cuts / test blocks, as scripts/actions/train_como.py), the same rule as eval_next_frame.load_history_batch.

    python scripts/actions/tok_features.py --h5 data/zelda_train_frames.h5 --tokenizer <ckpt dir> --mode quant --history 0 \
        --out data/zelda_train_tok_v5_quant.npy
"""

import argparse
import time

import h5py
import numpy as np
import torch

from datasets.split import segments
from models.como import TokenizerFeatures
from utils.utils import load_videotokenizer_from_checkpoint


def history_index(n, segs, history, skip):
    # -> [N, history+1] frame indices (oldest first) per frame, clamped to the frame's segment start
    seg_start = np.zeros(n, dtype=np.int64)
    for s, e in segs:
        seg_start[s:e] = s
    t = np.arange(n)[:, None]
    return np.maximum(t - skip * np.arange(history, -1, -1)[None], seg_start[:, None])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--tokenizer', required=True, help='video_tokenizer_step_N checkpoint dir')
    p.add_argument('--mode', choices=['quant', 'hidden'], required=True)
    p.add_argument('--history', type=int, default=0)
    p.add_argument('--skip', type=int, default=4)
    p.add_argument('--merge', type=int, default=2)
    p.add_argument('--out', required=True)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--limit', type=int, default=0, help='only the first N frames (smoke)')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, st = load_videotokenizer_from_checkpoint(a.tokenizer, 'cpu')
    assert a.history == 0 or not st['config'].get('per_frame'), 'per-frame tokenizer: use --history 0'
    feat = TokenizerFeatures(tok, a.mode, a.merge).to(a.device)
    with h5py.File(a.h5, 'r') as h5:
        n_file = len(h5['frames'])
        n = min(n_file, a.limit) if a.limit else n_file
        frames = h5['frames'][:n]  # uint8 [N, H, W, C]
    segs = [(s, min(e, n)) for s, e in segments(a.h5, n_file) if s < n]
    idx = history_index(n, segs, a.history, a.skip)  # [N, h+1]
    S = (tok.encoder.patch_embed.Hp // a.merge) ** 2
    out = np.lib.format.open_memmap(a.out, mode='w+', dtype=np.float16, shape=(n, S, feat.dim))
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, a.batch):
            x = torch.from_numpy(frames[idx[i:i + a.batch]]).to(a.device)  # [B, h+1, H, W, C]
            x = x.permute(0, 1, 4, 2, 3).float() / 127.5 - 1  # [B, h+1, C, H, W]
            out[i:i + len(x)] = feat(x).half().cpu().numpy()
            if (i // a.batch) % 50 == 0:
                print(f'{i + len(x)}/{n} frames, {time.time() - t0:.0f} s', flush=True)
    out.flush()
    f = torch.from_numpy(np.asarray(out[:min(n, 512)])).float()
    print(f'DONE {a.out} {out.shape}; feature mean {f.mean():.3f} std {f.std():.3f}; {len(segs)} segments; '
          f'{(idx[:, 0] == idx[:, -1]).mean() if a.history else 1:.3f} of frames without history', flush=True)


if __name__ == '__main__':
    main()
