"""Precompute frozen MAE ViT-L/16 features (models/como.py MAEFeatures, fixed patch order) for every frame of a frames .h5,
so CoMo training (scripts/train_como.py) only runs the motion IDM and the decoder.

Output: <out>.npy, float16 [N, S=197, 1024], row i = frame i of the .h5 (~404 KB per frame; zelda_train 64,850 frames
-> 26 GB). Features are computed in fp32 (tf32 matmuls), stored in fp16.

    python scripts/como_features.py --h5 data/zelda_train_frames.h5 --out data/zelda_train_mae_large.npy --device cuda
"""

import argparse
import time

import h5py
import numpy as np
import torch

from models.como import MAEFeatures, MAE_TOKENS, MAE_DIM


def check(mae, a):
    # our features == HF's default (randomly shuffled) output put back in image order with its ids_restore; and the
    # features are deterministic (the same frame twice -> identical rows)
    import torch.nn.functional as F
    with h5py.File(a.h5, 'r') as h5:
        x = torch.from_numpy(h5['frames'][:8]).to(a.device).permute(0, 3, 1, 2).float() / 127.5 - 1
    ours = mae(x)
    with torch.no_grad():
        px = (F.interpolate((x + 1) / 2, size=(224, 224), mode='bilinear', align_corners=False) - mae.mean) / mae.std
        o = mae.vit(pixel_values=px)  # random patch shuffle
        hf = o.last_hidden_state
        restored = torch.cat([hf[:, :1], torch.gather(hf[:, 1:], 1, o.ids_restore.unsqueeze(-1).expand(-1, -1, hf.shape[-1]))], 1)
    print(f'max |ours - HF unshuffled| = {(ours - restored).abs().max().item():.2e}; '
          f'max |ours - HF shuffled (CoMo code)| = {(ours - hf).abs().max().item():.2e}; '
          f'repeat diff = {(mae(x) - ours).abs().max().item():.2e}; feature std {ours.std().item():.3f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--limit', type=int, default=0, help='only the first N frames (smoke)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--check', action='store_true', help='verify the fixed patch order on 8 frames, then exit')
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    mae = MAEFeatures().to(a.device)
    if a.check:
        return check(mae, a)
    with h5py.File(a.h5, 'r') as h5:
        fr = h5['frames']
        n = min(len(fr), a.limit) if a.limit else len(fr)
        out = np.lib.format.open_memmap(a.out, mode='w+', dtype=np.float16, shape=(n, MAE_TOKENS, MAE_DIM))
        t0 = time.time()
        for i in range(0, n, a.batch):
            x = torch.from_numpy(fr[i:min(i + a.batch, n)]).to(a.device).permute(0, 3, 1, 2).float() / 127.5 - 1  # [N, C, H, W]
            out[i:i + len(x)] = mae(x).half().cpu().numpy()
            if (i // a.batch) % 20 == 0:
                print(f'{i + len(x)}/{n} frames, {time.time() - t0:.0f} s', flush=True)
        out.flush()
    print(f'wrote {a.out}: [{n}, {MAE_TOKENS}, {MAE_DIM}] fp16 in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
