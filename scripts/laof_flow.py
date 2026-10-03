"""Precompute the LAOF optical-flow targets for an .h5 of frames (STA-39).

For every stored frame i, flow[i] = RAFT flow from frame i to frame i + gap (gap 4 = ZeldaDataset frame_skip, so the
pairs are exactly the LAM's transitions). As in LAOF (sample_datasets_with_opticalflow_sam_masknums.py), RAFT runs on a
x4 nearest-upsampled copy (128 -> 512 px, 20 refinement iterations); the result is average-pooled back to the frame grid
and rescaled to frame pixels. Camera compensation (our replacement for LAOF's LangSAM agent mask): the per-pair median
flow (the dominant whole-screen shift: the camera follows Link) is subtracted, so the background goes to ~0 and what is
left is Link's (and the enemies') motion in world coordinates. The raw median is stored too.

Output h5 (next to the frames): flow [N, 2, H, W] float16 (camera-compensated, frame px, (dx, dy)), shift [N, 2] float32
(the subtracted median), valid [N] bool (False for the last `gap` rows), attrs gap/upsample/iters/weights/source.
RGB encoding (LAOF's "paper" HSV method, sigma) happens at train time (models/laof.py), so sigma can change without
recomputing.

Usage (repo root, PYTHONPATH=$PWD; GPU work runs on Modal: scripts/modal_train.py::laof_flow):
    python scripts/laof_flow.py --h5 data/zelda_test_frames.h5 --out data/zelda_test_flow_gap4.h5 [--limit 256] [--viz out.png]
"""

import argparse
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def load_raft(device):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    # C_T_SKHT_V2: chairs -> things -> sintel+kitti+hd1k+things, the torchvision analogue of LAOF's raft-sintel.pth
    return raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).eval().to(device)


@torch.no_grad()
def pair_flow(raft, a, b, upsample=4, iters=20):
    # a, b: uint8 [B, H, W, C] -> flow [B, 2, H, W] in frame px (dx, dy), median shift [B, 2]
    x = torch.stack([a, b]).permute(0, 1, 4, 2, 3).float() / 127.5 - 1  # [2, B, C, H, W] in [-1, 1]
    H, W = x.shape[-2:]
    x = F.interpolate(x.flatten(0, 1), scale_factor=upsample, mode='nearest').unflatten(0, (2, -1))  # [2, B, C, uH, uW]
    flow = raft(x[0], x[1], num_flow_updates=iters)[-1]  # [B, 2, uH, uW] in upsampled px
    flow = F.avg_pool2d(flow, upsample) / upsample  # [B, 2, H, W] in frame px
    shift = flow.flatten(2).median(dim=2).values  # [B, 2]
    return flow - shift[:, :, None, None], shift


def flow_to_rgb(flow, sigma):
    # flow [B, 2, H, W] -> LAOF "paper" encoding: hue = direction, saturation = value = min(1, |flow| / (sigma * diag))
    from models.laof import flow_to_rgb as f2r
    return f2r(flow, sigma)


def save_viz(path, frames_a, frames_b, flow, shift, sigma=0.02):
    from PIL import Image, ImageDraw
    rgb = ((flow_to_rgb(flow.float(), sigma) + 0.5).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, 3]
    d = np.clip(np.abs(frames_b.astype(np.int16) - frames_a.astype(np.int16)).max(-1) * 3, 0, 255).astype(np.uint8)
    rows = []
    for k in range(len(rgb)):
        rows.append(np.concatenate([frames_a[k], frames_b[k], np.repeat(d[k][..., None], 3, -1), rgb[k]], 1))
    img = Image.fromarray(np.concatenate(rows, 0)).resize((4 * 128 * 2, len(rows) * 128 * 2), Image.NEAREST)
    dr = ImageDraw.Draw(img)
    for k in range(len(rgb)):
        dr.text((4, k * 256 + 4), f'shift ({shift[k, 0]:+.1f}, {shift[k, 1]:+.1f}) px | max |flow| {flow[k].norm(dim=0).max():.1f}', fill='white')
    img.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--gap', type=int, default=4)
    p.add_argument('--upsample', type=int, default=4)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--limit', type=int, default=0, help='only the first N rows (smoke)')
    p.add_argument('--viz', default='', help='PNG of 16 spread-out pairs: frame t | t+gap | |change| | compensated flow')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()

    raft = load_raft(a.device)
    with h5py.File(a.h5, 'r') as src:
        frames = src['frames'][:a.limit] if a.limit else src['frames'][:]  # [N, H, W, C] uint8
    N, H, W, _ = frames.shape
    n_pairs = N - a.gap
    t0 = time.time()
    with h5py.File(a.out, 'w') as dst:
        fl = dst.create_dataset('flow', (N, 2, H, W), dtype='float16', chunks=(64, 2, H, W), compression='lzf')
        sh = dst.create_dataset('shift', (N, 2), dtype='float32')
        dst.create_dataset('valid', data=np.arange(N) < n_pairs)
        dst.attrs.update(gap=a.gap, upsample=a.upsample, iters=a.iters, weights='raft_large C_T_SKHT_V2', source=a.h5,
                         compensation='per-pair median subtracted')
        for s in range(0, n_pairs, a.batch):
            e = min(s + a.batch, n_pairs)
            fa = torch.from_numpy(frames[s:e]).to(a.device)
            fb = torch.from_numpy(frames[s + a.gap:e + a.gap]).to(a.device)
            flow, shift = pair_flow(raft, fa, fb, a.upsample, a.iters)
            fl[s:e] = flow.half().cpu().numpy()
            sh[s:e] = shift.cpu().numpy()
            if (s // a.batch) % 200 == 0:
                print(f'{e}/{n_pairs} pairs, {e / (time.time() - t0):.1f} pairs/s', flush=True)
        fl[n_pairs:] = 0
        mag = np.linalg.norm(sh[:n_pairs], axis=1)
        print(f'done {n_pairs} pairs in {time.time() - t0:.0f} s; |shift| > 0.5 px on {np.mean(mag > 0.5):.1%} of pairs')

        if a.viz:
            idx = np.linspace(0, n_pairs - 1, 16).astype(int)
            save_viz(a.viz, frames[idx], frames[idx + a.gap], torch.from_numpy(fl[np.sort(idx)].astype(np.float32)), sh[np.sort(idx)])
            print(f'viz -> {a.viz}')


if __name__ == '__main__':
    main()
