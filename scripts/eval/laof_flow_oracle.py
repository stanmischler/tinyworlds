"""Flow oracle for the lam_judge set (STA-39): how much of the judged action is in the LAOF flow target itself?

Per judged transition, reads the camera-compensated flow (scripts/laof_flow.py) of frame_a -> frame_b, averages it over
the moving pixels (|flow| > `--px`; with --mask-tol > 0 only those the camera shift does not explain, as in
training: models/laof.camera_mask), and maps it to a code: 0 = still (fewer than `--min-pixels` moving pixels), 1-8 =
the 8 compass sectors of the mean vector. Writes codes.json and scores it with lam_judge (--codes).

    python scripts/eval/laof_flow_oracle.py                                   # -> eval_results/lam_judge/oracle_flow_masked/
    python scripts/eval/laof_flow_oracle.py --mask-tol 0 --name oracle_flow   # compensated flow only
"""

import argparse
import json
import math
import os
import subprocess
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.laof import camera_mask  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--flow', default='data/zelda_test_flow_gap4.h5')
    p.add_argument('--px', type=float, default=1.0)
    p.add_argument('--min-pixels', type=int, default=20)
    p.add_argument('--mask-tol', type=float, default=0.1)
    p.add_argument('--mask-dilate', type=int, default=2)
    p.add_argument('--name', default='oracle_flow_masked')
    a = p.parse_args()
    meta = json.load(open('eval_results/lam_judge/set/transitions.json'))
    codes = {}
    with h5py.File(a.flow, 'r') as f, h5py.File('data/zelda_test_frames.h5', 'r') as h:
        assert int(f.attrs['gap']) == meta['frame_skip']
        for tr in meta['transitions']:
            fl = f['flow'][tr['frame_a']].astype(np.float32)  # [2, H, W] px
            mag = np.linalg.norm(fl, axis=0)
            m = mag > a.px
            if a.mask_tol > 0:
                fr = torch.from_numpy(np.stack([h['frames'][tr['frame_a']], h['frames'][tr['frame_b']]])).permute(0, 3, 1, 2).float() / 127.5 - 1
                m &= camera_mask(fr[:1], fr[1:], torch.from_numpy(f['shift'][tr['frame_a']][None]), a.mask_tol, a.mask_dilate)[0].numpy()
            if m.sum() < a.min_pixels:
                codes[str(tr['id'])] = 0
                continue
            u, v = fl[0][m].mean(), fl[1][m].mean()
            codes[str(tr['id'])] = 1 + int(round(math.atan2(v, u) / (math.pi / 4))) % 8
    os.makedirs('eval_results/lam_judge', exist_ok=True)
    path = f'eval_results/lam_judge/{a.name}_codes.json'
    json.dump(codes, open(path, 'w'))
    print(f'code usage {np.bincount(list(codes.values()), minlength=9).tolist()}')
    subprocess.run([sys.executable, 'scripts/eval/lam_judge.py', 'score', '--codes', f'{a.name}={path}'], check=True)


if __name__ == '__main__':
    main()
