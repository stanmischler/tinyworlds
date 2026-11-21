"""Long-rollout check of a flow dynamics model (evaluation/rollout_vs_real.py): dream `horizon` frames from held-out clips with
their true actions and compare with reality frame by frame (PSNR, object ratio). Clips come from the play assets
(scripts/play/export_play_assets.py, e.g. --n-starts 128 --horizon 60).

  python scripts/eval/eval_rollout.py --dynamics results/bigfish_dynamics/dynamics/checkpoints/dynamics_step_99999 \
      --out eval_results/rollout/bigfish_dynamics_s32 [--context-noise 0.0] [--steps 32]
-> <out>.json (per-frame curves, summary at frames 10/30/60) and <out>.png (dream vs reality strips, first clips).
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from evaluation.rollout_vs_real import rollout, rollout_metrics
from utils.utils import load_dynamics_from_checkpoint, load_videotokenizer_from_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dynamics', required=True)
    p.add_argument('--tokenizer', default='results/bigfish_tokenizer_long/video_tokenizer/checkpoints/video_tokenizer_step_99000')
    p.add_argument('--assets', default='results/bigfish_play/assets.npz')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--context-noise', type=float, default=0.0)  # sampling-time context noise (models trained with it)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-show', type=int, default=4)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()

    tok, _ = load_videotokenizer_from_checkpoint(a.tokenizer, a.device)
    dyn, _ = load_dynamics_from_checkpoint(a.dynamics, a.device)  # EMA weights
    tok.eval(), dyn.eval()
    dyn.context_noise = a.context_noise
    z = np.load(a.assets)
    frames, actions = z['start_frames'], z['start_actions']  # [S, 3+H, 64, 64, 3], [S, 2+H, A]
    dream = rollout(tok, dyn, frames, actions, a.steps, seed=a.seed, device=a.device)  # [S, H, 64, 64, 3]
    m = rollout_metrics(dream, frames[:, 3:])

    H = dream.shape[1]
    at = [t for t in (10, 30, 60) if t <= H]
    summary = {f'{k}_at_{t}': float(m[k][t - 1]) for k in ('psnr', 'object_ratio') for t in at}
    res = {'dynamics': a.dynamics, 'steps': a.steps, 'context_noise': a.context_noise, 'clips': len(frames),
           'horizon': H, 'summary': summary, **{k: v.tolist() for k, v in m.items()}}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out + '.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))

    # strips: per clip one dream row and one reality row at frames 1, 10, 20, ..., H
    ts = [0] + list(range(9, H, 10))
    rows = []
    for s in range(min(a.n_show, len(frames))):
        rows += [np.concatenate([dream[s, t] for t in ts], 1), np.concatenate([frames[s, 3 + t] for t in ts], 1)]
    img = np.concatenate(rows, 0)
    Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(a.out + '.png')


if __name__ == '__main__':
    main()
