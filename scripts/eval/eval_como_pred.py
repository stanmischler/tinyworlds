"""Next-frame prediction score of CoMo's forward dynamics decoder (models/como.py MotionDecoder), on the same held-out
windows as eval_next_frame.py, for reference next to the dynamics model. Informative only: the decoder is CoMo's training
aid, not a world model; it sees one frame (the last context frame t) plus the action inferred from (t, t+4).

Per window (eval_next_frame.test_windows, context 3, frame_skip 4, stride 8 -> 890 Zelda windows), target = frame t+4:
  - `como`      : decoder(frame t, z(t, t+4)), z = the IDM's full 128-d continuous action (true transition given, as
                  eval_next_frame --action-mode lam)
  - `como_k16`  : z replaced by its k-means centroid (16 clusters fit on all held-out z, seed 0): a 16-code action budget;
                  with --action-dir (STA-42, scripts/actions/como_actions.py) the centroids fit on train z that the k16 dynamics
                  model is conditioned on
  - `shuffled`  : z of another window (roll by 1): how much the prediction depends on the action
  - `copy`      : frame t (copy-last baseline)
Metrics: PSNR, SSIM (eval_next_frame's implementations, pixels in [0, 1]), LPIPS if installed, mean over windows.

    python scripts/eval/eval_como_pred.py --ckpt results/como_zelda_v1/como/checkpoints/como_step_50000 --name como_v1_50k
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch

from evaluation.action_metrics import kmeans
from evaluation.image_metrics import psnr, ssim, try_lpips
from evaluation.windows import test_windows, load_history_batch, to_model_range, to_unit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--test-h5', default='data/zelda_test_frames.h5')
    p.add_argument('--context', type=int, default=3)
    p.add_argument('--frame-skip', type=int, default=4)
    p.add_argument('--sample-stride', type=int, default=8)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--action-dir', help='CoMo action dir (scripts/actions/como_actions.py): use its train-fit centroids for como_k<k>')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out-dir', default='eval_results/como_pred')
    a = p.parse_args()
    from utils.utils import load_latent_actions_from_checkpoint
    lam, _ = load_latent_actions_from_checkpoint(a.ckpt, a.device)  # CoMoLAM: frozen features + IDM + decoder
    lam.eval()
    wins = test_windows(a.test_h5, a.context, a.frame_skip, a.sample_stride)
    zs, xs, ys = [], [], []
    with h5py.File(a.test_h5, 'r') as h5, torch.no_grad():
        for i in range(0, len(wins), a.batch_size):
            h = getattr(lam, 'history', 0)  # temporal-tokenizer CoMo: frame t and t+4 each with h earlier frames (STA-43)
            fr = to_model_range(load_history_batch(h5['frames'], wins[i:i + a.batch_size], a.context, a.frame_skip, h, a.test_h5), a.device)
            pair = fr[:, -2:]  # [B, 2, C, H, W]: last context frame t, target t+4
            zs.append(lam.encode(fr[:, -2 - h:])[:, -1].float())  # [B, A]: action (t, t+4)
            xs.append(pair[:, 0])
            ys.append(pair[:, 1])
    z, x, y = torch.cat(zs), torch.cat(xs), torch.cat(ys)  # [N, A], [N, C, H, W] x2
    if a.action_dir:
        c = torch.load(os.path.join(a.action_dir, 'state.pt'), map_location='cpu', weights_only=False)['centroids'].to(z.device)
        a.k = len(c)
    else:
        c = kmeans(z, a.k)
    zk = c[torch.cdist(z, c).argmin(1)]
    dec = lam.como.decoder
    shape = (-1, lam.como.idm.n_queries, lam.como.idm.down[-1].out_features)
    lpips_fn = try_lpips(a.device)
    per = {m: {'psnr': [], 'ssim': [], 'lpips': []} for m in ('como', f'como_k{a.k}', 'shuffled', 'copy')}
    with torch.no_grad():
        for i in range(0, len(z), a.batch_size):
            sl = slice(i, i + a.batch_size)
            tgt = to_unit(y[sl])
            preds = {'como': dec(x[sl], z[sl].view(shape)), f'como_k{a.k}': dec(x[sl], zk[sl].view(shape)),
                     'shuffled': dec(x[sl], z.roll(1, 0)[sl].view(shape)), 'copy': x[sl]}
            for m, pr in preds.items():
                pu = to_unit(pr)
                per[m]['psnr'] += psnr(pu, tgt).tolist()
                per[m]['ssim'] += ssim(pu, tgt).tolist()
                if lpips_fn is not None:
                    per[m]['lpips'] += lpips_fn(pu * 2 - 1, tgt * 2 - 1).flatten().tolist()
    summary = {m: {k: round(float(np.mean(v)), 4) for k, v in d.items() if v} for m, d in per.items()}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({'name': a.name, 'ckpt': a.ckpt, 'action_dir': a.action_dir, 'n_windows': len(wins), 'k': a.k, 'summary': summary},
              open(f'{a.out_dir}/{a.name}.json', 'w'), indent=1)
    print(a.name, len(wins), 'windows', json.dumps(summary))


if __name__ == '__main__':
    main()
