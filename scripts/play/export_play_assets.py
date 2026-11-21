"""Assets for the Bigfish play page (scripts/play/server.py), built where the data lives (CPU, about a minute):

  - commands: one 8-number action per control (8 directions + no move) = the mean standardized action of the training
    pairs whose true button is that control. The true buttons only name and place the keys; the dynamics model never
    saw them.
  - centroids: the 16 k-means centroids of the action dir (standardized), each named by its commonest true control,
    for the "16 learned commands" key map.
  - starts: held-out start windows: 3 real context frames, the 2 actions between them, and the next `--horizon` real
    frames with their actions (for "dream vs reality" replays).

  python scripts/play/export_play_assets.py            # -> results/bigfish_play/assets.npz (~20 MB)
"""
import argparse
import os

import h5py
import numpy as np
import torch

from evaluation.action_check import BIGFISH_CLASSES

CLASS_NAMES = ['left+down', 'left', 'left+up', 'down', 'no move', 'up', 'right+down', 'right', 'right+up']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-h5', default='data/bigfish_frames.h5')
    p.add_argument('--test-h5', default='data/bigfish_test_frames.h5')
    p.add_argument('--train-z', default='data/bigfish_como_z.npy')  # raw action per row (row t: frames t -> t+1)
    p.add_argument('--test-z', default='data/bigfish_test_como_z.npy')
    p.add_argument('--full-dir', default='results/bigfish_action_model_actions_full')  # mean / std of the training actions
    p.add_argument('--k-dir', default='results/bigfish_action_model_actions_k16')  # k-means centroids (raw z)
    p.add_argument('--n-starts', type=int, default=32)
    p.add_argument('--horizon', type=int, default=60)
    p.add_argument('--out', default='results/bigfish_play/assets.npz')
    a = p.parse_args()

    st = torch.load(os.path.join(a.full_dir, 'state.pt'), map_location='cpu', weights_only=False)
    mean, std = st['mean'].float().numpy(), st['std'].float().numpy()  # [A]
    cent = torch.load(os.path.join(a.k_dir, 'state.pt'), map_location='cpu', weights_only=False)['centroids'].float().numpy()

    # commands: per-control mean of the standardized training actions, inside episodes only
    z = (np.load(a.train_z).astype(np.float32) - mean) / std  # [N, A]
    with h5py.File(a.train_h5, 'r') as h5:
        buttons = h5['buttons'][:len(z)].astype(np.int64)
        ep = h5['episode_index'][:]
    same_ep = np.zeros(len(z), bool)
    same_ep[:len(ep) - 1] = ep[:-1] == ep[1:]  # row t and t+1 in one episode
    ok = same_ep[:len(z)] & np.isfinite(z).all(1)
    cls = np.asarray(BIGFISH_CLASSES)[buttons]  # [N] in 0..8
    commands = np.stack([z[ok & (cls == c)].mean(0) for c in range(9)]).astype(np.float32)  # [9, A]
    counts = np.array([(ok & (cls == c)).sum() for c in range(9)])

    # k-means centroids: standardized, named by the commonest true control of their training pairs
    cz = (cent - mean) / std  # [K, A]
    code = np.concatenate([np.linalg.norm(z[ok][i:i + 200000, None] - cz[None], axis=2).argmin(1)
                           for i in range(0, ok.sum(), 200000)])
    cmaj = np.array([np.bincount(cls[ok][code == k], minlength=9).argmax() if (code == k).any() else 4
                     for k in range(len(cz))])
    cpur = np.array([(cls[ok][code == k] == cmaj[k]).mean() if (code == k).any() else 0 for k in range(len(cz))])

    # starts: held-out windows of 3 + horizon frames inside one episode, spread over the file
    zt = (np.load(a.test_z).astype(np.float32) - mean) / std
    L = 3 + a.horizon
    with h5py.File(a.test_h5, 'r') as h5:
        ep_t, step_t = h5['episode_index'][:], h5['step_index'][:]
        n = len(ep_t)
        valid = [s for s in range(0, n - L, 7)
                 if ep_t[s] == ep_t[s + L - 1] and step_t[s + L - 1] - step_t[s] == L - 1 and np.isfinite(zt[s:s + L - 1]).all()]
        pick = [valid[i] for i in np.linspace(0, len(valid) - 1, a.n_starts).astype(int)]
        frames = np.stack([h5['frames'][s:s + L] for s in pick])  # [S, L, 64, 64, 3] uint8
        tbtn = np.stack([np.asarray(BIGFISH_CLASSES)[h5['buttons'][s:s + L - 1].astype(np.int64)] for s in pick])
    actions = np.stack([zt[s:s + L - 1] for s in pick]).astype(np.float32)  # [S, L-1, A]: actions[:, t] drives t -> t+1

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, commands=commands, command_names=np.array(CLASS_NAMES), command_counts=counts,
                        centroids=cz.astype(np.float32), centroid_class=cmaj, centroid_purity=cpur,
                        start_frames=frames, start_actions=actions, start_buttons=tbtn, start_rows=np.array(pick))
    print(f'wrote {a.out}: commands {commands.shape} (pairs per control {counts.tolist()}), centroids {cz.shape} '
          f'(purity {cpur.round(2).tolist()}), starts {frames.shape}')
    print('command norms', np.linalg.norm(commands, axis=1).round(2).tolist())


if __name__ == '__main__':
    main()
