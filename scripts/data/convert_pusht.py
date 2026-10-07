"""Convert DINO-WM's Push-T (`pusht_noise`, the dataset used by NanoWM, LeWM/stable-worldmodel and JEPA-WMs) into
the two .h5 files tinyworlds uses.

Source: https://osf.io/download/k2d8w/ (pusht_noise.zip, 2.79 GB): per split `obses/episode_NNN.mp4` (224x224, one
frame per env step), `rel_actions.pth` [N_ep, L_max, 2] (zero padded), `seq_lengths.pkl` (true length per episode).
Train 18,685 episodes / 2,336,736 frames, val 21 episodes / 2,514 frames.

Outputs (in --out-dir):
  pusht_frames.h5      training set, NanoWM's convention: frame_interval 5, the 5 env actions between two kept
                       frames concatenated into one 10-D action. Only phase-0 frames are kept (steps 0, 5, 10, ...).
                         frames        uint8   [N, S, S, 3]  INTER_AREA resize of the 224px frame
                         actions       float32 [N, 10]       z-scored rel_actions/100 of steps 5k..5k+4 (the action that
                                                             leads from kept frame k to k+1; zeros past the episode end)
                         episode_index int32   [N]
                         step_index    int32   [N]           env step of the frame
  pusht_val_frames.h5  the shipped val split at native 224px and every env step, for scripts/eval/eval_pusht.py
                         frames        uint8   [N, 224, 224, 3]
                         rel_actions   float32 [N, 2]        rel_actions/100, NOT normalised
                         episode_index, step_index           as above
  Both carry a seq_lengths dataset and attrs action_mean / action_std (per dim of rel_actions/100, over the valid train steps; std + 1e-6 as
  NanoWM), frame_interval and action_scale.

Usage (repo root):
    python scripts/data/convert_pusht.py --raw-dir data/pusht_noise          # downloads + unzips if --raw-dir is missing
    python scripts/data/convert_pusht.py --raw-dir data/pusht_noise --limit-episodes 50 --out-dir /tmp/pusht   # smoke
On Modal (no 23 GB upload from the laptop):  modal run scripts/infra/modal_train.py::convert_pusht
"""

import argparse
import os
import pickle
import time
import urllib.request
import zipfile
from multiprocessing import Pool

import cv2
import h5py
import numpy as np
import torch

OSF_URL = 'https://osf.io/download/k2d8w/'
FRAME_INTERVAL = 5
ACTION_SCALE = 100.0


def ensure_raw(raw_dir):
    """Download and unzip pusht_noise.zip next to raw_dir unless raw_dir/val/seq_lengths.pkl already exists."""
    if os.path.exists(os.path.join(raw_dir, 'val', 'seq_lengths.pkl')):
        return
    parent = os.path.dirname(os.path.abspath(raw_dir))
    os.makedirs(parent, exist_ok=True)
    zip_path = os.path.join(parent, 'pusht_noise.zip')
    if not os.path.exists(zip_path):
        print(f'downloading {OSF_URL} -> {zip_path}', flush=True)
        urllib.request.urlretrieve(OSF_URL, zip_path + '.part')
        os.replace(zip_path + '.part', zip_path)
    print(f'unzipping {zip_path}', flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(parent)  # creates <parent>/pusht_noise/{train,val}
    extracted = os.path.join(parent, 'pusht_noise')
    if os.path.abspath(extracted) != os.path.abspath(raw_dir):
        os.replace(extracted, raw_dir)
    os.remove(zip_path)


def load_split(raw_dir, split):
    d = os.path.join(raw_dir, split)
    with open(os.path.join(d, 'seq_lengths.pkl'), 'rb') as f:
        seq_lengths = [int(x) for x in pickle.load(f)]
    rel = torch.load(os.path.join(d, 'rel_actions.pth')).float().numpy() / ACTION_SCALE  # [N_ep, L_max, 2]
    return d, seq_lengths, rel


def decode_episode(job):
    """-> uint8 [n_kept, H, W, 3] RGB; every `step`-th frame, resized to `size` (None = native)."""
    mp4, length, step, size = job
    cap = cv2.VideoCapture(mp4)
    frames = []
    for t in range(length):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'{mp4}: decoded {t} frames, expected {length}')
        if t % step:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if size is not None and frame.shape[0] != size:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.stack(frames)


def write_split(out_path, split_dir, seq_lengths, rel, step, size, mean, std, workers, actions_mode):
    n_ep = len(seq_lengths)
    n_frames = sum((L + step - 1) // step for L in seq_lengths)
    res = size if size is not None else 224
    jobs = [(os.path.join(split_dir, 'obses', f'episode_{i:03d}.mp4'), seq_lengths[i], step, size) for i in range(n_ep)]
    t0 = time.time()
    with h5py.File(out_path + '.part', 'w') as f:
        frames_d = f.create_dataset('frames', (n_frames, res, res, 3), dtype=np.uint8,
                                    chunks=(min(64, n_frames), res, res, 3), compression='lzf')
        a_dim = 2 * step if actions_mode == 'chunked' else 2
        act_d = f.create_dataset('actions' if actions_mode == 'chunked' else 'rel_actions', (n_frames, a_dim), dtype=np.float32)
        ep_d = f.create_dataset('episode_index', (n_frames,), dtype=np.int32)
        st_d = f.create_dataset('step_index', (n_frames,), dtype=np.int32)
        pos = 0
        with Pool(workers) as pool:
            for i, frames in enumerate(pool.imap(decode_episode, jobs, chunksize=4)):
                L, n = seq_lengths[i], len(frames)
                steps = np.arange(0, L, step)
                assert len(steps) == n
                if actions_mode == 'chunked':
                    a = np.zeros((n * step, 2), dtype=np.float32)  # [n*step, 2]
                    a[:L] = (rel[i, :L] - mean) / std
                    a = a.reshape(n, step * 2)  # [n, 10]: actions of steps 5k..5k+4, flattened step-major as NanoWM
                else:
                    a = rel[i, :L]
                frames_d[pos:pos + n] = frames
                act_d[pos:pos + n] = a
                ep_d[pos:pos + n] = i
                st_d[pos:pos + n] = steps
                pos += n
                if (i + 1) % 500 == 0 or i + 1 == n_ep:
                    print(f'  {out_path}: {i + 1}/{n_ep} episodes, {pos} frames ({time.time() - t0:.0f}s)', flush=True)
        assert pos == n_frames
        f.attrs['action_mean'] = mean
        f.attrs['action_std'] = std
        f.attrs['frame_interval'] = step if actions_mode == 'chunked' else 1
        f.attrs['action_scale'] = ACTION_SCALE
        f.create_dataset('seq_lengths', data=np.array(seq_lengths, dtype=np.int32))  # too big for an attr (64 KB)
        f.attrs['source'] = 'DINO-WM pusht_noise (' + OSF_URL + ')'
    os.replace(out_path + '.part', out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-dir', default='data/pusht_noise')
    p.add_argument('--out-dir', default='data')
    p.add_argument('--size', type=int, default=128, help='training resolution')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--limit-episodes', type=int, help='convert only the first N train episodes (smoke test)')
    p.add_argument('--skip-train', action='store_true')
    args = p.parse_args()

    ensure_raw(args.raw_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    train_dir, train_len, train_rel = load_split(args.raw_dir, 'train')
    val_dir, val_len, val_rel = load_split(args.raw_dir, 'val')

    # action stats over the valid steps of ALL train episodes (as NanoWM), also with --limit-episodes
    valid = np.concatenate([train_rel[i, :L] for i, L in enumerate(train_len)])  # [sum L, 2]
    mean = valid.mean(0).astype(np.float32)
    std = (valid.std(0, ddof=1) + 1e-6).astype(np.float32)  # torch.std default is unbiased
    print(f'train: {len(train_len)} episodes, {sum(train_len)} frames; val: {len(val_len)} episodes, {sum(val_len)} frames')
    print(f'action mean {mean}, std {std}')

    write_split(os.path.join(args.out_dir, 'pusht_val_frames.h5'), val_dir, val_len, val_rel,
                step=1, size=None, mean=mean, std=std, workers=args.workers, actions_mode='raw')
    if not args.skip_train:
        n = args.limit_episodes or len(train_len)
        write_split(os.path.join(args.out_dir, 'pusht_frames.h5'), train_dir, train_len[:n], train_rel[:n],
                    step=FRAME_INTERVAL, size=args.size, mean=mean, std=std, workers=args.workers, actions_mode='chunked')
    print('done')


if __name__ == '__main__':
    main()
