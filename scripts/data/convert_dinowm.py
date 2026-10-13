"""Convert DINO-WM's Point Maze / Wall datasets (as used by NanoWM) into the two .h5 files tinyworlds uses.

Source: DINO-WM's OSF project (https://osf.io/bmw48/), `point_maze.zip` (0.72 GB, 28 GB unzipped) and `wall_single.zip`
(1.67 GB). One folder per env: `obses/episode_NNN.pth` (uint8 tensor [L_max, H, W, 3], one frame per env step),
`actions.pth` [N_ep, L_max, 2] float64 (zero padded), `states.pth`, `seq_lengths.pth` (true length per episode).
Point Maze: 2000 episodes x 100 steps, uint8 [100, 224, 224, 3]. Wall: 1920 episodes, float32 [51, 3, 224, 224] frames
in [0, 255] (not integer; rounded to uint8 here), no seq_lengths file -> NanoWM uses states.shape[1] = 50 steps.

Split (NanoWM, src/wm_datasets/world_model_dataset.py _split_trajectories_indices): no shipped val folder; the episodes
are shuffled with np.random.seed(42), the first 90% are train, the rest val (evaluation.nanowm.nanowm_val_order).
Action normalisation (src/configs/dataset/dino_wm/<env>.yaml): Point Maze z-scores the actions with the mean / std
(torch, unbiased, + 1e-6) of the valid steps of the TRAIN episodes; Wall does not normalise them.

Outputs (in --out-dir):
  <env>_frames.h5      the train episodes (ascending id), NanoWM's convention: frame_interval 5, the 5 env actions between
                       two kept frames concatenated into one 10-D action. Only phase-0 frames are kept (steps 0, 5, ...).
                         frames        uint8   [N, S, S, 3]  --size (default 256), resized like NanoWM builds its ground
                                                             truth: float bilinear stretch (align_corners False, no
                                                             antialias), round(x * 255); --resize area = cv2 INTER_AREA
                         actions       float32 [N, 10]       actions of steps 5k..5k+4 (normalised as NanoWM for this env;
                                                             zeros past the episode end)
                         episode_index int32   [N]           original DINO-WM episode id
                         step_index    int32   [N]           env step of the frame
  <env>_val_frames.h5  the val episodes in NanoWM's order, native resolution, every env step, for scripts/eval/eval_nanowm.py
                         frames        uint8   [N, H, W, 3]
                         actions       float32 [N, 2]        raw env actions, NOT normalised
                         episode_index, step_index           as above
  Both carry a seq_lengths dataset (file order) and attrs env, frame_interval, split_ratio, split_seed,
  normalize_action, val_traj_ids (NanoWM order), and action_mean / action_std (per env-action dim, only if normalised).

Usage (repo root, PYTHONPATH=$PWD):
    python scripts/data/convert_dinowm.py --env point_maze      # downloads + unzips into data/dinowm_raw/ if missing
    python scripts/data/convert_dinowm.py --env wall --size 128 --resize area --skip-val --out-dir /tmp/dinowm128
    python scripts/data/convert_dinowm.py --env wall --limit-episodes 20 --out-dir /tmp/dinowm   # smoke
"""

import argparse
import os
import time
import urllib.request
import zipfile
from multiprocessing import Pool

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from evaluation.nanowm import ENVS, FRAME_INTERVAL, nanowm_val_order

SOURCES = {  # env -> (OSF download url, folder inside the zip)
    'point_maze': ('https://osf.io/download/vr5gy/', 'point_maze'),
    'wall': ('https://osf.io/download/49rnx/', 'wall_single'),
}
SPLIT_SEED = 42


def ensure_raw(raw_dir, url, folder):
    """Download and unzip <folder>.zip next to raw_dir unless raw_dir/actions.pth already exists."""
    if os.path.exists(os.path.join(raw_dir, 'actions.pth')):
        return
    parent = os.path.dirname(os.path.abspath(raw_dir))
    os.makedirs(parent, exist_ok=True)
    zip_path = os.path.join(parent, f'{folder}.zip')
    if not os.path.exists(zip_path):
        print(f'downloading {url} -> {zip_path}', flush=True)
        urllib.request.urlretrieve(url, zip_path + '.part')
        os.replace(zip_path + '.part', zip_path)
    print(f'unzipping {zip_path}', flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(parent)  # creates <parent>/<folder>/
    extracted = os.path.join(parent, folder)
    if os.path.abspath(extracted) != os.path.abspath(raw_dir):
        os.replace(extracted, raw_dir)


def load_meta(raw_dir):
    """-> actions float32 [N_ep, L_max, 2], seq_lengths list[int], as DinoWorldModelDataSource reads them."""
    actions = torch.load(os.path.join(raw_dir, 'actions.pth'), weights_only=True).float()  # [N_ep, L_max, 2]
    sl_path = os.path.join(raw_dir, 'seq_lengths.pth')
    if os.path.exists(sl_path):
        seq_lengths = [int(x) for x in torch.as_tensor(torch.load(sl_path, weights_only=True))]
    else:  # their fallback (Wall ships no seq_lengths): every trajectory is states.shape[1] long (Wall: 50 of 51 frames)
        states = torch.load(os.path.join(raw_dir, 'states.pth'), weights_only=True)
        seq_lengths = [states.shape[1]] * len(states)
    return actions, seq_lengths


def load_episode(job):
    """-> uint8 [n_kept, S, S, 3] RGB; every `step`-th frame of the first `length`, resized to `size` (None = native)."""
    path, length, step, size, resize = job
    torch.set_num_threads(1)  # one episode per worker process
    video = torch.as_tensor(torch.load(path, weights_only=True))[:length:step]
    # Point Maze: uint8 [n, H, W, 3]; Wall: float32 [n, 3, H, W] in [0, 255], not integer. NanoWM reads both as
    # float / 255 and only permutes when the last dim is 1 or 3 (DinoWorldModelDataSource.load_visual_frames)
    x = (video.float() / 255.0).permute(0, 3, 1, 2) if video.shape[-1] in (1, 3) else video.float() / 255.0  # [n, 3, H, W]
    if size is not None and x.shape[-1] != size and resize == 'bilinear':
        # NanoWM (_load_slice, resize_mode 'stretch'): [0, 1] float, bilinear, align_corners False
        x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    frames = np.ascontiguousarray((x.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).numpy())  # [n, H, W, 3]
    if size is not None and frames.shape[1] != size:  # resize == 'area'
        frames = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])
    return frames


def write_split(out_path, raw_dir, ids, seq_lengths, actions, step, size, mean, std, workers, attrs, resize='bilinear'):
    """ids: episode ids in file order. step 5 -> chunked 10-D (normalised if mean is not None) actions; step 1 -> raw."""
    lengths = [seq_lengths[i] for i in ids]
    n_frames = sum((L + step - 1) // step for L in lengths)
    jobs = [(os.path.join(raw_dir, 'obses', f'episode_{i:03d}.pth'), L, step, size, resize) for i, L in zip(ids, lengths)]
    t0 = time.time()
    with h5py.File(out_path + '.part', 'w') as f, Pool(workers) as pool:
        frames_d = act_d = None
        ep_d = f.create_dataset('episode_index', (n_frames,), dtype=np.int32)
        st_d = f.create_dataset('step_index', (n_frames,), dtype=np.int32)
        pos = 0
        for k, frames in enumerate(pool.imap(load_episode, jobs, chunksize=2)):
            i, L, n = ids[k], lengths[k], len(frames)
            if frames_d is None:  # resolution from the data (native for val)
                res = frames.shape[1:]
                frames_d = f.create_dataset('frames', (n_frames, *res), dtype=np.uint8,
                                            chunks=(min(64, n_frames), *res), compression='lzf')
                act_d = f.create_dataset('actions', (n_frames, 2 * step), dtype=np.float32)
            steps = np.arange(0, L, step)
            assert len(steps) == n
            a = actions[i, :L].numpy()  # [L, 2]
            if step > 1:
                chunk = np.zeros((n * step, 2), dtype=np.float32)  # [n*step, 2]
                chunk[:L] = a if mean is None else (a - mean) / std
                a = chunk.reshape(n, step * 2)  # [n, 10]: actions of steps 5k..5k+4, flattened step-major as NanoWM
            frames_d[pos:pos + n] = frames
            act_d[pos:pos + n] = a
            ep_d[pos:pos + n] = i
            st_d[pos:pos + n] = steps
            pos += n
            if (k + 1) % 200 == 0 or k + 1 == len(ids):
                print(f'  {out_path}: {k + 1}/{len(ids)} episodes, {pos} frames ({time.time() - t0:.0f}s)', flush=True)
        assert pos == n_frames
        f.create_dataset('seq_lengths', data=np.array(lengths, dtype=np.int32))
        f.attrs['frame_interval'] = step
        for k, v in attrs.items():
            f.attrs[k] = v
    os.replace(out_path + '.part', out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', required=True, choices=sorted(SOURCES))
    p.add_argument('--raw-dir', help='default data/dinowm_raw/<point_maze|wall_single>')
    p.add_argument('--out-dir', default='data')
    p.add_argument('--size', type=int, default=256, help='training resolution')
    p.add_argument('--resize', choices=('bilinear', 'area'), default='bilinear',
                   help="bilinear = NanoWM's 256 ground truth exactly; area = INTER_AREA (as convert_pusht.py)")
    p.add_argument('--workers', type=int, default=min(8, os.cpu_count()))
    p.add_argument('--limit-episodes', type=int, help='convert only the first N train and first N val episodes (smoke test)')
    p.add_argument('--skip-train', action='store_true')
    p.add_argument('--skip-val', action='store_true')
    args = p.parse_args()

    url, folder = SOURCES[args.env]
    spec = ENVS[args.env]
    raw_dir = args.raw_dir or os.path.join('data', 'dinowm_raw', folder)
    ensure_raw(raw_dir, url, folder)
    os.makedirs(args.out_dir, exist_ok=True)
    actions, seq_lengths = load_meta(raw_dir)
    n_ep = len(actions)

    order = nanowm_val_order(n_ep, 0.0, SPLIT_SEED)  # all episodes, NanoWM's shuffle
    cut = int(n_ep * spec['split_ratio'])
    train_nanowm_order, val_ids = order[:cut], order[cut:]

    attrs = {'env': args.env, 'split_ratio': spec['split_ratio'], 'split_seed': SPLIT_SEED,
             'normalize_action': spec['normalize_action'], 'val_traj_ids': np.asarray(val_ids, dtype=np.int32),
             'source': f'DINO-WM {folder} ({url})'}
    mean = std = None
    if spec['normalize_action']:  # NanoWM _compute_action_stats: torch, train episodes in split order, std + 1e-6
        valid = torch.cat([actions[i, :seq_lengths[i]] for i in train_nanowm_order])  # [sum L, 2]
        mean = valid.mean(0).numpy()
        std = (valid.std(0) + 1e-6).numpy()
        attrs.update(action_mean=mean, action_std=std)
    print(f'{args.env}: {n_ep} episodes, {sum(seq_lengths)} env steps; train {len(train_nanowm_order)}, val {len(val_ids)}')
    print(f'action mean {mean}, std {std}')

    n = args.limit_episodes
    if not args.skip_val:
        write_split(os.path.join(args.out_dir, f'{args.env}_val_frames.h5'), raw_dir, [int(i) for i in val_ids[:n]],
                    seq_lengths, actions, step=1, size=None, mean=None, std=None, workers=args.workers, attrs=attrs)
    if not args.skip_train:
        train_ids = sorted(int(i) for i in train_nanowm_order)[:n]
        write_split(os.path.join(args.out_dir, f'{args.env}_frames.h5'), raw_dir, train_ids, seq_lengths, actions,
                    step=FRAME_INTERVAL, size=args.size, mean=mean, std=std, workers=args.workers,
                    attrs=dict(attrs, resize=args.resize), resize=args.resize)
    print('done')


if __name__ == '__main__':
    main()
