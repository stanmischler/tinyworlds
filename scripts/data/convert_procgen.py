"""Convert one Procgen game of the HF dataset EpicPinkPenguin/procgen into tinyworlds episode .h5 files.

Source: <game>/train-*.parquet (9M steps) and <game>/test-00000-of-00001.parquet (1M steps), 64x64 RGB frames played by a PPO
agent (25M steps, easy mode). One row = one env step: observation (PNG bytes), action (one of Procgen's 15 buttons),
reward, terminated, truncated. Rows are consecutive steps; episodes follow each other, the row with terminated or
truncated set is the last of its episode. Row groups hold 100 rows.

Outputs (in --out-dir):
  <game>_frames.h5        training set: --train-frames steps of the first train file, taken as --train-chunks contiguous
                          runs evenly spaced over the file (the head of a file holds the shortest episodes, so one run from
                          the start would be biased)
  <game>_test_frames.h5   held-out set: --test-frames steps of the test file, as row groups evenly spaced over it

Both files:
  frames          uint8 [N, 64, 64, 3]
  episode_index   int32 [N]   a new id after every episode end and at the start of every contiguous run
  step_index      int32 [N]   row count since the episode id started (a run may start mid-episode)
  buttons         int8  [N]   the true button pressed at this frame (leads to the next row); see BUTTONS
  reward          float32 [N]
  done            bool  [N]   terminated or truncated
  attrs: source, game, split, buttons (JSON list of the 15 names)
The test file also has attrs['test_blocks_local']: JSON [[start, end), ...] of its contiguous same-episode row ranges, the
blocks the held-out windows are cut from (evaluation/windows.py test_windows).

The buttons are deliberately not stored as `actions`: EpisodeH5Dataset serves `actions` to the models, and the true
buttons must only reach the action checks.

Usage (repo root, PYTHONPATH=$PWD):
    python scripts/data/convert_procgen.py --game bigfish                       # downloads the train file (8.7 GB) first
    python scripts/data/convert_procgen.py --game bigfish --remote --train-frames 2000 --test-frames 1000 --out-dir /tmp/x   # smoke
"""

import argparse
import io
import json
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
from PIL import Image

HF_REPO = 'EpicPinkPenguin/procgen'
# Procgen's 15 buttons, in the env's action order (procgen/env.py get_combos)
BUTTONS = ['left+down', 'left', 'left+up', 'down', 'noop', 'up', 'right+down', 'right', 'right+up', 'D', 'A', 'W', 'S', 'Q', 'E']
RES = 64


def decode_png(png_bytes):
    return np.asarray(Image.open(io.BytesIO(png_bytes)).convert('RGB'))  # [64, 64, 3] uint8


def open_parquet(filename, raw_dir, remote):
    """ParquetFile of one file of the HF repo: read over HTTP range requests (remote, small samples) or downloaded first."""
    import pyarrow.parquet as pq
    if remote:
        from huggingface_hub import HfFileSystem
        return pq.ParquetFile(HfFileSystem().open(f'datasets/{HF_REPO}/{filename}', 'rb', block_size=1 << 20))
    from huggingface_hub import hf_hub_download
    return pq.ParquetFile(hf_hub_download(HF_REPO, filename, repo_type='dataset', local_dir=raw_dir))


def spaced_runs(n_groups, groups_wanted, n_runs):
    """n_runs contiguous runs of row groups, evenly spaced over the file and never overlapping -> [(first, last + 1), ...]."""
    if groups_wanted >= n_groups:
        return [(0, n_groups)]
    n_runs = max(1, min(n_runs, groups_wanted))
    length = groups_wanted // n_runs
    firsts = np.linspace(0, n_groups - length, n_runs).round().astype(int)  # spacing >= length, as groups_wanted < n_groups
    return [(int(f), int(f) + length) for f in firsts]


def read_runs(parquet, runs, pool, batch_groups=20):
    """Rows of the row-group runs, in file order -> dict of arrays [N, ...] plus `run_start` [N] (True on a run's first row)."""
    n = sum(parquet.metadata.row_group(g).num_rows for first, last in runs for g in range(first, last))
    out = {'frames': np.empty((n, RES, RES, 3), np.uint8), 'buttons': np.empty(n, np.int8), 'reward': np.empty(n, np.float32),
           'done': np.zeros(n, bool), 'run_start': np.zeros(n, bool)}  # preallocated: a list of chunks would double the peak RAM
    row, t0 = 0, time.time()
    for r, (first, last) in enumerate(runs):
        out['run_start'][row] = True
        for g in range(first, last, batch_groups):
            table = parquet.read_row_groups(list(range(g, min(g + batch_groups, last))))
            rows = slice(row, row + table.num_rows)
            pngs = [o['bytes'] for o in table.column('observation').to_pylist()]
            out['frames'][rows] = np.stack(pool.map(decode_png, pngs, chunksize=64))
            out['buttons'][rows] = table.column('action').to_pylist()
            out['reward'][rows] = table.column('reward').to_pylist()
            out['done'][rows] = np.asarray(table.column('terminated').to_pylist()) | np.asarray(table.column('truncated').to_pylist())
            row += table.num_rows
        print(f'  run {r + 1}/{len(runs)} (row groups {first}..{last - 1}) read, {time.time() - t0:.0f}s', flush=True)
    return out


def episode_ids(done, run_start):
    """done, run_start [N] bool -> (episode_index [N], step_index [N]): a new id on the row after a done and on a run's first row."""
    new = run_start.copy()
    new[1:] |= done[:-1]
    new[0] = True
    episode = np.cumsum(new) - 1                                   # [N]
    first_row = np.maximum.accumulate(np.where(new, np.arange(len(new)), 0))  # [N] row where the row's episode id started
    return episode.astype(np.int32), (np.arange(len(new)) - first_row).astype(np.int32)


def blocks_of(episode):
    """episode_index [N] (non-decreasing) -> [[start, end), ...] of its constant runs."""
    edges = np.flatnonzero(np.diff(episode)) + 1
    bounds = np.concatenate([[0], edges, [len(episode)]])
    return [[int(a), int(b)] for a, b in zip(bounds[:-1], bounds[1:])]


def write_h5(path, rows, game, split):
    episode, step = episode_ids(rows['done'], rows['run_start'])
    with h5py.File(path, 'w') as f:
        # test file: one chunk per row group, as the held-out windows are read one by one (a 1024-frame chunk would be
        # decompressed for each of them)
        f.create_dataset('frames', data=rows['frames'], chunks=(100 if split == 'test' else 1024, RES, RES, 3), compression='lzf')
        f.create_dataset('episode_index', data=episode)
        f.create_dataset('step_index', data=step)
        f.create_dataset('buttons', data=rows['buttons'])
        f.create_dataset('reward', data=rows['reward'])
        f.create_dataset('done', data=rows['done'])
        f.attrs['source'] = f'HF {HF_REPO}, {game}/{split}'
        f.attrs['game'], f.attrs['split'] = game, split
        f.attrs['buttons'] = json.dumps(BUTTONS)
        if split == 'test':
            f.attrs['test_blocks_local'] = json.dumps(blocks_of(episode))
    print(f'{path}: {len(episode)} frames, {int(episode[-1]) + 1} episode ids, {int(rows["done"].sum())} episode ends')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game', default='bigfish')
    p.add_argument('--train-frames', type=int, default=1_000_000)
    p.add_argument('--train-chunks', type=int, default=10, help='contiguous runs the training frames are taken as')
    p.add_argument('--test-frames', type=int, default=50_000)
    p.add_argument('--out-dir', default='data')
    p.add_argument('--raw-dir', default='data/procgen_raw', help='where the train parquet file is downloaded')
    p.add_argument('--remote', action='store_true', help='read the train file over HTTP instead of downloading it (small samples)')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    from huggingface_hub import HfApi
    files = sorted(f for f in HfApi().list_repo_files(HF_REPO, repo_type='dataset') if f.startswith(f'{args.game}/'))
    train_file = next(f for f in files if '/train-' in f)
    test_file = next(f for f in files if '/test-' in f)
    os.makedirs(args.out_dir, exist_ok=True)

    with Pool(args.workers) as pool:
        for split, filename, n_frames, n_runs, remote in [
            ('train', train_file, args.train_frames, args.train_chunks, args.remote),
            ('test', test_file, args.test_frames, None, True),  # single row groups spread over the file: range reads
        ]:
            parquet = open_parquet(filename, args.raw_dir, remote)
            rows_per_group = parquet.metadata.row_group(0).num_rows
            groups_wanted = max(1, n_frames // rows_per_group)
            runs = spaced_runs(parquet.metadata.num_row_groups, groups_wanted, n_runs or groups_wanted)
            print(f'{split}: {filename}, {parquet.metadata.num_rows} rows, taking {len(runs)} runs of {runs[0][1] - runs[0][0]} row groups')
            rows = read_runs(parquet, runs, pool)
            name = f'{args.game}_frames.h5' if split == 'train' else f'{args.game}_test_frames.h5'
            write_h5(os.path.join(args.out_dir, name), rows, args.game, split)


if __name__ == '__main__':
    main()
