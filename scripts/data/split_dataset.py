"""Split a `<game>_frames.h5` cache into deterministic train/test files.

The test set is `n_blocks` equal blocks of contiguous frames, evenly spaced over the whole video so
every zone is represented. A margin of `margin` frames on each side of every block is dropped from
the train file, so no training window (num_frames * frame_skip <= 16 frames) can touch a test frame.
Everything is a fixed function of the frame count: no randomness, same split every time.

Outputs, next to the input (the input is left untouched):
    <game>_train_frames.h5   frames [N_train, H, W, C] uint8, source_index [N_train] int64
    <game>_test_frames.h5    frames [N_test,  H, W, C] uint8, source_index [N_test]  int64
`source_index[i]` is the row of frame i in the original file, so any frame traces back to its
position in the video. Both files carry the split parameters and the test-block bounds (in original
indices) as attributes; the test file also stores each block's bounds in its own local indices.

Usage (from the repo root):
    python scripts/data/split_dataset.py data/sonic_frames.h5
    python scripts/data/split_dataset.py data/sonic_frames.h5 --n-blocks 10 --block-fraction 0.01 --margin 16
"""

import argparse
import json
import os

import h5py
import numpy as np

from datasets.split import test_blocks, split_masks, write_subset



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='path to <game>_frames.h5')
    parser.add_argument('--n-blocks', type=int, default=10)
    parser.add_argument('--block-fraction', type=float, default=0.01, help='fraction of all frames per test block')
    parser.add_argument('--margin', type=int, default=16, help='frames dropped from train on each side of a test block')
    args = parser.parse_args()

    stem = args.input[:-len('_frames.h5')] if args.input.endswith('_frames.h5') else os.path.splitext(args.input)[0]
    train_path, test_path = f'{stem}_train_frames.h5', f'{stem}_test_frames.h5'

    with h5py.File(args.input, 'r') as f:
        src = f['frames']
        n = src.shape[0]
        blocks = test_blocks(n, args.n_blocks, args.block_fraction)
        is_test, is_train = split_masks(n, blocks, args.margin)
        test_idx, train_idx = np.flatnonzero(is_test), np.flatnonzero(is_train)

        common = {
            'source_file': os.path.basename(args.input),
            'source_n_frames': n,
            'n_blocks': args.n_blocks,
            'block_fraction': args.block_fraction,
            'margin': args.margin,
            'test_blocks_source': json.dumps(blocks),  # [[start, end), ...] in original indices
        }
        # each test block's bounds inside the test file itself
        local = []
        pos = 0
        for start, end in blocks:
            local.append((pos, pos + (end - start)))
            pos += end - start
        write_subset(test_path, src, test_idx, {**common, 'split': 'test', 'test_blocks_local': json.dumps(local)})
        write_subset(train_path, src, train_idx, {**common, 'split': 'train'})

    print(f'{args.input}: {n} frames')
    print(f'  test  -> {test_path}: {len(test_idx)} frames in {len(blocks)} blocks of {blocks[0][1] - blocks[0][0]}: {blocks}')
    print(f'  train -> {train_path}: {len(train_idx)} frames ({n - len(train_idx) - len(test_idx)} dropped as margins)')


if __name__ == '__main__':
    main()
