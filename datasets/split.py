"""Deterministic train/test split of a <game>_frames.h5 (scripts/data/split_dataset.py writes the files) and the
contiguous segments of a split file, so windows never straddle a cut."""

import json

import h5py
import numpy as np


def test_blocks(n_frames: int, n_blocks: int, block_fraction: float):
    """Return [(start, end), ...] in original indices: `n_blocks` blocks of floor(N * fraction) frames,
    centered on evenly spaced points over the video."""
    length = int(n_frames * block_fraction)
    blocks = []
    for k in range(n_blocks):
        center = int((k + 0.5) * n_frames / n_blocks)
        start = max(0, min(n_frames - length, center - length // 2))
        blocks.append((start, start + length))
    return blocks


def split_masks(n_frames: int, blocks, margin: int):
    """Boolean masks over original indices: which frames are test, which are train."""
    is_test = np.zeros(n_frames, dtype=bool)
    is_excluded = np.zeros(n_frames, dtype=bool)  # test frames plus the margins around them
    for start, end in blocks:
        is_test[start:end] = True
        is_excluded[max(0, start - margin):min(n_frames, end + margin)] = True
    return is_test, ~is_excluded


def write_subset(path: str, src: h5py.Dataset, indices: np.ndarray, attrs: dict, chunk: int = 2000, h5_chunk: int = 64):
    with h5py.File(path, 'w') as out:
        shape = (len(indices),) + src.shape[1:]
        # lzf like the original caches (about half the size on disk, so half the upload to the Modal volume);
        # whole-frame chunks of `h5_chunk` frames keep random single-frame reads (the eval script) cheap
        frames = out.create_dataset('frames', shape=shape, dtype=src.dtype, compression='lzf',
                                    chunks=(min(h5_chunk, len(indices)),) + src.shape[1:])
        # copy in contiguous runs so we read the source sequentially
        pos = 0
        run_start = 0
        while run_start < len(indices):
            run_end = run_start
            while run_end + 1 < len(indices) and indices[run_end + 1] == indices[run_end] + 1 and run_end + 1 - run_start < chunk:
                run_end += 1
            block = src[indices[run_start]:indices[run_end] + 1]
            frames[pos:pos + len(block)] = block
            pos += len(block)
            run_start = run_end + 1
        out.create_dataset('source_index', data=indices.astype(np.int64))
        for k, v in attrs.items():
            out.attrs[k] = v


def segments(h5_path, n_frames):
    """Contiguous [start, end) stretches of local indices: test_blocks_local for a test file, the cuts of
    split_dataset.py for a train file, else the whole file."""
    with h5py.File(h5_path, 'r') as f:
        attrs = dict(f.attrs)
    if 'test_blocks_local' in attrs:
        return [tuple(b) for b in json.loads(attrs['test_blocks_local'])]
    if attrs.get('split') == 'train':
        _, is_train = split_masks(int(attrs['source_n_frames']), json.loads(attrs['test_blocks_source']), int(attrs['margin']))
        src = np.flatnonzero(is_train)
        assert len(src) == n_frames, (len(src), n_frames)
        cuts = np.flatnonzero(np.diff(src) != 1) + 1
        bounds = [0, *cuts.tolist(), n_frames]
        return list(zip(bounds[:-1], bounds[1:]))
    return [(0, n_frames)]
