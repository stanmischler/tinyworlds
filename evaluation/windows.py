"""Deterministic held-out windows of a test .h5 (test_blocks_local attr, scripts/data/split_dataset.py) and the
uint8 <-> model-range conversions every eval shares."""

import json

import h5py
import numpy as np
import torch


def test_windows(h5_path, context, frame_skip, sample_stride):
    """Deterministic list of (block_id, local_start) windows; the target is local_start + context*frame_skip."""
    with h5py.File(h5_path, 'r') as f:
        blocks = json.loads(f.attrs['test_blocks_local'])
    span = context * frame_skip  # index offset of the target from the window start
    windows = []
    for block_id, (start, end) in enumerate(blocks):
        for s in range(start, end - span, sample_stride):
            windows.append((block_id, s))
    return windows


def load_window_batch(frames_dset, windows, context, frame_skip):
    # -> uint8 [B, T=context+1, H, W, C]
    idx = np.array([[s + k * frame_skip for k in range(context + 1)] for _, s in windows])
    out = np.stack([frames_dset[list(row)] for row in idx])
    return out


def load_history_batch(frames_dset, windows, context, frame_skip, history, h5_path):
    # load_window_batch with `history` extra frames before each window (temporal-tokenizer CoMo, STA-43), indices clamped
    # to the window's test block start (= scripts/actions/tok_features.py rule) -> uint8 [B, T=history+context+1, H, W, C]
    if not history:
        return load_window_batch(frames_dset, windows, context, frame_skip)
    with h5py.File(h5_path, 'r') as f:
        blocks = json.loads(f.attrs['test_blocks_local'])
    idx = np.array([[max(s + k * frame_skip, blocks[b][0]) for k in range(-history, context + 1)] for b, s in windows])
    return np.stack([frames_dset[sorted(set(row))][np.searchsorted(sorted(set(row)), row)] for row in idx])


def to_model_range(frames_u8, device):
    # uint8 [B, T, H, W, C] -> float [-1, 1] [B, T, C, H, W], same as the training transform
    x = torch.from_numpy(frames_u8).to(device).permute(0, 1, 4, 2, 3).float() / 255.0
    return x * 2 - 1


def to_unit(x):
    return ((x + 1) / 2).clamp(0, 1)
