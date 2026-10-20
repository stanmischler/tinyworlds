"""Frame pairs (frame t, frame t+4) for the visualisations, as uint8 [N, H, W, C] arrays plus per-pair metadata:
  - labelled pairs : the frozen judge set of scripts/eval/lam_eval.py (eval_results/lam_eval/zelda/set): the last
                     transition of each pool window, kept when both judges agree ('main' labels, e.g. WALK_L, STILL)
  - test windows   : the 890 held-out windows of eval_next_frame / eval_como_pred (context 3, frame_skip 4, stride 8);
                     the pair is (last context frame, target)
  - train pairs    : a seeded sample of train rows i with their precomputed z (data/zelda_train_como_z.npy, pair i, i+4),
                     keeping only pairs that do not straddle a cut between source segments"""

import json
import os

import h5py
import numpy as np

from evaluation.windows import test_windows

GAP = 4  # stored frames between t and t+4 (Zelda 15 fps, frame_skip 4)


def read_rows(dset, rows):
    # h5 fancy indexing needs sorted unique rows: read those, then put them back in the asked order -> [N, H, W, C]
    uniq, inv = np.unique(np.asarray(rows), return_inverse=True)
    return dset[uniq.tolist()][inv]


def labelled_pairs(set_dir, h5_path):
    pool = json.load(open(os.path.join(set_dir, 'pool.json')))
    labels = json.load(open(os.path.join(set_dir, 'labels.json')))['main']  # {window id: label}
    wins = [w for w in pool['windows'] if str(w['id']) in labels]
    with h5py.File(h5_path, 'r') as f:
        frame_t = read_rows(f['frames'], [w['frame_a'] for w in wins])
        frame_next = read_rows(f['frames'], [w['frame_b'] for w in wins])
    return {'frame_t': frame_t, 'frame_next': frame_next, 'label': np.array([labels[str(w['id'])] for w in wins]),
            'block': np.array([w['block'] for w in wins]), 'row': np.array([w['frame_a'] for w in wins]),
            'id': np.array([w['id'] for w in wins])}


def window_pairs(h5_path, n=None, context=3, frame_skip=4, sample_stride=8):
    wins = test_windows(h5_path, context, frame_skip, sample_stride)
    if n is not None:  # n windows evenly spaced over the test set, so they cover every test block
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, n).round().astype(int)]
    rows = np.array([s + (context - 1) * frame_skip for _, s in wins])  # last context frame
    with h5py.File(h5_path, 'r') as f:
        frame_t = read_rows(f['frames'], rows)
        frame_next = read_rows(f['frames'], rows + frame_skip)
    return {'frame_t': frame_t, 'frame_next': frame_next, 'block': np.array([b for b, _ in wins]), 'row': rows}


def valid_train_rows(source_index, gap=GAP):
    # rows i whose pair (i, i+gap) is gap frames apart in the source video (the train h5 concatenates segments)
    i = np.arange(len(source_index) - gap)
    return i[source_index[i + gap] - source_index[i] == gap]


def train_pairs(h5_path, z_path, n, seed=0):
    """All valid train z (for fitting the PCA) and a seeded sample of n pairs with their frames (for the grey cloud)."""
    z_all = np.load(z_path)  # [N_rows, A], row i = z(i, i+4)
    with h5py.File(h5_path, 'r') as f:
        valid = valid_train_rows(f['source_index'][:])
        rows = np.sort(np.random.default_rng(seed).choice(valid, n, replace=False))
        frame_t, frame_next = read_rows(f['frames'], rows), read_rows(f['frames'], rows + GAP)
    return {'z_all': z_all[valid], 'z': z_all[rows], 'row': rows, 'frame_t': frame_t, 'frame_next': frame_next}


def other_block_partner(blocks, seed):
    """For each pair i, a random partner j from another test block (another stretch of play, so usually another room)."""
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(np.flatnonzero(blocks != b)) for b in blocks])
