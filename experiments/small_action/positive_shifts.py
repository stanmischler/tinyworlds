"""The camera shifts to the InfoNCE positives on the YouTube rows of the 5 h shift table (five_hours.py), so InfoNCE can train on it.

train.py draws a pair (t, b = t + pair_gap[t]) and a positive frame b - 1 or b + 1, and reads the shift from t to it in row 0 or 2 of
the table. On our capture's rows (pair gap 4) these are the gaps 3 and 5, copied from data/zelda_train_shifts.npz. On the YouTube rows
(pair gap 2) they are the gaps 1 and 3, which the table left as NaN. This script measures them as shift.py does (best_shift, then
known_mse of the shifted frame) and writes them in place; every other value stays bit-identical, and the table as it was is kept beside
it. A shift is measured wherever both frames exist, so the last rows of a stretch hold one measured across its end: train.py never
draws them (trainable_starts). Run again on a filled table, it changes nothing.

    TINYWORLDS_GPU=H100 modal run --detach experiments/small_action/modal_small_action.py::positive_shifts
"""

import json
import os
import shutil
import time

import h5py
import numpy as np
import torch

from experiments.small_action.eval import GAP_ROW
from experiments.small_action.shift import best_shift, known_mse, shift_frames

MERGED_H5, SHIFTS = 'data/zelda_five_hours_frames.h5', 'data/zelda_five_hours_shifts.npz'
BEFORE, UNFINISHED = 'data/zelda_five_hours_shifts_before_positives.npz', 'data/zelda_five_hours_shifts_unfinished.npz'
POSITIVE_ROWS = (GAP_ROW - 1, GAP_ROW + 1)  # table rows of the positives b - 1 and b + 1 (train.py sample_batch: row delta + 1)
OUR_SOURCE = 0  # `source` of our capture's rows; the YouTube files follow
BATCH = 2048
PROGRESS_EVERY = 10  # batches between two progress lines
DEVICE = 'cuda'


def frames_between(pair_gap, row):
    # pair_gap [N], row: a table row -> [N] stored frames from t to the frame that row's shift points to: the pair's later frame at
    # GAP_ROW, one frame before or after it at the positives' rows
    return pair_gap.astype(np.int64) + (row - GAP_ROW)


def rows_to_measure(table, row):
    # -> the YouTube rows t whose frame at table row `row` exists
    n = len(table['source'])
    return np.flatnonzero((table['source'] != OUR_SOURCE) & (np.arange(n) + frames_between(table['pair_gap'], row) < n))


@torch.no_grad()
def measure_shifts(frames, earlier, later, batch, device):
    # frames uint8 [N, H, W, C]; earlier, later [R]: the rows of each pair -> shifts (dy, dx) [R, 2] float16, errors [R] float32
    shifts, errors = np.zeros((len(earlier), 2), np.float16), np.zeros(len(earlier), np.float32)
    to_float = lambda i: torch.from_numpy(frames[i]).to(device).permute(0, 3, 1, 2).float() / 255  # noqa: E731
    t0 = time.time()
    for i in range(0, len(earlier), batch):
        x, y = to_float(earlier[i:i + batch]), to_float(later[i:i + batch])  # [B, 3, H, W] in [0, 1]
        s = best_shift(x, y)
        shifts[i:i + batch] = s.cpu().numpy()
        errors[i:i + batch] = known_mse(shift_frames(x, s, (0, 1)), y, s).cpu().numpy()
        if (i // batch) % PROGRESS_EVERY == 0:
            print(f'  {min(i + batch, len(earlier))}/{len(earlier)} pairs in {(time.time() - t0) / 60:.1f} min', flush=True)
    return shifts, errors


def fill_positive_shifts(frames, table, batch, device):
    # -> the table's shifts [N, 3, 2] and errors [N, 3] with the positives' rows measured on the YouTube rows
    shifts, errors = table['shifts'].copy(), table['errors'].copy()
    for row in POSITIVE_ROWS:
        t = rows_to_measure(table, row)
        print(f'table row {row}: {len(t)} YouTube pairs', flush=True)
        shifts[t, row], errors[t, row] = measure_shifts(frames, t, t + frames_between(table['pair_gap'], row)[t], batch, device)
    return shifts, errors


def missing_positives(table):
    # -> how many shifts this script has to measure are still NaN
    return sum(int(np.isnan(table['shifts'][rows_to_measure(table, row), row]).any(1).sum()) for row in POSITIVE_ROWS)


def assert_rest_unchanged(table, shifts, errors):
    # the pair's row everywhere and our capture's three rows: the same bits as in the table
    ours = table['source'] == OUR_SOURCE
    for old, new in ((table['shifts'], shifts), (table['errors'], errors)):
        assert old.dtype == new.dtype and old.shape == new.shape
        assert old[:, GAP_ROW].tobytes() == new[:, GAP_ROW].tobytes(), 'the pairs\' shifts changed'
        assert old[ours].tobytes() == new[ours].tobytes(), 'our capture\'s rows changed'


def write_table(table, shifts, errors):
    # SHIFTS as it was -> BEFORE; the new table, every key kept, -> UNFINISHED, read back, then in SHIFTS' place
    shutil.copy(SHIFTS, BEFORE)
    np.savez(UNFINISHED, **{**{k: table[k] for k in table.files}, 'shifts': shifts, 'errors': errors})
    written = np.load(UNFINISHED)
    assert written.files == table.files
    assert_rest_unchanged(table, written['shifts'], written['errors'])
    os.replace(UNFINISHED, SHIFTS)


def print_summary(table, names):
    # per file and table row: frames between the two frames, median and p95 of |shift| (largest of |dy|, |dx|) on moving pairs
    # (>= 1 px), the share of still pairs, and the rows left without a shift
    for k, name in enumerate(names):
        here = table['source'] == k
        cells = []
        for row in range(table['shifts'].shape[1]):
            size = np.abs(table['shifts'][here, row].astype(np.float32)).max(1)  # [rows of the file], NaN where not measured
            measured = size[~np.isnan(size)]
            moving = measured[measured >= 1]
            p50, p95 = np.quantile(moving, (.5, .95))
            gap = frames_between(table['pair_gap'][here], row)[0]
            cells.append(f'gap {gap}: {p50:.2f} / {p95:.2f} px, still {1 - len(moving) / len(measured):.3f}, NaN {len(size) - len(measured)}')
        print(f'{os.path.basename(name)} ({here.sum()} rows) | ' + ' | '.join(cells), flush=True)


def main():
    table = np.load(SHIFTS)
    with h5py.File(MERGED_H5, 'r') as f:
        names = json.loads(f.attrs['sources'])
        if missing_positives(table) == 0:
            print(f'{SHIFTS} already holds the positives\' shifts on the YouTube rows: nothing to do', flush=True)
            return print_summary(table, names)
        frames = f['frames'][:]  # uint8 [N, H, W, C]
    assert len(frames) == len(table['shifts'])
    shifts, errors = fill_positive_shifts(frames, table, BATCH, DEVICE)
    assert_rest_unchanged(table, shifts, errors)
    write_table(table, shifts, errors)
    table = np.load(SHIFTS)
    print(f'{SHIFTS}: {missing_positives(table)} positives\' shifts left to measure; the table before is {BEFORE}', flush=True)
    print('median / p95 of |shift| on moving pairs, share of still pairs, rows without a shift:', flush=True)
    print_summary(table, names)


if __name__ == '__main__':
    main()
