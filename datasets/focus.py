"""The focus pool: pairs of frames with a still camera whose change is one compact patch, most often Link walking, and a batch
sampler that draws a fixed share of every batch from it. One definition for the small action model's trainer
(experiments/small_action/train.py, still_fraction) and the dynamics trainer (scripts/dynamics/train_dynamics.py, still_fraction).
The tables come from experiments/small_action/five_hours.py.
"""

import numpy as np
import torch
from torch.utils.data import Sampler


def focus_pairs(shift, motion=None, min_px=0, max_px=0, min_share=0.0, drop_dialog=False):
    """shift [N, 2]: measured camera shift of each row's pair, px. motion: the motion table (moving_px [N] changed pixels after the
    shift, peak_share [N] their share inside the best 32 px window, dialog [N]); None = the shift alone decides.
    -> [N] bool: camera still (under 1 px), changed pixels in [min_px, max_px], peak share >= min_share, and with drop_dialog no
    dialog box or menu on screen."""
    focus = np.abs(shift).max(1) < 1
    if motion is not None:
        focus &= (motion['moving_px'] >= min_px) & (motion['moving_px'] <= max_px) & (motion['peak_share'] >= min_share)
        if drop_dialog:
            assert 'dialog' in motion.files, 'the motion table holds no dialog flags: rebuild it with experiments/small_action/five_hours.py'
            focus &= ~motion['dialog']
    return focus


def focus_window_starts(data, focus):
    """data: a MultiVideoH5Dataset. focus [load_start_index + N] bool, one per table row (the base .h5 from row 0, then the extra
    files, as action_file). -> first rows [K] in data.data of the windows whose last pair, the predicted frame, is in focus. A
    window of an extra file is there frame_skip times, as a base window is one of the frame_skip phases of its slot, so a uniform
    draw keeps every second of video equally likely, as the default loader does."""
    last_pair = lambda starts, skip: data.load_start_index + starts + (data.num_frames - 2) * skip  # noqa: E731
    base = np.array(sorted(data.base_starts), dtype=np.int64)
    base = base[focus[last_pair(base, data.frame_skip)]]
    extra = data.extra_starts[focus[last_pair(data.extra_starts, data.extra_skip)]]
    return np.concatenate([base, np.repeat(extra, data.frame_skip)])


class FocusBatchSampler(Sampler):
    """Batches of window indices [batch_size] of a MultiVideoH5Dataset: round(fraction * batch_size) of them from data.focus_starts
    (index len(data) + k), the others uniform over all windows as the default loader draws them (so they hold the pool's natural
    share), in a random order. Draws are with replacement; an epoch is len(data) // batch_size batches."""

    def __init__(self, data, batch_size, fraction, generator=None):
        self.data, self.batch_size, self.n_focus, self.generator = data, batch_size, round(fraction * batch_size), generator

    def __len__(self):
        return len(self.data) // self.batch_size

    def __iter__(self):
        n, g = len(self.data), self.generator
        for _ in range(len(self)):
            focus = n + torch.randint(len(self.data.focus_starts), (self.n_focus,), generator=g)
            others = torch.randint(n, (self.batch_size - self.n_focus,), generator=g)
            yield torch.cat([focus, others])[torch.randperm(self.batch_size, generator=g)].tolist()


def focus_batch_sampler(data, batch_size, fraction, shift_table, motion_table, min_px, max_px, min_share, drop_dialog, generator=None):
    """The dynamics trainer's still_fraction: sets data.focus_starts from the two tables and returns the batch sampler."""
    assert hasattr(data, 'extra_starts'), 'still_fraction: multi-file datasets only (MultiVideoH5Dataset)'
    assert shift_table and motion_table, 'still_fraction needs still_shift_table and still_motion_table'
    table, motion = np.load(shift_table), np.load(motion_table)
    first, n = data.load_start_index, len(data.data)
    assert len(table['shifts']) == len(motion['moving_px']) == first + n, (len(table['shifts']), len(motion['moving_px']), first + n)
    gaps = table['pair_gap'][first:]  # the tables' pairs must be the loader's: rows (r, r + the file's skip)
    assert (gaps[:data.base_rows] == data.frame_skip).all() and (gaps[data.base_rows:] == data.extra_skip).all(), \
        f'{shift_table}: pairs at other skips than the loader ({data.frame_skip} base, {data.extra_skip} extra files)'
    shift = table['shifts'][:, int(np.flatnonzero(table['gaps'] == data.frame_skip)[0])]  # [N, 2], the row that holds each pair's own shift
    focus = focus_pairs(shift, motion, min_px, max_px, min_share, drop_dialog)
    data.focus_starts = focus_window_starts(data, focus)
    n_base = int((data.focus_starts < data.base_rows).sum())
    assert len(data.focus_starts) > 0, 'empty focus pool'
    print(f'batch: {fraction} from {n_base + (len(data.focus_starts) - n_base) // data.frame_skip} focus windows '
          f'({n_base} of the base file, {n_base / len(data.focus_starts):.4f} of the focus draws), the rest from all {len(data)} windows')
    return FocusBatchSampler(data, batch_size, fraction, generator)
