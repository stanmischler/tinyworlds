"""The 5 h capture for the small action model: our train file followed by the 10 YouTube runs, in one .h5 with its shift and motion tables.

The pair must span the same game motion as the held-out pairs. On our files (60 fps stored) a pair is rows (t, t + 4), and a scroll moves
8.75 px. On the YouTube files (15 fps stored) the same 1/15 s moves only 3.5 to 5.25 px, in steps of 1.75 px; rows (t, t + 2) move 8.75 px
(7 px on two slower files), so a pair there is rows (t, t + 2). Row t of the tables holds its pair's gap (`pair_gap`), its file (`source`)
and how many times it enters the draw (`row_weight`: 1 on our rows, 4 on YouTube rows, so every second of video is equally likely, as the
ZELDA_MORE_TRAIN loader draws). Our rows keep their shifts from data/zelda_train_shifts.npz; on YouTube rows only the pair's shift is
measured, so these tables serve runs without InfoNCE only. Then a report: per file, errors, shift sizes, the motion filter, and sheets.

    python experiments/small_action/five_hours.py --stage merge
    python experiments/small_action/five_hours.py --stage tables --out results/small_action_images/five_hours_data
"""

import argparse
import glob
import json
import os
import shutil
import tempfile
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from datasets.split import segments
from experiments.small_action.eval import GAP_ROW
from experiments.small_action.shift import GAPS, SEARCH, best_shift, known_mse, shift_frames

OUR_H5, OUR_SHIFTS = 'data/zelda_train_frames.h5', 'data/zelda_train_shifts.npz'
EXTRA_H5 = 'data/zelda_youtube/*.h5'
MERGED_H5, SHIFTS, MOTION = 'data/zelda_five_hours_frames.h5', 'data/zelda_five_hours_shifts.npz', 'data/zelda_five_hours_motion.npz'
OUR_GAP, EXTRA_GAP = 4, 2  # stored frames of a pair, the same scroll in px in both
EXTRA_WEIGHT = 4  # one YouTube row is as much video as 4 of our rows
START_INDEX, DROP_WORST = 1000, 0.05  # configs/experiments/small_action/base.yaml
STILL_MIN_PX, STILL_MAX_PX, STILL_MIN_SHARE = 80, 600, 0.7  # the still sampler's bounds, configs/experiments/small_action/late_weight_focus.yaml
CHANGE, WINDOW = 0.1, 32  # a pixel changed: max over channels of |aligned t - t+n| > CHANGE (pixels in [0, 1]); window side in px
DEVICE = 'cuda'


@torch.no_grad()
def motion_table(frames, s, gap, batch=1024):
    # frames uint8 [N, H, W, C], s [N, 2], gap [N]: stored frames of each row's pair -> per pair with |s| < 1 px, changed pixels after
    # the shift [N] and the share of them inside the best WINDOW x WINDOW window [N] (0 elsewhere). Pixels only, no Link position:
    # a compact patch of change on a still screen is most often Link walking (the still sampler of train.py)
    N = len(frames)
    moving, peak = np.zeros(N, np.int32), np.zeros(N, np.float32)
    rows = np.flatnonzero((np.abs(s).max(1) < 1) & (np.arange(N) + gap < N))
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        x_a, x_b = (torch.from_numpy(frames[k]).to(DEVICE).permute(0, 3, 1, 2).float() / 255 for k in (r, r + gap[r]))  # [B, 3, H, W] x 2
        moved = ((shift_frames(x_a, torch.from_numpy(s[r]).float().to(DEVICE), (0, 1)) - x_b).abs().amax(1, keepdim=True) > CHANGE).float()  # [B, 1, H, W]
        count = moved.sum((1, 2, 3))  # [B]
        best = F.avg_pool2d(moved, WINDOW, stride=1).flatten(1).amax(1) * WINDOW ** 2  # [B], changed pixels in the best window
        moving[r], peak[r] = count.cpu().numpy(), (best / count.clamp(min=1)).cpu().numpy()
    return moving, peak


def merge():
    # -> MERGED_H5: frames [N, H, W, C], segment [N] (no pair crosses one), source [N]; the file names in attrs
    extras = sorted(glob.glob(EXTRA_H5))
    frames, runs, sources, offset = [], [], [], 0
    for k, path in enumerate([OUR_H5, *extras]):
        with h5py.File(path, 'r') as f:
            frames.append(f['frames'][:])
        run = np.zeros(len(frames[-1]), np.int64)
        for i, (lo, hi) in enumerate(segments(path, len(run))):
            run[lo:hi] = i
        runs.append(run + offset)
        sources.append(np.full(len(run), k, np.uint8))
        offset = runs[-1].max() + 1
        print(f'{path}: {len(run)} frames, {run.max() + 1} stretches', flush=True)
    frames = np.concatenate(frames)
    with tempfile.TemporaryDirectory() as tmp:  # written locally first: a killed job leaves nothing half made on the volume
        local = f'{tmp}/merged.h5'
        with h5py.File(local, 'w') as out:
            out.create_dataset('frames', data=frames, compression='lzf', chunks=(64, *frames.shape[1:]))
            out.create_dataset('segment', data=np.concatenate(runs))
            out.create_dataset('source', data=np.concatenate(sources))
            out.attrs['sources'] = json.dumps([OUR_H5, *extras])
        shutil.copy(local, MERGED_H5)
    print(f'{MERGED_H5}: {len(frames)} frames, {offset} stretches', flush=True)


def measure_extra_shifts(frames, rows, batch):
    # the pair (r, r + EXTRA_GAP) for each r in rows: shifts [R, 2], errors [R]
    shifts, errors = np.zeros((len(rows), 2), np.float16), np.zeros(len(rows), np.float32)
    to_float = lambda i: torch.from_numpy(frames[i]).to(DEVICE).permute(0, 3, 1, 2).float() / 255  # noqa: E731
    t0 = time.time()
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        x, y = to_float(r), to_float(r + EXTRA_GAP)
        s = best_shift(x, y)
        shifts[i:i + batch] = s.cpu().numpy()
        errors[i:i + batch] = known_mse(shift_frames(x, s, (0, 1)), y, s).cpu().numpy()
        if (i // batch) % 10 == 0:
            print(f'shifts: {i + len(r)}/{len(rows)} in {(time.time() - t0) / 60:.1f} min', flush=True)
    return shifts, errors


def tables(batch):
    # -> SHIFTS, MOTION; both also return what the report reads
    with h5py.File(MERGED_H5, 'r') as f:
        frames, source = f['frames'][:], f['source'][:]
    N, ours = len(frames), source == 0
    pair_gap = np.where(ours, OUR_GAP, EXTRA_GAP).astype(np.uint8)
    if not os.path.exists(SHIFTS):
        table = np.load(OUR_SHIFTS)
        assert tuple(table['gaps']) == GAPS and len(table['shifts']) == ours.sum()
        shifts, errors = np.full((N, len(GAPS), 2), np.nan, np.float16), np.full((N, len(GAPS)), np.nan, np.float32)
        shifts[ours], errors[ours] = table['shifts'], table['errors']
        rows = np.flatnonzero(~ours & (np.arange(N) + EXTRA_GAP < N))
        shifts[rows, GAP_ROW], errors[rows, GAP_ROW] = measure_extra_shifts(frames, rows, batch)
        np.savez(SHIFTS, shifts=shifts, errors=errors, gaps=np.array(GAPS), pair_gap=pair_gap, source=source, row_weight=np.where(ours, 1, EXTRA_WEIGHT).astype(np.uint8))
    table = np.load(SHIFTS)
    s = np.nan_to_num(table['shifts'][:, GAP_ROW].astype(np.float32))
    moving, peak = motion_table(frames, s, pair_gap.astype(np.int64))
    np.savez_compressed(MOTION, moving_px=moving, peak_share=peak)
    return frames, table, moving, peak


def kept_rows(table, n):
    # every usable start per file, and those kept by DROP_WORST inside that file -> (usable [N] bool, kept [N] bool, limits {file: limit})
    with h5py.File(MERGED_H5, 'r') as f:
        run = f['segment'][:]
    usable = np.zeros(n, bool)
    cuts = [0, *(np.flatnonzero(np.diff(run) != 0) + 1).tolist(), n]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        usable[max(lo, START_INDEX):max(hi - (OUR_GAP + 1), 0)] = True
    kept, limits = np.zeros(n, bool), {}
    errors, source = table['errors'][:, GAP_ROW], table['source']
    for k in np.unique(source):
        rows = usable & (source == k)
        limits[int(k)] = float(np.quantile(errors[rows], 1 - DROP_WORST))
        kept |= rows & (errors <= limits[int(k)])
    return usable, kept, limits


def report(frames, table, moving, peak, out):
    os.makedirs(out, exist_ok=True)
    with h5py.File(MERGED_H5, 'r') as f:
        names, source = json.loads(f.attrs['sources']), f['source'][:]
    usable, kept, limits = kept_rows(table, len(frames))
    s, errors = np.nan_to_num(table['shifts'][:, GAP_ROW].astype(np.float32)), table['errors'][:, GAP_ROW]
    size = np.abs(s).max(1)
    gap = table['pair_gap'].astype(np.int64)
    rng = np.random.default_rng(0)
    rows_out = []
    for k, name in enumerate(names):
        here, on_kept = usable & (source == k), kept & (source == k)
        still, moved = on_kept & (size < 1), on_kept & (size >= 1)
        focus = still & (moving >= STILL_MIN_PX) & (moving <= STILL_MAX_PX) & (peak >= STILL_MIN_SHARE)
        row = {'file': os.path.basename(name), 'seconds': round(int((source == k).sum()) / (60 if k == 0 else 15)), 'usable': int(here.sum()), 'kept': int(on_kept.sum()),
               'error_p50': float(np.quantile(errors[here], .5)), 'error_p95_limit': limits[k], 'error_p99': float(np.quantile(errors[here], .99)), 'still': round(float(still.sum() / on_kept.sum()), 3),
               'exact_repeat': round(float((errors[here] < 1e-6).mean()), 3), 'moving_shift_px_p50_p95': [float(np.quantile(size[moved], q)) for q in (.5, .95)], 'at_search_edge': round(float((size[here] >= SEARCH - 1).mean()), 4),
               'focus_pool': int(focus.sum()), 'focus_of_still': round(float(focus.sum() / max(still.sum(), 1)), 3), 'changed_px_on_still_p50_p90': [float(np.quantile(moving[still], q)) for q in (.5, .9)]}
        if k:
            with h5py.File(name, 'r') as f:
                step, spacing, run = int(f.attrs['frame_step']), np.diff(f['source_frame'][:]), f['segment'][:]
                row['source_fps'], row['frame_step'] = float(f.attrs['source_fps']), step
            row['pairs_off_step'] = int((spacing[np.diff(run) == 0] != step).sum())
        rows_out.append(row)
        print(json.dumps(row), flush=True)
        picks = [rng.choice(np.flatnonzero(m), 2) for m in (still, moved, here & ~kept) if m.any()]
        strip = []
        for r in np.concatenate(picks):
            a, b = (torch.from_numpy(frames[i]).permute(2, 0, 1).float()[None] / 255 for i in (r, r + gap[r]))
            aligned = shift_frames(a, torch.from_numpy(s[[r]]), (0, 1))
            strip.append(torch.cat([a, aligned, b, ((aligned - b).abs() * 3).clamp(0, 1)], 3))
        Image.fromarray((torch.cat(strip, 2)[0] * 255).round().byte().permute(1, 2, 0).numpy()).save(f'{out}/pairs_{k:02d}_{row["file"]}.png')
    summary = {'pairs': rows_out, 'change_threshold': CHANGE, 'our_share_of_draws': float((kept & (source == 0)).sum() / table['row_weight'][kept].sum())}
    json.dump(summary, open(f'{out}/report.json', 'w'), indent=1)
    print(f"our share of draws: {summary['our_share_of_draws']:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['merge', 'tables'])
    p.add_argument('--out', default='results/small_action_images/five_hours_data')
    p.add_argument('--batch', type=int, default=2048)
    a = p.parse_args()
    if a.stage == 'merge':
        if not os.path.exists(MERGED_H5):
            merge()
    else:
        report(*tables(a.batch), a.out)


if __name__ == '__main__':
    main()
