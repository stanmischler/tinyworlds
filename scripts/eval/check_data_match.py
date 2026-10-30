"""Check every video .h5 against our Zelda training frames: 5 checks, each with a fixed pass threshold.

  zoom        |zoom_error| <= 3%  (from scripts/data/find_video_crop.py; Link's height scales with it: 12 px x (1 + err))
  colour      every channel mean within our 1-minute chunk means' mean +- 2 std
  sharpness   Laplacian variance within +-20% of ours
  motion      mean change between consecutive 15 fps frames within +-20% of ours
  tokenizer   frozen tokenizer reconstruction PSNR within 1 dB of ours
Our side: data/zelda_train_frames.h5 from row 1000 (as training reads it), 15 fps = every 4th frame. Video side: kept
frames only (`segment` >= 0), windows of 4 consecutive 15 fps frames inside one segment.

Writes a markdown table and a json (per video: values, ratios, pass flags).

Usage (repo root):
    python scripts/eval/check_data_match.py configs/data/zelda_youtube.yaml --h5-dir data/zelda_youtube \
        --tokenizer <run>/video_tokenizer/checkpoints/video_tokenizer_step_29000 --out eval_results/zelda_youtube/match
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch
import yaml

from evaluation.data_match import channel_means, chunk_channel_means, motion, recon_psnr, sharpness
from utils.utils import load_videotokenizer_from_checkpoint

OUR_TRAIN, OUR_START, OUR_SKIP = 'data/zelda_train_frames.h5', 1000, 4
T = 4                     # window length, as the tokenizer and dynamics train
N_WINDOWS = 256
ZOOM_TOL, COLOUR_STD, RATIO_TOL, PSNR_TOL = 0.03, 2.0, 0.20, 1.0


def windows_at(frames, starts, skip):
    """frames [N,H,W,C], starts [n] -> [n,T,H,W,C]."""
    return np.stack([frames[s:s + T * skip:skip] for s in starts])


def even(values, n):
    return values[np.linspace(0, len(values) - 1, min(n, len(values))).astype(int)]


def our_reference(tokenizer, device):
    with h5py.File(OUR_TRAIN, 'r') as f:
        frames = f['frames'][OUR_START:]                     # 60 fps
    at_15fps = frames[::OUR_SKIP]
    windows = windows_at(frames, even(np.arange(len(frames) - T * OUR_SKIP), N_WINDOWS), OUR_SKIP)
    chunk_means = chunk_channel_means(at_15fps, 15 * 60)    # one row per minute
    return {'colour_mean': chunk_means.mean(0), 'colour_std': chunk_means.std(0), 'sharpness': sharpness(at_15fps[::4]),
            'motion': motion(windows), 'psnr': recon_psnr(tokenizer, windows, device)}


def video_stats(path, tokenizer, device):
    with h5py.File(path, 'r') as f:
        frames, segment = f['frames'][:], f['segment'][:]
    kept = frames[segment >= 0]
    starts = np.nonzero((segment[:-T + 1] >= 0) & (segment[:-T + 1] == segment[T - 1:]))[0]
    windows = windows_at(frames, even(starts, N_WINDOWS), 1)
    return {'colour': channel_means(kept), 'sharpness': sharpness(kept[::4]), 'motion': motion(windows),
            'psnr': recon_psnr(tokenizer, windows, device), 'minutes': len(kept) / 15 / 60}


def judge(video, stats, ours):
    colour_z = (stats['colour'] - ours['colour_mean']) / ours['colour_std']
    row = {
        'zoom_error': video['zoom_error'], 'colour_z': colour_z.round(2).tolist(),
        'sharpness_ratio': stats['sharpness'] / ours['sharpness'], 'motion_ratio': stats['motion'] / ours['motion'],
        'psnr_diff': stats['psnr'] - ours['psnr'], 'psnr': stats['psnr'], 'minutes': stats['minutes'],
    }
    row['pass'] = {
        'zoom': abs(row['zoom_error']) <= ZOOM_TOL,
        'colour': bool(np.all(np.abs(colour_z) <= COLOUR_STD)),
        'sharpness': abs(row['sharpness_ratio'] - 1) <= RATIO_TOL,
        'motion': abs(row['motion_ratio'] - 1) <= RATIO_TOL,
        'tokenizer': abs(row['psnr_diff']) <= PSNR_TOL,
    }
    row['all_pass'] = all(row['pass'].values())
    return row


def markdown(videos, rows, ours):
    mark = lambda ok: 'pass' if ok else '**FAIL**'
    lines = [f"Ours: tokenizer PSNR {ours['psnr']:.2f} dB, sharpness {ours['sharpness']:.0f}, motion {ours['motion']:.2f}",
             '', '| video | min kept | zoom | colour (z R,G,B) | sharpness | motion | tokenizer PSNR | all |',
             '|---|---|---|---|---|---|---|---|']
    for v, r in zip(videos, rows):
        p = r['pass']
        lines.append(f"| {v['channel']} | {r['minutes']:.1f} | {r['zoom_error']:+.1%} {mark(p['zoom'])} | "
                     f"{', '.join(f'{z:+.1f}' for z in r['colour_z'])} {mark(p['colour'])} | "
                     f"×{r['sharpness_ratio']:.2f} {mark(p['sharpness'])} | ×{r['motion_ratio']:.2f} {mark(p['motion'])} | "
                     f"{r['psnr']:.2f} ({r['psnr_diff']:+.2f}) {mark(p['tokenizer'])} | {mark(r['all_pass'])} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('--h5-dir', default='data/zelda_youtube')
    parser.add_argument('--tokenizer', required=True, help='video tokenizer checkpoint directory')
    parser.add_argument('--out', default='eval_results/zelda_youtube/match')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = parser.parse_args()

    with open(args.config) as f:
        videos = yaml.safe_load(f)['videos']
    tokenizer, _ = load_videotokenizer_from_checkpoint(args.tokenizer, args.device)
    ours = our_reference(tokenizer, args.device)
    rows = []
    for video in videos:
        rows.append(judge(video, video_stats(os.path.join(args.h5_dir, f"{video['id']}.h5"), tokenizer, args.device), ours))
        print(video['channel'], rows[-1]['pass'])

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    table = markdown(videos, rows, ours)
    with open(args.out + '.md', 'w') as f:
        f.write(table + '\n')
    with open(args.out + '.json', 'w') as f:
        json.dump({'ours': {k: np.asarray(v).tolist() for k, v in ours.items()},
                   'videos': {v['id']: r for v, r in zip(videos, rows)}}, f, indent=1)
    print(table)


if __name__ == '__main__':
    main()
