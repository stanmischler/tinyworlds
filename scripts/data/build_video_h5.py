"""Crop every video of a video config into our Zelda frame at 15 fps and write one .h5 per video.

Per video (needs game_box and stretch from scripts/data/find_video_crop.py): read the mp4 sequentially, keep every
round(video fps / 15)-th frame (60 fps -> every 4th, 30 fps -> every 2nd), crop + resize with datasets/video_crop.py.

Output: <out-dir>/<id>.h5
    frames        uint8 [N, 128, 128, 3]
    source_frame  int64 [N]   frame index in the mp4
    attrs: video_id, channel, fps (=15), source_fps, frame_step, game_box, stretch
Every frame is kept here; scripts/data/flag_video_frames.py later removes the non-gameplay runs.

Usage (repo root):
    python scripts/data/build_video_h5.py configs/data/zelda_youtube.yaml --raw-dir data/zelda_youtube/raw \
        --out-dir data/zelda_youtube
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import h5py
import numpy as np
import yaml

from datasets.video_crop import FRAME_SIZE, GameCrop

TARGET_FPS = 15


def build(video, raw_dir, out_dir):
    out_path = os.path.join(out_dir, f"{video['id']}.h5")
    capture = cv2.VideoCapture(os.path.join(raw_dir, f"{video['id']}.mp4"))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    step = int(round(source_fps / TARGET_FPS))
    crop = GameCrop(*video['game_box'], stretch=video['stretch'])

    frames, source_frame, index = [], [], 0
    while True:
        if index % step == 0:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(crop.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            source_frame.append(index)
        elif not capture.grab():          # skipped frame: decode-free advance
            break
        index += 1
    capture.release()

    frames = np.stack(frames)             # [N,128,128,3] uint8
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('frames', data=frames, chunks=(64, FRAME_SIZE, FRAME_SIZE, 3), compression='gzip',
                         compression_opts=4)
        f.create_dataset('source_frame', data=np.asarray(source_frame, dtype=np.int64))
        f.attrs.update(video_id=video['id'], channel=video['channel'], fps=TARGET_FPS, source_fps=source_fps,
                       frame_step=step, game_box=video['game_box'], stretch=video['stretch'])
    return video['id'], len(frames), source_fps, step


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('--raw-dir', default='data/zelda_youtube/raw')
    parser.add_argument('--out-dir', default='data/zelda_youtube')
    args = parser.parse_args()

    with open(args.config) as f:
        videos = yaml.safe_load(f)['videos']
    os.makedirs(args.out_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=5) as pool:
        jobs = [pool.submit(build, v, args.raw_dir, args.out_dir) for v in videos]
        for job in jobs:
            video_id, n, source_fps, step = job.result()
            print(f'{video_id}: {n} frames ({n / TARGET_FPS / 60:.1f} min) from {source_fps:.2f} fps, every {step}')


if __name__ == '__main__':
    main()
