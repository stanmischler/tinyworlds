"""Find each video's crop into our Zelda frame, write it into the video config, and draw a review sheet.

Per video: detect the game screen box from 60 frames spread over the video, then measure the scale against our frames
(SIFT + RANSAC affine on up to 16 frames) to get the horizontal stretch and the zoom error of the crop rule
(datasets/video_crop.py). A video passes the zoom check when |zoom error| <= 3%.

Writes into the config, per video: game_box [x0, y0, x1, y1], stretch, zoom_error (measured / expected vertical scale
- 1), n_scale_matches. Sheet: one row of our frames, then 6 cropped frames per video.

Usage (repo root):
    python scripts/data/find_video_crop.py configs/data/zelda_youtube.yaml --raw-dir data/zelda_youtube/raw \
        --sheet eval_results/zelda_youtube/crop_sheet.png
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import yaml
from PIL import Image, ImageDraw

from datasets.video_crop import FRAME_SIZE, CROP_SIDE, GameCrop, ReferenceMatcher, detect_game_box, read_frames

OUR_FRAMES = 'data/zelda_frames.h5'
STRETCHES = (1.0, 4 / 3)       # 4:3 game shown as is, or stretched to 16:9


def our_reference_frames(step=150):
    with h5py.File(OUR_FRAMES, 'r') as f:
        return f['frames'][1000::step]       # ~470 frames over the whole video


def find_crop(video_path):
    frames, _ = read_frames(video_path, 60)
    x0, y0, x1, y1 = detect_game_box(frames)

    matcher = ReferenceMatcher(our_reference_frames())
    scales = [m for m in (matcher.match(frame) for frame in frames[::4]) if m is not None and m[0] >= 25]
    sx = float(np.median([m[2] for m in scales]))
    sy = float(np.median([m[3] for m in scales]))
    stretch = min(STRETCHES, key=lambda s: abs(sx / sy - s))
    expected_sy = CROP_SIDE * (y1 - y0) / FRAME_SIZE      # video px per our px under the crop rule
    return {'game_box': [x0, y0, x1, y1], 'stretch': round(stretch, 4), 'zoom_error': round(sy / expected_sy - 1, 4),
            'n_scale_matches': len(scales), 'measured_sx_over_sy': round(sx / sy, 3)}


def draw_sheet(videos, raw_dir, path, n_cols=6, zoom=2):
    with h5py.File(OUR_FRAMES, 'r') as f:
        ours = [f['frames'][i] for i in np.linspace(2000, 70000, n_cols).astype(int)]
    rows = [('ours', ours)]
    for video in videos:
        crop = GameCrop(*video['game_box'], stretch=video['stretch'])
        frames, _ = read_frames(os.path.join(raw_dir, f"{video['id']}.mp4"), n_cols)
        label = f"{video['channel']}\nzoom err {video['zoom_error']:+.1%}\nstretch {video['stretch']:.2f}"
        rows.append((label, [crop.apply(frame) for frame in frames]))

    cell, label_width = FRAME_SIZE * zoom + 6, 170
    sheet = Image.new('RGB', (label_width + n_cols * cell, len(rows) * cell), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    for r, (label, images) in enumerate(rows):
        draw.multiline_text((8, r * cell + 100), label, fill=(255, 255, 255))
        for c, image in enumerate(images):
            tile = Image.fromarray(np.ascontiguousarray(image)).resize((FRAME_SIZE * zoom,) * 2, Image.NEAREST)
            sheet.paste(tile, (label_width + c * cell, r * cell))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('--raw-dir', default='data/zelda_youtube/raw')
    parser.add_argument('--sheet', default='eval_results/zelda_youtube/crop_sheet.png')
    args = parser.parse_args()

    with open(args.config) as f:
        text = f.read()
    header = ''.join(line + '\n' for line in text.splitlines() if line.startswith('#'))
    config = yaml.safe_load(text)
    videos = config['videos']

    paths = [os.path.join(args.raw_dir, f"{v['id']}.mp4") for v in videos]
    with ProcessPoolExecutor(max_workers=5) as pool:
        for video, found in zip(videos, pool.map(find_crop, paths)):
            video.update(found)
            print(f"{video['id']} {video['channel']}: {found}")

    with open(args.config, 'w') as f:
        f.write(header)
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None, width=140)
    draw_sheet(videos, args.raw_dir, args.sheet)
    print(f'sheet: {args.sheet}')


if __name__ == '__main__':
    main()
