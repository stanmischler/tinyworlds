"""Download the videos listed in a YouTube video config (e.g. configs/data/zelda_youtube.yaml) as video-only mp4.

Picks the best H.264 stream at most --max-height tall at the config's fps (H.264 so OpenCV can decode it), falling back
to any codec at that height. Already-downloaded videos are skipped.

Output: <out-dir>/<id>.mp4

Usage (repo root; needs `pip install yt-dlp imageio-ffmpeg`, data prep only):
    python scripts/data/download_videos.py configs/data/zelda_youtube.yaml --out-dir data/zelda_youtube/raw
"""

import argparse
import os

import imageio_ffmpeg
import yaml
import yt_dlp


def stream_format(max_height, fps):
    best_h264 = f'bv*[height<={max_height}][fps>={fps - 5}][vcodec^=avc1]'
    best_any = f'bv*[height<={max_height}][fps>={fps - 5}]'
    return f'{best_h264}/{best_any}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='yaml with a `videos` list of {id, fps, ...}')
    parser.add_argument('--out-dir', default='data/zelda_youtube/raw')
    parser.add_argument('--max-height', type=int, default=720, help='our 128 px crop is ~550 px of a 720p frame')
    args = parser.parse_args()

    with open(args.config) as f:
        videos = yaml.safe_load(f)['videos']
    os.makedirs(args.out_dir, exist_ok=True)

    for video in videos:
        path = os.path.join(args.out_dir, f"{video['id']}.mp4")
        if os.path.exists(path):
            print(f'{path} exists, skipping')
            continue
        options = {
            'format': stream_format(args.max_height, video['fps']),
            'outtmpl': path,
            'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(),
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video['id']}"])
        print(f'{path} done')


if __name__ == '__main__':
    main()
