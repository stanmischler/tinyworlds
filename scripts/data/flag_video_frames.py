"""Flag non-gameplay runs in each video .h5 (datasets/frame_outliers.py) for review, then remove them.

Review pass (default): from black/blank frames and the hand-labelled `junk` ranges of the config, writes each video's
cut runs into the config (`cuts`: [start, end, reason] in 15 fps frame indices, end-exclusive) and a review page with
one thumbnail strip per cut.
Apply pass (--apply, after the review): rewrites each .h5 without the cut frames. `segment` int64 [N] gives the id of
the contiguous kept stretch each frame came from, so the training loader never builds a window across a cut;
`source_frame` still points into the original video.

Usage (repo root):
    python scripts/data/flag_video_frames.py configs/data/zelda_youtube.yaml --h5-dir data/zelda_youtube \
        --review eval_results/zelda_youtube/cuts.html
    python scripts/data/flag_video_frames.py configs/data/zelda_youtube.yaml --h5-dir data/zelda_youtube --apply
"""

import argparse
import base64
import html
import io
import os

import h5py
import numpy as np
import yaml
from PIL import Image

from datasets.frame_outliers import SAMPLE_SECONDS, black_flat_flags, cut_runs, nearest_label, sample_labels, segment_ids
from datasets.video_crop import thumbnail_features


def reasons(flags, start, end):
    """Flag names covering >= 30% of a run, e.g. 'black+junk'."""
    names = [name for name in ('black', 'flat', 'junk') if flags[name][start:end].mean() >= 0.3]
    return '+'.join(names) or 'mixed'


def thumbnail(frame, size=96):
    buffer = io.BytesIO()
    Image.fromarray(frame).resize((size, size), Image.NEAREST).save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def clock(frame_index, fps):
    seconds = int(frame_index / fps)
    return f'{seconds // 60}:{seconds % 60:02d}'


def review_section(video, frames, fps):
    kept = len(frames) - sum(e - s for s, e, _ in video['cuts'])
    rows = []
    for start, end, reason in video['cuts']:
        picks = np.linspace(start, end - 1, 5).astype(int)
        images = ''.join(f'<img src="{thumbnail(frames[i])}" alt="frame {i}">' for i in picks)
        rows.append(f'<tr><td>{clock(start, fps)}–{clock(end, fps)}</td><td>{(end - start) / fps:.0f} s</td>'
                    f'<td>{reason}</td><td class="strip">{images}</td></tr>')
    link = f"https://www.youtube.com/watch?v={video['id']}"
    return (f'<section><h2><a href="{link}">{html.escape(video["channel"])}</a> <code>{video["id"]}</code></h2>'
            f'<p>{len(frames) / fps / 60:.1f} min → {kept / fps / 60:.1f} min kept, {len(video["cuts"])} cuts</p>'
            f'<table><tr><th>video time</th><th>length</th><th>reason</th><th>frames</th></tr>{"".join(rows)}'
            f'</table></section>')


PAGE_STYLE = '''<style>
:root{--bg:#f6f7f3;--fg:#1d2419;--muted:#5d6656;--line:#d9ddd2;--accent:#2f6b3a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151a14;--fg:#e4e9df;--muted:#9aa591;--line:#323b2e;--accent:#8fd19a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#151a14;--fg:#e4e9df;--muted:#9aa591;--line:#323b2e;--accent:#8fd19a;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:24px}
main{max-width:1100px;margin:0 auto;display:flex;flex-direction:column;gap:24px}
h1{margin:0;color:var(--accent)} h2{margin:0;font-size:16px} a{color:var(--accent)} p{margin:4px 0;color:var(--muted)}
section{overflow-x:auto} table{border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:middle;white-space:nowrap}
.strip img{width:72px;height:72px;image-rendering:pixelated;margin-right:3px}
</style>'''


def remove_cuts(path, runs):
    """Rewrite the .h5 at `path` without the frames inside `runs`, adding `segment` (see the module docstring)."""
    with h5py.File(path, 'r') as f:
        assert 'segment' not in f, f'{path}: cuts already removed'
        segment = segment_ids(len(f['frames']), runs)
        keep = segment >= 0
        frames, source_frame, attrs = f['frames'][:][keep], f['source_frame'][:][keep], dict(f.attrs)
    tmp = path + '.tmp'
    with h5py.File(tmp, 'w') as f:
        f.create_dataset('frames', data=frames, compression='gzip', compression_opts=4,
                         chunks=(64,) + frames.shape[1:])
        f.create_dataset('source_frame', data=source_frame)
        f.create_dataset('segment', data=segment[keep])
        f.attrs.update(attrs)
    os.replace(tmp, path)
    print(f'{os.path.basename(path)}: {len(runs)} cuts, {keep.sum()} of {len(keep)} frames kept')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('--h5-dir', default='data/zelda_youtube')
    parser.add_argument('--review', default='eval_results/zelda_youtube/cuts.html')
    parser.add_argument('--apply', action='store_true', help='remove the config cuts from each .h5')
    args = parser.parse_args()

    with open(args.config) as f:
        text = f.read()
    header = ''.join(line + '\n' for line in text.splitlines() if line.startswith('#'))
    config = yaml.safe_load(text)

    if args.apply:
        for video in config['videos']:
            remove_cuts(os.path.join(args.h5_dir, f"{video['id']}.h5"), [(s, e) for s, e, _ in video['cuts']])
        return

    videos = []                                       # (video, fps, thumbs, sample labels); frames re-read later
    for video in config['videos']:
        with h5py.File(os.path.join(args.h5_dir, f"{video['id']}.h5"), 'r') as f:
            assert 'segment' not in f, f"{video['id']}: cuts already removed; rebuild it with build_video_h5.py"
            frames, fps = f['frames'][:], float(f.attrs['fps'])
        thumbs = thumbnail_features(frames)
        step = int(SAMPLE_SECONDS * fps)
        labels = sample_labels(len(range(0, len(frames), step)), video.get('junk', []))
        videos.append((video, fps, thumbs, labels))
    samples = np.concatenate([thumbs[::int(SAMPLE_SECONDS * fps)] for _, fps, thumbs, _ in videos])   # [S,D]
    sample_junk = np.concatenate([labels for *_, labels in videos])                                     # [S]

    sections, total_before, total_kept = [], 0, 0
    for video, fps, thumbs, labels in videos:
        with h5py.File(os.path.join(args.h5_dir, f"{video['id']}.h5"), 'r') as f:
            frames = f['frames'][:]
        black, flat = black_flat_flags(frames)
        flags = {'black': black, 'flat': flat, 'junk': nearest_label(thumbs, samples, sample_junk)}
        flags['any'] = black | flat | flags['junk']
        video['cuts'] = [[s, e, reasons(flags, s, e)] for s, e in cut_runs(flags['any'], fps)]
        kept = len(frames) - sum(e - s for s, e, _ in video['cuts'])
        total_before, total_kept = total_before + len(frames), total_kept + kept
        print(f"{video['id']}: {len(video['cuts'])} cuts, {kept / fps / 60:.1f} of {len(frames) / fps / 60:.1f} min kept "
              f"(raw flags: black {black.mean():.1%}, flat {flat.mean():.1%}, junk {flags['junk'].mean():.1%})")
        sections.append(review_section(video, frames, fps))

    with open(args.config, 'w') as f:
        f.write(header)
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None, width=140)
    os.makedirs(os.path.dirname(args.review), exist_ok=True)
    summary = (f'<p>{total_before / fps / 60:.0f} min built, {total_kept / fps / 60:.0f} min kept. Each row is one '
               f'cut: 5 frames spread over it. Reasons: black (fade), flat (blank screen), junk (closest to a frame '
               f'labelled non-gameplay on the contact sheets: title, menus, cutscene art, another game).</p>')
    with open(args.review, 'w') as f:
        f.write(f'<title>Zelda Video Cuts</title>{PAGE_STYLE}<main><h1>Cuts to review</h1>{summary}'
                f'{"".join(sections)}</main>')
    print(f'review page: {args.review}')


if __name__ == '__main__':
    main()
