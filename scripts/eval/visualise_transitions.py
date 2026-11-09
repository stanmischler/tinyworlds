"""Page of random frame-to-frame transitions of the Zelda train set, as the models see them (one 15 fps step).

Each row: frame t | frame t+1 | |t+1 - t| | |t+1 - best_shift(t)| | flow t -> t+1 | flow best_shift(t) -> t+1.
Pairs are uniform over the valid starts (never across a split cut), with no filtering by content.

Best shift: phase correlation (evaluation/motion.py global_shift), replaced by an exhaustive search over +-SEARCH px only
when that clearly leaves less difference: phase correlation can lock onto static content and miss a real scroll, while the
search alone can lock onto a repeating floor texture at the wrong distance. A nonzero shift must beat no shift by MIN_GAIN.
Rows where phase correlation was overruled say so.

    PYTHONPATH=$PWD ./.venv/bin/python scripts/eval/visualise_transitions.py --n 100 --seed 0 --out transitions/page.html
"""

import argparse
import base64
import html
import io
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb
from PIL import Image, ImageDraw
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from datasets.data_utils import MULTI_VIDEO
from datasets.split import segments
from evaluation.motion import global_shift

H5_PATH, STEP, START = MULTI_VIDEO['ZELDA_MORE_TRAIN'][0]  # the train file, stored frames per 15 fps step, first row read
H5_PATH = H5_PATH.lstrip('/')

SEARCH = 16             # px, exhaustive shift search range
MIN_GAIN = 0.10         # a nonzero shift must lower the mean difference by this fraction of the plain one
SEARCH_GAIN = 0.25      # the search overrules phase correlation only if it lowers the difference by this fraction
MIN_OVERLAP = 0.5       # a shift must keep this fraction of the frame
WEAK_SHIFT = 0.7        # nonzero shift leaving more than this fraction of the plain difference: flagged
LARGE_CHANGE = 10       # mean difference (0-255) still left after compensation: flagged as a change no shift explains
FLOW_DISAGREE = 1.5       # px between the shift and RAFT's median flow: flagged
DIFF_SCALE = 64         # difference (0-255) at the top of the colour map, on every row
MIN_FLOW_SCALE = 2      # px, so RAFT's sub-pixel noise on a static pair is not shown at full colour
RAFT_UPSAMPLE, RAFT_ITERS = 4, 20  # as the stored flow targets (data/zelda_test_flow_gap4.h5)
GREY = (128, 128, 128)
PANELS = ['frame t', 'frame t+1 (one model step)', '|t+1 - t|', '|t+1 - best_shift(t)|', 'flow t &rarr; t+1',
          'flow best_shift(t) &rarr; t+1']


def overlap(dy, dx, H, W):
    # slices (in t, in t+1) of the pixels both frames share when t+1[y, x] ~ t[y - dy, x - dx], with no wraparound
    y0, y1, x0, x1 = max(dy, 0), H + min(dy, 0), max(dx, 0), W + min(dx, 0)
    return (slice(y0 - dy, y1 - dy), slice(x0 - dx, x1 - dx)), (slice(y0, y1), slice(x0, x1))


def mean_diff(a, b, dy, dx):
    # a, b: float [H, W, C] -> mean |b - shifted a| over the shared pixels
    in_a, in_b = overlap(dy, dx, *a.shape[:2])
    return float(np.abs(a[in_a] - b[in_b]).mean())


def best_shift(a, b):
    # a, b: uint8 [H, W, C] -> (dy, dx) of the best shift, (dy, dx) of phase correlation alone
    a, b = a.astype(np.float32), b.astype(np.float32)
    H, W = a.shape[:2]
    phase = global_shift(torch.from_numpy(a.mean(-1))[None], torch.from_numpy(b.mean(-1))[None])[0].tolist()
    candidates = [(dy, dx) for dy in range(-SEARCH, SEARCH + 1) for dx in range(-SEARCH, SEARCH + 1)]
    errors = [mean_diff(a, b, *c) for c in candidates]
    best, error = candidates[int(np.argmin(errors))], min(errors)
    if (H - abs(phase[0])) * (W - abs(phase[1])) >= MIN_OVERLAP * H * W:
        phase_error = mean_diff(a, b, *phase)
        if error > (1 - SEARCH_GAIN) * phase_error:
            best, error = tuple(phase), phase_error
    if error > (1 - MIN_GAIN) * mean_diff(a, b, 0, 0):
        best = (0, 0)
    return best, tuple(phase)


@torch.no_grad()
def raft_flow(raft, a, b, device):
    # a, b: uint8 [H, W, C] of the same size -> numpy [2, H, W] flow in frame px (dx, dy); RAFT needs even sizes here, so
    # an odd last row or column is padded by repeating it
    x = torch.from_numpy(np.stack([a, b])).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1  # [2, C, H, W]
    H, W = x.shape[-2:]
    x = F.pad(x, (0, W % 2, 0, H % 2), mode='replicate')
    x = F.interpolate(x, scale_factor=RAFT_UPSAMPLE, mode='nearest')  # [2, C, uH, uW]
    flow = raft(x[:1], x[1:], num_flow_updates=RAFT_ITERS)[-1]  # [1, 2, uH, uW] in upsampled px
    flow = F.avg_pool2d(flow, RAFT_UPSAMPLE)[0] / RAFT_UPSAMPLE  # [2, H, W] in frame px
    return flow[:, :H, :W].cpu().numpy()


def flow_rgb(flow, scale):
    # flow: [2, H, W] px -> uint8 [H, W, 3]: hue = direction, saturation = |flow| / scale (white = no motion)
    hue = (np.arctan2(flow[1], flow[0]) / (2 * np.pi)) % 1
    sat = np.clip(np.hypot(flow[0], flow[1]) / scale, 0, 1)
    return (hsv_to_rgb(np.stack([hue, sat, np.ones_like(hue)], -1)) * 255).astype(np.uint8)


def diff_rgb(diff):
    # diff: float [H, W] 0-255 -> uint8 [H, W, 3] on the fixed scale, in a colour map that never passes through GREY
    return (colormaps['inferno'](np.clip(diff / DIFF_SCALE, 0, 1))[..., :3] * 255).astype(np.uint8)


def on_grey(panel, where, H, W):
    # panel: uint8 [h, w, 3] of the shared pixels -> uint8 [H, W, 3] with the strip the scroll reveals in neutral grey
    out = np.full((H, W, 3), GREY, np.uint8)
    out[where] = panel
    return out


def data_uri(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, 'PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def wheel_legend(size=160, margin=20):
    # the colour wheel with its four directions; y grows downward as in the images
    r = size // 2 - margin
    dy, dx = np.mgrid[-size // 2:size // 2, -size // 2:size // 2].astype(np.float32)
    rgb = flow_rgb(np.stack([dx, dy]), r)
    rgb[np.hypot(dx, dy) > r] = 255
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for xy, label in [((size - 16, size // 2 - 6), 'R'), ((6, size // 2 - 6), 'L'), ((size // 2 - 3, 4), 'U'), ((size // 2 - 3, size - 16), 'D')]:
        draw.text(xy, label, fill=(0, 0, 0))
    return data_uri(np.asarray(img))


def diff_legend():
    return data_uri(diff_rgb(np.tile(np.linspace(0, DIFF_SCALE, 256)[None], (14, 1))))


def transition(raft, a, b, device):
    # a, b: uint8 [H, W, C] -> dict with the panels, the numbers and the notes of one row
    H, W = a.shape[:2]
    (dy, dx), phase = best_shift(a, b)
    in_a, in_b = overlap(dy, dx, H, W)
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    plain = np.abs(fb - fa).mean(-1)  # [H, W]
    compensated = np.abs(fb[in_b] - fa[in_a]).mean(-1)  # [h, w]
    flow = raft_flow(raft, a, b, device)  # [2, H, W]
    flow_comp = raft_flow(raft, a[in_a], b[in_b], device)  # [2, h, w]
    scale = max(MIN_FLOW_SCALE, float(np.percentile(np.hypot(*flow), 99)),
                float(np.percentile(np.hypot(*flow_comp), 99)))
    mean_plain, mean_comp = float(plain[in_b].mean()), float(compensated.mean())  # same pixels, so they compare
    median_flow = np.median(flow, axis=(1, 2))  # (dx, dy)

    notes, flags = [], []
    if mean_plain == 0:
        notes.append('Duplicate frames: t+1 is identical to t.')
    if tuple(phase) != (dy, dx):
        notes.append(f'Phase correlation alone gave ({phase[1]:+d}, {phase[0]:+d}); overruled by the search or by no shift.')
    if (dy, dx) != (0, 0) and mean_comp > WEAK_SHIFT * mean_plain:
        flags.append('Shift unreliable: it removes little of the difference.')
    if mean_comp > LARGE_CHANGE:
        flags.append('Large change that no single shift explains (cut, fade, menu or text).')
    if np.hypot(median_flow[0] - dx, median_flow[1] - dy) > FLOW_DISAGREE:
        flags.append(f'Shift and RAFT disagree: median flow ({median_flow[0]:+.1f}, {median_flow[1]:+.1f}) px.')
    return {
        'panels': [a, b, diff_rgb(plain), on_grey(diff_rgb(compensated), in_b, H, W), flow_rgb(flow, scale),
                   on_grey(flow_rgb(flow_comp, scale), in_b, H, W)],
        'dx': dx, 'dy': dy, 'mean_plain': mean_plain, 'mean_compensated': mean_comp, 'flow_scale': scale,
        'phase_dx': phase[1], 'phase_dy': phase[0], 'notes': notes, 'flags': flags,
    }


CSS = """
body { margin: 0; background: #fff; color: #222; font: 14px/1.45 -apple-system, Helvetica, Arial, sans-serif; }
main { padding: 16px; }
h1 { font-size: 20px; margin: 0 0 8px; }
p, ul { max-width: 62em; margin: 6px 0; }
.legends { display: flex; flex-wrap: wrap; gap: 28px; align-items: center; margin: 12px 0 18px; }
.legends img { display: block; }
.grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 3px; }
.heads { position: sticky; top: 0; z-index: 1; background: #fff; border-bottom: 1px solid #bbb; margin-bottom: 3px; }
.head { padding: 6px 2px; font-weight: 600; font-size: 13px; }
.grid img { width: 100%; display: block; image-rendering: pixelated; }
.caption { grid-column: 1 / -1; padding: 4px 2px 20px; font-size: 13px; font-variant-numeric: tabular-nums; }
.note { color: #555; display: block; }
.note.bad { color: #b00020; }
"""


def write_page(path, rows, args):
    n_nonzero = sum((r['dx'], r['dy']) != (0, 0) for r in rows)
    n_flagged = sum(bool(r['flags']) for r in rows)
    n_phase_lost = sum((r['phase_dx'], r['phase_dy']) != (r['dx'], r['dy']) for r in rows)
    n_duplicate = sum(r['mean_plain'] == 0 for r in rows)
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Zelda frame transitions</title><style>{CSS}</style></head><body><main>
<h1>Zelda train set: {len(rows)} random transitions</h1>
<p>{html.escape(H5_PATH)}, seed {args.seed}. Frame t+1 is {STEP} stored frames after t: one model step at 15 fps. Pairs are uniform over valid
starts (from row {START}, never across a split cut), not filtered by content, in random order.</p>
<ul>
<li>Nonzero shift on {n_nonzero} of {len(rows)} rows; {n_flagged} rows flagged unreliable (red notes); phase correlation overruled on {n_phase_lost}; {n_duplicate} duplicate pairs.</li>
<li>Shift (dx, dy): t+1 looks like t moved by dx right and dy down. Shifts are whole pixels, but the source scrolls by fractions of a pixel after resizing, so edges stay in the compensated difference.</li>
<li>Grey strip: the part of t+1 that t cannot show after the shift. It is left out of the compensated difference and flow; nothing wraps around.</li>
<li>Mean differences are on the 0-255 scale, averaged over RGB and over the same shared pixels for both, so they compare. The |t+1 - t| panel still shows the full frame.</li>
<li>Flow is RAFT (raft_large, C_T_SKHT_V2) on frames upsampled x{RAFT_UPSAMPLE}, {RAFT_ITERS} iterations, in frame pixels. Both flow panels of a row share one scale, given under the row: full colour at that many px (99th percentile of the row, at least {MIN_FLOW_SCALE} px).</li>
<li>Flow on low-resolution sprites is unreliable: small sprites, animation frames, flat floors and repeating tiles give little to match, and new or vanished content (text, cuts) has no true flow. Trust the broad colour of a region more than a single sprite.</li>
</ul>
<div class="legends">
<div><img src="{wheel_legend()}" width="120" height="120" alt="Flow colour wheel"><div>Flow colour: direction content moves<br>white = no motion</div></div>
<div><img src="{diff_legend()}" width="256" height="14" alt="Difference scale"><div>Difference: 0 (black) to {DIFF_SCALE}+ (pale yellow), same on all rows and both panels</div></div>
</div>
<div class="grid heads">"""]
    parts += [f'<div class="head">{p}</div>' for p in PANELS]
    parts.append('</div><div class="grid">')
    for k, r in enumerate(rows):
        parts += [f'<img src="{data_uri(p)}" alt="">' for p in r['panels']]
        phrase = (f"shift ({r['dx']:+d}, {r['dy']:+d}) px &middot; mean difference {r['mean_plain']:.2f} plain, {r['mean_compensated']:.2f} compensated"
                  f" &middot; flow scale {r['flow_scale']:.1f} px")
        notes = ''.join(f'<span class="note">{html.escape(t)}</span>' for t in r['notes'])
        notes += ''.join(f'<span class="note bad">{html.escape(t)}</span>' for t in r['flags'])
        parts.append(f"""<div class="caption"><b>Sample {k + 1}</b> &middot; frame {r['index']} &rarr; {r['index'] + STEP} &middot; {phrase}{notes}</div>""")
    parts.append('</div></main></body></html>')
    with open(path, 'w') as f:
        f.write('\n'.join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='transitions.html')
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = p.parse_args()

    t0 = time.time()
    with h5py.File(H5_PATH, 'r') as f:
        frames = f['frames']  # [N, 128, 128, 3] uint8
        starts = np.concatenate([np.arange(max(a, START), b - STEP) for a, b in segments(H5_PATH, len(frames))])  # [S]
        picked = np.random.default_rng(args.seed).choice(starts, args.n, replace=False)
        pairs = [(frames[int(i)], frames[int(i) + STEP]) for i in picked]

    raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).eval().to(args.device)
    rows = []
    for k, (i, (a, b)) in enumerate(zip(picked, pairs)):
        rows.append({'index': int(i), **transition(raft, a, b, args.device)})
        print(f"{k + 1}/{args.n} frame {i}: shift ({rows[-1]['dx']:+d}, {rows[-1]['dy']:+d}), "
              f"mean {rows[-1]['mean_plain']:.2f} -> {rows[-1]['mean_compensated']:.2f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_page(args.out, rows, args)
    with open(os.path.splitext(args.out)[0] + '_samples.json', 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'panels'} for r in rows], f, indent=1)
    nonzero = np.mean([(r['dx'], r['dy']) != (0, 0) for r in rows])
    print(f'{args.out}: {os.path.getsize(args.out) / 1e6:.1f} MB, {time.time() - t0:.0f} s, nonzero shift on {nonzero:.0%}, '
          f"{sum(bool(r['flags']) for r in rows)} flagged")


if __name__ == '__main__':
    main()
