"""Interactive 3D PCA page: page.html (template next to this file) + thumbs/*.webp sprite sheets of frame pairs.

Each point is one (frame t, frame t+4) pair; hovering shows the pair. Thumbnails are sprite sheets of TILES_PER_SHEET
tiles (frame t | frame t+4 side by side): labelled test pairs at full 128 px, the grey train cloud at half size to keep
the page small."""

import json
import os

import numpy as np
from PIL import Image

from evaluation.lam_vis.figures import LABELS, display_label

TEMPLATE = os.path.join(os.path.dirname(__file__), 'page.html')
SHEET_COLS, TILES_PER_SHEET = 10, 100


def pair_tile(frame_t, frame_next, size):
    # uint8 [H, W, C] x2 -> PIL [size, 2*size] (frame t | frame t+4)
    tile = Image.new('RGB', (2 * size, size))
    for k, f in enumerate((frame_t, frame_next)):
        tile.paste(Image.fromarray(f).resize((size, size), Image.BILINEAR if size < f.shape[0] else Image.NEAREST), (k * size, 0))
    return tile


def write_sheets(frames_t, frames_next, size, prefix, out_dir):
    """-> list of (sheet file, tile index in sheet) per pair; sheets saved under out_dir/thumbs/."""
    os.makedirs(os.path.join(out_dir, 'thumbs'), exist_ok=True)
    where = []
    for s0 in range(0, len(frames_t), TILES_PER_SHEET):
        n = min(TILES_PER_SHEET, len(frames_t) - s0)
        rows = (n + SHEET_COLS - 1) // SHEET_COLS
        sheet = Image.new('RGB', (SHEET_COLS * 2 * size, rows * size))
        for k in range(n):
            sheet.paste(pair_tile(frames_t[s0 + k], frames_next[s0 + k], size), ((k % SHEET_COLS) * 2 * size, (k // SHEET_COLS) * size))
        name = f'thumbs/{prefix}_{s0 // TILES_PER_SHEET:03d}.webp'
        sheet.save(os.path.join(out_dir, name), quality=80)
        where += [(name, k) for k in range(n)]
    return where


def write_page(out_dir, train, train_xyz, lab, lab_xyz, var, meta):
    """train / lab: dicts from evaluation.lam_vis.data (frames, rows; lab also labels, ids); *_xyz: [N, 3] PCA coords."""
    os.makedirs(out_dir, exist_ok=True)
    train_where = write_sheets(train['frame_t'], train['frame_next'], 64, 'train', out_dir)
    lab_where = write_sheets(lab['frame_t'], lab['frame_next'], 128, 'test', out_dir)
    r = lambda v: [round(float(x), 3) for x in v]
    groups = [{'key': 'TRAIN', 'name': 'train pairs', 'symbol': 'circle',
               'x': r(train_xyz[:, 0]), 'y': r(train_xyz[:, 1]), 'z': r(train_xyz[:, 2]),
               'tiles': train_where, 'info': [f'train row {i}' for i in train['row']]}]
    shown = np.array([display_label(l) for l in lab['label']])
    symbols = {'D': 'diamond', 'o': 'circle', 's': 'square', 'x': 'x'}
    for key, (name, _, marker) in LABELS.items():
        m = np.flatnonzero(shown == key)
        if len(m):
            groups.append({'key': key, 'name': name, 'symbol': symbols[marker],
                           'x': r(lab_xyz[m, 0]), 'y': r(lab_xyz[m, 1]), 'z': r(lab_xyz[m, 2]),
                           'tiles': [lab_where[i] for i in m],
                           'info': [f"{lab['label'][i]} · judge pool window {lab['id'][i]}, test rows {lab['row'][i]}→{lab['row'][i] + 4}" for i in m]})
    data = {'groups': groups, 'var': r(var), 'sheet_cols': SHEET_COLS, **meta}
    payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')  # no '</script>' inside the inline script
    html = open(TEMPLATE).read().replace('__DATA__', payload)
    with open(os.path.join(out_dir, 'page.html'), 'w') as f:
        f.write(html)
