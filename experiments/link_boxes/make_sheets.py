"""Kept on purpose: the tooling that built evaluation/zelda_link_boxes.json (make_sheets -> blind judges -> merge_judges).
Annotation sheets for the judge-placed Link boxes: every held-out Zelda eval window (890, the eval_next_frame.py protocol:
3 context frames 4 stored frames apart + target, a window every 8 frames) as a pair 'last context frame | target', both
at 3x with a grid every 16 px labelled in target pixel coordinates (0-128), 6 windows per sheet in 2 columns.

    python experiments/link_boxes/make_sheets.py --out eval_results/link_boxes/sheets
Writes sheet_<k>.png and windows.json (sheet -> window ids, window id -> (block, local start)).
    python experiments/link_boxes/make_sheets.py --out eval_results/link_boxes/sheets_third --windows 1,2,46   # third judge
"""

import argparse
import json
import os

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluation.windows import test_windows, load_window_batch

H5 = 'data/zelda_test_frames.h5'
CONTEXT, SKIP, STRIDE = 3, 4, 8
SCALE, PER_SHEET, COLS, MARGIN = 3, 6, 2, 22


def gridded(frame, font):
    # uint8 [H, W, C] -> PIL image at SCALE x with grid lines every 16 px and pixel labels on the margins
    H, W, _ = frame.shape
    im = Image.new('RGB', (W * SCALE + MARGIN, H * SCALE + MARGIN), 'white')
    im.paste(Image.fromarray(frame).resize((W * SCALE, H * SCALE), Image.NEAREST), (MARGIN, MARGIN))
    d = ImageDraw.Draw(im)
    for v in range(0, W + 1, 16):
        p = MARGIN + v * SCALE
        d.line([(p, MARGIN), (p, MARGIN + H * SCALE)], fill=(0, 255, 255), width=1)
        d.line([(MARGIN, p), (MARGIN + W * SCALE, p)], fill=(0, 255, 255), width=1)
        d.text((p - 6, 4), str(v), fill='black', font=font)
        d.text((0, p - 6), str(v), fill='black', font=font)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='eval_results/link_boxes/sheets')
    p.add_argument('--windows', help='comma-separated window ids (third-judge sheets for the disagreements); default all')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    wins = test_windows(H5, CONTEXT, SKIP, STRIDE)
    chosen = [int(i) for i in args.windows.split(',')] if args.windows else list(range(len(wins)))
    font = ImageFont.load_default(size=13)
    sheets = {}
    with h5py.File(H5, 'r') as h5:
        for k, s in enumerate(range(0, len(chosen), PER_SHEET)):
            ids = chosen[s:s + PER_SHEET]
            x = load_window_batch(h5['frames'], [wins[i] for i in ids], CONTEXT, SKIP)  # uint8 [n, T, H, W, C]
            rows = []
            for j, i in enumerate(ids):
                last, tgt = gridded(x[j, -2], font), gridded(x[j, -1], font)
                row = Image.new('RGB', (last.width + tgt.width + 30, last.height + 22), 'white')
                ImageDraw.Draw(row).text((4, 2), f'WINDOW {i}:   left = previous frame (context)   |   right = TARGET (annotate this one)',
                                         fill='black', font=font)
                row.paste(last, (0, 22)); row.paste(tgt, (last.width + 30, 22))
                rows.append(row)
            w, h = rows[0].width + 40, rows[0].height + 10
            sheet = Image.new('RGB', (COLS * w, -(-len(rows) // COLS) * h), 'white')
            for j, r in enumerate(rows):
                sheet.paste(r, ((j % COLS) * w, (j // COLS) * h))
            sheet.save(f'{args.out}/sheet_{k}.png')
            sheets[k] = ids
    with open(f'{args.out}/windows.json', 'w') as f:
        json.dump({'sheets': sheets, 'windows': [list(w) for w in wins]}, f)
    print(f'{len(sheets)} sheets of {PER_SHEET} windows -> {args.out}')


if __name__ == '__main__':
    main()
