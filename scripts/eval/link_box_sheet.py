"""Sheets for the Link-focused next-frame metrics (evaluation/link_box.py).

  boxes   : N held-out windows evenly spaced: target with the judge box (green) and the motion box (red), and the
            motion-PSNR weight w (camera-compensated |target - last context frame|) as a heatmap. Also prints how often
            the motion box holds the judged Link (judge centre inside the motion box).
  examples: from an eval_next_frame.py run saved with --save-predictions, the k best and k worst windows by judge-box
            LPIPS: last context frame, target, prediction (judge box drawn), target crop, predicted crop, heatmap.
  compare : several eval runs (each saved with --save-predictions) side by side on N windows where Link moves (judge
            box and motion box both present), evenly spaced: target, target crop, each run's predicted crop with its
            judge-box LPIPS and motion-weighted PSNR; a header row holds each run's summary link metrics.

Usage (from the repo root, PYTHONPATH=$PWD):
    python scripts/eval/link_box_sheet.py boxes --out eval_results/link_boxes/boxes_heatmaps.png
    python scripts/eval/link_box_sheet.py examples --eval eval_results/zelda_bigger_fm_s32.json \
        --pred eval_results/zelda_bigger_fm_s32_pred.npz --out eval_results/link_boxes/examples.png
    python scripts/eval/link_box_sheet.py compare --evals eval_results/a_s32.json eval_results/b_s32.json \
        --labels "run a" "run b" --out eval_results/link_boxes/compare.png   # predictions read from <json stem>_pred.npz
"""

import argparse
import json

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from evaluation.link_box import BOX, overlay_windows, judge_boxes, motion_weights
from evaluation.motion import motion_box
from evaluation.windows import test_windows, load_window_batch

H5 = 'data/zelda_test_frames.h5'
JUDGE = 'evaluation/zelda_link_boxes.json'
CONTEXT, SKIP, STRIDE = 3, 4, 8
SCALE = 2
GREEN, RED = (0, 230, 0), (255, 40, 40)
INFERNO = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], float)  # colour stops


def heat(v):
    # v [H, W] >= 0 -> uint8 [H, W, 3], normalised per image, black -> purple -> orange -> yellow
    v = v / v.max() if v.max() > 0 else v
    t = np.clip(v, 0, 1) * (len(INFERNO) - 1)
    i = np.minimum(t.astype(int), len(INFERNO) - 2)
    f = (t - i)[..., None]
    return (INFERNO[i] * (1 - f) + INFERNO[i + 1] * f).astype(np.uint8)


def tile(img_u8, boxes=(), caption='', font=None):
    # uint8 [H, W, 3] -> PIL at SCALE x with (box [2], colour) outlines and a caption strip
    H, W, _ = img_u8.shape
    im = Image.new('RGB', (W * SCALE, H * SCALE + 14), 'black')
    im.paste(Image.fromarray(img_u8).resize((W * SCALE, H * SCALE), Image.NEAREST))
    d = ImageDraw.Draw(im)
    for (r, c), colour in boxes:
        if r >= 0:
            d.rectangle([c * SCALE, r * SCALE, (c + BOX) * SCALE - 1, (r + BOX) * SCALE - 1], outline=colour, width=2)
    d.text((2, H * SCALE + 1), caption, fill=(255, 255, 0), font=font)
    return im


def grid(rows, gap=4):
    # list of lists of PIL tiles -> one PIL image
    w = max(sum(t.width + gap for t in r) for r in rows)
    h = sum(max(t.height for t in r) + gap for r in rows)
    out = Image.new('RGB', (w, h), 'white')
    y = 0
    for r in rows:
        x = 0
        for t in r:
            out.paste(t, (x, y)); x += t.width + gap
        y += max(t.height for t in r) + gap
    return out


def window_data(ids):
    # -> windows, last [n, C, H, W], target [n, C, H, W] in [0, 1], motion boxes [n, 2], judge boxes [n, 2]
    wins = test_windows(H5, CONTEXT, SKIP, STRIDE)
    with h5py.File(H5, 'r') as h5:
        x = torch.from_numpy(load_window_batch(h5['frames'], [wins[i] for i in ids], CONTEXT, SKIP))
    x = x.permute(0, 1, 4, 2, 3).float() / 255  # [n, T, C, H, W]
    last, target = x[:, -2], x[:, -1]
    mbox, _ = motion_box(last, target)
    overlay = torch.tensor(overlay_windows(H5, wins, CONTEXT, SKIP))[ids]
    mbox[overlay] = -1
    jbox = judge_boxes(JUDGE, [wins[i] for i in ids], target.shape[-1])
    jbox[overlay] = -1  # as in eval_next_frame.py
    return wins, last, target, mbox, jbox


def u8(x):
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def hit_rate():
    # judge centre inside the motion box, over windows with both; and the windows only one of them boxes
    wins = test_windows(H5, CONTEXT, SKIP, STRIDE)
    _, _, _, mbox, jbox = window_data(list(range(len(wins))))
    has_m, has_j = mbox[:, 0] >= 0, jbox[:, 0] >= 0
    with open(JUDGE) as f:
        centres = torch.tensor([[-1, -1] if r['x'] is None else [r['y'], r['x']] for r in json.load(f)['windows']],
                               dtype=torch.float)  # [N, 2] (row, col)
    inside = ((centres >= mbox) & (centres < mbox + BOX)).all(1) & has_m & has_j
    both = has_m & has_j
    print(f'motion box on judged Link: {int(inside.sum())} / {int(both.sum())} = {inside.sum() / both.sum():.1%} of windows '
          f'with both boxes; motion box but Link absent {int((has_m & ~has_j).sum())}; Link but no motion box '
          f'{int((~has_m & has_j).sum())}; neither {int((~has_m & ~has_j).sum())}')


def boxes_sheet(args, font):
    n_all = len(test_windows(H5, CONTEXT, SKIP, STRIDE))
    ids = np.linspace(0, n_all - 1, args.n).round().astype(int).tolist()
    _, last, target, mbox, jbox = window_data(ids)
    w = motion_weights(last, target).numpy()  # [n, H, W]
    cells = []
    for j, i in enumerate(ids):
        t = u8(target[j])
        over = (0.45 * t + 0.55 * heat(w[j])).astype(np.uint8)
        cap = f'{i}' + ('' if jbox[j, 0] >= 0 else ' no Link') + ('' if mbox[j, 0] >= 0 else ' no motion box')
        cells += [tile(t, [(jbox[j].tolist(), GREEN), (mbox[j].tolist(), RED)], cap, font), tile(over, [], 'weight w', font)]
    per_row = 6  # 3 windows x (target, heatmap)
    sheet = grid([cells[k:k + per_row] for k in range(0, len(cells), per_row)])
    sheet.save(args.out)
    print(f'saved {args.out} ({args.n} windows; green = judge box, red = motion box)')
    hit_rate()


def examples_sheet(args, font):
    with open(args.eval) as f:
        res = json.load(f)
    pred = np.load(args.pred)['pred']  # [N, H, W, C] uint8, window order of the eval
    lp = np.array(res['windows']['judge_box_lpips'], dtype=float)
    psnr = np.array(res['windows']['psnr'])
    order = [i for i in np.argsort(lp) if not np.isnan(lp[i])]
    picks = [('best', i) for i in order[:args.k]] + [('worst', i) for i in order[::-1][:args.k]]
    ids = [i for _, i in picks]
    _, last, target, mbox, jbox = window_data(ids)
    w = motion_weights(last, target).numpy()
    zoom = lambda img, b: np.kron(img[b[0]:b[0] + BOX, b[1]:b[1] + BOX], np.ones((4, 4, 1), np.uint8))  # 32 -> 128 px
    rows = []
    for j, (tag, i) in enumerate(picks):
        b = jbox[j].tolist()
        t, p = u8(target[j]), pred[i]
        rows.append([tile(u8(last[j]), [], f'{tag}: window {i}, last ctx', font),
                     tile(t, [(b, GREEN)], 'target', font),
                     tile(p, [(b, GREEN)], f'pred  PSNR {psnr[i]:.1f}', font),
                     tile(zoom(t, b), [], 'target box', font),
                     tile(zoom(p, b), [], f'pred box  LPIPS {lp[i]:.3f}', font),
                     tile(heat(w[j]), [], 'weight w', font)])
    grid(rows).save(args.out)
    print(f"saved {args.out} ({args.k} best + {args.k} worst by judge-box LPIPS of {res['name']})")


def text_tile(lines, font, size=(128, 128)):
    # list of str -> PIL tile (same size as an image tile) with the lines written on white
    im = Image.new('RGB', (size[0] * SCALE, size[1] * SCALE + 14), 'white')
    d = ImageDraw.Draw(im)
    for k, line in enumerate(lines):
        d.text((6, 6 + 16 * k), line, fill='black', font=font)
    return im


def compare_sheet(args, font):
    runs = []
    for path in args.evals:
        with open(path) as f:
            res = json.load(f)
        runs.append((res, np.load(path.replace('.json', '_pred.npz'))['pred']))  # pred [N, H, W, C] uint8
    win = runs[0][0]['windows']
    moving = [i for i, (g, jb) in enumerate(zip(win['motion_group'], win['judge_box'])) if g == 'moving' and jb[0] >= 0]
    ids = [moving[k] for k in np.linspace(0, len(moving) - 1, args.n).round().astype(int)]
    _, _, target, _, jbox = window_data(ids)
    zoom = lambda img, b: np.kron(img[b[0]:b[0] + BOX, b[1]:b[1] + BOX], np.ones((4, 4, 1), np.uint8))  # 32 -> 128 px
    fmt = lambda v, d: '-' if v is None else f'{v:.{d}f}'
    header = [text_tile(['all 890 test windows,', '32 Euler steps', '', 'below: windows where', 'Link moves; crop =', 'judge box, 4x'], font),
              text_tile(['target (real frame)', '', f"judge box n = {runs[0][0]['n_link_windows']['judge_box']}",
                         f"motion box n = {runs[0][0]['n_link_windows']['motion_box']}"], font)]
    for label, (res, _) in zip(args.labels, runs):
        m = res['summary']['model']
        header.append(text_tile([label, '', f"full PSNR    {fmt(m['psnr'], 2)}", f"judge LPIPS  {fmt(m['judge_box_lpips'], 3)}",
                                 f"judge FID    {fmt(m['judge_box_fid'], 1)}", f"motion LPIPS {fmt(m['motion_box_lpips'], 3)}",
                                 f"motion FID   {fmt(m['motion_box_fid'], 1)}", f"motion PSNR  {fmt(m['motion_psnr'], 2)}"], font))
    rows = [header]
    for j, i in enumerate(ids):
        b = jbox[j].tolist()
        t = u8(target[j])
        row = [tile(t, [(b, GREEN)], f'window {i}', font), tile(zoom(t, b), [], 'target', font)]
        for label, (res, pred) in zip(args.labels, runs):
            w = res['windows']
            row.append(tile(zoom(pred[i], b), [], f"LPIPS {w['judge_box_lpips'][i]:.3f}  mPSNR {w['motion_psnr'][i]:.1f}", font))
        rows.append(row)
    grid(rows).save(args.out)
    print(f'saved {args.out} ({args.n} windows x {len(runs)} runs)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['boxes', 'examples', 'compare'])
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=30, help='boxes / compare: windows shown')
    p.add_argument('--eval', help='examples: eval_next_frame.py json')
    p.add_argument('--pred', help='examples: its <name>_pred.npz')
    p.add_argument('--k', type=int, default=6, help='examples: best and worst windows shown')
    p.add_argument('--evals', nargs='+', help='compare: eval_next_frame.py jsons, each next to its <stem>_pred.npz')
    p.add_argument('--labels', nargs='+', help='compare: one run name per json')
    args = p.parse_args()
    font = ImageFont.load_default(size=11)
    {'boxes': boxes_sheet, 'examples': examples_sheet, 'compare': compare_sheet}[args.mode](args, font)


if __name__ == '__main__':
    main()
