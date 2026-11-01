"""Side-by-side Link crops of every run at one checkpoint, from matched_windows.py outputs (box_crops.npz + metrics.json):
one row per window, columns = last context, target, then each run's 1-step prediction, 32x32 Link box zoomed, box PSNR
under each prediction. Rows: held-out windows the diagnosis judged BLURRY (where Link fades), then held-out windows it
judged SHARP, then a few train windows.

    python experiments/regularisation/link_sheet.py --evals results/evals --step 40000 --out link_sheet_40k.png
    python experiments/regularisation/link_sheet.py --step 50000 --run-step zelda_todays_recipe=40000 --out latest.png
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUNS = [('zelda_todays_recipe', "today's recipe"), ('zelda_regularised_shifted', 'regularised + shifted'),
        ('zelda_motion_weighted', 'motion-weighted'), ('zelda_error_weighted', 'error-weighted'),
        ('zelda_action_token', 'action token'), ('zelda_longer_fm', 'longer flow matching')]  # last = merged 13M baseline
ZOOM, PAD, HEAD = 4, 6, 28


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evals', default='results/evals')
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--run-step', action='append', default=[], help='run=step: another checkpoint for that run')
    p.add_argument('--n-heldout', type=int, default=12)
    p.add_argument('--n-sharp', type=int, default=0)
    p.add_argument('--n-train', type=int, default=4)
    p.add_argument('--runs', nargs='+', help='run folders to show, in order (default: all that exist)')
    p.add_argument('--suffix', default='', help="eval folder suffix, e.g. '_32steps' for step_N_32steps")
    p.add_argument('--out', required=True)
    a = p.parse_args()
    steps = {run: a.step for run, _ in RUNS} | {k: int(v) for k, v in (x.split('=') for x in a.run_step)}
    names = dict(RUNS)
    runs = [(run, names[run]) for run in a.runs] if a.runs else RUNS
    runs = [(run, name) for run, name in runs if os.path.isdir(f'{a.evals}/{run}/step_{steps[run]}{a.suffix}')]
    data = {}
    for run, _ in runs:
        d = f'{a.evals}/{run}/step_{steps[run]}{a.suffix}'
        data[run] = (json.load(open(f'{d}/metrics.json'))['windows'], np.load(f'{d}/box_crops.npz')['crops'])  # [N, 3, C, 32, 32]
    rows = data[runs[0][0]][0]
    blurry = [i for i, r in enumerate(rows) if r['split'] == 'heldout' and r['heldout_label'] == 'BLURRY']
    sharp = [i for i, r in enumerate(rows) if r['split'] == 'heldout' and r['heldout_label'] == 'SHARP']
    train = [i for i, r in enumerate(rows) if r['split'] == 'train']
    pick = lambda idx, n: [idx[int(k)] for k in np.linspace(0, len(idx) - 1, n)] if idx else []
    sel = ([('held-out blurry', i) for i in pick(blurry, a.n_heldout)] + [('held-out sharp', i) for i in pick(sharp, a.n_sharp)]
           + [('train', i) for i in pick(train, a.n_train)])

    cell = 32 * ZOOM
    cols = ['last context', 'target'] + [f'{name} {steps[run] // 1000}k' for run, name in runs]
    W = len(cols) * (cell + PAD) + 90
    H = HEAD + len(sel) * (cell + PAD + 14)
    sheet = Image.new('RGB', (W, H), (240, 240, 240))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for c, name in enumerate(cols):
        draw.text((90 + c * (cell + PAD), 8), name, fill='black', font=font)
    for r, (split, i) in enumerate(sel):
        y = HEAD + r * (cell + PAD + 14)
        draw.text((4, y + cell // 2 - 10), f'{split.replace(" ", chr(10))}\n{rows[i]["scene"]}', fill='black', font=font)
        crops = data[runs[0][0]][1][i]  # [3, C, 32, 32]: last context, target, prediction
        tiles = [crops[0], crops[1]] + [data[run][1][i][2] for run, _ in runs]
        for c, t in enumerate(tiles):
            x = 90 + c * (cell + PAD)
            sheet.paste(Image.fromarray(t.transpose(1, 2, 0)).resize((cell, cell), Image.NEAREST), (x, y))
            if c >= 2:
                run = runs[c - 2][0]
                assert data[run][0][i]['id'] == rows[i]['id']
                draw.text((x, y + cell + 1), f'box {data[run][0][i]["box_psnr"]:.1f} dB', fill='black', font=font)
    sheet.save(a.out)
    print(f'saved {a.out}: {len(sel)} rows')


if __name__ == '__main__':
    main()
