"""Link box crops (32x32, zoom 4) at 32 Euler steps on the held-out judged windows where the baseline's Link box is
worst (lowest box PSNR), one row per window: last context, target, then each run's prediction with its box PSNR.
Reads the outputs of experiments/regularisation/matched_windows.py --num-steps 32 (<evals>/<run>/step_N_32steps).

    python experiments/sampled_lpips/broken_link_sheet.py results/evals broken_link_sheet.png 12
"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EV, OUT, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
RUNS = [('zelda_longer_fm', 99999, 'longer flow matching'), ('finetune_base_flow_only', 1999, 'flow only control'),
        ('finetune_clean_guess_lpips_3', 1999, 'clean-guess LPIPS 3'), ('finetune_clean_guess_lpips_10', 1999, 'clean-guess LPIPS 10'),
        ('finetune_clean_guess_lpips_30', 1999, 'clean-guess LPIPS 30')]
ZOOM, PAD, HEAD, LEFT = 4, 6, 28, 70
data = {}
for run, step, _ in RUNS:
    d = f'{EV}/{run}/step_{step}_32steps'
    data[run] = (json.load(open(f'{d}/metrics.json'))['windows'], np.load(f'{d}/box_crops.npz')['crops'])
rows = data[RUNS[0][0]][0]
held = [i for i, r in enumerate(rows) if r['split'] == 'heldout']
sel = sorted(held, key=lambda i: rows[i]['box_psnr'])[:N]  # worst baseline Link boxes first
cell = 32 * ZOOM
cols = ['last context', 'target'] + [name for _, _, name in RUNS]
sheet = Image.new('RGB', (LEFT + len(cols) * (cell + PAD), HEAD + len(sel) * (cell + PAD + 14)), (240, 240, 240))
draw = ImageDraw.Draw(sheet); font = ImageFont.load_default()
for c, name in enumerate(cols):
    draw.text((LEFT + c * (cell + PAD), 8), name, fill='black', font=font)
for r, i in enumerate(sel):
    y = HEAD + r * (cell + PAD + 14)
    draw.text((4, y + cell // 2 - 10), f'held-out\n{rows[i]["scene"]}', fill='black', font=font)
    crops = data[RUNS[0][0]][1][i]
    tiles = [crops[0], crops[1]] + [data[run][1][i][2] for run, _, _ in RUNS]
    for c, t in enumerate(tiles):
        x = LEFT + c * (cell + PAD)
        sheet.paste(Image.fromarray(t.transpose(1, 2, 0)).resize((cell, cell), Image.NEAREST), (x, y))
        if c >= 2:
            run = RUNS[c - 2][0]; w = data[run][0][i]
            assert w['id'] == rows[i]['id']
            draw.text((x, y + cell + 1), f'box {w["box_psnr"]:.1f} dB', fill='black', font=font)
sheet.save(OUT)
for run, _, name in RUNS:  # mean box PSNR on these rows and on all held-out windows
    W = data[run][0]
    print(f'{name:22s} worst-{N} {np.mean([W[i]["box_psnr"] for i in sel]):.2f} dB | all held-out {np.mean([W[i]["box_psnr"] for i in held]):.2f} dB')
