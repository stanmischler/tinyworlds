"""Images of a small action model on held-out data: predictions, attention of the queries, and the judged confusion matrix.

  predictions.png  camera still and camera moving walks, one row each:
                   frame t | aligned frame t | prediction | frame t+4 | |prediction - frame t+4| x 3
  attention.png    aligned frame t | frame t+4 | attention of the 4 heaviest queries, over frame t+4, weight in the corner
  swap.png         frame t+4 | prediction | prediction with the action of the row below | |difference| x 5
  confusion.png    codes (16 k-means clusters) x judge labels, per camera group, codes ordered to put the most on the diagonal

    python experiments/small_action/visualise.py --ckpt <dir>/step_50000 --out eval_results/small_action/<run>
"""

import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from evaluation.action_eval import GROUPS, SCORED
from experiments.small_action.eval import GAP_ROW, SET, TEST_H5, TEST_SHIFTS, action_scores, load_model, subset_rows, to_pixels
from experiments.small_action.shift import shift_frames

N_ROWS = 12


def judged_walks(group, n, subset=None):
    # up to n judged walks of a camera group, evenly spaced through the subset: rows of pairs.json
    rows = subset_rows(subset)[0]
    kept = json.load(open(f'{SET}/labels.json'))['kept']
    still = lambda r: max(abs(r['shift'][0]), abs(r['shift'][1])) <= 1  # noqa: E731
    walks = [r for r in rows if kept.get(str(r['id'])) in ('U', 'D', 'L', 'R') and still(r) == (group == 'still')]
    return walks[:: max(1, len(walks) // n)][:n]


def load_rows(rows, device):
    # -> frame t, frame t+4 [N, 3, 128, 128] in [-1, 1], measured shift [N, 2]
    table = np.load(TEST_SHIFTS)['shifts'][:, GAP_ROW].astype(np.float32)
    with h5py.File(TEST_H5, 'r') as f:
        a = to_pixels(np.stack([f['frames'][r['a']] for r in rows]), device)
        b = to_pixels(np.stack([f['frames'][r['b']] for r in rows]), device)
    return a, b, torch.from_numpy(table[[r['a'] for r in rows]]).to(device)


def save_image(x, path):
    # x: [C, H, W] in [-1, 1]
    Image.fromarray(((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(path)


@torch.no_grad()
def prediction_sheet(model, path, device):
    columns = []
    for group in GROUPS:
        x_a, x_b, s = load_rows(judged_walks(group, N_ROWS), device)
        pred = model.predict(x_a, x_b, s)[0].clamp(-1, 1)
        error = ((pred - x_b).abs() * 3 - 1).clamp(-1, 1)
        columns.append(torch.cat([torch.cat(c, 2) for c in zip(x_a, shift_frames(x_a, s), pred, x_b, error)], 1))  # [C, nH, 5W]
    save_image(torch.cat(columns, 2), path)


@torch.no_grad()
def attention_sheet(model, path, device, top=4):
    x_a, x_b, s = load_rows(judged_walks('still', N_ROWS // 2) + judged_walks('moving', N_ROWS // 2), device)
    aligned = shift_frames(x_a, s)
    maps, weights = model.encoder.attention(aligned, x_b)  # [N, Q, 32, 32], [N, Q]
    panels = []
    for i in range(len(x_a)):
        row = [aligned[i], x_b[i]]
        for q in weights[i].argsort(descending=True)[:top]:
            m = F.interpolate(maps[i, q][None, None], size=128, mode='nearest')[0]  # [1, 128, 128]
            m = m / m.max().clamp(min=1e-8)
            heat = torch.stack([m[0], torch.zeros_like(m[0]), torch.zeros_like(m[0])]) * 2 - 1
            row.append(x_b[i] * (1 - m * 0.75) + heat * m * 0.75)
        panels.append(torch.cat(row, 2))
    save_image(torch.cat(panels, 1), path)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    for i in range(len(x_a)):
        for j, q in enumerate(weights[i].argsort(descending=True)[:top]):
            draw.text(((2 + j) * 128 + 3, i * 128 + 3), f'q{q.item()} {weights[i, q].item():.2f}', fill=(255, 255, 0))
    image.save(path)


@torch.no_grad()
def swap_sheet(model, path, device, subset=None):
    x_a, x_b, s = load_rows(judged_walks('still', N_ROWS // 2, subset) + judged_walks('moving', N_ROWS // 2, subset), device)
    aligned = shift_frames(x_a, s)
    a = model.encoder(aligned, x_b)
    true = (aligned + model.decoder(aligned, a)).clamp(-1, 1)
    other = (aligned + model.decoder(aligned, a.roll(-1, 0))).clamp(-1, 1)
    change = ((true - other).abs() * 5 - 1).clamp(-1, 1)
    save_image(torch.cat([torch.cat(c, 2) for c in zip(x_b, true, other, change)], 1), path)


def confusion_figure(action, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, group in zip(axes, GROUPS):
        used = [l for l in SCORED if ((action['labels'] == l) & (action['group'] == group)).sum() >= 5]
        table = np.zeros((16, len(used)))
        for code, label, g in zip(action['codes'], action['labels'], action['group']):
            if g == group and label in used:
                table[code, used.index(label)] += 1
        table = table[table.sum(1) > 0]
        fractions = table / table.sum(0, keepdims=True).clip(1)
        code_rows, label_cols = linear_sum_assignment(-fractions)
        order = list(code_rows[np.argsort(label_cols)]) + [r for r in range(len(table)) if r not in code_rows]
        ax.imshow(fractions[order], cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(used)), used, rotation=45)
        ax.set_yticks(range(len(order)), [str(i) for i in order])
        ax.set_xlabel('judge label')
        ax.set_ylabel('action cluster')
        ax.set_title(f"camera {group}: score {action['link'][group].get('nmi_adj')}")
        for r, row in enumerate(table[order].astype(int)):
            for c, v in enumerate(row):
                if v:
                    ax.text(c, r, str(v), ha='center', va='center', fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, _ = load_model(a.ckpt, a.device)
    prediction_sheet(model, f'{a.out}/predictions.png', a.device)
    attention_sheet(model, f'{a.out}/attention.png', a.device)
    swap_sheet(model, f'{a.out}/swap.png', a.device)
    confusion_figure(action_scores(model, a.device), f'{a.out}/confusion.png')


if __name__ == '__main__':
    main()
