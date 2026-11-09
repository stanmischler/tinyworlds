"""One sheet for the blind visual check (criterion 6): 12 random still-walk pairs of the confirm half, drawn with seed 0 fixed before
looking, columns frame t | target | prediction | other action (the next drawn pair's a), Link's box, no model name on the image.

    python experiments/small_action/blind_sheet.py --ckpt <dir>/step_N --out results/<dir>
"""

import argparse
import os

import numpy as np

from experiments.small_action.assess import crops, judged_groups, load_group, predict, rolled_predict, sheet, top_left
from experiments.small_action.eval import GAP_ROW, TEST_SHIFTS, load_model

ROWS, SEED = 12, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, _ = load_model(a.ckpt, a.device)
    groups, yx = judged_groups('confirm')
    rows = groups['still_walking']
    picks = np.sort(np.random.default_rng(SEED).choice(len(rows), min(ROWS, len(rows)), replace=False))
    rows = [rows[k] for k in picks]
    table = np.load(TEST_SHIFTS)['shifts'][:, GAP_ROW].astype(np.float32)
    x_a, x_b, s = load_group(rows, table, a.device)
    corners = top_left(rows, yx)
    columns = {'frame t': crops(x_a.cpu(), corners), 'target': crops(x_b.cpu(), corners),
               'prediction': crops(predict(lambda p, q, r: model.predict(p, q, r)[0], x_a, x_b, s), corners), 'other action': crops(rolled_predict(model, x_a, x_b, s), corners)}
    sheet(columns, list(columns), f'{a.out}/still_walking.png')
    print('BLIND_SHEET', a.ckpt, 'pairs a =', [r['a'] for r in rows], flush=True)


if __name__ == '__main__':
    main()
