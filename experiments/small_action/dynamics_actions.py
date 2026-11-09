"""Our action for the flow dynamics model: s (2, the measured shift) + a (8, the action encoder of a small action model), 10 numbers.

For a pair (frame t, frame t+n), exactly as the action model's decoder D receives them:
    s = best_shift(frame t, frame t+n)                 (experiments/small_action/shift.py, pixels in [0, 1])
    a = ActionEncoder(frame t moved by s, frame t+n)   (pixels in [-1, 1], fp32, no noise)
The dynamics model gets (s, a) standardised per number with the mean and std of the training pairs, through the same FiLM path as
CoMo's z (train_dynamics.py: action_file + an action dir as latent_actions_path).

Outputs, laid out like scripts/actions/como_actions.py:
  - data/<dataset>_small_action_<tag>.npy  float32 [N, 10]: row r = raw (s, a) of the pair (r, r + the file's frame skip) in
    MultiVideoH5Dataset row order (base .h5 from row 0, then the extra files at --extra-skip, the dynamics yaml's extra_frame_skip);
    the last skip rows of each file are 0
  - data/zelda_test_small_action_<tag>.npy  the same for the held-out .h5 at skip 4 (the eval recomputes it online; kept to check)
  - results/small_action_actions_<tag>/state.pt  action dir {model_type 'small_actions', action_ckpt, mean, std, centroids}:
    utils.load_latent_actions_from_checkpoint -> SmallActions (online encode for the eval and the test curve)
  - results/small_action_actions_<tag>/summary.json  frame step check per file, stats, online vs stored check

    python experiments/small_action/dynamics_actions.py --action-ckpt results/late_weight_focus/small_action/checkpoints/step_200000 \
        --tag late_weight_focus_200000_yt2 --extra-skip 2
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

from datasets.data_utils import multi_video_files
from datasets.split import segments
from evaluation.action_metrics import kmeans
from experiments.small_action.eval import GAP_ROW, load_model
from experiments.small_action.shift import best_shift, shift_frames
from models.como import _CentroidQuantizer

N_SHIFT = 2  # s comes first in the 10 numbers
FIVE_HOURS_SHIFTS = 'data/zelda_five_hours_shifts.npz'  # the action model's pairs: our rows at gap 4, YouTube rows at gap 2
OUR_SHIFTS = 'data/zelda_train_shifts.npz'  # our capture at gaps 3, 4, 5


def raw_actions(encoder, x_a, x_b):
    # frames t, t+n [B, 3, H, W] in [-1, 1] -> raw (s, a) [B, 10]
    s = best_shift((x_a + 1) / 2, (x_b + 1) / 2)  # [B, 2]
    return torch.cat([s, encoder(shift_frames(x_a, s), x_b)], 1)


class SmallActions(nn.Module):
    """Dynamics conditioning from a small action model, with the interface of models.como.CoMoActions: encode(frames [B, T, C, H, W]
    in [-1, 1]) -> standardised (s, a) [B, T-1, 10]; standardize(raw [..., 10]) for precomputed rows; quantizer = nearest of K k-means
    centroids (logging and the eval's lam_code only)."""

    continuous_actions = True

    def __init__(self, encoder, mean, std, centroids):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.action_dim = N_SHIFT + encoder.action_dim
        self.register_buffer('mean', mean.float())  # [10]
        self.register_buffer('std', std.float())  # [10]
        self.register_buffer('centroids', centroids.float())  # [K, 10] raw space
        self.quantizer = _CentroidQuantizer(self)

    def codes(self, raw):
        return torch.cdist(raw.reshape(-1, raw.shape[-1]).float(), self.centroids).argmin(1).view(raw.shape[:-1])

    def standardize(self, raw):
        return (raw.float() - self.mean) / self.std

    @torch.no_grad()
    def encode(self, frames):
        B, T = frames.shape[:2]
        x = frames.float()
        raw = raw_actions(self.encoder, x[:, :-1].flatten(0, 1), x[:, 1:].flatten(0, 1))  # [B*(T-1), 10]
        return self.standardize(raw.view(B, T - 1, -1))


def load_small_actions(action_dir, device):
    st = torch.load(f'{action_dir}/state.pt', map_location='cpu', weights_only=False)
    model, _ = load_model(st['action_ckpt'], device)
    return SmallActions(model.encoder, st['mean'], st['std'], st['centroids']).to(device), st


@torch.no_grad()
def file_actions(encoder, h5, gap, batch, device, limit=0):
    # every pair (r, r + gap) of one .h5 -> raw (s, a) [N, 10], last gap rows 0
    with h5py.File(h5, 'r') as f:
        frames = f['frames'][:limit or None]  # uint8 [N, H, W, C]
    n = len(frames)
    out = np.zeros((n, N_SHIFT + encoder.action_dim), np.float32)
    t0 = time.time()
    for i in range(0, n - gap, batch):
        j = min(i + batch, n - gap)
        x = torch.from_numpy(frames[i:j + gap]).to(device).permute(0, 3, 1, 2).float()  # [b + gap, C, H, W] in 0..255
        s = best_shift(x[:j - i] / 255, x[gap:] / 255)  # as shift.py: pixels / 255
        x = x / 127.5 - 1  # as the action model's training
        out[i:j] = torch.cat([s, encoder(shift_frames(x[:j - i], s), x[gap:])], 1).cpu().numpy()
        if (i // batch) % 20 == 0:
            print(f'  {os.path.basename(h5)}: {j}/{n - gap} pairs, {time.time() - t0:.0f} s', flush=True)
    return out


def step_size(s):
    return np.abs(s).max(1)  # px, the larger of |dy|, |dx|


def moving_quantiles(s):
    # shifts [N, 2] -> median and 95th percentile of the step size on pairs that move at least 1 px
    size = step_size(s)
    return [float(np.quantile(size[size >= 1], q)) for q in (.5, .95)] if (size >= 1).any() else None


def shift_equal(s, table_s):
    # share of pairs whose s equals the table's exactly (tables store float16; quarter pixels are exact there)
    return float((np.abs(s.astype(np.float16).astype(np.float32) - table_s).max(1) == 0).mean())


def frame_step_check(files, parts, fit_rows, sizes):
    # per file: the dynamics pair's skip against the action model's; at the same skip, s must equal the action model's table exactly
    table = np.load(FIVE_HOURS_SHIFTS) if os.path.exists(FIVE_HOURS_SHIFTS) else None
    report, offset = [], 0
    for (h5, gap, _), raw, rows, n_file in zip(files, parts, fit_rows, sizes):
        s = raw[rows, :N_SHIFT]
        size = step_size(s)
        row = {'file': os.path.basename(h5), 'dynamics_skip': gap, 'pairs': int(len(rows)), 'still_share': round(float((size < 1).mean()), 3),
               'moving_px_p50_p95': moving_quantiles(s),
               'a_std': raw[rows, N_SHIFT:].std(0).round(3).tolist()}
        if table is not None and offset + n_file <= len(table['pair_gap']):
            action_gap = int(table['pair_gap'][offset + rows[0]])
            ts = np.nan_to_num(table['shifts'][offset + rows, GAP_ROW].astype(np.float32))
            row.update({'action_model_skip': action_gap, 'match': action_gap == gap,  # another skip, another pair: nothing to compare
                        'shift_equal_to_table': shift_equal(s, ts) if action_gap == gap else None})
        report.append(row)
        print('FRAME_STEP', json.dumps(row), flush=True)
        offset += n_file  # the table holds whole files
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action-ckpt', required=True, help='small action model checkpoint dir')
    p.add_argument('--tag', required=True, help='names the outputs, e.g. late_weight_focus_200000')
    p.add_argument('--dataset', default='ZELDA_MORE_TRAIN')
    p.add_argument('--extra-skip', type=int, default=1, help="rows of a pair on the extra (YouTube) files: the dynamics yaml's extra_frame_skip")
    p.add_argument('--test-h5', default='data/zelda_test_frames.h5')
    p.add_argument('--test-skip', type=int, default=4, help='eval_next_frame --frame-skip')
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--limit', type=int, default=0, help='first N rows of each file (smoke); the outputs get a _smoke suffix')
    p.add_argument('--check', type=int, default=256, help='online SmallActions.encode vs the stored test rows on N test pairs')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    suffix = f'_{a.tag}' + ('_smoke' if a.limit else '')
    model, step = load_model(a.action_ckpt, a.device)
    encoder = model.encoder

    # training pairs, each file at its own skip (as the dynamics loader pairs them); the fit rows: pairs inside one segment, base from its start
    files = multi_video_files(a.dataset, a.extra_skip)
    parts, fit_rows, sizes = [], [], []
    for h5, gap, start in files:
        raw = file_actions(encoder, h5, gap, a.batch, a.device, a.limit)
        with h5py.File(h5, 'r') as f:
            n_file = len(f['frames'])
        spans = [(max(s, start), min(e, len(raw)) - gap) for s, e in segments(h5, n_file)]
        rows = np.concatenate([np.arange(lo, hi) for lo, hi in spans if hi > lo])
        parts.append(raw)
        fit_rows.append(rows)
        sizes.append(n_file)
    train = np.concatenate(parts)
    train_path = f'data/{a.dataset.lower()}_small_action{suffix}.npy'
    np.save(train_path, train)
    print(f'wrote {train_path} {train.shape}', flush=True)
    fit = torch.from_numpy(np.concatenate([raw[rows] for raw, rows in zip(parts, fit_rows)]))
    mean, std = fit.mean(0), fit.std(0).clamp_min(1e-6)
    centroids = kmeans(fit, a.k, seed=0)

    # our capture's shifts must equal the table the action model trained on (the same pair, the same measure)
    ours = parts[0][fit_rows[0], :N_SHIFT]
    table = np.nan_to_num(np.load(OUR_SHIFTS)['shifts'][fit_rows[0], GAP_ROW].astype(np.float32))
    shift_same = shift_equal(ours, table)
    print(f'our capture: shift equal to {OUR_SHIFTS} on {shift_same:.4f} of {len(ours)} pairs', flush=True)

    # held-out pairs at the eval's skip, and the online encode against them
    test = file_actions(encoder, a.test_h5, a.test_skip, a.batch, a.device, a.limit)
    test_path = f'data/{os.path.basename(a.test_h5).replace("_frames.h5", "")}_small_action{suffix}.npy'
    np.save(test_path, test)
    actions = SmallActions(encoder, mean, std, centroids).to(a.device)
    with h5py.File(a.test_h5, 'r') as f:
        rows = np.arange(min(a.check, len(test) - a.test_skip))
        fr = f['frames'][:rows[-1] + a.test_skip + 1]
    x = torch.from_numpy(np.stack([fr[rows], fr[rows + a.test_skip]], 1)).to(a.device).permute(0, 1, 4, 2, 3).float() / 127.5 - 1
    online = torch.cat([actions.encode(x[i:i + 64]) for i in range(0, len(x), 64)])[:, 0].cpu()  # [n, 10] standardised
    stored = actions.standardize(torch.from_numpy(test[rows]).to(a.device)).cpu()
    test_table = np.nan_to_num(np.load('data/zelda_test_shifts.npz')['shifts'][rows, GAP_ROW].astype(np.float32))
    shift_gap = ((online - stored)[:, :N_SHIFT] * std[:N_SHIFT]).abs().max(1).values  # px, raw space
    check = {'pairs': len(rows), 'max_abs_diff_standardised': float((online - stored).abs().max()),
             'shift_equal_online_vs_stored': float(shift_gap.lt(1e-4).float().mean()),
             'shift_equal_stored_vs_test_table': shift_equal(test[rows, :N_SHIFT], test_table)}
    print('CHECK', json.dumps(check), flush=True)

    out_dir = f'results/small_action_actions{suffix}'
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'model_type': 'small_actions', 'action_ckpt': a.action_ckpt, 'action_step': step, 'extra_skip': a.extra_skip, 'mean': mean, 'std': std, 'centroids': centroids,
                'config': {'model_type': 'small_actions', 'n_actions': a.k}}, f'{out_dir}/state.pt')
    summary = {'action_ckpt': a.action_ckpt, 'step': step, 'dataset': a.dataset, 'extra_skip': a.extra_skip, 'train_file': train_path, 'test_file': test_path, 'fit_pairs': len(fit),
               'mean': mean.tolist(), 'std': std.tolist(), 'our_shift_equal_to_table': shift_same, 'check': check,
               'frame_step': frame_step_check(files, parts, fit_rows, sizes),
               'test': {'skip': a.test_skip, 'moving_px_p50_p95': moving_quantiles(test[:-a.test_skip, :N_SHIFT])}}
    json.dump(summary, open(f'{out_dir}/summary.json', 'w'), indent=1)
    print(f'wrote {out_dir}/state.pt and summary.json', flush=True)
    print('SUMMARY', json.dumps({k: v for k, v in summary.items() if k != 'frame_step'}), flush=True)


if __name__ == '__main__':
    main()
