"""Held-out scores of a small action model, on its own checkpoint or during training.

  - movement-vs-camera action eval (evaluation/action_eval.py): 1,780 judged pairs. Link score per camera group, the worse
    group being the headline; the shift predictor scored against the measured shift; a linear read-out of the shift from a.
  - next frame on the 890 held-out windows (the windows of eval_next_frame.py): PSNR over the whole frame and over the pixels
    the shift does not reveal, judge-box LPIPS, and the same with actions rolled across the batch.
  - memorisation: the same PSNR on 890 training pairs.

    python experiments/small_action/eval.py --ckpt results/<run>/small_action/checkpoints/step_50000 --device mps
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from evaluation.action_eval import block_interval, camera_groups, codes_of, link_scores, linear_shift_readout, shift_scores
from evaluation.image_metrics import try_lpips
from evaluation.link_box import box_lpips, judge_boxes, overlay_windows
from evaluation.windows import test_windows
from experiments.small_action.model import SmallActionModel
from experiments.small_action.shift import known_mse, shift_frames

TEST_H5, TEST_SHIFTS = 'data/zelda_test_frames.h5', 'data/zelda_test_shifts.npz'
SET = 'eval_results/action_eval/zelda/set'
SPLIT = f'{SET}/split.json'  # experiments/small_action/split.py: frame ranges of dev and confirm
GAP_ROW = 1  # gaps (3, 4, 5): the row of gap 4 in a shift table
LINK_BOXES = 'evaluation/zelda_link_boxes.json'


def in_subset(frames, subset):
    # frames: rows of the test .h5 holding frame t of each pair -> bool [N]; subset: 'dev', 'confirm', or None / 'all' for every pair
    frames = np.asarray(frames)
    if subset in (None, 'all'):
        return np.ones(len(frames), bool)
    return np.any([(lo <= frames) & (frames <= hi) for lo, hi in json.load(open(SPLIT))[subset]], axis=0)


def subset_windows(subset):
    # the held-out windows whose frame t (start + 8) lies in the subset
    windows = test_windows(TEST_H5, 3, 4, 8)
    return [w for w, keep in zip(windows, in_subset([s + 8 for _, s in windows], subset)) if keep]


def to_pixels(frames, device):
    # uint8 numpy [N, H, W, C] -> float [-1, 1] [N, C, H, W]
    return torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1


def psnr_of_mse(mse_in_model_range):
    # MSE of pixels in [-1, 1] -> PSNR of pixels in [0, 1] (the range is twice as wide, so the MSE is 4 times larger)
    return -10 * torch.log10((mse_in_model_range / 4).clamp(min=1e-10))


@torch.no_grad()
def actions_and_shifts(model, frames_a, frames_b, shifts, device, batch=64):
    # uint8 frames [N, H, W, C] x 2, measured shifts [N, 2] -> a [N, A], predicted shifts [N, 2], both numpy
    actions, predicted = [], []
    for i in range(0, len(frames_a), batch):
        x_a, x_b = to_pixels(frames_a[i:i + batch], device), to_pixels(frames_b[i:i + batch], device)
        s = torch.from_numpy(shifts[i:i + batch]).float().to(device)
        a = model.encoder(shift_frames(x_a, s), x_b)
        actions.append(a.cpu().numpy())
        predicted.append(model.shift_predictor(x_a, a).cpu().numpy())
    return np.concatenate(actions), np.concatenate(predicted)


def subset_rows(subset):
    # rows of pairs.json in the subset, and their positions in the full list
    rows = json.load(open(f'{SET}/pairs.json'))
    keep = np.flatnonzero(in_subset([r['a'] for r in rows], subset))
    return [rows[k] for k in keep], keep


def action_scores(model, device, limit=0, n_boot=0, subset=None):
    # the judged pairs: index a of each pair is the frame row of the test .h5
    rows = subset_rows(subset)[0][:limit or None]
    judged = json.load(open(f'{SET}/labels.json'))
    labels = np.array([judged['kept'].get(str(r['id']), '') for r in rows], dtype=object)
    table = np.load(TEST_SHIFTS)['shifts'][:, GAP_ROW].astype(np.float32)
    with h5py.File(TEST_H5, 'r') as f:
        frames_a = np.stack([f['frames'][r['a']] for r in rows])
        frames_b = np.stack([f['frames'][r['b']] for r in rows])
    measured = np.array([r['shift'] for r in rows], dtype=np.float64)  # whole-pixel shift the judges saw: defines the groups
    actions, predicted = actions_and_shifts(model, frames_a, frames_b, table[[r['a'] for r in rows]], device)
    group, blocks = camera_groups(measured), np.array([r['block'] for r in rows])
    min_pairs = 5 if not limit else 1
    codes = codes_of(torch.from_numpy(actions))
    link = link_scores(codes, labels, group, blocks, min_pairs)
    if n_boot:
        link['block_95'] = block_interval(codes, labels, group, blocks, min_pairs, n_boot)
    valid = np.array([str(r['id']) not in set(judged['no_shift']) for r in rows])
    out = {'link': link, 'codes': codes, 'actions': actions, 'predicted_shift': predicted, 'labels': labels, 'group': group,
           'shift_predictor': shift_scores(predicted, measured, group, valid)}
    if not limit:
        out['shift_from_action'] = shift_scores(linear_shift_readout(actions.astype(np.float64), measured, blocks, valid), measured, group, valid)
    return out


def next_frame_batches(model, frames_a, frames_b, shifts, device, batch=64):
    # -> per pair: known-pixel PSNR with true / rolled a and with the predicted shift (play time), shift-only known-pixel PSNR,
    # full-frame PSNR, and the predictions in [0, 1]
    scores, predictions = {k: [] for k in ('known', 'rolled', 'predicted_shift', 'shift_known', 'full', 'shift_full')}, []
    for i in range(0, len(frames_a), batch):
        x_a, x_b = to_pixels(frames_a[i:i + batch], device), to_pixels(frames_b[i:i + batch], device)
        s = torch.from_numpy(shifts[i:i + batch]).float().to(device)
        aligned = shift_frames(x_a, s)
        a = model.encoder(aligned, x_b)
        pred = (aligned + model.decoder(aligned, a)).clamp(-1, 1)
        rolled = (aligned + model.decoder(aligned, a.roll(1, 0))).clamp(-1, 1)
        aligned_play = shift_frames(x_a, model.shift_predictor(x_a, a))
        play = (aligned_play + model.decoder(aligned_play, a)).clamp(-1, 1)
        scores['predicted_shift'].append(psnr_of_mse(known_mse(play, x_b, s)))
        scores['known'].append(psnr_of_mse(known_mse(pred, x_b, s)))
        scores['rolled'].append(psnr_of_mse(known_mse(rolled, x_b, s)))
        scores['shift_known'].append(psnr_of_mse(known_mse(aligned, x_b, s)))
        scores['full'].append(psnr_of_mse((pred - x_b).pow(2).mean((1, 2, 3))))
        scores['shift_full'].append(psnr_of_mse((aligned - x_b).pow(2).mean((1, 2, 3))))
        predictions.append(((pred + 1) / 2).cpu())
    return {k: torch.cat(v).cpu() for k, v in scores.items()}, torch.cat(predictions)


@torch.no_grad()
def next_frame_scores(model, device, lpips_fn, limit, train_h5, train_shifts, train_pairs, subset=None):
    windows = subset_windows(subset)[:limit or None]
    table = np.load(TEST_SHIFTS)['shifts'][:, GAP_ROW].astype(np.float32)
    with h5py.File(TEST_H5, 'r') as f:
        frames_a = np.stack([f['frames'][s + 8] for _, s in windows])
        frames_b = np.stack([f['frames'][s + 12] for _, s in windows])
    shifts = table[[s + 8 for _, s in windows]]
    per, predictions = next_frame_batches(model, frames_a, frames_b, shifts, device)
    boxes = torch.as_tensor(judge_boxes(LINK_BOXES, windows, frames_a.shape[1]))
    boxes[torch.tensor(overlay_windows(TEST_H5, windows, 3, 4))] = -1  # paused game under a dialog: nothing to judge
    target = torch.from_numpy(frames_b).permute(0, 3, 1, 2).float() / 255
    lpips_values = torch.cat([box_lpips(lpips_fn, predictions[i:i + 64].to(device), target[i:i + 64].to(device), boxes[i:i + 64]) for i in range(0, len(target), 64)])
    out = {'psnr_full': per['full'].mean().item(), 'psnr_known': per['known'].mean().item(), 'psnr_known_rolled_action': per['rolled'].mean().item(), 'psnr_known_predicted_shift': per['predicted_shift'].mean().item(),
           'shift_only_psnr_full': per['shift_full'].mean().item(), 'shift_only_psnr_known': per['shift_known'].mean().item(),
           'judge_box_lpips': lpips_values.nanmean().item() if lpips_fn is not None else None, 'n_windows': len(windows), 'n_boxes': int((boxes[:, 0] >= 0).sum())}
    if train_pairs is not None:  # memorisation: the same score on fixed training pairs
        table = np.load(train_shifts)['shifts'][:, GAP_ROW].astype(np.float32)
        order = np.argsort(train_pairs)
        rows = train_pairs[order]
        with h5py.File(train_h5, 'r') as f:
            train_a, train_b = f['frames'][rows], f['frames'][rows + 4]
        out['train_psnr_known'] = next_frame_batches(model, train_a, train_b, table[rows], device)[0]['known'].mean().item()
        out['gap_known'] = out['train_psnr_known'] - out['psnr_known']
    return out


def evaluate(model, device, lpips_fn, limit=0, train_pairs=None, train_h5='data/zelda_train_frames.h5', train_shifts='data/zelda_train_shifts.npz', n_boot=0, subset=None):
    # everything but images and saved vectors: a flat dictionary for the log
    model.eval()
    scores = next_frame_scores(model, device, lpips_fn, limit, train_h5, train_shifts, train_pairs, subset)
    action = action_scores(model, device, limit, n_boot, subset)
    model.train()
    link = action['link']
    flat = {f'val_{k}': v for k, v in scores.items()}
    flat.update({'link_still': link['still'].get('nmi_adj'), 'link_moving': link['moving'].get('nmi_adj'), 'link_headline': link['headline'],
                 'shift_error_still': action['shift_predictor']['still']['median'], 'shift_error_moving': action['shift_predictor']['moving']['median']})
    if 'shift_from_action' in action:
        flat.update({'shift_from_a_still': action['shift_from_action']['still']['median'], 'shift_from_a_moving': action['shift_from_action']['moving']['median']})
    return flat, action


def load_model(ckpt, device):
    state = torch.load(f'{ckpt}/state.pt', map_location='cpu', weights_only=False)
    model = SmallActionModel(**model_kwargs(OmegaConf.create(state['config']))).to(device)
    model.load_state_dict(torch.load(f'{ckpt}/model_state_dict.pt', map_location=device, weights_only=True))
    return model.eval().requires_grad_(False), state['step']


def model_kwargs(cfg):
    # keys missing from a config take the model's default (off), so old checkpoints load as they trained
    optional = ('kl_weight', 'error_weight', 'lpips_weight', 'lpips_crop')
    return {**{k: cfg[k] for k in ('action_dim', 'dec_dim', 'dec_depth', 'dec_heads', 'dec_mlp', 'contrastive_weight', 'temperature')},
            **{k: cfg[k] for k in optional if cfg.get(k) is not None}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--out', help='folder for score.json, vectors.npy and predicted_shift.npy')
    p.add_argument('--subset', default='all', choices=['all', 'dev', 'confirm'])
    a = p.parse_args()
    model, step = load_model(a.ckpt, a.device)
    flat, action = evaluate(model, a.device, try_lpips(a.device), n_boot=200, subset=a.subset)
    flat.update({'step': step, 'block_95': action['link']['block_95'], 'link': {k: action['link'][k] for k in ('still', 'moving')}})
    print(json.dumps(flat, indent=1))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(flat, open(f'{a.out}/score.json', 'w'), indent=1)
        np.save(f'{a.out}/vectors.npy', action['actions'])
        np.save(f'{a.out}/predicted_shift.npy', action['predicted_shift'])


if __name__ == '__main__':
    main()
