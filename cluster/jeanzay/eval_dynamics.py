"""Held-out eval of a dynamics run trained on the small action model's actions, on the dev or confirm half of the 890 windows: the
two commands of experiments/small_action/modal_small_action.py::eval_small_dynamics, for a machine without Modal.

    python cluster/jeanzay/eval_dynamics.py <run name> --split dev|confirm          # last checkpoint of results/<run name>/dynamics
    python cluster/jeanzay/eval_dynamics.py <run name> --split dev --check --device cpu   # loads every input, evaluates nothing

scripts/eval/eval_next_frame.py (32 Euler steps, judge boxes, predictions saved) -> results/evals/next_frame/<run>_<split>_s32.json,
then experiments/small_action/dynamics_score.py -> <run>_<split>_s32_score.json and one RESULT line: Link's box LPIPS on the still-walk
windows (the selection number) and the frame PSNR. Stops before the eval when LPIPS or Inception weights do not load (the eval would
score None without them: GPU nodes have no internet) or when the run was trained on another action dir.
"""

import argparse
import json
import os
import subprocess
import sys

import torch

from evaluation.image_metrics import try_lpips
from evaluation.link_box import judge_boxes, try_inception
from evaluation.windows import split_windows, test_windows
from utils.inference_utils import load_models

# as experiments/small_action/modal_small_action.py: DYN_TOKENIZER, action_paths(DEFAULT_ACTION, DEFAULT_EXTRA_SKIP)[0]
TOKENIZER = 'results/zelda_more_data_tokenizer/video_tokenizer/checkpoints/video_tokenizer_step_59000'
ACTIONS = 'results/small_action_actions_late_weight_focus_200000_yt2'
TEST_H5 = 'data/zelda_test_frames.h5'
LINK_BOXES = 'evaluation/zelda_link_boxes.json'
SPLIT_FILE = 'eval_results/action_eval/zelda/set/split.json'
OUT = 'results/evals/next_frame'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--split', default='dev', choices=['dev', 'confirm'])
    p.add_argument('--step', type=int, default=-1, help='checkpoint step; default the last one')
    p.add_argument('--num-steps', type=int, default=32, help='Euler steps; the split scores are computed at 32 only')
    p.add_argument('--limit', type=int, default=0, help='first N windows of the half (smoke); the outputs get a _smoke suffix')
    p.add_argument('--device', default='cuda')
    p.add_argument('--check', action='store_true', help='load the models, metric weights, windows and boxes, then stop')
    a = p.parse_args()

    root = f'results/{a.run}/dynamics/checkpoints'
    steps = [int(d.rsplit('_', 1)[1]) for d in os.listdir(root) if os.path.exists(f'{root}/{d}/state.pt')]
    ckpt = f'{root}/dynamics_step_{max(steps) if a.step < 0 else a.step}'
    trained = torch.load(f'{ckpt}/state.pt', map_location='cpu', weights_only=False)['config']
    trained = trained if isinstance(trained, dict) else vars(trained)
    assert trained.get('latent_actions_path') == ACTIONS, (trained.get('latent_actions_path'), ACTIONS)
    assert try_lpips('cpu') is not None, 'LPIPS does not load (lpips package, AlexNet weights in $TORCH_HOME)'
    assert try_inception('cpu') is not None, 'the FID Inception network does not load (pytorch-fid, weights in $TORCH_HOME)'

    if a.check:
        load_models(TOKENIZER, ACTIONS, ckpt, torch.device(a.device), use_actions=True)
        windows = split_windows(test_windows(TEST_H5, 3, 4, 8), SPLIT_FILE, a.split, 3, 4)
        boxes = judge_boxes(LINK_BOXES, windows, 128)  # [N, 2]
        print(f'CHECK OK {ckpt}: models, LPIPS and Inception load on {a.device}; {a.split}: {len(windows)} windows, '
              f'{int((boxes[:, 0] >= 0).sum())} with a judge box', flush=True)
        subprocess.run([sys.executable, 'experiments/small_action/dynamics_score.py', '--counts'], check=True)
        return

    name = f'{a.run}_{a.split}_s{a.num_steps}' + ('_smoke' if a.limit else '')
    subprocess.run([sys.executable, '-u', 'scripts/eval/eval_next_frame.py', '--device', a.device, '--test-h5', TEST_H5,
                    '--video-tokenizer-path', TOKENIZER, '--latent-actions-path', ACTIONS, '--dynamics-path', ckpt,
                    '--link-boxes', LINK_BOXES, '--split-file', SPLIT_FILE, '--split', a.split, '--num-steps', str(a.num_steps),
                    '--save-predictions', '--out-dir', OUT, '--name', name, *(['--limit', str(a.limit)] if a.limit else [])], check=True)
    print(f'EVAL DONE {ckpt} -> {OUT}/{name}.json', flush=True)
    if a.num_steps != 32:
        print('no split scores: they are defined at 32 Euler steps')
        return
    subprocess.run([sys.executable, 'experiments/small_action/dynamics_score.py', '--split', a.split, '--eval', f'{a.run}={OUT}/{name}.json',
                    '--out', f'{OUT}/{name}_score.json'], check=True)
    s = json.load(open(f'{OUT}/{name}_score.json'))[a.run]
    none = lambda v: 'none' if v is None else f'{v:.4f}'  # noqa: E731
    print(f"RESULT {a.run} {a.split}{' (first ' + str(a.limit) + ' windows)' if a.limit else ''}: still-walk judge-box LPIPS "
          f"{none(s['still_walking_judge_box_lpips'])} on {s['still_walking_n']} windows (copy last frame "
          f"{none(s['still_walking_copy_judge_box_lpips'])}), PSNR {s['psnr']:.2f} on {s['windows']} windows (copy {s['copy_psnr']:.2f})", flush=True)


if __name__ == '__main__':
    main()
