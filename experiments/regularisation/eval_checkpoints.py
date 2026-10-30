"""Score every saved checkpoint of one dynamics run (EMA weights), skipping the ones already scored:
  1. matched_windows.py: train vs held-out matched windows, 1 Euler step (the memorisation gap, box crops for judging)
  2. scripts/eval/eval_next_frame.py: the standard held-out eval (890 test windows, lam actions, default Euler steps), with
     the Link metrics: judge-box and motion-box LPIPS / FID, motion-weighted PSNR
Outputs: results/evals/<run>/step_<N>/{metrics.json, box_crops.npz} and results/evals/<run>/next_frame/step_<N>.{json,png}.

    python experiments/regularisation/eval_checkpoints.py --run zelda_todays_recipe
"""

import argparse
import glob
import os
import re
import subprocess
import sys

TOKENIZER = 'results/zelda_cont_final_tp/video_tokenizer/checkpoints/video_tokenizer_step_29000'
ACTIONS = 'results/como_actions_v1_full'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, help='run folder name under results/')
    p.add_argument('--min-step', type=int, default=1, help='skip checkpoints below this step (step 0 = untrained)')
    a = p.parse_args()
    ckpts = glob.glob(f'results/{a.run}/dynamics/checkpoints/dynamics_step_*')
    steps = sorted(int(re.search(r'_(\d+)$', c).group(1)) for c in ckpts)
    out_root = f'results/evals/{a.run}'
    for step in (s for s in steps if s >= a.min_step):
        ckpt = f'results/{a.run}/dynamics/checkpoints/dynamics_step_{step}'
        matched = f'{out_root}/step_{step}'
        if not os.path.exists(f'{matched}/metrics.json'):
            subprocess.run([sys.executable, 'experiments/regularisation/matched_windows.py', '--dynamics-path', ckpt,
                            '--out-dir', matched], check=True)
        if not os.path.exists(f'{out_root}/next_frame/step_{step}.json'):
            subprocess.run([sys.executable, 'scripts/eval/eval_next_frame.py', '--test-h5', 'data/zelda_test_frames.h5',
                            '--video-tokenizer-path', TOKENIZER, '--latent-actions-path', ACTIONS, '--dynamics-path', ckpt,
                            '--action-mode', 'lam', '--link-boxes', 'evaluation/zelda_link_boxes.json',
                            '--out-dir', f'{out_root}/next_frame', '--name', f'step_{step}'],
                           check=True)
        print(f'{a.run} step {step} scored', flush=True)


if __name__ == '__main__':
    main()
