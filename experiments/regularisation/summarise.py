"""Curves and table over checkpoints for the regularisation runs, from eval_checkpoints.py outputs:
  - held-out eval (890 test windows, default Euler steps): PSNR / SSIM / LPIPS per step
  - matched windows (1 step): train vs held-out PSNR / LPIPS / box PSNR and the gap (train - held-out)
Writes <evals>/summary.json, summary.md (table at every step) and curves.png.

    python experiments/regularisation/summarise.py --evals results/evals
"""

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt

RUNS = [('zelda_todays_recipe', "today's recipe"), ('zelda_regularised_shifted', 'regularised + shifted'),
        ('zelda_motion_weighted', 'motion-weighted loss'), ('zelda_error_weighted', 'error-weighted loss'),
        ('zelda_action_token', 'action token')]


def step_of(path):
    return int(re.search(r'step_(\d+)', path).group(1))


def load_run(root):
    # -> {step: {metric: value}} for one run's eval folder
    out = {}
    for f in glob.glob(f'{root}/next_frame/step_*.json'):
        m = json.load(open(f))['summary']['model']
        out.setdefault(step_of(f), {}).update(test_psnr=m['psnr'], test_ssim=m['ssim'], test_lpips=m['lpips'])
        for k in ('judge_box_lpips', 'judge_box_fid', 'motion_box_lpips', 'motion_box_fid', 'motion_psnr'):
            if k in m:  # Link metrics (evaluations run with --link-boxes)
                out[step_of(f)][k] = m[k]
    for f in glob.glob(f'{root}/step_*/metrics.json'):
        s = json.load(open(f))['summary']
        row = out.setdefault(step_of(f), {})
        for split in ('train', 'heldout'):
            for k in ('psnr', 'lpips', 'box_psnr', 'box_sharpness'):
                row[f'{split}_{k}'] = s[split][k]['mean']
        row['gap_psnr'] = row['train_psnr'] - row['heldout_psnr']
        row['gap_box_psnr'] = row['train_box_psnr'] - row['heldout_box_psnr']
    return dict(sorted(out.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evals', default='results/evals')
    a = p.parse_args()
    data = {name: load_run(f'{a.evals}/{run}') for run, name in RUNS if os.path.isdir(f'{a.evals}/{run}')}
    json.dump(data, open(f'{a.evals}/summary.json', 'w'), indent=1)

    cols = [('test_psnr', 'test PSNR', '.2f'), ('test_lpips', 'test LPIPS', '.4f'), ('judge_box_lpips', 'judge-box LPIPS', '.4f'),
            ('judge_box_fid', 'judge-box FID', '.1f'), ('motion_psnr', 'motion PSNR', '.2f'), ('train_psnr', 'train PSNR', '.2f'),
            ('heldout_psnr', 'held-out PSNR', '.2f'), ('gap_psnr', 'gap', '+.2f'), ('train_box_psnr', 'train box', '.2f'),
            ('heldout_box_psnr', 'held-out box', '.2f'), ('gap_box_psnr', 'box gap', '+.2f')]
    lines = ['| run | step | ' + ' | '.join(c[1] for c in cols) + ' |', '|' + '---|' * (len(cols) + 2)]
    for name, steps in data.items():
        for step, row in steps.items():
            cells = [format(row[k], f) if k in row else '' for k, _, f in cols]
            lines.append(f'| {name} | {step} | ' + ' | '.join(cells) + ' |')
    open(f'{a.evals}/summary.md', 'w').write('\n'.join(lines) + '\n')

    panels = [('test_psnr', 'held-out test PSNR (890 windows)'), ('judge_box_lpips', 'held-out judge-box LPIPS (Link)'),
              ('judge_box_fid', 'held-out judge-box FID (Link)'), ('motion_psnr', 'held-out motion-weighted PSNR'),
              ('gap_psnr', 'matched windows: train - held-out PSNR'), ('heldout_box_psnr', 'matched held-out Link-box PSNR'),
              ('gap_box_psnr', 'matched windows: train - held-out box PSNR'), ('heldout_box_sharpness', 'held-out box sharpness')]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, (k, title) in zip(axes.flat, panels):
        for name, steps in data.items():
            xs = [s for s, r in steps.items() if k in r]
            ax.plot([x / 1000 for x in xs], [steps[x][k] for x in xs], marker='o', ms=3, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('step (k)')
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{a.evals}/curves.png', dpi=120)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
