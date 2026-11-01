"""Pick the two FlowC2S-from-scratch runs to train to 100k, from the step-9999 evals (32 Euler steps, 890 held-out windows):
  plain:  best PSNR among the plain runs (sigma 0.1 / 0.3 / 0.5)
  motion: best judge-box LPIPS among the motion-weighted runs whose PSNR is within 0.5 dB of the plain winner

    python experiments/flowc2s_scratch/pick_winners.py [--evals results/evals]
"""

import argparse
import json
import os

SIGMAS = ('01', '03', '05')
PSNR_MARGIN = 0.5  # dB


def load(evals, run):
    path = f'{evals}/{run}/next_frame/step_9999.json'
    return json.load(open(path))['summary']['model'] if os.path.exists(path) else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evals', default='results/evals')
    a = p.parse_args()
    plain = {f'zelda_flowc2s_s{s}': load(a.evals, f'zelda_flowc2s_s{s}') for s in SIGMAS}
    motion = {f'zelda_flowc2s_s{s}_motion{m}': load(a.evals, f'zelda_flowc2s_s{s}_motion{m}') for s in SIGMAS for m in ('01', '03')}

    print('| run | PSNR | SSIM | LPIPS | judge-box LPIPS | judge-box FID | motion PSNR |')
    print('|---|---|---|---|---|---|---|')
    for run, m in {**plain, **motion}.items():
        fmt = lambda k, f: 'n/a' if m.get(k) is None else format(m[k], f)  # FID is None when Inception did not load
        cells = ['missing'] if m is None else [fmt('psnr', '.2f'), fmt('ssim', '.3f'), fmt('lpips', '.4f'),
                                               fmt('judge_box_lpips', '.4f'), fmt('judge_box_fid', '.1f'), fmt('motion_psnr', '.2f')]
        print(f'| {run} | ' + ' | '.join(cells) + ' |')
    missing = [r for r, m in {**plain, **motion}.items() if m is None]
    if missing:
        print(f'not all evals done: {missing}')
        return

    best_plain = max(plain, key=lambda r: plain[r]['psnr'])
    floor = plain[best_plain]['psnr'] - PSNR_MARGIN
    eligible = {r: m for r, m in motion.items() if m['psnr'] >= floor}
    print(f'plain winner: {best_plain} (PSNR {plain[best_plain]["psnr"]:.2f}); motion runs with PSNR >= {floor:.2f}: {sorted(eligible)}')
    if not eligible:
        print('motion winner: none within the PSNR margin')
        return
    best_motion = min(eligible, key=lambda r: eligible[r]['judge_box_lpips'])
    print(f'motion winner: {best_motion} (judge-box LPIPS {eligible[best_motion]["judge_box_lpips"]:.4f})')
    print(f'WINNERS {best_plain} {best_motion}')


if __name__ == '__main__':
    main()
