"""Switch step and loss scale of the focal frequency loss run, from the frequency-loss-only run (the paper's crossing rule,
Degeorge et al. 2026, App. "we identify where the f-loss and v-loss curves intersect ... and do not perform a grid search").

  f_switch_step: first test step where the f-only test PSNR falls to or below the v-only curve (longer flow matching,
                 longer_fm_test_curve.csv) and stays there for the next two evaluations, after having been above it;
                 linearly interpolated between the two evaluations around the crossing.
                 No crossing: the f curve never got above the v curve -> half the run (50000), or it is still above at
                 the last evaluation -> the last evaluated step. Both are flagged in the output.
  f_loss_scale:  mean v-loss / mean f-loss of the f-only run over its train_log.jsonl intervals within 2000 steps of
                 the switch step, so that both terms have the same size where they are mixed.

Usage: python experiments/focal_frequency/pick_switch.py results/zelda_frequency_only_fm/dynamics out.json [plot.png]"""

import csv
import json
import os
import sys


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def crossing_step(steps, f_psnr, v_psnr, half_run=50000):
    # steps, f_psnr, v_psnr: aligned lists (step 0 excluded) -> (step, rule)
    gap = [f - v for f, v in zip(f_psnr, v_psnr)]  # > 0: f-only ahead
    if not any(g > 0 for g in gap):
        return half_run, 'no crossing: f-only never ahead of v-only, half the run'
    first_ahead = next(k for k, g in enumerate(gap) if g > 0)
    for k in range(first_ahead + 1, len(gap)):
        if k + 3 <= len(gap) and all(g <= 0 for g in gap[k:k + 3]):  # behind at k and the next two evaluations
            frac = gap[k - 1] / (gap[k - 1] - gap[k])  # linear interpolation of the zero of the gap
            return int(round(steps[k - 1] + frac * (steps[k] - steps[k - 1]))), 'crossing of the test PSNR curves'
    return steps[-1], 'no crossing: f-only still ahead at its last evaluation, last evaluated step'


def main():
    run_dir, out_path = sys.argv[1], sys.argv[2]
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, 'longer_fm_test_curve.csv')) as f:
        v_curve = {int(r['step']): float(r['test_psnr']) for r in csv.DictReader(f)}
    f_curve = {r['step']: r['test/psnr'] for r in read_jsonl(os.path.join(run_dir, 'test_curve.jsonl'))}
    steps = sorted(s for s in f_curve if s in v_curve and s > 0)
    switch, rule = crossing_step(steps, [f_curve[s] for s in steps], [v_curve[s] for s in steps])

    near = [r for r in read_jsonl(os.path.join(run_dir, 'train_log.jsonl')) if abs(r['step'] - switch) <= 2000 and 'f_loss' in r]
    if not near:  # switch outside the logged range (e.g. half the run beyond a short smoke run): the last intervals
        near = [r for r in read_jsonl(os.path.join(run_dir, 'train_log.jsonl')) if 'f_loss' in r][-3:]
    scale = sum(r['v_loss'] for r in near) / sum(r['f_loss'] for r in near)

    out = {'f_switch_step': switch, 'f_loss_scale': scale, 'rule': rule, 'scale_from_steps': [r['step'] for r in near],
           'curves': {'step': steps, 'f_only_psnr': [f_curve[s] for s in steps], 'v_only_psnr': [v_curve[s] for s in steps]}}
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"f_switch_step {switch} ({rule}); f_loss_scale {scale:.4g} from steps {out['scale_from_steps']}")

    if len(sys.argv) > 3:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(sorted(v_curve)[1:], [v_curve[s] for s in sorted(v_curve)[1:]], label='longer flow matching (v-loss)')
        ax.plot(steps, out['curves']['f_only_psnr'], label='frequency loss only (f-loss)')
        ax.axvline(switch, color='grey', ls='--', lw=1, label=f'switch step {switch}')
        ax.set_xlabel('training step'); ax.set_ylabel('test PSNR (dB), 1 Euler step'); ax.set_ylim(max(30, min(out['curves']['f_only_psnr'] or [30]) - 1), None)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(sys.argv[3], dpi=130)


if __name__ == '__main__':
    main()
