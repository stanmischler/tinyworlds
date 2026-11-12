"""Scores of an action-model run, one line per scored checkpoint, read from its log.jsonl:

    python cluster/lix/scores.py                      # results/weights_and_window/small_action/log.jsonl
    python cluster/lix/scores.py results/<run_name>/small_action/log.jsonl

All scores are on the dev half of the held-out capture, on whole frames (training windows are never scored here). General PSNR leaves out
the strip the camera shift reveals. Link's box: the judged box around Link, on the pairs where he walks.
"""
import json
import sys

COLUMNS = [('step', 'step', '{:d}'),
           ('general PSNR', 'val_psnr_known', '{:.2f}'),
           ('box PSNR still', 'link_still_walking_psnr', '{:.2f}'),
           ('box LPIPS still', 'link_still_walking_lpips', '{:.3f}'),
           ('box PSNR moving', 'link_moving_walking_psnr', '{:.2f}'),
           ('box LPIPS moving', 'link_moving_walking_lpips', '{:.3f}'),
           ('gain still (dB)', 'link_still_walking_gain', '{:.2f}'),
           ('direction still', 'direction_still', '{:.2f}')]

path = sys.argv[1] if len(sys.argv) > 1 else 'results/weights_and_window/small_action/log.jsonl'
rows = [json.loads(line) for line in open(path)]
scored = {r['step']: r for r in rows if 'link_still_walking_lpips' in r}  # a resumed run logs a step again: the last one wins
training = [r for r in rows if 'recon' in r]
print(' | '.join(name for name, _, _ in COLUMNS))
for step in sorted(scored):
    print(' | '.join(form.format(scored[step][key]) if key in scored[step] else '-' for _, key, form in COLUMNS))
if training:
    last = training[-1]
    print(f"last training step {last['step']}: pixel loss {last['recon']:.5f}, {last['it_s']:.2f} steps/s")
print('still = camera still, Link walks (the hard case); moving = camera follows Link. First run at 200k: general PSNR 36.32, '
      'box LPIPS still .255, box PSNR still 22.81; copying frame t gives box LPIPS still .275; shipped 5-stage model .185')
