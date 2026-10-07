"""STA-46 figure: held-out next-frame PSNR (1 pass) vs training step of the v5 per-frame-tokenizer dynamics conditioned on
per-frame-tokenizer CoMo actions (pf_h, STA-43) vs MAE CoMo actions (STA-42), 16 codes and full z, with reference lines.
Reads eval_results/sta46/{pfcomo_<arm>,como_<arm>_v5}_<step>_s1.json.
    python experiments/como_dynamics/plot_psnr_perframe_tokenizer.py   ->   figures/sta46/psnr_vs_step.png
"""
import glob
import json
import os
import re

from experiments.como_dynamics.psnr_plot import COL, INK2, finish, new_figure, plot_arm

R = {}
for f in glob.glob('eval_results/sta46/*_s1.json'):
    m = re.search(r'(pfcomo_(k16|full)|como_(k16|full)_v5)_(\d+)_s1\.json$', f)
    if m:
        R.setdefault(m.group(1), {})[int(m.group(4))] = json.load(open(f))['summary']['model']['psnr']
fig, ax = new_figure([(32.11, 'v5 tokenizer recon (ceiling)', INK2), (27.80, 'v4 best (v3 tok, STA-36 LAM)', INK2), (27.43, 'copy-last', INK2)])
names = {'como_full_v5': 'full z, MAE CoMo (STA-42)', 'pfcomo_full': 'full z, per-frame-tok CoMo (STA-46)',
         'como_k16_v5': '16 codes, MAE CoMo (STA-42)', 'pfcomo_k16': '16 codes, per-frame-tok CoMo (STA-46)'}
for arm in names:
    st = sorted(R[arm])
    plot_arm(ax, st, [R[arm][s] for s in st], COL['full' if 'full' in arm else 'k16'], arm.startswith('como_'), names[arm])
os.makedirs('figures/sta46', exist_ok=True)
finish(ax, 32.5, 0.62, 'Zelda dynamics, v5 per-frame tokenizer: action source (890 held-out windows)', 'figures/sta46/psnr_vs_step.png')
