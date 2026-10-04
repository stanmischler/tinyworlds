"""STA-46 figure: held-out next-frame PSNR (1 pass) vs training step of the v5 per-frame-tokenizer dynamics conditioned on
per-frame-tokenizer CoMo actions (pf_h, STA-43) vs MAE CoMo actions (STA-42), 16 codes and full z, with reference lines.
Reads eval_results/sta46/{pfcomo_<arm>,como_<arm>_v5}_<step>_s1.json.
    python scripts/eval/plot_sta46.py   ->   figures/sta46/psnr_vs_step.png
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = {}
for f in glob.glob('eval_results/sta46/*_s1.json'):
    m = re.search(r'(pfcomo_(k16|full)|como_(k16|full)_v5)_(\d+)_s1\.json$', f)
    if m:
        R.setdefault(m.group(1), {})[int(m.group(4))] = json.load(open(f))['summary']['model']['psnr']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
col = {'full': '#2a78d6', 'k16': '#eb6834'}
fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor=SURF)
ax.set_facecolor(SURF)
refs = [(32.11, 'v5 tokenizer recon (ceiling)', INK2), (27.80, 'v4 best (v3 tok, STA-36 LAM)', INK2), (27.43, 'copy-last', INK2)]
for y, label, c in refs:
    ax.axhline(y, color=c, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(29600, y, label, va='center', fontsize=8, color=INK2)
names = {'como_full_v5': 'full z, MAE CoMo (STA-42)', 'pfcomo_full': 'full z, per-frame-tok CoMo (STA-46)',
         'como_k16_v5': '16 codes, MAE CoMo (STA-42)', 'pfcomo_k16': '16 codes, per-frame-tok CoMo (STA-46)'}
for arm in names:
    st = sorted(R[arm])
    mae = arm.startswith('como_')
    ax.plot(st, [R[arm][s] for s in st], color=col['full' if 'full' in arm else 'k16'], lw=2, ls='-' if mae else (0, (5, 2)),
            marker='o' if mae else 's', ms=5, mec=SURF, mew=1.5, label=names[arm], zorder=3)
ax.set_xlim(4000, 29200)
ax.set_ylim(27, 32.5)
ax.set_xlabel('training step', color=INK2)
ax.set_ylabel('held-out next-frame PSNR (dB), 1 pass', color=INK2)
ax.grid(axis='y', color=GRID, lw=.8)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8)
ax.legend(loc='center left', bbox_to_anchor=(0.01, 0.62), frameon=False, fontsize=8, labelcolor=INK, handlelength=4)
ax.set_title('Zelda dynamics, v5 per-frame tokenizer: action source (890 held-out windows)', fontsize=10, color=INK, loc='left')
plt.tight_layout()
os.makedirs('figures/sta46', exist_ok=True)
plt.savefig('figures/sta46/psnr_vs_step.png', dpi=150, facecolor=SURF)
