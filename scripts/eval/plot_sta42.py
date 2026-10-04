"""STA-42 figure: held-out next-frame PSNR (1 pass) vs training step of the 4 CoMo-conditioned Zelda dynamics arms, with
the reference lines (CoMo decoder, tokenizer recon, v4 best, copy-last). Reads eval_results/como/como_<arm>_<step>_s1.json.
    python scripts/eval/plot_sta42.py   ->   figures/sta42/psnr_vs_step.png
"""
import glob
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = {}
for f in glob.glob('eval_results/como/como_*_s1.json'):
    m = re.search(r'como_(\w+_v\d)_(\d+)_s1', f)
    R.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(f))['summary']['model']['psnr']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
col = {'full': '#2a78d6', 'k16': '#eb6834'}
fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor=SURF)
ax.set_facecolor(SURF)
refs = [(33.41, 'CoMo decoder, full z', col['full']), (32.53, 'tokenizer recon (ceiling)', INK2),
        (30.15, 'CoMo decoder, 16 codes', col['k16']), (27.80, 'v4 best (STA-36 LAM)', INK2), (27.43, 'copy-last', INK2)]
for y, label, c in refs:
    ax.axhline(y, color=c, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(29600, y, label, va='center', fontsize=8, color=INK2)
names = {'full_v3': 'full z, v3 tokenizer', 'full_v5': 'full z, v5 tokenizer', 'k16_v3': '16 codes, v3 tokenizer', 'k16_v5': '16 codes, v5 tokenizer'}
for arm in names:
    st = sorted(R[arm])
    v3 = arm.endswith('v3')
    ax.plot(st, [R[arm][s] for s in st], color=col[arm.split('_')[0]], lw=2, ls='-' if v3 else (0, (5, 2)),
            marker='o' if v3 else 's', ms=5, mec=SURF, mew=1.5, label=names[arm], zorder=3)
ax.set_xlim(4000, 29200)
ax.set_ylim(27, 34)
ax.set_xlabel('training step', color=INK2)
ax.set_ylabel('held-out next-frame PSNR (dB), 1 pass', color=INK2)
ax.grid(axis='y', color=GRID, lw=.8)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8)
ax.legend(loc='center left', bbox_to_anchor=(0.01, 0.6), frameon=False, fontsize=8, labelcolor=INK, handlelength=4)
ax.set_title('Zelda dynamics conditioned on CoMo actions (890 held-out windows)', fontsize=10, color=INK, loc='left')
plt.tight_layout()
plt.savefig('figures/sta42/psnr_vs_step.png', dpi=150, facecolor=SURF)
