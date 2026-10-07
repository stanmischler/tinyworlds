"""STA-42 figure: held-out next-frame PSNR (1 pass) vs training step of the 4 CoMo-conditioned Zelda dynamics arms, with
the reference lines (CoMo decoder, tokenizer recon, v4 best, copy-last). Reads eval_results/como/como_<arm>_<step>_s1.json.
    python experiments/como_dynamics/plot_psnr_como_actions.py   ->   figures/sta42/psnr_vs_step.png
"""
import glob
import json
import re

from experiments.como_dynamics.psnr_plot import COL, INK2, finish, new_figure, plot_arm

R = {}
for f in glob.glob('eval_results/como/como_*_s1.json'):
    m = re.search(r'como_(\w+_v\d)_(\d+)_s1', f)
    R.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(f))['summary']['model']['psnr']
fig, ax = new_figure([(33.41, 'CoMo decoder, full z', COL['full']), (32.53, 'tokenizer recon (ceiling)', INK2),
                      (30.15, 'CoMo decoder, 16 codes', COL['k16']), (27.80, 'v4 best (STA-36 LAM)', INK2), (27.43, 'copy-last', INK2)])
names = {'full_v3': 'full z, v3 tokenizer', 'full_v5': 'full z, v5 tokenizer', 'k16_v3': '16 codes, v3 tokenizer', 'k16_v5': '16 codes, v5 tokenizer'}
for arm in names:
    st = sorted(R[arm])
    plot_arm(ax, st, [R[arm][s] for s in st], COL[arm.split('_')[0]], arm.endswith('v3'), names[arm])
finish(ax, 34, 0.6, 'Zelda dynamics conditioned on CoMo actions (890 held-out windows)', 'figures/sta42/psnr_vs_step.png')
