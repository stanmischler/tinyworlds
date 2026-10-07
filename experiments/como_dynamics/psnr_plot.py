"""Shared look of the PSNR-vs-step figures of this folder: palette, dotted reference lines, axes styling and saving."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
COL = {'full': '#2a78d6', 'k16': '#eb6834'}


def new_figure(refs):
    # refs: [(psnr, label, color)] -> fig, ax with the dotted horizontal reference lines drawn
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor=SURF)
    ax.set_facecolor(SURF)
    for y, label, c in refs:
        ax.axhline(y, color=c, lw=1, ls=(0, (2, 3)), zorder=1)
        ax.text(29600, y, label, va='center', fontsize=8, color=INK2)
    return fig, ax


def plot_arm(ax, steps, psnrs, color, solid, label):
    ax.plot(steps, psnrs, color=color, lw=2, ls='-' if solid else (0, (5, 2)),
            marker='o' if solid else 's', ms=5, mec=SURF, mew=1.5, label=label, zorder=3)


def finish(ax, ymax, legend_y, title, path):
    ax.set_xlim(4000, 29200)
    ax.set_ylim(27, ymax)
    ax.set_xlabel('training step', color=INK2)
    ax.set_ylabel('held-out next-frame PSNR (dB), 1 pass', color=INK2)
    ax.grid(axis='y', color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(loc='center left', bbox_to_anchor=(0.01, legend_y), frameon=False, fontsize=8, labelcolor=INK, handlelength=4)
    ax.set_title(title, fontsize=10, color=INK, loc='left')
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=SURF)
