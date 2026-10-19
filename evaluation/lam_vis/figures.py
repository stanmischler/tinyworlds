"""PNGs of the action visualisations. Frame grids are pixel-exact (evaluation/image_grid.render_grid); plots use
matplotlib. Labels keep one colour and one marker each in every figure and on the 3D page."""

import numpy as np
import torch

from evaluation.image_grid import render_grid

# judge label -> (display name, colour, matplotlib marker); ATTACK and INTERACT (1 pair each) fold into 'other'.
# Keep the colours in sync with the --l-* tokens of page.html (light theme values).
LABELS = {
    'WALK_U': ('walk up', '#2a78d6', 'D'),
    'WALK_D': ('walk down', '#eb6834', 'D'),
    'WALK_L': ('walk left', '#1baf7a', 'D'),
    'WALK_R': ('walk right', '#6250d6', 'D'),
    'STILL': ('still', '#eda100', 'o'),
    'NONCONTROL': ('no control', '#e34948', 's'),
    'OTHER': ('other (attack, interact)', '#0b0b0b', 'x'),
}
TRAIN_GREY = '#b5b4ae'


def display_label(label):
    return label if label in LABELS else 'OTHER'


def unit(x):
    # uint8 [H, W, C] array or [C, H, W] tensor in [-1, 1] -> float [C, H, W] in [0, 1] (render_grid's input)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).permute(2, 0, 1).float() / 255
    return ((x.float().cpu() + 1) / 2).clamp(0, 1)


def blank(like):
    return torch.ones_like(unit(like))


def cycle_strips(rows, scale=2):
    """rows: dicts with frames a_t, a_next, direct, b_t, b_hat, a_hat and captions (label, psnr_direct, psnr_cycle, cos)."""
    headers = ['A: frame t', 'A: frame t+4 (real)', "A + A's action", 'B: frame t', "B + A's action (1st transfer)",
               'A + action re-inferred from B (2nd transfer)']
    frames = [[unit(r[k]) for k in ('a_t', 'a_next', 'direct', 'b_t', 'b_hat', 'a_hat')] for r in rows]
    captions = [['', '', f"{r['psnr_direct']:.1f} dB", '', f"cos(z, z') {r['cos']:.2f}", f"{r['psnr_cycle']:.1f} dB"]
                for r in rows]
    labels = [LABELS[display_label(r['label'])][0] for r in rows]
    return render_grid(frames, headers, labels, captions, scale)


def same_action_grid(sources, starts, preds, scale=2):
    """sources: list of (label, frame_t, frame_next) real pairs whose action is used; starts: list of S start frames;
    preds: [R][S] frames, preds[r][s] = Dec(starts[s], z of source r). First row: the start frames themselves."""
    headers = ['action taken from: frame t', 'frame t+4'] + [f'start {s + 1}' for s in range(len(starts))]
    rows = [[blank(starts[0]), blank(starts[0])] + [unit(s) for s in starts]]
    labels = ['start frames']
    for (label, f_t, f_next), row in zip(sources, preds):
        rows.append([unit(f_t), unit(f_next)] + [unit(p) for p in row])
        labels.append(LABELS[display_label(label)][0])
    return render_grid(rows, headers, labels, [[''] * len(headers)] * len(rows), scale)


def rollout_grid(rows, n_steps, scale=2):
    """rows: list of (label, frames) with frames = [start, Dec(start, z), Dec(Dec(start, z), z), ...] (n_steps + 1)."""
    headers = ['start'] + [f'x{k} (t+{4 * k})' for k in range(1, n_steps + 1)]
    frames = [[unit(f) for f in fr] for _, fr in rows]
    labels = [LABELS[display_label(lab)][0] for lab, _ in rows]
    return render_grid(frames, headers, labels, [[''] * len(headers)] * len(rows), scale)


def cosine_heatmap(mat, groups, counts, path):
    import matplotlib.pyplot as plt
    names = [f'{LABELS[g][0]} ({n})' for g, n in zip(groups, counts)]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)  # diverging, grey-white midpoint at 0
    ax.set_xticks(range(len(groups)), names, rotation=35, ha='right')
    ax.set_yticks(range(len(groups)), names)
    for i in range(len(groups)):
        for j in range(len(groups)):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(mat[i, j]) > 0.6 else '#0b0b0b')
    fig.colorbar(im, ax=ax, fraction=0.046, label='mean cosine of centred z')
    ax.set_title('Action similarity between judge labels', fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(left=0.22)  # room for the row labels
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pca_views(train_xyz, lab_xyz, labels, var, path):
    """Three 2D projections (PC1-PC2, PC1-PC3, PC2-PC3) of the 3D PCA: train cloud in grey, labelled pairs on top."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    shown = np.array([display_label(l) for l in labels])
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.scatter(train_xyz[:, i], train_xyz[:, j], s=3, c=TRAIN_GREY, alpha=0.4, linewidths=0, label='train pairs')
        for key, (name, colour, marker) in LABELS.items():
            m = shown == key
            if m.any():
                ax.scatter(lab_xyz[m, i], lab_xyz[m, j], s=30, c=colour, marker=marker, label=f'{name} ({m.sum()})',
                           edgecolors='white' if marker != 'x' else None, linewidths=0.6)
        ax.set_xlabel(f'PC{i + 1} ({100 * var[i]:.0f}% var)')
        ax.set_ylabel(f'PC{j + 1} ({100 * var[j]:.0f}% var)')
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc='best', markerscale=1.2)
    fig.suptitle('CoMo action z, PCA fit on train pairs', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
