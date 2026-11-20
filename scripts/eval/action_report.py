"""Action report: are the learned actions of a CoMo action model good enough to drive a dynamics model?
Run on a finished checkpoint, on held-out pairs (t, t+1) of a dataset that records the true buttons (Procgen Bigfish).

Three readings, written to --out-dir:
  action_report.json    probe accuracy (linear and small MLP, fitted on the first half of the held-out file and scored on
                        the second half), the majority-class score, recall per control, the confusion matrix, and the
                        purity of K k-means clusters (the share of pairs whose cluster's most frequent control is theirs:
                        the clusters become the play commands)
  action_confusion.png  confusion matrix of the MLP probe (rows: true control, each row sums to 1)
  action_pca.png        the actions projected on their first principal components, coloured by the true control

Usage (repo root, PYTHONPATH=$PWD):
    python scripts/eval/action_report.py --ckpt results/bigfish_action_model/como/checkpoints/como_step_100000 \
        --test-h5 data/bigfish_test_frames.h5 --out-dir eval_results/action_report/bigfish_action_model
"""

import argparse
import json
import os

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from datasets.split import segments  # noqa: E402
from evaluation.action_check import N_CLASSES, button_classes, fit_probe  # noqa: E402
from utils.utils import load_como_from_checkpoint  # noqa: E402

CONTROLS = ['left+down', 'left', 'left+up', 'down', 'no move', 'up', 'right+down', 'right', 'right+up']  # class id -> name
# one hue per direction, placed around the colour wheel at the direction's angle; grey for no move
COLOURS = ['#7b4fc9', '#2f6fde', '#1aa7a1', '#b23fb0', '#8a8a8a', '#3fa34d', '#d6423b', '#e8862a', '#c9b21e']


@torch.no_grad()
def held_out_actions(lam, test_h5, max_pairs, device, batch=256):
    """Actions of held-out pairs (t, t+1) -> (z [N, A], class [N]), pairs in file order.
    t, t+1 and t+2 stay in one episode: the last row of an episode duplicates the one before, so it is never a t+1."""
    with h5py.File(test_h5, 'r') as f:
        frames, buttons = f['frames'][:], f['buttons'][:]  # uint8 [M, H, W, C], int8 [M]
    starts = np.concatenate([np.arange(s, e - 2) for s, e in segments(test_h5, len(frames)) if e - 2 > s])
    if max_pairs and len(starts) > max_pairs:
        starts = starts[np.linspace(0, len(starts) - 1, max_pairs).round().astype(int)]  # evenly spread, still in file order
    z = []
    for i in range(0, len(starts), batch):
        t = starts[i:i + batch]
        pair = torch.from_numpy(np.stack([frames[t], frames[t + 1]], 1)).to(device)  # [B, 2, H, W, C]
        pair = pair.permute(0, 1, 4, 2, 3).float() / 127.5 - 1                        # [B, 2, C, H, W] in [-1, 1]
        z.append(lam.encode(pair)[:, 0].float().cpu())                                # [B, A]
    return torch.cat(z), button_classes(torch.from_numpy(buttons[starts]))


def kmeans(z, k, steps=50, seed=0):
    """Plain k-means on z [N, A] -> cluster id [N]."""
    gen = torch.Generator().manual_seed(seed)
    centroids = z[torch.randperm(len(z), generator=gen)[:k]].clone()  # [K, A]
    for _ in range(steps):
        cluster = torch.cdist(z, centroids).argmin(1)                  # [N]
        for c in range(k):
            if (cluster == c).any():
                centroids[c] = z[cluster == c].mean(0)
    return torch.cdist(z, centroids).argmin(1)


def cluster_purity(cluster, labels, k):
    # share of pairs whose cluster's most frequent control is their own
    counts = torch.zeros(k, N_CLASSES).index_put_((cluster, labels), torch.ones(len(labels)), accumulate=True)  # [K, 9]
    return (counts.max(1).values.sum() / len(labels)).item()


def plot_confusion(confusion, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(confusion, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES), CONTROLS, rotation=45, ha='right')
    ax.set_yticks(range(N_CLASSES), CONTROLS)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f'{confusion[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if confusion[i, j] > 0.5 else 'black')
    ax.set_xlabel('control predicted from the action')
    ax.set_ylabel('true control')
    ax.set_title('Probe on held-out pairs (each row sums to 1)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pca(z, labels, path, max_points=6000):
    """z [N, A] standardised -> scatter of principal components (1, 2) and (3, 4), coloured by the true control."""
    z = z - z.mean(0)
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    share = s ** 2 / (s ** 2).sum()                 # variance explained by each component
    proj = z @ vt[:4].T                             # [N, 4]
    keep = np.random.default_rng(0).permutation(len(z))[:max_points]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, (a, b) in zip(axes, [(0, 1), (2, 3)]):
        for c in range(N_CLASSES):
            pts = keep[labels[keep] == c]
            ax.scatter(proj[pts, a], proj[pts, b], s=5, alpha=0.35, color=COLOURS[c], linewidths=0)
        for c in range(N_CLASSES):  # class means on top, labelled
            mean = proj[labels == c][:, [a, b]].mean(0)
            ax.scatter(*mean, s=90, color=COLOURS[c], edgecolors='black', linewidths=1, zorder=3, label=CONTROLS[c])
            ax.annotate(CONTROLS[c], mean, xytext=(6, 6), textcoords='offset points', fontsize=9, zorder=4)
        ax.set_xlabel(f'component {a + 1} ({share[a]:.0%} of the variance)')
        ax.set_ylabel(f'component {b + 1} ({share[b]:.0%} of the variance)')
        ax.grid(alpha=0.2)
    axes[0].legend(title='true control (large dot = its mean)', fontsize=8, loc='best')
    fig.suptitle(f'Held-out actions, principal components (small dots: {len(keep)} pairs)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return share


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='CoMo checkpoint dir (como_step_N)')
    p.add_argument('--test-h5', default='data/bigfish_test_frames.h5')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--max-pairs', type=int, default=0, help='0 = every held-out pair')
    p.add_argument('--clusters', type=int, default=16, help='k-means clusters (the play commands)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    lam, state = load_como_from_checkpoint(args.ckpt, args.device)
    z, labels = held_out_actions(lam.eval(), args.test_h5, args.max_pairs, args.device)  # [N, A], [N]
    z = (z - z.mean(0)) / z.std(0).clamp_min(1e-6)

    # probe: fitted on the first half of the file, scored on the second half (different episodes)
    half = len(z) // 2
    y_test = labels[half:]
    linear = fit_probe(z[:half], labels[:half], z[half:])
    mlp = fit_probe(z[:half], labels[:half], z[half:], hidden=64)
    confusion = torch.zeros(N_CLASSES, N_CLASSES).index_put_((y_test, mlp), torch.ones(len(y_test)), accumulate=True)
    confusion = (confusion / confusion.sum(1, keepdim=True).clamp_min(1)).numpy()  # rows: true control
    cluster = kmeans(z, args.clusters)

    plot_confusion(confusion, f'{args.out_dir}/action_confusion.png')
    share = plot_pca(z.numpy(), labels.numpy(), f'{args.out_dir}/action_pca.png')
    report = {
        'checkpoint': args.ckpt, 'step': state.get('step'), 'pairs': len(z), 'action_dim': z.shape[1],
        'probe_linear_acc': (linear == y_test).float().mean().item(),
        'probe_mlp_acc': (mlp == y_test).float().mean().item(),
        'majority_acc': torch.bincount(y_test, minlength=N_CLASSES).max().item() / len(y_test),
        'recall_per_control': {CONTROLS[c]: float(confusion[c, c]) for c in range(N_CLASSES)},
        'confusion_rows_true': confusion.tolist(),
        f'kmeans{args.clusters}_purity': cluster_purity(cluster, labels, args.clusters),
        'pca_variance_share': share.tolist(),
    }
    json.dump(report, open(f'{args.out_dir}/action_report.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != 'confusion_rows_true'}, indent=1))


if __name__ == '__main__':
    main()
