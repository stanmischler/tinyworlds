"""PCA of the continuous action z, and cosines of z. Cosines are taken on z centred on the train mean, the same origin
as the PCA, so a cosine and a PCA direction mean the same thing (|train mean| is ~0.3 |z|, so this changes little)."""

import numpy as np


def fit_pca(z, k=3):
    # z [N, A] -> mean [A], components [k, A] (unit rows), explained variance ratio [k]
    mean = z.mean(0)
    _, s, vt = np.linalg.svd(z - mean, full_matrices=False)
    return mean, vt[:k], (s ** 2 / (s ** 2).sum())[:k]


def project(z, mean, components):
    # z [N, A] -> coordinates [N, k]
    return (z - mean) @ components.T


def centred_cosine(z1, z2, mean):
    # z1, z2 [N, A] -> cosine of (z1 - mean, z2 - mean) [N]
    u, v = z1 - mean, z2 - mean
    return (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))


def group_cosine(z, labels, groups, mean):
    """Mean centred cosine between every two label groups -> [G, G]; within a group, pairs i != j only."""
    u = z - mean
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    sim = u @ u.T  # [N, N]
    out = np.zeros((len(groups), len(groups)))
    for gi, a in enumerate(groups):
        for gj, b in enumerate(groups):
            block = sim[np.ix_(labels == a, labels == b)]
            if a == b:
                n = len(block)
                out[gi, gj] = (block.sum() - np.trace(block)) / max(n * (n - 1), 1)
            else:
                out[gi, gj] = block.mean()
    return out


def most_typical(z, mask, n=1):
    """Indices (into z) of the n pairs of a group whose z is closest to the group's mean z: its most typical actions."""
    idx = np.flatnonzero(mask)
    return idx[np.argsort(np.linalg.norm(z[idx] - z[idx].mean(0), axis=1))[:n]]
