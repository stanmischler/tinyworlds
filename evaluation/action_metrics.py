"""Scores of discrete / continuous action codes: entropy, NMI against labels, seeded k-means, held-out linear R^2."""

import math

import numpy as np
import torch


def heldout_r2(feats, target):
    # feats: [N, D], target: [N, 2] -> R^2 of least squares (with bias) fit on the first half, scored on the second
    X = torch.cat([feats.double(), torch.ones(len(feats), 1, dtype=torch.double)], 1)
    y = target.double()
    h = len(X) // 2
    w = torch.linalg.lstsq(X[:h], y[:h]).solution
    resid = y[h:] - X[h:] @ w
    return float(1 - resid.pow(2).sum() / (y[h:] - y[h:].mean(0)).pow(2).sum().clamp_min(1e-8))


def entropy_nats(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def nmi(table):
    # table: [n_codes, n_classes] joint counts -> I(X;Y) / sqrt(H(X) H(Y))
    pxy = table / table.sum()
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    hx, hy = entropy_nats(table.sum(1)), entropy_nats(table.sum(0))
    return mi / math.sqrt(hx * hy) if hx > 0 and hy > 0 else 0.0


def kmeans(z, k, iters=50, seed=0):
    # z: [N, A] -> centres [k, A] (Lloyd, k-means++ init, seeded)
    g = torch.Generator().manual_seed(seed)
    zc = z.cpu().float()
    centres = [zc[torch.randint(len(zc), (1,), generator=g)].squeeze(0)]
    for _ in range(1, k):
        d = torch.cdist(zc, torch.stack(centres)).min(1).values.pow(2)
        centres.append(zc[torch.multinomial(d / d.sum(), 1, generator=g)].squeeze(0))
    c = torch.stack(centres)
    for _ in range(iters):
        a = torch.cdist(zc, c).argmin(1)
        c = torch.stack([zc[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
    return c.to(z.device)


def label_table(codes, labels, n_codes, classes):
    # codes [n] ints, labels [n] class names -> joint counts [n_codes, len(classes)]
    tab = np.zeros((n_codes, len(classes)))
    for c, l in zip(codes, labels):
        tab[c, classes.index(l)] += 1
    return tab


def label_metrics(codes, labels, n_codes, classes, extended=False):
    """Codes vs judge labels: NMI, chance NMI (mean over 200 seeded code permutations), chance-corrected NMI_adj, purity,
    majority baseline and the code x label table. extended (scripts/eval/lam_eval.py): {'n': 0} on an empty set, plus
    completeness and codes_used."""
    if extended and not len(codes):
        return {'n': 0}
    tab = label_table(codes, labels, n_codes, classes)
    rng = np.random.default_rng(0)
    perm = np.mean([nmi(label_table(rng.permutation(codes), labels, n_codes, classes)) for _ in range(200)])
    v, n = nmi(tab), tab.sum()
    out = {'n': int(len(codes)), 'nmi': round(v, 4), 'nmi_chance': round(float(perm), 4), 'nmi_adj': round((v - perm) / (1 - perm), 4),
           'purity': round(float(tab.max(1).sum() / max(n, 1)), 4)}
    if extended:
        out['completeness'] = round(float(tab.max(0).sum() / n), 4)
    out['majority_baseline'] = round(float(tab.sum(0).max() / max(n, 1)), 4)
    if extended:
        out['codes_used'] = int((tab.sum(1) > 0).sum())
    out['table'] = {f'code {k}': {cl: int(tab[k, j]) for j, cl in enumerate(classes) if tab[k, j]} for k in range(n_codes) if tab[k].sum()}
    return out
