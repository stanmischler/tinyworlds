"""STA-35: can token similarity between two frames find the moving character (Link)?

Heuristic: encode frame t and frame t+gap with the video tokenizer (each frame alone, T=1), take the
cosine similarity between every token of t and every token of t+gap ([P, P] with P = Hp*Wp), match each
token i of t to its most similar token f(i) of t+gap (exact ties -> the candidate nearest to i), and call
the token with the largest displacement d(i) = |pos(i) - pos(f(i))| (in cells) the mover. On a still
background this should be Link.

Token types: `fsq` = FSQ-quantized latent [L] (what the dynamics model sees), `hidden` = encoder last
hidden state before the latent head [E]. Flat token index p = Wp*row + col (row-major, as in PatchEmbedding).

Subcommands (from the repo root, PYTHONPATH=$PWD):
    # 1. contact sheet of auto-filtered candidate pairs (localized change, no scroll, not static)
    python experiments/patch_similarity/patch_similarity.py candidates
    # 2. AC1 grid figure for the chosen pairs + AC2 similarity figures for a subset (indices are local
    #    to the test .h5, as printed on the contact sheet)
    python experiments/patch_similarity/patch_similarity.py figures --pairs 120,900,... --ac2 120,900,1500
    # 3. same, with the search restricted to a 12x12-cell square around the largest pixel change
    python experiments/patch_similarity/patch_similarity.py figures --pairs 120,900,... --ac2 120,900,1500 --window 12
    # 4. optimal transport between the two token sets, cost (1 - cos) + lam * distance, for several lam
    python experiments/patch_similarity/patch_similarity.py ot --pairs 120,900,... --lams 0,0.01,0.03,0.1,0.3,1
    # 5. unbalanced OT (tokens may be created / destroyed at cost tau): calibrate (lam, tau) against a pixel
    #    ground truth on static-camera test pairs, then draw the plans on the display pairs
    python experiments/patch_similarity/patch_similarity.py uot-calib --exclude 120,900,...
    python experiments/patch_similarity/patch_similarity.py uot --pairs 120,900,...
    # 6. pixel-level action labels (background median + mover centroid shift) on every test transition
    python experiments/patch_similarity/patch_similarity.py pixel-labels
    # 7. the calibrated unbalanced OT plan of every pair (t, t + gap) of an .h5, as an input of the OT-conditioned LAM
    python experiments/patch_similarity/patch_similarity.py ot-plans --h5 data/zelda_train_frames.h5 --out data/zelda_train_uot_gap4.npz

Outputs go to --out-dir (default eval_results/sta35_patch_similarity/).
"""

import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import scipy.sparse as sp
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from utils.utils import load_videotokenizer_from_checkpoint

DEFAULT_TOKENIZER = '../test/results/zelda_v3_2026_09_26/video_tokenizer/checkpoints/video_tokenizer_step_29000'


# ----------------------------------------------------------------------------- data
def load_pair(frames_dset, t, gap):
    # -> uint8 [2, H, W, C]: frames t and t+gap of the test .h5
    return np.stack([frames_dset[t], frames_dset[t + gap]])


def cell_diff(pair_u8, patch):
    # mean |frame t+gap - frame t| per patch cell, in [0, 1] -> [Hp, Wp]
    x = pair_u8.astype(np.float32) / 255.0
    d = np.abs(x[1] - x[0])  # [H, W, C]
    H, W, _ = d.shape
    return d.reshape(H // patch, patch, W // patch, patch, -1).mean((1, 3, 4))


def test_blocks(h5):
    return json.loads(h5.attrs['test_blocks_local'])


# ----------------------------------------------------------------------------- tokens
@torch.no_grad()
def encode_tokens(tok, pair_u8, device):
    # each frame encoded alone (T=1) -> {'fsq': [2, P, L], 'hidden': [2, P, E]} on cpu, float64
    x = torch.from_numpy(pair_u8).to(device).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1  # [2, C, H, W]
    x = x.unsqueeze(1)  # [B=2, T=1, C, H, W]
    enc = tok.encoder
    hidden = enc.transformer(enc.patch_embed(x))  # [2, 1, P, E]
    fsq = tok.quantizer(enc.latent_head(hidden))  # [2, 1, P, L]
    return {'fsq': fsq[:, 0].cpu().double(), 'hidden': hidden[:, 0].cpu().double()}


def cosine_matrix(a, b):
    # a: [P, D] tokens of frame t, b: [P, D] tokens of frame t+gap -> [P, P] cosine similarity
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return a @ b.T


def match_and_displacement(sim, Wp, mask=None, tie_tol=1e-9):
    # sim: [P, P]; f(i) = argmax_j sim[i, j], exact ties (within float64 tol) broken toward the j nearest to i
    # mask: optional [P] bool window; then both i and its candidates j are restricted to the window, d = -1 outside
    # -> f [P] (flat index in frame t+gap), d [P] (Euclidean displacement in cells)
    P = sim.shape[0]
    pos = torch.stack([torch.arange(P) // Wp, torch.arange(P) % Wp], -1).double()  # [P, 2] (row, col)
    dist = torch.cdist(pos, pos)  # [P, P]
    if mask is not None:
        sim = sim.masked_fill(~mask[None, :], float('-inf'))  # [P, P]
    is_max = sim >= sim.max(dim=1, keepdim=True).values - tie_tol  # [P, P]
    f = torch.where(is_max, dist, torch.full_like(dist, float('inf'))).argmin(dim=1)  # [P]
    d = dist[torch.arange(P), f]  # [P]
    if mask is not None:
        d = d.masked_fill(~mask, -1.0)
    return f, d


def diff_window(pair_u8, patch, size):
    # cell of the pixel with the largest |frame t+gap - frame t| (summed over channels), and a size x size cell
    # square centred on it (shifted to stay inside the grid) -> (r0, c0), (top, left), mask [P] bool
    diff = np.abs(pair_u8[1].astype(np.int32) - pair_u8[0].astype(np.int32)).sum(-1)  # [H, W]
    y, x = np.unravel_index(int(diff.argmax()), diff.shape)
    Hp, Wp = diff.shape[0] // patch, diff.shape[1] // patch
    r0, c0 = y // patch, x // patch
    top = int(np.clip(r0 - size // 2, 0, Hp - size)); left = int(np.clip(c0 - size // 2, 0, Wp - size))
    m = np.zeros((Hp, Wp), dtype=bool); m[top:top + size, left:left + size] = True
    return (int(r0), int(c0)), (top, left), torch.from_numpy(m.reshape(-1))


def ot_plan(sim, Wp, lam):
    # balanced OT between the P tokens of frame t and the P tokens of frame t+gap, uniform mass 1/P each:
    # cost C_ij = (1 - cos_ij) + lam * |pos_i - pos_j| (cells). With equal uniform marginals the optimal plan
    # is a permutation (Birkhoff), so it is solved exactly as an assignment -> sigma [P], d [P] (cells), cost
    P = sim.shape[0]
    pos = torch.stack([torch.arange(P) // Wp, torch.arange(P) % Wp], -1).double()  # [P, 2] (row, col)
    dist = torch.cdist(pos, pos)  # [P, P]
    C = (1 - sim) + lam * dist  # [P, P]
    _, sigma = linear_sum_assignment(C.numpy())
    sigma = torch.from_numpy(sigma)  # [P]
    d = dist[torch.arange(P), sigma]  # [P]
    return sigma, d, float(C[torch.arange(P), sigma].mean())


def uot_plan(sim, Wp, lam, tau):
    # unbalanced OT as an exact assignment on a [2P, 2P] matrix: a token of frame t either moves to a token of
    # frame t+gap at cost (1 - cos) + lam * distance, or is destroyed at cost tau; a token of frame t+gap not
    # reached is created at cost tau (TV-penalised unbalanced OT with uniform masses)
    # -> sigma [P] (destination, -1 = destroyed), created [P] bool, d [P] (cells, 0 for destroyed)
    # Exact and sparse: a move costing >= 2 tau is never better than destroy + create, so only cheaper moves are
    # edges (radius < 2 tau / lam cells); the dummy-dummy block only needs the transpose of that pattern.
    # Same plans and cost as the dense [2P, 2P] linear_sum_assignment, ~100x faster.
    P = sim.shape[0]
    pos = torch.stack([torch.arange(P) // Wp, torch.arange(P) % Wp], -1).double()  # [P, 2] (row, col)
    dist = torch.cdist(pos, pos)  # [P, P]
    C = ((1 - sim) + lam * dist).numpy()  # [P, P]
    i, j = np.nonzero(C < 2 * tau)
    eps = 1e-9  # sparse matrices drop explicit zeros, and every cost must stay an edge
    ar = np.arange(P)
    rows = np.concatenate([i, ar, P + j, P + ar])
    cols = np.concatenate([j, P + ar, P + i, ar])
    vals = np.concatenate([C[i, j] + eps, np.full(P, tau), np.full(len(i), eps), np.full(P, tau)])
    _, cols_of = min_weight_full_bipartite_matching(sp.csr_matrix((vals, (rows, cols)), shape=(2 * P, 2 * P)))  # [2P]
    col_of = cols_of[:P]  # [P] column of each frame-t token
    sigma = torch.from_numpy(np.where(col_of < P, col_of, -1))  # [P]
    created = torch.zeros(P, dtype=torch.bool)
    created[torch.from_numpy(cols_of[P:][cols_of[P:] < P])] = True  # frame t+gap tokens fed by a dummy source
    d = torch.where(sigma >= 0, dist[torch.arange(P), sigma.clamp_min(0)], torch.zeros(P, dtype=torch.double))  # [P]
    return sigma, created, d


def uot_motion(sigma, d, Wp, Hp):
    # predicted mover displacement: median arrow (drow, dcol) over the largest 8-connected group of transported
    # tokens (d > 0), grouped by their source cell -> (drow, dcol) in cells, mask [Hp, Wp] of that group
    moved = (d > 0).reshape(Hp, Wp).numpy()
    lab, n = ndimage.label(moved, structure=np.ones((3, 3)))
    if n == 0:
        return (0.0, 0.0), np.zeros((Hp, Wp), dtype=bool)
    big = lab == (1 + int(np.argmax(ndimage.sum(moved, lab, range(1, n + 1)))))  # [Hp, Wp]
    src = torch.from_numpy(np.flatnonzero(big))
    dr = (sigma[src] // Wp - src // Wp).double(); dc = (sigma[src] % Wp - src % Wp).double()
    return (float(dr.median()), float(dc.median())), big


def gt_motion(X, t, gap, block, patch, cell_thresh=0.08, pad=2, span=160, step=4, screen_tol=0.02, fg_thresh=60):
    # pixel ground truth for the mover, independent of the tokens. Background B = per-pixel temporal median of the
    # frames within +-span of t (same test block, every `step`) that show the same screen (mean |f_k - f_t| outside
    # the moving blob < screen_tol); the mover moves around, so it drops out of the median. Mover mask at t and t+gap
    # = pixels inside the padded box of the largest changed blob whose summed RGB |f - B| > fg_thresh; motion =
    # shift of the mask centroid.
    # -> (dy, dx) in pixels (float; None if no usable background), blob mask [Hp, Wp], info dict
    pair = load_pair(X, t, gap)
    ch = cell_diff(pair, patch) > cell_thresh  # [Hp, Wp]
    lab, n = ndimage.label(ch, structure=np.ones((3, 3)))
    if n == 0:
        return None, ch, dict(reason='no change')
    blob = lab == (1 + int(np.argmax(ndimage.sum(ch, lab, range(1, n + 1)))))
    rr, cc = np.where(blob)
    Hp, Wp = ch.shape
    r0, r1 = max(0, rr.min() - pad), min(Hp, rr.max() + 1 + pad)
    c0, c1 = max(0, cc.min() - pad), min(Wp, cc.max() + 1 + pad)
    box_px = np.zeros(pair.shape[1:3], dtype=bool); box_px[r0 * patch:r1 * patch, c0 * patch:c1 * patch] = True
    s0, s1 = block
    ks = [k for k in range(max(s0, t - span), min(s1, t + gap + span + 1), step) if not (t - 8 <= k <= t + gap + 8)]
    f_t = pair[0].astype(np.float32)
    same = [k for k in ks if np.abs(X[k].astype(np.float32) - f_t)[~box_px].mean() / 255 < screen_tol]
    if len(same) < 8:
        return None, blob, dict(reason=f'only {len(same)} same-screen frames')
    B = np.median(np.stack([X[k] for k in same]).astype(np.float32), axis=0)  # [H, W, C]
    cents = []
    for f in pair.astype(np.float32):
        fg = (np.abs(f - B).sum(-1) > fg_thresh) & box_px  # [H, W]
        if fg.sum() < 8:
            return None, blob, dict(reason='empty mover mask')
        ys, xs = np.nonzero(fg)
        R, G, Bc = f[fg].T
        green = float(((G > R + 20) & (G > Bc + 20) & (G > 80)).mean())  # Link's tunic / cap
        cents.append((ys.mean(), xs.mean(), int(fg.sum()), int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1), green))
    dy, dx = cents[1][0] - cents[0][0], cents[1][1] - cents[0][1]
    return (float(dy), float(dx)), blob, dict(n_bg=len(same), centroid_t=cents[0][:2], mask_px=[cents[0][2], cents[1][2]],
                                              mask_hw=[cents[0][3:5], cents[1][3:5]], green=min(cents[0][5], cents[1][5]))


def pixel_label(X, t, gap, block, patch, cell_thresh=0.08, screen_tol=0.02, gt_min_px=2.0,
                fade_tol=0.01, max_sprite_px=28, mask_area=(30, 450), max_motion_px=10.0, min_green=0.0):
    # pixel-level action label of the transition (t, t+gap), no tokens involved. Status:
    #   'still'    nothing changed -> action none
    #   'fade'     global brightness change (screen fade / flash) -> no label
    #   'scroll'   the camera moved (mean |diff| outside the dilated moving blob > screen_tol) -> no label
    #   'text'     the changed region is a dialogue box (mostly near-black with white glyphs) -> no label
    #   'no_bg'    no static background estimate (too few same-screen frames, empty mask) -> no label
    #   'not_link' mover mask not sprite-sized (> max_sprite_px a side or area outside mask_area), e.g. animated water
    #   'not_green' < min_green of the mover mask is Link-green in one of the frames (text, lights, thrown objects)
    #   'implausible' |motion| > max_motion_px in one transition (Link walks ~3 px per 4 frames at 128 px)
    #   'labelled' mover motion from gt_motion -> 5-way direction
    pair = load_pair(X, t, gap)
    f0, f1 = pair.astype(np.float32)
    ch = cell_diff(pair, patch) > cell_thresh  # [Hp, Wp]
    if not ch.any():
        return dict(t=t, status='still', dir='none', dy=0.0, dx=0.0, n_blobs=0)
    if abs(f1.mean() - f0.mean()) / 255 > fade_tol:
        return dict(t=t, status='fade', dir=None, n_blobs=0)
    n_blobs = ndimage.label(ch, structure=np.ones((3, 3)))[1]
    near = np.kron(ndimage.binary_dilation(ch, structure=np.ones((3, 3)), iterations=2), np.ones((patch, patch), dtype=bool))
    if (~near).mean() < 0.25:  # change everywhere: screen transition or scroll
        return dict(t=t, status='scroll', dir=None, n_blobs=int(n_blobs))
    outside = np.abs(pair[1].astype(np.float32) - pair[0].astype(np.float32))[~near].mean() / 255
    if outside > screen_tol:
        return dict(t=t, status='scroll', dir=None, n_blobs=int(n_blobs))
    box = np.kron(ndimage.binary_dilation(ch, structure=np.ones((3, 3))), np.ones((patch, patch), dtype=bool))  # [H, W]
    for f in (f0, f1):
        dark = (f[box] < 40).all(-1).mean(); white = (f[box] > 200).all(-1).mean()
        if dark > 0.4 and white > 0.03:
            return dict(t=t, status='text', dir=None, n_blobs=int(n_blobs))
    g, _, info = gt_motion(X, t, gap, block, patch, cell_thresh=cell_thresh, screen_tol=screen_tol)
    if g is None:
        return dict(t=t, status='no_bg', dir=None, n_blobs=int(n_blobs), reason=info.get('reason'))
    if any(max(hw) > max_sprite_px for hw in info['mask_hw']) or not all(mask_area[0] <= a <= mask_area[1] for a in info['mask_px']):
        return dict(t=t, status='not_link', dir=None, n_blobs=int(n_blobs), mask_hw=info['mask_hw'], mask_px=info['mask_px'])
    if info['green'] < min_green:
        return dict(t=t, status='not_green', dir=None, n_blobs=int(n_blobs), green=info['green'])
    if np.hypot(*g) > max_motion_px:
        return dict(t=t, status='implausible', dir=None, n_blobs=int(n_blobs), dy=g[0], dx=g[1])
    return dict(t=t, status='labelled', dir=direction(*g, gt_min_px), dy=g[0], dx=g[1], n_blobs=int(n_blobs),
                centroid=[float(v) for v in info['centroid_t']])


def direction(dy, dx, min_mag):
    # 5-way action label from a displacement: none if both components are under min_mag, else the dominant axis
    if max(abs(dy), abs(dx)) < min_mag:
        return 'none'
    if abs(dy) >= abs(dx):
        return 'down' if dy > 0 else 'up'
    return 'right' if dx > 0 else 'left'


# ----------------------------------------------------------------------------- drawing helpers
def show_frame(ax, img_u8, Hp, Wp, grid=True, cmap=None):
    # image drawn in cell units: cell (row, col) spans [col, col+1] x [row, row+1]
    ax.imshow(img_u8, extent=(0, Wp, Hp, 0), interpolation='nearest', cmap=cmap)
    if grid:
        for k in range(1, Wp):
            ax.axvline(k, color='white', lw=0.3, alpha=0.45)
        for k in range(1, Hp):
            ax.axhline(k, color='white', lw=0.3, alpha=0.45)
    ax.set_xticks(np.arange(0, Wp, 4) + 0.5, [str(c) for c in range(0, Wp, 4)], fontsize=6)
    ax.set_yticks(np.arange(0, Hp, 4) + 0.5, [str(r) for r in range(0, Hp, 4)], fontsize=6)
    ax.set_xlim(0, Wp); ax.set_ylim(Hp, 0)


def box(ax, p, Wp, color, lw=2.0):
    r, c = divmod(int(p), Wp)
    ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=color, lw=lw))


# ----------------------------------------------------------------------------- subcommands
def local_change_pairs(frames, s, b, gap, patch, cell_thresh, min_cells, max_cells, max_extent):
    # pairs (t, t+gap) of one test block whose changed cells are few and packed in a small box (no scroll, not static)
    # frames: uint8 [N, H, W, C] of the block starting at local index s -> [(block, t, n_changed, bbox_h, bbox_w)]
    ok = []
    for t in range(0, len(frames) - gap):
        ch = cell_diff(frames[[t, t + gap]], patch) > cell_thresh  # [Hp, Wp]
        n = int(ch.sum())
        if not (min_cells <= n <= max_cells):
            continue
        rows, cols = np.where(ch)
        bh, bw = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
        if bh <= max_extent and bw <= max_extent:
            ok.append((b, s + t, n, int(bh), int(bw)))
    return ok


def cmd_candidates(args):
    h5 = h5py.File(args.test_h5, 'r')
    X = h5['frames']
    blocks = test_blocks(h5)
    picks = []  # (block, t, n_changed, bbox_h, bbox_w)
    for b, (s, e) in enumerate(blocks):
        ok = local_change_pairs(X[s:e], s, b, args.gap, args.patch, args.cell_thresh, args.min_cells, args.max_cells, args.max_extent)
        # up to `per_block` picks at the interior quantiles of the valid pairs, at least `min_spacing` frames apart
        chosen = []
        for q in (np.arange(args.per_block) + 0.5) / args.per_block if ok else []:
            cand = ok[int(q * len(ok))]
            if all(abs(cand[1] - c[1]) >= args.min_spacing for c in chosen):
                chosen.append(cand)
        print(f'block {b}: {len(ok)} valid pairs, picked {[c[1] for c in chosen]}')
        picks += chosen
    picks = picks[:args.max_candidates]

    # contact sheet: each candidate = frame t | frame t+gap | |diff|, 2 candidates per row
    per_row = 2
    n_rows = (len(picks) + per_row - 1) // per_row
    fig, axes = plt.subplots(n_rows, 3 * per_row, figsize=(3 * per_row * 2.2, n_rows * 2.4), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for k, (b, t, n, bh, bw) in enumerate(picks):
        pair = load_pair(X, t, args.gap)
        r, c0 = divmod(k, per_row)
        diff = np.abs(pair[1].astype(np.float32) - pair[0].astype(np.float32)).mean(-1)
        for j, (img, lab) in enumerate([(pair[0], f't = {t}'), (pair[1], f't+{args.gap}'), (diff, '|diff|')]):
            ax = axes[r, 3 * c0 + j]
            ax.imshow(img, interpolation='nearest', cmap='magma' if j == 2 else None)
            ax.set_title(lab if j else f'#{t}  (block {b}, {n} cells)', fontsize=8)
    fig.suptitle(f'STA-35 candidate pairs (t, t+{args.gap}) from {os.path.basename(args.test_h5)}: '
                 f'changed cells (> {args.cell_thresh}) in [{args.min_cells}, {args.max_cells}], extent <= {args.max_extent} cells',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'candidates.png')
    fig.savefig(out, dpi=110); plt.close(fig)
    with open(os.path.join(args.out_dir, 'candidates.json'), 'w') as fh:
        json.dump([dict(block=b, t=t, n_changed=n, extent=[bh, bw]) for b, t, n, bh, bw in picks], fh, indent=1)
    print(f'{len(picks)} candidates -> {out}')


def fig_grid(X, ts, gap, patch, out):
    # AC1: per pair, frame t | frame t+gap | |diff|, all on the Hp x Wp token grid
    H, W = X.shape[1:3]
    Hp, Wp = H // patch, W // patch
    fig, axes = plt.subplots(len(ts), 3, figsize=(3 * 4.2, len(ts) * 4.2), squeeze=False)
    for r, t in enumerate(ts):
        pair = load_pair(X, t, gap)
        diff = np.abs(pair[1].astype(np.float32) - pair[0].astype(np.float32)).mean(-1)
        show_frame(axes[r, 0], pair[0], Hp, Wp); axes[r, 0].set_title(f'frame t = {t}', fontsize=9)
        show_frame(axes[r, 1], pair[1], Hp, Wp); axes[r, 1].set_title(f'frame t+{gap} = {t + gap}', fontsize=9)
        show_frame(axes[r, 2], diff, Hp, Wp, cmap='magma'); axes[r, 2].set_title('|frame t+gap - frame t|', fontsize=9)
        axes[r, 0].set_ylabel('token row', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('token col', fontsize=8)
    fig.suptitle(f'{Hp}x{Wp} token grid (patch {patch}px); flat token index p = {Wp}*row + col', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=100); plt.close(fig)


def fig_similarity(pair, sim, f, d, kind, t, gap, Wp, Hp, out, win=None):
    # AC2: cosine matrix | frame t with i* | frame t+gap with f(i*) | sim row of i* over frame t+gap
    # win: optional ((r0, c0), (top, left), size, mask) diff window; the matrix panel then shows only the window tokens
    i_star = int(torch.argmax(d))  # first index among equal maxima
    j_star = int(f[i_star])
    ri, ci = divmod(i_star, Wp); rj, cj = divmod(j_star, Wp)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1, 1.15]))

    ax = axes[0]
    if win is None:
        im = ax.imshow(sim.numpy(), cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        ticks = np.arange(0, Hp * Wp, Wp * 4)
        ax.set_xticks(ticks, [f'r{k // Wp}' for k in ticks], fontsize=6); ax.set_yticks(ticks, [f'r{k // Wp}' for k in ticks], fontsize=6)
        ax.set_xticks(np.arange(0, Hp * Wp, Wp), minor=True); ax.set_yticks(np.arange(0, Hp * Wp, Wp), minor=True)
        ax.tick_params(which='minor', length=2)
        ax.set_title(f'cosine similarity [{Hp * Wp} x {Hp * Wp}]', fontsize=9)
    else:
        (r0, c0), (top, left), size, mask = win
        idx = torch.nonzero(mask).squeeze(1)  # [size*size] flat indices, row-major
        im = ax.imshow(sim[idx][:, idx].numpy(), cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        ticks = np.arange(0, size * size, size * 2)
        ax.set_xticks(ticks, [f'r{top + k // size}' for k in ticks], fontsize=6); ax.set_yticks(ticks, [f'r{top + k // size}' for k in ticks], fontsize=6)
        ax.set_xticks(np.arange(0, size * size, size), minor=True); ax.set_yticks(np.arange(0, size * size, size), minor=True)
        ax.tick_params(which='minor', length=2)
        ax.set_title(f'cosine similarity inside the window [{size * size} x {size * size}]', fontsize=9)
    ax.set_xlabel(f'token p of frame t+{gap}', fontsize=8); ax.set_ylabel('token p of frame t', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    if win is not None:
        for a in axes[1:]:
            a.add_patch(Rectangle((left, top), size, size, fill=False, edgecolor='yellow', lw=1.5, ls='--'))
        axes[1].plot(c0 + 0.5, r0 + 0.5, marker='x', color='yellow', ms=8, mew=2)
    show_frame(axes[1], pair[0], Hp, Wp); box(axes[1], i_star, Wp, 'red')
    axes[1].set_title(f'frame t = {t}: i* = ({ri}, {ci})  p={i_star}', fontsize=9)
    show_frame(axes[2], pair[1], Hp, Wp); box(axes[2], j_star, Wp, 'red')
    axes[2].set_title(f'frame t+{gap}: f(i*) = ({rj}, {cj})  p={j_star}', fontsize=9)
    fig.add_artist(ConnectionPatch(xyA=(ci + 0.5, ri + 0.5), coordsA=axes[1].transData,
                                   xyB=(cj + 0.5, rj + 0.5), coordsB=axes[2].transData,
                                   arrowstyle='->', color='red', lw=1.5))

    ax = axes[3]
    show_frame(ax, pair[1], Hp, Wp, grid=False)
    im = ax.imshow(sim[i_star].reshape(Hp, Wp).numpy(), extent=(0, Wp, Hp, 0), cmap='RdBu_r', vmin=-1, vmax=1,
                   alpha=0.6, interpolation='nearest')
    box(ax, i_star, Wp, 'black', lw=1.2); box(ax, j_star, Wp, 'red')
    ax.set_title(f'sim(i*, j) over frame t+{gap} (black = pos of i*, red = f(i*))', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    n_max = int((d == d.max()).sum())
    fig.suptitle(f'{kind} tokens, pair t = {t}: token that moved most d(i*) = {float(d[i_star]):.2f} cells'
                 + (f'  ({n_max} tokens tie at this d)' if n_max > 1 else '')
                 + (f'  |  search restricted to the {win[2]}x{win[2]} window around the max |diff| pixel (yellow x)' if win else ''),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=100); plt.close(fig)
    return dict(i_star=i_star, i_star_rc=[ri, ci], f_i_star=j_star, f_i_star_rc=[rj, cj], d=float(d[i_star]), n_tied_max=n_max)


def cmd_figures(args):
    h5 = h5py.File(args.test_h5, 'r')
    X = h5['frames']
    ts = [int(s) for s in args.pairs.split(',')]
    ac2 = [int(s) for s in args.ac2.split(',')] if args.ac2 else []
    os.makedirs(args.out_dir, exist_ok=True)

    fig_grid(X, ts, args.gap, args.patch, os.path.join(args.out_dir, 'ac1_grid.png'))
    print(f'AC1 -> {args.out_dir}/ac1_grid.png')
    if not ac2:
        return

    device = torch.device(args.device)
    tok, _ = load_videotokenizer_from_checkpoint(args.tokenizer, device)
    tok.eval()
    Hp, Wp = X.shape[1] // args.patch, X.shape[2] // args.patch
    summary = {}
    suffix = f'_win{args.window}' if args.window else ''
    for t in ac2:
        pair = load_pair(X, t, args.gap)
        tokens = encode_tokens(tok, pair, device)
        win = None
        if args.window:
            (r0, c0), (top, left), mask = diff_window(pair, args.patch, args.window)
            win = ((r0, c0), (top, left), args.window, mask)
        for kind in ('fsq', 'hidden'):
            sim = cosine_matrix(tokens[kind][0], tokens[kind][1])  # [P, P]
            f, d = match_and_displacement(sim, Wp, mask=win[3] if win else None)
            out = os.path.join(args.out_dir, f'ac2_t{t}_{kind}{suffix}.png')
            res = fig_similarity(pair, sim, f, d, kind, t, args.gap, Wp, Hp, out, win=win)
            res['n_moved'] = int((d > 0).sum())
            if win:
                res['diff_max_cell'] = [r0, c0]; res['window_top_left'] = [top, left]
            summary[f'{t}_{kind}'] = res
            print(f't={t} {kind}: i*={res["i_star_rc"]} -> f(i*)={res["f_i_star_rc"]} d={res["d"]:.2f} '
                  f'({res["n_moved"]} tokens with d>0, {res["n_tied_max"]} tied at max) -> {out}')
    with open(os.path.join(args.out_dir, f'ac2_summary{suffix}.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)


def cmd_ot(args):
    # OT plan for each pair and each distance weight lam: rows = pairs, cols = lam; frame t with an arrow from every
    # transported token to its destination (static tokens drawn nothing), changed cells outlined in cyan
    h5 = h5py.File(args.test_h5, 'r')
    X = h5['frames']
    ts = [int(s) for s in args.pairs.split(',')]
    lams = [float(s) for s in args.lams.split(',')]
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    tok, _ = load_videotokenizer_from_checkpoint(args.tokenizer, device)
    tok.eval()
    Hp, Wp = X.shape[1] // args.patch, X.shape[2] // args.patch
    summary = {}
    pairs = {t: load_pair(X, t, args.gap) for t in ts}
    tokens = {t: encode_tokens(tok, pairs[t], device) for t in ts}
    for kind in ('fsq', 'hidden'):
        fig, axes = plt.subplots(len(ts), len(lams), figsize=(len(lams) * 3.6, len(ts) * 3.75), squeeze=False)
        for r, t in enumerate(ts):
            changed = cell_diff(pairs[t], args.patch) > args.cell_thresh  # [Hp, Wp]
            sim = cosine_matrix(tokens[t][kind][0], tokens[t][kind][1])  # [P, P]
            for c, lam in enumerate(lams):
                sigma, d, cost = ot_plan(sim, Wp, lam)
                moved = d > 0  # [P]
                src = torch.nonzero(moved).squeeze(1)
                ri, ci = src // Wp, src % Wp
                rj, cj = sigma[src] // Wp, sigma[src] % Wp
                ch = torch.from_numpy(changed.reshape(-1))
                n_mov = int(moved.sum())
                prec = float((moved & ch).sum()) / max(n_mov, 1)  # moved tokens that sit on changed cells
                rec = float((moved & ch).sum()) / max(int(ch.sum()), 1)  # changed cells whose token moved
                ax = axes[r, c]
                show_frame(ax, pairs[t][0], Hp, Wp, grid=False)
                ax.contour(np.arange(Wp) + 0.5, np.arange(Hp) + 0.5, changed.astype(float), levels=[0.5],
                           colors='cyan', linewidths=0.8)
                if n_mov:
                    q = ax.quiver((ci + 0.5).numpy(), (ri + 0.5).numpy(), (cj - ci).numpy(), (rj - ri).numpy(),
                                  d[src].numpy(), angles='xy', scale_units='xy', scale=1, cmap='autumn',
                                  width=0.006, headwidth=3.5, headlength=4, clim=(0, 8))
                ax.set_title(f't={t}  lam={lam:g}\nmoved {n_mov}  prec {prec:.2f}  rec {rec:.2f}', fontsize=8)
                summary[f'{kind}_{t}_{lam:g}'] = dict(n_moved=n_mov, precision=prec, recall=rec,
                                                      mean_d_moved=float(d[moved].mean()) if n_mov else 0.0, cost=cost)
        fig.suptitle(f'{kind} tokens: exact OT plan (permutation) with cost (1 - cos) + lam * distance [cells]; '
                     f'arrows = token of frame t -> destination in frame t+{args.gap} (colour = distance, 0..8 cells); '
                     f'cyan = changed cells (|diff| > {args.cell_thresh})', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = os.path.join(args.out_dir, f'ot_{kind}.png')
        fig.savefig(out, dpi=90); plt.close(fig)
        print(f'{kind} -> {out}')
    with open(os.path.join(args.out_dir, 'ot_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    for k, v in summary.items():
        print(f"{k:22s} moved {v['n_moved']:4d}  prec {v['precision']:.2f}  rec {v['recall']:.2f}  mean d {v['mean_d_moved']:.2f}")


def static_activity(sigma, created, d, changed, P):
    # fraction of static tokens (cells outside the changed region dilated by 1) that the plan touches:
    # transported away, destroyed, or created
    static = ~torch.from_numpy(ndimage.binary_dilation(changed, structure=np.ones((3, 3))).reshape(-1))  # [P]
    active = (d > 0) | (sigma < 0) | created  # [P]
    return float((active & static).sum()) / max(int(static.sum()), 1)


def cmd_uot_calib(args):
    # calibrate (lam, tau) of the unbalanced OT plan against the pixel ground truth of gt_motion, on static-camera
    # pairs of the whole test split that are not near the display pairs
    h5 = h5py.File(args.test_h5, 'r')
    X = h5['frames'][:]  # uint8 [N, H, W, C], in memory (~360 MB for zelda)
    blocks = test_blocks(h5)
    held = [int(v) for v in args.exclude.split(',')] if args.exclude else []
    Hp, Wp = X.shape[1] // args.patch, X.shape[2] // args.patch
    P = Hp * Wp

    pool = []
    for b, (s, e) in enumerate(blocks):
        ok = local_change_pairs(X[s:e], s, b, args.gap, args.patch, args.cell_thresh, 4, 60, 10)
        pool += [c for c in ok if all(abs(c[1] - h) > 16 for h in held)]
    pool = pool[::max(1, len(pool) // (4 * args.n_calib))]  # thin before the (slower) ground-truth pass
    calib, last = [], -10 ** 9
    for (b, t, *_) in pool:
        if t - last < args.min_spacing:
            continue
        g, blob, info = gt_motion(X, t, args.gap, blocks[b], args.patch, cell_thresh=args.cell_thresh)
        if g is not None:
            calib.append(dict(block=b, t=t, gt=g, gt_dir=direction(*g, args.gt_min_px), changed=cell_diff(load_pair(X, t, args.gap), args.patch) > args.cell_thresh))
            last = t
    calib = [calib[int(k)] for k in np.linspace(0, len(calib) - 1, min(len(calib), args.n_calib))]
    gt_dirs = [c['gt_dir'] for c in calib]
    print(f'{len(calib)} calibration pairs with ground truth, blocks {sorted(set(c["block"] for c in calib))}, '
          f'gt directions {dict((k, gt_dirs.count(k)) for k in sorted(set(gt_dirs)))}')

    device = torch.device(args.device)
    tok, _ = load_videotokenizer_from_checkpoint(args.tokenizer, device)
    tok.eval()
    lams = [float(v) for v in args.lams.split(',')]
    taus = [float(v) for v in args.taus.split(',')]
    sims = {kind: [] for kind in ('fsq', 'hidden')}
    for c in calib:
        tk = encode_tokens(tok, load_pair(X, c['t'], args.gap), device)
        for kind in sims:
            sims[kind].append(cosine_matrix(tk[kind][0], tk[kind][1]))  # [P, P]

    results = {}
    for kind in sims:
        # resolvable = gt motion >= one cell (a token method cannot see sub-cell motion); still = gt < gt_min_px
        big = np.array([np.hypot(*c['gt']) >= args.patch for c in calib])
        still = np.array([c['gt_dir'] == 'none' for c in calib])
        acc = np.zeros((len(lams), len(taus))); epe = np.zeros_like(acc); act = np.zeros_like(acc)
        acc_big = np.zeros_like(acc); none_still = np.zeros_like(acc)
        for a, lam in enumerate(lams):
            for bt, tau in enumerate(taus):
                hits, errs, acts = [], [], []
                for c, sim in zip(calib, sims[kind]):
                    sigma, created, d = uot_plan(sim, Wp, lam, tau)
                    (dr, dc), _ = uot_motion(sigma, d, Wp, Hp)
                    pred_px = (dr * args.patch, dc * args.patch)
                    hits.append(direction(*pred_px, args.gt_min_px) == c['gt_dir'])
                    errs.append(float(np.hypot(pred_px[0] - c['gt'][0], pred_px[1] - c['gt'][1])))
                    acts.append(static_activity(sigma, created, d, c['changed'], P))
                hits = np.array(hits)
                acc[a, bt], epe[a, bt], act[a, bt] = hits.mean(), np.mean(errs), np.mean(acts)
                acc_big[a, bt], none_still[a, bt] = hits[big].mean(), hits[still].mean()
            print(f'{kind} lam={lam:g}: acc>=1cell {np.round(acc_big[a], 2).tolist()}  none|still {np.round(none_still[a], 2).tolist()}  '
                  f'all {np.round(acc[a], 2).tolist()}  static activity {np.round(act[a] * 100, 2).tolist()} %')
        # selection: balanced score (direction accuracy on >= 1-cell moves, "none" rate on still pairs) among settings
        # that leave the background alone, ties -> lowest EPE
        bal = (acc_big + none_still) / 2
        score = np.where(act <= args.max_static_activity, bal - 1e-3 * epe, -np.inf)
        a, bt = np.unravel_index(int(np.argmax(score)), score.shape)
        results[kind] = dict(lam=lams[a], tau=taus[bt], balanced=float(bal[a, bt]), acc_big=float(acc_big[a, bt]),
                             none_still=float(none_still[a, bt]), acc=float(acc[a, bt]), epe_px=float(epe[a, bt]),
                             static_activity=float(act[a, bt]), n_big=int(big.sum()), n_still=int(still.sum()),
                             grid=dict(lams=lams, taus=taus, balanced=bal.tolist(), acc_big=acc_big.tolist(),
                                       none_still=none_still.tolist(), acc=acc.tolist(), epe_px=epe.tolist(),
                                       static_activity=act.tolist()))
        print(f'{kind}: selected lam={lams[a]:g} tau={taus[bt]:g} balanced {bal[a, bt]:.2f} (acc>=1cell {acc_big[a, bt]:.2f} '
              f'on {big.sum()}, none|still {none_still[a, bt]:.2f} on {still.sum()}) all {acc[a, bt]:.2f} '
              f'epe {epe[a, bt]:.1f}px static activity {act[a, bt] * 100:.2f}%')

    # heatmaps: rows = kind, cols = metric
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for r, kind in enumerate(sims):
        g = results[kind]['grid']
        for c, (key, title, cmap, fmt) in enumerate([('balanced', 'balanced score (selection)', 'viridis', '{:.2f}'),
                                                      ('acc_big', f'direction acc, moves >= 1 cell (n={results[kind]["n_big"]})', 'viridis', '{:.2f}'),
                                                      ('none_still', f'"none" rate, still pairs (n={results[kind]["n_still"]})', 'viridis', '{:.2f}'),
                                                      ('static_activity', 'static tokens touched [%]', 'magma_r', '{:.2f}')]):
            M = np.array(g[key]) * (100 if key == 'static_activity' else 1)
            ax = axes[r, c]
            ax.imshow(M, cmap=cmap, aspect='auto')
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center', fontsize=7, color='white')
            ax.set_xticks(range(len(taus)), [f'{v:g}' for v in taus]); ax.set_yticks(range(len(lams)), [f'{v:g}' for v in lams])
            ax.set_xlabel('tau (create / destroy cost)'); ax.set_ylabel('lam (cost per cell)')
            sel = results[kind]
            ax.add_patch(Rectangle((taus.index(sel['tau']) - 0.5, lams.index(sel['lam']) - 0.5), 1, 1, fill=False, edgecolor='red', lw=2))
            ax.set_title(f'{kind}: {title}', fontsize=10)
    fig.suptitle(f'Unbalanced OT calibration on {len(calib)} static-camera test pairs (gap {args.gap}); '
                 f'red = selected (best balanced score with static activity <= {args.max_static_activity * 100:g}%); '
                 f'"none" if |motion| < {args.gt_min_px:g}px', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, 'uot_calibration.png'), dpi=100); plt.close(fig)
    results['calibration_pairs'] = [dict(block=c['block'], t=c['t'], gt_px=c['gt'], gt_dir=c['gt_dir']) for c in calib]
    with open(os.path.join(args.out_dir, 'uot_calibration.json'), 'w') as fh:
        json.dump(results, fh, indent=1)
    print(f'-> {args.out_dir}/uot_calibration.{{png,json}}')


def cmd_uot(args):
    # unbalanced OT plans on the display pairs at the calibrated (lam, tau) and at tau / 2, tau * 2 around it:
    # arrows = transported tokens, x = destroyed (frame t), o = created (frame t+gap), blue = predicted motion
    # (median arrow of the largest moving group), green = pixel ground truth when the camera is static
    h5 = h5py.File(args.test_h5, 'r')
    X = h5['frames']
    blocks = test_blocks(h5)
    ts = [int(v) for v in args.pairs.split(',')]
    cal = json.load(open(os.path.join(args.out_dir, 'uot_calibration.json')))
    device = torch.device(args.device)
    tok, _ = load_videotokenizer_from_checkpoint(args.tokenizer, device)
    tok.eval()
    Hp, Wp = X.shape[1] // args.patch, X.shape[2] // args.patch
    summary = {}
    for kind in ('fsq', 'hidden'):
        lam, tau = cal[kind]['lam'], cal[kind]['tau']
        settings = [(lam, tau / 2), (lam, tau), (lam, tau * 2)]
        fig, axes = plt.subplots(len(ts), len(settings), figsize=(len(settings) * 4.6, len(ts) * 4.7), squeeze=False)
        for r, t in enumerate(ts):
            pair = load_pair(X, t, args.gap)
            b = next(i for i, (s0, s1) in enumerate(blocks) if s0 <= t < s1)
            g, _, _ = gt_motion(X, t, args.gap, blocks[b], args.patch, cell_thresh=args.cell_thresh)
            changed = cell_diff(pair, args.patch) > args.cell_thresh
            tk = encode_tokens(tok, pair, device)
            sim = cosine_matrix(tk[kind][0], tk[kind][1])  # [P, P]
            for c, (lm, ta) in enumerate(settings):
                sigma, created, d = uot_plan(sim, Wp, lm, ta)
                (dr, dc), group = uot_motion(sigma, d, Wp, Hp)
                ax = axes[r, c]
                show_frame(ax, pair[0], Hp, Wp, grid=False)
                ax.contour(np.arange(Wp) + 0.5, np.arange(Hp) + 0.5, changed.astype(float), levels=[0.5], colors='cyan', linewidths=0.7)
                src = torch.nonzero(d > 0).squeeze(1)
                if len(src):
                    ax.quiver((src % Wp + 0.5).numpy(), (src // Wp + 0.5).numpy(), (sigma[src] % Wp - src % Wp).numpy(),
                              (sigma[src] // Wp - src // Wp).numpy(), angles='xy', scale_units='xy', scale=1,
                              color='orange', width=0.006, headwidth=3.5, headlength=4)
                dst = torch.nonzero(sigma < 0).squeeze(1)
                ax.plot((dst % Wp + 0.5).numpy(), (dst // Wp + 0.5).numpy(), 'x', color='red', ms=4, mew=1.2)
                crt = torch.nonzero(created).squeeze(1)
                ax.plot((crt % Wp + 0.5).numpy(), (crt // Wp + 0.5).numpy(), 'o', mfc='none', mec='white', ms=4, mew=1)
                if group.any():
                    gy, gx = np.argwhere(group).mean(0) + 0.5
                    ax.annotate('', xy=(gx + 3 * dc, gy + 3 * dr), xytext=(gx, gy),
                                arrowprops=dict(arrowstyle='-|>', color='deepskyblue', lw=2.5))
                    if g is not None:
                        ax.annotate('', xy=(gx + 3 * g[1] / args.patch, gy + 3 * g[0] / args.patch), xytext=(gx, gy),
                                    arrowprops=dict(arrowstyle='-|>', color='lime', lw=2, ls='--'))
                pred_dir = direction(dr * args.patch, dc * args.patch, args.gt_min_px)
                gt_dir = direction(*g, args.gt_min_px) if g is not None else 'n/a (camera moves)'
                ax.set_title(f't={t} lam={lm:g} tau={ta:g}{"  [calibrated]" if c == 1 else ""}\n'
                             f'moved {int((d > 0).sum())} destroyed {len(dst)} created {len(crt)}\n'
                             f'pred {pred_dir} ({dr:+.0f},{dc:+.0f}) cells | gt {gt_dir}', fontsize=8)
                summary[f'{kind}_{t}_{lm:g}_{ta:g}'] = dict(pred_cells=[dr, dc], pred_dir=pred_dir, gt_px=g, gt_dir=gt_dir,
                                                         n_moved=int((d > 0).sum()), n_destroyed=len(dst), n_created=len(crt))
        fig.suptitle(f'{kind} tokens, unbalanced OT: cost (1 - cos) + lam * distance, create/destroy cost tau. '
                     f'orange = transported, red x = destroyed, white o = created, cyan = changed cells; '
                     f'blue = predicted motion (x3), green dashed = pixel ground truth (x3)', fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        out = os.path.join(args.out_dir, f'uot_{kind}.png')
        fig.savefig(out, dpi=90); plt.close(fig)
        print(f'{kind} (lam={lam:g}, tau={tau:g}) -> {out}')
    with open(os.path.join(args.out_dir, 'uot_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    for k, v in summary.items():
        print(f"{k:28s} pred {v['pred_dir']:5s} {v['pred_cells']}  gt {v['gt_dir']}  moved {v['n_moved']} destroyed {v['n_destroyed']} created {v['n_created']}")


STATUSES = ('still', 'labelled', 'fade', 'scroll', 'text', 'no_bg', 'not_link', 'not_green', 'implausible')


def cmd_pixel_labels(args):
    # run pixel_label on every transition (t, t+gap) of the test split, t every `stride` frames inside each block;
    # report coverage and the label distribution per block, and draw a stratified check sheet of labelled pairs
    h5 = h5py.File(args.test_h5, 'r')
    X = h5['frames'][:]  # uint8 [N, H, W, C]
    blocks = test_blocks(h5)
    rows = []
    for b, (s0, s1) in enumerate(blocks):
        for t in range(s0, s1 - args.gap, args.stride):
            r = pixel_label(X, t, args.gap, (s0, s1), args.patch, args.cell_thresh, args.screen_tol, args.gt_min_px)
            r['block'] = b
            rows.append(r)
        sub = [r for r in rows if r['block'] == b]
        stat = {k: sum(r['status'] == k for r in sub) for k in STATUSES}
        dirs = {k: sum(r.get('dir') == k for r in sub) for k in ('none', 'up', 'down', 'left', 'right')}
        print(f'block {b}: {len(sub)} transitions  {stat}  dirs {dirs}')
    n = len(rows)
    stat = {k: sum(r['status'] == k for r in rows) for k in STATUSES}
    dirs = {k: sum(r.get('dir') == k for r in rows) for k in ('none', 'up', 'down', 'left', 'right')}
    multi = sum(r['status'] == 'labelled' and r['n_blobs'] > 1 for r in rows)
    print(f'total {n}: ' + ', '.join(f'{k} {v} ({100 * v / n:.0f}%)' for k, v in stat.items()))
    print(f'labels (still + labelled = {stat["still"] + stat["labelled"]}): {dirs}; labelled with > 1 moving blob: {multi}')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'pixel_labels.json'), 'w') as fh:
        json.dump(dict(gap=args.gap, stride=args.stride, status=stat, dirs=dirs, multi_blob=multi, rows=rows), fh, indent=1)

    # check sheet: up to `per_dir` labelled transitions per direction, spread over the split; crop around the mover
    rng = np.random.default_rng(0)
    picks = []
    for k in ('up', 'down', 'left', 'right', 'none'):
        cand = [r for r in rows if r['status'] == 'labelled' and r['dir'] == k]
        picks += [cand[i] for i in sorted(rng.choice(len(cand), min(args.per_dir, len(cand)), replace=False))]
    fig, axes = plt.subplots(len(picks) // 4 + (len(picks) % 4 > 0), 8, figsize=(8 * 2.3, (len(picks) // 4 + 1) * 2.5), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for i, r in enumerate(picks):
        pair = load_pair(X, r['t'], args.gap)
        cy, cx = r['centroid']
        for j in range(2):
            ax = axes[i // 4, 2 * (i % 4) + j]
            ax.imshow(pair[j], interpolation='nearest')
            ax.set_xlim(cx - 20, cx + 20); ax.set_ylim(cy + 20, cy - 20)
            if j == 0:
                ax.annotate('', xy=(cx + 3 * r['dx'], cy + 3 * r['dy']), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle='-|>', color='lime', lw=2))
                ax.set_title(f'#{r["t"]} {r["dir"]} ({r["dy"]:+.1f},{r["dx"]:+.1f})px' + (' multi' if r['n_blobs'] > 1 else ''), fontsize=8)
            else:
                ax.set_title(f't+{args.gap}', fontsize=8)
    fig.suptitle(f'pixel labeller check sheet: {len(picks)} labelled transitions (gap {args.gap}), '
                 f'arrow = mover centroid shift x3 on frame t; crop 40x40 px around the mover', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(args.out_dir, 'pixel_labels_check.png'), dpi=90); plt.close(fig)
    print(f'-> {args.out_dir}/pixel_labels.json, pixel_labels_check.png')


def cmd_ot_plans(args):
    # calibrated unbalanced OT plan (hidden tokens) for every pair (t, t + gap) of an .h5, as LAM inputs:
    # sigma [N, P] int16 (destination of token i of frame t, -1 = destroyed), created [N, P] bool (token of frame
    # t + gap fed by nothing); the last `gap` frames get the identity plan
    X = h5py.File(args.h5, 'r')['frames']
    N, H, W = X.shape[:3]
    Hp, Wp = H // args.patch, W // args.patch
    P = Hp * Wp
    device = torch.device(args.device)
    tok, _ = load_videotokenizer_from_checkpoint(args.tokenizer, device)
    tok.eval()
    sigma = np.tile(np.arange(P, dtype=np.int16), (N, 1))  # [N, P]
    created = np.zeros((N, P), dtype=bool)
    n_moved = np.zeros(N, dtype=np.int32)
    for s in tqdm(range(0, N - args.gap, args.chunk)):
        e = min(s + args.chunk, N - args.gap)
        with torch.no_grad():
            x = torch.from_numpy(X[s:e + args.gap]).to(device).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1  # [n, C, H, W]
            h = tok.encoder.transformer(tok.encoder.patch_embed(x.unsqueeze(1)))[:, 0]  # [n, P, E]
            h = (h / h.norm(dim=-1, keepdim=True)).cpu().double()
        for t in range(s, e):
            sg, cr, d = uot_plan(h[t - s] @ h[t - s + args.gap].T, Wp, args.lam, args.tau)
            sigma[t], created[t], n_moved[t] = sg.numpy(), cr.numpy(), int((d > 0).sum())
    np.savez(args.out, sigma=sigma, created=created, lam=args.lam, tau=args.tau, gap=args.gap, patch=args.patch,
             tokenizer=args.tokenizer, h5=args.h5)
    destroyed = (sigma < 0).sum(1)
    print(f'{args.out}: {N} pairs; per pair mean moved {n_moved.mean():.1f}, destroyed {destroyed.mean():.1f}, '
          f'created {created.sum(1).mean():.1f}; all-static pairs {np.mean((n_moved == 0) & (destroyed == 0)):.2f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--test-h5', default='data/zelda_test_frames.h5')
    p.add_argument('--out-dir', default='eval_results/sta35_patch_similarity')
    p.add_argument('--gap', type=int, default=4, help='stored frames between the two frames (= frame_skip in training)')
    p.add_argument('--patch', type=int, default=4, help='tokenizer patch size (cell size of the grid)')
    sub = p.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('candidates')
    c.add_argument('--cell-thresh', type=float, default=0.08, help='mean |diff| in [0,1] above which a cell counts as changed')
    c.add_argument('--min-cells', type=int, default=4)
    c.add_argument('--max-cells', type=int, default=60)
    c.add_argument('--max-extent', type=int, default=10, help='max bbox height/width of the changed cells, in cells')
    c.add_argument('--per-block', type=int, default=2)
    c.add_argument('--min-spacing', type=int, default=60)
    c.add_argument('--max-candidates', type=int, default=20)

    fg = sub.add_parser('figures')
    fg.add_argument('--pairs', required=True, help='comma-separated local indices t of the AC1 pairs')
    fg.add_argument('--ac2', default='', help='comma-separated subset of --pairs for the similarity figures')
    fg.add_argument('--window', type=int, default=0,
                    help='if > 0: restrict i and f(i) to a window x window cell square centred on the max |diff| pixel '
                         '(12 = 3 Link heights of ~4 cells at 128px)')
    fg.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    fg.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')

    o = sub.add_parser('ot')
    o.add_argument('--pairs', required=True, help='comma-separated local indices t')
    o.add_argument('--lams', default='0,0.01,0.03,0.1,0.3,1', help='distance weights (cost per cell)')
    o.add_argument('--cell-thresh', type=float, default=0.08)
    o.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    o.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')

    u = sub.add_parser('uot-calib')
    u.add_argument('--exclude', default='78,1050,2539,4000,6222,2979', help='display pairs kept out of calibration')
    u.add_argument('--n-calib', type=int, default=48)
    u.add_argument('--min-spacing', type=int, default=24, help='min frames between calibration pairs')
    u.add_argument('--lams', default='0.01,0.02,0.03,0.05,0.1,0.2')
    u.add_argument('--taus', default='0.05,0.1,0.2,0.3,0.5,1')
    u.add_argument('--max-static-activity', type=float, default=0.01)
    for q in (u, sub.add_parser('uot')):
        q.add_argument('--cell-thresh', type=float, default=0.08)
        q.add_argument('--gt-min-px', type=float, default=2.0, help='motion under this (px, both axes) counts as "none"')
        q.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
        q.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
        if q is not u:
            q.add_argument('--pairs', required=True)

    pl = sub.add_parser('pixel-labels')
    pl.add_argument('--stride', type=int, default=4, help='frames between consecutive labelled transitions')
    pl.add_argument('--cell-thresh', type=float, default=0.08)
    pl.add_argument('--screen-tol', type=float, default=0.02)
    pl.add_argument('--gt-min-px', type=float, default=2.0)
    pl.add_argument('--per-dir', type=int, default=4)

    op = sub.add_parser('ot-plans')
    op.add_argument('--h5', required=True)
    op.add_argument('--out', required=True, help='.npz path')
    op.add_argument('--lam', type=float, default=0.3, help='calibrated on hidden tokens (uot-calib)')
    op.add_argument('--tau', type=float, default=0.5)
    op.add_argument('--chunk', type=int, default=256)
    op.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    op.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')

    args = p.parse_args()
    {'ot-plans': cmd_ot_plans, 'pixel-labels': cmd_pixel_labels, 'candidates': cmd_candidates, 'figures': cmd_figures, 'ot': cmd_ot,
     'uot-calib': cmd_uot_calib, 'uot': cmd_uot}[args.cmd](args)


if __name__ == '__main__':
    main()
