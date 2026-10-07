"""STA-35: Identifiable Token Correspondence (ITC) between two consecutive frames (Sonic or Zelda).

ITC (Articles/Identifiable Token Correspondence for World Models) decides, for every position j of the next frame,
whether to reuse a token u_i of the previous frame (moved from position i) or to keep the candidate token generated
for j. Here the candidates are the tokenizer's own tokens of frame t+1 (no dynamics model), so the plan reads as
"which tokens of frame t survive into frame t+1, and from where".

Candidates: frame t+1 encoded alone (T=1); the soft candidate distribution p_j over the num_bins**latent_dim FSQ
codebook is the product over latent dims of softmax_k(-(b_d - k)^2 / temp), with b_d the bounded pre-rounding latent
in [0, num_bins - 1] (its argmax is the FSQ code). Previous tokens u_i: the FSQ codes of frame t (one-hot).

Paper, eq. 1-2 and Algorithms 1-2:
    A_prev[i, j] = <p_j, u_i> - c_d * D(pos_i, pos_j)          (L x L, D = Euclidean distance in token cells)
    A_gen[k, j]  = ||p_j||_inf - c_w  if k == j, else -inf      (L x L)
    A = [[A_prev, 0], [A_gen, 0]]  (2L x 2L, the zero columns absorb unused rows)
    P = Sinkhorn(-A, eps, n_iter) with unit row and column marginals; binarize -> Pi_prev, Pi_gen
    s_hat[j] = u_i if Pi_prev[i, j] = 1, else candidate j (Pi_gen[j, j] = 1)
Binarization (Algorithm 2) runs on the full 2L x 2L plan: on the 2L x L slice it cannot terminate (2L rows compete
for L columns, so some row always loses). Entries with -inf affinity (zero mass) are masked out of the binarization,
so Pi_gen stays diagonal. The binarized plan is checked against an exact assignment (Hungarian) on the same P.

Variant `--affinity ratio` (not in the paper): A_prev uses p_j(u_i) / ||p_j||_inf and A_gen = 1 - c_w. This tokenizer's
FSQ latents sit near bin boundaries and its codes depend on the whole frame, so ~85% of code changes between two
frames happen in cells whose pixels did not change; the ratio scores a boundary flip as (almost) the same token.

Usage (from the repo root, PYTHONPATH=$PWD):
    python experiments/itc_actions/itc_correspondence.py --pairs test:3657,train:6307,train:33983 --out-dir eval_results/sta35_itc/paper
    python experiments/itc_actions/itc_correspondence.py --affinity ratio --temp 0.1 --c-w 0.5 --out-dir eval_results/sta35_itc/ratio
    python experiments/itc_actions/itc_correspondence.py --dataset zelda --out-dir eval_results/sta35_itc/zelda/paper
    python experiments/itc_actions/itc_correspondence.py --dataset zelda --affinity ratio --temp 0.1 --c-w 0.5 --out-dir eval_results/sta35_itc/zelda/ratio
Outputs: <out_dir>/itc_<split>_<t>.png per pair, <out_dir>/summary.json.
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
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from utils.utils import load_videotokenizer_from_checkpoint

# per dataset: tokenizer checkpoint, default pairs (static camera, the player moves > 1 token between t and t+1)
DATASETS = {
    'sonic': ('../sta-25-analyse-v3-training-on-wandb-and-launch-longer-training-if/results/'
              'sonic_v4_bigtok_2026_10_01/attempt2/video_tokenizer/checkpoints/video_tokenizer_step_37000',
              'test:3657,train:6307,train:33983'),
    'zelda': ('../test/results/zelda_v3_2026_09_26/video_tokenizer/checkpoints/video_tokenizer_step_29000',
              'test:3840,test:77,test:279'),
}
DEFAULT_TOKENIZER = DATASETS['sonic'][0]


# ----------------------------------------------------------------------------- tokens
@torch.no_grad()
def encode(tok, pair_u8, device, temp):
    # each frame encoded alone (T=1); pair_u8: uint8 [2, H, W, C]
    # -> codes [2, P] (FSQ indices), digit_prob [2, P, L, num_bins] (per-dim soft bin probabilities; one-hot if temp == 0)
    x = torch.from_numpy(pair_u8).to(device).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1  # [2, C, H, W]
    z = tok.encoder(x.unsqueeze(1))[:, 0]  # [2, P, L]
    q = tok.quantizer
    b = q.scale_and_shift(torch.tanh(z))  # [2, P, L] in [0, num_bins - 1]
    codes = q.get_indices_from_latents(q(z), dim=-1)  # [2, P]
    k = torch.arange(q.num_bins, device=device, dtype=b.dtype)  # [num_bins]
    if temp > 0:
        digit_prob = torch.softmax(-(b.unsqueeze(-1) - k) ** 2 / temp, dim=-1)  # [2, P, L, num_bins]
    else:
        digit_prob = (torch.round(b).unsqueeze(-1) == k).to(b.dtype)  # [2, P, L, num_bins]
    return codes.cpu(), digit_prob.cpu().double(), q


def code_digits(codes, q):
    # codes [*] -> digits [*, L] (bin index per latent dim)
    return (codes.unsqueeze(-1) // q.basis.cpu()) % q.num_bins


def prob_of_codes(digit_prob, digits):
    # digit_prob [P, L, nb] (candidates), digits [P', L] (previous tokens) -> [P', P] p_j(u_i) = prod_d p_jd(u_id)
    Pj, Ld, _ = digit_prob.shape
    g = digit_prob.permute(1, 0, 2)  # [L, P, nb]
    out = torch.ones(digits.shape[0], Pj, dtype=digit_prob.dtype)
    for d in range(Ld):
        out = out * g[d][:, digits[:, d]].T  # [P', P]
    return out


# ----------------------------------------------------------------------------- ITC
def affinities(p_prev, p_max, Wp, c_d, c_w):
    # p_prev [L, L] = <p_j, u_i>, p_max [L] = ||p_j||_inf -> A [2L, 2L] (-inf = forbidden), dist [L, L]
    L = p_prev.shape[0]
    pos = torch.stack([torch.arange(L) // Wp, torch.arange(L) % Wp], -1).double()  # [L, 2] (row, col)
    dist = torch.cdist(pos, pos)  # [L, L]
    A_prev = p_prev - c_d * dist  # [L, L]
    A_gen = torch.full((L, L), float('-inf'), dtype=torch.double)
    A_gen[torch.arange(L), torch.arange(L)] = p_max - c_w  # [L, L]
    A = torch.cat([torch.cat([A_prev, torch.zeros(L, L, dtype=torch.double)], 1),
                   torch.cat([A_gen, torch.zeros(L, L, dtype=torch.double)], 1)], 0)  # [2L, 2L]
    return A, dist


def sinkhorn(A, eps, n_iter):
    # entropic OT with cost -A and unit marginals on the square [2L, 2L] matrix, in the log domain -> P [2L, 2L]
    logK = A / eps  # [2L, 2L], -inf stays -inf
    f = torch.zeros(A.shape[0], dtype=A.dtype); g = torch.zeros(A.shape[1], dtype=A.dtype)
    for _ in range(n_iter):
        f = -torch.logsumexp(logK + g[None, :], dim=1)
        g = -torch.logsumexp(logK + f[:, None], dim=0)
    return torch.exp(logK + f[:, None] + g[None, :])


def binarize(P, allowed, v=1e6, max_iter=100000):
    # Algorithm 2 on a square plan P [n, n]; allowed [n, n] bool (False = -inf affinity, never selected)
    # -> Pi [n, n] permutation matrix, number of iterations
    P = np.where(allowed, P, -np.inf)
    n, m = P.shape
    rows = np.arange(n)
    for it in range(max_iter):
        target = P.argmax(1)  # [n]
        init = np.zeros((n, m), bool); init[rows, target] = True
        C = np.where(init, P, -v)  # P * Pi_init - v (1 - Pi_init)
        source = C.argmax(0)  # [m]
        out = np.zeros((n, m), bool); out[source, np.arange(m)] = True
        out &= init
        R = init & ~out  # rows that targeted a column won by another row
        if not R.any():
            return out, it + 1
        P = P - v * R
    raise RuntimeError('binarization did not converge')


# ----------------------------------------------------------------------------- ground truth (pixels, for annotation)
def player_gt(X, t, win=12, thresh=80, blue_only=True):
    # foreground of frames t and t+1 against the median of a +-win frame window (camera assumed static). Player = the
    # largest foreground component (holding Sonic-blue pixels if blue_only) in frame t, then the one nearest to it in t+1
    # -> centroid (row, col) px of frame t and t+1 (None if not found), masks [2, H, W]
    lo, hi = max(0, t - win), min(len(X), t + win + 2)
    W = X[lo:hi].astype(np.int16)
    same_cam = (np.abs(W - X[t].astype(np.int16)).sum(-1) > 60).mean((1, 2)) < 0.05  # frames sharing frame t's camera
    bg = np.median(W[same_cam], 0)
    cen, m, prev = [], [], None
    for k in (0, 1):
        fr = X[t + k].astype(np.int16)
        fg = np.abs(fr - bg).sum(-1) > thresh
        blue = fg & (fr[..., 2] > 120) & (fr[..., 2] > fr[..., 0] + 40)
        lab, _ = ndimage.label(ndimage.binary_dilation(fg, iterations=1))
        comps = [c for c in np.setdiff1d(np.unique(lab[blue if blue_only else fg]), [0])]
        if not comps:
            cen.append(None); m.append(np.zeros_like(fg)); continue
        cs = [np.array(ndimage.center_of_mass(fg & (lab == c))) for c in comps]
        if prev is None:
            pick = int(np.argmax([(fg & (lab == c)).sum() for c in comps]))
        else:
            dists = [np.hypot(*(c - prev)) for c in cs]
            pick = int(np.argmin(dists))
            if dists[pick] > 3 * win:
                cen.append(None); m.append(np.zeros_like(fg)); continue
        prev = cs[pick]; cen.append(cs[pick]); m.append(fg & (lab == comps[pick]))
    return cen, np.stack(m)


# ----------------------------------------------------------------------------- figure
def grid(ax, Hp, Wp, patch, alpha=0.25):
    for v in range(Hp + 1):
        ax.axhline(v * patch - 0.5, color='w', lw=0.3, alpha=alpha)
    for v in range(Wp + 1):
        ax.axvline(v * patch - 0.5, color='w', lw=0.3, alpha=alpha)


def dir_color(dr, dc):
    # hue = direction of the move (right = red, down = green-ish, ...), full saturation
    h = (np.arctan2(dr, dc) / (2 * np.pi)) % 1.0
    return hsv_to_rgb(np.stack([h, np.ones_like(h), np.ones_like(h)], -1))


def draw_moves(ax, img, src, dst, Wp, patch, Hp, zoom=None):
    ax.imshow(img, interpolation='nearest'); grid(ax, Hp, Wp, patch)
    if len(src):
        r0, c0 = src // Wp, src % Wp; r1, c1 = dst // Wp, dst % Wp
        cols = dir_color(r1 - r0, c1 - c0)
        ax.quiver((c0 + .5) * patch - .5, (r0 + .5) * patch - .5, (c1 - c0) * patch, (r1 - r0) * patch, color=cols,
                  angles='xy', scale_units='xy', scale=1, width=0.008 if zoom is None else 0.012, headwidth=3,
                  headlength=3.5)
    if zoom is not None:
        (y0, y1), (x0, x1) = zoom
        ax.set_xlim(x0 - .5, x1 - .5); ax.set_ylim(y1 - .5, y0 - .5)


def figure(out, title, pair, codes, A, P, Pi, Hp, Wp, patch, gt, info):
    # 2 x 5 panels: frames, diff, s_hat composition, moves | zooms, move histogram, affinity, plan
    L = Hp * Wp
    reuse_src = np.full(L, -1)  # [L] for each position j of s_hat: source i of the reused token, -1 = candidate kept
    ii, jj = np.nonzero(Pi[:L, :L]); reuse_src[jj] = ii
    kept = reuse_src < 0
    stayed = (~kept) & (reuse_src == np.arange(L))
    moved = (~kept) & ~stayed
    dst = np.nonzero(moved)[0]; src = reuse_src[dst]

    fig, axs = plt.subplots(2, 5, figsize=(22, 10))
    ax = [axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3], axs[0, 4], axs[1, 0], axs[1, 1], None, None, axs[1, 2]]
    axA, axP = axs[1, 3], axs[1, 4]

    ax[0].imshow(pair[0], interpolation='nearest'); grid(ax[0], Hp, Wp, patch); ax[0].set_title('frame t (previous tokens u)')
    ax[1].imshow(pair[1], interpolation='nearest'); grid(ax[1], Hp, Wp, patch); ax[1].set_title('frame t+1 (candidates s~)')
    diff = np.abs(pair[1].astype(int) - pair[0].astype(int)).sum(-1)
    ax[2].imshow(diff, cmap='magma', interpolation='nearest'); grid(ax[2], Hp, Wp, patch)
    same = (codes[0] == codes[1]).numpy().reshape(Hp, Wp)
    ax[2].contour(np.kron(~same, np.ones((patch, patch))), levels=[0.5], colors='cyan', linewidths=0.6)
    ax[2].set_title(f'|pixel diff| + pixel player move (white)\ncyan = code changed in place ({int((~same).sum())}/{L} cells)')
    if gt[0][0] is not None and gt[0][1] is not None:
        for a in ax[:3]:
            a.plot(gt[0][0][1], gt[0][0][0], '+', color='white', ms=8, mew=1.2)
            a.annotate('', xy=(gt[0][1][1], gt[0][1][0]), xytext=(gt[0][0][1], gt[0][0][0]),
                       arrowprops=dict(arrowstyle='->', color='white', lw=1.2))

    # composition of s_hat (paper colors): blue = reused from frame t, green = candidate kept
    comp = np.zeros((Hp, Wp, 4))
    comp[..., :3] = np.where(kept.reshape(Hp, Wp)[..., None], [0.2, 0.85, 0.2], [0.3, 0.4, 1.0])
    comp[..., 3] = np.where(kept.reshape(Hp, Wp), 0.75, np.where(moved.reshape(Hp, Wp), 0.75, 0.25))
    ax[3].imshow(pair[1], interpolation='nearest')
    ax[3].imshow(np.kron(comp, np.ones((patch, patch, 1))), interpolation='nearest'); grid(ax[3], Hp, Wp, patch)
    mv = moved.reshape(Hp, Wp)
    ax[3].contour(np.kron(mv, np.ones((patch, patch))), levels=[0.5], colors='yellow', linewidths=1.0)
    ax[3].set_title(f's^ on frame t+1: blue = reused from frame t\n({int(stayed.sum())} in place, yellow outline = '
                    f'{int(moved.sum())} moved), green = candidate kept ({int(kept.sum())})')

    draw_moves(ax[4], pair[0], src, dst, Wp, patch, Hp)
    ax[4].set_title(f'moved reused tokens on frame t\narrow i -> j, hue = direction ({len(src)} arrows)')

    # zoom on the gt mover (or on the moved tokens)
    if gt[0][0] is not None:
        cy, cx = gt[0][0]
    elif len(src):
        cy, cx = (src // Wp).mean() * patch, (src % Wp).mean() * patch
    else:
        cy, cx = 32, 32
    h = 12 * pair.shape[1] // 64  # zoom half-size in px
    ms = 10 * 64 / pair.shape[1]
    y0 = int(np.clip(cy - h, 0, pair.shape[1] - 2 * h)); x0 = int(np.clip(cx - h, 0, pair.shape[2] - 2 * h))
    zoom = ((y0, y0 + 2 * h), (x0, x0 + 2 * h))
    draw_moves(ax[5], pair[0], src, dst, Wp, patch, Hp, zoom=zoom)
    k = np.nonzero(kept)[0]
    ax[5].plot((k % Wp + .5) * patch - .5, (k // Wp + .5) * patch - .5, 's', mfc='none', mec='lime', ms=ms, mew=1.2)
    ax[5].set_title('zoom, frame t: arrows = moved tokens,\ngreen squares = positions given a new candidate')
    ax[6].imshow(pair[1], interpolation='nearest'); grid(ax[6], Hp, Wp, patch, alpha=.5)
    ax[6].plot((k % Wp + .5) * patch - .5, (k // Wp + .5) * patch - .5, 's', mfc='none', mec='lime', ms=ms, mew=1.2)
    ax[6].plot((dst % Wp + .5) * patch - .5, (dst // Wp + .5) * patch - .5, 'o', mfc='none', mec='yellow', ms=0.7 * ms, mew=1.2)
    ax[6].set_xlim(zoom[1][0] - .5, zoom[1][1] - .5); ax[6].set_ylim(zoom[0][1] - .5, zoom[0][0] - .5)
    ax[6].set_title('zoom, frame t+1: green = candidate kept,\nyellow = moved token landed here')
    # displacement histogram of the moved tokens
    if len(src):
        dr = dst // Wp - src // Wp; dc = dst % Wp - src % Wp
        vals, cnt = np.unique(np.stack([dr, dc], 1), axis=0, return_counts=True)
        lab = [f'({a:+d},{b:+d})' for a, b in vals]
        ax[9].barh(range(len(lab)), cnt, color=dir_color(vals[:, 0], vals[:, 1]))
        ax[9].set_yticks(range(len(lab))); ax[9].set_yticklabels(lab, fontsize=8)
    ax[9].set_title('moves of reused tokens\n(drow, dcol) in cells, count'); ax[9].set_xlabel('tokens')

    A_show = A[:, :L].copy(); A_show[np.isinf(A_show)] = np.nan
    im = axA.imshow(A_show, aspect='auto', cmap='gray_r', interpolation='nearest')
    axA.axhline(L - .5, color='r', lw=.8); axA.set_title('affinity A (rows: u then s~; cols: positions j)\nwhite = -inf')
    axA.set_ylabel('rows 0..L-1 prev, L..2L-1 gen'); plt.colorbar(im, ax=axA, fraction=.04)
    axP.imshow(P[:, :L], aspect='auto', cmap='gray_r', interpolation='nearest', vmax=1)
    pi, pj = np.nonzero(Pi[:, :L]); axP.plot(pj, pi, '.', color='r', ms=1.5)
    axP.axhline(L - .5, color='b', lw=.8); axP.set_title('Sinkhorn plan P (gray) and\nbinarized plan (red dots)')
    for a in ax[:7]:
        a.set_xticks([]); a.set_yticks([])
    for a in fig.axes:
        a.title.set_fontsize(9)
    fig.suptitle(title + '\n' + info, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return reuse_src, kept, stayed, moved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=list(DATASETS), default='sonic')
    p.add_argument('--pairs', default=None, help='<split>:<t> pairs (t, t+1) of data/<dataset>_<split>_frames.h5')
    p.add_argument('--tokenizer', default=None)
    p.add_argument('--temp', type=float, default=0.0, help='soft-FSQ temperature of the candidate distribution p_j '
                   '(0 = one-hot on the frame t+1 code)')
    p.add_argument('--c-d', type=float, default=0.1, help='distance cost per token cell')
    p.add_argument('--affinity', choices=['paper', 'ratio'], default='paper',
                   help='paper: A_prev = <p_j, u_i>, A_gen = ||p_j||_inf; ratio: both divided by ||p_j||_inf')
    p.add_argument('--c-w', type=float, default=0.3, help='cost of keeping a candidate (c_w / c_d = reuse radius in cells)')
    p.add_argument('--eps', type=float, default=0.02, help='Sinkhorn entropic regularisation')
    p.add_argument('--n-iter', type=int, default=500)
    p.add_argument('--out-dir', default='eval_results/sta35_itc')
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = p.parse_args()
    args.tokenizer = args.tokenizer or DATASETS[args.dataset][0]
    args.pairs = args.pairs or DATASETS[args.dataset][1]

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    tok, _ = load_videotokenizer_from_checkpoint(args.tokenizer, device)
    tok.eval()
    summary = {'args': vars(args), 'pairs': {}}
    for spec in args.pairs.split(','):
        split, t = spec.split(':'); t = int(t)
        X = h5py.File(f'data/{args.dataset}_{split}_frames.h5', 'r')['frames']
        pair = np.stack([X[t], X[t + 1]])  # uint8 [2, H, W, C]
        patch = tok.encoder.patch_embed.patch_size if hasattr(tok.encoder.patch_embed, 'patch_size') else 4
        Hp, Wp = pair.shape[1] // patch, pair.shape[2] // patch
        L = Hp * Wp

        codes, digit_prob, q = encode(tok, pair, device, args.temp)  # [2, L], [2, L, Ld, nb]
        u_digits = code_digits(codes[0], q)  # [L, Ld]
        p_prev = prob_of_codes(digit_prob[1], u_digits)  # [L(i), L(j)] = <p_j, u_i>
        p_max = digit_prob[1].max(-1).values.prod(-1)  # [L] = ||p_j||_inf
        if args.affinity == 'ratio':  # likelihood ratio p_j(u_i) / max_c p_j(c): a code flipped across a bin boundary ~ 1
            p_prev, p_max = p_prev / p_max[None, :], torch.ones_like(p_max)
        A, dist = affinities(p_prev, p_max, Wp, args.c_d, args.c_w)  # [2L, 2L]
        P = sinkhorn(A, args.eps, args.n_iter).numpy()  # [2L, 2L]
        allowed = ~np.isinf(A.numpy())
        Pi, n_bin = binarize(P, allowed)
        r, c = linear_sum_assignment(np.where(allowed, P, -1e9), maximize=True)
        hung = np.zeros_like(Pi); hung[r, c] = True
        marg = (np.abs(P.sum(0) - 1).max(), np.abs(P.sum(1) - 1).max())
        assert not Pi[L:, :L][~np.eye(L, dtype=bool)].any(), 'gen block must stay diagonal'

        gt = player_gt(X, t, blue_only=args.dataset == 'sonic')
        gt_move = None if gt[0][0] is None or gt[0][1] is None else (gt[0][1] - gt[0][0]).round(1).tolist()
        info = (f'affinity {args.affinity}, temp {args.temp}, c_d {args.c_d}, c_w {args.c_w}, eps {args.eps}, {args.n_iter} Sinkhorn iters '
                f'(marginal err {max(marg):.1e}); binarization {n_bin} iters, agrees with Hungarian on '
                f'{int((Pi[:, :L] & hung[:, :L]).sum())}/{L} columns; pixel gt player move (drow, dcol) px = {gt_move}')
        out = os.path.join(args.out_dir, f'itc_{split}_{t}.png')
        reuse_src, kept, stayed, moved = figure(out, f'ITC on {args.dataset} {split} frames {t} -> {t + 1}', pair, codes, A.numpy(),
                                                P, Pi, Hp, Wp, patch, gt, info)
        dst = np.nonzero(moved)[0]; src = reuse_src[dst]
        moves = [{'from': [int(s // Wp), int(s % Wp)], 'to': [int(d // Wp), int(d % Wp)]} for s, d in zip(src, dst)]
        kept_cells = [[int(j // Wp), int(j % Wp)] for j in np.nonzero(kept)[0]]
        rec = {'reused_in_place': int(stayed.sum()), 'reused_moved': int(moved.sum()), 'candidates_kept': int(kept.sum()),
               'codes_changed_in_place': int((codes[0] != codes[1]).sum()), 'moves': moves, 'kept_cells': kept_cells,
               'gt_player_frame_t_px': None if gt[0][0] is None else gt[0][0].round(1).tolist(), 'gt_move_px': gt_move,
               'hungarian_agree': int((Pi[:, :L] & hung[:, :L]).sum()), 'binarize_iters': n_bin}
        summary['pairs'][spec] = rec
        print(f'{spec}: in place {rec["reused_in_place"]}, moved {rec["reused_moved"]}, kept {rec["candidates_kept"]}, '
              f'codes changed {rec["codes_changed_in_place"]}, gt move {gt_move}, gt at {rec["gt_player_frame_t_px"]}')
        for m in moves:
            print('   move', m['from'], '->', m['to'])
        print('   kept', kept_cells)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
