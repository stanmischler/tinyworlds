"""The judge-labelled held-out Zelda action set (scripts/eval/lam_judge.py builds and scores it): label vocabulary,
window protocol, LAM -> code helpers, NMI/purity metrics against the consensus labels and per-code example grids."""

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from evaluation.action_metrics import kmeans, label_metrics
from evaluation.windows import load_history_batch, to_model_range

LABELS = ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR', 'STILL', 'ATTACK', 'NONCONTROL', 'UNSURE']
MOVES = ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR']
H5 = 'data/zelda_test_frames.h5'
SEQ, SKIP, STRIDE = 4, 4, 8  # LAM window: 4 frames, 4 stored frames apart; window starts every 8 stored frames
OUT = 'eval_results/lam_judge'
SET = f'{OUT}/set'


def font_draw(img):
    return ImageDraw.Draw(img)


def transition_panel(fr_a, fr_b, scale=2, tag=''):
    # fr_a, fr_b: uint8 [H, W, C] -> PIL image: frame t | frame t+1 | amplified |change| (magma-free grayscale-hot)
    d = np.clip(np.abs(fr_b.astype(np.int16) - fr_a.astype(np.int16)).max(-1) * 3, 0, 255).astype(np.uint8)  # [H, W]
    heat = np.stack([d, (d.astype(np.float32) * 0.6).astype(np.uint8), np.zeros_like(d)], -1)  # orange on black
    H, W = d.shape
    gap, top = 6, 16
    img = Image.new('RGB', (3 * W * scale + 2 * gap, H * scale + top), 'white')
    for k, a in enumerate([fr_a, fr_b, heat]):
        img.paste(Image.fromarray(a).resize((W * scale, H * scale), Image.NEAREST), (k * (W * scale + gap), top))
    font_draw(img).text((2, 2), tag, fill='black')
    return img


def stack_rows(rows, title):
    W = max(r.width for r in rows)
    img = Image.new('RGB', (W, sum(r.height + 8 for r in rows) + 20), 'white')
    font_draw(img).text((4, 4), title, fill='black')
    y = 20
    for r in rows:
        img.paste(r, (0, y))
        y += r.height + 8
    return img


def lam_codes(ckpt, wins, device, batch=32, k=0):
    # -> codes [N, T-1], number of codes, continuous actions [N, T-1, A] (None for a discrete LAM)
    from utils.utils import load_latent_actions_from_checkpoint
    lam, _ = load_latent_actions_from_checkpoint(ckpt, device)
    lam.eval()
    if getattr(lam, 'continuous_actions', False):
        zq = encode_windows(lam, wins, device, batch).float()
        k = k or lam.quantizer.codebook_size  # --k overrides the number of k-means clusters
        c = kmeans(zq.reshape(-1, lam.action_dim), k)
        return torch.cdist(zq.reshape(-1, lam.action_dim), c).argmin(1).reshape(zq.shape[:-1]).cpu().numpy(), k, zq.cpu()
    return (*codes_from_lam(lam, wins, device, batch), None)


def encode_windows(lam, wins, device, batch=32):
    # lam in eval mode -> actions [N, T-1, A] for every held-out window
    with h5py.File(H5, 'r') as h5, torch.no_grad():
        h = getattr(lam, 'history', 0)  # temporal-tokenizer CoMo: extra earlier frames per window (STA-43)
        return torch.cat([lam.encode(to_model_range(load_history_batch(h5['frames'], wins[i:i + batch], SEQ - 1, SKIP, h, H5), device))
                          for i in range(0, len(wins), batch)])


def codes_from_lam(lam, wins, device, batch=32):
    # lam in eval mode -> codes [N, T-1] for every held-out window, n_codes
    q, A = lam.quantizer, lam.action_dim
    zq = encode_windows(lam, wins, device, batch)  # [N, T-1, A]
    if getattr(lam, 'continuous_actions', False):
        c = kmeans(zq.reshape(-1, A).float(), q.codebook_size)
        return torch.cdist(zq.reshape(-1, A).float(), c).argmin(1).reshape(zq.shape[:-1]).cpu().numpy(), q.codebook_size  # [N, T-1]
    return q.get_indices_from_latents(zq).cpu().numpy(), q.codebook_size


def probe(z, labels, folds=5, l2=1e-2, seed=0):
    """Cross-validated linear probe: does the continuous action predict the judge label, clusters aside? (score()
    feeds the top --probe-pcs principal components of all held-out actions: 144 labels cannot fit 128 raw dims)
    z [n, A] float, labels list[str] -> held-out accuracy of a standardised multinomial logistic regression (L-BFGS,
    L2 `l2`), stratified-free `folds`-fold split (seeded), and the majority-class accuracy on the same folds."""
    classes = sorted(set(labels))
    y = torch.tensor([classes.index(l) for l in labels])
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    correct = majority = 0
    for f in range(folds):
        te = perm[f::folds]
        tr = perm[torch.isin(perm, te, invert=True)]
        mu, sd = z[tr].mean(0), z[tr].std(0) + 1e-6
        xtr, xte = (z[tr] - mu) / sd, (z[te] - mu) / sd
        W = torch.zeros(z.shape[1], len(classes), requires_grad=True)
        b = torch.zeros(len(classes), requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(xtr @ W + b, y[tr]) + l2 * W.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            correct += int(((xte @ W + b).argmax(1) == y[te]).sum())
        majority += int((y[te] == torch.bincount(y[tr], minlength=len(classes)).argmax()).sum())
    return {'acc': round(correct / len(y), 4), 'majority': round(majority / len(y), 4), 'n': len(y)}


def judged_metrics(meta, labels, n_codes, codes_all=None, code_of_id=None):
    # codes_all [N windows, T-1] (or code_of_id {"<transition id>": code}) -> metrics on the judged set
    ids, codes, labs = [], [], []
    for tr in meta['transitions']:
        i = str(tr['id'])
        if i not in labels:
            continue
        ids.append(i)
        codes.append(int(codes_all[tr['window'], tr['t']]) if codes_all is not None else int(code_of_id[i]))
        labs.append(labels[i])
    return {'n_codes': n_codes, 'all': label_metrics(codes, labs, n_codes, LABELS),
            'moves_only': label_metrics([c for c, l in zip(codes, labs) if l in MOVES], [l for l in labs if l in MOVES], n_codes, MOVES),
            'per_transition': dict(zip(ids, codes))}


def code_grids(name, codes_all, wins, n_codes, out_dir, per_code=6):
    # codes_all: [N, T-1] -> one PNG per code with up to `per_code` random held-out transitions mapped to it
    rng = np.random.default_rng(0)
    flat = [(w, t, int(codes_all[w, t])) for w in range(len(wins)) for t in range(codes_all.shape[1])]
    usage = np.bincount([c for _, _, c in flat], minlength=n_codes) / len(flat)
    paths = []
    with h5py.File(H5, 'r') as h5:
        fr = h5['frames']
        for k in range(n_codes):
            cand = [(w, t) for w, t, c in flat if c == k]
            if not cand:
                continue
            sel = [cand[i] for i in rng.choice(len(cand), min(per_code, len(cand)), replace=False)]
            rows = []
            for j, (w, t) in enumerate(sel):
                blk, s = wins[w]
                rows.append(transition_panel(fr[s + t * SKIP], fr[s + (t + 1) * SKIP], tag=f'code {k} example {j} (block {blk})'))
            p = f'{out_dir}/code_{k}.png'
            stack_rows(rows, f'{name}: code {k}, {usage[k]:.1%} of held-out transitions (n={len(cand)}); frame t | frame t+1 | |change| x3').save(p)
            paths.append(p)
    return usage, paths
