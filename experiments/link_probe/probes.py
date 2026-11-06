"""Small probes that read Link's motion or the camera shift from one stage of the action model, on the pairs of labels.py.

Every probe sees a fixed input and is cross-validated by test block, so no clip is on both sides. Hyperparameters are fixed
here before any score is seen; the ridge penalty is picked inside the training blocks only.
  - position_net : for grids of patches or pixels, keeps position (learned position per cell, cross-attention pool)
  - mlp          : for vectors (z, IDM output, averaged MAE features)
  - ridge        : kernel ridge on anything, one-vs-rest least squares for the direction
Direction is trained on windows where it is defined (moved >= MOVED_PX, or every scroll for the camera)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.link_probe.labels import MOVED_PX, direction

CROP_PATCHES = 7    # patches per side around Link, about 64 px on the 128 px frame
CROP_PX = 64
MAE_GRID = 14
LAMBDAS = np.logspace(-4, 4, 9)  # times the mean kernel diagonal
SCALE_PX = 10.0     # displacement scale in the Huber loss
DIRS = 4
NET_STEPS, MLP_STEPS, BATCH = 600, 400, 64
NET_SEEDS, RIDGE_SHUFFLES = 3, 5
MATCHED_N = 77      # camera-still windows with Link moved: the camera is also probed at this count


# ---------------------------------------------------------------- inputs

def crop_box(link_t, grid, size):
    # link_t [N, 2] (y, x) in 128 px frame -> top-left cell [N, 2] of a size x size crop on a grid x grid map, kept inside
    cell = (link_t / 128 * grid).astype(int)
    return np.clip(cell - size // 2, 0, grid - size)


def crop_grid(x, tl, size):
    # x [N, C, H, W], tl [N, 2] -> [N, C, size, size]
    return torch.stack([x[i, :, r:r + size, c:c + size] for i, (r, c) in enumerate(tl.tolist())])


def pixel_pair(fr_a, fr_b):
    # uint8 [N, H, W, C] x2 -> [N, 6, H, W] (frame a, b - a) in [-1, 1]
    a, b = (torch.from_numpy(f).permute(0, 3, 1, 2).float() / 127.5 - 1 for f in (fr_a, fr_b))
    return torch.cat([a, b - a], 1)


def mae_pair(m_a, m_b):
    # float16 [N, S=197, 1024] x2 -> [N, 2048, 14, 14] (features of a, b - a), class token dropped, row-major patches
    a, b = (torch.from_numpy(m[:, 1:]).float() for m in (m_a, m_b))
    x = torch.cat([a, b - a], -1)  # [N, 196, 2048]
    return x.transpose(1, 2).reshape(len(x), 2048, MAE_GRID, MAE_GRID)


def load_input(name, d):
    # d: arrays of the prep step -> ('grid', [N, C, H, W]) or ('vec', [N, D])
    fr, mae, link = d['frames'], d['mae'], d['link_t']
    if name == 'raw pixels':
        return 'grid', pixel_pair(fr[:, 0], fr[:, 2])
    if name == 'raw pixels, last 4 frames only':
        return 'grid', pixel_pair(fr[:, 1], fr[:, 2])
    if name == 'raw pixels around Link':
        return 'grid', crop_grid(pixel_pair(fr[:, 0], fr[:, 2]), crop_box(link, 128, CROP_PX), CROP_PX)
    if name == 'MAE patches':
        return 'grid', mae_pair(mae[:, 0], mae[:, 2])
    if name == 'MAE patches, last 4 frames only':
        return 'grid', mae_pair(mae[:, 1], mae[:, 2])
    if name == 'MAE patches around Link':
        return 'grid', crop_grid(mae_pair(mae[:, 0], mae[:, 2]), crop_box(link, MAE_GRID, CROP_PATCHES), CROP_PATCHES)
    if name == 'MAE averaged':
        return 'vec', mae_pair(mae[:, 0], mae[:, 2]).mean((2, 3))
    run, kind = name.split(' | ')  # e.g. 'z 20 min | two steps'
    key = {'z 20 min': ('z', 'short'), 'z 5 h': ('z', 'long'), 'IDM output 20 min': ('idm', 'short'), 'IDM output 5 h': ('idm', 'long')}[run]
    first, second, direct = (d[f'{key[0]}_{key[1]}_{p}'] for p in ('first', 'second', 'direct'))  # [N, D] pairs t,t+4 / t+4,t+8 / t,t+8
    v = {'two steps': lambda: np.concatenate([first, second], 1), 'last step': lambda: second, 'direct 8 frames': lambda: direct,
         'two steps summed': lambda: first + second}[kind]()
    return 'vec', torch.from_numpy(v.astype(np.float32))


# ---------------------------------------------------------------- targets

def task_targets(set_name, d, seed):
    """Set name -> (idx into the pairs, y [n, 2] px, dir [n], dir_ok [n] bool). Draws of the same-count set depend on seed."""
    still, scroll = d['still'], d['scroll']
    if set_name.startswith('link'):
        idx = {'link, camera still': np.flatnonzero(still), 'link, scroll': np.flatnonzero(scroll),
               'link, all windows': np.arange(len(still))}[set_name]
        y = d['disp'][idx].astype(np.float32)
        return idx, y, direction(y), np.abs(y).max(1) >= MOVED_PX
    idx = np.flatnonzero(scroll)
    if set_name == 'camera, scroll, same count':
        idx = np.sort(np.random.default_rng(1000 + seed).choice(idx, min(MATCHED_N, len(idx)), replace=False))
    y = d['cam'][idx].astype(np.float32)
    return idx, y, direction(y), np.ones(len(idx), bool)


def shuffled(y, dirs, ok, seed):
    p = np.random.default_rng(5000 + seed).permutation(len(y))
    return y[p], dirs[p], ok[p]


# ---------------------------------------------------------------- ridge

def linear_kernels(kind, x):
    # -> kernels(train, test) -> (K_trtr, K_tetr): grids get one fold-centred kernel per block (frame t, change), each with
    # unit mean diagonal, so the small change block is not drowned; vectors are standardised on the training fold
    if kind == 'vec':
        x = x.double()

        def vec(tr, te):
            mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
            a, b = (x[tr] - mu) / sd, (x[te] - mu) / sd
            return (a @ a.T).numpy(), (b @ a.T).numpy()
        return vec
    half = x.shape[1] // 2
    blocks = [x[:, :half].flatten(1), x[:, half:].flatten(1)]
    full = [(b @ b.T).double().numpy() for b in blocks]

    def grid(tr, te):
        Ka, Kb = 0, 0
        for K in full:
            Ktr, Kte = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
            m = Ktr.mean(0, keepdims=True)
            c = Ktr - m - m.T + Ktr.mean()
            s = np.diag(c).mean() + 1e-12
            Ka = Ka + c / s
            Kb = Kb + (Kte - Kte.mean(1, keepdims=True) - m + Ktr.mean()) / s
        return Ka, Kb
    return grid


def solve_ridge(K, Y, lams):
    # K [n, n], Y [n, q] centred -> alphas [L, n, q]
    w, V = np.linalg.eigh(K)
    w = np.clip(w, 0, None)
    return np.stack([V @ ((V.T @ Y) / (w + lam)[:, None]) for lam in lams])


def fit_ridge(K, Kte, Y, g):
    # lambda from leave-one-block-out on the training blocks only; g [n] their blocks. Returns predictions [m, q] for Kte
    mean = Y.mean(0)
    scale = np.diag(K).mean() + 1e-12
    lams = LAMBDAS * scale
    groups = np.unique(g)
    if len(groups) >= 3:
        errs = np.zeros(len(lams))
        for b in groups:
            tr, va = g != b, g == b
            a = solve_ridge(K[np.ix_(tr, tr)], Y[tr] - Y[tr].mean(0), lams)
            pred = K[np.ix_(va, tr)] @ a + Y[tr].mean(0)
            errs += ((pred - Y[va]) ** 2).sum((1, 2))
        lam = lams[int(errs.argmin())]
    else:
        lam = scale
    return Kte @ solve_ridge(K, Y - mean, [lam])[0] + mean


def ridge_oof(kernels, idx, block, y, dirs, ok):
    # out-of-fold predictions on idx by block: displacement [n, 2], direction [n]
    pred, dir_pred = np.zeros_like(y, dtype=np.float64), np.zeros(len(y), int)
    onehot = np.eye(DIRS)[dirs]
    for b in np.unique(block[idx]):
        tr, te = np.flatnonzero(block[idx] != b), np.flatnonzero(block[idx] == b)
        K, Kte = kernels(idx[tr], idx[te])
        pred[te] = fit_ridge(K, Kte, y[tr].astype(np.float64), block[idx][tr])
        m = ok[tr]
        if m.sum() >= DIRS:
            s = fit_ridge(K[np.ix_(m, m)], Kte[:, m], onehot[tr][m], block[idx][tr][m])
            dir_pred[te] = s.argmax(1)
    return pred, dir_pred


# ---------------------------------------------------------------- trained probes

class PositionNet(nn.Module):
    """Grid [B, C, H, W] -> 64-channel map (conv stem for pixels, 1x1 projection for features) -> two convs -> learned position
    per cell -> 4 learned queries cross-attend -> displacement (2) and direction (4). Position survives the pool."""

    def __init__(self, in_ch, hw, is_pixels, dim=64):
        super().__init__()
        if is_pixels:  # 128 -> 16 cells, or 64 -> 8
            self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, 48, 3, 2, 1), nn.GELU(), nn.Conv2d(48, dim, 3, 2, 1))
            hw = hw // 8
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_ch, dim, 1), nn.Dropout2d(0.1))
        self.conv = nn.Sequential(nn.GELU(), nn.Conv2d(dim, dim, 3, 1, 1), nn.GELU(), nn.Conv2d(dim, dim, 3, 1, 1))
        self.pos = nn.Parameter(torch.randn(1, hw * hw, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, 4, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(4 * dim), nn.Linear(4 * dim, 128), nn.GELU(), nn.Dropout(0.1))
        self.disp, self.dir = nn.Linear(128, 2), nn.Linear(128, DIRS)

    def forward(self, x):
        x = self.conv(self.stem(x)).flatten(2).transpose(1, 2) + self.pos  # [B, P, E]
        q, _ = self.attn(self.queries.expand(len(x), -1, -1), x, x, need_weights=False)  # [B, 4, E]
        h = self.mlp(q.flatten(1))
        return self.disp(h), self.dir(h)


class MLP(nn.Module):

    def __init__(self, in_dim, width=256):
        super().__init__()
        self.body = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(0.2), nn.Linear(width, width), nn.GELU(), nn.Dropout(0.2))
        self.disp, self.dir = nn.Linear(width, 2), nn.Linear(width, DIRS)

    def forward(self, x):
        h = self.body(x)
        return self.disp(h), self.dir(h)


def fit_net(model, x_tr, y_tr, d_tr, ok_tr, steps, seed, device):
    # full AdamW run with a cosine rate; x_tr on device. Returns the final training loss
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    y_tr, d_tr, ok_tr = y_tr.to(device), d_tr.to(device), ok_tr.to(device)
    counts = torch.bincount(d_tr[ok_tr], minlength=DIRS).float().clamp_min(1)
    weights = counts.sum() / (DIRS * counts)  # balanced direction loss
    n = len(x_tr)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        i = torch.randint(0, n, (min(BATCH, n),), generator=gen).to(device)
        p, c = model(x_tr[i])
        loss = F.huber_loss(p / SCALE_PX, y_tr[i] / SCALE_PX)
        if ok_tr[i].any():
            loss = loss + F.cross_entropy(c[ok_tr[i]], d_tr[i][ok_tr[i]], weight=weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return float(loss.detach())


@torch.no_grad()
def predict(model, x, device, batch=256):
    model.eval()
    out = [model(x[i:i + batch].to(device)) for i in range(0, len(x), batch)]
    return torch.cat([o[0] for o in out]).cpu().numpy(), torch.cat([o[1] for o in out]).argmax(1).cpu().numpy()


def net_oof(kind, x, idx, block, y, dirs, ok, seed, device, steps=None):
    # kind 'grid' -> PositionNet, 'vec' -> MLP (standardised on the training fold). -> displacement [n, 2], direction [n], train R2
    pred, dir_pred = np.zeros_like(y, dtype=np.float64), np.zeros(len(y), int)
    y_t, d_t, ok_t = torch.from_numpy(y).float(), torch.from_numpy(dirs).long(), torch.from_numpy(ok)
    is_pixels = kind == 'grid' and x.shape[1] == 6
    train_r2 = []
    for b in np.unique(block[idx]):
        tr, te = np.flatnonzero(block[idx] != b), np.flatnonzero(block[idx] == b)
        x_tr, x_te = x[idx[tr]], x[idx[te]]
        torch.manual_seed(seed)
        if kind == 'vec':
            mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
            x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
            model = MLP(x.shape[1])
        else:
            mu, sd = x_tr.mean((0, 2, 3), keepdim=True), x_tr.std((0, 2, 3), keepdim=True) + 1e-6
            x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
            model = PositionNet(x.shape[1], x.shape[-1], is_pixels)
        x_tr = x_tr.to(device)
        fit_net(model, x_tr, y_t[tr], d_t[tr], ok_t[tr], steps or (NET_STEPS if kind == 'grid' else MLP_STEPS), seed, device)
        pred[te], dir_pred[te] = predict(model, x_te, device)
        p_tr, _ = predict(model, x_tr.cpu(), device)
        train_r2.append(float(1 - ((p_tr - y[tr]) ** 2).sum() / ((y[tr] - y[tr].mean(0)) ** 2).sum().clip(1e-9)))
    return pred, dir_pred, float(np.mean(train_r2))
