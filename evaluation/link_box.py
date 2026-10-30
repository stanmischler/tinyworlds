"""Link-focused next-frame metrics (scripts/eval/eval_next_frame.py): a 32x32 box per test window, placed either by
judges (Link's centre on the target, a saved file) or by motion (evaluation.motion.motion_box), scored with LPIPS on the
crops and FID over all crops; and a motion-weighted PSNR. Windows without a box are dropped from that box's metrics.
Images are [B, C, H, W] in [0, 1]; a box is its top-left (row, col), (-1, -1) when the window has none."""

import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from evaluation.motion import camera_compensated_diff, overlay_intervals, revealed_mask

BOX = 32
LPIPS_SIZE = 128  # crops are upsampled to this before LPIPS (AlexNet features are too coarse on 32 px)
INCEPTION_SIZE = 299


def overlay_windows(h5_path, windows, context, frame_skip):
    """-> list[bool], one per (block_id, local_start) window: True when the last context frame or the target shows an
    overlay (dialog box) over a paused game. Overlays are found once per test block on its frames `frame_skip` apart,
    from the block start, which is the grid every window's frames sit on (window starts step by a multiple of it)."""
    with h5py.File(h5_path, 'r') as f:
        blocks = json.loads(f.attrs['test_blocks_local'])
        assert all((s - blocks[b][0]) % frame_skip == 0 for b, s in windows), \
            'overlays are found on the frame_skip grid from each block start: sample_stride must be a multiple of frame_skip'
        in_overlay = set()  # local frame indices inside an overlay
        for start, end in blocks:
            idx = list(range(start, end, frame_skip))
            frames = torch.from_numpy(f['frames'][idx]).permute(0, 3, 1, 2).float() / 255  # [N, C, H, W]
            for a, b in overlay_intervals(frames):
                in_overlay.update(idx[a:b])
    span = context * frame_skip
    return [(s + span - frame_skip) in in_overlay or (s + span) in in_overlay for _, s in windows]


def judge_boxes(path, windows, frame_size):
    """Judge file (evaluation/zelda_link_boxes.json: one {"block", "start", "x", "y"} per window, x/y = Link's centre on
    the target or null) -> top-left boxes [N, 2] for `windows`, the box centred on Link and clamped inside the frame."""
    with open(path) as f:
        by_window = {(r['block'], r['start']): r for r in json.load(f)['windows']}
    missing = [w for w in windows if tuple(w) not in by_window]
    assert not missing, f'{path} has no box for {len(missing)} windows (e.g. {missing[0]}): it was built for the Zelda ' \
                        'test split with context 3, frame skip 4, stride 8'
    clamp = lambda c: min(max(int(round(c)) - BOX // 2, 0), frame_size - BOX)  # top-left of a box centred on c
    boxes = []
    for w in windows:
        r = by_window[tuple(w)]
        boxes.append((-1, -1) if r['x'] is None else (clamp(r['y']), clamp(r['x'])))
    return torch.tensor(boxes)  # [N, 2]


def crop(images, boxes):
    # images [B, C, H, W] (any device), boxes [B, 2] on the CPU -> (crops [n, C, BOX, BOX] of the windows with a box,
    # their row indices in the batch)
    rows = [(i, r, c) for i, (r, c) in enumerate(boxes.tolist()) if r >= 0]
    if not rows:
        return None, []
    return torch.stack([images[i, :, r:r + BOX, c:c + BOX] for i, r, c in rows]), [i for i, _, _ in rows]


def box_lpips(lpips_fn, pred, target, boxes):
    # -> [B] LPIPS between the pred and target crops (CPU), NaN where there is no box (or no lpips package)
    out = torch.full((pred.shape[0],), float('nan'))
    (p, idx), (t, _) = crop(pred, boxes), crop(target, boxes)
    if lpips_fn is None or not idx:
        return out
    up = lambda x: F.interpolate(x, size=LPIPS_SIZE, mode='bilinear', align_corners=False)
    out[idx] = lpips_fn(up(p) * 2 - 1, up(t) * 2 - 1).flatten().float().cpu()
    return out


def try_inception(device):
    try:
        from pytorch_fid.inception import InceptionV3  # optional dependency
        return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    except Exception as e:
        print(f'WARNING: no Inception network ({e!r}): box FIDs will be None')
        return None


def box_inception_features(inception, images, boxes):
    # -> [n, 2048] Inception pool features of the crops of the windows with a box (n = number of such windows)
    x, idx = crop(images, boxes)
    if inception is None or not idx:
        return np.zeros((0, 2048))
    x = F.interpolate(x, size=INCEPTION_SIZE, mode='bilinear', align_corners=False)
    return inception(x)[0].flatten(1).float().cpu().double().numpy()


def fid(feats_a, feats_b):
    # Frechet distance between two feature sets [n, 2048]; None when either is too small to estimate a covariance
    if len(feats_a) < 2 or len(feats_b) < 2:
        return None
    from scipy import linalg  # pytorch_fid's calculate_frechet_distance breaks on scipy >= 1.16 (sqrtm lost `disp`)
    mu_a, mu_b = feats_a.mean(0), feats_b.mean(0)
    cov_a, cov_b = np.cov(feats_a, rowvar=False), np.cov(feats_b, rowvar=False)
    cov_sqrt = linalg.sqrtm(cov_a @ cov_b)
    if not np.isfinite(cov_sqrt).all():  # singular product (fewer crops than 2048 dims): offset, as pytorch_fid does
        offset = np.eye(len(cov_a)) * 1e-6
        cov_sqrt = linalg.sqrtm((cov_a + offset) @ (cov_b + offset))
    cov_sqrt = cov_sqrt.real  # tiny imaginary parts are numerical noise
    # ||mu_a - mu_b||^2 + Tr(cov_a + cov_b - 2 (cov_a cov_b)^1/2)
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * cov_sqrt))


def motion_weights(last, target, tol=1):
    # -> w [B, H, W] >= 0: camera-compensated |target - last| (max over channels), 0 on the strip the scroll reveals
    diff, cam = camera_compensated_diff(last, target, tol)
    return diff * ~revealed_mask(cam, *diff.shape[1:], tol)


def motion_weighted_psnr(pred, target, w):
    # pred, target [B, C, H, W], w [B, H, W] -> [B]: PSNR of the w-weighted MSE, sum w (pred - target)^2 / sum w
    sq_err = ((pred - target) ** 2).mean(1)  # [B, H, W], mean over channels
    weighted_mse = (w * sq_err).flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / weighted_mse.clamp_min(1e-10))
