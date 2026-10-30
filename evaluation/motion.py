"""Pixel-motion proxies between consecutive frames (camera scroll by phase correlation, local block matching),
used to check what latent actions encode."""

import torch
import torch.nn.functional as F


MOTION_LABELS = [f'{v}{h}' for v in ('U', '0', 'D') for h in ('L', '0', 'R')]  # 9 classes, '00' = still


def global_shift(a, b):
    # a, b: [N, H, W] grayscale -> integer (dy, dx) [N, 2] such that b ~ roll(a, (dy, dx)), by phase correlation
    Fa, Fb = torch.fft.fft2(a), torch.fft.fft2(b)
    r = Fb * Fa.conj()
    r = r / r.abs().clamp_min(1e-8)
    corr = torch.fft.ifft2(r).real  # [N, H, W]
    N, H, W = corr.shape
    flat = corr.flatten(1).argmax(1)
    dy, dx = flat // W, flat % W
    dy = torch.where(dy > H // 2, dy - H, dy)
    dx = torch.where(dx > W // 2, dx - W, dx)
    return torch.stack([dy, dx], 1)  # [N, 2]


def motion_class(shift, tol=1):
    # shift: [N, 2] (dy, dx) -> class index [N] in 0..8 (row = vertical U/0/D, col = horizontal L/0/R)
    v = torch.where(shift[:, 0] < -tol, 0, torch.where(shift[:, 0] > tol, 2, 1))
    h = torch.where(shift[:, 1] < -tol, 0, torch.where(shift[:, 1] > tol, 2, 1))
    return v * 3 + h


def local_shift(a, b, max_shift=4, block=8, thr=0.08, min_frac=0.05):
    # a, b: [N, H, W] grayscale in [0, 1] -> dominant integer (dy, dx) [N, 2] of the blocks that changed (0 if none)
    pool = lambda t: F.avg_pool2d(t[:, None], block)[:, 0]  # [N, h, w]
    changed = pool(((a - b).abs() > thr).float()) > min_frac  # [N, h, w]
    shifts = [(dy, dx) for dy in range(-max_shift, max_shift + 1) for dx in range(-max_shift, max_shift + 1)]
    err = torch.stack([pool((torch.roll(a, (dy, dx), (1, 2)) - b).abs()) for dy, dx in shifts])  # [S, N, h, w]
    best = torch.tensor(shifts, device=a.device)[err.argmin(0)]  # [N, h, w, 2]
    moving = changed & (best.abs().sum(-1) > 0)  # [N, h, w]
    out = torch.zeros(a.shape[0], 2, dtype=torch.long, device=a.device)
    for n in torch.nonzero(moving.flatten(1).any(1)).flatten().tolist():
        v, c = best[n][moving[n]].unique(dim=0, return_counts=True)
        out[n] = v[c.argmax()]
    return out  # [N, 2]


def player_move(a, b):
    # a, b: [N, H, W] -> (dy, dx) [N, 2]: minus the camera scroll if there is one, else the dominant local move
    cam = global_shift(a, b)  # content shift
    scrolling = (cam.abs() > 1).any(1, keepdim=True)
    return torch.where(scrolling, -cam, local_shift(a, b))


def camera_compensated_diff(last, target, tol=1):
    # last, target: [B, C, H, W] in [0, 1] -> (diff [B, H, W], cam [B, 2]): per pixel, the max-channel |target - last
    # shifted by the camera|, taking the best match within +-tol px of the phase-correlation shift (absorbs sub-pixel
    # scroll on edges). On a near-flat background the phase correlation can lock onto the sprite itself (rare on Zelda)
    B, C, H, W = target.shape
    cam = global_shift(last.mean(1).float(), target.mean(1).float())  # [B, 2] (dy, dx): target ~ roll(last, cam)
    rows, cols = torch.arange(H, device=target.device), torch.arange(W, device=target.device)
    b = torch.arange(B, device=target.device)[:, None, None]  # [B, 1, 1]
    diffs = []
    for a in range(-tol, tol + 1):
        for c in range(-tol, tol + 1):
            r = (rows[None] - cam[:, :1] - a) % H  # [B, H] roll: out[i] = in[i - shift]
            k = (cols[None] - cam[:, 1:] - c) % W  # [B, W]
            shifted = last.permute(0, 2, 3, 1)[b, r[:, :, None], k[:, None, :]]  # [B, H, W, C]
            diffs.append((target.permute(0, 2, 3, 1) - shifted).abs().amax(-1))  # [B, H, W]
    return torch.stack(diffs).amin(0), cam  # [B, H, W], [B, 2]


def revealed_mask(cam, H, W, tol=1):
    # cam: [B, 2] (dy, dx) -> [B, H, W] bool: the strip revealed by the scroll (first dy rows if dy > 0, last |dy| if
    # dy < 0, same for columns), widened by tol, plus a tol-px frame border where the +-tol rolls wrap around
    rows, cols = torch.arange(H, device=cam.device), torch.arange(W, device=cam.device)
    dy, dx = cam[:, 0, None], cam[:, 1, None]  # [B, 1]
    bad_r = (rows[None] < dy.clamp_min(0) + tol) | (rows[None] >= H + dy.clamp_max(0) - tol)  # [B, H]
    bad_c = (cols[None] < dx.clamp_min(0) + tol) | (cols[None] >= W + dx.clamp_max(0) - tol)  # [B, W]
    return bad_r[:, :, None] | bad_c[:, None, :]


def moving_pixel_mask(last, target, tol=1, thr=0.1):
    # last, target: [B, C, H, W] in [0, 1] -> (moved [B, H, W] bool, cam [B, 2]): pixels that changed beyond the camera
    # scroll (camera-compensated diff > thr); the revealed strip and the wrap border are not counted
    diff, cam = camera_compensated_diff(last, target, tol)
    return (diff > thr) & ~revealed_mask(cam, *diff.shape[1:], tol), cam


def moving_token_mask(last, target, patch_size, tol=1, thr=0.1):
    # last, target: [B, C, H, W] in [0, 1] -> [B, P] bool, tokens (patch_size x patch_size) holding a moving pixel
    moved, _ = moving_pixel_mask(last, target, tol, thr)  # [B, H, W]
    tokens = F.max_pool2d(moved[:, None].float(), patch_size)  # [B, 1, Hp, Wp]
    return tokens.flatten(1) > 0  # [B, P], row-major patches as the tokenizer


def overlay_intervals(frames, max_px=2048):
    # frames: [N, C, H, W] in [0, 1], consecutive frames of one clip -> list of (start, end): frames start..end-1 show an
    # overlay (dialog box, menu) drawn over a paused game. An overlay opens with a large change (> max_px moving px) at
    # `start` and closes at the next large change `end` that brings back the frame before `start` (<= max_px moving px
    # vs frames[start - 1]). A scene cut or fade never brings back the old frame, so it opens no overlay
    n = moving_pixel_mask(frames[:-1], frames[1:])[0].flatten(1).sum(1)  # [N-1] moving px between frames k and k+1
    events = [k + 1 for k in torch.nonzero(n > max_px).flatten().tolist()]  # first frame after each large change
    intervals, i = [], 0
    while i < len(events) - 1:
        start, end = events[i], events[i + 1]
        back = int(moving_pixel_mask(frames[start - 1:start], frames[end:end + 1])[0].sum())
        if back <= max_px:
            intervals.append((start, end))
            i += 2  # the closing change cannot open the next overlay
        else:
            i += 1
    return intervals


MIN_MOVING_PX = 20  # fewer moving pixels than this = nothing moves (Link standing still)


def motion_box(last, target, size=32, min_px=MIN_MOVING_PX, max_px=2048, max_cam=12, tol=1, thr=0.1):
    # last, target: [B, C, H, W] in [0, 1] -> (top-left [B, 2] (row, col), -1 where no box; moving px count [B]).
    # The box is the size x size window holding the most moving pixels, among windows that do not touch the strip the
    # scroll reveals. No box when nothing moves (< min_px: e.g. Link standing) or when a large area changes
    # (> max_px: dialog text, fades, floor texture that does not line up during a scroll), or when the camera jumps more
    # than max_cam px (room transitions; walking scrolls 9 px per Zelda eval step). Thresholds tuned on Zelda at 128 px
    moved, cam = moving_pixel_mask(last, target, tol, thr)  # [B, H, W]
    n_moving = moved.flatten(1).sum(1)  # [B]
    count = F.avg_pool2d(moved[:, None].float(), size, stride=1)[:, 0]  # [B, H-size+1, W-size+1] fraction moving
    H, W = moved.shape[1:]
    strip = revealed_mask(cam, H, W, tol) & ~revealed_mask(torch.zeros_like(cam), H, W, tol)  # [B, H, W] minus wrap border
    touches_strip = F.max_pool2d(strip[:, None].float(), size, stride=1)[:, 0] > 0  # [B, H-size+1, W-size+1]
    count = count.masked_fill(touches_strip, -1)
    best = count.flatten(1).argmax(1)  # [B]
    box = torch.stack([best // count.shape[-1], best % count.shape[-1]], 1)  # [B, 2]
    ok = (n_moving >= min_px) & (n_moving <= max_px) & (cam.abs().amax(1) <= max_cam) & (count.flatten(1).amax(1) > 0)
    return torch.where(ok[:, None], box, torch.full_like(box, -1)), n_moving
