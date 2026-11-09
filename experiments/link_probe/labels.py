"""Held-out Zelda pairs for the Link motion probes, labelled from the judges' Link centres (evaluation/zelda_link_boxes.json).

The judges placed Link on the target of each eval window (context 3, frame skip 4, stride 8), so consecutive windows give
Link's on-screen displacement over 8 stored frames, the only span the labels can measure. One pair per consecutive window
inside a test block: frame t (previous target), frame t+4, frame t+8 (this target). The camera over the span is the sum of
the two 4-frame shifts, since a walking scroll of about 9 px per 4 frames can leave the +-16 px search over 8 frames."""

import json

import h5py
import numpy as np

from evaluation.lam_vis.data import read_rows
from evaluation.windows import test_windows
from scripts.eval.visualise_transitions import LARGE_CHANGE, WEAK_SHIFT, best_shift, mean_diff

H5 = 'data/zelda_test_frames.h5'
BOXES = 'evaluation/zelda_link_boxes.json'
SPAN = 8            # stored frames between the labelled frames t and t+8
MOVED_PX = 3        # Link "moved": largest axis of the displacement, as the earlier z analysis
STILL_PX = 1        # camera "still": both 4-frame shifts within this
SCROLL_PX = 3       # camera "scroll": the shift over the span reaches this
REST_BOX = 40       # px, box around Link left out of the "rest of the frame changed" check
REST_THR = 0.03     # camera still also needs under this fraction of pixels changing outside Link's boxes
DIR_NAMES = ['up', 'down', 'left', 'right']


def direction(d):
    # d [N, 2] (dy, dx) -> [N] 0 up, 1 down, 2 left, 3 right, by the larger axis
    vertical = np.abs(d[:, 0]) >= np.abs(d[:, 1])
    return np.where(vertical, np.where(d[:, 0] < 0, 0, 1), np.where(d[:, 1] < 0, 2, 3))


def rest_change(a, b, centres):
    # a, b uint8 [H, W, C], centres [(y, x), ...] -> fraction of pixels changing by over 10% outside the boxes around Link
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1) / 255 > 0.1
    H, W = d.shape
    for y, x in centres:
        y0, x0 = [int(np.clip(round(v) - REST_BOX // 2, 0, s - REST_BOX)) for v, s in ((y, H), (x, W))]
        d[y0:y0 + REST_BOX, x0:x0 + REST_BOX] = False
    return float(d.mean())


def build_pairs(h5=H5, boxes=BOXES, limit=None):
    """-> dict of arrays, one row per consecutive-window pair with no dialog and Link visible at both ends:
    rows [N, 3] (t, t+4, t+8), block [N], link_t [N, 2] (y, x) on frame t, disp [N, 2] Link (dy, dx) on screen, cam [N, 2]
    content shift over the span, cam_half [N, 2] over t+4 -> t+8 alone, still / scroll bool [N], rest, cam_fit."""
    with open(boxes) as f:
        by = {(r['block'], r['start']): r for r in json.load(f)['windows']}
    wins = test_windows(h5, 3, 4, 8)
    rec = [by[tuple(w)] for w in wins]
    keep = [i for i in range(1, len(wins))
            if wins[i][0] == wins[i - 1][0] and wins[i][1] - wins[i - 1][1] == SPAN
            and not rec[i]['dialog'] and not rec[i - 1]['dialog'] and rec[i]['x'] is not None and rec[i - 1]['x'] is not None]
    n_pairs = sum(wins[i][0] == wins[i - 1][0] for i in range(1, len(wins)))
    if limit:
        keep = keep[:: max(1, len(keep) // limit)][:limit]
    rows = np.array([[wins[i - 1][1] + 12, wins[i - 1][1] + 16, wins[i][1] + 12] for i in keep])
    link_a = np.array([[rec[i - 1]['y'], rec[i - 1]['x']] for i in keep], float)
    link_b = np.array([[rec[i]['y'], rec[i]['x']] for i in keep], float)
    with h5py.File(h5, 'r') as f:
        fr = [read_rows(f['frames'], rows[:, k]) for k in range(3)]
    cam1, cam2, weak, large, rest = [], [], [], [], []
    for a, m, b, la, lb in zip(*fr, link_a, link_b):
        s1, s2 = best_shift(a, m)[0], best_shift(m, b)[0]
        cam1.append(s1), cam2.append(s2)
        fa, fm, fb = a.astype(np.float32), m.astype(np.float32), b.astype(np.float32)
        comp = [mean_diff(fa, fm, *s1), mean_diff(fm, fb, *s2)]
        plain = [mean_diff(fa, fm, 0, 0), mean_diff(fm, fb, 0, 0)]
        weak.append(any(s != (0, 0) and c > WEAK_SHIFT * p for s, c, p in zip((s1, s2), comp, plain)))
        large.append(max(comp) > LARGE_CHANGE)
        rest.append(max(rest_change(a, m, [la, lb]), rest_change(m, b, [la, lb])))
    cam1, cam2, rest = np.array(cam1), np.array(cam2), np.array(rest)
    weak, large = np.array(weak), np.array(large)
    cam = cam1 + cam2
    still = (np.abs(cam1).max(1) <= STILL_PX) & (np.abs(cam2).max(1) <= STILL_PX) & (rest <= REST_THR) & ~large
    scroll = (np.abs(cam).max(1) >= SCROLL_PX) & ~weak & ~large
    return {'rows': rows, 'block': np.array([wins[i][0] for i in keep]), 'link_t': link_a, 'disp': link_b - link_a,
            'cam': cam.astype(np.float32), 'cam_half': cam2.astype(np.float32), 'still': still, 'scroll': scroll, 'rest': rest,
            'weak': weak, 'large': large, 'n_consecutive_pairs': n_pairs}
