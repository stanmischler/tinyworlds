"""Flag the frames of a gameplay video that are not gameplay, so they can be cut from the dataset.

A frame is flagged when it is
  - black:  mean intensity < BLACK_MEAN (fades, loading),
  - flat:   intensity std < FLAT_STD (blank screens),
  - junk:   like a hand-labelled non-gameplay frame (title screen, menus, cutscene art, another game). A person labels
            one frame every SAMPLE_SECONDS on a contact sheet (junk or gameplay); every frame then takes the label of
            its nearest labelled sample (16x16 pixels) over all videos, which also catches a menu opened between
            two samples.
Flags are smoothed over one second (median filter) and padded by PAD_SECONDS on each side, so fades in and out of a
cut go with it.
"""

import numpy as np
from scipy.ndimage import median_filter

BLACK_MEAN = 12.0
FLAT_STD = 6.0
SAMPLE_SECONDS = 5
PAD_SECONDS = 0.5


def sample_labels(n_samples, junk_ranges):
    """bool [n_samples]: sample k (at k * SAMPLE_SECONDS) is junk if its time is inside a [start_s, end_s] range."""
    times = np.arange(n_samples) * SAMPLE_SECONDS
    labels = np.zeros(n_samples, dtype=bool)
    for start, end in junk_ranges:
        labels |= (times >= start) & (times <= end)
    return labels


def nearest_label(x, samples, labels, chunk=4096):
    """x [N,D], samples [S,D], labels bool [S] -> bool [N]: label of each row's nearest sample (Euclidean)."""
    s_sq = (samples ** 2).sum(1)
    out = []
    for start in range(0, len(x), chunk):
        sq = -2 * x[start:start + chunk] @ samples.T + s_sq                       # [n,S] squared distance - |x|^2
        out.append(labels[sq.argmin(1)])
    return np.concatenate(out)


def black_flat_flags(frames):
    """frames [N,H,W,3] uint8 -> (black, flat) bool [N] arrays."""
    gray = frames.mean(3)                                                         # [N,H,W]
    return gray.mean((1, 2)) < BLACK_MEAN, gray.std((1, 2)) < FLAT_STD


def cut_runs(flagged, fps):
    """bool [N] -> list of (start, end) frame runs to cut, end-exclusive: 1 s median-smoothed, padded, merged."""
    smooth = median_filter(flagged.astype(np.uint8), size=int(fps) | 1, mode='nearest').astype(bool)
    pad = int(round(PAD_SECONDS * fps))
    padded = np.convolve(smooth, np.ones(2 * pad + 1), 'same') > 0 if pad else smooth
    edges = np.diff(np.concatenate([[0], padded.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def segment_ids(n_frames, runs):
    """int64 [n_frames]: id of the contiguous kept segment of each frame, -1 for cut frames."""
    cut = np.zeros(n_frames, dtype=bool)
    for start, end in runs:
        cut[start:end] = True
    starts_new = ~cut & np.concatenate([[True], cut[:-1]])                       # first kept frame after a cut
    ids = np.cumsum(starts_new) - 1
    return np.where(cut, -1, ids).astype(np.int64)
