"""Put gameplay videos of a game into the frame of an existing dataset: same crop, zoom and 128x128 size.

The Zelda training frames are a square window of the game screen, measured by matching SIFT features of our frames
against YouTube frames of the same game (scale identical in x and y, window centred horizontally, HUD cut at the top):
    side = CROP_SIDE * H,   top = y0 + CROP_TOP * H,   left = centre_x - side * stretch / 2
with H the game screen's height in the video and stretch the video's horizontal / vertical scale (4/3 for a 4:3 game
stretched to 16:9).
"""

from dataclasses import dataclass

import cv2
import numpy as np

CROP_SIDE = 0.762   # window side / game-screen height
CROP_TOP = 0.1985   # window top below the game screen's top edge / game-screen height (dialogue-box borders on our rows 74 and 126)
FRAME_SIZE = 128


@dataclass
class GameCrop:
    x0: int           # game screen box in video pixels, end-exclusive
    y0: int
    x1: int
    y1: int
    stretch: float = 1.0

    def window(self):
        """(left, top, width, height) of our crop window in video pixels."""
        height = self.y1 - self.y0
        side = CROP_SIDE * height
        width = side * self.stretch
        left = (self.x0 + self.x1) / 2 - width / 2
        top = self.y0 + CROP_TOP * height
        return int(round(left)), int(round(top)), int(round(width)), int(round(side))

    def apply(self, frame):
        """frame [H,W,3] uint8 RGB -> [FRAME_SIZE,FRAME_SIZE,3] uint8, resized as the original dataset (INTER_AREA)."""
        left, top, width, height = self.window()
        window = frame[top:top + height, left:left + width]
        return cv2.resize(window, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)


def detect_game_box(frames, change_threshold=8.0, min_col_fraction=0.5, min_row_fraction=0.3):
    """Game screen box from frames [N,H,W,3] spread over a video: the pixels that change over time.

    Bands, side art and borders are static; a webcam changes but covers too few rows of its columns to pass
    `min_col_fraction`. Returns (x0, y0, x1, y1), end-exclusive.
    """
    changing = frames.astype(np.float32).std(0).mean(2) > change_threshold   # [H,W]
    cols = np.where(changing.mean(0) > min_col_fraction)[0]
    rows = np.where(changing.mean(1) > min_row_fraction)[0]
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


class ReferenceMatcher:
    """Affine scale of a video frame relative to our frames: SIFT on our frames (upsampled), RANSAC affine fit.

    The translation is not used: the camera scrolls, so a video frame and its best-matching frame of ours are rarely
    at the same camera position. Scale is position-independent.
    """

    def __init__(self, reference_frames, upsample=4):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.upsample = upsample
        self.reference = [self._features(frame, upsample) for frame in reference_frames]

    def _features(self, frame, upsample):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if upsample != 1:
            gray = cv2.resize(gray, None, fx=upsample, fy=upsample, interpolation=cv2.INTER_CUBIC)
        return self.sift.detectAndCompute(gray, None)

    def match(self, video_frame, min_matches=12):
        """Best reference match: (inliers, reference index, sx, sy) with sx, sy = video px per reference px, or None."""
        video_keys, video_desc = self._features(video_frame, 1)
        if video_desc is None:
            return None
        best = None
        for index, (ref_keys, ref_desc) in enumerate(self.reference):
            if ref_desc is None or len(ref_desc) < 2:
                continue
            pairs = [a for a, b in self.matcher.knnMatch(ref_desc, video_desc, k=2) if a.distance < 0.7 * b.distance]
            if len(pairs) < min_matches:
                continue
            src = np.float32([ref_keys[p.queryIdx].pt for p in pairs]) / self.upsample
            dst = np.float32([video_keys[p.trainIdx].pt for p in pairs])
            affine, inliers = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=3)
            if affine is None:
                continue
            n = int(inliers.sum())
            if best is None or n > best[0]:
                best = (n, index, float(affine[0, 0]), float(affine[1, 1]))
        return best


def read_frames(video_path, n_frames, start_fraction=0.05, end_fraction=0.95):
    """n_frames frames [n,H,W,3] uint8 RGB evenly spread over the video (seeking), and their frame indices."""
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(total * start_fraction, total * end_fraction, n_frames).astype(int)
    frames = []
    for index in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = video.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video.release()
    return np.stack(frames), indices[:len(frames)]


def thumbnail_features(frames, size=16):
    """frames [N,H,W,3] uint8 -> [N, size*size*3] float32 (area-averaged), for nearest-neighbour scene lookup."""
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames]
                    ).reshape(len(frames), -1).astype(np.float32)
