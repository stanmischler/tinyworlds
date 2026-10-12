"""NanoWM's evaluation protocol (arXiv 2605.23993, github.com/simchowitzlabpublic/nano-world-model @ cc1f151a), shared by
scripts/eval/eval_pusht.py and scripts/eval/eval_nanowm.py: fixed val clips, ground truth at 128 / 256, and their metric
stack (piqa PSNR / SSIM, LPIPS-VGG, pytorch-fid FID).

Clip rule (src/wm_datasets/world_model_dataset.py): the trajectories are shuffled with np.random.seed(42) and the last
(1 - split_ratio) of them are the val split (_split_trajectories_indices; Push-T ships a separate val folder, read with
split_ratio 0, i.e. all of it, still shuffled); every stride-1 window of span NUM_FRAMES * FRAME_INTERVAL in that
trajectory order (_create_slices_exhaustive); then np.random.RandomState(42).choice(n_slices, 256, replace=False), in
that order (sample_fixed_slice_specs). A clip is env frames s + {0, 5, 10, 15}; actions[i] = the 5 env actions of steps
s+5i .. s+5i+4 (normalised with the train stats when the env normalises them), flattened step-major to 10-D.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

FRAME_INTERVAL = 5
NUM_FRAMES = 4  # 1 context + 3 predicted
RESOLUTIONS = (128, 256)

# per env: NanoWM dataset config (src/configs/dataset/dino_wm/<config>.yaml), its split and the paper's numbers (Tab. 7 /
# their README, 256px, 1 -> 3 frames); the released checkpoints are in experiments/pusht_nanowm/nanowm_dump.py
ENVS = {
    'pusht': dict(config='dino_wm/pusht', split_ratio=0.0, normalize_action=True, val_h5='data/pusht_val_frames.h5',
                  published={'psnr': 33.19, 'ssim': 0.982, 'lpips': 0.016, 'fid': 13.63}),
    'point_maze': dict(config='dino_wm/point_maze', split_ratio=0.9, normalize_action=True, val_h5='data/point_maze_val_frames.h5',
                       published={'psnr': 36.74, 'ssim': 0.984, 'lpips': 0.019, 'fid': 9.66}),
    'wall': dict(config='dino_wm/wall', split_ratio=0.9, normalize_action=False, val_h5='data/wall_val_frames.h5',
                 published={'psnr': 34.05, 'ssim': 0.994, 'lpips': 0.010, 'fid': 2.64}),
}


# ----------------------------------------------------------------------------- clips
def nanowm_val_order(n_traj, split_ratio, seed=42):
    """NanoWM's val trajectory ids, in their order (_split_trajectories_indices)."""
    np.random.seed(seed)
    order = np.arange(n_traj)
    np.random.shuffle(order)
    return order[int(n_traj * split_ratio):]


def nanowm_clips(traj_ids, seq_lengths, num_clips, seed=42):
    """NanoWM's fixed val subset. traj_ids: val trajectories in NanoWM order, seq_lengths: their lengths (same order).
    -> [(traj_id, start)] in selection order, number of val slices."""
    span = NUM_FRAMES * FRAME_INTERVAL
    slices = [(int(t), s) for t, L in zip(traj_ids, seq_lengths) for s in range(0, int(L) - span + 1)]
    sel = np.random.RandomState(seed).choice(len(slices), size=min(num_clips, len(slices)), replace=False)
    return [slices[i] for i in sel], len(slices)


def load_clips(h5, row_offset, clips, mean=None, std=None, action_key='rel_actions'):
    """row_offset[traj_id] = first .h5 row of that trajectory (all env steps stored).
    -> frames uint8 [B, T, H, W, 3] (native res), actions float32 [B, T - 1, 10] (normalised iff mean is given)."""
    frames, actions = [], []
    raw = h5[action_key]
    for t, s in clips:
        base = int(row_offset[t]) + s
        frames.append(h5['frames'][[base + k * FRAME_INTERVAL for k in range(NUM_FRAMES)]])
        a = raw[base:base + (NUM_FRAMES - 1) * FRAME_INTERVAL]  # [15, 2]
        if mean is not None:
            a = (a - mean) / std
        actions.append(a.reshape(NUM_FRAMES - 1, -1))
    return np.stack(frames), np.stack(actions).astype(np.float32)


# ----------------------------------------------------------------------------- resizing
def area_resize(frames, size):
    # [..., H, W, 3] uint8 (or float, kept float32) -> [..., size, size, 3], INTER_AREA as the scripts/data/convert_*.py
    frames = frames if frames.dtype == np.uint8 else frames.astype(np.float32)
    flat = frames.reshape(-1, *frames.shape[-3:])
    out = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in flat])
    return out.reshape(*frames.shape[:-3], size, size, 3)


def to_chw01(frames, device):
    # uint8 [B, T, H, W, 3], or float already in [0, 1] -> float [B, T, 3, H, W] in [0, 1]
    x = torch.from_numpy(frames).to(device).permute(0, 1, 4, 2, 3).float()
    return x / 255.0 if frames.dtype == np.uint8 else x


def resize_bt(x, size, mode):
    # float [B, T, C, H, W] -> [B, T, C, size, size]
    B, T = x.shape[:2]
    kw = {} if mode == 'nearest' else {'align_corners': False}
    return F.interpolate(x.flatten(0, 1), size=(size, size), mode=mode, **kw).unflatten(0, (B, T))


def ground_truth(frames):
    """frames uint8 [B, T, H, W, 3] native -> {128: INTER_AREA as the training set, 256: float bilinear stretch as
    NanoWM (_load_slice, resize_mode 'stretch')}, each float [B, T, C, S, S] in [0, 1] on cpu."""
    return {128: to_chw01(area_resize(frames, 128), 'cpu'),
            256: resize_bt(to_chw01(frames, 'cpu'), 256, 'bilinear')}


def upsample(x, size):
    # [B, T, C, h, w] in [0, 1] -> size, bicubic (a 128px prediction scored at 256)
    return x if x.shape[-1] == size else resize_bt(x, size, 'bicubic').clamp(0, 1)


# ----------------------------------------------------------------------------- metrics
class Metrics:
    """NanoWM's metric stack (src/utils/metrics.py Evaluator); every input float [N, 3, H, W] in [0, 1]."""

    def __init__(self, device, fid=True):
        import lpips
        import piqa
        self.device = device
        self.psnr = piqa.PSNR(epsilon=1e-08, value_range=1.0, reduction='none').to(device)
        self.ssim = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3, reduction='none').to(device)
        self.lpips = lpips.LPIPS(net='vgg', verbose=False).to(device).eval()
        self.inception = None
        if fid:
            from pytorch_fid.inception import InceptionV3
            self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    @torch.no_grad()
    def per_frame(self, pred, gt, bs=32):
        out = {'psnr': [], 'ssim': [], 'lpips': []}
        for i in range(0, len(pred), bs):
            p, g = pred[i:i + bs].to(self.device).clamp(0, 1), gt[i:i + bs].to(self.device).clamp(0, 1)
            out['psnr'].append(self.psnr(p, g).cpu())
            out['ssim'].append(self.ssim(p, g).cpu())
            out['lpips'].append(self.lpips(p * 2 - 1, g * 2 - 1).flatten().cpu())
        return {k: torch.cat(v).numpy() for k, v in out.items()}  # each [N]

    @torch.no_grad()
    def fid(self, pred, gt, bs=32):
        if self.inception is None:
            return None
        feats = []
        for x in (gt, pred):
            f = []
            for i in range(0, len(x), bs):
                b = F.interpolate(x[i:i + bs].to(self.device).clamp(0, 1), size=(299, 299), mode='bilinear', align_corners=False)
                f.append(self.inception(b)[0].squeeze(3).squeeze(2).cpu().numpy())
            feats.append(np.concatenate(f))
        (fr, ff) = feats
        return float(calculate_frechet_distance(fr.mean(0), np.cov(fr, rowvar=False), ff.mean(0), np.cov(ff, rowvar=False)))


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """pytorch-fid 0.3.0's calculate_frechet_distance, minus its `sqrtm(..., disp=False)` (removed in scipy >= 1.16)."""
    import scipy.linalg
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def summarise(metrics, pred, gt, with_fid):
    """pred, gt: float [B, 3, C, H, W] (the predicted steps) -> {psnr, ssim, lpips, fid, per_step}."""
    B, S = pred.shape[:2]
    pf = metrics.per_frame(pred.flatten(0, 1), gt.flatten(0, 1))  # each [B * S], step-minor
    out = {k: float(v.mean()) for k, v in pf.items()}
    out['fid'] = metrics.fid(pred.flatten(0, 1), gt.flatten(0, 1)) if with_fid else None
    out['per_step'] = {str(t + 1): {k: float(v.reshape(B, S)[:, t].mean()) for k, v in pf.items()} for t in range(S)}
    return out


def reference_tables(metrics, gt, with_fid):
    """copy-last at both resolutions and the resize ceiling at 256 (the 128 ground truth bicubic-upsampled: the best any
    128px model can score at 256). gt: ground_truth(...) -> {128: {...}, 256: {...}}."""
    tables = {res: {} for res in RESOLUTIONS}
    for res in RESOLUTIONS:
        copy = gt[res][:, :1].expand(-1, NUM_FRAMES - 1, -1, -1, -1)
        tables[res]['copy_last'] = summarise(metrics, copy, gt[res][:, 1:], with_fid)
    tables[256]['resize_ceiling'] = summarise(metrics, upsample(gt[128], 256)[:, 1:], gt[256][:, 1:], with_fid)
    return tables


def score_nanowm_npz(metrics, npz_path, clips, gt, tables, with_fid):
    """NanoWM's own predictions (experiments/pusht_nanowm/nanowm_rescore_*), same clips, same metric code; adds
    tables[res]['nanowm'] (at 128 their 256 prediction is INTER_AREA-downsampled) and returns extra result fields."""
    nz = np.load(npz_path)
    theirs = [(int(e), int(s)) for e, s in zip(nz['traj_idx'], nz['start_frame'])]
    order = [theirs.index(c) for c in clips]  # raises ValueError if a clip is missing
    extra = {'nanowm_clip_order_matches': order == list(range(len(clips)))}
    tables[256]['nanowm'] = summarise(metrics, to_chw01(nz['pred'][order], 'cpu')[:, 1:], gt[256][:, 1:], with_fid)
    tables[128]['nanowm'] = summarise(metrics, to_chw01(area_resize(nz['pred'][order], 128), 'cpu')[:, 1:], gt[128][:, 1:], with_fid)
    if 'gt' in nz.files:  # their own 256 ground truth vs ours (decoder / resize differences)
        extra['nanowm_gt_vs_ours_psnr'] = float(metrics.per_frame(to_chw01(nz['gt'][order], 'cpu').flatten(0, 1), gt[256].flatten(0, 1))['psnr'].mean())
    return extra


def print_tables(tables):
    for res in RESOLUTIONS:
        print(f'@{res}')
        for g, v in tables[res].items():
            fid = f"{v['fid']:.2f}" if v['fid'] is not None else '-'
            steps = '  '.join(f"t{t}: {s['psnr']:.2f}" for t, s in v['per_step'].items())
            print(f"  {g:16s} PSNR {v['psnr']:.2f}  SSIM {v['ssim']:.4f}  LPIPS {v['lpips']:.4f}  FID {fid}   ({steps})")
