"""Push-T evaluation with NanoWM's protocol (STA-29), for a tinyworlds model trained with action_source=gt.

Protocol (NanoWM, src/scripts/eval_single_model.sh + src/wm_datasets/world_model_dataset.py):
  - clips: the 21 shipped val episodes of DINO-WM's pusht_noise shuffled by np.random.seed(42), every start s in range(0, L - 20 + 1) in that episode
    order (stride 1), then np.random.RandomState(42).choice(n_slices, 256, replace=False), in that order
  - a clip is env frames s + {0, 5, 10, 15}; actions[i] = the 5 rel_actions/100 of steps s+5i .. s+5i+4, z-scored
    with the train stats and flattened to 10-D
  - 1 real context frame -> 3 predicted frames, ground-truth actions, autoregressive (one frame at a time),
    greedy MaskGIT (temperature 0); the 4 frames' tokens are decoded together, as in training
  - metrics on the 3 predicted frames only, mean over the 768 frames: PSNR (piqa, value_range 1), SSIM (piqa,
    11x11 gaussian, sigma 1.5), LPIPS-VGG (lpips 0.1.4, inputs in [-1, 1]), FID (pytorch-fid InceptionV3 pool3,
    bilinear resize to 299, inputs in [0, 1]); plus per predicted step (no FID)
Scored at two resolutions:
  128: prediction as is vs the 224px frame INTER_AREA-resized to 128 (as the training set)   <- headline
  256: prediction bicubic-upsampled vs the 224px frame bilinear-resized to 256 (NanoWM's ground truth exactly)
References in every table: copy-last (frame 0 repeated), tokenizer_recon (the 4 real frames tokenized and decoded:
the ceiling of any token prediction), resize_ceiling (256 only: the 128 ground truth bicubic-upsampled, the best any
128px model can score at 256), and nanowm (if --nanowm-npz: NanoWM's own predictions from
scripts/eval/nanowm_rescore_modal.py, scored with the same code; use predictions_f16.npz: rounding their float
predictions to uint8 alone raises LPIPS by ~0.003 and FID by ~3; at 128 they are INTER_AREA-downsampled).

Outputs <out_dir>/<name>.json and <out_dir>/<name>.png (8 clips: context, targets, predictions).

Usage (repo root, PYTHONPATH=$PWD):
    python scripts/eval/eval_pusht.py --run-dir results/pusht_v1_<date> --nanowm-npz results/nanowm_pusht_rescore/predictions_f16.npz
    python scripts/eval/eval_pusht.py --run-dir results/<smoke> --limit 8 --no-fid          # smoke
    python scripts/eval/eval_pusht.py --nanowm-npz results/nanowm_pusht_rescore/predictions_f16.npz   # NanoWM only
"""

import argparse
import json
import os
import time

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import find_latest_checkpoint, load_videotokenizer_from_checkpoint, load_dynamics_from_checkpoint

FRAME_INTERVAL = 5
NUM_FRAMES = 4  # 1 context + 3 predicted
RESOLUTIONS = (128, 256)


# ----------------------------------------------------------------------------- clips
def nanowm_clips(seq_lengths, num_clips, seed):
    """NanoWM's fixed val subset: [(episode, start)] in selection order."""
    span = NUM_FRAMES * FRAME_INTERVAL
    # their val split (split_ratio 0) still shuffles the episodes first (_split_trajectories_indices, np.random.seed(42));
    # verified against their fixed_subset.json by scripts/eval/nanowm_rescore_modal.py: identical list and order
    np.random.seed(seed)
    order = np.arange(len(seq_lengths))
    np.random.shuffle(order)
    slices = [(int(ep), s) for ep in order for s in range(0, seq_lengths[ep] - span + 1)]
    sel = np.random.RandomState(seed).choice(len(slices), size=min(num_clips, len(slices)), replace=False)
    return [slices[i] for i in sel], len(slices)


def load_clips(h5, ep_offset, clips, mean, std):
    """-> frames uint8 [B, T, 224, 224, 3], actions float32 [B, T - 1, 10] (normalised, NanoWM order)."""
    frames, actions = [], []
    rel = h5['rel_actions']
    for ep, s in clips:
        base = ep_offset[ep] + s
        frames.append(h5['frames'][[base + k * FRAME_INTERVAL for k in range(NUM_FRAMES)]])
        a = (rel[base:base + (NUM_FRAMES - 1) * FRAME_INTERVAL] - mean) / std  # [15, 2]
        actions.append(a.reshape(NUM_FRAMES - 1, FRAME_INTERVAL * 2))
    return np.stack(frames), np.stack(actions).astype(np.float32)


def area_resize(frames, size):
    # [..., H, W, 3] uint8 (or float, kept float32) -> [..., size, size, 3], INTER_AREA as scripts/convert_pusht.py
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


# ----------------------------------------------------------------------------- model
@torch.no_grad()
def rollout(tok, dyn, x, actions, num_steps, temperature, batch_size):
    """x: float [B, T, C, H, W] in [-1, 1] (only frame 0 is used as context); actions [B, T - 1, A].
    -> pred, recon float [B, T, C, H, W] in [0, 1]; pred_idx, gt_idx [B, T, P]."""
    preds, recons, pidx, gidx = [], [], [], []

    def idx_to_latents(idx):
        return tok.quantizer.get_latents_from_indices(idx, dim=-1)

    for i in range(0, len(x), batch_size):
        xb, ab = x[i:i + batch_size], actions[i:i + batch_size]
        gt_idx = tok.tokenize(xb)  # [b, T, P]; causal tokenizer: frame 0's codes do not depend on later frames
        lat = idx_to_latents(gt_idx[:, :1])  # [b, 1, P, L]
        for t in range(1, xb.shape[1]):
            lat = dyn.forward_inference(lat, prediction_horizon=1, num_steps=num_steps, index_to_latents_fn=idx_to_latents,
                                        conditioning=ab[:, :t], temperature=temperature)  # [b, t + 1, P, L]
        preds.append(((tok.detokenize(lat) + 1) / 2).clamp(0, 1).cpu())  # decode the 4 frames together, as trained
        recons.append(((tok.detokenize(idx_to_latents(gt_idx)) + 1) / 2).clamp(0, 1).cpu())
        pidx.append(tok.quantizer.get_indices_from_latents(lat, dim=-1).cpu())
        gidx.append(gt_idx.cpu())
        print(f'  rollout {min(i + batch_size, len(x))}/{len(x)}', flush=True)
    return torch.cat(preds), torch.cat(recons), torch.cat(pidx), torch.cat(gidx)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--val-h5', default='data/pusht_val_frames.h5')
    p.add_argument('--run-dir', help='results dir holding the tokenizer + dynamics checkpoints (latest step of each)')
    p.add_argument('--video-tokenizer-path'); p.add_argument('--dynamics-path')
    p.add_argument('--nanowm-npz', help='NanoWM predictions (scripts/eval/nanowm_rescore_modal.py) to score alongside')
    p.add_argument('--num-clips', type=int, default=256)
    p.add_argument('--seed', type=int, default=42, help="NanoWM's validation_fixed_subset_seed")
    p.add_argument('--num-steps', type=int, default=10, help='MaskGIT unmasking iterations per frame')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    p.add_argument('--limit', type=int, help='first N clips only (smoke test; FID needs many)')
    p.add_argument('--no-fid', action='store_true')
    p.add_argument('--out-dir', default='eval_results')
    p.add_argument('--name')
    args = p.parse_args()

    has_model = bool(args.run_dir or args.dynamics_path)
    if args.run_dir:
        args.video_tokenizer_path = find_latest_checkpoint('.', 'video_tokenizer', run_root_dir=args.run_dir)
        args.dynamics_path = find_latest_checkpoint('.', 'dynamics', run_root_dir=args.run_dir)
    # with neither a model nor --nanowm-npz only the references are computed (copy-last, resize ceiling)
    name = args.name or (os.path.basename(os.path.normpath(args.run_dir)) if args.run_dir else ('nanowm' if args.nanowm_npz else 'references')) + '_pusht'
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    # ---- clips and ground truth
    h5 = h5py.File(args.val_h5, 'r')
    seq_lengths = [int(L) for L in h5['seq_lengths'][:]]
    ep_offset = np.concatenate([[0], np.cumsum(seq_lengths)[:-1]])
    mean, std = h5.attrs['action_mean'], h5.attrs['action_std']
    clips, n_slices = nanowm_clips(seq_lengths, args.num_clips, args.seed)
    if args.limit:
        clips = clips[:args.limit]
    print(f'{len(clips)} clips (of {n_slices} val slices, seed {args.seed}) from {args.val_h5}')
    frames224, actions = load_clips(h5, ep_offset, clips, mean, std)  # [B, T, 224, 224, 3], [B, T-1, 10]
    gt = {128: to_chw01(area_resize(frames224, 128), 'cpu'),  # [B, T, C, 128, 128]
          256: resize_bt(to_chw01(frames224, 'cpu'), 256, 'bilinear')}  # NanoWM: float bilinear 224 -> 256
    with_fid = not args.no_fid
    metrics = Metrics(device, fid=with_fid)

    def up(x, size):  # [B, T, C, h, w] in [0, 1] -> size, bicubic
        return x if x.shape[-1] == size else resize_bt(x, size, 'bicubic').clamp(0, 1)

    tables = {res: {} for res in RESOLUTIONS}
    for res in RESOLUTIONS:
        copy = gt[res][:, :1].expand(-1, NUM_FRAMES - 1, -1, -1, -1)
        tables[res]['copy_last'] = summarise(metrics, copy, gt[res][:, 1:], with_fid)
    tables[256]['resize_ceiling'] = summarise(metrics, up(gt[128], 256)[:, 1:], gt[256][:, 1:], with_fid)

    result = {'name': name, 'n_clips': len(clips), 'n_val_slices': n_slices, 'val_h5': args.val_h5,
              'clips': [[int(e), int(s)] for e, s in clips],
              'config': {k: v for k, v in vars(args).items() if k not in ('out_dir', 'name')}}

    # ---- our model
    pred = None
    if has_model:
        print(f'checkpoints:\n  {args.video_tokenizer_path}\n  {args.dynamics_path}')
        tok, _ = load_videotokenizer_from_checkpoint(args.video_tokenizer_path, device)
        dyn, _ = load_dynamics_from_checkpoint(args.dynamics_path, device)
        tok.eval(); dyn.eval()
        x = gt[128].to(device) * 2 - 1  # [B, T, C, 128, 128] in [-1, 1], the training transform
        pred, recon, pred_idx, gt_idx = rollout(tok, dyn, x, torch.from_numpy(actions).to(device),
                                                args.num_steps, args.temperature, args.batch_size)
        for res in RESOLUTIONS:
            tables[res]['model'] = summarise(metrics, up(pred, res)[:, 1:], gt[res][:, 1:], with_fid)
            tables[res]['tokenizer_recon'] = summarise(metrics, up(recon, res)[:, 1:], gt[res][:, 1:], with_fid)
        acc = (pred_idx[:, 1:] == gt_idx[:, 1:]).float().mean((0, 2))  # [3]
        copy_acc = (gt_idx[:, :1] == gt_idx[:, 1:]).float().mean((0, 2))
        result['token_acc'] = {'model': float(acc.mean()), 'copy_last': float(copy_acc.mean()),
                               'model_per_step': {str(t + 1): float(a) for t, a in enumerate(acc)}}

    # ---- NanoWM's own predictions, same clips, same metric code
    if args.nanowm_npz:
        nz = np.load(args.nanowm_npz)
        theirs = [(int(e), int(s)) for e, s in zip(nz['traj_idx'], nz['start_frame'])]
        order = [theirs.index(c) for c in clips]  # KeyError-free: raises ValueError if a clip is missing
        result['nanowm_clip_order_matches'] = order == list(range(len(clips)))
        npred = to_chw01(nz['pred'][order], 'cpu')  # [B, T, C, 256, 256]
        tables[256]['nanowm'] = summarise(metrics, npred[:, 1:], gt[256][:, 1:], with_fid)
        tables[128]['nanowm'] = summarise(metrics, to_chw01(area_resize(nz['pred'][order], 128), 'cpu')[:, 1:], gt[128][:, 1:], with_fid)
        if 'gt' in nz.files:  # their own 256 ground truth vs ours (decoder / resize differences)
            result['nanowm_gt_vs_ours_psnr'] = float(metrics.per_frame(to_chw01(nz['gt'][order], 'cpu').flatten(0, 1), gt[256].flatten(0, 1))['psnr'].mean())

    result['tables'] = {str(r): t for r, t in tables.items()}
    result['seconds'] = round(time.time() - t0, 1)
    json_path = os.path.join(args.out_dir, f'{name}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=1)

    # ---- PNG: first 8 clips, context + 3 targets + 3 predictions at 128
    if pred is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = min(8, len(clips))
        fig, axes = plt.subplots(n, 7, figsize=(11, 1.65 * n))
        axes = np.atleast_2d(axes)
        for r in range(n):
            imgs = [gt[128][r, 0]] + [gt[128][r, t] for t in range(1, 4)] + [pred[r, t] for t in range(1, 4)]
            titles = ['context'] + [f'target +{5 * t}' for t in range(1, 4)] + [f'pred +{5 * t}' for t in range(1, 4)]
            for c, (img, title) in enumerate(zip(imgs, titles)):
                ax = axes[r, c]
                ax.imshow(img.permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=8)
                if c == 0:
                    ax.set_ylabel(f'ep {clips[r][0]} s {clips[r][1]}', fontsize=7)
        m = tables[128]['model']
        fig.suptitle(f"{name} @128: PSNR {m['psnr']:.2f}  SSIM {m['ssim']:.3f}  LPIPS {m['lpips']:.3f}"
                     + (f"  FID {m['fid']:.1f}" if m['fid'] is not None else ''), fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'{name}.png'), dpi=130); plt.close(fig)

    print()
    for res in RESOLUTIONS:
        print(f'@{res}')
        for g, v in tables[res].items():
            fid = f"{v['fid']:.2f}" if v['fid'] is not None else '-'
            steps = '  '.join(f"t{t}: {s['psnr']:.2f}" for t, s in v['per_step'].items())
            print(f"  {g:16s} PSNR {v['psnr']:.2f}  SSIM {v['ssim']:.4f}  LPIPS {v['lpips']:.4f}  FID {fid}   ({steps})")
    if 'token_acc' in result:
        print(f"token acc {result['token_acc']['model']:.3f} (copy {result['token_acc']['copy_last']:.3f})")
    print(f'saved {json_path} ({result["seconds"]}s)')


if __name__ == '__main__':
    main()
