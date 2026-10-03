"""Held-out tokenizer eval: reconstruction quality and token stability across time, for one or more checkpoints.

    PYTHONPATH=. python scripts/eval/tokenizer_frame_eval.py \
        --run "v3 temporal=../test/results/zelda_v3_2026_09_26/video_tokenizer/checkpoints/video_tokenizer_step_29000" \
        --run "v5 per-frame=results/zelda_v5_frametok/video_tokenizer/checkpoints/video_tokenizer_step_29000" \
        --test-h5 data/zelda_test_frames.h5 --out eval_results/zelda_v5_frametok_tokenizer

Windows follow eval_next_frame.py (test blocks, frames `frame_skip` stored frames apart). Each window holds 2*T-1
frames f_0..f_{2T-2}; the first T are the training-style window W1 = f_0..f_{T-1}, the last T are W2 = f_{T-1}..f_{2T-2},
so f_{T-1} is the last frame of W1 and the first of W2. Metrics:
  - recon_window_{psnr,ssim}: W1 encoded -> FSQ indices -> decoded, as in training (all T frames)
  - recon_single_{psnr,ssim}: each frame of W1 encoded and decoded alone (T=1)
  - token_mismatch_position: fraction of patches of f_{T-1} whose code differs between W1 (position T-1) and W2 (position 0)
  - static_token_change: among patches pixel-identical (uint8) between consecutive frames of W1, fraction whose code changes
  - token_copy_rate: fraction of all patches whose code equals the previous frame's (copy-last token accuracy ceiling),
    next to pixel_static_rate, the fraction of patches that are pixel-identical to the previous frame
Writes <out>.json and <out>.png (rows: ground truth, then each run's T=4 reconstruction of the last frame of W1).
"""
import argparse, json, os, sys
import h5py, numpy as np, torch
sys.path.insert(0, os.getcwd())
from utils.utils import load_videotokenizer_from_checkpoint
from scripts.eval.eval_next_frame import psnr, ssim, test_windows, to_model_range, to_unit

p = argparse.ArgumentParser()
p.add_argument('--run', action='append', required=True, help='label=<checkpoint dir>')
p.add_argument('--test-h5', default='data/zelda_test_frames.h5')
p.add_argument('--context', type=int, default=4)
p.add_argument('--frame-skip', type=int, default=4)
p.add_argument('--sample-stride', type=int, default=32)
p.add_argument('--batch-size', type=int, default=8)
p.add_argument('--n-vis', type=int, default=6)
p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
p.add_argument('--out', default='eval_results/tokenizer_frame_eval')
args = p.parse_args()

T, k = args.context, args.frame_skip
# windows long enough for 2T-1 frames; test_windows(context=n) reserves n*k frames after the start
windows = test_windows(args.test_h5, 2 * T - 2, k, args.sample_stride)
with h5py.File(args.test_h5) as f:
    frames = np.stack([f['frames'][[s + i * k for i in range(2 * T - 1)]] for _, s in windows])  # [N, 2T-1, H, W, C] uint8
print(f'{len(windows)} windows of {2 * T - 1} frames from {args.test_h5}', flush=True)
vis_ids = np.linspace(0, len(windows) - 1, args.n_vis).astype(int)


def patch_static(u8, ps):
    # u8: [N, T, H, W, C] -> [N, T-1, P] True where the patch is pixel-identical to the previous frame's
    same = (u8[:, 1:] == u8[:, :-1]).all(-1)  # [N, T-1, H, W]
    N, Tm, H, W = same.shape
    same = same.reshape(N, Tm, H // ps, ps, W // ps, ps).all((3, 5))  # [N, T-1, Hp, Wp]
    return torch.from_numpy(same.reshape(N, Tm, -1))


def evaluate(ckpt):
    model, _ = load_videotokenizer_from_checkpoint(ckpt, device=args.device)
    model.eval()
    ps = model.encoder.patch_embed.patch_size
    acc = {key: [] for key in ['wp', 'ws', 'sp', 'ss']}
    mism, st_change, st_n, copy_eq, copy_n, vis = 0, 0, 0, 0, 0, {}
    with torch.no_grad():
        for i in range(0, len(frames), args.batch_size):
            u8 = frames[i:i + args.batch_size]
            x = to_model_range(u8, args.device)                          # [B, 2T-1, C, H, W]
            B = x.shape[0]
            w1, w2 = x[:, :T], x[:, T - 1:]
            idx1 = model.tokenize(w1)                                    # [B, T, P]
            idx2 = model.tokenize(w2)                                    # [B, T, P]
            rec = model.detokenize(model.quantizer.get_latents_from_indices(idx1)).clamp(-1, 1)  # [B, T, C, H, W]
            idx_s = model.tokenize(w1.reshape(B * T, 1, *w1.shape[2:]))  # [B*T, 1, P]
            rec_s = model.detokenize(model.quantizer.get_latents_from_indices(idx_s)).clamp(-1, 1).reshape(rec.shape)
            gt = to_unit(w1).flatten(0, 1)                               # [B*T, C, H, W]
            for key, r in (('w', rec), ('s', rec_s)):
                r = to_unit(r).flatten(0, 1)
                acc[key + 'p'].append(psnr(r, gt).cpu())
                acc[key + 's'].append(ssim(r, gt).cpu())
            mism += (idx1[:, T - 1] != idx2[:, 0]).sum().item()
            static = patch_static(u8[:, :T], ps)                        # [B, T-1, P]
            changed = (idx1[:, 1:] != idx1[:, :-1]).cpu()                # [B, T-1, P]
            st_change += (changed & static).sum().item(); st_n += static.sum().item()
            copy_eq += (~changed).sum().item(); copy_n += changed.numel()
            for j in range(B):
                if i + j in vis_ids:
                    vis[i + j] = to_unit(rec[j, T - 1]).permute(1, 2, 0).cpu().numpy()
    P = idx1.shape[-1]
    m = {key: float(torch.cat(v).mean()) for key, v in acc.items()}
    return {
        'recon_window_psnr': m['wp'], 'recon_window_ssim': m['ws'],
        'recon_single_psnr': m['sp'], 'recon_single_ssim': m['ss'],
        'token_mismatch_position': mism / (len(frames) * P),
        'static_token_change': st_change / max(st_n, 1),
        'token_copy_rate': copy_eq / copy_n,
        'pixel_static_rate': st_n / copy_n,
        'n_windows': len(frames), 'checkpoint': ckpt,
    }, vis


results, vis_all = {}, {}
for spec in args.run:
    label, ckpt = spec.split('=', 1)
    results[label], vis_all[label] = evaluate(ckpt)
    r = results[label]
    print(f"{label:24s} window {r['recon_window_psnr']:.2f} dB / {r['recon_window_ssim']:.3f}  single {r['recon_single_psnr']:.2f} dB / "
          f"{r['recon_single_ssim']:.3f}  pos-mismatch {r['token_mismatch_position']:.3f}  static-change {r['static_token_change']:.3f}  "
          f"copy {r['token_copy_rate']:.3f} (pixel-static {r['pixel_static_rate']:.3f})", flush=True)

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
with open(args.out + '.json', 'w') as f:
    json.dump({'test_h5': args.test_h5, 'context': T, 'frame_skip': k, 'sample_stride': args.sample_stride, 'runs': results}, f, indent=1)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = ['ground truth'] + list(results)
fig, axes = plt.subplots(len(rows), len(vis_ids), figsize=(2.2 * len(vis_ids), 2.3 * len(rows)), squeeze=False)
for c, w in enumerate(vis_ids):
    axes[0, c].imshow(frames[w, T - 1])
    for r, label in enumerate(results, start=1):
        img = vis_all[label][w]
        axes[r, c].imshow(img)
        gt = frames[w, T - 1].astype(np.float32) / 255
        axes[r, c].set_xlabel(f'{10 * np.log10(1 / max(((img - gt) ** 2).mean(), 1e-10)):.1f} dB', fontsize=8)
for r, label in enumerate(rows):
    axes[r, 0].set_ylabel(label, fontsize=8)
for a in axes.flat:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(args.out + '.png', dpi=120)
print('wrote', args.out + '.json', args.out + '.png')
