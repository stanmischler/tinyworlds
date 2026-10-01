"""Held-out reconstruction PSNR of video-tokenizer checkpoints vs training step, for several runs on one figure.

    PYTHONPATH=. python scripts/eval/tokenizer_psnr_vs_step.py \
        --run "v3 small (128/256/4)=../test/results/sonic_v3_2026_09_23/video_tokenizer/checkpoints" \
        --run "v4 small, warm start +0=results/sonic_v4_2026_10_01/video_tokenizer/checkpoints:7500" \
        --out eval_results/sonic_v4_tokenizer_psnr_vs_step.png

Each --run is "label=<checkpoints dir>[:step_offset]"; the offset is added to the step parsed from the checkpoint
directory names (for warm-started runs, pass the step the warm start came from). Windows are context_length
consecutive stored frames, spread uniformly over the test .h5 (same frames for every checkpoint, encoded and decoded
through the full tokenizer: encoder -> FSQ indices -> latents -> decoder).
"""
import argparse, glob, json, os, re, sys
import h5py, numpy as np, torch
sys.path.insert(0, os.getcwd())
from utils.utils import load_videotokenizer_from_checkpoint

p = argparse.ArgumentParser()
p.add_argument('--run', action='append', required=True, help='label=<checkpoints dir>[:step_offset]')
p.add_argument('--test-h5', default='data/sonic_test_frames.h5')
p.add_argument('--windows', type=int, default=64)
p.add_argument('--context', type=int, default=4)
p.add_argument('--batch-size', type=int, default=16)
p.add_argument('--device', default='cpu')
p.add_argument('--out', default='eval_results/tokenizer_psnr_vs_step.png')
args = p.parse_args()

with h5py.File(args.test_h5) as f:
    N = f['frames'].shape[0]
    starts = np.linspace(0, N - args.context, args.windows).astype(int)
    x = np.stack([f['frames'][s:s + args.context] for s in starts])  # [W, T, H, W, C] uint8
x = torch.from_numpy(x).float().permute(0, 1, 4, 2, 3) / 127.5 - 1.0  # [W, T, C, H, W] in [-1, 1]

def psnr_of(ckpt):
    model, _ = load_videotokenizer_from_checkpoint(ckpt, device=args.device)
    model.eval()
    mses, codes = [], set()
    with torch.no_grad():
        for i in range(0, len(x), args.batch_size):
            xb = x[i:i + args.batch_size].to(args.device)
            idx = model.tokenize(xb)                                    # [B, T, P]
            z = model.quantizer.get_latents_from_indices(idx)           # [B, T, P, L]
            xh = model.detokenize(z).clamp(-1, 1)                       # [B, T, C, H, W]
            mses.append(((xh - xb) ** 2).mean(dim=(1, 2, 3, 4)).cpu())  # per window
            codes.update(idx.unique().tolist())
    mse = torch.cat(mses)
    return float((10 * torch.log10(4.0 / mse)).mean()), len(codes), model.codebook_size

results = {}
for spec in args.run:
    label, rest = spec.split('=', 1)
    ckpt_dir, _, off = rest.partition(':')
    off = int(off) if off else 0
    pts = []
    for d in sorted(glob.glob(os.path.join(ckpt_dir, '*_step_*')), key=lambda d: int(re.search(r'_step_(\d+)', d).group(1))):
        step = int(re.search(r'_step_(\d+)', d).group(1)) + off
        ps, nc, cs = psnr_of(d)
        pts.append({'step': step, 'psnr': ps, 'codes_used': nc, 'codebook': cs})
        print(f'{label:32s} step {step:6d}  PSNR {ps:6.2f} dB  codes {nc}/{cs}', flush=True)
    results[label] = pts

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
    json.dump({'test_h5': args.test_h5, 'windows': args.windows, 'context': args.context, 'runs': results}, f, indent=1)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.5))
for label, pts in results.items():
    ax.plot([q['step'] for q in pts], [q['psnr'] for q in pts], marker='o', ms=3, label=label)
ax.set_xlabel('training step (incl. warm-start offset)'); ax.set_ylabel(f'held-out PSNR (dB), {args.windows} windows')
ax.set_title(os.path.basename(args.test_h5)); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(args.out, dpi=130)
print('wrote', args.out)
