"""Side-by-side one-step predictions of several models on held-out test windows, as one pixel-exact PNG.

Rows: N test windows, drawn as the first N of a fixed-seed shuffle of all windows of the test .h5 (same windows as
scripts/eval/eval_next_frame.py: `context` real frames + 1 target, `frame_skip` stored frames apart), so --n 20 and
--n 30 share their first 20 rows. Columns: last context frame (caption: copy-last PSNR), target, then one prediction
per model (caption: PSNR vs target; header: mean PSNR over the N rows). Flow-model noise is seeded per window, so a
row does not change with N or with the other models. Windows run one at a time (the eval batches 32), so per-window
PSNRs can differ from eval_next_frame's by float noise.

Models (evaluation/predictors.py), from a yaml list (configs/visualise/zelda_all.yaml) and/or repeated --model flags:
    --model "FlowC2S=results/zelda_copyflow_c2s_final/dynamics/checkpoints/dynamics_step_29000"
    --model "bridge 4 steps=results/zelda_bridge_final/dynamics/checkpoints/dynamics_step_29000,steps=4"
    --model "CoMo decoder=results/como_zelda_v1/como/checkpoints/como_step_50000"
  a dynamics checkpoint takes its tokenizer and actions from its own config unless tokenizer=/actions= are given;
  other keys: steps= (sampler / MaskGIT steps, default 1), bf_eps= (Bi-flow), centroids= (CoMo action dir: snap z to
  its train-fit centroids)

Outputs <out_dir>/<name>.png and <out_dir>/<name>.json (windows and per-cell PSNR).

Usage (from the repo root, PYTHONPATH=$PWD; on Modal: scripts/infra/modal_train.py::visualise_predictions):
    python scripts/eval/visualise_predictions.py --models configs/visualise/zelda_all.yaml --n 30 --name zelda_all
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch
import yaml

from evaluation.image_grid import render_grid
from evaluation.image_metrics import psnr
from evaluation.predictors import build_predictor
from evaluation.windows import load_window_batch, test_windows, to_model_range, to_unit


def parse_model_flag(flag):
    # "label=<checkpoint>[,key=value...]" -> spec dict; a CoMo checkpoint (config model_type 'como') is a decoder model
    head, *rest = flag.split(',')
    label, path = head.split('=', 1)
    cfg = torch.load(os.path.join(path, 'state.pt'), map_location='cpu', weights_only=False)['config']
    spec = {'label': label, ('como' if cfg.get('model_type') == 'como' else 'dynamics'): path}
    spec.update(kv.split('=', 1) for kv in rest)
    return spec


def pick_windows(test_h5, context, frame_skip, sample_stride, n, seed):
    """First n of a fixed-seed shuffle of all test windows -> ([(index in the full list, (block, start))], total windows)."""
    windows = test_windows(test_h5, context, frame_skip, sample_stride)
    order = np.random.default_rng(seed).permutation(len(windows))[:n]
    return [(int(i), windows[i]) for i in order], len(windows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', help='yaml with a `models:` list of specs (see configs/visualise/zelda_all.yaml)')
    p.add_argument('--model', action='append', default=[], help='"label=<checkpoint>[,key=value...]", repeatable')
    p.add_argument('--n', type=int, default=30, help='number of windows (rows)')
    p.add_argument('--seed', type=int, default=0, help='window shuffle and per-window flow noise')
    p.add_argument('--test-h5', default='data/zelda_test_frames.h5')
    p.add_argument('--context', type=int, default=3)
    p.add_argument('--frame-skip', type=int, default=4)
    p.add_argument('--sample-stride', type=int, default=8)
    p.add_argument('--scale', type=int, default=2, help='nearest-neighbour upscale of each frame')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    p.add_argument('--out-dir', default='eval_results/visualise')
    p.add_argument('--name', default='predictions')
    args = p.parse_args()

    specs = (yaml.safe_load(open(args.models))['models'] if args.models else []) + [parse_model_flag(f) for f in args.model]
    assert specs, 'give --models and/or --model'
    device = torch.device(args.device)
    picked, n_windows = pick_windows(args.test_h5, args.context, args.frame_skip, args.sample_stride, args.n, args.seed)
    with h5py.File(args.test_h5, 'r') as h5:
        source_index = h5['source_index'][:]
        x = to_model_range(load_window_batch(h5['frames'], [w for _, w in picked], args.context, args.frame_skip),
                           device)  # [N, T, C, H, W] in [-1, 1]
    last, target = to_unit(x[:, -2]), to_unit(x[:, -1])  # [N, C, H, W] last context frame, target
    copy_psnr = psnr(last, target).tolist()

    preds, psnrs = [], []  # per model: [N, C, H, W] on cpu, [N]
    for spec in specs:
        print(f"{spec['label']}: {spec.get('dynamics') or spec.get('como')}", flush=True)
        predictor = build_predictor(spec, device)
        out = []
        for k, (i, _) in enumerate(picked):
            gen = torch.Generator(device=device).manual_seed(args.seed * n_windows + i)  # one noise stream per window
            out.append(predictor(x[k:k + 1], gen))  # [C, H, W]
        pred = torch.stack(out)  # [N, C, H, W]
        preds.append(pred.cpu())
        psnrs.append(psnr(pred, target).tolist())
        print(f'  mean PSNR {np.mean(psnrs[-1]):.2f} dB over {len(picked)} windows (copy {np.mean(copy_psnr):.2f})', flush=True)
        del predictor
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    headers = [f'last frame ({np.mean(copy_psnr):.2f} dB)', 'target'] + \
              [f"{s['label']} ({np.mean(v):.2f} dB)" for s, v in zip(specs, psnrs)]
    rows = [[last[k].cpu(), target[k].cpu()] + [pr[k] for pr in preds] for k in range(len(picked))]
    captions = [[f'copy {copy_psnr[k]:.2f}', ''] + [f'{v[k]:.2f}' for v in psnrs] for k in range(len(picked))]
    row_labels = [f'#{k + 1}\nblock {b}\nsrc {source_index[s]}' for k, (_, (b, s)) in enumerate(picked)]
    os.makedirs(args.out_dir, exist_ok=True)
    png_path = os.path.join(args.out_dir, f'{args.name}.png')
    render_grid(rows, headers, row_labels, captions, scale=args.scale).save(png_path)

    result = {
        'config': {k: v for k, v in vars(args).items() if k != 'model'}, 'models': specs,
        'windows': [{'index': i, 'block': b, 'start': s, 'source_start': int(source_index[s]), 'copy_psnr': copy_psnr[k],
                     'psnr': {sp['label']: v[k] for sp, v in zip(specs, psnrs)}}
                    for k, (i, (b, s)) in enumerate(picked)],
        'mean_psnr': {'copy': float(np.mean(copy_psnr)), **{sp['label']: float(np.mean(v)) for sp, v in zip(specs, psnrs)}},
    }
    json_path = os.path.join(args.out_dir, f'{args.name}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=1)
    print(f'saved {png_path} and {json_path}')


if __name__ == '__main__':
    main()
