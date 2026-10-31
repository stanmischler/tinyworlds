"""Gradient size of the flow loss vs the sampled-frame loss (models/sampled_frame_loss.py) on a few training batches of
the starting checkpoint, to place the λ grid: λ * |grad sampled| should be within ~0.1-1x |grad flow|.

    python experiments/sampled_lpips/grad_ratio.py --dynamics results/zelda_longer_fm/dynamics/checkpoints/dynamics_step_99999 \
        --tokenizer results/zelda_cont_final_tp/video_tokenizer/checkpoints/video_tokenizer_step_29000 \
        --actions results/como_actions_v1_full --batches 4
"""

import argparse
import glob
import os

import torch

from datasets.datasets import MultiVideoH5Dataset
from models.sampled_frame_loss import frozen_lpips, boxed_windows, box_lpips_l1
from utils.inference_utils import load_models


def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dynamics', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--actions', required=True)
    p.add_argument('--batches', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=32, help='64 does not fit uncompiled (training compiles)')
    p.add_argument('--sampled-batch', type=int, default=16)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--grad-steps', type=int, default=1)
    p.add_argument('--target', choices=['endpoint', 'clean_guess'], default='endpoint', help='as sampled_loss_target')
    p.add_argument('--base-only', action='store_true', help='base train frames only (no YouTube files); actions encoded online, close to but not exactly the training loader')
    p.add_argument('--extra-files', type=int, default=2, help='YouTube .h5 files to load (all = slow, ~16 GB)')
    a = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    tok, lam, dyn = load_models(a.tokenizer, a.actions, a.dynamics, device, use_actions=True)
    dyn.train()
    params = [q for q in dyn.parameters() if q.requires_grad]
    lpips_fn = frozen_lpips(device)
    extras = [] if a.base_only else sorted(glob.glob(os.getcwd() + '/data/zelda_youtube/*.h5'))[:a.extra_files]
    data = MultiVideoH5Dataset((os.getcwd() + '/data/zelda_train_frames.h5', 4, 1000), extras, num_frames=4)
    loader = torch.utils.data.DataLoader(data, batch_size=a.batch_size, shuffle=True)

    for b, (x, _) in zip(range(a.batches), loader):
        x = x.to(device)  # [B, T, C, H, W] in [-1, 1]
        with torch.no_grad():
            latents = tok.encode(x)  # [B, T, P, L]
            actions = lam.encode(x)  # [B, T-1, A]
        amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')  # as training (amp: true)
        with amp:
            _, _, flow_loss = dyn(latents, training=True, conditioning=actions)
        g_flow = grad_norm(flow_loss, params)

        unit = (x[:, -2:].float() + 1) / 2  # [B, 2, C, H, W]
        rows, boxes = boxed_windows(unit[:, 0], unit[:, 1], a.sampled_batch)
        if not len(rows):
            print(f'batch {b}: no boxed window', flush=True)
            continue
        context = latents[rows, :-1].float()  # [n, Tc, P, L]
        with amp:
            if a.target == 'clean_guess':
                z_next = dyn.clean_guess_with_grad(context, actions[rows], a.steps)  # [n, P, L]
            else:
                z_next = dyn.sample_next_with_grad(context, actions[rows], a.steps, a.grad_steps)  # [n, P, L]
        pred = (tok.decode(torch.cat([context, z_next[:, None]], dim=1))[:, -1] + 1) / 2  # [n, C, H, W]
        s_lpips, s_l1 = box_lpips_l1(lpips_fn, pred, unit[rows, 1], boxes)
        g_samp = grad_norm(s_lpips + s_l1, params)
        peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
        print(f'batch {b}: {len(rows)}/{len(x)} boxed | flow loss {flow_loss.item():.4f} grad {g_flow:.4f} | '
              f'sampled LPIPS {s_lpips.item():.4f} L1 {s_l1.item():.4f} grad {g_samp:.4f} | '
              f'ratio sampled/flow {g_samp / g_flow:.3f} | peak {peak_gb:.1f} GB', flush=True)


if __name__ == '__main__':
    main()
