"""Frame-pair action encoder distilled from teacher pseudo-labels (STA-35 itc_loop iteration 5, arm A).

A small CNN reads (frame t, frame t+gap) and predicts the teacher's code (0..7 = R, DR, D, DL, L, UL, U, UR, 8 = STILL)
from scripts/eval/itc_pseudo.py (column --label-key: code_itc = ITC teacher, code_pix = pixel-crop ablation teacher).
All train frames sit on the GPU as uint8; a held-out 5% of the train rows (every 20th block of 1000 frames) measures
teacher agreement. At the end it writes the judge codes {id: code} for scripts/eval/lam_judge.py score --codes.

  python scripts/train_action_encoder.py train --labels data/zelda_train_itc_pseudo_gap4.npz --out-dir results/<run>/action_encoder
  python scripts/train_action_encoder.py codes --ckpt results/<run>/action_encoder/encoder.pt --out codes.json
"""

import argparse
import json
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 9
MIRROR = torch.tensor([4, 3, 2, 1, 0, 7, 6, 5, 8])  # code under a horizontal flip (R <-> L, DR <-> DL, UR <-> UL)


class PairEncoder(nn.Module):
    def __init__(self, width=32, n_classes=N_CLASSES):
        super().__init__()
        chans = [9, width, 2 * width, 4 * width, 8 * width]  # input = frame t, frame t+1, their difference
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):  # 4 stride-2 stages: 128 -> 8
            blocks += [nn.Conv2d(cin, cout, 3, 2, 1), nn.GroupNorm(8, cout), nn.GELU(),
                       nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.pool_score = nn.Conv2d(chans[-1], 1, 1)  # attention pooling: the action lives in a few cells (Link)
        self.head = nn.Sequential(nn.LayerNorm(chans[-1]), nn.Linear(chans[-1], n_classes))

    def forward(self, fa, fb):
        # fa, fb: [B, C, H, W] in [-1, 1] -> logits [B, n_classes]
        h = self.body(torch.cat([fa, fb, fb - fa], 1))  # [B, E, Hp, Wp]
        w = self.pool_score(h).flatten(2).softmax(-1)  # [B, 1, Hp*Wp]
        return self.head((h.flatten(2) * w).sum(-1))  # [B, n_classes]


def to_float(x):
    # [B, H, W, C] uint8 -> [B, C, H, W] float in [-1, 1]
    return x.permute(0, 3, 1, 2).float() / 127.5 - 1.0


def train(args):
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    z = np.load(args.labels)
    gap = int(z['gap'])
    y_all = z[args.label_key].astype(np.int64)  # [N] label of the pair (t, t + gap), -1 = none
    t0 = time.time()
    with h5py.File(args.frames, 'r') as h:
        X = torch.from_numpy(h['frames'][:]).to(dev)  # [N, H, W, C] uint8
    print(f'frames {tuple(X.shape)} on {dev} in {time.time() - t0:.0f}s', flush=True)
    rows = np.nonzero(y_all >= 0)[0]
    val_mask = (rows // 1000) % 20 == 7
    tr, va = rows[~val_mask], rows[val_mask]
    counts = np.bincount(y_all[tr], minlength=N_CLASSES)
    print('train', len(tr), 'val', len(va), 'class counts', counts.tolist(), flush=True)
    w = (counts.sum() / np.maximum(counts, 1)) ** args.balance  # inverse-frequency^balance class weights
    w = torch.tensor(w / (w * counts).sum() * counts.sum(), dtype=torch.float32, device=dev)
    y = torch.from_numpy(y_all).to(dev)
    tr_t = torch.from_numpy(tr).to(dev)
    model = PairEncoder(args.width).to(dev)
    print('params', sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0))))
    mirror = MIRROR.to(dev)
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda')
    log = []
    for step in range(args.steps + 1):
        model.train()
        idx = tr_t[torch.randint(len(tr_t), (args.batch,), device=dev)]  # [B] rows t
        fa, fb, lab = to_float(X[idx]), to_float(X[idx + gap]), y[idx]
        if args.flip:  # Link's L / R sprites are mirror images
            f = torch.rand(args.batch, device=dev) < 0.5
            fa, fb = torch.where(f[:, None, None, None], fa.flip(-1), fa), torch.where(f[:, None, None, None], fb.flip(-1), fb)
            lab = torch.where(f, mirror[lab], lab)
        with torch.autocast(**amp):
            logits = model(fa, fb)
        loss = F.cross_entropy(logits.float(), lab, weight=w, label_smoothing=args.smooth)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.log_every == 0 or step == args.steps:
            acc, per = evaluate(model, X, y_all, va, gap, dev, amp)
            rec = dict(step=step, loss=round(loss.item(), 4), val_acc=acc, val_per_class=per, s=round(time.time() - t0))
            log.append(rec)
            print(json.dumps(rec), flush=True)
    torch.save({'model': model.state_dict(), 'args': vars(args), 'gap': gap}, os.path.join(args.out_dir, 'encoder.pt'))
    json.dump(log, open(os.path.join(args.out_dir, 'train_log.json'), 'w'), indent=0)
    if args.transitions:
        codes = judge_codes(model, args.transitions, args.test_frames, dev, amp)
        json.dump(codes, open(os.path.join(args.out_dir, 'codes_judge.json'), 'w'))
        print('judge code counts', np.bincount(list(codes.values()), minlength=N_CLASSES).tolist())
    print('ENCODER DONE', args.out_dir)


@torch.no_grad()
def evaluate(model, X, y_all, va, gap, dev, amp, bs=1024):
    # teacher agreement on held-out train rows -> (accuracy, per-class recall list)
    model.eval()
    pred = []
    for i in range(0, len(va), bs):
        idx = torch.from_numpy(va[i:i + bs]).to(dev)
        with torch.autocast(**amp):
            pred.append(model(to_float(X[idx]), to_float(X[idx + gap])).argmax(-1).cpu())
    pred, yt = torch.cat(pred).numpy(), y_all[va]
    per = [round(float((pred[yt == c] == c).mean()), 3) if (yt == c).any() else None for c in range(N_CLASSES)]
    return round(float((pred == yt).mean()), 4), per


@torch.no_grad()
def judge_codes(model, transitions, frames, dev, amp):
    # judge transitions (eval_results/lam_judge/set/transitions.json) -> {id: code}
    model.eval()
    tr = json.load(open(transitions))['transitions']
    with h5py.File(frames, 'r') as h:
        F_ = h['frames']
        fa = torch.from_numpy(np.stack([F_[t['frame_a']] for t in tr])).to(dev)
        fb = torch.from_numpy(np.stack([F_[t['frame_b']] for t in tr])).to(dev)
    with torch.autocast(**amp):
        pred = model(to_float(fa), to_float(fb)).argmax(-1).cpu().tolist()
    return {str(t['id']): int(c) for t, c in zip(tr, pred)}


def codes(args):
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = PairEncoder(ck['args']['width'])
    model.load_state_dict(ck['model'])
    dev = torch.device(args.device)
    c = judge_codes(model.to(dev), args.transitions, args.test_frames, dev, dict(device_type=dev.type, enabled=False))
    json.dump(c, open(args.out, 'w'))
    print('saved', args.out, np.bincount(list(c.values()), minlength=N_CLASSES).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['train', 'codes'])
    p.add_argument('--labels', default='data/zelda_train_itc_pseudo_gap4.npz')
    p.add_argument('--label-key', default='code_itc')
    p.add_argument('--frames', default='data/zelda_train_frames.h5')
    p.add_argument('--test-frames', default='data/zelda_test_frames.h5')
    p.add_argument('--transitions', default='eval_results/lam_judge/set/transitions.json')
    p.add_argument('--out-dir', default='results/action_encoder')
    p.add_argument('--ckpt', default='')
    p.add_argument('--out', default='codes_judge.json')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--balance', type=float, default=0.5, help='class weight = (1 / frequency)^balance (0 = plain CE)')
    p.add_argument('--smooth', type=float, default=0.05)
    p.add_argument('--flip', type=int, default=1)
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    {'train': train, 'codes': codes}[args.cmd](args)


if __name__ == '__main__':
    main()
