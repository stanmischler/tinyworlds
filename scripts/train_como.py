"""Train the CoMo motion IDM (models/como.py) on precomputed MAE features (scripts/como_features.py).

Each sample is a transition (t, t+gap) inside one contiguous stretch of the .h5 (train split cuts and the first
`start_index` frames excluded, as ZeldaDataset's load_start_index), plus a jittered end t+gap+delta (delta uniform in
{-jitter..jitter} minus 0) for the InfoNCE positive. The feature table and the frames sit on the GPU, so a step is a
gather + the IDM/decoder; no dataloader.

Logged: train recon MSE, InfoNCE, copy-frame-t MSE (the baseline a motion-blind decoder can reach), and on a fixed
held-out batch from the test .h5: recon / copy / recon with z shuffled across the batch (shuffle gap = how much the
decoder uses z), z spread (mean per-dim std, mean pairwise cosine). Every ckpt_every steps: checkpoint
(results/<run>/como/checkpoints/como_step_N, loadable by utils.utils.load_latent_actions_from_checkpoint), a
prediction PNG, and the held-out LAM judge (scripts/eval/lam_judge.py, k-means k = n_actions, and k = 8).

    python scripts/train_como.py --config configs/como/zelda.yaml -- run_name=como_zelda n_updates=50000
    python scripts/train_como.py --config configs/como/zelda.yaml -- fake_features=1500 device=cpu ...  # CPU code check
"""

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval'))
from models.como import CoMo, MAE_TOKENS, MAE_DIM  # noqa: E402
from utils.utils import COMO_ARCH_KEYS, MODEL_CHECKPOINT, OPTIMIZER_CHECKPOINT, STATE  # noqa: E402


def segments(h5_path, n_frames):
    """Contiguous [start, end) stretches of local indices: test_blocks_local for a test file, the cuts of
    split_dataset.py for a train file, else the whole file."""
    with h5py.File(h5_path, 'r') as f:
        attrs = dict(f.attrs)
    if 'test_blocks_local' in attrs:
        return [tuple(b) for b in json.loads(attrs['test_blocks_local'])]
    if attrs.get('split') == 'train':
        from split_dataset import split_masks
        _, is_train = split_masks(int(attrs['source_n_frames']), json.loads(attrs['test_blocks_source']), int(attrs['margin']))
        src = np.flatnonzero(is_train)
        assert len(src) == n_frames, (len(src), n_frames)
        cuts = np.flatnonzero(np.diff(src) != 1) + 1
        bounds = [0, *cuts.tolist(), n_frames]
        return list(zip(bounds[:-1], bounds[1:]))
    return [(0, n_frames)]


def valid_starts(segs, span, start_index=0):
    # t such that t .. t + span stay inside one segment: [M]
    return np.concatenate([np.arange(max(s, start_index), e - span) for s, e in segs if e - span > max(s, start_index)])


def load_split(cfg, h5_path, feat_path, device, fake=0):
    with h5py.File(h5_path, 'r') as f:
        n = fake or min(len(f['frames']), cfg.max_frames or len(f['frames']))
        frames = torch.from_numpy(f['frames'][:n]).to(device)  # uint8 [N, H, W, C]
    if fake:
        feats = torch.randn(n, MAE_TOKENS, MAE_DIM, dtype=torch.float16, device=device)
    else:
        mm = np.load(feat_path, mmap_mode='r')
        assert mm.shape[0] >= n, (mm.shape, n)
        feats = torch.empty((n, MAE_TOKENS, MAE_DIM), dtype=torch.float16, device=device)
        for i in range(0, n, 2048):  # chunked: the host never holds the whole table
            feats[i:i + 2048] = torch.from_numpy(np.ascontiguousarray(mm[i:min(i + 2048, n)])).to(device)
    if fake:
        return frames, feats, [(0, n)]
    with h5py.File(h5_path, 'r') as f:
        n_file = len(f['frames'])
    segs = [(s, min(e, n)) for s, e in segments(h5_path, n_file) if s < n]  # max_frames cuts the last segment
    return frames, feats, segs


def to_pixels(fr):
    # uint8 [B, H, W, C] -> float [-1, 1] [B, C, H, W]
    return fr.permute(0, 3, 1, 2).float() / 127.5 - 1


def batch(frames, feats, starts, idx, gap, jitter, gen, device):
    t = torch.as_tensor(starts[idx], device=device)  # [B]
    d = torch.randint(1, jitter + 1, (len(t),), generator=gen, device='cpu').to(device)
    d = d * (torch.randint(0, 2, (len(t),), generator=gen, device='cpu').to(device) * 2 - 1)  # +-1..jitter
    b, j = t + gap, t + gap + d
    return feats[t].float(), feats[b].float(), feats[j].float(), to_pixels(frames[t]), to_pixels(frames[b])


def save_png(path, x_a, x_b, pred, n=8):
    rows = torch.cat([torch.cat([x_a[i], x_b[i], pred[i].clamp(-1, 1)], 2) for i in range(min(n, len(x_a)))], 1)  # [C, nH, 3W]
    Image.fromarray(((rows + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(path)


@torch.no_grad()
def held_out(model, vb, amp):
    f_a, f_b, f_j, x_a, x_b = vb
    model.eval()
    with amp():
        out = model(f_a, f_b, f_j, x_a, x_b)
        z = out['z'].float().flatten(1)  # [B, A]
        shuf = F.mse_loss(model.decoder(x_a, out['z'].roll(1, 0)).float(), x_b)
    model.train()
    zn = F.normalize(z, dim=1)
    cos = (zn @ zn.T)[~torch.eye(len(z), dtype=torch.bool, device=z.device)].mean()
    return {'val/recon': out['recon'].item(), 'val/copy': F.mse_loss(x_a, x_b).item(), 'val/shuffled': shuf.item(),
            'val/shuffle_gap': shuf.item() - out['recon'].item(), 'val/nce': out.get('nce', torch.zeros(())).item(),
            'val/z_std': z.std(0).mean().item(), 'val/z_cos': cos.item()}, out['pred']


def judge(ckpt, run_dir, step, cfg):
    res = {}
    for k in sorted({cfg.n_actions, 8}):
        name = f'{cfg.run_name}_s{step}_k{k}'
        cmd = [sys.executable, 'scripts/eval/lam_judge.py', 'score', '--device', cfg.judge_device, '--lam', f'{name}={ckpt}', '--k', str(k)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:] if r.returncode else '', flush=True)
        if r.returncode == 0:
            sc = json.load(open(f'eval_results/lam_judge/{name}/score.json'))
            res[k] = sc
            dst = f'{run_dir}/evals/{name}'
            os.makedirs(dst, exist_ok=True)
            subprocess.run(['cp', '-r', f'eval_results/lam_judge/{name}/.', dst])
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/como/zelda.yaml')
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_dotlist([o for o in a.overrides if o != '--']))
    device = cfg.device
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = f'results/{cfg.run_name}'
    ckpt_dir = f'{run_dir}/como/checkpoints'
    viz_dir = f'{run_dir}/como/visualizations'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    t0 = time.time()
    frames, feats, segs = load_split(cfg, cfg.train_h5, cfg.train_features, device, cfg.fake_features)
    span = cfg.gap + cfg.jitter
    starts = valid_starts(segs, span, cfg.start_index if not cfg.fake_features else 0)
    vframes, vfeats, vsegs = load_split(cfg, cfg.test_h5, cfg.test_features, device, cfg.fake_features)
    vstarts = valid_starts(vsegs, span)
    vgen = torch.Generator().manual_seed(1)
    vidx = torch.randperm(len(vstarts), generator=vgen)[:cfg.val_batch].numpy()
    vb = batch(vframes, vfeats, vstarts, vidx, cfg.gap, cfg.jitter, vgen, device)
    print(f'train: {len(frames)} frames, {len(segs)} segments, {len(starts)} transitions; held-out {len(vstarts)} '
          f'(batch {len(vidx)}); loaded in {time.time() - t0:.0f} s', flush=True)

    model = CoMo(**{k: cfg[k] for k in COMO_ARCH_KEYS if k in cfg}).to(device)
    n_idm = sum(p.numel() for p in model.idm.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f'params: IDM {n_idm / 1e6:.1f}M, decoder {n_dec / 1e6:.1f}M', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=tuple(cfg.betas), eps=1e-8, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(cfg.lr_floor, 1 - s / cfg.n_updates))
    step0 = 0
    if cfg.checkpoint:
        model.load_state_dict(torch.load(f'{cfg.checkpoint}/{MODEL_CHECKPOINT}', map_location=device, weights_only=True))
        st = torch.load(f'{cfg.checkpoint}/{STATE}', map_location='cpu', weights_only=False)
        opt.load_state_dict(torch.load(f'{cfg.checkpoint}/{OPTIMIZER_CHECKPOINT}', map_location=device, weights_only=False))
        sched.load_state_dict(st['scheduler'])
        step0 = st['step']
        gen.manual_seed(cfg.seed + step0)
        print(f'resumed from {cfg.checkpoint} at step {step0}', flush=True)
    amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if cfg.amp and device.startswith('cuda') else contextlib.nullcontext
    fwd = torch.compile(model) if cfg.compile else model

    wb = None
    if cfg.use_wandb:
        import wandb
        wb = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=OmegaConf.to_container(cfg), resume='allow', id=cfg.run_name)
    log = open(f'{run_dir}/como/log.jsonl', 'a')
    model.train()
    tl = time.time()
    for step in range(step0, cfg.n_updates + 1):
        idx = torch.randint(0, len(starts), (cfg.batch_size,), generator=gen).numpy()
        f_a, f_b, f_j, x_a, x_b = batch(frames, feats, starts, idx, cfg.gap, cfg.jitter, gen, device)
        with amp():
            out = fwd(f_a, f_b, f_j, x_a, x_b)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if step % cfg.log_interval == 0:
            rec = {'step': step, 'loss': out['loss'].item(), 'recon': out['recon'].item(), 'nce': out.get('nce', torch.zeros(())).item(),
                   'copy': F.mse_loss(x_a, x_b).item(), 'grad_norm': gn.item(), 'lr': sched.get_last_lr()[0],
                   'it_s': cfg.log_interval / max(time.time() - tl, 1e-6) if step > step0 else 0.0}
            tl = time.time()
            if step % cfg.val_interval == 0:
                v, pred = held_out(model, vb, amp)
                rec.update(v)
                if step % cfg.ckpt_interval == 0:
                    save_png(f'{viz_dir}/pred_step_{step}.png', vb[3], vb[4], pred)
            print(' '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in rec.items()), flush=True)
            log.write(json.dumps(rec) + '\n')
            log.flush()
            if wb:
                wb.log(rec, step=step)
        if step > step0 and (step % cfg.ckpt_interval == 0 or step == cfg.n_updates):
            ck = f'{ckpt_dir}/como_step_{step}'
            os.makedirs(ck, exist_ok=True)
            torch.save(model.state_dict(), f'{ck}/{MODEL_CHECKPOINT}')
            torch.save(opt.state_dict(), f'{ck}/{OPTIMIZER_CHECKPOINT}')
            torch.save({'config': {**OmegaConf.to_container(cfg), 'model_type': 'como'}, 'scheduler': sched.state_dict(), 'step': step}, f'{ck}/{STATE}')
            print(f'CHECKPOINT {ck}', flush=True)
            if cfg.judge and not cfg.fake_features:
                res = judge(ck, run_dir, step, cfg)
                for k, sc in res.items():
                    rec = {'step': step, f'judge_k{k}/moves_nmi_adj': sc['moves_only']['nmi_adj'], f'judge_k{k}/all_nmi_adj': sc['all']['nmi_adj'],
                           f'judge_k{k}/moves_purity': sc['moves_only']['purity'], f'judge_k{k}/entropy': sc.get('entropy_nats'),
                           **{f'judge_k{k}/probe_{s}': v for s, v in sc.get('probe', {}).items()}}
                    print('JUDGE ' + json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + '\n')
                    log.flush()
                    if wb:
                        wb.log({kk: vv for kk, vv in rec.items() if kk != 'step' and vv is not None}, step=step)
    print(f'DONE {cfg.n_updates} steps in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
