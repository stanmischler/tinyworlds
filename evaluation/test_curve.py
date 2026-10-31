"""Held-out curves during training, on the test windows of scripts/eval/eval_next_frame.py (the test .h5 of a *_TRAIN
dataset, see datasets.data_utils.TEST_H5):
- TokenizerTestCurve: each window encoded and decoded noise-free (eval mode), target frame scored; the same loss on fixed
  training windows, so test/loss vs train_fixed/loss is the overfitting gap.
- TestCurve: flow dynamics, eval_next_frame.py's protocol (same windows, batch size and seed, actions from the action
  model on the window, decoded with the context), so at a given checkpoint test/psnr matches the eval's 1-step number.
For watching a run only; the reported numbers are still eval_next_frame.py's. Each pass also draws the same 8 windows
(first window of the first 8 test blocks, as in the eval's PNG)."""

from contextlib import contextmanager

import h5py
import numpy as np
import torch

from evaluation.image_grid import render_grid
from evaluation.image_metrics import psnr, ssim
from evaluation.windows import test_windows, load_window_batch, to_model_range, to_unit

N_IMAGE_WINDOWS = 8


def first_window_per_block(windows, n=N_IMAGE_WINDOWS):
    # [(block_id, start), ...] -> the first window of each of the first n blocks
    first = {}
    for block_id, s in windows:
        first.setdefault(block_id, (block_id, s))
    return [first[b] for b in sorted(first)[:n]]


@contextmanager
def full_fp32():
    # the eval runs without tf32; restore the training setting afterwards
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


class HeldOutWindows:
    """The eval's test windows (first `limit` of them, all if None) and the 8 image windows, as uint8 [N, T, H, W, C]."""

    def __init__(self, h5_path, context=3, frame_skip=4, sample_stride=8, batch_size=32, limit=None):
        assert limit is None or limit > 0, limit
        windows = test_windows(h5_path, context, frame_skip, sample_stride)
        self.image_windows = first_window_per_block(windows)  # from the full list, so a limit keeps the same picture
        self.windows = windows[:limit]
        with h5py.File(h5_path, 'r') as f:
            self.frames = load_window_batch(f['frames'], self.windows, context, frame_skip)  # uint8 [N, T, H, W, C]
            self.image_frames = load_window_batch(f['frames'], self.image_windows, context, frame_skip)  # uint8 [8, T, H, W, C]
        self.context, self.batch_size = context, batch_size

    def batches(self, device, frames=None):
        frames = self.frames if frames is None else frames
        for i in range(0, len(frames), self.batch_size):
            yield to_model_range(frames[i:i + self.batch_size], device)  # [B, T, C, H, W]


class TokenizerTestCurve(HeldOutWindows):
    def __init__(self, h5_path, train_data, context=3, frame_skip=4, sample_stride=8, batch_size=32, limit=None):
        # train_data: the training dataset (VideoHDF5Dataset); its in-memory frames give as many fixed training windows,
        # evenly spaced over the start indices training samples from
        super().__init__(h5_path, context, frame_skip, sample_stride, batch_size, limit)
        starts = np.linspace(0, len(train_data) - 1, len(self.windows)).round().astype(int)  # [N]
        if hasattr(train_data, 'window_rows'):  # MultiVideoH5Dataset: index != row, files at different frame rates
            idx = np.stack([train_data.window_rows(i) for i in starts])  # [N, T]
        else:
            idx = starts[:, None] + frame_skip * np.arange(context + 1)[None]  # [N, T]
        self.train_frames = train_data.data[idx]  # uint8 [N, T, H, W, C]

    @torch.no_grad()
    def __call__(self, tok, device):
        # tok: the unwrapped VideoTokenizer -> ({'test/recon_psnr', 'test/recon_ssim', 'test/loss', 'train_fixed/loss'},
        # PIL image of the 8 windows: one column per window, target row over reconstruction row)
        was_training = tok.training
        tok.eval()  # noise-free: mean latents, no latent noise, latent_scale not updated
        psnrs, ssims, test_loss, train_loss = [], [], 0.0, 0.0
        with full_fp32():
            for x in self.batches(device):
                loss, x_hat = tok(x)  # training-formula loss (smooth L1 + KL), x_hat [B, T, C, H, W]
                pred, tgt = to_unit(x_hat[:, -1]), to_unit(x[:, -1])  # [B, C, H, W]
                psnrs += psnr(pred, tgt).tolist()
                ssims += ssim(pred, tgt).tolist()
                test_loss += loss.item() * x.shape[0]  # every window has the same size: weighted mean = global mean
            for x in self.batches(device, self.train_frames):
                train_loss += tok(x)[0].item() * x.shape[0]
            x = next(self.batches(device, self.image_frames))  # [8, T, C, H, W]
            tgt, rec = to_unit(x[:, -1]).cpu(), to_unit(tok(x)[1][:, -1]).cpu()  # [8, C, H, W]
        tok.train(was_training)
        test_loss, train_loss = test_loss / len(self.frames), train_loss / len(self.train_frames)
        image = render_grid([list(tgt), list(rec)], [f'block {b}' for b, _ in self.image_windows], ['target', 'recon'],
                            [[f'{v:.1f} dB' if r else '' for v in psnr(rec, tgt).tolist()] for r in (0, 1)])
        metrics = {'test/recon_psnr': float(np.mean(psnrs)), 'test/recon_ssim': float(np.mean(ssims)),
                   'test/loss': test_loss, 'train_fixed/loss': train_loss}
        return metrics, image


class TestCurve(HeldOutWindows):
    def __init__(self, h5_path, context=3, frame_skip=4, sample_stride=8, batch_size=32, limit=None):
        super().__init__(h5_path, context, frame_skip, sample_stride, batch_size, limit)
        # copy-last baseline: the last context frame as the prediction; fixed, so computed once
        copy = [psnr(to_unit(x[:, -2]), to_unit(x[:, -1])) for x in self.batches('cpu')]  # [B] per batch
        self.copy_psnr = float(torch.cat(copy).mean())

    @torch.no_grad()
    def __call__(self, dyn, tok, actions, device, num_steps=1, seed=0):
        # dyn: flow dynamics (EMA copy); tok: frozen tokenizer; actions: action model with encode(frames) -> [B, T-1, A]
        # -> ({'test/psnr', 'test/ssim', 'test/flow_loss', 'test/copy_psnr'}, PIL image: one row per window), means over all
        # windows; for an EDM model test/flow_loss is its preconditioned denoiser loss and for a regression model its latent MSE, neither comparable with a flow model's; a corrected model also gets test/psnr_k0 (corrector off) and test/corr_loss, and test/flow_loss stays
        # the dynamics net's own loss
        was_training = dyn.training
        dyn.eval()
        correction = getattr(dyn, 'correction', False)
        psnrs, ssims, psnrs_k0, loss_sum, corr_sum = [], [], [], 0.0, 0.0

        def predict(x):
            # x [B, T, C, H, W] -> predicted target frame [B, C, H, W] in [0, 1], decoded with the context as in the eval
            lat = tok.encode(x)  # [B, T, P, L]
            pred_lat = dyn.forward_inference(lat[:, :self.context], prediction_horizon=1, num_steps=num_steps,
                                             conditioning=actions.encode(x))  # [B, T, P, L]
            return to_unit(tok.decode(pred_lat)[:, -1])

        with full_fp32(), torch.random.fork_rng(devices=[device] if torch.device(device).type == 'cuda' else []):
            torch.manual_seed(seed)  # the eval's seed: same sampling noise per window
            for x in self.batches(device):
                pred = predict(x)  # [B, C, H, W]
                tgt = to_unit(x[:, -1])  # [B, C, H, W]
                psnrs += psnr(pred, tgt).tolist()
                ssims += ssim(pred, tgt).tolist()
                if correction:  # same windows with the corrector off: the fine-tuned dynamics alone
                    k = dyn.correction_k
                    dyn.correction_k = 0
                    psnrs_k0 += psnr(predict(x), tgt).tolist()
                    dyn.correction_k = k

            torch.manual_seed(seed)  # fixed noise and flow times, so the loss curve is not noisy
            token_loss = getattr(dyn, 'token_loss', None)
            if token_loss is not None:
                dyn.token_loss = 'mse'  # plain velocity MSE whatever the training loss, so runs compare
            for x in self.batches(device):
                _, aux, loss = dyn(tok.encode(x), training=True, conditioning=actions.encode(x))  # plain velocity MSE
                if correction:  # aux = (x1_hat, s, loss terms)
                    loss, corr_sum = aux[2]['flow_loss'], corr_sum + aux[2]['corr_loss'].item() * x.shape[0]
                loss_sum += loss.item() * x.shape[0]
            if token_loss is not None:
                dyn.token_loss = token_loss

            torch.manual_seed(seed)
            x = next(self.batches(device, self.image_frames))  # [8, T, C, H, W]
            pred, tgt = predict(x).cpu(), to_unit(x[:, -1]).cpu()  # [8, C, H, W]
        dyn.train(was_training)

        context = to_unit(x[:, :self.context]).cpu()  # [8, context, C, H, W]
        err = (pred - tgt).abs().mean(1, keepdim=True).expand_as(pred)  # [8, C, H, W] grey |error|, mean over channels
        err = err / err.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)  # per-window scale, so small errors show
        image = render_grid([[*context[j], tgt[j], pred[j], err[j]] for j in range(len(tgt))],
                            [f'context {k + 1}' for k in range(self.context)] + ['target', 'prediction', '|error| (scaled)'],
                            [f'block {b}' for b, _ in self.image_windows],
                            [[''] * (self.context + 1) + [f'{v:.1f} dB', ''] for v in psnr(pred, tgt).tolist()])
        out = {'test/psnr': float(np.mean(psnrs)), 'test/ssim': float(np.mean(ssims)),
               'test/flow_loss': loss_sum / len(self.frames), 'test/copy_psnr': self.copy_psnr}
        if correction:
            out.update({'test/psnr_k0': float(np.mean(psnrs_k0)), 'test/corr_loss': corr_sum / len(self.frames)})
        return out, image
