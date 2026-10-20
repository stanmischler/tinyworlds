"""Held-out score of a flow dynamics model during training: scripts/eval/eval_next_frame.py's protocol (same windows,
batch size and seed, actions from the action model on the window, decoded with the context), so at a given checkpoint
test/psnr matches the eval's 1-step number. For watching a run only; the reported number is still eval_next_frame.py."""

import h5py
import numpy as np
import torch

from evaluation.image_metrics import psnr, ssim
from evaluation.windows import test_windows, load_window_batch, to_model_range, to_unit


class TestCurve:
    def __init__(self, h5_path, context=3, frame_skip=4, sample_stride=8, batch_size=32):
        self.windows = test_windows(h5_path, context, frame_skip, sample_stride)
        with h5py.File(h5_path, 'r') as f:
            self.frames = load_window_batch(f['frames'], self.windows, context, frame_skip)  # uint8 [N, T, H, W, C]
        self.context, self.batch_size = context, batch_size

    def batches(self, device):
        for i in range(0, len(self.frames), self.batch_size):
            yield to_model_range(self.frames[i:i + self.batch_size], device)  # [B, T, C, H, W]

    @torch.no_grad()
    def __call__(self, dyn, tok, actions, device, num_steps=1, seed=0):
        # dyn: flow dynamics (EMA copy); tok: frozen tokenizer; actions: action model with encode(frames) -> [B, T-1, A]
        # -> {'test/psnr', 'test/ssim', 'test/flow_loss'}, means over all windows; a corrected model also gets
        # test/psnr_k0 (corrector off) and test/corr_loss, and test/flow_loss stays the dynamics net's own loss
        was_training, tf32 = dyn.training, torch.backends.cuda.matmul.allow_tf32
        dyn.eval()
        torch.backends.cuda.matmul.allow_tf32 = False  # full fp32, as the eval
        correction = getattr(dyn, 'correction', False)
        psnrs, ssims, psnrs_k0, loss_sum, corr_sum = [], [], [], 0.0, 0.0
        with torch.random.fork_rng(devices=[device] if torch.device(device).type == 'cuda' else []):
            torch.manual_seed(seed)  # the eval's seed: same sampling noise per window
            for x in self.batches(device):
                lat = tok.encode(x)  # [B, T, P, L]
                pred_lat = dyn.forward_inference(lat[:, :self.context], prediction_horizon=1, num_steps=num_steps,
                                                 conditioning=actions.encode(x))  # [B, T, P, L]
                pred = to_unit(tok.decode(pred_lat)[:, -1])  # [B, C, H, W], decoded with the context as in the eval
                tgt = to_unit(x[:, -1])  # [B, C, H, W]
                psnrs += psnr(pred, tgt).tolist()
                ssims += ssim(pred, tgt).tolist()
                if correction:  # same windows with the corrector off: the fine-tuned dynamics alone
                    k = dyn.correction_k
                    dyn.correction_k = 0
                    pred_lat = dyn.forward_inference(lat[:, :self.context], prediction_horizon=1, num_steps=num_steps,
                                                     conditioning=actions.encode(x))  # [B, T, P, L]
                    dyn.correction_k = k
                    psnrs_k0 += psnr(to_unit(tok.decode(pred_lat)[:, -1]), tgt).tolist()

            torch.manual_seed(seed)  # fixed noise and flow times, so the loss curve is not noisy
            for x in self.batches(device):
                _, aux, loss = dyn(tok.encode(x), training=True, conditioning=actions.encode(x))  # velocity MSE, as in training
                if correction:  # aux = (x1_hat, s, loss terms)
                    loss, corr_sum = aux[2]['flow_loss'], corr_sum + aux[2]['corr_loss'].item() * x.shape[0]
                loss_sum += loss.item() * x.shape[0]
        torch.backends.cuda.matmul.allow_tf32 = tf32
        dyn.train(was_training)
        out = {'test/psnr': float(np.mean(psnrs)), 'test/ssim': float(np.mean(ssims)), 'test/flow_loss': loss_sum / len(self.frames)}
        if correction:
            out.update({'test/psnr_k0': float(np.mean(psnrs_k0)), 'test/corr_loss': corr_sum / len(self.frames)})
        return out
