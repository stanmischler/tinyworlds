"""Frame-pair action classifier (frame t, frame t+gap -> one of N_CLASSES teacher codes), trained by
experiments/itc_actions/train_action_encoder.py and scored by scripts/eval/lam_eval.py --pair-encoder."""

import torch
import torch.nn as nn

N_CLASSES = 9


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
