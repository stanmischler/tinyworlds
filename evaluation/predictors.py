"""One-step next-frame predictors for the visual comparison grid (scripts/eval/visualise_predictions.py), on the same inputs
as the results-table evals:
  - tokenizer + actions + dynamics: as scripts/eval/eval_next_frame.py (teacher forced, true action into the target,
    target decoded together with its context)
  - CoMo decoder: as scripts/eval/eval_como_pred.py (frame t + CoMo action z(t, t+4), no tokenizer)
Each predictor maps one window x [1, T=context+1, C, H, W] in [-1, 1] and a torch.Generator to the predicted target
frame [C, H, W] in [0, 1]. The generator only feeds flow / Bi-flow / bridge noise, so a window's prediction does not
depend on which other windows are drawn."""

import os
from pathlib import Path

import torch

from evaluation.windows import to_unit
from utils.inference_utils import load_models
from utils.utils import load_latent_actions_from_checkpoint


def dynamics_paths(dynamics_path):
    """(tokenizer, actions) checkpoint paths saved in a dynamics checkpoint's config, for specs that omit them. Runs on
    Modal saved absolute container paths (/root/tinyworlds/results/...): keep them relative from 'results/' on."""
    cfg = torch.load(Path(dynamics_path) / 'state.pt', map_location='cpu', weights_only=False)['config']
    paths = []
    for key in ('video_tokenizer_path', 'latent_actions_path'):
        p = cfg.get(key)
        assert p, f'{dynamics_path} has no {key} in its config: give it in the model spec'
        if not os.path.exists(p) and '/results/' in p:
            p = 'results/' + p.split('/results/', 1)[1]
        paths.append(p)
    return paths


class DynamicsPredictor:
    """Frozen tokenizer + action model (LAM or CoMo actions) + dynamics model (MaskGIT, flow, FlowC2S, Bi-flow, bridge)."""

    def __init__(self, tokenizer_path, actions_path, dynamics_path, device, num_steps=1, bf_eps=None):
        self.tok, self.actions, self.dyn = load_models(tokenizer_path, actions_path, dynamics_path, device)  # as eval_next_frame
        if bf_eps is not None:
            assert hasattr(self.dyn, 'bf_eps'), 'bf_eps needs a Bi-flow dynamics checkpoint'
            self.dyn.bf_eps = bf_eps
        self.num_steps = num_steps
        self.is_flow = hasattr(self.dyn, 'sample_next')  # flow models draw noise; MaskGIT at temperature 0 does not

    @torch.no_grad()
    def __call__(self, x, generator):
        tok = self.tok
        lat = tok.encode(x)  # [1, T, P, L]; causal encoder: context latents equal encoding the context alone
        cond = self.actions.encode(x)  # [1, T-1, A]: every transition, incl. the true one into the target
        idx_to_lat = lambda idx: tok.quantizer.get_latents_from_indices(idx, dim=-1)  # MaskGIT only
        kw = {'generator': generator} if self.is_flow else {}
        pred_lat = self.dyn.forward_inference(lat[:, :-1], prediction_horizon=1, num_steps=self.num_steps,
                                              index_to_latents_fn=idx_to_lat, conditioning=cond, temperature=0.0,
                                              **kw)  # [1, T, P, L]: context + predicted target
        return to_unit(tok.decode(pred_lat)[0, -1])  # [C, H, W], decoded with its context as in training


class CoMoDecoderPredictor:
    """Frozen CoMo checkpoint used as the world model: decoder(frame t, z(t, t+4)). With an action dir, z is first snapped
    to the nearest of its train-fit k-means centroids (the '16 codes' rows)."""

    def __init__(self, como_path, device, centroids_dir=None):
        self.lam, _ = load_latent_actions_from_checkpoint(como_path, device)  # CoMoLAM: frozen features + IDM + decoder
        self.lam.eval()
        assert getattr(self.lam, 'history', 0) == 0, 'CoMo with history frames is not supported'
        assert not hasattr(self.lam.como, 'draw'), 'drawing-model CoMo is not supported'
        self.centroids = None
        if centroids_dir:
            st = torch.load(Path(centroids_dir) / 'state.pt', map_location='cpu', weights_only=False)
            self.centroids = st['centroids'].float().to(device)  # [K, A] raw z space
        idm = self.lam.como.idm
        self.z_shape = (1, idm.n_queries, idm.down[-1].out_features)  # [1, Q, L]

    @torch.no_grad()
    def __call__(self, x, generator):
        frame_t = x[:, -2]  # [1, C, H, W] last context frame
        z = self.lam.encode(x[:, -2:])[:, -1].float()  # [1, A]: action (t, t+4)
        if self.centroids is not None:
            z = self.centroids[torch.cdist(z, self.centroids).argmin(1)]  # [1, A] nearest train centroid
        return to_unit(self.lam.como.decoder(frame_t, z.view(self.z_shape))[0])  # [C, H, W]


def build_predictor(spec, device):
    """spec: dict with either `como` (+ optional `centroids`) or `dynamics` (+ optional `tokenizer`, `actions`, `steps`,
    `bf_eps`); see configs/visualise/zelda_all.yaml. Unknown keys fail, so a typo never renders the wrong model."""
    keys = {'label', 'como', 'centroids'} if 'como' in spec else {'label', 'dynamics', 'tokenizer', 'actions', 'steps', 'bf_eps'}
    assert set(spec) <= keys, f"{spec['label']}: unknown keys {sorted(set(spec) - keys)}, expected a subset of {sorted(keys)}"
    if 'como' in spec:
        return CoMoDecoderPredictor(spec['como'], device, spec.get('centroids'))
    tok_path, act_path = spec.get('tokenizer'), spec.get('actions')
    if not (tok_path and act_path):
        saved_tok, saved_act = dynamics_paths(spec['dynamics'])
        tok_path, act_path = tok_path or saved_tok, act_path or saved_act
    return DynamicsPredictor(tok_path, act_path, spec['dynamics'], device, int(spec.get('steps', 1)),
                             None if spec.get('bf_eps') is None else float(spec['bf_eps']))
