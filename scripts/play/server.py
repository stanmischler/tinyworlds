"""Play Bigfish inside the trained world model: a local web page (scripts/play/page/index.html) driving the frozen
tokenizer + flow dynamics model frame by frame over a WebSocket.

  PYTHONPATH="$PWD" ./.venv/bin/python scripts/play/server.py          # then open http://localhost:8765

Needs, from the repo root (Jean Zay run folder -> local `results/`):
  results/bigfish_tokenizer_long/video_tokenizer/checkpoints/video_tokenizer_step_99000
  results/bigfish_dynamics/dynamics/checkpoints/dynamics_step_99999      (ema_state_dict.pt, model_state_dict.pt, state.pt)
  results/bigfish_play/assets.npz                                         (scripts/play/export_play_assets.py)

Each step the page sends one 8-number action (a key's command, already standardized); the server appends it to the
last two actions, samples the next latent frame from noise in `steps` Euler steps, decodes it and sends it back. The
context is the last 3 latent frames, as in training. "Replay" mode feeds the true actions of the held-out clip instead
and also returns the real frame, to compare the dream with what happened.
"""
import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import numpy as np
import torch
from aiohttp import web, WSMsgType

from utils.utils import load_dynamics_from_checkpoint, load_videotokenizer_from_checkpoint

PAGE = Path(__file__).parent / 'page' / 'index.html'
CONTEXT = 3  # context frames the dynamics model was trained with (context_length 4 = 3 + the predicted one)


def b64(img):
    # uint8 [H, W, 3] -> base64 of the raw RGB bytes
    return base64.b64encode(np.ascontiguousarray(img).tobytes()).decode()


class World:
    """One player's game: latent context, action history, and the frozen models (shared)."""

    def __init__(self, tok, dyn, assets, device):
        self.tok, self.dyn, self.a, self.device = tok, dyn, assets, device
        self.start, self.t, self.mode = 0, 0, 'play'
        self.gen = torch.Generator(device=device)

    @torch.no_grad()
    def reset(self, start, seed):
        self.start, self.t = int(start), 0
        self.gen.manual_seed(int(seed))
        fr = torch.from_numpy(self.a['start_frames'][self.start, :CONTEXT]).to(self.device)  # [3, H, W, C] uint8
        x = fr.permute(0, 3, 1, 2)[None].float() / 127.5 - 1  # [1, 3, C, H, W] in [-1, 1]
        self.lat = self.tok.encode(x)  # [1, 3, P, L]
        self.act = torch.from_numpy(self.a['start_actions'][self.start, :CONTEXT - 1]).to(self.device)[None]  # [1, 2, A]
        return [self.a['start_frames'][self.start, i] for i in range(CONTEXT)]

    @torch.no_grad()
    def step(self, action, steps):
        # action: [A] standardized -> next frame uint8 [H, W, 3]
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device)[None, None]  # [1, 1, A]
        cond = torch.cat([self.act[:, -(CONTEXT - 1):], a], 1)  # [1, 3, A]; the last drives the new frame
        new = self.dyn.sample_next(self.lat[:, -CONTEXT:], cond, int(steps), generator=self.gen)  # [1, P, L]
        img = self.tok.decode(new[:, None].to(self.lat.dtype))[0, 0]  # [C, H, W] in [-1, 1]
        self.lat = torch.cat([self.lat[:, -(CONTEXT - 1):], new[:, None].to(self.lat.dtype)], 1)
        self.act = torch.cat([self.act[:, -(CONTEXT - 2):], a], 1)
        self.t += 1
        return ((img.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()

    def true_action(self):
        # the held-out clip's own action for the next transition (None past the stored horizon)
        acts = self.a['start_actions'][self.start]  # [L-1, A]
        i = CONTEXT - 1 + self.t
        return None if i >= len(acts) else acts[i]

    def real_frame(self):
        return self.a['start_frames'][self.start, CONTEXT + self.t - 1]


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))


def build_app(tok, dyn, assets, device):
    lock = asyncio.Lock()  # one GPU: steps of all pages run one at a time
    loop_exec = None

    async def index(_):
        return web.FileResponse(PAGE)

    async def meta(_):
        a = assets
        return web.json_response({
            'commands': a['commands'].tolist(), 'command_names': a['command_names'].tolist(),
            'command_counts': a['command_counts'].tolist(),
            'centroids': a['centroids'].tolist(), 'centroid_class': a['centroid_class'].tolist(),
            'centroid_purity': a['centroid_purity'].tolist(),
            'starts': [b64(a['start_frames'][i, CONTEXT - 1]) for i in range(len(a['start_frames']))],
            'horizon': int(a['start_frames'].shape[1] - CONTEXT), 'device': device,
        })

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=1 << 22)
        await ws.prepare(request)
        world = World(tok, dyn, assets, device)
        aloop = asyncio.get_running_loop()
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            m = json.loads(msg.data)
            op = m.get('op')
            if op == 'reset':
                async with lock:
                    frames = await aloop.run_in_executor(loop_exec, world.reset, m.get('start', 0), m.get('seed', 0))
                world.mode = m.get('mode', 'play')
                await ws.send_json({'op': 'reset', 'start': world.start, 'frames': [b64(f) for f in frames]})
            elif op == 'step':
                replay = world.mode == 'replay'
                action = world.true_action() if replay else np.asarray(m['action'], np.float32)
                if action is None:
                    await ws.send_json({'op': 'end'})
                    continue
                t0 = time.perf_counter()
                async with lock:
                    img = await aloop.run_in_executor(loop_exec, world.step, action, m.get('steps', 8))
                out = {'op': 'frame', 'id': m.get('id'), 't': world.t, 'frame': b64(img),
                       'ms': round(1000 * (time.perf_counter() - t0)), 'action': np.asarray(action).tolist()}
                if replay:
                    real = world.real_frame()
                    out.update(real=b64(real), psnr=psnr(img, real),
                               button=int(assets['start_buttons'][world.start, CONTEXT - 1 + world.t - 1]))
                await ws.send_json(out)
        return ws

    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/meta', meta)
    app.router.add_get('/ws', ws_handler)
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', default='results/bigfish_tokenizer_long/video_tokenizer/checkpoints/video_tokenizer_step_99000')
    p.add_argument('--dynamics', default='results/bigfish_dynamics/dynamics/checkpoints/dynamics_step_99999')
    p.add_argument('--assets', default='results/bigfish_play/assets.npz')
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    a = p.parse_args()

    tok, _ = load_videotokenizer_from_checkpoint(a.tokenizer, a.device)
    dyn, _ = load_dynamics_from_checkpoint(a.dynamics, a.device)  # EMA weights, as in the evals
    tok.eval(), dyn.eval()
    assets = dict(np.load(a.assets))
    world = World(tok, dyn, assets, a.device)  # warm-up: first MPS kernels compile here, not on the first key press
    world.reset(0, 0), world.step(assets['commands'][4], 2)
    print(f'Bigfish world model on {a.device}: open http://{a.host}:{a.port}', flush=True)
    web.run_app(build_app(tok, dyn, assets, a.device), host=a.host, port=a.port, print=None)


if __name__ == '__main__':
    main()
