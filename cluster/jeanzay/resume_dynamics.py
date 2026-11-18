"""Train a dynamics model under a fixed run name, continuing from its last checkpoint when there is one: the same command for
the first job and for every chained job past the 20 h limit.

    python cluster/jeanzay/resume_dynamics.py <run name> --config <dynamics yaml> --training_config <training yaml> -- key=val

Runs scripts/dynamics/train_dynamics.py with the same arguments and outputs in results/<run name>/dynamics/. With a checkpoint
there, adds checkpoint=<the last complete one> (weights, EMA, optimizer, schedule and step count continue) and
seed=<the run's seed + the step it continues at>: the trainer seeds its batch and noise draws from `seed` alone, so the same
seed would replay the batches and noise of the run's first steps. Exits at once when the last step is already saved.
"""

import os
import sys

import torch

FILES = ('model_state_dict.pt', 'optim_state_dict.pt', 'state.pt')


def last_checkpoint(root):
    # -> (dir, state.pt dict) of the highest step saved whole, or (None, None). The EMA file is written last: a checkpoint
    # without it, when others have it, was cut by the end of a job
    steps = sorted((int(d.rsplit('_', 1)[1]) for d in os.listdir(root) if d.startswith('dynamics_step_')), reverse=True) if os.path.isdir(root) else []
    has_ema = lambda step: os.path.exists(f'{root}/dynamics_step_{step}/ema_state_dict.pt')  # noqa: E731
    for step in steps:
        ckpt = f'{root}/dynamics_step_{step}'
        if all(os.path.exists(f'{ckpt}/{f}') for f in FILES) and (has_ema(step) or not any(has_ema(s) for s in steps)):
            return ckpt, torch.load(f'{ckpt}/state.pt', map_location='cpu', weights_only=False)
    return None, None


def main():
    run, args = sys.argv[1], sys.argv[2:]
    os.environ['NG_RUN_ROOT_DIR'] = os.path.join(os.getcwd(), 'results', run)
    ckpt, state = last_checkpoint(f'results/{run}/dynamics/checkpoints')
    if ckpt:
        config = state['config'] if isinstance(state['config'], dict) else vars(state['config'])
        given = dict(a.lstrip('-').split('=', 1) for a in args if '=' in a)
        end = int(given.get('stop_step') or given.get('n_updates') or config.get('stop_step') or config['n_updates'])
        if state['step'] + 1 >= end:
            print(f'{run}: step {state["step"]} of {end} is saved, nothing left to train')
            return
        args = [*args, *([] if '--' in args else ['--']), f'checkpoint={ckpt}']
        if config.get('seed') is not None and 'seed' not in given:
            args.append(f'seed={config["seed"] + state["step"] + 1}')
    print(f'{run}: {"continuing from " + ckpt if ckpt else "first job"}', flush=True)
    os.execv(sys.executable, [sys.executable, '-u', 'scripts/dynamics/train_dynamics.py', *args])


if __name__ == '__main__':
    main()
