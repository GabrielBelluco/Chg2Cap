"""Validated epoch-boundary resume; no writes to the source run."""
import math
import shutil
from pathlib import Path

import torch


def validate_resume(state, vocab, config, epochs):
    required = ('encoder_dict', 'encoder_trans_dict', 'decoder_dict', 'optimizer_dict',
                'torch_rng', 'cuda_rng', 'data_generator_rng', 'epoch', 'steps', 'validation_loss', 'config')
    if any(k not in state for k in required):
        raise ValueError('Checkpoint lacks model/optimizer/RNG metadata')
    if state.get('vocab') != vocab:
        raise ValueError('Resume vocabulary mismatch')
    for key in ('manifest_sha256', 'vocab_sha256', 'max_length', 'batch_size', 'seed'):
        if state['config'].get(key) != config[key]:
            raise ValueError('Resume configuration mismatch: ' + key)
    if not state.get('full_epoch', True):
        raise ValueError('Partial-epoch resume is not supported; use a full-epoch best checkpoint')
    if not 1 <= state['epoch'] < epochs or state['steps'] <= 0 or not math.isfinite(state['validation_loss']):
        raise ValueError('Invalid epoch/step/loss or no epochs remaining')


def restore_training(state, models, optimizer, generator, cuda=True):
    for model, key in zip(models, ('encoder_dict', 'encoder_trans_dict', 'decoder_dict')):
        model.load_state_dict(state[key], strict=True)
    optimizer.load_state_dict(state['optimizer_dict'])
    # load_state_dict moves Adam parameter states to the parameter devices.
    generator.set_state(state['data_generator_rng'])
    torch.set_rng_state(state['torch_rng'])
    if cuda:
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def require_disk_space(directory, checkpoint_bytes, copies=1):
    directory = Path(directory)
    while not directory.exists():
        directory = directory.parent
    required = copies * checkpoint_bytes + 1024 ** 3
    available = shutil.disk_usage(directory).free
    if available < required:
        raise OSError(f'Insufficient disk space: need {required / 1024**3:.1f} GiB, have {available / 1024**3:.1f} GiB. Source checkpoint preserved.')
