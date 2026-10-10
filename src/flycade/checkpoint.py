"""Versioned full-state checkpoints, validated before reconstruction and atomic publication."""
import fcntl
import importlib.metadata
import json
import os
import platform
import random
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from flycade.errors import PreparationError
from flycade.graph import digest, inspect_graph
from flycade.game import GameEnv
from flycade.rom import inspect_registration

FORMAT = 1
STATE_CONTRACT = {
    'schedule': 'constant learning rate; fixed total update budget',
    'normalization': 'pixels / 255; per-rollout advantage normalization; no running statistics',
    'scaler': None, 'hidden_state': None, 'discounted_environment_returns': None,
    'sampler': 'torch global RNG; full rollout batch, no shuffle',
    'rollout': 'consumed before save; never restored',
    'periodic_save_or_evaluation': None,
    'reset': 'new episode; emulator, frame stack, action, position, timers and episode accumulators reset',
}


@contextmanager
def run_lock(output: Path) -> Iterator[None]:
    with (output / '.writer.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PreparationError('run_busy', 'Another process is writing this Run') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, allow_nan=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def capture_rng(env: GameEnv, device: str) -> dict[str, Any]:
    return {'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else [],
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'environment_rng': env.np_random.bit_generator.state,
            'action_rng': env.action_space.np_random.bit_generator.state,
            'observation_rng': env.observation_space.np_random.bit_generator.state}


def restore_rng(state: dict[str, Any], env: GameEnv, device: str) -> None:
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if device == 'cuda':
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
    env.np_random.bit_generator.state = state['environment_rng']
    env.action_space.np_random.bit_generator.state = state['action_rng']
    env.observation_space.np_random.bit_generator.state = state['observation_rng']


def save_checkpoint(output: Path, state: dict[str, Any]) -> dict[str, Any]:
    if not state['progress']['safe_boundary']:
        raise ValueError('Checkpoint requires a completed optimizer update')
    directory = output / 'checkpoints'
    directory.mkdir(exist_ok=True)
    checkpoint_id = str(uuid.uuid4())
    state = {**state, 'format_version': FORMAT, 'checkpoint_id': checkpoint_id,
             'state_contract': STATE_CONTRACT}
    destination = directory / f'{checkpoint_id}.pt'
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.partial', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
            checked = torch.load(temporary, map_location='cpu', weights_only=False)
            if checked['checkpoint_id'] != checkpoint_id or not checked['progress']['safe_boundary']:
                raise ValueError('Checkpoint readback failed')
            checksum = digest(temporary)
            os.replace(temporary, destination)
            sync_directory(directory)
        finally:
            temporary.unlink(missing_ok=True)
    metadata = {'format_version': FORMAT, 'checkpoint_id': checkpoint_id,
                'file': f'checkpoints/{checkpoint_id}.pt', 'sha256': checksum,
                'manifest_sha256': digest(output / 'run.json'),
                'run_id': state['manifest']['run_id'], 'session_id': state['progress']['session_id'],
                'updates': state['progress']['updates']}
    atomic_json(destination.with_suffix('.json'), metadata)
    # The single authoritative commit point. An interrupted save cannot replace it partially.
    atomic_json(output / 'latest.json', metadata)
    # Compatibility convenience for A3 consumers; resume always uses latest.json.
    alias = output / '.final.pt.partial'
    alias.unlink(missing_ok=True)
    os.symlink(metadata['file'], alias)
    os.replace(alias, output / 'final.pt')
    sync_directory(output)
    return metadata


def load_checkpoint(output: Path, home: Path) -> dict[str, Any]:
    def reject(message: str) -> None:
        raise PreparationError('checkpoint_incompatible', message)

    try:
        metadata = json.loads((output / 'latest.json').read_text())
        if metadata['format_version'] != FORMAT:
            reject('Unsupported checkpoint format')
        identifier = str(uuid.UUID(metadata['checkpoint_id']))
        if metadata['file'] != f'checkpoints/{identifier}.pt':
            reject('Invalid checkpoint path')
        path = output / metadata['file']
        if digest(output / 'run.json') != metadata['manifest_sha256']:
            reject('Run configuration/manifest changed')
        manifest = json.loads((output / 'run.json').read_text())
        if manifest['checkpoint_format'] != FORMAT:
            reject('Unsupported Run checkpoint format')
        if manifest['python'] != platform.python_version():
            reject('Python version differs')
        for package, version in manifest['versions'].items():
            if importlib.metadata.version(package) != version:
                reject(f'Package version differs: {package}')
        source = Path(__file__).parent
        current = {p.name: digest(p) for p in sorted(source.glob('*.py'))}
        if current != manifest['code']['files']:
            reject('Code differs; resume requires identical Python source hashes')
        lock = source.parent.parent / 'uv.lock'
        if (digest(lock) if lock.exists() else None) != manifest['code']['uv_lock_sha256']:
            reject('Package lock differs')
        if torch.version.cuda != manifest['cuda_runtime']:
            reject('CUDA runtime differs')
        current_graph = inspect_graph(output / 'graph')
        if any(current_graph[key] != value for key, value in manifest['graph'].items() if key != 'output'):
            reject('Graph artifact differs')
        if digest(output / 'initial.pt') != manifest['initial_sha256']:
            reject('Initial policy artifact differs')
        if not manifest['fixture'] and inspect_registration(home) != manifest['registration']:
            reject('ROM/state/integration differs')
        if digest(path) != metadata['sha256']:
            reject('Checkpoint checksum differs')
        # Only local trusted Run files: full Python/NumPy state requires pickle.
        state: dict[str, Any] = torch.load(path, map_location='cpu', weights_only=False)
        if (state['manifest'] != manifest or state['format_version'] != FORMAT
                or state['checkpoint_id'] != identifier or state['state_contract'] != STATE_CONTRACT
                or state['progress']['run_id'] != manifest['run_id']
                or not state['progress']['safe_boundary']):
            reject('Checkpoint identity or state contract differs')
        return state
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise PreparationError('checkpoint_incompatible', f'Missing or invalid checkpoint/artifact: {exc}') from exc
