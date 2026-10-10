"""Versioned full-state checkpoints, validated before reconstruction and atomic publication."""
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import pickle
import random
import tempfile
import time
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
    'schedule': 'constant learning rate; total update budget extended only by explicit budget ledger',
    'normalization': 'pixels / 255; per-rollout advantage normalization; no running statistics',
    'scaler': None, 'hidden_state': None, 'discounted_environment_returns': None,
    'sampler': 'torch global RNG; full rollout batch, no shuffle',
    'rollout': 'consumed before save; never restored',
    'periodic_save_or_evaluation': 'next_autosave_seconds in cumulative training time; preserved across sessions',
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
                'updates': state['progress']['updates'], 'saved_unix': time.time()}
    atomic_json(destination.with_suffix('.json'), metadata)
    # The single authoritative commit point. An interrupted save cannot replace it partially.
    atomic_json(output / 'latest.json', metadata)
    atomic_json(directory / f'{checkpoint_id}.committed.json', metadata)
    # Compatibility convenience for A3 consumers; resume always uses latest.json.
    alias = output / '.final.pt.partial'
    alias.unlink(missing_ok=True)
    os.symlink(metadata['file'], alias)
    os.replace(alias, output / 'final.pt')
    sync_directory(output)
    return metadata


def read_checkpoint(output: Path, home: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    def reject(message: str) -> None:
        raise PreparationError('checkpoint_incompatible', message)

    try:
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
                or metadata['run_id'] != manifest['run_id']
                or metadata['session_id'] != state['progress']['session_id']
                or metadata['updates'] != state['progress']['updates']
                or not state['progress']['safe_boundary']):
            reject('Checkpoint identity or state contract differs')
        state['progress']['last_save'] = {'checkpoint_id': identifier, 'updates': metadata['updates'],
            'transitions': state['progress']['transitions'], 'completed_unix': metadata['saved_unix']}
        return state
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise PreparationError('checkpoint_incompatible', f'Missing or invalid checkpoint/artifact: {exc}') from exc


def read_metadata(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Checkpoint metadata must be an object')
    timestamp = value.get('saved_unix')
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp):
        raise ValueError('Invalid checkpoint timestamp')
    return value


def last_log_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('rb') as log:
        log.seek(max(0, path.stat().st_size - 65536))
        for line in reversed(log.read().splitlines()):
            try:
                row: dict[str, Any] = json.loads(line)
                if isinstance(row, dict):
                    return row
            except ValueError:
                continue
    return {}


def observed_progress(output: Path) -> dict[str, Any]:
    try:
        value: dict[str, Any] = json.loads((output / 'report.json').read_text())
        if not isinstance(value, dict):
            raise ValueError('Invalid progress report')
    except (OSError, ValueError):
        value = last_log_row(output / 'updates.jsonl')
    for name, key in (('updates', 'updates'), ('transitions', 'transition')):
        row = last_log_row(output / f'{name}.jsonl')
        if row.get('session_id') == value.get('session_id'):
            value[name] = max(value.get(name, 0), row.get(key, 0))
    return value


def load_checkpoint(output: Path, home: Path) -> dict[str, Any]:
    atomic_json(output / 'recovery-status.json', {'state': 'validating', 'started_unix': time.time()})
    rejected: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    try:
        candidates.append(read_metadata(output / 'latest.json'))
    except (OSError, ValueError) as exc:
        rejected.append({'checkpoint_id': 'latest', 'error': str(exc)})
    committed = []
    for path in (output / 'checkpoints').glob('*.committed.json'):
        try:
            committed.append(read_metadata(path))
        except (OSError, ValueError) as exc:
            rejected.append({'checkpoint_id': path.name, 'error': str(exc)})
    candidates.extend(sorted(committed, key=lambda m: m.get('saved_unix', 0), reverse=True))
    seen = set()
    before = observed_progress(output)
    for metadata in candidates:
        identifier = str(metadata.get('checkpoint_id'))
        if identifier in seen:
            continue
        seen.add(identifier)
        try:
            state = read_checkpoint(output, home, metadata)
        except (PreparationError, OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            rejected.append({'checkpoint_id': identifier, 'error': str(exc)})
            continue
        marker = output / 'checkpoints' / f'{identifier}.committed.json'
        if not marker.exists():
            atomic_json(marker, metadata)
        progress = state['progress']
        lost_updates = max(0, before.get('updates', 0) - progress['updates'])
        lost_transitions = max(0, before.get('transitions', 0) - progress['transitions'])
        recovery = {'state': 'validated', 'checkpoint_id': identifier, 'restored_updates': progress['updates'],
            'restored_transitions': progress['transitions'], 'lost_updates': lost_updates,
            'lost_transitions': lost_transitions, 'from_session_id': before.get('session_id'),
            'rejected_candidates': rejected, 'observed_updates': before.get('updates'),
            'note': 'Rollback is relative to last durable progress report; unlogged work may also be lost.'}
        atomic_json(output / 'recovery-status.json', recovery)
        if rejected or lost_updates or lost_transitions:
            recovery.update(recovery_id=str(uuid.uuid4()), run_id=state['manifest']['run_id'], created_unix=time.time())
            state['_recovery'] = recovery
            # Repair only after full state validation; no checkpoint content is rewritten.
            atomic_json(output / 'latest.json', metadata)
            alias = output / '.final.pt.partial'
            alias.unlink(missing_ok=True)
            os.symlink(metadata['file'], alias)
            os.replace(alias, output / 'final.pt')
            sync_directory(output)
        return state
    failure: dict[str, Any] = {'state': 'failed', 'rejected_candidates': rejected,
        'error': 'No valid checkpoint/artifact. Restore a matching local backup or create a new Run; partial weights are not resume.'}
    atomic_json(output / 'recovery-status.json', failure)
    raise PreparationError('checkpoint_incompatible', failure['error'] + ' ' + json.dumps(rejected))
