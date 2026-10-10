"""Immutable policy-only artifacts for evaluation, independent of optimizer/resume loading."""
import importlib.metadata
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import torch

from flycade.checkpoint import atomic_json, sync_directory
from flycade.errors import PreparationError
from flycade.graph import digest, inspect_graph
from flycade.policy import ConnectomePolicy, policy_from_graph
from flycade.rom import inspect_registration

POLICY_SOURCES = ('policy.py', 'game.py', 'nes.py', 'rom.py', 'training_fixture.py')


def publish_snapshot(run: Path, state: dict[str, Any], manifest: dict[str, Any],
                     snapshot_id: str, updates: int) -> None:
    directory = run / 'snapshots'
    directory.mkdir(exist_ok=True)
    metadata_path = directory / f'{snapshot_id}.json'
    if metadata_path.exists():
        raise ValueError(f'Snapshot already exists: {snapshot_id}')
    if snapshot_id == 'initial':
        weights = run / 'initial.pt'
    else:
        weights = directory / f'{snapshot_id}.pt'
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.partial', delete=False) as handle:
            temporary = Path(handle.name)
            try:
                torch.save({name: value.detach().cpu() for name, value in state.items()}, handle)
                handle.flush()
                os.fsync(handle.fileno())
                torch.load(temporary, map_location='cpu', weights_only=True)
                os.replace(temporary, weights)
            finally:
                temporary.unlink(missing_ok=True)
        weights.chmod(0o444)
        sync_directory(directory)
    atomic_json(metadata_path, {'format_version': 1, 'snapshot_id': snapshot_id,
        'run_id': manifest['run_id'], 'updates': updates, 'weights': str(weights.relative_to(run)),
        'weights_sha256': digest(weights), 'manifest_sha256': digest(run / 'run.json'),
        'normalization': 'pixels / 255; no running observation or reward statistics',
        'checkpoint_id': None if snapshot_id == 'initial' else snapshot_id})
    metadata_path.chmod(0o444)
    atomic_json(directory / 'latest.json', {'snapshot_id': snapshot_id})


def load_snapshot(run: Path, selection: str, home: Path, device: str) -> tuple[ConnectomePolicy, dict[str, Any], dict[str, Any]]:
    if device == 'cuda' and not torch.cuda.is_available():
        raise PreparationError('cuda_unavailable', 'CUDA requested but unavailable')
    if selection == 'latest':
        selection = json.loads((run / 'snapshots' / 'latest.json').read_text())['snapshot_id']
    if selection != 'initial':
        selection = str(uuid.UUID(selection))
    metadata = json.loads((run / 'snapshots' / f'{selection}.json').read_text())
    manifest = json.loads((run / 'run.json').read_text())
    if manifest['fixture'] and device != 'cpu':
        raise ValueError('Synthetic evaluation fixture requires --device cpu')
    expected_weights = 'initial.pt' if selection == 'initial' else f'snapshots/{selection}.pt'
    if (metadata['format_version'] != 1 or metadata['snapshot_id'] != selection
            or metadata['run_id'] != manifest['run_id'] or metadata['weights'] != expected_weights
            or metadata['manifest_sha256'] != digest(run / 'run.json')
            or metadata['weights_sha256'] != digest(run / expected_weights)):
        raise PreparationError('snapshot_incompatible', 'Snapshot identity, manifest or weights changed')
    if inspect_graph(run / 'graph')['graph_sha256'] != manifest['graph']['graph_sha256']:
        raise PreparationError('snapshot_incompatible', 'Snapshot graph differs')
    for filename in POLICY_SOURCES:
        if digest(Path(__file__).parent / filename) != manifest['code']['files'][filename]:
            raise PreparationError('snapshot_incompatible', f'Policy/environment code differs: {filename}')
    for package in ('torch', 'numpy', 'gymnasium', 'pillow', 'stable-retro'):
        if importlib.metadata.version(package) != manifest['versions'][package]:
            raise PreparationError('snapshot_incompatible', f'Package differs: {package}')
    if not manifest['fixture'] and inspect_registration(home) != manifest['registration']:
        raise PreparationError('snapshot_incompatible', 'ROM/state/integration differs')
    graph = run / 'graph'
    model, game = manifest['model'], manifest['game']
    policy = policy_from_graph(graph, model['state_dim'], model['propagation_steps'],
        game['frame_stack'] * (3 if game['color'] == 'RGB' else 1)).to(device)
    policy.load_state_dict(torch.load(run / expected_weights, map_location=device, weights_only=True), strict=True)
    policy.eval()
    policy.requires_grad_(False)
    return policy, metadata, manifest
