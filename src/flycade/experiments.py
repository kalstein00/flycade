"""Explicit experiment lineage; parent artifacts are read-only inputs."""
import copy
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from flycade.checkpoint import atomic_json, read_checkpoint, read_metadata, run_lock, save_checkpoint, sync_directory
from flycade.errors import PreparationError
from flycade.graph import digest, inspect_graph
from flycade.snapshot import publish_snapshot


def selected_state(parent: Path, checkpoint_id: str, home: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identifier = str(uuid.UUID(checkpoint_id))
    metadata = read_metadata(parent / 'checkpoints' / f'{identifier}.committed.json')
    state = read_checkpoint(parent, home, metadata)
    lineage = {'parent_run_id': state['manifest']['run_id'], 'parent_path': str(parent.resolve()),
               'parent_checkpoint_id': identifier, 'parent_checkpoint_sha256': metadata['sha256'],
               'parent_manifest_sha256': metadata['manifest_sha256'],
               'parent_updates': state['progress']['updates'],
               'graph_sha256': state['manifest']['graph']['graph_sha256']}
    return state, lineage


def branch(parent: Path, checkpoint_id: str, output: Path, home: Path) -> dict[str, Any]:
    if output.exists():
        raise PreparationError('output_exists', f'Refusing to overwrite Run {output}')
    state, lineage = selected_state(parent, checkpoint_id, home)
    lineage.update(kind='full_state_branch', inherited='model, optimizer, RNG, counters and schedules',
                   initial_policy='inherited ancestor initial; not the branch starting checkpoint')
    from flycade.budget import budget_info
    budget = budget_info(parent, state['manifest'])
    if state['progress'].get('budget', {}).get('total_updates', 0) > budget['total_updates']:
        raise ValueError('Parent budget ledger is older than checkpoint')
    lineage['parent_budget'] = budget
    manifest = copy.deepcopy(state['manifest'])
    manifest['training']['updates'] = budget['total_updates']
    manifest.update(run_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()), created_unix=time.time(), lineage=lineage)
    output.mkdir(parents=True)
    with run_lock(output):
        shutil.copytree(parent / 'graph', output / 'graph')
        if inspect_graph(output / 'graph')['graph_sha256'] != lineage['graph_sha256']:
            raise ValueError('Parent graph changed while copying')
        origin = output / 'origin'
        origin.mkdir()
        shutil.copyfile(parent / 'checkpoints' / f"{lineage['parent_checkpoint_id']}.pt", origin / 'checkpoint.pt')
        shutil.copyfile(parent / 'run.json', origin / 'run.json')
        if (digest(origin / 'checkpoint.pt') != lineage['parent_checkpoint_sha256']
                or digest(origin / 'run.json') != lineage['parent_manifest_sha256']):
            raise ValueError('Parent checkpoint changed while copying')
        lineage['protected_origin'] = 'origin/checkpoint.pt; original parent bytes, independent of parent retention'
        shutil.copyfile(parent / 'initial.pt', output / 'initial.pt')
        (output / 'initial.pt').chmod(0o444)
        if digest(output / 'initial.pt') != manifest['initial_sha256']:
            raise ValueError('Parent initial policy changed while copying')
        for path in [output / 'initial.pt', *origin.iterdir(), *(output / 'graph').iterdir()]:
            if path.is_file():
                with path.open('rb') as handle:
                    os.fsync(handle.fileno())
        sync_directory(origin)
        sync_directory(output / 'graph')
        atomic_json(output / 'run.json', manifest)
        progress = copy.deepcopy(state['progress'])
        protocol = parent / 'initial-evaluation-protocol.json'
        if protocol.exists():
            shutil.copyfile(protocol, output / protocol.name)
            (output / protocol.name).chmod(0o444)
        if progress.get('evaluation_history'):
            from flycade.evaluation_index import evaluation_catalog
            catalog = evaluation_catalog(parent)
            atomic_json(origin / 'evaluation-index.json', catalog)
            # Preserve ancestor evidence separately: it must never enter the child's ranking.
            for directory in ('evaluations', 'snapshots', 'protocols'):
                if (parent / directory).exists():
                    shutil.copytree(parent / directory, origin / directory)
            progress['inherited_evaluation_history'] = {
                'run_id': state['manifest']['run_id'],
                'evaluation_ids': progress['evaluation_history'],
                'archive': 'origin/evaluation-index.json'}
            progress['evaluation_history'] = []
            progress.pop('initial_evaluation_id', None)
            for path in origin.rglob('*'):
                if path.is_file():
                    with path.open('rb') as handle:
                        os.fsync(handle.fileno())
            sync_directory(origin)
        for key in ('recovery', 'error', 'checkpoint_id', 'last_save', 'resumed_from'):
            progress.pop(key, None)
        progress.update(run_id=manifest['run_id'], session_id=manifest['session_id'], status='saved', lineage=lineage)
        progress['budget'] = budget_info(output, manifest)
        state.update(manifest=manifest, progress=progress)
        metadata = save_checkpoint(output, state)
        initial = torch.load(output / 'initial.pt', map_location='cpu', weights_only=True)
        publish_snapshot(output, initial, manifest, 'initial', 0)
        publish_snapshot(output, state['model'], manifest, metadata['checkpoint_id'], progress['updates'])
        progress.update(checkpoint_id=metadata['checkpoint_id'], last_save={
            'checkpoint_id': metadata['checkpoint_id'], 'updates': progress['updates'],
            'transitions': progress['transitions'], 'completed_unix': metadata['saved_unix']})
        atomic_json(output / 'report.json', progress)
        atomic_json(output / 'control.json', {'run_id': manifest['run_id'],
            'session_id': progress['session_id'], 'active': False, 'state': 'complete',
            'stop': True, 'last_save': progress['last_save']})
    return {'output': str(output.resolve()), **progress}


def warm_start(parent: Path, checkpoint_id: str, output: Path, home: Path,
               settings: dict[str, Any], game_settings: dict[str, Any] | None,
               stop_after_updates: int | None, observe_hz: int = 3) -> dict[str, Any]:
    from flycade.game import GameConfig
    from flycade.training import TrainingConfig, train
    state, lineage = selected_state(parent, checkpoint_id, home)
    lineage.update(kind='warm_start', loaded='all model state_dict keys; strict shapes; no partial load',
                   excluded=['optimizer', 'RNG', 'counters', 'save/evaluation schedules', 'parent training config'])
    manifest = state['manifest']
    game = GameConfig(**(manifest['game'] if game_settings is None else game_settings))
    return train(home, parent / 'graph', output, TrainingConfig(**settings), game,
                 manifest['device'], manifest['fixture'], stop_after_updates, observe_hz=observe_hz,
                 warm_state={'model': state['model'], 'lineage': lineage})
