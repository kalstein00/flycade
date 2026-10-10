"""Checkpoint protection references, shared by recovery, user pins and future best selection."""
import json
import uuid
from pathlib import Path
from typing import Any

from flycade.checkpoint import atomic_json, read_metadata, run_lock, sync_directory
from flycade.errors import PreparationError
from flycade.graph import digest


def references(run: Path) -> dict[str, Any]:
    path = run / 'checkpoint-references.json'
    if not path.exists():
        return {'pins': [], 'best': []}
    value: dict[str, Any] = json.loads(path.read_text())
    for key in ('pins', 'best'):
        if not isinstance(value.get(key), list):
            raise ValueError(f'Invalid checkpoint references: {key}')
        for identifier in value[key]:
            uuid.UUID(identifier)
    return value


def checkpoint_history(run: Path) -> dict[str, Any]:
    refs = references(run)
    protected = set(refs['pins'] + refs['best'])
    rows = []
    manifest_sha = digest(run / 'run.json')
    for path in (run / 'checkpoints').glob('*.committed.json'):
        try:
            row = read_metadata(path)
            identifier = str(uuid.UUID(row['checkpoint_id']))
            valid = (row['file'] == f'checkpoints/{identifier}.pt'
                     and row['manifest_sha256'] == manifest_sha
                     and digest(run / row['file']) == row['sha256'])
            rows.append({**row, 'integrity_valid': valid, 'protected': identifier in protected})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rows.append({'checkpoint_id': path.name.removesuffix('.committed.json'),
                         'integrity_valid': False, 'protected': True, 'error': str(exc)})
    rows.sort(key=lambda row: row.get('saved_unix', 0), reverse=True)
    return {'checkpoints': rows, 'references': refs, 'initial_protected': True,
            'note': 'Integrity check only; resume also validates source, environment and complete learning state.'}


def pin_checkpoint(run: Path, identifier: str, remove: bool = False) -> dict[str, Any]:
    identifier = str(uuid.UUID(identifier))
    with run_lock(run):
        history = checkpoint_history(run)
        if not any(row['checkpoint_id'] == identifier and row['integrity_valid'] for row in history['checkpoints']):
            raise PreparationError('checkpoint_unavailable', 'Only an intact committed checkpoint can be pinned')
        refs = history['references']
        pins = set(refs['pins'])
        pins.discard(identifier) if remove else pins.add(identifier)
        refs['pins'] = sorted(pins)
        atomic_json(run / 'checkpoint-references.json', refs)
        return {'checkpoint_id': identifier, 'protected': identifier in set(refs['pins'] + refs['best']), **refs}


def prune_checkpoints(run: Path, keep: int) -> list[str]:
    history = checkpoint_history(run)
    valid = [row for row in history['checkpoints'] if row['integrity_valid']]
    retained = {row['checkpoint_id'] for row in valid[:keep]}
    removed = []
    for row in valid:
        identifier = row['checkpoint_id']
        if identifier in retained or row['protected']:
            continue
        # Remove the recovery eligibility marker first. A crash can leave only an unused artifact.
        (run / 'checkpoints' / f'{identifier}.committed.json').unlink()
        sync_directory(run / 'checkpoints')
        for suffix in ('.json', '.pt'):
            (run / 'checkpoints' / f'{identifier}{suffix}').unlink(missing_ok=True)
        removed.append(identifier)
    if removed:
        sync_directory(run / 'checkpoints')
    return removed
