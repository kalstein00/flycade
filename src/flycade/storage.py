"""Run-local retention. Protected evidence and immutable origins are never GC roots to delete."""
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, TextIO

DEFAULT_POLICY = {'keep_evaluations': 3, 'keep_videos': 10, 'log_bytes': 8 * 1024 * 1024}


def storage_policy(run: Path) -> dict[str, int]:
    path = run / 'storage-policy.json'
    policy = {**DEFAULT_POLICY, **(json.loads(path.read_text()) if path.exists() else {})}
    return validate_policy(policy)


def validate_policy(policy: dict[str, int]) -> dict[str, int]:
    for key, value in policy.items():
        if key not in DEFAULT_POLICY or type(value) is not int or value < (1024 if key == 'log_bytes' else 1):
            raise ValueError(f'Invalid storage policy: {key}')
    return policy


def storage_status(run: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(run)
    path = run / 'storage-status.json'
    control = run / 'control.json'
    return {'policy': storage_policy(run), 'free_bytes': disk.free, 'total_bytes': disk.total,
            'low_space': disk.free < 64 * 1024 * 1024,
            'cleanup': json.loads(path.read_text()) if path.exists() else {'status': 'not_run'},
            'last_recovery': json.loads(control.read_text()).get('last_save') if control.exists() else None,
            'scope': 'This Run only; initial, protocol best/latest, pins, live evaluations, graph and origin are protected.'}


def trim_log(path: Path, limit: int) -> bool:
    if not path.exists() or path.stat().st_size <= limit:
        return False
    # Keep complete recent records using the same inode, including when the trainer has it open.
    with path.open('r+b') as handle:
        handle.seek(-limit // 2, os.SEEK_END)
        handle.readline()
        tail = handle.read()
        handle.seek(0)
        handle.write(tail)
        handle.truncate()
    return True


def append_log(log: TextIO, row: dict[str, Any], limit: int = DEFAULT_POLICY['log_bytes']) -> None:
    log.write(json.dumps(row, allow_nan=False) + '\n')
    log.flush()
    if log.tell() > limit:
        trim_log(Path(log.name), limit)
        log.seek(0, os.SEEK_END)


def _local(path: Path, run: Path) -> None:
    if path.is_symlink() or not path.resolve().is_relative_to(run.resolve()):
        raise ValueError(f'Refusing nonlocal retention path: {path}')


def cleanup_storage(run: Path, *, inactive: bool = False) -> dict[str, Any]:
    """Caller owns the trainer lock; evaluator references are serialized independently."""
    from flycade.checkpoint import atomic_json
    from flycade.evaluation_index import evaluation_catalog, refresh_evaluations
    from flycade.retention import reference_lock, references, evaluation_leases, _prune_checkpoints
    result: dict[str, Any] = {'status': 'complete', 'at_unix': time.time(), 'removed': [], 'trimmed_logs': [], 'errors': []}
    try:
        policy = storage_policy(run)
        for folder in ('evaluations', 'snapshots', 'checkpoints', 'videos', 'sessions', 'save-results'):
            _local(run / folder, run)
        # Recover directories left behind when index publication succeeded but removal failed.
        # Ranking keeps its previous incumbent on ties; protected results are never removed.
        if any((run / 'evaluations').glob('*/report.json')):
            refresh_evaluations(run)
        with reference_lock(run):
            refs = references(run)
            leased = evaluation_leases(run)
            protected = set(refs['pins'] + refs['best'] + refs.get('latest', []) + leased) | {'initial'}
            catalog = evaluation_catalog(run)
            removed_evaluations: list[str] = []
            retained_snapshots = set(protected)
            active = set()
            for path in (run / 'evaluation-leases').glob('*.json'):
                lease = json.loads(path.read_text())
                try:
                    os.kill(lease['pid'], 0)
                except ProcessLookupError:
                    continue
                active.add(path.stem)
            for group in catalog['protocols']:
                rows = group['results']
                keep = {row['evaluation_id'] for row in rows[-policy['keep_evaluations']:]}
                keep.update([group['initial'], group['best'], group['latest']])
                keep.update(row['evaluation_id'] for row in rows if row['snapshot_id'] in refs['pins'] + leased)
                keep.update(active)
                group['results'] = [row for row in rows if row['evaluation_id'] in keep]
                retained_snapshots.update(row['snapshot_id'] for row in group['results'])
                removed_evaluations.extend(row['evaluation_id'] for row in rows if row['evaluation_id'] not in keep)
            failed: dict[str, list[dict[str, Any]]] = {}
            for path in (run / 'evaluations').glob('*/report.json'):
                row = json.loads(path.read_text())
                if row['status'] != 'completed' and row['evaluation_id'] not in active and row['snapshot_id'] not in refs['pins']:
                    failed.setdefault(row['protocol_id'], []).append(row)
            for rows in failed.values():
                rows.sort(key=lambda row: (row['created_unix'], row['evaluation_id']), reverse=True)
                removed_evaluations.extend(row['evaluation_id'] for row in rows[policy['keep_evaluations']:])
            # Publish a consistent catalog before moving whole video+observation bundles.
            # Any crash leaves an unreferenced bundle, never a half-published protected result.
            atomic_json(run / 'evaluation-index.json', catalog)
            trash = run / '.storage-trash'
            _local(trash, run)
            trash.mkdir(exist_ok=True)
            for identifier in removed_evaluations:
                identifier = str(uuid.UUID(identifier))
                path = run / 'evaluations' / identifier
                _local(path, run)
                if path.exists():
                    path.rename(trash / identifier)
                    result['removed'].append(str(path.relative_to(run)))
            result['pruned_checkpoints'] = _prune_checkpoints(run, json.loads((run / 'run.json').read_text())['training']['keep_checkpoints'])
            retained_snapshots.update(path.stem for path in (run / 'checkpoints').glob('*.pt'))
            latest = run / 'snapshots/latest.json'
            if latest.exists():
                retained_snapshots.add(json.loads(latest.read_text())['snapshot_id'])
            # Include every remaining report (including failed or in-flight evaluations).
            for path in (run / 'evaluations').glob('*/report.json'):
                retained_snapshots.add(json.loads(path.read_text())['snapshot_id'])
            for path in (run / 'snapshots').glob('*.pt'):
                _local(path, run)
                if path.stem not in retained_snapshots:
                    path.unlink()
                    path.with_suffix('.json').unlink(missing_ok=True)
                    result['removed'].append(str(path.relative_to(run)))
            videos = sorted((run / 'videos').glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            for path in videos[policy['keep_videos']:]:
                _local(path, run)
                video = path.with_suffix('.webm')
                _local(video, run)
                # Removing the catalog record first makes an interrupted delete unavailable as a unit.
                path.unlink()
                video.unlink(missing_ok=True)
                result['removed'].append(str(video.relative_to(run)))
            # A failed unlink after catalog removal can leave an uncatalogued video.
            for video in (run / 'videos').glob('*.webm'):
                _local(video, run)
                if not video.with_suffix('.json').exists():
                    video.unlink()
                    result['removed'].append(str(video.relative_to(run)))
            if inactive:
                for folder in ('snapshots', 'checkpoints', 'videos'):
                    for path in (run / folder).glob('*.partial'):
                        _local(path, run)
                        path.unlink()
                        result['removed'].append(str(path.relative_to(run)))
            for path in trash.iterdir():
                _local(path, run)
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        for name in ('transitions.jsonl', 'updates.jsonl', 'save-events.jsonl'):
            path = run / name
            _local(path, run)
            if trim_log(path, policy['log_bytes']):
                result['trimmed_logs'].append(name)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['status'] = 'failed'
        result['errors'].append(str(exc))
    try:
        atomic_json(run / 'storage-status.json', result)
    except OSError as exc:
        result['status'] = 'failed'
        result['errors'].append(f'Cannot publish storage status: {exc}')
    return result


def manage_storage(run: Path, apply: bool = False, **changes: int | None) -> dict[str, Any]:
    from flycade.checkpoint import atomic_json, run_lock
    if apply or any(value is not None for value in changes.values()):
        with run_lock(run):
            policy = storage_policy(run)
            for key, value in changes.items():
                if value is not None:
                    policy[key] = value
            atomic_json(run / 'storage-policy.json', validate_policy(policy))
            if apply:
                cleanup = cleanup_storage(run, inactive=True)
                return {**storage_status(run), 'cleanup': cleanup}
    return storage_status(run)
