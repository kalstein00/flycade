"""Protocol-separated evaluation results and durable protection references."""
import hashlib
import json
from pathlib import Path
from typing import Any


def protocol_identity(protocol: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def evaluation_catalog(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / 'run.json').read_text())
    from flycade.budget import budget_info
    path = run / 'evaluation-index.json'
    result: dict[str, Any] = (json.loads(path.read_text()) if path.exists() else
        {'run_id': manifest['run_id'], 'protocols': [], 'note': 'Small fixed-seed samples; no general learning-success claim.'})
    if result['run_id'] != manifest['run_id']:
        raise ValueError('Evaluation index Run identity differs')
    refs = run / 'checkpoint-references.json'
    result['references'] = json.loads(refs.read_text()) if refs.exists() else {'pins': [], 'best': [], 'latest': []}
    result.update(model_kind=manifest['model'].get('kind', 'connectome'),
                  parameter_count=manifest['model'].get('parameter_count'),
                  training_budget=budget_info(run, manifest), training_config=manifest['training'])
    return result


def refresh_evaluations(run: Path) -> dict[str, Any]:
    from flycade.checkpoint import atomic_json
    from flycade.retention import reference_lock, references
    with reference_lock(run):
        previous = evaluation_catalog(run)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for path in sorted((run / 'evaluations').glob('*/report.json')):
            row = json.loads(path.read_text())
            if row['status'] != 'completed' or row['run_id'] != previous['run_id']:
                continue
            protocol = json.loads((path.parent / 'protocol.json').read_text())
            if protocol_identity(protocol) != row['protocol_id']:
                raise ValueError('Evaluation protocol checksum differs')
            grouped.setdefault(row['protocol_id'], []).append(row)
        old_best = {group['protocol_id']: group['best'] for group in previous['protocols']}
        groups = []
        for identifier, rows in sorted(grouped.items()):
            rows.sort(key=lambda row: (row['created_unix'], row['evaluation_id']))
            best = next((row for row in rows if row['evaluation_id'] == old_best.get(identifier)), rows[0])
            for row in rows:
                if (row['completion_rate'], row['mean_distance']) > (best['completion_rate'], best['mean_distance']):
                    best = row
            initial = next((row['evaluation_id'] for row in rows if row['snapshot_id'] == 'initial'), None)
            groups.append({'protocol_id': identifier, 'initial': initial,
                'latest': rows[-1]['evaluation_id'], 'best': best['evaluation_id'],
                'latest_snapshot': rows[-1]['snapshot_id'], 'best_snapshot': best['snapshot_id'],
                'results': rows, 'ranking_rule': 'completion rate, then mean distance; keep incumbent on ties'})
        result = {**previous, 'protocols': groups}
        refs = references(run)
        best_ids = sorted({group['best_snapshot'] for group in groups} - {'initial'})
        latest_ids = sorted({group['latest_snapshot'] for group in groups} - {'initial'})
        # Protect both old and new identities until the new index has committed.
        atomic_json(run / 'checkpoint-references.json', {**refs,
            'best': sorted(set(refs['best'] + best_ids)),
            'latest': sorted(set(refs.get('latest', []) + latest_ids))})
        atomic_json(run / 'evaluation-index.json', result)
        atomic_json(run / 'checkpoint-references.json', {**refs, 'best': best_ids, 'latest': latest_ids})
        return result
