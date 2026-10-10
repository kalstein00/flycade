"""Read-only bounded replay loading, without importing the training stack."""
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid


def evaluation_streams(run: Path) -> list[dict[str, Any]]:
    run_id = json.loads((run / 'run.json').read_text())['run_id']
    rows = []
    for path in (run / 'evaluations').glob('*/report.json'):
        row = json.loads(path.read_text())
        if row['run_id'] == run_id:
            rows.append({key: row[key] for key in ('evaluation_id', 'snapshot_id', 'snapshot_updates', 'created_unix', 'status')})
    return sorted(rows, key=lambda row: row['created_unix'])


def read_replay(run: Path, identifier: str, live: bool = False) -> dict[str, Any]:
    identifier = str(uuid.UUID(identifier))
    run_id = json.loads((run / 'run.json').read_text())['run_id']
    directory = run / 'evaluations' / identifier
    report = json.loads((directory / 'report.json').read_text())
    if report['run_id'] != run_id or report['evaluation_id'] != identifier:
        raise ValueError('Replay Run/evaluation identity differs')
    if live:
        path = directory / 'live.json'
        document = json.loads(path.read_text()) if path.exists() else None
        if document is not None and document['status'] == 'running':
            try:
                os.kill(document['evaluator_pid'], 0)
            except ProcessLookupError:
                document['status'] = 'evaluator_stopped'
    else:
        metadata = report.get('replay')
        document = None
        if metadata and metadata['status'] == 'complete':
            if metadata['file'] != 'observations.json.gz':
                raise ValueError('Unexpected replay artifact path')
            path = directory / metadata['file']
            if path.stat().st_size > 16 * 2**20:
                raise ValueError('Replay file exceeds bounded size')
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
                raise ValueError('Replay checksum differs')
            with gzip.open(path, 'rb') as handle:
                content = handle.read(20 * 2**20 + 1)
            if len(content) > 20 * 2**20:
                raise ValueError('Expanded replay exceeds bounded size')
            document = json.loads(content)
    if document is not None:
        if (document['format_version'] != 1 or document['run_id'] != run_id
                or document['evaluation_id'] != identifier or document['snapshot_id'] != report['snapshot_id']
                or document['protocol_id'] != report['protocol_id']):
            raise ValueError('Replay identity differs')
        samples = [document['sample']] if live and document.get('sample') else document.get('samples', [])
        if len(samples) > 180:
            raise ValueError('Replay sample limit exceeded')
        for sample in samples:
            if (sample['run_id'] != run_id or sample['session_id'] != identifier
                    or sample['snapshot_id'] != report['snapshot_id'] or sample['stream'] != 'evaluation'):
                raise ValueError('Replay sample identity differs')
    return {'report': report, 'document': document}
