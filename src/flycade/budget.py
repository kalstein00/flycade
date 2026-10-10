"""Explicit append-only budget change ledger; constant-LR schedules never restart."""
import json
import time
import uuid
from pathlib import Path
from typing import Any

SCHEDULE_RULE = 'constant learning rate; cumulative counters and save/evaluation schedules preserved'


def budget_info(run: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = manifest or json.loads((run / 'run.json').read_text())
    total = manifest['training']['updates']
    path = run / 'budget.json'
    events = json.loads(path.read_text())['events'] if path.exists() else []
    if not isinstance(events, list):
        raise ValueError('Invalid budget ledger')
    identifiers: set[str] = set()
    for event in events:
        identifier = str(uuid.UUID(event['event_id']))
        if (identifier in identifiers or event['run_id'] != manifest['run_id']
                or event['previous_updates'] != total or type(event['total_updates']) is not int
                or event['total_updates'] <= total or event['schedule_rule'] != SCHEDULE_RULE):
            raise ValueError('Invalid budget event chain; restore the local budget ledger')
        total = event['total_updates']
        identifiers.add(identifier)
    return {'total_updates': total, 'original_updates': manifest['training']['updates'],
            'events': events, 'schedule_rule': SCHEDULE_RULE}


def extend_budget(run: Path, total_updates: int, home: Path) -> dict[str, Any]:
    from flycade.checkpoint import atomic_json, load_checkpoint, run_lock
    with run_lock(run):
        state = load_checkpoint(run, home)
        info = budget_info(run, state['manifest'])
        if total_updates <= info['total_updates']:
            raise ValueError('Budget extension must increase the current total updates')
        if state['progress'].get('budget', {}).get('total_updates', 0) > info['total_updates']:
            raise ValueError('Budget ledger is older than the checkpoint; restore the ledger')
        event = {'event_id': str(uuid.uuid4()), 'run_id': state['manifest']['run_id'],
                 'created_unix': time.time(), 'previous_updates': info['total_updates'],
                 'total_updates': total_updates, 'at_updates': state['progress']['updates'],
                 'checkpoint_id': state['checkpoint_id'], 'schedule_rule': SCHEDULE_RULE}
        atomic_json(run / 'budget.json', {'events': [*info['events'], event]})
        return event
