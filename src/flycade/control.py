"""File-based save requests: CLI requests, the sole trainer commits at safe boundaries."""
import fcntl
import json
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from flycade.errors import PreparationError


def request_save(run: Path, stop: bool = False, wait_seconds: float = 30) -> dict[str, Any]:
    from flycade.checkpoint import atomic_json
    if not math.isfinite(wait_seconds) or wait_seconds < 0:
        raise ValueError('wait-seconds must be finite and nonnegative')
    with (run / '.request.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (run / '.writer.lock').open('a') as writer:
            try:
                fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise PreparationError('run_inactive', 'No active trainer; resume the Run first')
        state = json.loads((run / 'control.json').read_text())
        if not state['active'] or state.get('stop'):
            raise PreparationError('run_inactive', 'Trainer is closing; wait for exit before resume')
        request_path = run / 'save-request.json'
        if request_path.exists():
            raise PreparationError('save_pending', 'A save request is already pending; inspect status')
        request = {'request_id': str(uuid.uuid4()), 'session_id': state['session_id'],
                   'requested_unix': time.time(), 'stop': stop, 'reason': 'manual'}
        atomic_json(request_path, request)
    print('Save request accepted; waiting for rollout/optimizer boundary.', file=sys.stderr)
    if wait_seconds == 0:
        return {**request, 'state': 'requested'}
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        receipt = run / 'save-results' / f"{request['request_id']}.json"
        current: dict[str, Any] = json.loads((receipt if receipt.exists() else run / 'control.json').read_text())
        if current.get('request_id') == request['request_id'] and current['state'] in ('complete', 'failed'):
            if current['state'] == 'failed':
                raise PreparationError('save_failed', current.get('error', 'Save failed; previous checkpoint retained'))
            if not stop or not current['active']:
                return current
        time.sleep(.05)
    raise PreparationError('save_wait_timeout', 'Request retained; inspect status before ending the process')


def run_status(run: Path) -> dict[str, Any]:
    state: dict[str, Any] = json.loads((run / 'control.json').read_text())
    with (run / '.writer.lock').open('a') as writer:
        try:
            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            if state['active']:
                state.update(active=False, state='trainer_stopped')
    request = run / 'save-request.json'
    if request.exists():
        pending = json.loads(request.read_text())
        if pending['session_id'] == state['session_id'] and pending['request_id'] != state.get('request_id'):
            state['pending_request'] = pending
    recovery = run / 'recovery-status.json'
    if recovery.exists():
        state['recovery'] = json.loads(recovery.read_text())
    state['delay_seconds'] = max(0., time.time() - state['requested_unix']) if state['state'] in ('waiting_boundary', 'saving') or (state.get('stop') and state['active']) else 0
    return state


class SaveControl:
    def __init__(self, run: Path, session: dict[str, Any], report: dict[str, Any]):
        self.run = run
        self.pending: dict[str, Any] | None = None
        self.state: dict[str, Any] = {'run_id': session['run_id'], 'session_id': session['session_id'],
            'active': True, 'state': 'idle', 'last_save': report.get('last_save'),
            'reset': session['reset']}
        (run / 'save-request.json').unlink(missing_ok=True)
        self.publish()

    def publish(self) -> None:
        from flycade.checkpoint import atomic_json
        self.state['updated_unix'] = time.time()
        atomic_json(self.run / 'control.json', self.state)

    def event(self, state: str) -> None:
        self.state['state'] = state
        self.publish()
        with (self.run / 'save-events.jsonl').open('a') as log:
            from flycade.storage import append_log, storage_policy
            append_log(log, self.state, storage_policy(self.run)['log_bytes'])
        print(f'Save {state}: {self.state.get("reason", "")} (last recovery: {self.state.get("last_save")})', file=sys.stderr)

    def poll(self, stop: bool = False, automatic_due: bool = False) -> None:
        path = self.run / 'save-request.json'
        if self.pending is None and path.exists():
            request = json.loads(path.read_text())
            if request['session_id'] == self.state['session_id']:
                self.pending = request
        if (stop or automatic_due) and self.pending is None:
            self.pending = {'request_id': str(uuid.uuid4()), 'requested_unix': time.time(),
                            'stop': stop, 'reason': 'stop' if stop else 'automatic'}
        if self.pending is not None:
            self.pending['stop'] |= stop
            self.state['stop'] = self.pending['stop']
            if self.state.get('request_id') != self.pending['request_id']:
                self.state.update(self.pending)
                self.event('waiting_boundary')

    def saving(self) -> None:
        self.event('saving')

    def complete(self, metadata: dict[str, Any], report: dict[str, Any]) -> None:
        now = time.time()
        report['last_save'] = {'checkpoint_id': metadata['checkpoint_id'], 'updates': metadata['updates'],
                              'transitions': report['transitions'], 'completed_unix': now}
        self.state.update(last_save=report['last_save'], completed_unix=now)
        self.event('complete')
        path = self.run / 'save-request.json'
        if path.exists() and json.loads(path.read_text())['request_id'] == self.state['request_id']:
            path.unlink()
        self.receipt()
        self.pending = None

    def receipt(self) -> None:
        if self.state.get('reason') == 'manual':
            from flycade.checkpoint import atomic_json
            directory = self.run / 'save-results'
            directory.mkdir(exist_ok=True)
            atomic_json(directory / f"{self.state['request_id']}.json", self.state)

    def close(self, error: str | None = None) -> None:
        self.state['active'] = False
        if error:
            self.state['error'] = error
            self.event('failed')
        else:
            self.publish()
        self.receipt()
