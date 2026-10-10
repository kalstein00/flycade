"""Bounded sequential CPU evaluation; training state stays in the parent process."""
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from flycade.checkpoint import atomic_json
from flycade.control import SaveControl
from flycade.errors import PreparationError
from flycade.workers import kill_worker


def evaluate_snapshot(output: Path, home: Path, snapshot: str, protocol: Path,
                      timeout: float, report: dict[str, Any], control: SaveControl,
                      poll: Callable[[], None]) -> dict[str, Any]:
    report.update(status='evaluating_initial' if snapshot == 'initial' else 'evaluating_checkpoint',
                  training_paused=True)
    control.state.update(training_paused=True, evaluating_snapshot=snapshot)
    control.publish()
    atomic_json(output / 'report.json', report)
    worker = subprocess.Popen([sys.executable, '-m', 'flycade', 'evaluate', str(output),
        '--snapshot', snapshot, '--protocol', str(protocol), '--home', str(home),
        '--device', 'cpu', '--training-paused'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)
    report['evaluation_worker_pid'] = worker.pid
    deadline = time.monotonic() + timeout
    try:
        while True:
            poll()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PreparationError('evaluation_timeout', 'Initial evaluator exceeded bounded wait; initial policy retained' if snapshot == 'initial'
                                       else 'Periodic evaluator exceeded bounded wait; pre-evaluation checkpoint retained')
            try:
                stdout, stderr = worker.communicate(timeout=min(.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if worker.returncode != 0:
            raise PreparationError('initial_evaluation_failed' if snapshot == 'initial' else 'periodic_evaluation_failed', stdout + stderr)
        result: dict[str, Any] = json.loads(stdout)
        return result
    finally:
        if worker.poll() is None:
            kill_worker(worker)
        report.update(evaluation_worker_reaped=worker.poll() is not None,
                      training_paused=False, status='running')
        control.state.update(training_paused=False, evaluating_snapshot=None)
        control.publish()
