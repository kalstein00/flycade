"""Bounded cleanup for owned evaluator processes and their encoder descendants."""
import os
from pathlib import Path
import signal
import subprocess
from typing import Any


def descendants(pid: int) -> list[int]:
    try:
        direct = [int(value) for value in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except OSError:
        return []
    return direct + [child for parent in direct for child in descendants(parent)]


def kill_worker(process: subprocess.Popen[Any]) -> None:
    # Encoders start their own session, so killing the evaluator's group alone is insufficient.
    for pid in reversed(descendants(process.pid)):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate(timeout=5)
