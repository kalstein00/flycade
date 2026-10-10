#!/usr/bin/env python3
"""Verify a real WSL boot boundary before resuming the pinned acceptance checkpoint."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('handoff', type=Path)
    parser.add_argument('--home', type=Path, default=Path('.flycade'))
    args = parser.parse_args()
    try:
        handoff = json.loads(args.handoff.read_text())
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if boot == handoff['boot_id']:
            raise ValueError('WSL boot ID is unchanged; actual WSL shutdown or PC restart is still required')
        run = Path(handoff['run'])
        latest = json.loads((run / 'latest.json').read_text())
        if (latest['checkpoint_id'] != handoff['checkpoint']['checkpoint_id']
                or sha(run / 'final.pt') != handoff['checkpoint_sha256']
                or sha(run / 'initial.pt') != handoff['initial_sha256']):
            raise ValueError('Restart handoff checkpoint differs; preserve artifacts and inspect before resume')
        result = subprocess.run([sys.executable, '-m', 'flycade', 'resume', str(run),
                                 '--home', str(args.home), '--stop-after-updates', '3'],
                                capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise ValueError(result.stdout + result.stderr)
        report: dict[str, Any] = json.loads(result.stdout)
        config = json.loads((run / 'run.json').read_text())['training']
        if (report['resumed_from'] != latest['checkpoint_id']
                or report['session_id'] == handoff['session_id']
                or report['updates'] != handoff['updates'] + 3
                or report['optimizer_steps'] != handoff['optimizer_steps'] + 3 * config['epochs']
                or report['transitions'] != handoff['transitions'] + 3 * config['rollout_steps']
                or not report['environment_closed']
                or sha(run / 'initial.pt') != handoff['initial_sha256']):
            raise ValueError('Post-restart progress or identity verification failed; inspect the saved Run')
        evidence = {'boot_id_before': handoff['boot_id'], 'boot_id_after': boot,
                    'checkpoint_before': latest['checkpoint_id'], 'checkpoint_after': report['checkpoint_id'],
                    'report': report, 'windows_screen_acceptance': 'User confirmation required separately'}
        output = args.handoff.with_name('restart-verification.json')
        output.write_text(json.dumps(evidence, indent=2))
        print(json.dumps({'status': 'restart_and_resume_verified', 'output': str(output),
                          'updates': report['updates'], 'session_id': report['session_id']}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({'status': 'not_verified', 'error': str(exc)}, indent=2))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
