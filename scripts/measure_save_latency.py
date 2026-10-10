#!/usr/bin/env python3
"""Observe CLI stop-request log, atomic checkpoint publication and process exit."""
import argparse
import json
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    rows: dict[str, dict[str, float | str]] = {}
    if any('Save-and-stop requested' in p.read_text() for p in args.directory.glob('phase-*.stderr.log')):
        raise SystemExit('Start the watcher before the first save request; late baselines are invalid')
    last_checkpoint = ''
    try:
        last_checkpoint = json.loads((args.directory / 'run/latest.json').read_text())['checkpoint_id']
    except FileNotFoundError:
        pass
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            latest = json.loads((args.directory / 'run/latest.json').read_text())
        except (OSError, ValueError):
            latest = {'checkpoint_id': ''}
        for index in range(4):
            label = str(index)
            error = args.directory / f'phase-{index}.stderr.log'
            if not error.exists() or 'Save-and-stop requested' not in error.read_text():
                continue
            if label not in rows:
                rows[label] = {'request_observed_unix': time.time()}
                rows[label]['previous_checkpoint'] = last_checkpoint
            row = rows[label]
            now = time.time()
            if latest['checkpoint_id'] and latest['checkpoint_id'] != row['previous_checkpoint'] and 'publication_observed_unix' not in row:
                row['publication_observed_unix'] = now
                row['request_to_publication_seconds'] = now - float(row['request_observed_unix'])
            result = args.directory / f'phase-{index}.stdout.json'
            if result.exists() and result.stat().st_size and 'completion_observed_unix' not in row:
                row['completion_observed_unix'] = now
                row['request_to_completion_seconds'] = now - float(row['request_observed_unix'])
        if latest['checkpoint_id']:
            last_checkpoint = latest['checkpoint_id']
        (args.directory / 'save-latency.json').write_text(json.dumps({'poll_resolution_seconds': .05, 'phases': rows}, indent=2))
        if len(rows) == 4 and all('completion_observed_unix' in row and 'publication_observed_unix' in row for row in rows.values()):
            return
        time.sleep(.05)
    raise SystemExit('Latency observation timed out; partial evidence retained')


if __name__ == '__main__':
    main()
