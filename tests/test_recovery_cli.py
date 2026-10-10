"""Recovery and retention contracts through CLI processes and durable Run artifacts."""
import json
from pathlib import Path

from test_graph_cli import cli
from test_training_cli import prepared_graph


def saved_run(tmp_path, updates=3):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    first = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                '--updates', 20, '--rollout-steps', 4, '--stop-after-updates', 1)
    assert first.returncode == 0, first.stdout + first.stderr
    saves = [json.loads((run / 'latest.json').read_text())]
    for _ in range(updates - 1):
        result = cli('resume', run, '--stop-after-updates', 1)
        assert result.returncode == 0, result.stdout + result.stderr
        saves.append(json.loads((run / 'latest.json').read_text()))
    return run, saves


def test_corrupt_latest_rolls_back_with_explicit_event_and_new_session(tmp_path):
    run, saves = saved_run(tmp_path)
    before = json.loads((run / 'report.json').read_text())
    history = (run / 'updates.jsonl').read_bytes()
    (run / saves[-1]['file']).write_bytes(b'corrupt checkpoint')
    result = cli('resume', run, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['run_id'] == before['run_id'] and report['session_id'] != before['session_id']
    assert report['resumed_from'] == saves[-2]['checkpoint_id']
    assert report['updates'] == 3 and report['transitions'] == 12
    recovery = report['recovery']
    assert recovery['lost_updates'] == 1 and recovery['lost_transitions'] == 4
    assert recovery['from_session_id'] == before['session_id']
    assert recovery['to_session_id'] == report['session_id']
    assert recovery['rejected_candidates'][0]['checkpoint_id'] == saves[-1]['checkpoint_id']
    assert 'checksum' in recovery['rejected_candidates'][0]['error']
    assert (run / 'updates.jsonl').read_bytes().startswith(history)
    assert len(list((run / 'recoveries').glob('*.json'))) == 1
    assert report['episodes'] == 0


def test_recent_three_and_user_reference_survive_pruning(tmp_path):
    run, saves = saved_run(tmp_path, 1)
    initial = (run / 'initial.pt').read_bytes()
    pinned = saves[0]['checkpoint_id']
    result = cli('pin-checkpoint', run, pinned)
    assert result.returncode == 0, result.stdout + result.stderr
    for _ in range(4):
        result = cli('resume', run, '--stop-after-updates', 1)
        assert result.returncode == 0, result.stdout + result.stderr
    result = cli('checkpoints', run)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(result.stdout)['checkpoints']
    assert sorted(row['updates'] for row in rows if row['integrity_valid']) == [1, 3, 4, 5]
    assert next(row for row in rows if row['checkpoint_id'] == pinned)['protected']
    assert (run / 'initial.pt').read_bytes() == initial
    result = cli('pin-checkpoint', run, pinned, '--remove')
    assert result.returncode == 0, result.stdout + result.stderr
    result = cli('resume', run, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((run / 'checkpoints').glob('*.pt'))) == 3


def test_unpublished_complete_files_are_not_recovery_candidates(tmp_path):
    run, saves = saved_run(tmp_path)
    unpublished = saves[-1]
    (run / 'latest.json').write_text(json.dumps(saves[-2]))
    (run / 'checkpoints' / f"{unpublished['checkpoint_id']}.committed.json").unlink()
    result = cli('resume', run, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['resumed_from'] == saves[-2]['checkpoint_id']
    assert report['updates'] == 3


def test_broken_pointer_uses_verified_history_and_reports_no_valid_backup(tmp_path):
    run, saves = saved_run(tmp_path, 2)
    (run / 'latest.json').write_text('{broken')
    result = cli('resume', run, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['resumed_from'] == saves[-1]['checkpoint_id']
    for path in (run / 'checkpoints').glob('*.pt'):
        path.write_bytes(b'broken')
    before = (run / 'updates.jsonl').read_bytes()
    result = cli('resume', run)
    assert result.returncode == 2 and 'No valid checkpoint' in result.stdout
    assert 'local backup' in result.stdout
    assert (run / 'updates.jsonl').read_bytes() == before


import pytest


@pytest.mark.parametrize('fault', ['enospc', 'fsync', 'publish'])
def test_write_sync_or_publication_failure_preserves_previous_commit(tmp_path, fault):
    import os
    import subprocess
    import sys
    run, saves = saved_run(tmp_path, 1)
    library = tmp_path / 'fault.so'
    subprocess.run(['cc', '-shared', '-fPIC', '-o', str(library), 'tests/faults/checkpoint_io.c', '-ldl'], check=True)
    before = (run / 'latest.json').read_bytes()
    result = subprocess.run([sys.executable, '-m', 'flycade', 'resume', str(run), '--stop-after-updates', '1'],
        env={**os.environ, 'LD_PRELOAD': str(library), 'FLYCADE_FAULT_RUN': str(run), 'FLYCADE_FAULT_MODE': fault},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    assert (run / 'latest.json').read_bytes() == before
    assert not list((run / 'checkpoints').glob('*.partial'))
    status = json.loads(cli('status', run).stdout)
    assert status['state'] == 'failed'
    assert status['last_save']['checkpoint_id'] == saves[-1]['checkpoint_id']
    result = cli('resume', run, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['updates'] == 2


@pytest.mark.parametrize('boundary', ['temporary_created', 'data_renamed', 'pointer_published'])
def test_sigkill_at_filesystem_boundaries_can_resume(tmp_path, boundary):
    import ctypes
    import os
    import select
    import signal
    import struct
    import subprocess
    import sys
    import time
    run, saves = saved_run(tmp_path, 1)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    assert fd >= 0
    directory = run if boundary == 'pointer_published' else run / 'checkpoints'
    assert libc.inotify_add_watch(fd, os.fsencode(directory), 0x100 | 0x80) >= 0  # CREATE, MOVED_TO
    process = subprocess.Popen([sys.executable, '-m', 'flycade', 'resume', str(run), '--stop-after-updates', '1'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    children = set()
    hit = False
    try:
        deadline = time.monotonic() + 20
        while not hit and time.monotonic() < deadline:
            child_file = Path(f'/proc/{process.pid}/task/{process.pid}/children')
            if child_file.exists():
                children.update(child_file.read_text().split())
            if not select.select([fd], [], [], .02)[0]:
                assert process.poll() is None, process.communicate()
                continue
            data = os.read(fd, 65536)
            offset = 0
            while offset < len(data):
                _, mask, _, length = struct.unpack_from('iIII', data, offset)
                name = data[offset+16:offset+16+length].split(b'\0', 1)[0].decode()
                offset += 16 + length
                hit = ((boundary == 'temporary_created' and mask & 0x100 and name.endswith('.partial'))
                       or (boundary == 'data_renamed' and mask & 0x80 and name.endswith('.pt'))
                       or (boundary == 'pointer_published' and mask & 0x80 and name == 'latest.json'))
                if hit:
                    process.send_signal(signal.SIGKILL)
                    break
        assert hit, 'Requested publication boundary was not observed'
        process.communicate(timeout=10)
        assert process.returncode == -signal.SIGKILL
        # A notification is asynchronous: either old or newly committed state may win the race.
        pointer = json.loads((run / 'latest.json').read_text())
        assert pointer['updates'] in (1, 2)
        result = cli('resume', run, '--stop-after-updates', 1)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout)
        assert report['updates'] == pointer['updates'] + 1
        assert report['transitions'] == report['updates'] * 4 and report['episodes'] == 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            alive = [pid for pid in children if Path(f'/proc/{pid}/stat').exists()
                     and Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z']
            if not alive:
                break
            time.sleep(.05)
        assert not alive, 'Encoder must exit when killed trainer closes its pipe'
    finally:
        os.close(fd)
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


@pytest.mark.parametrize('value', [None, [], {'checkpoint_id': 'invalid', 'saved_unix': 'not-a-time'}])
def test_malformed_pointer_schema_falls_back_to_committed_history(tmp_path, value):
    run, saves = saved_run(tmp_path, 1)
    (run / 'latest.json').write_text(json.dumps(value))
    result = cli('resume', run, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['resumed_from'] == saves[0]['checkpoint_id']


def test_best_reference_uses_same_protection_contract(tmp_path):
    run, saves = saved_run(tmp_path, 1)
    best = saves[0]['checkpoint_id']
    # Public reference artifact consumed by the later periodic evaluator.
    (run / 'checkpoint-references.json').write_text(json.dumps({'pins': [], 'best': [best]}))
    for _ in range(3):
        result = cli('resume', run, '--stop-after-updates', 1)
        assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(cli('checkpoints', run).stdout)['checkpoints']
    assert next(row for row in rows if row['checkpoint_id'] == best)['protected']
    assert len([row for row in rows if row['integrity_valid']]) == 4
