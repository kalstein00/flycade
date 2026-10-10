"""Daily operations through separate CLI processes and persistent public reports."""
import json
import subprocess
import sys
import time

from test_graph_cli import cli
from test_training_cli import prepared_graph


def wait_report(run, predicate, process, timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert process.poll() is None, process.communicate()
        try:
            row = json.loads((run / 'report.json').read_text())
            if predicate(row):
                return row
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(.03)
    raise AssertionError('Run did not reach expected progress')


def test_manual_save_continues_and_separate_stop_resumes_same_run(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'daily'
    process = subprocess.Popen([sys.executable, '-m', 'flycade', 'train', '--fixture', '--device', 'cpu',
        '--graph', str(graph), '--output', str(run), '--updates', '10000', '--rollout-steps', '16'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        first = wait_report(run, lambda r: r['updates'] > 0, process)
        saved = cli('save', run)
        assert saved.returncode == 0, saved.stdout + saved.stderr
        state = json.loads(saved.stdout)
        assert state['state'] == 'complete' and state['active']
        assert state['completed_unix'] >= state['requested_unix']
        assert state['last_save']['updates'] >= first['updates']
        wait_report(run, lambda r: r['updates'] > state['last_save']['updates'], process)
        stopped = cli('save', run, '--stop')
        assert stopped.returncode == 0, stopped.stdout + stopped.stderr
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        final = json.loads(stdout)
        assert final['status'] == 'saved' and final['transitions'] == final['updates'] * 16
        result = cli('resume', run, '--stop-after-updates', 1)
        assert result.returncode == 0, result.stdout + result.stderr
        resumed = json.loads(result.stdout)
        assert resumed['run_id'] == first['run_id'] and resumed['session_id'] != first['session_id']
        assert resumed['updates'] == final['updates'] + 1
        assert 'new episode' in result.stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_autosave_schedule_survives_resume_and_releases_writer(tmp_path):
    import torch
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'automatic'
    settings = tmp_path / 'training.json'
    settings.write_text(json.dumps({'updates': 6, 'rollout_steps': 16, 'autosave_seconds': .01}))
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', run, '--training-config', settings, '--stop-after-updates', 2)
    assert result.returncode == 0, result.stdout + result.stderr
    saved = torch.load(run / 'final.pt', weights_only=False)
    due = saved['progress']['next_autosave_seconds']
    assert due > saved['progress']['training_seconds']
    events = [json.loads(x) for x in (run / 'save-events.jsonl').read_text().splitlines()]
    assert any(e.get('reason') == 'automatic' and e['state'] == 'complete' for e in events)
    assert all(e['last_save']['transitions'] % 16 == 0 for e in events if e['state'] == 'complete')
    result = cli('resume', run, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    resumed = torch.load(run / 'final.pt', weights_only=False)
    assert resumed['progress']['updates'] == 3
    assert resumed['progress']['next_autosave_seconds'] > due
    assert json.loads(cli('status', run).stdout)['active'] is False
    assert cli('save', run).returncode == 2


def test_busy_writer_is_rejected_and_dead_owner_does_not_block_resume(tmp_path):
    import signal
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'ownership'
    first = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                '--output', run, '--updates', 10000, '--rollout-steps', 16, '--stop-after-updates', 1)
    assert first.returncode == 0, first.stdout + first.stderr
    process = subprocess.Popen([sys.executable, '-m', 'flycade', 'resume', str(run)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_report(run, lambda r: r['session_id'] != json.loads(first.stdout)['session_id'] and r['updates'] > 1, process)
        duplicate = cli('resume', run, '--stop-after-updates', 1)
        assert duplicate.returncode == 2 and 'run_busy' in duplicate.stdout
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        # The persistent lock file remains; an ended owner's advisory lock must not.
        assert (run / '.writer.lock').exists()
        result = cli('resume', run, '--stop-after-updates', 1)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)['updates'] == json.loads(stdout)['updates'] + 1
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_stalled_encoder_is_bounded_and_retains_safe_save(tmp_path):
    import os
    import shutil
    from pathlib import Path
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'stalled'
    executables = tmp_path / 'bin'
    executables.mkdir()
    (executables / 'git').symlink_to(shutil.which('git'))
    encoder = executables / 'ffmpeg'
    # External process fault; no training internals are replaced.
    encoder.write_text('#!/bin/sh\necho $$ > "$ENCODER_PID_FILE"\nexec /bin/sleep 300\n')
    encoder.chmod(0o755)
    started = time.monotonic()
    result = subprocess.run([sys.executable, '-m', 'flycade', 'train', '--fixture', '--device', 'cpu',
        '--graph', str(graph), '--output', str(run), '--updates', '1', '--rollout-steps', '4'],
        env={**os.environ, 'PATH': str(executables), 'ENCODER_PID_FILE': str(tmp_path / 'encoder.pid')}, capture_output=True, text=True, timeout=50)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert time.monotonic() - started < 45
    assert report['recording_status'] == 'failed' and 'stalled' in report['recording_error']
    assert report['environment_closed'] and (run / 'latest.json').exists()
    assert not list((run / 'videos').glob('*.partial'))
    assert not Path(f"/proc/{(tmp_path / 'encoder.pid').read_text().strip()}").exists()


def test_manual_request_is_not_lost_during_frequent_autosave(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'busy-saver'
    config = tmp_path / 'frequent.json'
    config.write_text(json.dumps({'updates': 10000, 'rollout_steps': 8, 'autosave_seconds': .001}))
    process = subprocess.Popen([sys.executable, '-m', 'flycade', 'train', '--fixture', '--device', 'cpu',
        '--graph', str(graph), '--output', str(run), '--training-config', str(config)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    try:
        wait_report(run, lambda r: r['updates'] > 2, process)
        result = cli('save', run, '--stop')
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)['active'] is False
        process.wait(timeout=30)
        assert process.returncode == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_initial_evaluation_worker_timeout_is_bounded_and_reported(tmp_path):
    from pathlib import Path
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'evaluation-timeout'
    config = tmp_path / 'timeout.json'
    config.write_text(json.dumps({'updates': 2, 'rollout_steps': 4, 'evaluation_timeout_seconds': .05}))
    evaluation = tmp_path / 'evaluation.json'
    evaluation.write_text(json.dumps({'seeds': [11], 'max_frames': 120}))
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                 '--training-config', config, '--initial-evaluation-config', evaluation)
    assert result.returncode == 2 and 'evaluation_timeout' in result.stdout
    report = json.loads((run / 'report.json').read_text())
    assert report['updates'] == 0 and report['evaluation_worker_reaped']
    assert not Path(f"/proc/{report['evaluation_worker_pid']}").exists()
    status = json.loads(cli('status', run).stdout)
    assert status['state'] == 'failed' and not status['active']


def test_first_ctrl_c_during_initial_evaluation_waits_then_saves(tmp_path):
    import signal
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'initial-stop'
    evaluation = tmp_path / 'initial.json'
    evaluation.write_text(json.dumps({'seeds': [11], 'max_frames': 120}))
    process = subprocess.Popen([sys.executable, '-m', 'flycade', 'train', '--fixture', '--device', 'cpu',
        '--graph', str(graph), '--output', str(run), '--updates', '20', '--rollout-steps', '4',
        '--initial-evaluation-config', str(evaluation)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_report(run, lambda r: r['status'] == 'evaluating_initial', process)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        report = json.loads(stdout)
        assert report['initial_evaluation_id'] and report['evaluation_worker_reaped']
        assert report['status'] == 'saved' and report['updates'] == 1
        events = [json.loads(x) for x in (run / 'save-events.jsonl').read_text().splitlines()]
        assert events[0]['state'] == 'waiting_boundary'
        assert events[0]['requested_unix'] < report['training_started_unix']
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
