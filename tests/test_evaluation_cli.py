"""Fixed-policy evaluation through CLI processes and immutable Run artifacts."""
import hashlib
import json

from test_graph_cli import cli
from test_training_cli import prepared_graph


def protected_files(run):
    return {str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in run.rglob('*') if path.is_file()
            and path.name not in ('checkpoint-references.json', 'evaluation-index.json', '.references.lock')
            and not any(part in ('evaluations', 'protocols', 'evaluation-leases') for part in path.relative_to(run).parts)}


def test_initial_and_trained_snapshots_evaluate_without_changing_training(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', run, '--updates', 2, '--rollout-steps', 4, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    training = json.loads(result.stdout)
    protocol = tmp_path / 'protocol.json'
    result = cli('evaluation-protocol', run, '--output', protocol, '--seeds', '11,22', '--max-frames', 16,
                 '--video-seconds', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    before = protected_files(run)
    results = []
    for snapshot in ('initial', 'latest'):
        result = cli('evaluate', run, '--snapshot', snapshot, '--protocol', protocol, '--device', 'cpu')
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout)
        results.append(report)
        assert report['status'] == 'completed'
        assert report['run_id'] == training['run_id']
        assert report['mode'] == 'stochastic_evaluation'
        assert report['evaluation_count'] == 2 and report['completion_count'] == 0
        assert report['termination_counts'] == {'external_limit': 2}
        assert report['distance_unit'] == 'pixels' and report['reward_unit'] == 'raw game reward'
        assert len(report['distances']) == 2
        assert report['environment_closed'] and report['workers_alive'] == 0
        assert report['wall_seconds'] > 0 and report['peak_rss_bytes'] > 0
        assert report['videos'] and report['videos'][0]['snapshot_id'] == report['snapshot_id']
        assert report['videos'][0]['protocol_id'] == report['protocol_id']
        assert protected_files(run) == before
    repeat = cli('evaluate', run, '--snapshot', 'initial', '--protocol', protocol)
    assert repeat.returncode == 0, repeat.stdout + repeat.stderr
    repeated = json.loads(repeat.stdout)
    assert (run / 'evaluations' / repeated['evaluation_id'] / 'transitions.jsonl').read_bytes() == (
        run / 'evaluations' / results[0]['evaluation_id'] / 'transitions.jsonl').read_bytes()
    assert protected_files(run) == before
    assert results[0]['snapshot_id'] == 'initial'
    assert results[1]['snapshot_id'] == training['checkpoint_id']
    result = cli('compare-evaluations', run, results[0]['evaluation_id'], results[1]['evaluation_id'])
    assert result.returncode == 0, result.stdout + result.stderr
    comparison = json.loads(result.stdout)
    assert comparison['protocol_id'] == results[0]['protocol_id'] == results[1]['protocol_id']
    assert len(comparison['results']) == 2
    result = cli('resume', run)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['updates'] == 2


def test_initial_evaluation_finishes_before_learning_and_preserves_next_update(tmp_path):
    import torch
    graph = prepared_graph(tmp_path)
    options = tmp_path / 'evaluation.json'
    options.write_text(json.dumps({'seeds': [31, 32], 'max_frames': 16, 'video_seconds': 1}))
    runs = [tmp_path / 'control', tmp_path / 'baseline']
    for index, run in enumerate(runs):
        extra = ('--initial-evaluation-config', options) if index else ()
        result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                     '--updates', 1, '--rollout-steps', 4, *extra)
        assert result.returncode == 0, result.stdout + result.stderr
    control = torch.load(runs[0] / 'final.pt', weights_only=False)
    baseline = torch.load(runs[1] / 'final.pt', weights_only=False)
    for name, value in control['model'].items():
        assert torch.equal(value.to_dense(), baseline['model'][name].to_dense())
    assert torch.equal(control['torch_rng'], baseline['torch_rng'])
    report = json.loads((runs[1] / 'report.json').read_text())
    evaluation = json.loads((runs[1] / 'evaluations' / report['initial_evaluation_id'] / 'report.json').read_text())
    assert evaluation['snapshot_id'] == 'initial' and evaluation['snapshot_updates'] == 0
    assert evaluation['training_paused'] and evaluation['status'] == 'completed'
    assert evaluation['finished_unix'] <= report['training_started_unix']
    first_update = json.loads((runs[1] / 'updates.jsonl').read_text().splitlines()[0])
    assert first_update['initial_evaluation_id'] == evaluation['evaluation_id']


def test_protocol_identity_and_deterministic_spectating_are_not_mixed(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', run, '--updates', 1, '--rollout-steps', 4)
    assert result.returncode == 0, result.stdout + result.stderr
    stochastic, deterministic = tmp_path / 'sample.json', tmp_path / 'watch.json'
    for path, mode, seeds in [(stochastic, 'stochastic', '1,2'), (deterministic, 'deterministic', '1')]:
        result = cli('evaluation-protocol', run, '--output', path, '--mode', mode, '--seeds', seeds,
                     '--max-frames', 120, '--video-seconds', 1)
        assert result.returncode == 0, result.stdout + result.stderr
    ids = []
    for protocol in [stochastic, deterministic, deterministic]:
        result = cli('evaluate', run, '--protocol', protocol)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout)
        ids.append(report['evaluation_id'])
        assert sum(video['frames'] for video in report['videos']) <= 12
        trace = [json.loads(row) for row in (run / 'evaluations' / ids[-1] / 'transitions.jsonl').read_text().splitlines()]
        for episode in report['episodes']:
            steps = [row for row in trace if row['episode'] == episode['episode']]
            assert episode['reward'] == sum(row['reward'] for row in steps)
            assert episode['distance'] == max(row['max_progress'] for row in steps)
            assert episode['emulator_frames'] == sum(row['executed_frames'] for row in steps)
        if protocol == deterministic:
            assert report['mode'] == 'deterministic_spectating' and report['evaluation_count'] == 1
    assert (run / 'evaluations' / ids[1] / 'transitions.jsonl').read_bytes() == (
        run / 'evaluations' / ids[2] / 'transitions.jsonl').read_bytes()
    result = cli('compare-evaluations', run, *ids[:2])
    assert result.returncode == 2 and 'protocol_incompatible' in result.stdout
    result = cli('evaluation-protocol', run, '--output', tmp_path / 'invalid.json',
                 '--mode', 'deterministic', '--seeds', '1,2')
    assert result.returncode == 2 and 'one representative episode' in result.stdout


def test_missing_snapshot_and_changed_protocol_are_rejected_without_touching_run(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', run, '--updates', 1, '--rollout-steps', 4)
    assert result.returncode == 0, result.stdout + result.stderr
    protocol = tmp_path / 'protocol.json'
    assert cli('evaluation-protocol', run, '--output', protocol, '--max-frames', 16).returncode == 0
    original = json.loads(protocol.read_text())
    changed = {**original, 'actions': [['LEFT']]}
    protocol.write_text(json.dumps(changed))
    before = protected_files(run)
    result = cli('evaluate', run, '--protocol', protocol)
    assert result.returncode == 2 and 'protocol_incompatible' in result.stdout
    assert protected_files(run) == before and not (run / 'evaluations').exists()
    protocol.write_text(json.dumps(original))
    snapshot = json.loads((run / 'snapshots' / 'initial.json').read_text())
    (run / snapshot['weights']).chmod(0o644)
    (run / snapshot['weights']).write_bytes(b'changed')
    result = cli('evaluate', run, '--protocol', protocol)
    assert result.returncode == 2 and 'snapshot_incompatible' in result.stdout
    assert not (run / 'evaluations').exists()


def test_policy_snapshot_evaluates_without_any_resume_files(tmp_path):
    import shutil
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', run, '--updates', 1, '--rollout-steps', 4)
    assert result.returncode == 0, result.stdout + result.stderr
    snapshot_id = json.loads(result.stdout)['checkpoint_id']
    (run / 'final.pt').unlink()
    (run / 'latest.json').unlink()
    shutil.rmtree(run / 'checkpoints')
    protocol = tmp_path / 'protocol.json'
    assert cli('evaluation-protocol', run, '--output', protocol, '--max-frames', 8).returncode == 0
    result = cli('evaluate', run, '--snapshot', 'latest', '--protocol', protocol)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['snapshot_id'] == snapshot_id


def test_realtime_spectating_reaps_encoder_without_mutating_training(tmp_path):
    import subprocess
    import sys
    import time
    from pathlib import Path
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', run, '--updates', 1, '--rollout-steps', 4)
    assert result.returncode == 0, result.stdout + result.stderr
    protocol = tmp_path / 'watch.json'
    assert cli('evaluation-protocol', run, '--output', protocol, '--mode', 'deterministic',
               '--seeds', '1', '--max-frames', 120).returncode == 0
    before = protected_files(run)
    process = subprocess.Popen([sys.executable, '-m', 'flycade', 'evaluate', str(run),
        '--protocol', str(protocol), '--realtime'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    children = set()
    try:
        deadline = time.monotonic() + 20
        while process.poll() is None and time.monotonic() < deadline:
            path = Path(f'/proc/{process.pid}/task/{process.pid}/children')
            if path.exists():
                children.update(path.read_text().split())
            time.sleep(.02)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stdout + stderr
        report = json.loads(stdout)
        assert report['wall_seconds'] >= 2
        assert children and all(not Path(f'/proc/{pid}').exists() for pid in children)
        assert protected_files(run) == before
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
