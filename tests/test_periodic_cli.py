"""Periodic evaluation and best selection at the public CLI/artifact boundary."""
import json

import torch

from test_graph_cli import cli
from test_training_cli import prepared_graph


def test_periodic_evaluation_survives_resume_and_preserves_training_state(tmp_path):
    graph = prepared_graph(tmp_path)
    protocol = tmp_path / 'evaluation.json'
    protocol.write_text(json.dumps({'seeds': [11,22,33], 'max_frames': 16, 'video_seconds': 1}))
    runs = [tmp_path / 'control', tmp_path / 'evaluated']
    for index, run in enumerate(runs):
        config = tmp_path / f'config-{index}.json'
        config.write_text(json.dumps({'updates': 4, 'rollout_steps': 4, 'evaluation_every_updates': 2 if index else 0}))
        result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                     '--training-config', config, '--initial-evaluation-config', protocol,
                     '--stop-after-updates', 2)
        assert result.returncode == 0, result.stdout + result.stderr
        if index:
            report = json.loads(result.stdout)
            assert report['next_evaluation_update'] == 4
            assert len(report['evaluation_history']) == 2
        result = cli('resume', run)
        assert result.returncode == 0, result.stdout + result.stderr
    control, evaluated = [torch.load(run / 'final.pt', weights_only=False) for run in runs]
    for name, value in control['model'].items():
        assert torch.equal(value.to_dense(), evaluated['model'][name].to_dense())
    assert torch.equal(control['torch_rng'], evaluated['torch_rng'])
    assert evaluated['progress']['next_evaluation_update'] == 6
    assert len(evaluated['progress']['evaluation_history']) == 3
    results = [json.loads(path.read_text()) for path in (runs[1] / 'evaluations').glob('*/report.json')]
    assert sorted(row['snapshot_updates'] for row in results) == [0,2,4]
    assert len({row['protocol_id'] for row in results}) == 1
    assert all(row['training_paused'] and row['environment_closed'] for row in results)
    assert evaluated['progress']['updates'] == 4 and evaluated['progress']['optimizer_steps'] == 8


def test_best_keeps_ties_separates_protocols_and_protects_checkpoint(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                 '--updates', 8, '--rollout-steps', 4, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = json.loads(result.stdout)['checkpoint_id']
    protocols = [tmp_path / 'protocol-a.json', tmp_path / 'protocol-b.json']
    reports = []
    for path, seeds in zip(protocols, ('11,22,33', '41,42,43')):
        assert cli('evaluation-protocol', run, '--output', path, '--seeds', seeds, '--max-frames', 16, '--video-seconds', 1).returncode == 0
        for _ in range(2):
            result = cli('evaluate', run, '--snapshot', checkpoint, '--protocol', path)
            assert result.returncode == 0, result.stdout + result.stderr
            reports.append(json.loads(result.stdout))
    result = cli('evaluations', run)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert len(summary['protocols']) == 2
    for group in summary['protocols']:
        matching = [r for r in reports if r['protocol_id'] == group['protocol_id']]
        assert group['best'] == matching[0]['evaluation_id']
        assert group['latest'] == matching[-1]['evaluation_id']
        assert group['best_snapshot'] == checkpoint
    for _ in range(4):
        result = cli('resume', run, '--stop-after-updates', 1)
        assert result.returncode == 0, result.stdout + result.stderr
    history = json.loads(cli('checkpoints', run).stdout)
    assert checkpoint in history['references']['best']
    assert next(row for row in history['checkpoints'] if row['checkpoint_id'] == checkpoint)['protected']


def test_branch_preserves_evaluation_schedule_and_parent_history(tmp_path):
    graph = prepared_graph(tmp_path)
    parent, child = tmp_path / 'parent', tmp_path / 'child'
    config, evaluation = tmp_path / 'training.json', tmp_path / 'evaluation.json'
    config.write_text(json.dumps({'updates': 4, 'rollout_steps': 4, 'evaluation_every_updates': 1}))
    evaluation.write_text(json.dumps({'seeds': [11,22,33], 'max_frames': 16, 'video_seconds': 1}))
    trained = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', parent,
                  '--training-config', config, '--initial-evaluation-config', evaluation, '--stop-after-updates', 1)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    report = json.loads(trained.stdout)
    before = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    result = cli('branch', parent, '--checkpoint', report['checkpoint_id'], '--output', child)
    assert result.returncode == 0, result.stdout + result.stderr
    result = cli('resume', child, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    resumed = json.loads(result.stdout)
    assert resumed['updates'] == 2 and resumed['next_evaluation_update'] == 3
    assert resumed['inherited_evaluation_history']['run_id'] == report['run_id']
    assert resumed['inherited_evaluation_history']['evaluation_ids'] == report['evaluation_history']
    archived = json.loads((child / 'origin' / 'evaluation-index.json').read_text())
    assert archived['run_id'] == report['run_id']
    assert len(resumed['evaluation_history']) == 1  # Child results keep the child's identity.
    assert {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob('*') if p.is_file()} == before


def test_ranking_prioritizes_completion_then_distance_and_keeps_incumbent(tmp_path):
    """Controlled report artifacts exercise the public index command, not gameplay claims."""
    import uuid
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'rank'
    trained = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                  '--updates', 2, '--rollout-steps', 4)
    assert trained.returncode == 0
    protocol = tmp_path / 'protocol.json'
    assert cli('evaluation-protocol', run, '--output', protocol, '--max-frames', 16, '--video-seconds', 1).returncode == 0
    evaluated = cli('evaluate', run, '--snapshot', 'initial', '--protocol', protocol)
    assert evaluated.returncode == 0
    base = json.loads(evaluated.stdout)
    fixtures = [(0.,900.), (1/3,10.), (1/3,20.), (1/3,20.), (0.,1000.)]
    identifiers = []
    for offset, (rate, distance) in enumerate(fixtures, 1):
        identifier = str(uuid.uuid4()); identifiers.append(identifier)
        directory = run / 'evaluations' / identifier; directory.mkdir()
        (directory / 'protocol.json').write_bytes(protocol.read_bytes())
        row = {**base, 'evaluation_id': identifier, 'created_unix': base['created_unix']+offset,
               'completion_rate': rate, 'mean_distance': distance, 'evidence': 'controlled ranking fixture'}
        (directory / 'report.json').write_text(json.dumps(row))
        result = cli('evaluations', run, '--refresh')
        assert result.returncode == 0, result.stdout + result.stderr
        best = json.loads(result.stdout)['protocols'][0]['best']
        assert best == identifiers[min(offset-1, 2)]


def test_completed_budget_can_recover_pending_final_evaluation(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    config, evaluation = tmp_path / 'training.json', tmp_path / 'evaluation.json'
    config.write_text(json.dumps({'updates': 1, 'rollout_steps': 4, 'evaluation_every_updates': 1}))
    evaluation.write_text(json.dumps({'seeds': [11,22,33], 'max_frames': 16, 'video_seconds': 1}))
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                 '--training-config', config, '--initial-evaluation-config', evaluation)
    assert result.returncode == 0, result.stdout + result.stderr
    trained = torch.load(run / 'final.pt', weights_only=False)
    latest = json.loads((run / 'latest.json').read_text())
    (run / latest['file']).write_bytes(b'failure after final evaluation')
    result = cli('resume', run)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['updates'] == 1 and report['next_evaluation_update'] == 2
    assert report['status'] == 'completed'
    recovered = torch.load(run / 'final.pt', weights_only=False)
    for name, value in trained['model'].items():
        assert torch.equal(value.to_dense(), recovered['model'][name].to_dense())
    assert recovered['progress']['optimizer_steps'] == trained['progress']['optimizer_steps']
    assert cli('resume', run).returncode == 2  # No budget restart after recovery.


def test_stop_during_periodic_evaluation_saves_without_extra_training(tmp_path):
    import signal
    import subprocess
    import sys
    from pathlib import Path
    from test_daily_cli import wait_report
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'stopped'
    config, evaluation = tmp_path / 'training.json', tmp_path / 'evaluation.json'
    config.write_text(json.dumps({'updates': 20, 'rollout_steps': 4, 'evaluation_every_updates': 1}))
    evaluation.write_text(json.dumps({'seeds': [11,22,33], 'max_frames': 120, 'video_seconds': 1}))
    process = subprocess.Popen([sys.executable, '-m', 'flycade', 'train', '--fixture', '--device', 'cpu',
        '--graph', str(graph), '--output', str(run), '--training-config', str(config),
        '--initial-evaluation-config', str(evaluation)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_report(run, lambda r: r['status'] == 'evaluating_checkpoint', process)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        report = json.loads(stdout)
        assert report['updates'] == 1 and report['status'] == 'saved'
        assert report['next_evaluation_update'] == 2
        assert report['evaluation_worker_reaped'] and not report['training_paused']
        assert not Path(f"/proc/{report['evaluation_worker_pid']}").exists()
    finally:
        if process.poll() is None:
            process.kill(); process.communicate(timeout=10)


def test_stop_during_retried_evaluation_does_not_advance_optimizer(tmp_path):
    import signal
    import subprocess
    import sys
    from test_daily_cli import wait_report
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'retry-stop'
    config, evaluation = tmp_path / 'training.json', tmp_path / 'evaluation.json'
    config.write_text(json.dumps({'updates': 20, 'rollout_steps': 4, 'evaluation_every_updates': 1}))
    evaluation.write_text(json.dumps({'seeds': [11,22,33], 'max_frames': 120, 'video_seconds': 1}))
    trained = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                  '--training-config', config, '--initial-evaluation-config', evaluation, '--stop-after-updates', 1)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    latest = json.loads((run / 'latest.json').read_text())
    (run / latest['file']).write_bytes(b'force pending-evaluation fallback')
    process = subprocess.Popen([sys.executable, '-m', 'flycade', 'resume', str(run)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_report(run, lambda r: r['status'] == 'evaluating_checkpoint', process)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        result = json.loads(stdout)
        assert result['updates'] == 1 and result['optimizer_steps'] == 2
        assert result['status'] == 'saved' and result['next_evaluation_update'] == 2
    finally:
        if process.poll() is None:
            process.kill(); process.communicate(timeout=10)
