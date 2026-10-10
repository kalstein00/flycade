"""Reward and action audit survives checkpoint resume through the public CLI."""
import json

import pytest

from test_graph_cli import cli
from test_training_cli import prepared_graph


def test_reward_totals_and_action_counts_survive_resume(tmp_path):
    graph = prepared_graph(tmp_path)
    game = tmp_path / 'game.json'
    game.write_text(json.dumps({'progress_reward': .01, 'survival_reward': -.001,
                               'max_frames': 8}))
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', run, '--updates', 2, '--rollout-steps', 4,
                 '--stop-after-updates', 1, '--config', game)
    assert result.returncode == 0, result.stdout + result.stderr
    first = json.loads(result.stdout)
    assert sum(first['action_counts']) == 4
    result = cli('resume', run)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    rows = [json.loads(line) for line in (run / 'transitions.jsonl').read_text().splitlines()]
    assert len(rows) == 8
    assert sum(report['action_counts']) == 8
    assert report['reward_components']['time'] == pytest.approx(-.032)
    assert sum(report['reward_components'].values()) == pytest.approx(report['reward_sum'])
    for name, total in report['reward_components'].items():
        assert total == pytest.approx(sum(row['reward_components'][name] for row in rows))
    for action, count in enumerate(report['action_counts']):
        assert count == sum(row['action'] == action for row in rows)
    assert sum(report['rollout_action_counts']) == 4
    assert sum(report['rollout_mean_probabilities']) == pytest.approx(1)
    assert 0 <= report['rollout_entropy'] <= 1.946


def test_trial_refuses_to_overwrite_an_existing_run(tmp_path):
    import subprocess
    import sys
    run = tmp_path / 'run'
    run.mkdir()
    marker = run / 'keep'
    marker.write_text('existing experiment')
    result = subprocess.run([sys.executable, 'scripts/reward_trial.py', '--output', str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 2 and 'Run already exists' in result.stderr
    assert marker.read_text() == 'existing experiment'
    assert not (tmp_path / 'ready.json').exists()


def test_trial_rejects_missing_resume_run_without_creating_output(tmp_path):
    import subprocess
    import sys
    output = tmp_path / 'segment'
    result = subprocess.run([sys.executable, 'scripts/reward_trial.py', '--output', str(output),
                             '--resume-run', str(tmp_path / 'missing')], capture_output=True, text=True)
    assert result.returncode == 2 and 'Resume Run is missing' in result.stderr
    assert not output.exists()
