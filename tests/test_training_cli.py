"""New Run lifecycle through a real CLI process, synthetic CPU evidence only."""
import json
import os

import pytest
pytest.importorskip('torch')
from test_graph_cli import cli, fixture


def prepared_graph(tmp_path):
    cache, source, config = fixture(tmp_path)
    settings = json.loads(config.read_text())
    settings['min_synapses'] = 4  # Three edges: exercise the [2, E] artifact layout.
    config.write_text(json.dumps(settings))
    graph = tmp_path / 'graph'
    result = cli('prepare-graph', '--cache', cache, '--source-manifest', source,
                 '--config', config, '--output', graph)
    assert result.returncode == 0, result.stdout + result.stderr
    return graph


def test_new_run_protects_initial_policy_and_updates_at_safe_boundary(tmp_path):
    graph = prepared_graph(tmp_path)
    output = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', output, '--updates', 2, '--rollout-steps', 8)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'completed'
    assert report['evidence'] == 'synthetic CPU fixture; not NES/GPU acceptance'
    assert report['transitions'] == 16
    assert report['emulator_frames'] == 64
    assert report['updates'] == 2
    assert report['optimizer_steps'] == 4
    assert report['environment_closed'] and report['workers_alive'] == 0
    assert report['safe_boundary']
    assert report['initial_sha256'] != report['final_sha256']
    assert all(value > 0 for value in report['parameter_delta_l2'].values())
    assert report['graph_influence']['max_probability_delta'] > 1e-7
    manifest = json.loads((output / 'run.json').read_text())
    assert manifest['run_id'] and manifest['session_id']
    assert manifest['graph']['graph_sha256']
    assert manifest['versions']['torch']
    assert manifest['code']['files']
    assert not os.stat(output / 'initial.pt').st_mode & 0o222
    rows = [json.loads(line) for line in (output / 'updates.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and all(row['nonfinite_count'] == 0 for row in rows)
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', output, '--updates', 1, '--rollout-steps', 8)
    assert result.returncode == 2
    assert json.loads(result.stdout)['error']['code'] == 'output_exists'


def test_training_config_is_honored_and_short_propagation_is_rejected(tmp_path):
    graph = prepared_graph(tmp_path)
    config = tmp_path / 'training.json'
    config.write_text(json.dumps({'updates': 1, 'rollout_steps': 3, 'epochs': 1, 'seed': 19}))
    output = tmp_path / 'configured'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', output, '--training-config', config)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['transitions'] == 3 and report['updates'] == 1 and report['optimizer_steps'] == 1
    assert json.loads((output / 'run.json').read_text())['seed'] == 19
    config.write_text(json.dumps({'propagation_steps': 1}))
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', tmp_path / 'unreachable', '--training-config', config)
    assert result.returncode == 2
    assert 'No input-to-output path' in result.stdout
    assert not (tmp_path / 'unreachable').exists()


def test_external_episode_limit_is_counted_separately_from_run_budget(tmp_path):
    graph = prepared_graph(tmp_path)
    game = tmp_path / 'game.json'
    game.write_text(json.dumps({'max_frames': 3}))
    output = tmp_path / 'limited'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', output, '--updates', 1, '--rollout-steps', 4, '--config', game)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['episodes'] == 4 and report['transitions'] == 4
    assert report['emulator_frames'] == 12
    assert report['episode_outcomes'] == {'external_limit': 4}
    assert report['safe_boundary'] and report['environment_closed']
    rows = [json.loads(line) for line in (output / 'transitions.jsonl').read_text().splitlines()]
    assert all(row['truncated'] and not row['terminated'] for row in rows)
    torch = pytest.importorskip('torch')
    initial = torch.load(output / 'initial.pt', weights_only=True)
    final = torch.load(output / 'final.pt', weights_only=False)
    manifest = json.loads((output / 'run.json').read_text())
    for name in manifest['model']['trainable']:
        assert torch.isfinite(final['model'][name]).all()
        assert not torch.equal(initial[name], final['model'][name]), name
    assert final['progress']['safe_boundary']
    assert all(int(state['step']) == 2 for state in final['optimizer']['state'].values())


def test_nonfinite_update_fails_without_claiming_safe_boundary_and_closes_environment(tmp_path):
    graph = prepared_graph(tmp_path)
    game = tmp_path / 'game.json'
    game.write_text(json.dumps({'survival_reward': 1e30}))
    output = tmp_path / 'failed'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', output, '--updates', 1, '--rollout-steps', 4, '--config', game)
    assert result.returncode == 2
    assert json.loads(result.stdout)['error']['code'] == 'nonfinite_training'
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'failed' and not report['safe_boundary']
    assert report['updates'] == 0 and report['nonfinite_count'] == 1
    assert report['environment_closed'] and report['workers_alive'] == 0
    assert (output / 'initial.pt').is_file() and not (output / 'final.pt').exists()
