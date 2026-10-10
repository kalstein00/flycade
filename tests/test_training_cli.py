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


def test_resume_preserves_budget_and_matches_reset_control(tmp_path):
    import torch
    graph = prepared_graph(tmp_path)
    game = tmp_path / 'reset-each-rollout.json'
    game.write_text(json.dumps({'max_frames': 16}))
    control, resumed = tmp_path / 'control', tmp_path / 'resumed'
    common = ('--fixture', '--device', 'cpu', '--graph', graph,
              '--updates', 3, '--rollout-steps', 4, '--config', game)
    result = cli('train', *common, '--output', control)
    assert result.returncode == 0, result.stdout + result.stderr
    result = cli('train', *common, '--output', resumed, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    first = json.loads(result.stdout)
    assert first['status'] == 'saved' and first['updates'] == 1
    saved = torch.load(resumed / 'final.pt', weights_only=False)
    result = cli('resume', resumed)
    assert result.returncode == 0, result.stdout + result.stderr
    final = json.loads(result.stdout)
    assert final['run_id'] == first['run_id']
    assert final['session_id'] != first['session_id']
    assert final['updates'] == 3 and final['transitions'] == 12
    assert final['optimizer_steps'] == 6 and final['episodes'] == 3
    assert final['environment_closed'] and final['workers_alive'] == 0
    actual = torch.load(resumed / 'final.pt', weights_only=False)
    expected = torch.load(control / 'final.pt', weights_only=False)
    for name, tensor in expected['model'].items():
        assert torch.equal(tensor.to_dense(), actual['model'][name].to_dense()), name
    for key, state in expected['optimizer']['state'].items():
        for name, tensor in state.items():
            assert torch.equal(tensor, actual['optimizer']['state'][key][name])
    assert torch.equal(expected['torch_rng'], actual['torch_rng'])
    assert saved['progress']['updates'] == 1
    assert len(list((resumed / 'checkpoints').glob('*.pt'))) == 2


def test_training_video_history_survives_resume(tmp_path):
    import subprocess
    graph = prepared_graph(tmp_path)
    output = tmp_path / 'video-run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', output, '--updates', 2, '--rollout-steps', 8, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    first = json.loads(result.stdout)
    result = cli('resume', output)
    assert result.returncode == 0, result.stdout + result.stderr
    result = cli('history', output)
    assert result.returncode == 0, result.stdout + result.stderr
    history = json.loads(result.stdout)
    assert len(history['recordings']) == 2
    assert {r['session_id'] for r in history['recordings']} == {
        first['session_id'], json.loads((output / 'report.json').read_text())['session_id']}
    for recording in history['recordings']:
        assert recording['mode'] == 'training_recording'
        assert recording['run_id'] == first['run_id']
        assert recording['status'] == 'complete'
        path = output / recording['file']
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)],
                               capture_output=True, text=True, check=True)
        streams = json.loads(probe.stdout)['streams']
        assert len(streams) == 1 and streams[0]['codec_name'] == 'vp9'
        assert streams[0]['r_frame_rate'] == '12/1'
        assert path.stat().st_size < 20000


def test_resume_rejects_changed_identity_and_missing_artifacts_before_training(tmp_path):
    import shutil
    graph = prepared_graph(tmp_path)
    original = tmp_path / 'original'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', original, '--updates', 2, '--rollout-steps', 4, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    for field, value in [('game', {'width': 32}), ('training', {'state_dim': 4}),
                         ('checkpoint_format', 999), ('model', {'architecture': 'different'})]:
        run = tmp_path / field
        shutil.copytree(original, run, symlinks=True)
        manifest = json.loads((run / 'run.json').read_text())
        manifest[field] = value
        (run / 'run.json').write_text(json.dumps(manifest))
        before = (run / 'transitions.jsonl').read_bytes()
        result = cli('resume', run)
        assert result.returncode == 2 and 'checkpoint_incompatible' in result.stdout
        assert (run / 'transitions.jsonl').read_bytes() == before
    for filename in ['graph/weight.npy', 'initial.pt']:
        run = tmp_path / filename.replace('/', '-')
        shutil.copytree(original, run, symlinks=True)
        (run / filename).unlink()
        result = cli('resume', run)
        assert result.returncode == 2 and ('Missing' in result.stdout or 'missing' in result.stdout)
    run = tmp_path / 'corrupt'
    shutil.copytree(original, run, symlinks=True)
    (run / 'final.pt').write_bytes(b'corrupt')
    result = cli('resume', run)
    assert result.returncode == 2 and 'checksum' in result.stdout


def test_interrupted_episode_is_reset_without_counting_a_terminal(tmp_path):
    graph = prepared_graph(tmp_path)
    output = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', output, '--updates', 2, '--rollout-steps', 4, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    result = cli('resume', output)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['episodes'] == 0 and report['episode_outcomes'] == {}
    assert report['resume_reset'] == {'frames': 0, 'max_progress': 0, 'stack_frames_equal': True}
    rows = [json.loads(line) for line in (output / 'transitions.jsonl').read_text().splitlines()]
    assert rows[4]['observation_sha256'] == rows[0]['observation_sha256']
    assert rows[4]['transition'] == 5


def test_sigint_saves_at_update_boundary_and_reaps_encoder(tmp_path):
    import select
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path
    graph = prepared_graph(tmp_path)
    output = tmp_path / 'interrupt'
    process = subprocess.Popen([sys.executable, '-m', 'flycade', 'train', '--fixture', '--device', 'cpu',
                                '--graph', str(graph), '--output', str(output), '--updates', '10000',
                                '--rollout-steps', '16'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    children = set()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            child_file = Path(f'/proc/{process.pid}/task/{process.pid}/children')
            if child_file.exists():
                children.update(child_file.read_text().split())
            if (output / 'updates.jsonl').exists() and (output / 'updates.jsonl').stat().st_size:
                break
            assert process.poll() is None
            time.sleep(.02)
        else:
            pytest.fail('training did not reach first update')
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        report = json.loads(stdout)
        assert report['status'] == 'saved' and report['safe_boundary']
        assert report['transitions'] == report['updates'] * 16
        assert 'Save-and-stop requested' in stderr
        assert children and all(not Path(f'/proc/{pid}').exists() for pid in children)
        assert report['environment_closed'] and report['recording_status'] == 'complete'
        result = cli('resume', output, '--stop-after-updates', 1)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)['updates'] == report['updates'] + 1
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_failed_checkpoint_write_keeps_last_published_state(tmp_path):
    import hashlib
    import resource
    import signal
    import subprocess
    import sys
    graph = prepared_graph(tmp_path)
    output = tmp_path / 'write-failure'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', output, '--updates', 3, '--rollout-steps', 4, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    pointer = (output / 'latest.json').read_bytes()
    checksum = hashlib.sha256((output / 'final.pt').read_bytes()).hexdigest()
    limit = (output / 'final.pt').stat().st_size // 2

    def limit_file_size():
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))

    failed = subprocess.run([sys.executable, '-m', 'flycade', 'resume', str(output), '--stop-after-updates', '1'],
                            preexec_fn=limit_file_size, capture_output=True, text=True, timeout=30)
    assert failed.returncode == 2, failed.stdout + failed.stderr
    assert (output / 'latest.json').read_bytes() == pointer
    assert hashlib.sha256((output / 'final.pt').read_bytes()).hexdigest() == checksum
    assert not list((output / 'checkpoints').glob('*.partial'))
    result = cli('resume', output, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['updates'] == 2


def test_encoder_failure_does_not_prevent_checkpoint_or_resume(tmp_path):
    import shutil
    import subprocess
    import sys
    graph = prepared_graph(tmp_path)
    output = tmp_path / 'encoder-failure'
    executables = tmp_path / 'bin'
    executables.mkdir()
    (executables / 'git').symlink_to(shutil.which('git'))
    encoder = executables / 'ffmpeg'
    encoder.write_text('#!/bin/sh\nexit 1\n')
    encoder.chmod(0o755)
    result = subprocess.run([sys.executable, '-m', 'flycade', 'train', '--fixture', '--device', 'cpu',
        '--graph', str(graph), '--output', str(output), '--updates', '2', '--rollout-steps', '8',
        '--stop-after-updates', '1'], env={**os.environ, 'PATH': str(executables)},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'saved' and report['recording_status'] == 'failed'
    assert report['recording_error'] and (output / 'latest.json').exists()
    assert not list((output / 'videos').glob('*.partial'))
    result = cli('resume', output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['recording_status'] == 'complete'


def test_unwritable_recording_directory_does_not_block_learning_resume(tmp_path):
    graph = prepared_graph(tmp_path)
    output = tmp_path / 'video-directory-failure'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', output, '--updates', 2, '--rollout-steps', 4, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    (output / 'videos').rename(output / 'preserved-videos')
    (output / 'videos').write_text('not a directory')
    result = cli('resume', output)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['updates'] == 2 and report['status'] == 'completed'
    assert report['recording_status'] == 'failed' and report['recording_error']
    assert report['environment_closed']
