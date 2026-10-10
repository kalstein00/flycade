"""Bounded actual-forward evidence through CLI and recorded public artifacts."""
import base64
import gzip
import io
import json

import numpy as np
from PIL import Image
import torch

from flycade.policy import policy_from_graph
from test_graph_cli import cli
from test_training_cli import prepared_graph


def test_evaluation_records_bounded_actual_forward_samples(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    evaluation = tmp_path / 'evaluation.json'
    evaluation.write_text(json.dumps({'seeds': [11,22,33], 'max_frames': 120, 'video_seconds': 1}))
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                 '--updates', 1, '--rollout-steps', 4, '--initial-evaluation-config', evaluation)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    directory = run / 'evaluations' / report['initial_evaluation_id']
    evaluated = json.loads((directory / 'report.json').read_text())
    replay = evaluated['replay']
    assert replay['status'] == 'complete' and replay['bytes'] < 2**20
    document = json.loads(gzip.decompress((directory / replay['file']).read_bytes()))
    assert document['format_version'] == 1
    assert document['run_id'] == report['run_id']
    assert document['snapshot_id'] == 'initial'
    assert document['interval_emulator_frames'] == 20
    assert document['maximum_samples'] == 3
    assert [sample['video_time_seconds'] for sample in document['samples']] == [0., 1/3, 2/3]
    assert len(document['graph']['nodes']) <= 128 and len(document['graph']['edges']) <= 512
    policy = policy_from_graph(run / 'graph', 8, 2, 12)
    policy.load_state_dict(torch.load(run / 'initial.pt', weights_only=True))
    indices = [node['index'] for node in document['graph']['nodes']]
    for sample in document['samples']:
        pixels = np.stack([np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(',')[1])))) for uri in sample['pixels']])
        observed = []
        with torch.inference_mode():
            distribution, _ = policy(torch.from_numpy(pixels).unsqueeze(0), observe=lambda state: observed.append(state[0,indices]))
        assert torch.allclose(observed[0], torch.tensor(sample['activity']), atol=1e-7)
        assert torch.allclose(distribution.probs[0], torch.tensor(sample['probabilities']), atol=1e-7)
        assert sample['stream'] == 'evaluation' and sample['episode'] == 0
    assert document['video_file'] == evaluated['videos'][0]['file']


def test_replay_write_failure_keeps_checkpoint_and_reports_error(tmp_path):
    import subprocess
    import sys
    import time
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'fault'
    assert cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
               '--updates', 1, '--rollout-steps', 4).returncode == 0
    before = (run / 'final.pt').read_bytes()
    protocol = tmp_path / 'protocol.json'
    assert cli('evaluation-protocol', run, '--output', protocol, '--seeds', '11', '--mode', 'deterministic',
               '--max-frames', 120, '--video-seconds', 2).returncode == 0
    process = subprocess.Popen([sys.executable, '-m', 'flycade', 'evaluate', str(run), '--snapshot', 'initial',
        '--protocol', str(protocol), '--realtime'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic()+15
        path = None
        while time.monotonic()<deadline:
            paths = list((run / 'evaluations').glob('*/live.json'))
            if paths:
                path = paths[0].parent
                (path / 'observations.json.gz').mkdir()  # Public filesystem publication fault.
                break
            assert process.poll() is None, process.communicate()
            time.sleep(.01)
        assert path is not None
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 2, stdout + stderr
        report = json.loads((path / 'report.json').read_text())
        assert report['status'] == 'failed' and report['replay']['status'] == 'failed'
        assert report['replay']['error'] and report['environment_closed']
        assert not list(path.glob('*.partial'))
        assert (run / 'final.pt').read_bytes() == before
        assert not list((run / 'evaluation-leases').glob('*.json'))
    finally:
        if process.poll() is None:
            process.kill();process.communicate(timeout=10)
