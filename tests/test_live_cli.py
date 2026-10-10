"""Live observation through CLI artifacts: actual action forward, no RNG intervention."""
import base64
import hashlib
import io
import json

import numpy as np
import torch
from PIL import Image

from test_graph_cli import cli
from test_training_cli import prepared_graph


def test_observation_matches_action_and_preserves_training_rng(tmp_path):
    graph = prepared_graph(tmp_path)
    runs = []
    for hz in (0, 5):
        run = tmp_path / f'run-{hz}'
        result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                     '--output', run, '--updates', 2, '--rollout-steps', 32, '--observe-hz', hz)
        assert result.returncode == 0, result.stdout + result.stderr
        runs.append(run)
    off, on = [torch.load(run / 'final.pt', weights_only=False) for run in runs]
    for name in off['model']:
        a, b = off['model'][name], on['model'][name]
        assert torch.equal(a.to_dense() if a.is_sparse else a, b.to_dense() if b.is_sparse else b)
    assert torch.equal(off['torch_rng'], on['torch_rng'])
    live = json.loads((runs[1] / 'live' / 'latest.json').read_text())
    assert live['status'] == 'completed'
    sample = live['sample']
    rows = [json.loads(line) for line in (runs[1] / 'transitions.jsonl').read_text().splitlines()]
    row = rows[sample['step'] - 1]
    assert sample['action'] == row['action']
    assert sample['probabilities'] == row['probabilities']
    assert sample['transition']['reward'] == row['reward']
    assert sample['episode'] == row['episode']
    assert sample['session_id'] == row['session_id']
    assert sample['observation_sha256'] == row['observation_sha256']
    frames = [np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(',')[1])))) for uri in sample['pixels']]
    assert hashlib.sha256(np.stack(frames).tobytes()).hexdigest() == row['observation_sha256']
    assert len(sample['activity']) == len(live['graph']['nodes'])
    assert all(len(state) == 8 for state in sample['activity'])
    assert live['observer']['queue_capacity'] == 1
    assert live['observer']['history_capacity'] == 0
    assert live['observer']['published'] > 0
    assert not (runs[0] / 'live').exists()


def test_resume_without_observation_clears_previous_session_sample(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                 '--updates', 2, '--rollout-steps', 8, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    previous = json.loads((run / 'live' / 'latest.json').read_text())
    result = cli('resume', run, '--observe-hz', 0)
    assert result.returncode == 0, result.stdout + result.stderr
    current = json.loads((run / 'live' / 'latest.json').read_text())
    assert current['generation'] > previous['generation']
    assert current['session_id'] == json.loads(result.stdout)['session_id']
    assert current['status'] == 'disabled'
    assert current['sample'] is None


def test_live_graph_keeps_display_bound_for_long_real_paths(tmp_path):
    import pyarrow as pa
    import pyarrow.feather as feather
    from test_graph_cli import fixture
    cache, source, config = fixture(tmp_path)
    ids = list(range(720575940000000001, 720575940000000131))
    feather.write_feather(pa.table({'pre_pt_root_id': ids[:-1], 'post_pt_root_id': ids[1:],
        'neuropil': ['ME_L'] * 129, 'syn_count': [5] * 129}), cache / 'connections.feather')
    np.save(cache / 'roots.npy', np.array(ids, dtype=np.uint64))
    (cache / 'annotations.tsv').write_text('root_id\tcell_type\tsuper_class\n' + ''.join(
        f'{rid}\t{kind}\toptic\n' for rid, kind in zip(ids, ['input'] + ['middle'] * 128 + ['output'])))
    manifest = json.loads(source.read_text())
    for spec in manifest['files'].values():
        spec['sha256'] = hashlib.sha256((cache / spec['name']).read_bytes()).hexdigest()
    source.write_text(json.dumps(manifest))
    graph = tmp_path / 'long-graph'
    result = cli('prepare-graph', '--cache', cache, '--source-manifest', source, '--config', config, '--output', graph)
    assert result.returncode == 0, result.stdout + result.stderr
    training = tmp_path / 'training.json'
    training.write_text(json.dumps({'updates': 1, 'rollout_steps': 2, 'epochs': 1, 'propagation_steps': 129, 'state_dim': 2}))
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run, '--training-config', training)
    assert result.returncode == 0, result.stdout + result.stderr
    envelope = json.loads((run / 'live' / 'latest.json').read_text())
    assert envelope['graph']['used_nodes'] == 130
    assert len(envelope['graph']['nodes']) <= 128
    assert len(envelope['graph']['edges']) <= 512
    assert any(node['input'] for node in envelope['graph']['nodes'])
    assert any(node['output'] for node in envelope['graph']['nodes'])
