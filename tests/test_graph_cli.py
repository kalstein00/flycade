"""Preparation contract at the agreed process/artifact boundary."""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

pa = pytest.importorskip('pyarrow', reason='Install the graph extra for graph CLI integration tests')
feather = pytest.importorskip('pyarrow.feather')
pytest.importorskip('duckdb')


IDS = [str(720575940000000001 + i) for i in range(5)]


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'flycade', *map(str, args)],
                          capture_output=True, text=True)


def fixture(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    # A->B crosses the threshold only after summing neuropils. B->A does not.
    feather.write_feather(pa.table({
        'pre_pt_root_id': [int(IDS[i]) for i in [0, 0, 1, 1, 2, 3]],
        'post_pt_root_id': [int(IDS[i]) for i in [1, 1, 0, 2, 2, 0]],
        'neuropil': ['ME_L', 'LO_L', 'ME_L', 'ME_L', 'ME_L', 'ME_L'],
        'syn_count': [2, 3, 4, 7, 9, 6],
    }), cache / 'connections.feather')
    np.save(cache / 'roots.npy', np.array(IDS, dtype=np.uint64))
    (cache / 'annotations.tsv').write_text('root_id\tcell_type\tsuper_class\n' + ''.join(
        f'{rid}\t{kind}\toptic\n' for rid, kind in zip(IDS, ['input', 'middle', 'output', 'excluded', 'output'])))
    manifest = {'dataset': 'synthetic-fixture', 'release': 'test', 'files': {}}
    for role, name in [('connections', 'connections.feather'), ('neurons', 'roots.npy'), ('annotations', 'annotations.tsv')]:
        manifest['files'][role] = {'name': name, 'url': (cache / name).as_uri(),
            'sha256': hashlib.sha256((cache / name).read_bytes()).hexdigest(),
            'license': 'CC0-1.0', 'citation': 'synthetic test fixture'}
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(manifest))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'selection': {'cell_types': ['input', 'middle', 'output']},
        'inputs': {'cell_types': ['input']}, 'outputs': {'cell_types': ['output']},
        'min_synapses': 5, 'self_loops': 'drop', 'seed': 0}))
    return cache, source, config


def test_prepare_aggregates_before_threshold_and_preserves_direction_ids(tmp_path):
    cache, source, config = fixture(tmp_path)
    output = tmp_path / 'graph'
    result = cli('prepare-graph', '--cache', cache, '--source-manifest', source,
                 '--config', config, '--output', output)
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = json.loads((output / 'nodes.json').read_text())
    assert [node['root_id'] for node in nodes] == [IDS[i] for i in [0, 1, 2, 4]]
    assert np.load(output / 'edge_index.npy').tolist() == [[0, 1], [1, 2]]
    assert np.load(output / 'syn_count.npy').tolist() == [5, 7]
    assert np.load(output / 'weight.npy').tolist() == [5.0, 7.0]
    report = json.loads((output / 'report.json').read_text())
    assert report['counts']['source_neurons'] == 5
    assert report['counts']['source_region_edges'] == 6
    assert report['counts']['source_pair_edges'] == 5
    assert report['counts']['source_synapses'] == 31
    assert report['counts']['used_neurons'] == 4
    assert report['counts']['used_edges'] == 2
    assert report['counts']['used_synapses'] == 12
    assert report['connectivity']['input_reach_fraction'] == 1.0
    assert report['connectivity']['output_reach_fraction'] == 0.5
    assert report['connectivity']['example_path'] == IDS[:3]


def test_reconstructs_offline_and_detects_missing_or_corrupt_artifacts(tmp_path):
    cache, source, config = fixture(tmp_path)
    outputs = [tmp_path / 'first', tmp_path / 'second']
    for output in outputs:
        result = cli('prepare-graph', '--cache', cache, '--source-manifest', source,
                     '--config', config, '--output', output)
        assert result.returncode == 0, result.stdout + result.stderr
    first, second = [json.loads((out / 'manifest.json').read_text()) for out in outputs]
    assert first['graph_sha256'] == second['graph_sha256']
    assert first['artifacts'] == second['artifacts']
    result = cli('inspect-graph', outputs[0], '--cache', cache)
    assert result.returncode == 0, result.stdout + result.stderr
    (outputs[0] / 'edge_index.npy').unlink()
    result = cli('inspect-graph', outputs[0])
    assert json.loads(result.stdout)['error']['code'] == 'artifact_missing'
    (outputs[1] / 'nodes.json').write_text('[]')
    result = cli('inspect-graph', outputs[1])
    assert json.loads(result.stdout)['error']['code'] == 'artifact_corrupt'
    (cache / 'connections.feather').write_bytes(b'broken')
    result = cli('prepare-graph', '--cache', cache, '--source-manifest', source,
                 '--config', config, '--output', tmp_path / 'third')
    assert json.loads(result.stdout)['error']['code'] == 'artifact_corrupt'
    assert not (tmp_path / 'third').exists()


def test_fetch_verifies_then_reuses_cache_without_contacting_source(tmp_path):
    cache, source, _ = fixture(tmp_path)
    downloaded = tmp_path / 'downloaded'
    result = cli('fetch-graph-data', '--cache', downloaded, '--source-manifest', source)
    assert result.returncode == 0, result.stdout + result.stderr
    for path in cache.iterdir():
        path.unlink()
    result = cli('fetch-graph-data', '--cache', downloaded, '--source-manifest', source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(item['status'] == 'cached' for item in json.loads(result.stdout)['files'].values())
    (downloaded / 'roots.npy').write_bytes(b'bad cache')
    result = cli('fetch-graph-data', '--cache', downloaded, '--source-manifest', source)
    assert result.returncode == 2
    assert json.loads(result.stdout)['error']['code'] == 'artifact_corrupt'


def test_rejects_reverse_only_path_and_keeps_existing_graph(tmp_path):
    cache, source, config = fixture(tmp_path)
    settings = json.loads(config.read_text())
    settings['inputs'], settings['outputs'] = settings['outputs'], settings['inputs']
    config.write_text(json.dumps(settings))
    output = tmp_path / 'graph'
    result = cli('prepare-graph', '--cache', cache, '--source-manifest', source,
                 '--config', config, '--output', output)
    assert result.returncode == 2
    assert json.loads(result.stdout)['error']['code'] == 'graph_disconnected'
    assert not output.exists()
    output.mkdir()
    (output / 'keep').write_text('existing graph')
    result = cli('prepare-graph', '--cache', cache, '--source-manifest', source,
                 '--config', config, '--output', output)
    assert json.loads(result.stdout)['error']['code'] == 'output_exists'
    assert (output / 'keep').read_text() == 'existing graph'


def test_self_loops_are_explicit_and_numeric_ids_are_rejected(tmp_path):
    cache, source, config = fixture(tmp_path)
    settings = json.loads(config.read_text())
    settings['self_loops'] = 'keep'
    config.write_text(json.dumps(settings))
    output = tmp_path / 'graph'
    result = cli('prepare-graph', '--cache', cache, '--source-manifest', source,
                 '--config', config, '--output', output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert np.load(output / 'edge_index.npy').tolist() == [[0, 1, 2], [1, 2, 2]]
    assert np.load(output / 'syn_count.npy').tolist() == [5, 7, 9]
    settings['inputs'] = {'root_ids': [int(IDS[0])]}
    config.write_text(json.dumps(settings))
    result = cli('prepare-graph', '--cache', cache, '--source-manifest', source,
                 '--config', config, '--output', tmp_path / 'bad')
    assert result.returncode == 2
    assert 'strings' in json.loads(result.stdout)['error']['message']


def test_inspection_checks_measurement_and_rejects_malformed_source_manifest(tmp_path):
    cache, source, config = fixture(tmp_path)
    output = tmp_path / 'graph'
    assert cli('prepare-graph', '--cache', cache, '--source-manifest', source,
               '--config', config, '--output', output).returncode == 0
    (output / 'measurement.json').unlink()
    result = cli('inspect-graph', output)
    assert result.returncode == 2
    assert json.loads(result.stdout)['error']['code'] == 'artifact_missing'
    source.write_text('[]')
    result = cli('fetch-graph-data', '--cache', cache, '--source-manifest', source)
    assert result.returncode == 2
    assert 'Traceback' not in result.stderr


def test_graph_identity_is_independent_of_source_row_order(tmp_path):
    cache, source, config = fixture(tmp_path)
    arguments = ['--cache', cache, '--source-manifest', source, '--config', config]
    first = cli('prepare-graph', *arguments, '--output', tmp_path / 'first')
    assert first.returncode == 0, first.stdout + first.stderr
    path = cache / 'connections.feather'
    table = feather.read_table(path)
    feather.write_feather(table.take(pa.array([5, 4, 3, 2, 1, 0])), path)
    manifest = json.loads(source.read_text())
    manifest['files']['connections']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    source.write_text(json.dumps(manifest))
    second = cli('prepare-graph', *arguments, '--output', tmp_path / 'second')
    assert second.returncode == 0, second.stdout + second.stderr
    assert json.loads(first.stdout)['graph_sha256'] == json.loads(second.stdout)['graph_sha256']
    assert json.loads(first.stdout)['source_manifest_sha256'] != json.loads(second.stdout)['source_manifest_sha256']
