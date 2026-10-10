"""Fixed-source graph preparation. Whole-connectome aggregation stays on disk."""
from collections import deque
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import tempfile
import time
from typing import Any

import numpy as np

from flycade.errors import PreparationError


DEFAULT_SOURCE = Path(__file__).with_name('data') / 'flywire-v783.json'
GRAPH_FILES = ('nodes.json', 'edge_index.npy', 'syn_count.npy', 'weight.npy')


def digest(path: Path, algorithm: str = 'sha256') -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def graph_identity(hashes: dict[str, str]) -> str:
    payload = ''.join(f'{name}:{hashes[name]}\n' for name in GRAPH_FILES)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def read_sources(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('files'), dict):
        raise ValueError('Source manifest must be an object with a files object')
    if set(data['files']) != {'connections', 'neurons', 'annotations'}:
        raise ValueError('Source manifest requires connections, neurons, annotations')
    for spec in data['files'].values():
        if not isinstance(spec, dict) or any(not isinstance(spec.get(k), str) for k in
                ('name', 'sha256', 'url', 'license', 'citation')):
            raise ValueError('Every source requires string name, sha256, url, license, citation')
        if Path(spec['name']).name != spec['name'] or spec['name'] in {'.', '..'}:
            raise ValueError('Source names must be plain filenames')
        if len(spec['sha256']) != 64 or any(c not in '0123456789abcdef' for c in spec['sha256']):
            raise ValueError('Source requires a lowercase SHA-256 digest')
        for field in ('url', 'license', 'citation'):
            if not spec[field]:
                raise ValueError(f'Source requires {field}')
    return data


def check_file(path: Path, expected: str) -> None:
    if not path.is_file():
        raise PreparationError('artifact_missing', f'Missing artifact: {path}')
    if digest(path) != expected:
        raise PreparationError('artifact_corrupt', f'SHA-256 mismatch: {path}')


def configuration(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Graph config must be an object')
    allowed = {'selection', 'inputs', 'outputs', 'min_synapses', 'self_loops', 'seed'}
    if set(config) - allowed:
        raise ValueError(f'Unknown graph settings: {sorted(set(config) - allowed)}')
    config = {'min_synapses': 5, 'self_loops': 'drop', 'seed': 0, **config}
    if type(config['min_synapses']) is not int or config['min_synapses'] < 1:
        raise ValueError('min_synapses must be a positive integer')
    if config['self_loops'] not in ('drop', 'keep') or type(config['seed']) is not int:
        raise ValueError('self_loops must be drop/keep; seed must be an integer')
    for group in ('selection', 'inputs', 'outputs'):
        selector = config[group]
        if not isinstance(selector, dict) or len(selector) != 1 or not set(selector) <= {'root_ids', 'cell_types'}:
            raise ValueError(f'{group} requires exactly one of root_ids or cell_types')
        key, values = next(iter(selector.items()))
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f'{group} must contain a nonempty list of strings (including IDs)')
        if key == 'root_ids':
            for value in values:
                root_id(value)
        selector[key] = sorted(set(values))
    return config


def root_id(value: str) -> str:
    if not value.isascii() or not value.isdigit() or str(int(value)) != value or not 0 < int(value) < 2**64:
        raise ValueError(f'Invalid uint64 root ID: {value!r}')
    return value


def read_neurons(path: Path, annotations_path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    roots = np.load(path, allow_pickle=False)
    if roots.ndim != 1 or roots.dtype.kind not in 'iu':
        raise ValueError('Neuron list must be a one-dimensional integer NPY array')
    ids = [root_id(str(int(value))) for value in roots]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate proofread neuron IDs')
    annotations = {}
    with annotations_path.open(newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {'root_id', 'cell_type', 'super_class'} <= set(reader.fieldnames or []):
            raise ValueError('Annotation columns required: root_id, cell_type, super_class')
        for row in reader:
            rid = root_id(row['root_id'])
            if rid in annotations:
                raise ValueError(f'Duplicate annotation ID: {rid}')
            annotations[rid] = {key: row[key] for key in ('root_id', 'cell_type', 'super_class')}
    return ids, annotations


def select(selector: dict[str, list[str]], ids: list[str], annotations: dict[str, dict[str, str]]) -> list[str]:
    if 'root_ids' in selector:
        selected = set(selector['root_ids'])
        missing = selected - set(ids)
        if missing:
            raise ValueError(f'Selected IDs are not proofread: {sorted(missing)}')
    else:
        kinds = set(selector['cell_types'])
        selected = {rid for rid in ids if annotations.get(rid, {}).get('cell_type') in kinds}
        missing_kinds = kinds - {annotations[rid]['cell_type'] for rid in selected}
        if missing_kinds:
            raise ValueError(f'No proofread neurons for cell types: {sorted(missing_kinds)}')
    if not selected:
        raise ValueError('Neuron group must not be empty')
    return sorted(selected, key=int)


def connectivity(ids: list[str], edges: Any, inputs: list[str], outputs: list[str]) -> dict[str, Any]:
    index = {rid: i for i, rid in enumerate(ids)}
    forward: list[list[int]] = [[] for _ in ids]
    reverse: list[list[int]] = [[] for _ in ids]
    for pre, post in edges.T:
        forward[int(pre)].append(int(post))
        reverse[int(post)].append(int(pre))

    def traverse(starts: list[str], adjacency: list[list[int]]) -> dict[int, int | None]:
        parents: dict[int, int | None] = {index[rid]: None for rid in starts}
        queue = deque(parents)
        while queue:
            current = queue.popleft()
            for nxt in adjacency[current]:
                if nxt not in parents:
                    parents[nxt] = current
                    queue.append(nxt)
        return parents

    parents = traverse(inputs, forward)
    backwards = traverse(outputs, reverse)
    reached_outputs = [rid for rid in outputs if index[rid] in parents]
    reached_inputs = [rid for rid in inputs if index[rid] in backwards]
    path = []
    if reached_outputs:
        node: int | None = index[reached_outputs[0]]
        while node is not None:
            path.append(ids[node])
            node = parents[node]
        path.reverse()
    return {'input_reach_fraction': len(reached_inputs) / len(inputs),
            'output_reach_fraction': len(reached_outputs) / len(outputs),
            'inputs_reaching_output': reached_inputs, 'outputs_reached': reached_outputs,
            'example_path': path, 'definition': 'directed paths from any input to any output'}


def aggregate(connection_path: Path, ids: list[str], selected: list[str], config: dict[str, Any],
              work: Path) -> tuple[Any, Any, dict[str, int]]:
    import duckdb
    import pyarrow as pa
    import pyarrow.ipc as ipc

    columns = ['pre_pt_root_id', 'post_pt_root_id', 'syn_count']
    with pa.memory_map(str(connection_path), 'r') as source, duckdb.connect(str(work / 'aggregate.db')) as db:
        db.execute("SET memory_limit='512MB'")
        db.execute('SET threads=1')
        db.execute('SET preserve_insertion_order=false')
        reader = ipc.open_file(source)
        if not set(columns + ['neuropil']) <= set(reader.schema.names):
            raise ValueError('Connections require pre_pt_root_id, post_pt_root_id, syn_count, neuropil')
        if any(not pa.types.is_integer(reader.schema.field(c).type) for c in columns):
            raise ValueError('Connection IDs and syn_count must have integer types')
        schema = pa.schema([reader.schema.field(c) for c in columns])
        batches = pa.RecordBatchReader.from_batches(schema, (
            reader.get_batch(i).select(columns) for i in range(reader.num_record_batches)))
        db.register('batches', batches)
        db.execute('CREATE TABLE raw AS SELECT pre_pt_root_id AS pre, post_pt_root_id AS post, syn_count AS n FROM batches')
        db.register('roots', pa.table({'id': pa.array([int(i) for i in ids], type=pa.uint64())}))
        invalid = db.execute('''SELECT count(*) FROM raw WHERE pre IS NULL OR post IS NULL OR n IS NULL OR n <= 0
            OR pre NOT IN (SELECT id FROM roots) OR post NOT IN (SELECT id FROM roots)''').fetchone()
        if invalid and invalid[0]:
            raise ValueError('Connections contain null/invalid counts or non-proofread endpoints')
        db.execute('CREATE TABLE pairs AS SELECT pre, post, sum(n)::BIGINT AS n FROM raw GROUP BY pre, post')
        db.register('selected', pa.table({'id': pa.array([int(i) for i in selected], type=pa.uint64()),
                                        'idx': list(range(len(selected)))}))
        db.execute('CREATE TABLE eligible AS SELECT * FROM pairs WHERE n >= ?', [config['min_synapses']])
        if config['self_loops'] == 'drop':
            db.execute('DELETE FROM eligible WHERE pre = post')
        db.execute('''CREATE TABLE used AS SELECT s.idx AS pre, t.idx AS post, p.n
            FROM eligible p JOIN selected s ON p.pre=s.id JOIN selected t ON p.post=t.id''')
        counts = {'source_neurons': len(ids), 'used_neurons': len(selected)}
        for name, table in [('source_region', 'raw'), ('source_pair', 'pairs'), ('eligible', 'eligible'), ('used', 'used')]:
            row = db.execute(f'SELECT count(*), coalesce(sum(n), 0) FROM {table}').fetchone()
            assert row is not None
            counts[f'{name}_edges'] = int(row[0])
            counts[f'{name}_synapses'] = int(row[1])
        counts['source_synapses'] = counts.pop('source_region_synapses')
        counts['excluded_neurons'] = len(ids) - len(selected)
        counts['excluded_pair_edges'] = counts['source_pair_edges'] - counts['used_edges']
        counts['excluded_synapses'] = counts['source_synapses'] - counts['used_synapses']
        rows = db.execute('SELECT pre, post, n FROM used ORDER BY pre, post').fetchnumpy()
        edges = np.vstack((rows['pre'], rows['post'])).astype(np.int64)
        return edges, np.asarray(rows['n'], dtype=np.int64), counts


def prepare_graph(cache: Path, source_manifest: Path, config_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if output.exists():
        raise PreparationError('output_exists', f'Output already exists: {output}; use inspect-graph or a new path')
    sources = read_sources(source_manifest)
    paths = {}
    for role, spec in sources['files'].items():
        paths[role] = cache / spec['name']
        check_file(paths[role], spec['sha256'])
    config = configuration(config_path)
    ids, annotations = read_neurons(paths['neurons'], paths['annotations'])
    selected = select(config['selection'], ids, annotations)
    inputs = select(config['inputs'], ids, annotations)
    outputs = select(config['outputs'], ids, annotations)
    if not set(inputs + outputs) <= set(selected) or set(inputs) & set(outputs):
        raise ValueError('Input/output groups must be disjoint subsets of selected neurons')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.graph-', dir=output.parent) as temporary:
        work = Path(temporary)
        edges, counts, stats = aggregate(paths['connections'], ids, selected, config, work)
        reach = connectivity(selected, edges, inputs, outputs)
        if not reach['example_path']:
            raise PreparationError('graph_disconnected', 'No directed input-to-output path; change selection or threshold')
        artifact = work / 'artifact'
        artifact.mkdir()
        input_set, output_set = set(inputs), set(outputs)
        nodes = [{**annotations.get(rid, {'root_id': rid, 'cell_type': '', 'super_class': ''}),
                  'index': i, 'input': rid in input_set, 'output': rid in output_set}
                 for i, rid in enumerate(selected)]
        write_json(artifact / 'nodes.json', nodes)
        write_json(artifact / 'config.json', config)
        write_json(artifact / 'sources.json', sources)
        np.save(artifact / 'edge_index.npy', edges, allow_pickle=False)
        np.save(artifact / 'syn_count.npy', counts, allow_pickle=False)
        np.save(artifact / 'weight.npy', counts.astype(np.float64), allow_pickle=False)
        report = {'counts': stats, 'connectivity': reach,
                  'coverage': {'neuron_fraction': len(selected) / len(ids),
                               'synapse_fraction': stats['used_synapses'] / stats['source_synapses'],
                               'annotation_neurons': len(annotations),
                               'proofread_without_annotation': len(set(ids) - annotations.keys()),
                               'annotation_outside_proofread': len(annotations.keys() - set(ids)),
                               'selection': 'induced subgraph of selected proofread neurons', 'contracted': False},
                  'assumptions': {'direction': 'presynaptic to postsynaptic',
                                  'threshold': 'sum all neuropils per directed neuron pair, then filter',
                                  'normalization': 'none; weight = synapse count as float64',
                                  'sign': 'nonnegative structural weights; neurotransmitter does not establish sign',
                                  'self_loops': config['self_loops'], 'seed': config['seed'],
                                  'randomness': 'none; ascending numeric IDs and lexicographic edge indices'},
                  'tools': {name: version(name) for name in ('flycade', 'numpy', 'pyarrow', 'duckdb')},
                  'python': platform.python_version(),
                  'implementation_sha256': digest(Path(__file__))}
        write_json(artifact / 'report.json', report)
        hashes = {p.name: digest(p) for p in sorted(artifact.iterdir())}
        graph_hash = graph_identity(hashes)
        manifest = {'format_version': 1, 'graph_sha256': graph_hash, 'artifacts': hashes,
                    'config_sha256': hashes['config.json'], 'source_manifest_sha256': hashes['sources.json']}
        measurement = {'preprocessing_seconds': time.perf_counter() - started,
                       'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       'measurement_scope': 'Linux CLI process peak RSS; preparation includes source hashing, excludes download',
                       'duckdb_memory_limit': '512MB', 'threads': 1}
        write_json(artifact / 'measurement.json', measurement)
        manifest['measurement_sha256'] = digest(artifact / 'measurement.json')
        write_json(artifact / 'manifest.json', manifest)
        artifact.rename(output)
    return {'output': str(output), **manifest, 'counts': stats, 'connectivity': reach, 'measurement': measurement}


def inspect_graph(output: Path, cache: Path | None = None) -> dict[str, Any]:
    manifest_path = output / 'manifest.json'
    if not manifest_path.is_file():
        raise PreparationError('artifact_missing', f'Missing artifact: {manifest_path}')
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    required = {'nodes.json', 'config.json', 'sources.json', 'edge_index.npy', 'syn_count.npy', 'weight.npy', 'report.json'}
    if not isinstance(manifest, dict):
        raise PreparationError('artifact_corrupt', 'Graph manifest must be an object')
    if manifest.get('format_version') != 1 or set(manifest.get('artifacts', {})) != required:
        raise PreparationError('artifact_corrupt', 'Invalid graph manifest version or artifact inventory')
    for name, expected in manifest['artifacts'].items():
        check_file(output / name, expected)
    check_file(output / 'measurement.json', manifest['measurement_sha256'])
    hashes = manifest['artifacts']
    graph_hash = graph_identity(hashes)
    if (graph_hash != manifest['graph_sha256'] or hashes['config.json'] != manifest['config_sha256']
            or hashes['sources.json'] != manifest['source_manifest_sha256']):
        raise PreparationError('artifact_corrupt', 'Inconsistent graph/config/source identity')
    if cache is not None:
        sources = read_sources(output / 'sources.json')
        for spec in sources['files'].values():
            check_file(cache / spec['name'], spec['sha256'])
    return {'output': str(output), 'verified': True, 'sources_verified': cache is not None,
            **manifest, 'report': json.loads((output / 'report.json').read_text())}


def fetch_graph_data(cache: Path, source_manifest: Path) -> dict[str, Any]:
    """Download fixed, hash-pinned files; a verified cache never needs a network."""
    import shutil
    import urllib.error
    import urllib.request

    sources = read_sources(source_manifest)
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for role, spec in sources['files'].items():
        target = cache / spec['name']
        if target.exists():
            check_file(target, spec['sha256'])
            status = 'cached'
        else:
            with tempfile.TemporaryDirectory(prefix='.download-', dir=cache) as temporary:
                partial = Path(temporary) / spec['name']
                try:
                    with urllib.request.urlopen(spec['url'], timeout=120) as response, partial.open('wb') as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                except (OSError, urllib.error.URLError) as exc:
                    raise PreparationError('download_failed', f'{spec["url"]}: {exc}') from exc
                check_file(partial, spec['sha256'])
                partial.replace(target)
            status = 'downloaded'
        result[role] = {'path': str(target), 'sha256': spec['sha256'], 'status': status}
    return {'dataset': sources['dataset'], 'release': sources['release'], 'files': result}
