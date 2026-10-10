"""Read-only rollout observation. One pending bundle, one on disk; no consumer backpressure."""
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import resource
import threading
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from flycade.game import ACTIONS, Pixels


def graph_view(graph: Path) -> dict[str, Any]:
    """Deterministic bounded structural layout, cached for the lifetime of a Run."""
    nodes = json.loads((graph / 'nodes.json').read_text())
    edges = np.load(graph / 'edge_index.npy', allow_pickle=False).T.tolist()
    report = json.loads((graph / 'report.json').read_text())
    by_id = {node['root_id']: node['index'] for node in nodes}
    path = report['connectivity']['example_path']
    # Keep both ends of long paths while reserving room for input/output groups.
    selected = {by_id[root] for root in path[:32] + path[-32:]}
    for group in ('input', 'output'):
        selected.update(node['index'] for node in [n for n in nodes if n[group]][:12])
    # Grow along real connections, then fill with stable node IDs if disconnected.
    for _ in range(3):
        for source, target in edges:
            if len(selected) >= 128:
                break
            if source in selected or target in selected:
                selected.update((source, target))
    for node in nodes:
        if len(selected) >= min(128, len(nodes)):
            break
        selected.add(node['index'])
    visible = [dict(nodes[index]) for index in sorted(selected)]
    for group in ('input', 'internal', 'output'):
        members = [n for n in visible if ('input' if n['input'] else 'output' if n['output'] else 'internal') == group]
        for index, node in enumerate(members):
            # Four narrow lanes per group; coordinates never depend on activity.
            node.update(group=group, x={'input': 35, 'internal': 145, 'output': 255}[group] + index % 4 * 14,
                        y=32 + (index // 4 + 1) * 292 / ((len(members) + 3) // 4 + 1))
    links = [[a, b] for a, b in edges if a in selected and b in selected][:512]
    return {'sha256': json.loads((graph / 'manifest.json').read_text())['graph_sha256'],
            'version': json.loads((graph / 'sources.json').read_text()).get('release', 'fixture; see sources.json'),
            'nodes': visible, 'edges': links, 'used_nodes': len(nodes), 'used_edges': len(edges),
            'excluded_nodes': len(nodes) - len(visible), 'excluded_edges': len(edges) - len(links),
            'selection': 'first/last 32 of example input-output path + first 12 input/output nodes; 3 undirected neighbor passes in edge order; fill by index; max 128 nodes, first 512 induced directed edges',
            'layout': 'fixed structural diagram; not anatomical coordinates',
            'activity_rule': 'signed arithmetic mean of final tanh state vector; range [-1, 1]'}


def policy_view(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / 'run.json').read_text())
    if manifest['model'].get('kind', 'connectome') == 'cnn':
        return {'applicable': False, 'model_kind': 'cnn', 'nodes': [], 'edges': [],
                'used_nodes': 0, 'used_edges': 0, 'reason': 'CNN has no connectome circuit'}
    return {**graph_view(run / 'graph'), 'applicable': True, 'model_kind': 'connectome'}


def png_uri(pixels: Pixels) -> str:
    buffer = io.BytesIO()
    Image.fromarray(pixels[:, :, 0] if pixels.shape[-1] == 1 else pixels).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def disable_observation(output: Path, session: dict[str, Any]) -> None:
    """Invalidate any previous session when observation is disabled on resume."""
    path = output / 'live' / 'latest.json'
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    envelope = {'run_id': session['run_id'], 'session_id': session['session_id'],
                'generation': previous.get('generation', 0) + 1, 'sequence': 0,
                'updated_unix': time.time(), 'status': 'disabled', 'sample': None}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(envelope))
    temporary.replace(path)


class RolloutObserver:
    """Capture only the action forward. Serialization/disk I/O run on a bounded worker."""

    def __init__(self, output: Path, session: dict[str, Any], hz: int, device: str):
        self.directory = output / 'live'
        self.directory.mkdir(exist_ok=True)
        self.path = self.directory / 'latest.json'
        previous = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.generation = previous.get('generation', 0) + 1
        self.session = dict(session)
        self.graph = policy_view(output)
        self.indices = torch.tensor([n['index'] for n in self.graph['nodes']], device=device, dtype=torch.long)
        self.hz = hz
        self.sequence = 0
        self.first_capture = 0.
        self.next_due = 0.
        self.activity: list[list[float]] = []
        self.pending: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.stats: dict[str, Any] = {'requested_hz': hz, 'queue_capacity': 1, 'history_capacity': 0,
                                     'captured': 0, 'published': 0, 'dropped': 0, 'selected_forward_seconds': 0., 'actual_hz': 0.,
                                     'encode_seconds': 0., 'error': None}
        self.envelope: dict[str, Any] = {'format_version': 1, 'run_id': session['run_id'],
            'session_id': session['session_id'], 'generation': self.generation, 'sequence': 0,
            'trainer_pid': os.getpid(), 'status': 'running', 'sample': None, 'graph': self.graph,
            'observer': self.stats, 'updated_unix': time.time()}
        self._write()
        self.worker = threading.Thread(target=self._work, name='rollout-observer', daemon=True)
        self.worker.start()

    def due(self) -> bool:
        now = time.monotonic()
        if self.stats['error'] or now < self.next_due:
            return False
        self.next_due = now + 1 / self.hz
        return True

    def capture(self, state: Tensor) -> None:
        """Called by the same forward that supplies the sampled action, before env.step."""
        self.activity = state.detach()[0, self.indices].cpu().tolist()

    def submit(self, raw: Pixels, pixels: Pixels, probabilities: list[float], action: int,
               report: dict[str, Any], update: int, observed: float, episode_step: int,
               transition: dict[str, Any], device: str, capture_seconds: float) -> None:
        self.sequence += 1
        self.stats['captured'] += 1
        self.stats['selected_forward_seconds'] += capture_seconds
        if not self.first_capture:
            self.first_capture = time.monotonic()
        self.stats['actual_hz'] = (self.sequence - 1) / max(time.monotonic() - self.first_capture, 1 / self.hz)
        sample = {'run_id': self.session['run_id'], 'session_id': self.session['session_id'],
            'stream': 'training', 'worker': 0, 'episode': report['episodes'],
            'step': report['transitions'], 'episode_step': episode_step, 'policy_version': update,
            'observed_unix': observed, 'sample_id': f"{self.session['session_id']}:{self.sequence}",
            'raw': raw, 'pixels': pixels, 'observation_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
            'activity': self.activity, 'probabilities': probabilities, 'action': action,
            'argmax': int(np.argmax(probabilities)), 'buttons': list(ACTIONS[action]),
            'actions': [list(a) for a in ACTIONS], 'selection': 'stochastic categorical',
            'transition': transition, 'stage': '1-1',
            'metrics': {key: copy.deepcopy(report.get(key)) for key in ('updates', 'optimizer_steps', 'transitions',
                'emulator_frames', 'episodes', 'max_progress', 'reward_sum', 'training_seconds',
                'transitions_per_second', 'loss', 'entropy', 'episode_outcomes', 'checkpoint_id')},
            'resources': {'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          'cuda_allocated_bytes': torch.cuda.memory_allocated() if device == 'cuda' else 0},
            'metrics_unix': time.time()}
        # Transfer owned copies; the rollout never waits for the encoder or a browser.
        try:
            self.pending.get_nowait()
            self.stats['dropped'] += 1
        except queue.Empty:
            pass
        self.pending.put_nowait({'sequence': self.sequence, 'sample': sample})

    def _write(self) -> None:
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.envelope, allow_nan=False))
        temporary.replace(self.path)

    def _work(self) -> None:
        while not self.closed.is_set() or not self.pending.empty():
            try:
                bundle = self.pending.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                started = time.perf_counter()
                sample = bundle['sample']
                sample['raw'] = png_uri(sample['raw'])
                sample['pixel_shape'] = list(sample['pixels'].shape)
                sample['pixels'] = [png_uri(frame) for frame in sample['pixels']]
                self.stats['encode_seconds'] += time.perf_counter() - started
                self.stats['published'] += 1
                self.envelope.update(bundle, updated_unix=time.time())
                self._write()
            except Exception as exc:
                self.stats['error'] = str(exc)

    def close(self, report: dict[str, Any]) -> None:
        self.closed.set()
        self.worker.join(timeout=2)
        if self.worker.is_alive():
            self.stats['error'] = 'Observer writer did not stop within 2 seconds'
            return
        self.stats['actual_hz'] = max(0, self.sequence - 1) / max(time.monotonic() - self.first_capture, 1 / self.hz)
        self.envelope.update(status=report['status'], updated_unix=time.time(), final_metrics=copy.deepcopy(report))
        try:
            self._write()
        except OSError as exc:
            self.stats['error'] = str(exc)
