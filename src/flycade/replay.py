"""Bounded evaluation observations tied to the first episode's video clock."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from flycade.checkpoint import atomic_json, sync_directory
from flycade.game import ACTIONS, Pixels
from flycade.graph import digest
from flycade.observation import policy_view, png_uri


class EvaluationReplay:
    """Observe the action forward only; no new forward, random draw or interpolation."""
    def __init__(self, run: Path, output: Path, report: dict[str, Any], seconds: int, device: str):
        self.output = output
        self.next_frame = 0
        self.limit_frames = seconds * 60
        self.last_live = 0.
        self.error: str | None = None
        self.activity: list[list[float]] = []
        self.samples: list[dict[str, Any]] = []
        self.byte_count = 0
        self.maximum_bytes = 16 * 2**20
        graph = policy_view(run)
        self.indices = torch.tensor([node['index'] for node in graph['nodes']], device=device, dtype=torch.long)
        self.document: dict[str, Any] = {
            'format_version': 1, 'run_id': report['run_id'], 'evaluation_id': report['evaluation_id'],
            'snapshot_id': report['snapshot_id'], 'protocol_id': report['protocol_id'],
            'model_sha256': report['model_sha256'], 'snapshot_updates': report['snapshot_updates'],
            'episode': 0, 'graph': graph, 'interval_emulator_frames': 20,
            'maximum_samples': seconds * 3, 'maximum_uncompressed_sample_bytes': self.maximum_bytes,
            'video_file': None, 'samples': self.samples,
            'time_mapping': 'video_time_seconds = emulator frames before action / 60; video starts at first post-step frame; 12 fps video quantization plus <=1 emulator frame offset',
            'selection_rule': 'last sample at or before video time, valid for <20/60 seconds; no interpolation; missing intervals have no data',
            'status': 'recording'}

    def due(self, episode: int, frames: int) -> bool:
        return self.error is None and episode == 0 and self.next_frame <= frames < self.limit_frames

    def capture(self, state: Tensor) -> None:
        self.activity = state.detach()[0, self.indices].cpu().tolist()

    def append(self, raw: Pixels, pixels: Pixels, probabilities: list[float], action: int,
               frames: int, step: int, observed: float, seed: int, mode: str) -> None:
        self.next_frame = frames + 20
        try:
            sample = {'run_id': self.document['run_id'], 'session_id': self.document['evaluation_id'],
                'snapshot_id': self.document['snapshot_id'], 'stream': 'evaluation', 'worker': 0,
                'episode': 0, 'seed': seed, 'episode_step': step, 'step': step,
                'policy_version': self.document['snapshot_updates'], 'observed_unix': observed,
                'sample_id': f"{self.document['evaluation_id']}:{len(self.samples)+1}",
                'sequence': len(self.samples)+1, 'emulator_frames': frames, 'video_time_seconds': frames / 60,
                'raw': png_uri(raw), 'pixels': [png_uri(frame) for frame in pixels],
                'pixel_shape': list(pixels.shape), 'observation_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                'activity': self.activity, 'probabilities': probabilities, 'action': action,
                'argmax': int(np.argmax(probabilities)), 'buttons': list(ACTIONS[action]),
                'actions': [list(buttons) for buttons in ACTIONS], 'selection': mode}
            size = len(json.dumps(sample, allow_nan=False).encode())
            if self.byte_count + size > self.maximum_bytes or len(self.samples) >= self.document['maximum_samples']:
                raise ValueError('Bounded replay sample budget exceeded')
            self.samples.append(sample)
            self.byte_count += size
            if time.monotonic() - self.last_live >= 1/3:
                self.publish_live('running')
        except (OSError, ValueError) as exc:
            self.error = str(exc)

    def publish_live(self, status: str) -> None:
        self.last_live = time.monotonic()
        atomic_json(self.output / 'live.json', {**self.document, 'samples': [],
            'sample': self.samples[-1] if self.samples else None, 'status': status,
            'updated_unix': time.time(), 'evaluator_pid': os.getpid(), 'error': self.error})

    def close(self, videos: list[dict[str, Any]]) -> dict[str, Any]:
        path = self.output / 'observations.json.gz'
        try:
            self.document.update(video_file=videos[0]['file'] if videos else None,
                                 status='failed' if self.error else 'complete', error=self.error)
            encoded = gzip.compress(json.dumps(self.document, allow_nan=False).encode(), compresslevel=9, mtime=0)
            with tempfile.NamedTemporaryFile(dir=self.output, suffix='.partial', delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
                    os.replace(temporary, path)
                    sync_directory(self.output)
                finally:
                    temporary.unlink(missing_ok=True)
            self.publish_live('completed' if not self.error else 'failed')
            return {'status': self.document['status'], 'file': path.name, 'format_version': 1,
                    'sha256': digest(path), 'bytes': path.stat().st_size, 'samples': len(self.samples),
                    'interval_emulator_frames': 20, 'maximum_samples': self.document['maximum_samples'],
                    'maximum_uncompressed_sample_bytes': self.maximum_bytes, 'error': self.error}
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            return {'status': 'failed', 'file': None, 'error': self.error, 'samples': len(self.samples)}
