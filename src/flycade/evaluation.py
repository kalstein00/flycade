"""Sequential fixed-policy evaluation in a dedicated CLI process."""
import hashlib
import json
import random
import resource
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flycade.checkpoint import atomic_json
from flycade.errors import PreparationError
from flycade.game import ACTIONS, GameConfig, GameEnv
from flycade.graph import digest
from flycade.nes import NesEmulator
from flycade.recording import RecordingEmulator
from flycade.snapshot import load_snapshot
from flycade.training_fixture import FixtureEmulator


@dataclass(frozen=True)
class EvaluationConfig:
    seeds: tuple[int, ...] = (11, 22, 33)
    mode: str = 'stochastic'
    max_frames: int = 1800
    video_seconds: int = 15

    def __post_init__(self) -> None:
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds)):
            raise ValueError('Evaluation seeds must be a nonempty unique list of uint32 integers')
        if self.mode not in ('stochastic', 'deterministic'):
            raise ValueError('Evaluation mode must be stochastic or deterministic')
        if self.mode == 'deterministic' and len(self.seeds) != 1:
            raise ValueError('Deterministic spectating uses one representative episode, not seed diversity')
        if type(self.max_frames) is not int or self.max_frames <= 0:
            raise ValueError('Evaluation max_frames must be a positive integer')
        if type(self.video_seconds) is not int or not 1 <= self.video_seconds <= 60:
            raise ValueError('Evaluation video_seconds must be an integer in [1, 60]')


def protocol_identity(protocol: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def make_protocol(manifest: dict[str, Any], config: EvaluationConfig) -> dict[str, Any]:
    game = replace(GameConfig(**manifest['game']), max_frames=min(manifest['game']['max_frames'], config.max_frames))
    return {'format_version': 1, 'evaluation_code_sha256': digest(Path(__file__)), 'game': asdict(game), 'registration': manifest['registration'],
            'fixture': manifest['fixture'], 'actions': [list(action) for action in ACTIONS],
            'preprocessing': 'Pillow bilinear; oldest-to-newest frame stack; pixels / 255',
            'normalization': 'no running observation or reward statistics',
            'evaluation_count': len(config.seeds), 'seeds': list(config.seeds), 'mode': config.mode,
            'action_selection': 'torch.multinomial on CPU probabilities; local CPU Generator seeded per episode' if config.mode == 'stochastic'
                                else 'argmax probability; lowest action ID breaks ties',
            'video_seconds': config.video_seconds, 'distance_unit': 'pixels', 'reward_unit': 'raw game reward',
            'comparison_rule': 'same protocol only; report distributions, not a learning-success claim',
            'limitations': 'Small fixed seed sample; changing seeds does not diversify deterministic NES play.'}


def create_protocol(run: Path, output: Path, config: EvaluationConfig) -> dict[str, Any]:
    if output.exists():
        raise PreparationError('output_exists', f'Refusing to overwrite protocol {output}')
    manifest = json.loads((run / 'run.json').read_text())
    protocol = make_protocol(manifest, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(protocol, handle, indent=2)
        handle.write('\n')
    return {'protocol_id': protocol_identity(protocol), 'output': str(output.resolve()), 'protocol': protocol}


def validate_protocol(protocol: dict[str, Any], manifest: dict[str, Any]) -> None:
    config = EvaluationConfig(tuple(protocol['seeds']), protocol['mode'],
                              protocol['game']['max_frames'], protocol['video_seconds'])
    if protocol != make_protocol(manifest, config):
        raise PreparationError('protocol_incompatible', 'Protocol game/state/preprocessing/actions or format differs')


def evaluate(run: Path, selection: str, protocol_path: Path, home: Path, device: str,
             realtime: bool = False, training_paused: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(1)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    policy, snapshot, manifest = load_snapshot(run, selection, home, device)
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol, manifest)
    if realtime and protocol['mode'] != 'deterministic':
        raise ValueError('--realtime is for deterministic spectating only')
    protocol_id = protocol_identity(protocol)
    protocols = run / 'protocols'
    protocols.mkdir(exist_ok=True)
    persisted = protocols / f'{protocol_id}.json'
    if persisted.exists():
        if json.loads(persisted.read_text()) != protocol:
            raise PreparationError('protocol_incompatible', 'Stored protocol changed')
    else:
        atomic_json(persisted, protocol)
        persisted.chmod(0o444)
    evaluation_id = str(uuid.uuid4())
    output = run / 'evaluations' / evaluation_id
    output.mkdir(parents=True)
    atomic_json(output / 'protocol.json', protocol)
    report: dict[str, Any] = {'evaluation_id': evaluation_id, 'run_id': manifest['run_id'],
        'snapshot_id': snapshot['snapshot_id'], 'model_sha256': snapshot['weights_sha256'],
        'snapshot_updates': snapshot['updates'], 'protocol_id': protocol_id,
        'created_unix': time.time(), 'mode': 'stochastic_evaluation' if protocol['mode'] == 'stochastic'
                                          else 'deterministic_spectating',
        'device': device, 'status': 'running', 'training_paused': training_paused,
        'training_pause_reason': 'initial evaluation before first rollout' if training_paused else None,
        'execution': 'one separate evaluation process; episodes sequential; no optimizer',
        'evidence': 'synthetic CPU fixture; not NES/GPU acceptance' if manifest['fixture'] else f'real NES on {device}',
        'distance_unit': 'pixels', 'reward_unit': 'raw game reward', 'episodes': [], 'videos': [],
        'environment_closed': False, 'workers_alive': 0, 'realtime_requested': realtime,
        'normalization': snapshot['normalization'], 'limitations': protocol['limitations']}
    atomic_json(output / 'report.json', report)
    env: GameEnv | None = None
    recorder: RecordingEmulator | None = None
    try:
        video_session = {'session_id': evaluation_id, 'run_id': manifest['run_id'],
            'created_unix': report['created_unix'], 'start_updates': snapshot['updates'],
            'evaluation_id': evaluation_id, 'snapshot_id': snapshot['snapshot_id'],
            'model_sha256': snapshot['weights_sha256'], 'protocol_id': protocol_id,
            'mode': report['mode'], 'episode': 0, 'seed': protocol['seeds'][0]}
        recorder = RecordingEmulator(FixtureEmulator() if manifest['fixture'] else NesEmulator(home),
                                     output, video_session, max_video_frames=protocol['video_seconds'] * 12)
        env = GameEnv(recorder, GameConfig(**protocol['game']))
        with (output / 'transitions.jsonl').open('x') as trace:
            for episode, seed in enumerate(protocol['seeds']):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                sampler = torch.Generator(device='cpu').manual_seed(seed)
                env.action_space.seed(seed)
                obs, _ = env.reset(seed=seed)
                reward_sum, transitions = 0., 0
                episode_started = time.perf_counter()
                with torch.inference_mode():
                    while True:
                        distribution, _ = policy(torch.from_numpy(obs).unsqueeze(0).to(device))
                        action = int(torch.multinomial(distribution.probs.cpu(), 1, generator=sampler).item() if protocol['mode'] == 'stochastic'
                                     else distribution.probs.argmax(dim=-1).item())
                        obs, reward, terminated, truncated, info = env.step(action)
                        reward_sum += reward
                        transitions += 1
                        if not np.isfinite(reward_sum):
                            raise PreparationError('nonfinite_evaluation', 'Nonfinite evaluation reward')
                        if info['reason'] in ('position_discontinuity', 'unexpected_game_state', 'backend_end_unclassified'):
                            raise PreparationError('game_contract_violation', f"Invalid evaluation transition: {info['reason']}")
                        trace.write(json.dumps({'episode': episode, 'seed': seed, 'transition': transitions,
                            'action': action, 'reward': reward, 'terminated': terminated, 'truncated': truncated,
                            'max_progress': info['max_progress'], 'executed_frames': info['executed_frames'],
                            'reason': info['reason']}, allow_nan=False) + '\n')
                        if realtime:
                            delay = info['frames'] / 60 - (time.perf_counter() - episode_started)
                            if delay > 0:
                                time.sleep(delay)
                        if terminated or truncated:
                            break
                report['episodes'].append({'episode': episode, 'seed': seed, 'distance': info['max_progress'],
                    'reward': reward_sum, 'reason': info['reason'], 'emulator_frames': info['frames'],
                    'transitions': transitions, 'terminated': terminated, 'truncated': truncated})
                # A short representative clip of episode 0; never splice multiple seeds into one video.
                if episode == 0:
                    recorder.stop_recording()
        distances = [episode['distance'] for episode in report['episodes']]
        reasons: dict[str, int] = {}
        for episode_result in report['episodes']:
            reason = episode_result['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
        report.update(evaluation_count=len(distances), completion_count=reasons.get('completion', 0),
            completion_rate=reasons.get('completion', 0) / len(distances), termination_counts=reasons,
            distances=distances, mean_distance=float(np.mean(distances)), median_distance=float(np.median(distances)),
            min_distance=min(distances), max_distance=max(distances),
            rewards=[episode['reward'] for episode in report['episodes']])
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=str(exc) or type(exc).__name__)
        raise
    finally:
        if env is not None:
            env.close()
            report['environment_closed'] = env.closed
        if recorder is not None:
            report['recording_error'] = recorder.error
            if recorder.error:
                report.update(status='failed', error=recorder.error)
        report['videos'] = [json.loads(path.read_text()) for path in sorted((output / 'videos').glob('*.json'))]
        report['finished_unix'] = time.time()
        report['wall_seconds'] = time.perf_counter() - started
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report['encoder_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
        atomic_json(output / 'report.json', report)
    if report['status'] != 'completed':
        raise PreparationError('evaluation_recording_failed', str(report.get('error')))
    return {'output': str(output.resolve()), **report}


def compare_evaluations(run: Path, identifiers: list[str]) -> dict[str, Any]:
    if len(identifiers) < 2:
        raise ValueError('Choose at least two evaluations to compare')
    rows = [json.loads((run / 'evaluations' / str(uuid.UUID(identifier)) / 'report.json').read_text())
            for identifier in identifiers]
    manifest = json.loads((run / 'run.json').read_text())
    for row in rows:
        if row['status'] != 'completed' or row['run_id'] != manifest['run_id']:
            raise PreparationError('evaluation_incompatible', 'Only completed evaluations of this Run can be compared')
        protocol = json.loads((run / 'evaluations' / row['evaluation_id'] / 'protocol.json').read_text())
        if protocol_identity(protocol) != row['protocol_id']:
            raise PreparationError('protocol_incompatible', 'Evaluation protocol artifact changed')
    if len({row['protocol_id'] for row in rows}) != 1:
        raise PreparationError('protocol_incompatible', 'Different protocols cannot be compared as identical conditions')
    return {'run_id': manifest['run_id'], 'protocol_id': rows[0]['protocol_id'], 'results': rows,
            'note': 'Small-sample fixed-protocol comparison; not proof of improved gameplay.'}
