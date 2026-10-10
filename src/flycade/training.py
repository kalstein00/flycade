"""Bounded single-environment PPO experiment and auditable new Run artifacts."""
import hashlib
import importlib.metadata
import json
import platform
import os
import random
import resource
import signal
import sys
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from torch import Tensor

from flycade.checkpoint import (FORMAT, atomic_json, capture_rng, load_checkpoint, restore_rng,
                                run_lock, save_checkpoint, sync_directory)
from flycade.errors import PreparationError
from flycade.control import SaveControl
from flycade.game import GameConfig, GameEnv, Pixels
from flycade.graph import digest, inspect_graph, write_json
from flycade.nes import NesEmulator
from flycade.policy import ConnectomePolicy, Policy, policy_for_kind
from flycade.recording import RecordingEmulator
from flycade.rom import inspect_registration
from flycade.training_fixture import FixtureEmulator
from flycade.snapshot import publish_snapshot
from flycade.observation import RolloutObserver, disable_observation
from flycade.observation_config import validate_observation_rate


@dataclass(frozen=True)
class TrainingConfig:
    model_kind: str = 'connectome'
    updates: int = 2
    rollout_steps: int = 32
    epochs: int = 2
    state_dim: int = 8
    propagation_steps: int = 2
    seed: int = 7
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    autosave_seconds: float = 600
    evaluation_timeout_seconds: float = 120
    keep_checkpoints: int = 3
    evaluation_every_updates: int = 1000

    def __post_init__(self) -> None:
        if self.model_kind not in ('connectome', 'cnn'):
            raise ValueError('model_kind must be connectome or cnn')
        for name in ('updates', 'rollout_steps', 'epochs', 'state_dim', 'propagation_steps', 'keep_checkpoints'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.evaluation_every_updates) is not int or self.evaluation_every_updates < 0:
            raise ValueError('evaluation_every_updates must be a nonnegative integer')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        for name in ('learning_rate', 'gamma', 'gae_lambda', 'clip_ratio', 'entropy_coefficient',
                     'value_coefficient', 'max_grad_norm'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_ratio < 1:
            raise ValueError('Invalid discount, GAE lambda or PPO clip ratio')
        if not np.isfinite(self.autosave_seconds) or self.autosave_seconds <= 0:
            raise ValueError('autosave_seconds must be finite and positive')
        if not np.isfinite(self.evaluation_timeout_seconds) or self.evaluation_timeout_seconds <= 0:
            raise ValueError('evaluation_timeout_seconds must be finite and positive')
        if self.learning_rate == 0 or self.max_grad_norm == 0:
            raise ValueError('learning_rate and max_grad_norm must be positive')


def log_row(log: TextIO, row: dict[str, Any]) -> None:
    from flycade.storage import append_log, storage_policy
    append_log(log, row, storage_policy(Path(log.name).parent)['log_bytes'])


def pixel_tensor(obs: Pixels, device: str) -> Tensor:
    return torch.from_numpy(obs).unsqueeze(0).to(device)


def graph_influence(policy: ConnectomePolicy, pixels: Tensor) -> dict[str, Any]:
    with torch.no_grad():
        original, _ = policy(pixels)
        removed, _ = policy(pixels, edge_scale=0.)
    delta = float((original.probs - removed.probs).abs().max())
    return {'intervention': 'all edges removed; identical pixels and parameters',
            'input_sha256': hashlib.sha256(pixels.cpu().numpy().tobytes()).hexdigest(),
            'original_probabilities': original.probs.cpu().tolist(),
            'removed_probabilities': removed.probs.cpu().tolist(),
            'max_probability_delta': delta, 'passed': delta > 1e-7}


@dataclass(frozen=True)
class RolloutTransition:
    observation: Pixels
    action: int
    log_probability: float
    value: float
    reward: float
    next_value: float
    ended: bool


def optimize(policy: Policy, optimizer: torch.optim.Optimizer,
             rollout: list[RolloutTransition], config: TrainingConfig, device: str) -> dict[str, Any]:
    # next_values already masks true termination, but bootstraps truncation.
    advantages = [0.] * len(rollout)
    carry = 0.
    for i in reversed(range(len(rollout))):
        transition = rollout[i]
        delta = transition.reward + config.gamma * transition.next_value - transition.value
        carry = delta + config.gamma * config.gae_lambda * (not transition.ended) * carry
        advantages[i] = carry
    advantage = torch.tensor(advantages, device=device)
    returns = advantage + torch.tensor([t.value for t in rollout], device=device)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    pixels = torch.from_numpy(np.stack([t.observation for t in rollout])).to(device)
    action = torch.tensor([t.action for t in rollout], device=device)
    old_logp = torch.tensor([t.log_probability for t in rollout], device=device)
    before = {name: param.detach().clone() for name, param in policy.named_parameters()}
    metrics: dict[str, Any] = {}
    for _ in range(config.epochs):
        distribution, value = policy(pixels)
        logp = distribution.log_prob(action)
        ratio = (logp - old_logp).exp()
        actor_loss = -torch.minimum(ratio * advantage,
            ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantage).mean()
        value_loss = (value - returns).square().mean()
        entropy = distribution.entropy().mean()
        loss = actor_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
        if not torch.isfinite(loss):
            raise PreparationError('nonfinite_training', 'Nonfinite PPO loss; update aborted')
        optimizer.zero_grad()
        loss.backward()
        gradient_norms = {}
        for name, param in policy.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise PreparationError('nonfinite_training', f'Missing or nonfinite gradient: {name}')
            gradient_norms[name] = float(param.grad.norm())
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in policy.parameters()):
            raise PreparationError('nonfinite_training', 'Nonfinite parameters after optimizer step')
        metrics = {'loss': float(loss.detach()), 'policy_loss': float(actor_loss.detach()),
                   'value_loss': float(value_loss.detach()), 'entropy': float(entropy.detach()),
                   'approx_kl': float(((ratio - 1) - (logp - old_logp)).mean().detach()),
                   'clip_fraction': float(((ratio - 1).abs() > config.clip_ratio).float().mean()),
                   'gradient_l2': gradient_norms, 'nonfinite_count': 0}
    metrics['parameter_delta_l2'] = {name: float((param.detach() - before[name]).norm())
                                     for name, param in policy.named_parameters()}
    return metrics


def _train(home: Path, graph: Path, output: Path, config: TrainingConfig,
          game: GameConfig, device: str, fixture: bool = False, stop_after_updates: int | None = None,
           resume_state: dict[str, Any] | None = None,
           initial_evaluation: dict[str, Any] | None = None, observe_hz: int = 3,
          warm_state: dict[str, Any] | None = None) -> dict[str, Any]:
    if fixture and device != 'cpu':
        raise ValueError('Synthetic fixture requires --device cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise PreparationError('cuda_unavailable', 'CUDA requested but unavailable; no silent CPU fallback')
    graph_report = inspect_graph(graph)
    registration = {'evidence': 'synthetic fixture'} if fixture else inspect_registration(home)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    channels = game.frame_stack * (3 if game.color == 'RGB' else 1)
    policy = policy_for_kind(graph, config.model_kind, config.state_dim, config.propagation_steps, channels).to(device)
    if warm_state is not None:
        policy.load_state_dict(warm_state['model'], strict=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    initial_parameters = {name: p.detach().clone() for name, p in policy.named_parameters()}
    if resume_state is None:
        shutil.copytree(graph, output / 'graph')
        # Check the actual copied artifact too: a Run never silently changes graph identity.
        if inspect_graph(output / 'graph')['graph_sha256'] != graph_report['graph_sha256']:
            raise ValueError('Graph changed during Run creation')
        source = Path(__file__).parent
        repo = source.parent.parent
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True, text=True)
        manifest: dict[str, Any] = {'format_version': 2, 'checkpoint_format': FORMAT, 'fixture': fixture, 'run_id': str(uuid.uuid4()), 'session_id': str(uuid.uuid4()),
            'created_unix': time.time(), 'seed': config.seed, 'training': asdict(config),
            'evaluation': initial_evaluation,
            'game': asdict(game), 'registration': registration, 'graph': graph_report,
            'device': device, 'device_name': torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
            'python': platform.python_version(), 'cuda_runtime': torch.version.cuda,
            'versions': {name: importlib.metadata.version(name) for name in
                         ('torch', 'numpy', 'stable-retro', 'gymnasium', 'pillow', 'flycade')},
            'code': {'git_head': git.stdout.strip() if git.returncode == 0 else None,
                     'files': {p.name: digest(p) for p in sorted(source.glob('*.py'))},
                     'uv_lock_sha256': digest(repo / 'uv.lock') if (repo / 'uv.lock').exists() else None},
            'model': {'kind': config.model_kind, 'parameter_count': sum(p.numel() for p in policy.parameters()), 'architecture': 'conv/pool/linear pixel encoder -> COO propagation -> output-only actor/critic',
                      'state_dim': config.state_dim, 'propagation_steps': config.propagation_steps,
                      'fixed': 'topology and incoming-synapse-normalized nonnegative weights',
                      'sign_assumption': 'structural positive weights; not neurotransmitter sign inference',
                      'state': 'zero per observation, repeated input drive, tanh propagation; no temporal recurrence',
                      'mapping': 'learned engineering mapping to input nodes; output nodes flattened for readout',
                      'trainable': list(initial_parameters), 'pretrained_or_teacher': False},
            'algorithm': 'PPO-Clip; GAE; full-rollout batch per epoch; Adam; FP32; no running normalization',
            'worker_model': 'one in-process environment; one bounded ffmpeg encoder, reaped on close'}
        if config.model_kind == 'cnn':
            manifest['model'].update(architecture='Conv16(8/4), Conv32(4/2), adaptive pool3x3, dense64, actor/critic; ReLU',
                fixed='no graph; same pixel/game/action/reward contract', graph_used=False,
                sign_assumption='not applicable', state='feedforward CNN; no temporal recurrence',
                mapping='pixels -> convolutional features -> actor/critic; no connectome mapping')
        else:
            manifest['model']['graph_used'] = True
        if warm_state is not None:
            manifest['lineage'] = warm_state['lineage']
            manifest['model']['pretrained_or_teacher'] = 'local checkpoint warm start; no teacher'
        torch.save(policy.state_dict(), output / 'initial.pt')
        (output / 'initial.pt').chmod(0o444)
        manifest['initial_sha256'] = digest(output / 'initial.pt')
        write_json(output / 'run.json', manifest)
        for artifact in [output / 'run.json', output / 'initial.pt', *(output / 'graph').iterdir()]:
            if artifact.is_file():
                with artifact.open('rb') as handle:
                    os.fsync(handle.fileno())
        sync_directory(output / 'graph')
        sync_directory(output)
        publish_snapshot(output, policy.state_dict(), manifest, 'initial', 0)
        report: dict[str, Any] = {'run_id': manifest['run_id'], 'session_id': manifest['session_id'],
            'evidence': 'synthetic CPU fixture; not NES/GPU acceptance' if fixture else f'real NES on {device}',
            'status': 'running', 'transitions': 0, 'emulator_frames': 0, 'updates': 0,
            'optimizer_steps': 0, 'episodes': 0, 'episode_outcomes': {}, 'reward_sum': 0.,
            'max_progress': 0, 'safe_boundary': False, 'environment_closed': False,
            'workers_alive': 0, 'nonfinite_count': 0, 'initial_sha256': digest(output / 'initial.pt')}
    else:
        manifest = resume_state['manifest']
        policy.load_state_dict(resume_state['model'], strict=True)
        optimizer.load_state_dict(resume_state['optimizer'])
        report = dict(resume_state['progress'])
        report.update(session_id=str(uuid.uuid4()), status='running', environment_closed=False,
                      resumed_from=resume_state['checkpoint_id'])
        report.pop('error', None)
        if '_recovery' in resume_state:
            recovery = {**resume_state['_recovery'], 'state': 'restored', 'to_session_id': report['session_id']}
            report['recovery'] = recovery
            directory = output / 'recoveries'
            directory.mkdir(exist_ok=True)
            atomic_json(directory / f"{recovery['recovery_id']}.json", recovery)
            atomic_json(output / 'recovery-status.json', recovery)
            print(f"Recovered checkpoint {recovery['checkpoint_id']}; rollback {recovery['lost_updates']} updates / {recovery['lost_transitions']} transitions; new session.", file=sys.stderr)

    from flycade.budget import budget_info
    report['budget'] = budget_info(output, manifest)
    if 'lineage' in manifest:
        report['lineage'] = manifest['lineage']
    session = {'run_id': report['run_id'], 'session_id': report['session_id'],
               'resumed_from': report.get('resumed_from'), 'start_updates': report['updates'],
               'created_unix': time.time(), 'reset': 'new episode; learning state preserved'}
    sessions = output / 'sessions'
    sessions.mkdir(exist_ok=True)
    write_json(sessions / f"{report['session_id']}.json", session)
    report.setdefault('next_autosave_seconds', config.autosave_seconds)
    report.setdefault('evaluation_history', [])
    report.setdefault('next_evaluation_update', config.evaluation_every_updates if manifest.get('evaluation') is not None and config.evaluation_every_updates else None)
    control = SaveControl(output, session, report)
    write_json(output / 'report.json', report)
    if resume_state is not None:
        print('Resumed learning state; new episode (interrupted episode is not counted).', file=sys.stderr)
    start_updates = report['updates']
    previous_seconds = report.get('training_seconds', 0.)
    stop_requested = False

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print('Save-and-stop requested; finishing current rollout and optimizer update.', file=sys.stderr)

    previous_signal = signal.signal(signal.SIGINT, request_stop)
    started = time.perf_counter()
    env: GameEnv | None = None
    recording: RecordingEmulator | None = None
    observer: RolloutObserver | None = None
    try:
        if initial_evaluation is not None:
            from flycade.evaluation import EvaluationConfig, create_protocol
            protocol = output / 'initial-evaluation-protocol.json'
            create_protocol(output, protocol, EvaluationConfig(**initial_evaluation))
            from flycade.evaluation_worker import evaluate_snapshot
            evaluation = evaluate_snapshot(output, home, 'initial', protocol,
                config.evaluation_timeout_seconds, report, control, lambda: control.poll(stop_requested))
            report['initial_evaluation_id'] = evaluation['evaluation_id']
            report['evaluation_history'].append(evaluation['evaluation_id'])
            report['evaluation_history'] = report['evaluation_history'][-100:]
        report['training_started_unix'] = time.time()
        learning_started = time.perf_counter()
        recording = RecordingEmulator(FixtureEmulator() if fixture else NesEmulator(home), output,
            {**session, 'created_unix': report['training_started_unix']})
        env = GameEnv(recording, game)
        report['contract'] = env.describe()
        env.action_space.seed(config.seed + 1)
        env.observation_space.seed(config.seed + 2)
        if resume_state is None:
            obs, _ = env.reset(seed=config.seed)
        else:
            # Reconstruct first, restore RNG last. reset may consume the restored environment RNG,
            # but construction and seed initialization must never consume training RNG.
            restore_rng(resume_state, env, device)
            obs, _ = env.reset()
            report['resume_reset'] = {'frames': env.frames, 'max_progress': env.maximum - env.start_position,
                                      'stack_frames_equal': all(np.array_equal(obs[0], f) for f in obs)}
        if resume_state is None:
            np.save(output / 'control-pixels.npy', obs, allow_pickle=False)
        report['graph_influence'] = (graph_influence(policy, pixel_tensor(obs, device)) if isinstance(policy, ConnectomePolicy)
                                     else {'applicable': False, 'reason': 'CNN has no connectome graph'})
        if isinstance(policy, ConnectomePolicy) and not report['graph_influence']['passed']:
            raise PreparationError('graph_no_influence', 'Controlled edge removal did not change policy distribution')
        try:
            if observe_hz:
                observer = RolloutObserver(output, session, observe_hz, device)
            else:
                disable_observation(output, session)
        except (OSError, ValueError) as exc:
            report['observer_error'] = str(exc)
        def persist() -> None:
            assert env is not None
            control.saving()
            if report['training_seconds'] >= report['next_autosave_seconds']:
                report['next_autosave_seconds'] += (int((report['training_seconds'] - report['next_autosave_seconds']) / config.autosave_seconds) + 1) * config.autosave_seconds
            metadata = save_checkpoint(output, {'model': policy.state_dict(),
                'optimizer': optimizer.state_dict(), 'progress': report, 'manifest': manifest,
                **capture_rng(env, device)})
            publish_snapshot(output, policy.state_dict(), manifest, metadata['checkpoint_id'], report['updates'])
            report['checkpoint_id'] = metadata['checkpoint_id']
            report['final_sha256'] = digest(output / 'final.pt')
            control.complete(metadata, report)
            from flycade.storage import cleanup_storage
            report['storage'] = cleanup_storage(output)
            report['pruned_checkpoints'] = report['storage'].get('pruned_checkpoints', [])

        def periodic_evaluation() -> None:
            nonlocal learning_started
            control.poll(stop_requested)
            if stop_requested or (control.pending is not None and control.pending['stop']):
                return
            due = report.get('next_evaluation_update')
            if due is None or report['updates'] < due:
                return
            from flycade.evaluation_worker import evaluate_snapshot
            # Commit before starting another process. A failed evaluator leaves the due
            # position in this full-state checkpoint, so resume retries it.
            control.poll(stop_requested, True)
            persist()
            paused_at = time.perf_counter()
            evaluation = evaluate_snapshot(output, home, report['checkpoint_id'],
                output / 'initial-evaluation-protocol.json', config.evaluation_timeout_seconds,
                report, control, lambda: control.poll(stop_requested))
            paused_seconds = time.perf_counter() - paused_at
            learning_started += paused_seconds
            report['evaluation_seconds'] = report.get('evaluation_seconds', 0.) + paused_seconds
            report['evaluation_history'].append(evaluation['evaluation_id'])
            report['evaluation_history'] = report['evaluation_history'][-100:]
            report['next_evaluation_update'] = due + (int((report['updates'] - due) / config.evaluation_every_updates) + 1) * config.evaluation_every_updates
            write_json(output / 'report.json', report)

        if resume_state is not None:
            periodic_evaluation()
        episode_step = 0
        with (output / 'transitions.jsonl').open('a') as transitions, (output / 'updates.jsonl').open('a') as updates:
            for update in range(start_updates, config.updates):
                if resume_state is not None and (stop_requested or (control.pending is not None and control.pending['stop'])):
                    break
                report['safe_boundary'] = False
                rollout: list[RolloutTransition] = []
                for _ in range(config.rollout_steps):
                    if not env.observation_space.contains(obs):
                        raise ValueError('Invalid policy observation')
                    observing = observer is not None and observer.due()
                    observed = time.time()
                    capture_started = time.perf_counter()
                    raw = env.last_frame.copy() if observing and env.last_frame is not None else None
                    with torch.no_grad():
                        distribution, value = policy(pixel_tensor(obs, device),
                            observe=observer.capture if observing and observer else None)
                        sampled = distribution.sample()
                        action = int(sampled.item())
                        logp = float(distribution.log_prob(sampled).item())
                    capture_seconds = time.perf_counter() - capture_started
                    following, reward, terminated, truncated, info = env.step(action)
                    control.poll(stop_requested, previous_seconds + time.perf_counter() - learning_started >= report['next_autosave_seconds'])
                    episode_step += 1
                    report['transitions'] += 1
                    report['emulator_frames'] += info['executed_frames']
                    if not all(np.isfinite(number) for number in (reward, logp, float(value.item()))):
                        raise PreparationError('nonfinite_training', 'Nonfinite rollout reward, value or log probability')
                    log_row(transitions, {'run_id': report['run_id'], 'session_id': report['session_id'],
                        'transition': report['transitions'], 'episode': report['episodes'],
                        'observation_sha256': hashlib.sha256(obs.tobytes()).hexdigest(),
                        'next_observation_sha256': hashlib.sha256(following.tobytes()).hexdigest(),
                        'action': action, 'probabilities': distribution.probs[0].cpu().tolist(),
                        'reward': reward, 'terminated': terminated, 'truncated': truncated, **info})
                    if info['reason'] in ('position_discontinuity', 'unexpected_game_state', 'backend_end_unclassified'):
                        raise PreparationError('game_contract_violation', f"Invalid training transition: {info['reason']}")
                    if not env.observation_space.contains(following):
                        raise ValueError('Invalid next observation')
                    with torch.no_grad():
                        _, bootstrap = policy(pixel_tensor(following, device))
                    rollout.append(RolloutTransition(obs, action, logp, float(value.item()), float(reward),
                        0. if terminated else float(bootstrap.item()), terminated or truncated))
                    report['reward_sum'] += reward
                    report['max_progress'] = max(report['max_progress'], info['max_progress'])
                    if observing and observer is not None and raw is not None:
                        observer.submit(raw, obs.copy(), distribution.probs[0].cpu().tolist(), action,
                            report, update, observed, episode_step,
                            {'reward': reward, 'terminated': terminated, 'truncated': truncated,
                             'reason': info['reason'], 'max_progress': info['max_progress'],
                             'position': info['position']}, device, capture_seconds)
                    if terminated or truncated:
                        report['episodes'] += 1
                        outcomes = report['episode_outcomes']
                        outcomes[info['reason']] = outcomes.get(info['reason'], 0) + 1
                        obs, _ = env.reset()
                        episode_step = 0
                    else:
                        obs = following
                metrics = optimize(policy, optimizer, rollout, config, device)
                report.update(metrics)
                report['updates'] = update + 1
                report['optimizer_steps'] += config.epochs
                report['safe_boundary'] = True
                elapsed = time.perf_counter() - learning_started
                report['training_seconds'] = previous_seconds + elapsed
                report['transitions_per_second'] = report['transitions'] / report['training_seconds']
                log_row(updates, report)
                write_json(output / 'report.json', report)
                periodic_evaluation()
                control.poll(stop_requested)
                if control.pending is not None and not control.pending['stop']:
                    persist()
                if stop_requested or (control.pending is not None and control.pending['stop']) or (stop_after_updates is not None and update + 1 - start_updates >= stop_after_updates):
                    break
        report['parameter_delta_l2'] = {name: float((p.detach() - initial_parameters[name]).norm())
                                        for name, p in policy.named_parameters()}
        if not all(delta > 0 for delta in report['parameter_delta_l2'].values()):
            raise PreparationError('parameters_unchanged', 'At least one declared trainable parameter did not change')
        control.poll(True)
        persist()
        if digest(output / 'initial.pt') != report['initial_sha256']:
            raise ValueError('Protected initial policy changed')
        report['status'] = 'completed' if report['updates'] == config.updates else 'saved'
    except BaseException as exc:
        report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        report['error'] = str(exc) or type(exc).__name__
        if isinstance(exc, PreparationError) and exc.code == 'nonfinite_training':
            report['nonfinite_count'] += 1
        raise
    finally:
        if env is not None:
            env.close()
            report['environment_closed'] = env.closed
        if recording is not None:
            report['recording_error'] = recording.error
            report['recording_status'] = 'failed' if recording.error else 'complete'
            if recording.error:
                print(recording.error, file=sys.stderr)
        if observer is not None:
            observer.close(report)
            report['observer'] = dict(observer.stats)
        from flycade.storage import cleanup_storage
        report['storage'] = cleanup_storage(output)
        report['wall_seconds'] = time.perf_counter() - started
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
        write_json(output / 'report.json', report)
        session.update(end_updates=report['updates'], status=report['status'],
                       checkpoint_id=report.get('checkpoint_id'), recording_error=report.get('recording_error'))
        write_json(sessions / f"{report['session_id']}.json", session)
        control.close(report.get('error'))
        signal.signal(signal.SIGINT, previous_signal)
    return {'output': str(output.resolve()), **report}


def train(home: Path, graph: Path, output: Path, config: TrainingConfig,
          game: GameConfig, device: str, fixture: bool = False,
          stop_after_updates: int | None = None,
          initial_evaluation: dict[str, Any] | None = None, observe_hz: int = 3,
          warm_state: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_observation_rate(observe_hz)
    if output.exists():
        raise PreparationError('output_exists', f'Refusing to overwrite Run {output}')
    if stop_after_updates is not None and stop_after_updates <= 0:
        raise ValueError('stop-after-updates must be positive')
    if initial_evaluation is not None:
        from flycade.evaluation import EvaluationConfig
        EvaluationConfig(**initial_evaluation)
    # Validate before reserving the directory, preserving the CLI no-artifacts-on-invalid-config contract.
    inspect_graph(graph)
    nodes = json.loads((graph / 'nodes.json').read_text())
    edges = np.load(graph / 'edge_index.npy', allow_pickle=False).T.tolist()
    reachable = {n['index'] for n in nodes if n['input']}
    for _ in range(config.propagation_steps):
        reachable |= {dst for src, dst in edges if src in reachable}
    if not reachable.intersection(n['index'] for n in nodes if n['output']):
        raise ValueError('No input-to-output path within propagation_steps')
    if warm_state is not None:
        channels = game.frame_stack * (3 if game.color == 'RGB' else 1)
        candidate = policy_for_kind(graph, config.model_kind, config.state_dim, config.propagation_steps, channels)
        candidate.load_state_dict(warm_state['model'], strict=True)
    output.mkdir(parents=True)
    with run_lock(output):
        return _train(home, graph, output, config, game, device, fixture, stop_after_updates,
                      initial_evaluation=initial_evaluation, observe_hz=observe_hz, warm_state=warm_state)


def resume(output: Path, home: Path, stop_after_updates: int | None = None, observe_hz: int = 3) -> dict[str, Any]:
    if stop_after_updates is not None and stop_after_updates <= 0:
        raise ValueError('stop-after-updates must be positive')
    validate_observation_rate(observe_hz)
    with run_lock(output):
        state = load_checkpoint(output, home)
        manifest = state['manifest']
        from flycade.budget import budget_info
        budget = budget_info(output, manifest)
        if state['progress'].get('budget', {}).get('total_updates', 0) > budget['total_updates']:
            raise ValueError('Budget ledger is older than the checkpoint; restore the ledger')
        config = replace(TrainingConfig(**manifest['training']), updates=budget['total_updates'])
        due = state['progress'].get('next_evaluation_update')
        pending_evaluation = due is not None and due <= state['progress']['updates']
        if state['progress']['updates'] >= config.updates and not pending_evaluation:
            raise PreparationError('budget_completed', 'Run budget is already complete; no schedule restart')
        return _train(home, output / 'graph', output, config, GameConfig(**manifest['game']),
                      manifest['device'], manifest['fixture'], stop_after_updates, state, observe_hz=observe_hz)
