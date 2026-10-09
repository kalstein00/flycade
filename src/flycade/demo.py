"""Bounded real NES execution with streaming transition evidence."""
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from flycade.diagnostics import diagnose
from flycade.errors import PreparationError
from flycade.game import ACTIONS, GameConfig, GameEnv, Pixels
from flycade.nes import NesEmulator
from flycade.rom import inspect_registration


def save_observation(output: Path, name: str, observation: Pixels, raw: Pixels | None) -> None:
    np.save(output / f'{name}-policy.npy', observation, allow_pickle=False)
    if raw is not None:
        Image.fromarray(raw).save(output / f'{name}-screen.png')
    final = observation[-1]
    Image.fromarray(final if final.shape[-1] == 3 else final[:, :, 0]).save(output / f'{name}-policy.png')


def demo(home: Path, output: Path, steps: int, actions: list[int], config: GameConfig) -> dict[str, Any]:
    if steps <= 0 or not actions or any(type(a) is not int or not 0 <= a < len(ACTIONS) for a in actions):
        raise ValueError('steps must be positive; actions must be comma-separated IDs 0..6')
    manifest = inspect_registration(home)
    if output.exists():
        raise PreparationError('output_exists', f'Refusing to overwrite {output}; choose a new output directory.')
    output.mkdir(parents=True)
    config = replace(config, max_frames=min(config.max_frames, steps * config.action_repeat))
    report: dict[str, Any] = {'started_utc': datetime.now(timezone.utc).isoformat(),
                              'evidence': 'real NES; execution pending', 'manifest': manifest,
                              'diagnostics': diagnose(output), 'action_schedule': actions,
                              'requested_steps': steps, 'transitions': 0,
                              'semantic_validation': 'individual terminal events require trace/image review; see docs/validation/A1.md'}
    try:
        with GameEnv(NesEmulator(home), config) as env:
            report['contract'] = env.describe()
            with (output / 'transitions.jsonl').open('w') as log:
                obs, info = env.reset()
                report['reset'] = info
                report['native_shape'] = list(env.last_frame.shape) if env.last_frame is not None else None
                save_observation(output, 'reset', obs, env.last_frame)
                for index in range(steps):
                    action = actions[index % len(actions)]
                    obs, reward, terminated, truncated, info = env.step(action)
                    row = {'transition': index + 1, 'action': action, 'buttons': ACTIONS[action],
                           'reward': reward, 'terminated': terminated, 'truncated': truncated,
                           'observation_sha256': hashlib.sha256(obs.tobytes()).hexdigest(), **info}
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    report['transitions'] = index + 1
                    report['last_transition'] = row
                    if terminated or truncated:
                        break
                save_observation(output, 'last', obs, env.last_frame)
                report['evidence'] = 'real NES reset/step executed; review artifacts for AC-01'
                if info['reason'] in ('position_discontinuity', 'unexpected_game_state', 'backend_end_unclassified'):
                    raise PreparationError('game_contract_violation', f"Review transition trace: {info['reason']}")
    except BaseException as exc:
        report['evidence'] = 'real NES attempt failed or interrupted'
        report['error'] = str(exc) or type(exc).__name__
        raise
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return {'output': str(output.resolve()), **report}
