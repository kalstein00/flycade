"""Pixel-only NES transition API; RAM is used exclusively for reward and info."""
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, Self, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from flycade.errors import PreparationError

Pixels = NDArray[np.uint8]
ACTIONS = ((), ('RIGHT',), ('RIGHT', 'A'), ('RIGHT', 'B'),
           ('RIGHT', 'B', 'A'), ('A',), ('LEFT',))


class Emulator(Protocol):
    buttons: list[str | None]

    def reset(self) -> tuple[Pixels, dict[str, int]]: ...
    def step(self, buttons: list[int]) -> tuple[Pixels, dict[str, int], bool, bool]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class GameConfig:
    width: int = 84
    height: int = 84
    color: str = 'RGB'
    frame_stack: int = 4
    action_repeat: int = 4
    progress_reward: float = 1.0
    survival_reward: float = 0.0
    death_reward: float = -10.0
    completion_reward: float = 100.0
    stall_reward: float = 0.0
    stall_grace_frames: int = 60
    no_progress_reward: float = 0.0
    terminate_on_no_progress: bool = False
    enemy_reward: float = 0.0
    enemy_reward_cap: float = 5.0
    no_progress_frames: int = 600
    max_frames: int = 18000

    def __post_init__(self) -> None:
        for name in ['width', 'height', 'frame_stack', 'action_repeat', 'no_progress_frames', 'max_frames', 'stall_grace_frames']:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.color not in ('RGB', 'L'):
            raise ValueError('color must be RGB or L')
        for name in ['progress_reward', 'survival_reward', 'death_reward', 'completion_reward',
                     'stall_reward', 'no_progress_reward', 'enemy_reward', 'enemy_reward_cap']:
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if self.stall_reward > 0 or self.no_progress_reward > 0:
            raise ValueError('Stall and no-progress costs must be nonpositive')
        if self.enemy_reward < 0 or self.enemy_reward_cap < 0:
            raise ValueError('Enemy reward and cap must be nonnegative')
        if type(self.terminate_on_no_progress) is not bool:
            raise ValueError('terminate_on_no_progress must be boolean')


class GameEnv(gym.Env[Pixels, int]):
    """One reset/step stream. Owns and closes its emulator, also on context errors."""

    def __init__(self, emulator: Emulator, config: GameConfig = GameConfig()):
        self.emulator = emulator
        self.config = config
        self.action_space = cast(gym.Space[int], gym.spaces.Discrete(len(ACTIONS)))
        channels = 3 if config.color == 'RGB' else 1
        self.observation_space = gym.spaces.Box(0, 255,
            shape=(config.frame_stack, config.height, config.width, channels), dtype=np.uint8)
        self.history: deque[Pixels] = deque(maxlen=config.frame_stack)
        self.finished = True
        self.closed = False
        self.last_frame: Pixels | None = None
        if not all(button in emulator.buttons for combo in ACTIONS for button in combo):
            emulator.close()
            raise PreparationError('buttons_mismatch', 'Required NES buttons are missing.')

    def _pixels(self, frame: Pixels) -> Pixels:
        self.last_frame = frame.copy()
        converted = Image.fromarray(frame).convert(self.config.color).resize(
            (self.config.width, self.config.height), Image.Resampling.BILINEAR)
        result = np.asarray(converted, dtype=np.uint8)
        if result.ndim == 2:
            result = result[:, :, None]
        return result

    @staticmethod
    def _position(info: dict[str, int]) -> int:
        return info['player_page'] * 256 + info['player_x']

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Pixels, dict[str, Any]]:
        if self.closed:
            raise RuntimeError('Environment is closed')
        super().reset(seed=seed)
        self.finished = True
        frame, info = self.emulator.reset()
        if (info['world'], info['level'], info['mode']) != (0, 0, 1) or info['time'] <= 0:
            raise PreparationError('start_state_mismatch', 'Reset did not produce active World 1-1 with a running timer.')
        if self.config.enemy_reward and any(f'enemy_{field}_{slot}' not in info
                for slot in range(5) for field in ('active', 'id', 'state')):
            raise PreparationError('enemy_events_missing', 'Enemy reward requires verified NES enemy states.')
        self.credited_enemies: set[int] = set()
        self.enemy_reward_paid = 0.0
        self.enemy_defeats = 0
        self.components = dict.fromkeys(('progress', 'time', 'stall', 'death', 'completion', 'no_progress', 'enemy'), 0.)
        self.maximum = self._position(info)
        self.start_position = self.maximum
        self.previous = info
        self.frames = self.stalled = 0
        self.finished = False
        self.history.clear()
        pixel = self._pixels(frame)
        self.history.extend(pixel.copy() for _ in range(self.config.frame_stack))
        return np.stack(self.history), self._info(info, None, 0)

    def _info(self, info: dict[str, int], reason: str | None, executed: int) -> dict[str, Any]:
        return {'position': self._position(info),
                'screen_position': info['screen_page'] * 256 + info['screen_x'],
                'max_progress': self.maximum - self.start_position,
                'frames': self.frames, 'executed_frames': executed, 'reason': reason,
                'game': info.copy(), 'reward_components': self.components.copy(),
                'enemy_defeats': self.enemy_defeats}

    def _defeats(self, info: dict[str, int]) -> int:
        count = 0
        for slot in range(5):
            active, kind, state = (info.get(f'enemy_{field}_{slot}', 0)
                                   for field in ('active', 'id', 'state'))
            previous_active, previous_kind, previous_state = (self.previous.get(f'enemy_{field}_{slot}', 0)
                                   for field in ('active', 'id', 'state'))
            if active != 1 or previous_active != 1 or kind != previous_kind:
                self.credited_enemies.discard(slot)
                continue
            # SMB World1-1: defeat bit5; stomped Goomba state4. Koopa shell != kill.
            defeated = (bool(state & 0x20) and not previous_state & 0x20) or (kind == 6 and state == 4 and previous_state != 4)
            if kind <= 0x14 and defeated and slot not in self.credited_enemies:
                self.credited_enemies.add(slot)
                count += 1
        return count

    def step(self, action: int) -> tuple[Pixels, float, bool, bool, dict[str, Any]]:
        if self.finished or self.closed:
            raise RuntimeError('Call reset before stepping a new episode')
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action}')
        mask = [int(button in ACTIONS[action]) for button in self.emulator.buttons]
        self.components = dict.fromkeys(self.components, 0.)
        self.enemy_defeats = 0
        reason = None
        terminated = truncated = False
        for executed in range(1, self.config.action_repeat + 1):
            frame, info, backend_done, backend_truncated = self.emulator.step(mask)
            self.frames += 1
            # Inspect every emulator frame, before animation/respawn can create reward.
            if info['engine'] == 5:
                reason, terminated = 'completion', True
                self.components['completion'] += self.config.completion_reward
            elif info['timer_expired']:
                reason, terminated = 'game_timeout', True
                self.components['death'] += self.config.death_reward
            elif info['engine'] in (6, 11) or info['lives'] < self.previous['lives']:
                reason, terminated = 'death', True
                self.components['death'] += self.config.death_reward
            elif (info['world'], info['level'], info['mode']) != (0, 0, 1):
                reason, truncated = 'unexpected_game_state', True
            elif info.get('player_y_high', 1) >= 2:
                # World1-1, no DOWN/pipe entry: below-screen fall is irreversible.
                # The engine waits for death music before decrementing lives.
                reason, terminated = 'death', True
                self.components['death'] += self.config.death_reward
            elif backend_done or backend_truncated:
                reason, truncated = 'backend_end_unclassified', True
            elif abs(self._position(info) - self._position(self.previous)) > 16:
                reason, truncated = 'position_discontinuity', True
            else:
                progress = max(0, self._position(info) - self.maximum)
                self.maximum = max(self.maximum, self._position(info))
                self.stalled = 0 if progress else self.stalled + 1
                self.components['progress'] += progress * self.config.progress_reward
                self.components['time'] += self.config.survival_reward
                if self.stalled > self.config.stall_grace_frames:
                    self.components['stall'] += self.config.stall_reward
                defeats = self._defeats(info)
                self.enemy_defeats += defeats
                earned = min(defeats * self.config.enemy_reward,
                             max(0., self.config.enemy_reward_cap - self.enemy_reward_paid))
                self.enemy_reward_paid += earned
                self.components['enemy'] += earned
                if self.stalled >= self.config.no_progress_frames:
                    reason = 'no_progress'
                    terminated = self.config.terminate_on_no_progress
                    truncated = not terminated
                    self.components['no_progress'] += self.config.no_progress_reward
                elif self.frames >= self.config.max_frames:
                    reason, truncated = 'external_limit', True
            self.previous = info
            if terminated or truncated:
                self.finished = True
                break
        self.history.append(self._pixels(frame))
        return np.stack(self.history), sum(self.components.values()), terminated, truncated, self._info(info, reason, executed)

    def describe(self) -> dict[str, Any]:
        return {'config': asdict(self.config), 'actions': [list(a) for a in ACTIONS],
                'button_order': self.emulator.buttons, 'observation_shape': list(self.observation_space.shape or ()),
                'observation_dtype': 'uint8', 'resize': 'Pillow bilinear',
                'frame_stack_order': 'oldest to newest; last frame of each action repeat',
                'policy_input': 'pixels only', 'reward_unit': 'per emulated frame'}

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.finished = True
            self.emulator.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> Literal[False]:
        self.close()
        return False
