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
    no_progress_frames: int = 600
    max_frames: int = 18000

    def __post_init__(self) -> None:
        for name in ['width', 'height', 'frame_stack', 'action_repeat', 'no_progress_frames', 'max_frames']:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.color not in ('RGB', 'L'):
            raise ValueError('color must be RGB or L')
        for name in ['progress_reward', 'survival_reward', 'death_reward', 'completion_reward']:
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')


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
                'game': info.copy()}

    def step(self, action: int) -> tuple[Pixels, float, bool, bool, dict[str, Any]]:
        if self.finished or self.closed:
            raise RuntimeError('Call reset before stepping a new episode')
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action}')
        mask = [int(button in ACTIONS[action]) for button in self.emulator.buttons]
        reward = 0.0
        reason = None
        terminated = truncated = False
        for executed in range(1, self.config.action_repeat + 1):
            frame, info, backend_done, backend_truncated = self.emulator.step(mask)
            self.frames += 1
            # Inspect every emulator frame, before animation/respawn can create reward.
            if info['engine'] == 5:
                reason, terminated = 'completion', True
                reward += self.config.completion_reward
            elif info['timer_expired']:
                reason, terminated = 'game_timeout', True
                reward += self.config.death_reward
            elif info['engine'] in (6, 11) or info['lives'] < self.previous['lives']:
                reason, terminated = 'death', True
                reward += self.config.death_reward
            elif (info['world'], info['level'], info['mode']) != (0, 0, 1):
                reason, truncated = 'unexpected_game_state', True
            elif backend_done or backend_truncated:
                reason, truncated = 'backend_end_unclassified', True
            elif abs(self._position(info) - self._position(self.previous)) > 16:
                reason, truncated = 'position_discontinuity', True
            else:
                progress = max(0, self._position(info) - self.maximum)
                self.maximum = max(self.maximum, self._position(info))
                self.stalled = 0 if progress else self.stalled + 1
                reward += progress * self.config.progress_reward + self.config.survival_reward
                if self.stalled >= self.config.no_progress_frames:
                    reason, truncated = 'no_progress', True
                elif self.frames >= self.config.max_frames:
                    reason, truncated = 'external_limit', True
            self.previous = info
            if terminated or truncated:
                self.finished = True
                break
        self.history.append(self._pixels(frame))
        return np.stack(self.history), reward, terminated, truncated, self._info(info, reason, executed)

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
