"""Synthetic emulator boundary: these tests do not execute or validate NES RAM."""
import numpy as np

from flycade.game import GameConfig, GameEnv


def ram(x=250, screen=0, **changes):
    return dict(player_page=x // 256, player_x=x % 256,
                screen_page=screen // 256, screen_x=screen % 256,
                engine=8, lives=2, world=0, level=0, time=400,
                timer_expired=0, mode=1, **changes)


class SyntheticEmulator:
    buttons = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

    def __init__(self, frames):
        self.frames = frames
        self.closed = False
        self.pressed = []

    def reset(self):
        self.index = 0
        return np.full((240, 256, 3), 30, dtype=np.uint8), self.frames[0]

    def step(self, buttons):
        self.pressed.append(buttons)
        self.index += 1
        return np.full((240, 256, 3), 30 + self.index, dtype=np.uint8), self.frames[self.index], False, False

    def close(self):
        self.closed = True


def test_pixels_buttons_scroll_byte_wrap_backtracking_and_reset():
    backend = SyntheticEmulator([ram(), ram(258, 10), ram(250, 10), ram(258, 10), ram(262, 14)])
    with GameEnv(backend, GameConfig(action_repeat=1, frame_stack=2)) as env:
        obs, info = env.reset()
        assert obs.shape == (2, 84, 84, 3)
        assert obs.dtype == np.uint8
        assert info['position'] == 250
        first = env.step(4)
        assert first[1] == 8
        assert backend.pressed[0] == [1, 0, 0, 0, 0, 0, 0, 1, 1]
        assert env.step(6)[1] == 0
        assert env.step(1)[1] == 0
        assert env.step(1)[1] == 4
        env.reset()
        assert env.step(1)[1] == 8
    assert backend.closed


def test_death_stops_repeat_immediately_and_requires_reset():
    import pytest
    dead = ram(252)
    dead['engine'] = 11
    backend = SyntheticEmulator([ram(), dead])
    with GameEnv(backend, GameConfig(action_repeat=4)) as env:
        env.reset()
        _, reward, terminated, truncated, info = env.step(1)
        assert (reward, terminated, truncated, info['reason'], info['executed_frames']) == (-10, True, False, 'death', 1)
        with pytest.raises(RuntimeError):
            env.step(1)


def test_game_events_and_external_limits_have_distinct_bootstrap_semantics():
    cases = [({'engine': 5, 'time': 0}, {}, 'completion', True, False, 100),
             ({'engine': 11, 'timer_expired': 1, 'time': 0}, {}, 'game_timeout', True, False, -10),
             ({}, {'no_progress_frames': 1}, 'no_progress', False, True, 0),
             ({}, {'max_frames': 1}, 'external_limit', False, True, 0),
             ({'player_page': 50}, {}, 'position_discontinuity', False, True, 0)]
    for changes, config, reason, terminated, truncated, reward in cases:
        frame = ram()
        frame.update(changes)
        with GameEnv(SyntheticEmulator([ram(), frame]), GameConfig(action_repeat=4, **config)) as env:
            env.reset()
            _, actual_reward, term, trunc, info = env.step(0)
            assert (info['reason'], term, trunc, actual_reward) == (reason, terminated, truncated, reward)


def test_frame_stack_color_repeat_and_close_on_error():
    import pytest
    backend = SyntheticEmulator([ram(), ram(252), ram(254)])
    with pytest.raises(ValueError, match='Invalid action'):
        with GameEnv(backend, GameConfig(color='L', frame_stack=2, action_repeat=2)) as env:
            obs, _ = env.reset()
            assert obs.shape == (2, 84, 84, 1)
            obs, reward, term, trunc, info = env.step(1)
            assert reward == 4
            assert not term and not trunc
            assert info['executed_frames'] == 2
            assert int(obs[0, 0, 0, 0]) == 30
            assert int(obs[1, 0, 0, 0]) == 32
            env.step(7)
    assert backend.closed


def test_reset_rejects_other_world_and_invalid_configuration():
    import pytest
    from flycade.errors import PreparationError
    wrong = ram()
    wrong['world'] = 1
    with GameEnv(SyntheticEmulator([wrong])) as env:
        with pytest.raises(PreparationError, match='World 1-1'):
            env.reset()
        with pytest.raises(RuntimeError):
            env.step(0)
    for kwargs in [{'action_repeat': 0}, {'color': 'RAM'}, {'progress_reward': float('nan')}]:
        with pytest.raises(ValueError):
            GameConfig(**kwargs)
