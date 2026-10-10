"""Reward redesign at the pixel environment boundary, synthetic events only."""
import pytest

from flycade.game import GameConfig, GameEnv
from test_game import SyntheticEmulator, ram


def event(x=40, *, active=1, enemy_id=6, state=0, **changes):
    info = ram(x)
    info.update(changes)
    for slot in range(5):
        info.update({f'enemy_active_{slot}': active if slot == 0 else 0,
                     f'enemy_id_{slot}': enemy_id if slot == 0 else 0,
                     f'enemy_state_{slot}': state if slot == 0 else 0})
    return info


def test_reward_components_progress_backtracking_defeat_and_death():
    backend = SyntheticEmulator([event(), event(50), event(45), event(50, state=4),
                                 event(50, state=4), event(50, engine=11)])
    config = GameConfig(action_repeat=1, progress_reward=.01, survival_reward=-.001,
                        death_reward=-10, enemy_reward=.5, enemy_reward_cap=5)
    with GameEnv(backend, config) as env:
        env.reset()
        rows = [env.step(4) for _ in range(5)]
    assert [r[1] for r in rows] == pytest.approx([.099, -.001, .499, -.001, -10])
    assert rows[2][4]['enemy_defeats'] == 1
    assert rows[3][4]['enemy_defeats'] == 0
    assert rows[-1][2] and not rows[-1][3]
    for _, reward, _, _, info in rows:
        assert sum(info['reward_components'].values()) == pytest.approx(reward)


def test_stall_terminal_cost_has_no_bootstrap_and_external_limit_stays_truncated():
    config = GameConfig(action_repeat=4, survival_reward=-.001, stall_reward=-.005,
                        stall_grace_frames=1, no_progress_frames=3,
                        no_progress_reward=-10, terminate_on_no_progress=True)
    with GameEnv(SyntheticEmulator([event()] * 5), config) as env:
        env.reset()
        _, reward, term, trunc, info = env.step(0)
        assert (term, trunc, info['reason'], info['executed_frames']) == (True, False, 'no_progress', 3)
        assert reward == pytest.approx(-10.013)
        assert info['reward_components']['stall'] == pytest.approx(-.01)
    with GameEnv(SyntheticEmulator([event()] * 5), GameConfig(max_frames=2, no_progress_reward=-10,
                 terminate_on_no_progress=True)) as env:
        env.reset()
        _, _, term, trunc, info = env.step(0)
        assert (term, trunc, info['reason']) == (False, True, 'external_limit')


def test_enemy_reward_is_deduplicated_bounded_and_reset_per_episode():
    frames = [event(), event(state=4), event(state=0), event(state=4),
              event(active=0), event(), event(state=4), event(active=0), event(), event(state=4)]
    with GameEnv(SyntheticEmulator(frames), GameConfig(action_repeat=1, enemy_reward=.5,
                 enemy_reward_cap=.75)) as env:
        env.reset()
        rows = [env.step(4) for _ in range(9)]
        assert [r[4]['reward_components']['enemy'] for r in rows] == [.5, 0, 0, 0, 0, .25, 0, 0, 0]
        env.reset()
        assert env.step(4)[4]['reward_components']['enemy'] == .5
    # A Koopa becoming a shell, or an inactive slot, is not a confirmed defeat.
    with GameEnv(SyntheticEmulator([event(enemy_id=0), event(enemy_id=0, state=4),
                 event(enemy_id=0, state=0x24), event(active=0, state=0x20)]),
                 GameConfig(action_repeat=1, enemy_reward=.5)) as env:
        env.reset()
        assert env.step(4)[4]['enemy_defeats'] == 0
        assert env.step(4)[4]['enemy_defeats'] == 1
        assert env.step(4)[4]['enemy_defeats'] == 0


def test_enabled_enemy_reward_rejects_missing_signals_and_invalid_costs():
    from flycade.errors import PreparationError
    with GameEnv(SyntheticEmulator([ram()]), GameConfig(enemy_reward=.5)) as env:
        with pytest.raises(PreparationError, match='verified NES enemy states'):
            env.reset()
        with pytest.raises(RuntimeError, match='reset'):
            env.step(0)
    for config in ({'enemy_reward': -.5}, {'enemy_reward_cap': -1},
                   {'stall_reward': .1}, {'no_progress_reward': 1},
                   {'terminate_on_no_progress': 'yes'}, {'enemy_reward': float('nan')}):
        with pytest.raises(ValueError):
            GameConfig(**config)


def test_fall_below_playfield_is_terminal_before_respawn_or_extra_progress():
    frames = [event(player_y_high=1), event(44, player_y_high=0),
              event(48, player_y_high=1), event(52, player_y_high=2)]
    with GameEnv(SyntheticEmulator(frames), GameConfig(action_repeat=1, progress_reward=.01)) as env:
        env.reset()
        assert not env.step(4)[2]  # Above the screen during a high jump is not a fall.
        assert not env.step(4)[2]
        _, reward, term, trunc, info = env.step(4)
        assert (term, trunc, info['reason']) == (True, False, 'death')
        assert reward == -10
        assert info['reward_components']['progress'] == 0
        assert info['max_progress'] == 8
