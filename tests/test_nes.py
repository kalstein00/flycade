"""Opt-in real NES checks: FLYCADE_TEST_ROM must name a user-supplied ROM."""
import json
import os
from pathlib import Path

import pytest

from test_cli import cli

ROM = os.environ.get('FLYCADE_TEST_ROM')
pytestmark = pytest.mark.skipif(not ROM, reason='No user ROM: real NES is not validated by synthetic tests')


def test_real_rom_registration_fixed_reset_buttons_and_tamper_rejection(tmp_path):
    registered = cli('register-rom', ROM, '--home', tmp_path / 'home')
    assert registered.returncode == 0, registered.stdout
    inspected = cli('inspect', '--home', tmp_path / 'home')
    assert inspected.returncode == 0, inspected.stdout
    reports = []
    for name, action in [('idle', 0), ('right', 1), ('right-again', 1)]:
        output = tmp_path / name
        result = cli('demo', '--home', tmp_path / 'home', '--output', output,
                     '--actions', action, '--steps', 20)
        assert result.returncode == 0, result.stdout
        report = json.loads(result.stdout)
        assert report['reset']['position'] == 40
        assert report['reset']['game']['time'] == 400
        assert report['last_transition']['reason'] == 'external_limit'
        assert report['native_shape'] == [224, 240, 3]
        assert (output / 'reset-policy.npy').exists()
        assert (output / 'last-screen.png').exists()
        reports.append(report)
    assert reports[0]['last_transition']['position'] == 40
    assert reports[1]['last_transition']['position'] > 100
    assert reports[1]['last_transition'] == reports[2]['last_transition']
    game = tmp_path / 'home' / 'SuperMarioBros-Nes-v0'
    state = game / 'Level1-1.state'
    state.write_bytes(b'corrupt')
    bad = cli('inspect', '--home', tmp_path / 'home')
    assert json.loads(bad.stdout)['error']['code'] == 'artifact_hash_mismatch'
    state.unlink()
    missing = cli('inspect', '--home', tmp_path / 'home')
    assert json.loads(missing.stdout)['error']['code'] == 'artifact_missing'


def test_real_emulator_single_owner_and_release_on_context_exception(tmp_path):
    from flycade.errors import PreparationError
    from flycade.game import GameEnv
    from flycade.nes import NesEmulator

    assert cli('register-rom', ROM, '--home', tmp_path).returncode == 0
    with pytest.raises(ValueError, match='test exception'):
        with GameEnv(NesEmulator(tmp_path)) as env:
            env.reset()
            with pytest.raises(PreparationError, match='separate process'):
                NesEmulator(tmp_path)
            raise ValueError('test exception')
    with GameEnv(NesEmulator(tmp_path)) as env:
        assert env.reset()[1]['position'] == 40


def test_real_completion_replay_from_recorded_actions_file(tmp_path):
    assert cli('register-rom', ROM, '--home', tmp_path / 'home').returncode == 0
    result = cli('demo', '--home', tmp_path / 'home', '--output', tmp_path / 'completion',
                 '--steps', 342, '--actions-file', Path(__file__).parents[1] / 'docs/validation/completion-actions.json')
    assert result.returncode == 0, result.stdout + result.stderr
    event = json.loads(result.stdout)['last_transition']
    assert (event['reason'], event['terminated'], event['truncated']) == ('completion', True, False)
    assert event['frames'] == 1361
