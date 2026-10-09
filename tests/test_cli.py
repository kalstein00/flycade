import json
import subprocess
import sys


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'flycade', *map(str, args)],
                          capture_output=True, text=True)


def test_diagnostics_distinguish_host_memory_and_training_evidence(tmp_path):
    result = cli('diagnose', '--directory', tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['memory']['linux_total_bytes'] > 0
    assert 'windows_total_bytes' in report['memory']
    assert 'effective_limit_bytes' in report['memory']
    assert report['disk']['free_bytes'] > 0
    assert report['capabilities']['gpu_training'] == 'unverified: follow-up ticket'
    assert report['packages']['stable-retro'] == '1.0.1'


def test_rom_registration_distinguishes_missing_and_wrong_hash(tmp_path):
    rom = tmp_path / 'user.nes'
    missing = cli('register-rom', rom, '--home', tmp_path / 'home')
    assert missing.returncode == 2
    assert json.loads(missing.stdout)['error']['code'] == 'rom_missing'
    rom.write_bytes(b'NES\x1a' + bytes(12) + b'not a commercial ROM')
    wrong = cli('register-rom', rom, '--home', tmp_path / 'home')
    assert wrong.returncode == 2
    assert json.loads(wrong.stdout)['error']['code'] == 'rom_hash_mismatch'
    assert not (tmp_path / 'home').exists()


def test_catalog_records_selected_release_and_demo_refuses_missing_rom(tmp_path):
    result = cli('catalog')
    assert result.returncode == 0, result.stdout
    catalog = json.loads(result.stdout)
    assert catalog['game'] == 'SuperMarioBros-Nes-v0'
    assert 'Level1-1' in catalog['available_states']
    assert catalog['buttons'] == ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    missing = cli('demo', '--home', tmp_path, '--output', tmp_path / 'demo')
    assert missing.returncode == 2
    assert json.loads(missing.stdout)['error']['code'] == 'rom_missing'
