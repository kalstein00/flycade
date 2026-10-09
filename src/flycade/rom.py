"""Validate the selected release and install only user-supplied local ROMs."""
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from flycade.errors import PreparationError

INTEGRATION = Path(__file__).parent / 'integration'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog() -> dict[str, Any]:
    import stable_retro as retro

    release: dict[str, Any] = json.loads((INTEGRATION / 'release.json').read_text())
    version = importlib.metadata.version('stable-retro')
    if version != release['stable_retro']:
        raise PreparationError('release_mismatch', 'Run uv sync --locked; selected Stable-Retro release differs.')
    game = release['game']
    if game not in retro.data.list_games() or release['state'] not in retro.data.list_states(game):
        raise PreparationError('integration_missing', 'Selected game or Level1-1 state is absent; run uv sync --locked.')
    for filename, expected in release['upstream_sha256'].items():
        path = retro.data.get_file_path(game, filename, retro.data.Integrations.STABLE)
        if not path or digest(Path(path)) != expected:
            raise PreparationError('integration_mismatch', f'Installed integration differs: {filename}')
    core = retro.get_system_info('Nes')
    if core['buttons'] != release['buttons'] or core['lib'] != release['core']:
        raise PreparationError('core_mismatch', 'NES core/button definition differs from selected release.')
    return {**release, 'available_states': retro.data.list_states(game),
            'core_sha256': digest(Path(retro.get_core_path('Nes'))),
            'custom_sha256': {name: digest(INTEGRATION / name) for name in ['data.json', 'scenario.json']}}


def validate_rom(path: Path, expected: str) -> dict[str, str]:
    if not path.is_file():
        raise PreparationError('rom_missing', f'ROM not found: {path}. Supply your own local .nes file.')
    data = path.read_bytes()
    if len(data) < 16 or data[:4] != b'NES\x1a':
        raise PreparationError('rom_format', 'Expected an uncompressed iNES .nes file with a 16-byte header.')
    # Same NES checksum convention as stable_retro.data.read_rom: exclude 16-byte header.
    actual = hashlib.sha1(data[16:]).hexdigest()
    if actual != expected:
        raise PreparationError('rom_hash_mismatch', f'Expected payload SHA-1 {expected}; got {actual}. Check ROM edition; no ROM is downloaded.')
    return {'rom_sha256': hashlib.sha256(data).hexdigest(), 'rom_payload_sha1': actual}


def register_rom(path: Path, home: Path) -> dict[str, Any]:
    import stable_retro as retro

    # Missing ROM must remain distinguishable even before dependency inspection.
    if not path.is_file():
        raise PreparationError('rom_missing', f'ROM not found: {path}')
    release = catalog()
    hashes = validate_rom(path, release['rom_payload_sha1'])
    home.mkdir(parents=True, exist_ok=True)
    destination = home / release['game']
    if destination.exists():
        raise PreparationError('already_registered', f'{destination} exists. Use inspect to verify it, or choose another --home.')
    with tempfile.TemporaryDirectory(dir=home) as staging:
        game_dir = Path(staging) / release['game']
        game_dir.mkdir()
        shutil.copyfile(path, game_dir / 'rom.nes')
        for name in ['Level1-1.state', 'rom.sha', 'metadata.json']:
            shutil.copyfile(retro.data.get_file_path(release['game'], name, retro.data.Integrations.STABLE), game_dir / name)
        for name in ['data.json', 'scenario.json']:
            shutil.copyfile(INTEGRATION / name, game_dir / name)
        report = {**release, **hashes, 'evidence': 'registered; NES execution unverified'}
        (game_dir / 'registration.json').write_text(json.dumps(report, indent=2) + '\n')
        os.rename(game_dir, destination)
    return report


def inspect_registration(home: Path) -> dict[str, Any]:
    release = catalog()
    directory = home / release['game']
    hashes = validate_rom(directory / 'rom.nes', release['rom_payload_sha1'])
    for name, expected in {**release['custom_sha256'],
                           **{k: release['upstream_sha256'][k] for k in ['Level1-1.state', 'rom.sha', 'metadata.json']}}.items():
        path = directory / name
        if not path.is_file():
            raise PreparationError('artifact_missing', f'Missing {path}; register again into a fresh --home.')
        if digest(path) != expected:
            raise PreparationError('artifact_hash_mismatch', f'Changed artifact: {path}; register again into a fresh --home.')
    registration = directory / 'registration.json'
    if not registration.is_file():
        raise PreparationError('artifact_missing', f'Missing {registration}')
    original = json.loads(registration.read_text())
    if original['rom_sha256'] != hashes['rom_sha256']:
        raise PreparationError('rom_hash_mismatch', 'ROM bytes changed since registration (including the iNES header).')
    return {**release, **hashes, 'evidence': 'artifacts verified; NES execution unverified'}
