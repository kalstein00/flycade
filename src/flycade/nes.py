"""Stable-Retro system boundary. Only one live emulator in this process."""
import threading
from pathlib import Path
from typing import Any, cast

from flycade.errors import PreparationError
from flycade.game import Pixels
from flycade.rom import inspect_registration

_EMULATOR_LOCK = threading.Lock()


class NesEmulator:
    def __init__(self, home: Path):
        import stable_retro as retro

        self.manifest = inspect_registration(home)
        if not _EMULATOR_LOCK.acquire(blocking=False):
            raise PreparationError('emulator_busy', 'Use a separate process for another emulator.')
        self.saved_paths = list(retro.data.Integrations.CUSTOM_ONLY.paths)
        self.closed = False
        self.env: Any = None
        try:
            retro.data.Integrations.clear_custom_paths()
            retro.data.Integrations.add_custom_path(str(home.resolve()))
            self.env = retro.make(self.manifest['game'], state=self.manifest['state'],
                                  inttype=retro.data.Integrations.CUSTOM_ONLY,
                                  use_restricted_actions=retro.Actions.ALL,
                                  obs_type=retro.Observations.IMAGE, render_mode='rgb_array')
            self.buttons: list[str | None] = list(self.env.buttons)
            if self.buttons != self.manifest['buttons']:
                raise PreparationError('buttons_mismatch', 'Runtime buttons differ from the selected release.')
        except BaseException:
            self.close()
            raise

    def _info(self, info: dict[str, Any]) -> dict[str, int]:
        result = {k: int(v) for k, v in info.items()}
        # Supported SMB ROM: Enemy_Flag=$0f, Enemy_ID=$16, Enemy_State=$1e.
        # Reward diagnostics only; none of these values enters policy observations.
        ram = self.env.get_ram()
        for slot in range(5):
            for field, address in (('active', 0x0f), ('id', 0x16), ('state', 0x1e)):
                result[f'enemy_{field}_{slot}'] = int(ram[address + slot])
        return result

    def reset(self) -> tuple[Pixels, dict[str, int]]:
        frame, _ = self.env.reset()
        return cast(Pixels, frame), self._info(self.env.data.lookup_all())

    def step(self, buttons: list[int]) -> tuple[Pixels, dict[str, int], bool, bool]:
        frame, _, terminated, truncated, info = self.env.step(buttons)
        return cast(Pixels, frame), self._info(info), bool(terminated), bool(truncated)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                if self.env is not None:
                    self.env.close()
            finally:
                import stable_retro as retro
                retro.data.Integrations.clear_custom_paths()
                for path in self.saved_paths:
                    retro.data.Integrations.add_custom_path(path)
                _EMULATOR_LOCK.release()
