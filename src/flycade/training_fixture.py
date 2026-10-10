"""Explicit synthetic emulator for CPU contract checks; never NES evidence."""
import numpy as np
from flycade.game import Pixels


class FixtureEmulator:
    buttons: list[str | None] = ['B', 'A', 'LEFT', 'RIGHT']

    def reset(self) -> tuple[Pixels, dict[str, int]]:
        self.x = 40
        return self._frame()

    def _frame(self) -> tuple[Pixels, dict[str, int]]:
        return np.full((16, 16, 3), self.x % 256, dtype=np.uint8), {
            'player_page': self.x // 256, 'player_x': self.x % 256,
            'screen_page': 0, 'screen_x': 0, 'engine': 8, 'lives': 2,
            'world': 0, 'level': 0, 'time': 400, 'timer_expired': 0, 'mode': 1}

    def step(self, buttons: list[int]) -> tuple[Pixels, dict[str, int], bool, bool]:
        self.x += 2 if buttons[3] else 0
        frame, info = self._frame()
        return frame, info, False, False

    def close(self) -> None:
        pass
