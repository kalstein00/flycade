"""Bounded, low-bandwidth gameplay recording; no policy or RNG access."""
import os
import select
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from flycade.checkpoint import atomic_json, sync_directory
from flycade.game import Emulator, Pixels
from flycade.graph import digest


class RecordingEmulator:
    """Stream every fifth NES frame to VP9; publish independently playable 60s clips."""

    def __init__(self, emulator: Emulator, output: Path, session: dict[str, Any]):
        self.emulator = emulator
        self.buttons = emulator.buttons
        self.directory = output / 'videos'
        self.session = session
        self.process: subprocess.Popen[bytes] | None = None
        self.frames = 0
        self.clip_frames = 0
        self.clip = 0
        self.error: str | None = None
        self.temporary: Path | None = None
        self.row: dict[str, Any] = {}
        self.encoder = shutil.which('ffmpeg')
        try:
            self.directory.mkdir(exist_ok=True)
            if self.encoder is None:
                self.error = 'ffmpeg is missing; install ffmpeg to record subsequent sessions'
        except OSError as exc:
            self.error = f'Video directory unavailable: {exc}'

    def reset(self) -> tuple[Pixels, dict[str, int]]:
        return self.emulator.reset()

    def step(self, buttons: list[int]) -> tuple[Pixels, dict[str, int], bool, bool]:
        result = self.emulator.step(buttons)
        if self.frames % 5 == 0 and self.error is None:
            try:
                self._record(result[0])
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                self.error = f'Video recording failed: {exc}'
                self._abort()
        self.frames += 1
        return result

    def _record(self, frame: Pixels) -> None:
        if self.process is None:
            self.clip += 1
            name = f"{self.session['session_id']}-{self.clip:05d}"
            self.temporary = self.directory / f'{name}.partial'
            height, width, _ = frame.shape
            self.row = {**self.session, 'mode': 'training_recording', 'status': 'recording',
                        'file': f'videos/{name}.webm', 'codec': 'vp9', 'crf': 45,
                        'fps': 12, 'audio': False, 'width': width, 'height': height,
                        'session_frame_start': self.frames, 'segment': self.clip}
            assert self.encoder is not None
            self.process = subprocess.Popen([self.encoder, '-nostdin', '-v', 'error', '-n',
                '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                '-framerate', '12', '-i', 'pipe:0', '-an', '-c:v', 'libvpx-vp9',
                '-crf', '45', '-b:v', '0', '-deadline', 'good', '-cpu-used', '4',
                '-threads', '1', '-pix_fmt', 'yuv420p', '-f', 'webm', str(self.temporary)],
                start_new_session=True, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            assert self.process.stdin is not None
            os.set_blocking(self.process.stdin.fileno(), False)
            self.clip_frames = 0
        assert self.process.stdin is not None
        fd = self.process.stdin.fileno()
        remaining = memoryview(frame.tobytes())
        deadline = time.monotonic() + 10
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0 or not select.select([], [fd], [], timeout)[1]:
                raise RuntimeError('encoder stalled for 10 seconds')
            try:
                written = os.write(fd, remaining)
                remaining = remaining[written:]
            except BlockingIOError:
                continue
        self.clip_frames += 1
        if self.clip_frames == 720:
            self._finish()

    def _finish(self) -> None:
        if self.process is None:
            return
        assert self.process.stdin is not None and self.temporary is not None
        self.process.stdin.close()
        result = self.process.wait(timeout=30)
        self.process = None
        if result != 0:
            raise RuntimeError(f'ffmpeg exited with status {result}')
        with self.temporary.open('rb') as handle:
            os.fsync(handle.fileno())
        final = self.temporary.with_suffix('.webm')
        os.replace(self.temporary, final)
        sync_directory(self.directory)
        self.row.update(status='complete', frames=self.clip_frames,
                        duration_seconds=self.clip_frames / 12, bytes=final.stat().st_size,
                        sha256=digest(final))
        atomic_json(final.with_suffix('.json'), self.row)
        self.temporary = None

    def _abort(self) -> None:
        if self.process is not None:
            self.process.kill()
            self.process.wait(timeout=10)
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process = None
        if self.temporary is not None:
            self.temporary.unlink(missing_ok=True)
            self.temporary = None

    def close(self) -> None:
        try:
            try:
                self._finish()
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                self.error = f'Video finalization failed: {exc}'
                self._abort()
        finally:
            self.emulator.close()


def recording_history(output: Path) -> dict[str, Any]:
    import json
    manifest = json.loads((output / 'run.json').read_text())
    rows = [json.loads(path.read_text()) for path in sorted((output / 'videos').glob('*.json'))]
    rows.sort(key=lambda row: (row['created_unix'], row['segment']))
    for row in rows:
        row['available'] = (output / row['file']).is_file()
    return {'run_id': manifest['run_id'], 'recordings': rows,
            'note': 'Recorded training gameplay, not evaluation or live play; no circuit samples recorded.'}
