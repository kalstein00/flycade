"""Read-only loopback recording history, with browser-native video playback."""
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from flycade.recording import recording_history

PAGE = '''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>플레이 영상 내역 · Flycade</title>
<style>
:root {color-scheme:light dark;font-family:system-ui,sans-serif;line-height:1.6}
body {max-width:70rem;margin:2rem auto;padding:0 1rem}
h1 {font-size:1.7rem;margin-bottom:.5rem} h2 {font-size:1.2rem}
p {max-width:72ch} #run, #detail {overflow-wrap:anywhere}
video {display:block;width:100%;max-width:48rem;max-height:55vh;background:#111}
ul {padding:0;list-style:none} li {border-bottom:1px solid #888;padding:1rem 0}
button {font:inherit;padding:.5rem 1rem;cursor:pointer;min-height:44px}
button:focus-visible,a:focus-visible {outline:3px solid #007dbe;outline-offset:3px}
button[aria-current=true] {font-weight:bold;text-decoration:underline}
small {display:block} #error {color:light-dark(#a11919,#ffb3b3)}
</style>
<main><h1>플레이 영상 내역</h1>
<p id="mode">학습 기록 영상</p><p>학습 플레이와 고정 정책 평가의 저장된 영상을 다시 봅니다. 실시간 화면이 아니며 소리와 회로 활동 이력은 포함하지 않습니다.</p>
<p id="run"></p><video controls preload="metadata" aria-label="선택한 플레이 영상"></video>
<p id="detail" aria-live="polite">목록에서 영상을 선택하세요.</p><p id="error" role="alert"></p>
<h2>저장된 영상</h2><p>각 영상은 최대 1분입니다. 새 기록을 보려면 페이지를 새로고침하세요.</p><ul id="recordings"></ul></main>
<script>
const history = __HISTORY__;
document.querySelector('#run').textContent = 'Run ' + history.run_id;
const player = document.querySelector('video');
const detail = document.querySelector('#detail');
const error = document.querySelector('#error');
player.addEventListener('error', () => {error.textContent = '영상을 읽지 못했습니다. 파일을 확인하고 페이지를 새로고침하세요.';});
for (const [index, recording] of history.recordings.entries()) {
  const item = document.createElement('li');
  const button = document.createElement('button');
  const date = new Date(recording.created_unix * 1000).toLocaleString('ko-KR');
  const modes = {training_recording: '학습 기록 영상', stochastic_evaluation: '확률적 평가 기록 영상',
    deterministic_spectating: '결정론적 관전 기록 영상'};
  const model = recording.snapshot_id === 'initial' ? '초기 모델' :
    (recording.snapshot_id ? '갱신 ' + recording.start_updates + ' 모델' : '학습 플레이');
  const title = date + ' · ' + model + ' · 구간 ' + recording.segment;
  button.textContent = title + ' 재생';
  const info = document.createElement('small');
  info.textContent = '세션 시작: ' + recording.start_updates + ' 갱신 · ' +
    recording.duration_seconds.toFixed(1) + '초 · ' + (recording.bytes / 1024).toFixed(1) + ' KB · ' + modes[recording.mode];
  button.disabled = !recording.available;
  if (!recording.available) info.textContent += ' · 파일 없음';
  button.addEventListener('click', () => {
    for (const other of document.querySelectorAll('button')) other.removeAttribute('aria-current');
    button.setAttribute('aria-current', 'true');
    error.textContent = '';
    player.src = '/' + recording.file;
    player.load();
    document.querySelector('#mode').textContent = modes[recording.mode] || recording.mode;
    detail.textContent = title + ' · 세션 ' + recording.session_id +
      (recording.snapshot_id ? ' · snapshot ' + recording.snapshot_id + ' · protocol ' + recording.protocol_id : '');
  });
  item.append(button, info); document.querySelector('#recordings').append(item);
}
if (!history.recordings.length) document.querySelector('#recordings').textContent =
  '저장된 영상이 없습니다. 녹화가 끝난 뒤 새로고침하세요. 녹화 실패 여부는 Run 보고서에서 확인할 수 있습니다.';
</script></html>'''


def serve_history(output: Path, port: int) -> dict[str, Any]:
    # Validate Run before starting the service. No writes or training imports are needed by the handler.
    recording_history(output)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            history = recording_history(output)
            if path == '/':
                data = PAGE.replace('__HISTORY__', json.dumps(history).replace('<', '\\u003c')).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(data)
                return
            files = {'/' + row['file'] for row in history['recordings'] if row['available']}
            if path not in files or not re.fullmatch(r'/(?:evaluations/[a-f0-9-]+/)?videos/[a-f0-9-]+-\d+\.webm', path):
                self.send_error(404)
                return
            file = output / path.lstrip('/')
            size = file.stat().st_size
            start, end = 0, size - 1
            byte_range = self.headers.get('Range')
            if byte_range:
                match = re.fullmatch(r'bytes=(\d+)-(\d*)', byte_range)
                if not match:
                    self.send_error(416)
                    return
                start = int(match[1])
                end = min(int(match[2]), end) if match[2] else end
                if start > end:
                    self.send_error(416)
                    return
            self.send_response(206 if byte_range else 200)
            self.send_header('Content-Type', 'video/webm')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            if byte_range:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with file.open('rb') as handle:
                    handle.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = handle.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with HTTPServer(('127.0.0.1', port), Handler) as server:
        server.timeout = 10
        print(json.dumps({'url': f'http://127.0.0.1:{server.server_port}/'}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return {'status': 'closed'}
