"""Loopback-only read-only live viewer. Polling reads one atomically replaced bundle."""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


class LiveServer(HTTPServer):
    allow_reuse_address = True


def serve_live(run: Path, port: int = 8766) -> dict[str, Any]:
    manifest = json.loads((run / 'run.json').read_text())
    assets = Path(__file__).with_name('web')

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(2)

        def do_GET(self) -> None:
            if self.path == '/api/latest':
                path = run / 'live' / 'latest.json'
                try:
                    envelope = json.loads(path.read_text())
                    if envelope['run_id'] != manifest['run_id']:
                        raise ValueError('Run mismatch')
                    if envelope['status'] == 'running':
                        try:
                            os.kill(envelope['trainer_pid'], 0)
                        except ProcessLookupError:
                            envelope['status'] = 'trainer_stopped'
                except FileNotFoundError:
                    envelope = {'run_id': manifest['run_id'], 'status': 'no_data', 'sample': None}
                except (OSError, ValueError, KeyError):
                    self.send_error(503, 'Observation temporarily unavailable')
                    return
                body = json.dumps(envelope, allow_nan=False).encode()
                content_type = 'application/json'
            elif self.path in ('/', '/live.js', '/live.css'):
                filename = 'live.html' if self.path == '/' else self.path[1:]
                body = (assets / filename).read_bytes()
                content_type = {'live.html': 'text/html', 'live.js': 'text/javascript', 'live.css': 'text/css'}[filename]
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with LiveServer(('127.0.0.1', port), Handler) as server:
        print(json.dumps({'url': f'http://localhost:{server.server_port}', 'run_id': manifest['run_id']}), flush=True)
        server.serve_forever()
    return {'status': 'closed'}
