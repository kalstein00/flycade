"""Loopback-only read-only live viewer. Polling reads one atomically replaced bundle."""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


class LiveServer(HTTPServer):
    allow_reuse_address = True


def serve_live(run: Path, port: int = 8766, additional_runs: list[Path] | None = None) -> dict[str, Any]:
    paths = [run, *(additional_runs or [])]
    if len(paths) > 32:
        raise ValueError('A viewer accepts at most 32 explicit Runs')
    registry = {json.loads((path / 'run.json').read_text())['run_id']: path for path in paths}
    manifests = {identifier: json.loads((path / 'run.json').read_text()) for identifier, path in registry.items()}
    default_id = json.loads((run / 'run.json').read_text())['run_id']

    def workers(path: Path) -> list[int]:
        return [0, *sorted(int(p.name) for p in (path / 'live' / 'workers').glob('*')
                           if p.name.isdecimal() and 0 < int(p.name) < 64 and (p / 'latest.json').is_file())]
    assets = Path(__file__).with_name('web')

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(2)

        def do_GET(self) -> None:
            request = urlsplit(self.path)
            if request.path == '/api/catalog':
                body = json.dumps({'default_run': default_id, 'runs': [
                    {'run_id': identifier, 'label': path.name, 'workers': workers(path)}
                    for identifier, path in registry.items()]}).encode()
                content_type = 'application/json'
            elif request.path == '/api/latest':
                query = parse_qs(request.query)
                identifier = query.get('run', [default_id])[0]
                try:
                    selected_run = registry[identifier]
                    worker = int(query.get('worker', ['0'])[0])
                    if worker not in workers(selected_run):
                        raise ValueError('Unknown worker')
                except (KeyError, ValueError):
                    self.send_error(404, 'Unknown Run or worker')
                    return
                manifest = manifests[identifier]
                path = selected_run / 'live' / 'latest.json' if worker == 0 else selected_run / 'live' / 'workers' / str(worker) / 'latest.json'
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
                from flycade.control import run_status
                try:
                    envelope['control'] = run_status(selected_run)
                except (OSError, ValueError, KeyError):
                    envelope['control'] = None
                from flycade.budget import budget_info
                try:
                    envelope['run_info'] = {'run_id': manifest['run_id'],
                        'lineage': manifest.get('lineage'), 'budget': budget_info(selected_run, manifest)}
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    envelope['run_info'] = {'run_id': manifest['run_id'],
                        'lineage': manifest.get('lineage'), 'error': str(exc)}
                envelope['selected_worker'] = worker
                body = json.dumps(envelope, allow_nan=False).encode()
                content_type = 'application/json'
            elif self.path in ('/', '/live.js', '/live.css', '/inspector.js'):
                filename = 'live.html' if self.path == '/' else self.path[1:]
                body = (assets / filename).read_bytes()
                content_type = {'live.html': 'text/html', 'live.js': 'text/javascript', 'live.css': 'text/css', 'inspector.js': 'text/javascript'}[filename]
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
        print(json.dumps({'url': f'http://localhost:{server.server_port}', 'run_id': default_id}), flush=True)
        server.serve_forever()
    return {'status': 'closed'}
