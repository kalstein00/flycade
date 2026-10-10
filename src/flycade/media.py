"""Bounded streaming of an already authorized local video."""
import re
from http.server import BaseHTTPRequestHandler
from pathlib import Path


def send_video(handler: BaseHTTPRequestHandler, path: Path) -> None:
    try:
        handle = path.open('rb')
    except OSError:
        handler.send_error(404, 'Video unavailable')
        return
    with handle:
        size = path.stat().st_size
        start, end = 0, size - 1
        requested = handler.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d+)-(\d*)', requested)
            if not match:
                handler.send_error(416)
                return
            start = int(match[1])
            end = min(int(match[2]), end) if match[2] else end
        if start > end:
            handler.send_error(416)
            return
        handler.send_response(206 if requested else 200)
        handler.send_header('Content-Type', 'video/webm')
        handler.send_header('Accept-Ranges', 'bytes')
        handler.send_header('Content-Length', str(end - start + 1))
        handler.send_header('Cache-Control', 'no-store')
        if requested:
            handler.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        handler.end_headers()
        try:
            handle.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
