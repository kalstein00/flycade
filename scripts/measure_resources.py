#!/usr/bin/env python3
"""Reproducible four-phase A5 experiment using only public CLI/browser boundaries."""
import argparse
import json
import math
import threading
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from typing import Any


def children(pid: int) -> list[int]:
    try:
        direct = [int(p) for p in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except OSError:
        return []
    return direct + [child for parent in direct for child in children(parent)]


def resources() -> dict[str, Any]:
    processes = []
    for pid in [os.getpid(), *children(os.getpid())]:
        try:
            fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
            stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            processes.append({'pid': pid, 'name': fields['Name'].strip(),
                'rss': int(fields.get('VmRSS', '0').split()[0]) * 1024,
                'swap': int(fields.get('VmSwap', '0').split()[0]) * 1024,
                'major_faults': int(stat[9])})
        except (OSError, ValueError, KeyError):
            continue
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return {'unix': time.time(), 'processes': processes, 'rss': sum(p['rss'] for p in processes),
            'swap': sum(p['swap'] for p in processes),
            'pswpin': int(vm['pswpin']), 'pswpout': int(vm['pswpout'])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--home', type=Path, default=Path('.flycade'))
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--phase-seconds', type=float, default=150)
    args = parser.parse_args()
    if not math.isfinite(args.phase_seconds) or args.phase_seconds <= 0:
        parser.error('phase-seconds must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.output / 'run'
    prefix = [sys.executable, '-m', 'flycade']
    processes: list[subprocess.Popen[str]] = []
    handles: list[Any] = []
    samples: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    evaluation_config = args.output / 'evaluation.json'
    evaluation_config.write_text(json.dumps({'seeds': [11] if args.fixture else [11,22,33],
        'max_frames': 120 if args.fixture else 1800, 'video_seconds': 2 if args.fixture else 15}))
    host = subprocess.run([*prefix, 'diagnose'], capture_output=True, text=True, check=True)
    host_data = json.loads(host.stdout)
    if host_data['wsl']:
        windows = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
            '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'], capture_output=True, text=True, timeout=15)
        host_data['memory']['windows_available_bytes'] = int(windows.stdout.strip()) * 1024 if windows.returncode == 0 and windows.stdout.strip().isdigit() else None
    (args.output / 'host.json').write_text(json.dumps(host_data, indent=2))

    def launch(arguments: list[str], label: str) -> subprocess.Popen[str]:
        stdout = (args.output / f'{label}.stdout.json').open('w')
        stderr = (args.output / f'{label}.stderr.log').open('w')
        handles.extend((stdout, stderr))
        process = subprocess.Popen([*prefix, *arguments], stdout=stdout, stderr=stderr, text=True)
        processes.append(process)
        return process

    def sample() -> None:
        row = resources()
        if not args.fixture:
            gpu = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi',
                '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10)
            row['gpu_mib'] = gpu.stdout.strip()
        samples.append(row)
        with (args.output / 'resources.jsonl').open('a') as log:
            log.write(json.dumps(row) + '\n')

    def wait(process: subprocess.Popen[str], timeout: float = 90) -> None:
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if time.monotonic() > deadline:
                raise RuntimeError('Process failed to finish in bounded time')
            time.sleep(.1 if args.fixture else 1)
        if process.returncode:
            raise RuntimeError(f'CLI process {process.pid} failed with {process.returncode}; inspect stderr artifacts')

    def evaluate(label: str) -> subprocess.Popen[str]:
        return launch(['evaluate', str(run), '--snapshot', 'latest', '--protocol',
            str(run / 'initial-evaluation-protocol.json'), '--home', str(args.home), '--device', 'cpu'], label)

    sampler_stop = threading.Event()
    sampling_errors: list[str] = []

    def monitor() -> None:
        while not sampler_stop.is_set():
            try:
                sample()
            except Exception as exc:
                sampling_errors.append(str(exc))
                return
            sampler_stop.wait(.1 if args.fixture else 1)

    sample()  # Baseline before initial evaluation or any trainer/restore allocation.
    sampler = threading.Thread(target=monitor, name='resource-sampler', daemon=True)
    sampler.start()
    browser = playwright = service = None
    started = time.monotonic()
    try:
        for index, mode in enumerate(('off', 'live', 'concurrent_evaluation', 'sequential_evaluation')):
            before = json.loads((run / 'report.json').read_text()) if index else {'updates': 0, 'transitions': 0}
            evaluation = None
            if mode == 'sequential_evaluation':
                evaluation = evaluate('sequential')
                wait(evaluation)
                evaluations.append(json.loads((args.output / 'sequential.stdout.json').read_text()))
            arguments = ['resume', str(run), '--home', str(args.home)] if index else [
                'train', '--graph', str(args.graph), '--output', str(run), '--home', str(args.home),
                '--device', 'cpu' if args.fixture else 'cuda', '--updates', '1000000', '--rollout-steps', '32',
                '--initial-evaluation-config', str(evaluation_config)] + (['--fixture'] if args.fixture else [])
            arguments += ['--observe-hz', '0' if mode == 'off' else '3']
            trainer = launch(arguments, f'phase-{index}')
            deadline = time.monotonic() + 90
            while True:
                if trainer.poll() is not None:
                    raise RuntimeError(f'Trainer ended early: see phase-{index}.stderr.log')
                try:
                    current = json.loads((run / 'report.json').read_text())
                    if current['status'] == 'running' and current['updates'] > before['updates']:
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Trainer did not reach first update')
                time.sleep(.1)
            browser = playwright = service = None
            if mode != 'off':
                from playwright.sync_api import sync_playwright
                service = subprocess.Popen([*prefix, 'live', str(run), '--port', '0'],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                processes.append(service)
                assert service.stdout is not None
                if not select.select([service.stdout], [], [], 15)[0]:
                    raise RuntimeError('Live server did not start')
                url = json.loads(service.stdout.readline())['url']
                playwright = sync_playwright().start()
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={'width': 1280, 'height': 720})
                page.goto(url)
                page.wait_for_selector('[data-sample-id]')
            if mode == 'concurrent_evaluation':
                evaluation = evaluate('concurrent')
            phase_started = time.monotonic()
            print(f'{mode}: measuring {args.phase_seconds}s', flush=True)
            while time.monotonic() - phase_started < args.phase_seconds:
                if trainer.poll() is not None:
                    raise RuntimeError('Trainer failed during measurement')
                if sampling_errors:
                    raise RuntimeError(sampling_errors[0])
                if browser is not None:
                    page.wait_for_timeout(100 if args.fixture else 1000)
                else:
                    time.sleep(.1 if args.fixture else 1)
            requested = time.monotonic()
            trainer.send_signal(signal.SIGINT)
            wait(trainer)
            end = time.monotonic()
            after = json.loads((run / 'report.json').read_text())
            phases.append({'mode': mode, 'session_id': after['session_id'], 'status': after['status'],
                'updates': after['updates'], 'new_updates': after['updates'] - before['updates'],
                'new_transitions': after['transitions'] - before['transitions'],
                'measured_seconds': requested - phase_started, 'stop_save_exit_seconds': end - requested,
                'training_seconds': after['training_seconds'] - before.get('training_seconds', 0),
                'peak_rss_bytes': after['peak_rss_bytes'], 'peak_cuda_allocated_bytes': after['peak_cuda_allocated_bytes'],
                'observer': after.get('observer'), 'checkpoint_id': after['checkpoint_id']})
            if evaluation is not None and mode == 'concurrent_evaluation':
                wait(evaluation)
                evaluations.append(json.loads((args.output / 'concurrent.stdout.json').read_text()))
            if browser is not None:
                page.screenshot(path=str(args.output / f'{mode}.png'), full_page=True)
                browser.close()
                browser = None
            if playwright is not None:
                playwright.stop()
                playwright = None
            if service is not None:
                service.terminate()
                service.wait(timeout=10)
            print(json.dumps(phases[-1]), flush=True)
    finally:
        sampler_stop.set()
        sampler.join(timeout=12)
        owned = children(os.getpid())
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        for pid in owned:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 2
        while any(Path(f'/proc/{pid}').exists() for pid in owned) and time.monotonic() < deadline:
            time.sleep(.05)
        for pid in owned:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for handle in handles:
            handle.close()
    elapsed = time.monotonic() - started
    if sampling_errors or sampler.is_alive():
        raise RuntimeError(f'Resource sampling failed: {sampling_errors}')
    report = {'evidence': 'synthetic CPU fixture' if args.fixture else 'real WSL / NES / CUDA',
        'elapsed_seconds': elapsed, 'a5_duration_met': not args.fixture and sum(p['measured_seconds'] for p in phases) >= 600,
        'phases': phases, 'evaluations': evaluations,
        'peak_process_tree_rss_bytes': max(row['rss'] for row in samples),
        'peak_process_tree_swap_bytes': max(row['swap'] for row in samples),
        'system_swap_in_pages': samples[-1]['pswpin'] - samples[0]['pswpin'],
        'system_swap_out_pages': samples[-1]['pswpout'] - samples[0]['pswpout'],
        'orphan_children': children(os.getpid())}
    (args.output / 'measurement.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
