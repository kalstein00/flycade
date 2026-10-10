#!/usr/bin/env python3
"""One-hour local acceptance runner through CLI, files and a real browser."""
import argparse
import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Any, TextIO

from measure_resources import resources


def main() -> None:
    from playwright.sync_api import expect, sync_playwright
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=3605)
    args = parser.parse_args()
    if args.seconds < 3600:
        parser.error('Acceptance requires at least 3600 seconds')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = output / 'run'
    if run.exists():
        parser.error('Use a fresh output directory; an existing Run will not be overwritten')
    prefix = [sys.executable, '-m', 'flycade']
    host = subprocess.run([*prefix, 'diagnose'], capture_output=True, text=True, check=True)
    (output / 'host.json').write_text(host.stdout)
    handles: list[TextIO] = []
    processes: list[subprocess.Popen[str]] = []

    def launch(arguments: list[str | Path], label: str) -> subprocess.Popen[str]:
        stdout = (output / (label + '.stdout')).open('w')
        stderr = (output / (label + '.stderr')).open('w')
        handles.extend([stdout, stderr])
        process = subprocess.Popen([*prefix, *map(str, arguments)], stdout=stdout, stderr=stderr, text=True)
        processes.append(process)
        return process

    def read(path: Path) -> dict[str, Any]:
        # Progress reports are diagnostic files written in place, unlike checkpoint pointers.
        for attempt in range(50):
            try:
                value: dict[str, Any] = json.loads(path.read_text())
                return value
            except (FileNotFoundError, json.JSONDecodeError):
                if attempt == 49:
                    raise
                time.sleep(.02)
        raise RuntimeError(f'Could not read progress: {path}')

    (output / 'boot-id-before.txt').write_text(Path('/proc/sys/kernel/random/boot_id').read_text())
    events: list[dict[str, Any]] = []
    timings: dict[int, float] = {}
    samples: list[dict[str, Any]] = []
    sampler_stop = threading.Event()
    sampler_ready = threading.Event()
    sampler_errors: list[str] = []
    sampling_started = time.monotonic()
    phase = {'name': 'initialization'}

    def sample_loop() -> None:
        next_windows = 0.
        try:
            while not sampler_stop.is_set():
                tick = time.monotonic()
                row = resources()
                row.update(elapsed=tick-sampling_started, phase=phase['name'])
                if (run / 'report.json').exists():
                    progress = read(run / 'report.json')
                    row.update(updates=progress['updates'], transitions=progress['transitions'],
                               training_seconds=progress.get('training_seconds', 0.))
                if (run / 'control.json').exists():
                    row['evaluating'] = read(run / 'control.json').get('training_paused', False)
                memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
                row['linux_available_bytes'] = int(memory['MemAvailable'].split()[0]) * 1024
                gpu = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                row['gpu_used_bytes'] = int(gpu.stdout.strip()) * 1024 * 1024
                if tick >= next_windows:
                    windows = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                        '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'], capture_output=True, text=True, timeout=15)
                    row['windows_available_bytes'] = int(windows.stdout.strip()) * 1024 if windows.returncode == 0 and windows.stdout.strip().isdigit() else None
                    next_windows = tick + 60
                samples.append(row)
                with (output / 'resources.jsonl').open('a') as log:
                    log.write(json.dumps(row) + '\n')
                sampler_ready.set()
                sampler_stop.wait(max(0., 1 - (time.monotonic()-tick)))
        except Exception as exc:
            sampler_errors.append(str(exc))
            sampler_ready.set()

    sampler = threading.Thread(target=sample_loop, name='acceptance-resources', daemon=True)
    trainer: subprocess.Popen[str] | None = None
    viewer = None
    completed = False
    sampler.start()
    try:
        if not sampler_ready.wait(30) or sampler_errors:
            raise RuntimeError(f'Resource sampler did not start: {sampler_errors}')
        trainer = launch(['train', '--graph', '.flycade/graphs/visual-a2-final-001', '--output', run,
                          '--training-config', 'configs/training-small.json',
                          '--initial-evaluation-config', 'configs/evaluation-small.json'], 'train')
        deadline = time.monotonic() + 120
        while (not (run / 'live/latest.json').exists() or not (run / 'report.json').exists()
               or read(run / 'report.json').get('updates', 0) == 0):
            if trainer.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Trainer did not begin observations; inspect train.stderr')
            time.sleep(.25)
        viewer_errors = (output / 'viewer.stderr').open('w')
        handles.append(viewer_errors)
        viewer = subprocess.Popen([*prefix, 'live', str(run), '--port', '0'], stdout=subprocess.PIPE,
                                  stderr=viewer_errors, text=True)
        processes.append(viewer)
        assert viewer.stdout is not None
        url = json.loads(viewer.stdout.readline())['url']
        (output / 'viewer.json').write_text(json.dumps({'url': url, 'pid': viewer.pid}))
        started = time.monotonic()
        started_unix = time.time()
        phase['name'] = 'training_and_observation'
        next_browser = next_timings = 0.
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={'width': 1920, 'height': 1080})
            page.goto(url)
            expect(page.locator('#circuit')).to_be_visible(timeout=15000)
            while time.monotonic() - started < args.seconds:
                elapsed = time.monotonic() - started
                if trainer.poll() is not None:
                    raise RuntimeError('Trainer stopped before the acceptance duration')
                if sampler_errors:
                    raise RuntimeError(f'Resource sampler failed: {sampler_errors}')
                row = samples[-1]
                report = read(run / 'report.json')
                if elapsed >= next_timings:
                    for line in (run / 'updates.jsonl').read_text().splitlines():
                        try:
                            update = json.loads(line)
                        except ValueError:
                            continue
                        timings[update['updates']] = update['training_seconds']
                    next_timings = elapsed + 20
                if elapsed >= next_browser:
                    before = report['updates']
                    page.goto(url)
                    expect(page.locator('#circuit')).to_be_visible(timeout=15000)
                    node = page.locator('#circuit [data-node]').first
                    node.focus(); page.keyboard.press('Enter')
                    expect(page.locator('#selected-value')).to_be_visible()
                    page.locator('#node-filter').select_option('neighbors')
                    page.locator('#zoom-in').click()
                    if elapsed > 0:
                        page.close()
                        page = browser.new_page(viewport={'width': 1920, 'height': 1080})
                        page.goto(url)
                        expect(page.locator('#circuit')).to_be_visible(timeout=15000)
                    page.locator('#storage-info summary').click()
                    if len(events) in (0, 6, 11):
                        page.screenshot(path=str(output / f'live-{len(events)}.png'), full_page=True)
                    catalog = read(run / 'evaluation-index.json')
                    if catalog['protocols']:
                        initial = catalog['protocols'][0]['initial']
                        page.goto(url + '/compare')
                        expect(page.locator('#left-identity')).to_contain_text('initial')
                        page.locator('#right-video').evaluate('(v)=>v.play()')
                        page.wait_for_function('()=>document.querySelector("#right-video").currentTime>0')
                        page.goto(url + '/replay?evaluation=' + initial)
                        expect(page.locator('#circuit')).to_be_visible()
                        page.locator('#replay-video').evaluate('(v)=>v.play()')
                        page.wait_for_function('()=>document.querySelector("#replay-video").currentTime>.2')
                        page.goto(url)
                    events.append({'elapsed': elapsed, 'updates_before': before,
                        'updates_after': read(run / 'report.json')['updates'], 'node_inspection': True,
                        'reconnected': elapsed > 0, 'comparison_and_replay': True})
                    (output / 'browser-events.json').write_text(json.dumps(events, indent=2))
                    next_browser = elapsed + 300
                (output / 'progress.json').write_text(json.dumps({'elapsed_seconds': elapsed,
                    'required_seconds': args.seconds, 'updates': report['updates'], 'rss': row['rss'],
                    'gpu_used_bytes': row['gpu_used_bytes'], 'evaluating': row.get('evaluating', False)}))
                time.sleep(1)
            phase['name'] = 'saving_and_shutdown'
            stop_started = time.monotonic()
            stopped = subprocess.run([*prefix, 'save', str(run), '--stop', '--wait-seconds', '150'],
                                     capture_output=True, text=True, timeout=160, check=True)
            trainer.wait(timeout=30)
            if trainer.returncode:
                raise RuntimeError('Trainer failed during final save')
            save_stop_seconds = time.monotonic() - stop_started
            (output / 'stop.json').write_text(stopped.stdout)
            page.goto(url)
            expect(page.locator('#operation')).to_contain_text('종료 완료')
            page.screenshot(path=str(output / 'saved.png'), full_page=True)
            browser.close()
        phase['name'] = 'saved'
        time.sleep(1.1)
        sampler_stop.set(); sampler.join(timeout=30)
        if sampler.is_alive() or sampler_errors:
            raise RuntimeError(f'Resource sampler failed: {sampler_errors}')
        final = read(run / 'report.json')
        for line in (run / 'updates.jsonl').read_text().splitlines():
            update = json.loads(line)
            timings[update['updates']] = update['training_seconds']
        deltas = sorted(timings[key] - timings[key-1] for key in timings if key-1 in timings)
        p95 = deltas[min(len(deltas)-1, int(len(deltas)*.95))]
        windows_samples = [row['windows_available_bytes'] for row in samples if row.get('windows_available_bytes') is not None]
        intervals = sorted(b['unix']-a['unix'] for a,b in zip(samples,samples[1:]))
        summary = {'sampling_interval_max_seconds': max(intervals),
            'sampling_interval_p95_seconds': intervals[int(len(intervals)*.95)],
            'actual_seconds': time.monotonic()-started, 'started_unix': started_unix,
            'final': final, 'resource_samples': len(samples), 'update_intervals': len(deltas),
            'p95_update_seconds': p95, 'save_stop_seconds': save_stop_seconds,
            'peak_tree_rss_bytes': max(row['rss'] for row in samples),
            'peak_gpu_used_bytes': max(row['gpu_used_bytes'] for row in samples),
            'min_linux_available_bytes': min(row['linux_available_bytes'] for row in samples),
            'min_windows_available_bytes': min(windows_samples) if windows_samples else None,
            'max_tree_swap_bytes': max(row['swap'] for row in samples),
            'swap_in_delta': samples[-1]['pswpin']-samples[0]['pswpin'],
            'swap_out_delta': samples[-1]['pswpout']-samples[0]['pswpout'],
            'browser_checks': events,
            'evaluation_catalog': read(run / 'evaluation-index.json'),
            'manual_restart': 'pending; no WSL/PC restart performed by this script',
            'windows_browser': 'pending; Playwright Chromium in WSL is not a Windows screen acceptance'}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({key: value for key, value in summary.items() if key not in ('final','evaluation_catalog','browser_checks')}, indent=2), flush=True)
        completed = True
    finally:
        if not completed and trainer is not None and trainer.poll() is None:
            try:
                subprocess.run([*prefix, 'save', str(run), '--stop', '--wait-seconds', '150'], timeout=160)
            except subprocess.TimeoutExpired:
                pass
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
        sampler_stop.set(); sampler.join(timeout=30)
        for handle in handles:
            handle.close()
    if completed:
        seen = {entry['pid'] for sample in samples for entry in sample['processes']} - {os.getpid()}
        alive = []
        for pid in seen:
            stat = Path(f'/proc/{pid}/stat')
            if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z':
                alive.append(pid)
        summary['orphan_children'] = sorted(alive)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        if alive:
            raise RuntimeError(f'Unexpected surviving child processes: {alive}')


if __name__ == '__main__':
    main()
