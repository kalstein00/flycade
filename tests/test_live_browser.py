"""Live GUI against the real local HTTP service and controlled CPU policy artifacts."""
import json
import select
import subprocess
import sys

from playwright.sync_api import expect, sync_playwright

from test_graph_cli import cli
from test_training_cli import prepared_graph


def start_service(run, port=0):
    service = subprocess.Popen([sys.executable, '-m', 'flycade', 'live', str(run), '--port', str(port)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert select.select([service.stdout], [], [], 15)[0], 'live service did not start'
    line = service.stdout.readline()
    assert line, service.stderr.read()
    response = json.loads(line)
    assert 'url' in response, response
    return service, response['url']


def test_browser_renders_one_coherent_sample_and_desktop_layout(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', run, '--updates', 1, '--rollout-steps', 32)
    assert result.returncode == 0, result.stdout + result.stderr
    envelope = json.loads((run / 'live' / 'latest.json').read_text())
    sample = envelope['sample']
    before = (run / 'final.pt').read_bytes()
    service, url = start_service(run)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1280, 'height': 720})
            page.goto(url)
            page.wait_for_selector('[data-sample-id]')
            assert page.locator('#sample-id').text_content() == sample['sample_id']
            assert page.locator('#raw').get_attribute('src') == sample['raw']
            assert page.locator('#inputs img').count() == 4
            assert page.locator('#chosen').inner_text() == ' + '.join(sample['buttons'] or ['NOOP'])
            assert page.locator('#argmax').inner_text() == str(sample['argmax'])
            assert page.locator('#reward').inner_text() == str(sample['transition']['reward']).removesuffix('.0')
            assert page.locator('#status').inner_text() == '학습 종료 · 저장 완료'
            for index, probability in enumerate(sample['probabilities']):
                assert page.locator(f'[data-action="{index}"] meter').get_attribute('value') == str(probability)
            assert page.locator('#circuit [data-node]').count() == len(envelope['graph']['nodes'])
            for index, node in enumerate(envelope['graph']['nodes']):
                actual = float(page.locator(f'[data-node="{node["index"]}"]').get_attribute('data-mean'))
                assert abs(actual - sum(sample['activity'][index]) / len(sample['activity'][index])) < 1e-7
            for width, height in [(1280, 720), (1920, 1080), (390, 844)]:
                page.set_viewport_size({'width': width, 'height': height})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                for target in ('#raw', '#circuit', '#chosen', '#metrics', '#status'):
                    assert page.locator(target).is_visible()
            browser.close()
        assert (run / 'final.pt').read_bytes() == before
    finally:
        service.terminate()
        service.communicate(timeout=10)


def publish(run, envelope):
    path = run / 'live' / 'latest.json'
    temporary = path.with_suffix('.test')
    temporary.write_text(json.dumps(envelope))
    temporary.replace(path)


def test_reconnect_rejects_old_sessions_and_slow_consumer_skips_backlog(tmp_path):
    import copy
    import time
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', run, '--updates', 1, '--rollout-steps', 8)
    assert result.returncode == 0, result.stdout + result.stderr
    original = json.loads((run / 'live' / 'latest.json').read_text())
    (run / 'live' / 'latest.json').unlink()
    service, url = start_service(run)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.get_by_text('관측 데이터 없음', exact=True).wait_for()
            publish(run, original)
            page.wait_for_selector('[data-sample-id]')
            original_id = page.locator('#sample-id').text_content()
            port = int(url.rsplit(':', 1)[1])
            service.terminate()
            service.communicate(timeout=10)
            page.get_by_text('연결 끊김 · 재연결 중', exact=True).wait_for()
            assert page.locator('#sample-id').text_content() == original_id
            # The fixture producer advances while the consumer/service is absent.
            latest = copy.deepcopy(original)
            latest['generation'] += 1
            latest['session_id'] = 'new-session'
            latest['sample']['session_id'] = 'new-session'
            for sequence in range(1, 21):
                latest['sequence'] = sequence
                latest['updated_unix'] = time.time()
                latest['sample'].update(sample_id=f'new-session:{sequence}', step=sequence, policy_version=2)
                publish(run, latest)
            service, _ = start_service(run, port)
            page.get_by_text('new-session:20', exact=True).wait_for(state='attached')
            assert page.locator('[data-sample-id]').count() == 1
            # Old-session packets and backwards policy versions cannot overwrite the visible bundle.
            stale_packets = [original, copy.deepcopy(latest), copy.deepcopy(latest)]
            stale_packets[1]['sequence'] = 21
            stale_packets[1]['sample']['policy_version'] = 1
            stale_packets[2]['sequence'] = 21
            stale_packets[2]['sample']['worker'] = 1
            for stale in stale_packets:
                stale['updated_unix'] = time.time() + 1
                publish(run, stale)
                page.wait_for_timeout(450)
                assert page.locator('#sample-id').text_content() == 'new-session:20'
            # Same generation with a lower sequence is rejected, even with a newer timestamp.
            stale = copy.deepcopy(latest)
            stale['sequence'] = 19
            stale['sample']['sample_id'] = 'new-session:19'
            stale['updated_unix'] = time.time() + 2
            publish(run, stale)
            page.wait_for_timeout(450)
            assert page.locator('#sample-id').text_content() == 'new-session:20'
            publish(run, latest)
            page.reload()
            page.get_by_text('new-session:20', exact=True).wait_for(state='attached')
            disabled = copy.deepcopy(latest)
            disabled.update(generation=latest['generation'] + 1, session_id='disabled-session',
                            sequence=0, sample=None, status='disabled', updated_unix=time.time() + 3)
            publish(run, disabled)
            page.get_by_text('관측 꺼짐 · 데이터 없음', exact=True).wait_for()
            # A temporary missing-file response cannot erase the accepted session generation.
            (run / 'live' / 'latest.json').unlink()
            page.wait_for_timeout(450)
            publish(run, original)
            page.wait_for_timeout(450)
            assert page.locator('#raw').count() == 0
            assert page.locator('#status').inner_text() == '관측 꺼짐 · 데이터 없음'
            browser.close()
    finally:
        if service.poll() is None:
            service.terminate()
            service.communicate(timeout=10)


def test_active_browser_close_does_not_stop_or_change_training(tmp_path):
    import time
    import torch
    graph = prepared_graph(tmp_path)
    runs = [tmp_path / 'observed', tmp_path / 'unobserved']
    args = ['--fixture', '--device', 'cpu', '--graph', str(graph), '--updates', '128', '--rollout-steps', '16']
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        trainer = subprocess.Popen([sys.executable, '-m', 'flycade', 'train', *args,
                                    '--output', str(runs[0]), '--observe-hz', '5'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        service = None
        try:
            deadline = time.monotonic() + 20
            while not (runs[0] / 'live' / 'latest.json').exists() and trainer.poll() is None:
                assert time.monotonic() < deadline
                time.sleep(.05)
            service, url = start_service(runs[0])
            page = browser.new_page()
            page.goto(url)
            page.wait_for_selector('[data-sample-id]')
            assert trainer.poll() is None
            first = json.loads((runs[0] / 'live' / 'latest.json').read_text())
            before = first['sample']['step']
            initial_bytes = (runs[0] / 'live' / 'latest.json').stat().st_size
            page.context.set_offline(True)
            page.get_by_text('연결 끊김 · 재연결 중', exact=True).wait_for()
            frozen = page.locator('#sample-id').text_content()
            page.wait_for_timeout(1000)
            produced = json.loads((runs[0] / 'live' / 'latest.json').read_text())
            assert trainer.poll() is None
            assert produced['sample']['step'] > before
            assert produced['sequence'] > first['sequence']
            assert produced['observer']['queue_capacity'] == 1
            assert produced['observer']['history_capacity'] == 0
            assert len(list((runs[0] / 'live').glob('*.json'))) == 1
            assert (runs[0] / 'live' / 'latest.json').stat().st_size < initial_bytes * 2
            assert page.locator('#sample-id').text_content() == frozen
            page.context.set_offline(False)
            page.wait_for_function('(minimum) => Number(document.querySelector("#sample-id").textContent.split(":").at(-1)) >= minimum', arg=produced['sequence'])
            assert page.locator('[data-sample-id]').count() == 1
            browser.close()
            stdout, stderr = trainer.communicate(timeout=60)
            assert trainer.returncode == 0, stdout + stderr
            assert json.loads(stdout)['transitions'] > before
        finally:
            if trainer.poll() is None:
                trainer.terminate()
                trainer.communicate(timeout=10)
            if service is not None:
                service.terminate()
                service.communicate(timeout=10)
    result = cli('train', *args, '--output', runs[1], '--observe-hz', 0)
    assert result.returncode == 0, result.stdout + result.stderr
    on, off = [torch.load(run / 'final.pt', weights_only=False) for run in runs]
    for key in ('torch_rng',):
        assert torch.equal(on[key], off[key])
    assert on['python_rng'] == off['python_rng']
    assert on['environment_rng'] == off['environment_rng']
    for name, parameter in on['model'].items():
        other = off['model'][name]
        assert torch.equal(parameter.to_dense() if parameter.is_sparse else parameter,
                           other.to_dense() if other.is_sparse else other)


def test_browser_tracks_real_manual_save_stop_and_resume(tmp_path):
    from playwright.sync_api import expect
    from test_daily_cli import wait_report
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'daily-browser'
    trainer = subprocess.Popen([sys.executable, '-m', 'flycade', 'train', '--fixture', '--device', 'cpu',
        '--graph', str(graph), '--output', str(run), '--updates', '10000', '--rollout-steps', '64'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    service = None
    try:
        wait_report(run, lambda r: r['updates'] > 0, trainer)
        service, url = start_service(run)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1280, 'height': 720})
            page.goto(url)
            expect(page.locator('#operation')).to_contain_text('저장 전')
            request = cli('save', run, '--wait-seconds', 0)
            assert request.returncode == 0, request.stdout + request.stderr
            expect(page.locator('#operation')).to_contain_text('저장 완료', timeout=20000)
            assert trainer.poll() is None
            assert 'update' in page.locator('#recovery').inner_text()
            stopped = cli('save', run, '--stop')
            assert stopped.returncode == 0, stopped.stdout + stopped.stderr
            trainer.communicate(timeout=30)
            expect(page.locator('#operation')).to_contain_text('종료 완료')
            result = cli('resume', run, '--stop-after-updates', 1)
            assert result.returncode == 0, result.stdout + result.stderr
            expect(page.locator('#reset-notice')).to_contain_text('새 에피소드')
            browser.close()
    finally:
        if trainer.poll() is None:
            trainer.kill()
            trainer.communicate(timeout=10)
        if service is not None:
            service.terminate()
            service.communicate(timeout=10)


def test_stop_cleanup_is_visible_until_encoder_finishes(tmp_path):
    import os
    import shutil
    from pathlib import Path
    from playwright.sync_api import expect
    from test_daily_cli import wait_report
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'closing'
    executables = tmp_path / 'bin'
    executables.mkdir()
    (executables / 'git').symlink_to(shutil.which('git'))
    encoder = executables / 'ffmpeg'
    encoder.write_text(f'#!{sys.executable}\nimport sys,time\nsys.stdin.buffer.read()\ntime.sleep(3)\nsys.exit(1)\n')
    encoder.chmod(0o755)
    trainer = subprocess.Popen([sys.executable, '-m', 'flycade', 'train', '--fixture', '--device', 'cpu',
        '--graph', str(graph), '--output', str(run), '--updates', '10000', '--rollout-steps', '16'],
        env={**os.environ, 'PATH': str(executables)}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    service = None
    try:
        wait_report(run, lambda r: r['updates'] > 0, trainer)
        service, url = start_service(run)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1280, 'height': 720})
            page.goto(url)
            page.wait_for_selector('[data-sample-id]')
            result = cli('save', run, '--stop', '--wait-seconds', 0)
            assert result.returncode == 0, result.stdout + result.stderr
            expect(page.locator('#operation')).to_contain_text('종료 정리 중', timeout=10000)
            expect(page.locator('#operation')).to_contain_text('대기')
            expect(page.locator('#status')).to_have_text('저장 완료 · 종료 정리 중')
            screenshots = os.environ.get('FLYCADE_SCREENSHOTS')
            if screenshots:
                path = Path(screenshots)
                path.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(path / 'closing-fixture.png'), full_page=True)
            trainer.communicate(timeout=30)
            expect(page.locator('#operation')).to_contain_text('종료 완료')
            browser.close()
    finally:
        if trainer.poll() is None:
            trainer.kill()
            trainer.communicate(timeout=10)
        if service is not None:
            service.terminate()
            service.communicate(timeout=10)


def test_browser_shows_recovered_checkpoint_and_rollback(tmp_path):
    from playwright.sync_api import expect
    from test_recovery_cli import saved_run
    run, saves = saved_run(tmp_path)
    (run / saves[-1]['file']).write_bytes(b'corrupt')
    service, url = start_service(run)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1280, 'height': 720})
            page.goto(url)
            result = cli('resume', run, '--stop-after-updates', 1)
            assert result.returncode == 0, result.stdout + result.stderr
            expect(page.locator('#rollback')).to_contain_text('1 update · 4전이')
            expect(page.locator('#rollback')).to_contain_text('복구 완료')
            page.locator('#recovery-details summary').click()
            expect(page.locator('#recovery-errors')).to_contain_text('checksum')
            for path in (run / 'checkpoints').glob('*.pt'):
                path.write_bytes(b'broken')
            result = cli('resume', run)
            assert result.returncode == 2
            expect(page.locator('#rollback')).to_contain_text('복구 실패')
            expect(page.locator('#status')).to_contain_text('복구 실패')
            browser.close()
    finally:
        service.terminate()
        service.communicate(timeout=10)


def test_browser_identifies_full_branch_warm_start_and_budget_events(tmp_path):
    from test_recovery_cli import saved_run
    parent, saves = saved_run(tmp_path, 1)
    branch = tmp_path / 'branch'
    warm = tmp_path / 'warm'
    result = cli('branch', parent, '--checkpoint', saves[0]['checkpoint_id'], '--output', branch)
    assert result.returncode == 0, result.stdout + result.stderr
    config = tmp_path / 'warm.json'
    config.write_text(json.dumps({'updates': 1, 'rollout_steps': 4}))
    result = cli('warm-start', parent, '--checkpoint', saves[0]['checkpoint_id'], '--output', warm,
                 '--training-config', config)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cli('extend-budget', branch, '--updates', 25).returncode == 0
    parent_id = json.loads((parent / 'run.json').read_text())['run_id']
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for run, label in [(branch, '전체 상태 분기'), (warm, '가중치 재사용 · 새 실험')]:
            service, url = start_service(run)
            try:
                page = browser.new_page()
                page.goto(url)
                expect(page.locator('#run-kind')).to_have_text(label)
                assert page.locator('#run-kind').inner_text() == label
                page.locator('#run-info summary').click()
                assert parent_id in page.locator('#lineage').inner_text()
                assert saves[0]['checkpoint_id'] in page.locator('#lineage').inner_text()
                if run == branch:
                    assert '25' in page.locator('#budget-info').inner_text()
                    assert '20 → 25' in page.locator('#budget-events').inner_text()
                for width in (1280, 1920, 390):
                    page.set_viewport_size({'width': width, 'height': 844})
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.close()
            finally:
                service.terminate()
                service.wait(timeout=5)
        browser.close()
