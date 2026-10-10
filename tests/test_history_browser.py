"""Recording selection and playback through a browser connected to the actual local service."""
import json
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

from test_graph_cli import cli
from test_training_cli import prepared_graph


def test_user_selects_recordings_and_plays_without_mutating_training(tmp_path):
    graph = prepared_graph(tmp_path)
    output = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', output, '--updates', 2, '--rollout-steps', 32, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    result = cli('resume', output)
    assert result.returncode == 0, result.stdout + result.stderr
    before = (output / 'final.pt').read_bytes()
    service = subprocess.Popen([sys.executable, '-m', 'flycade', 'history', str(output), '--serve', '--port', '0'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        import select
        assert select.select([service.stdout], [], [], 15)[0], 'history server did not start'
        line = service.stdout.readline()
        assert line, service.stderr.read()
        url = json.loads(line)['url']
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1280, 'height': 720})
            page.goto(url)
            assert page.get_by_role('heading', name='플레이 영상 내역').is_visible()
            assert page.get_by_text('학습 기록 영상', exact=True).is_visible()
            choices = page.get_by_role('button', name='재생', exact=False)
            assert choices.count() == 2
            choices.nth(0).click()
            page.locator('video').evaluate('(v) => v.play()')
            page.wait_for_function('document.querySelector("video").currentTime > 0')
            source = page.locator('video').get_attribute('src')
            choices.nth(1).click()
            assert page.locator('video').get_attribute('src') != source
            page.locator('video').evaluate('(v) => v.play()')
            page.wait_for_function('document.querySelector("video").currentTime > 0')
            page.locator('video').evaluate('(v) => {v.pause(); v.currentTime = 1;}')
            page.wait_for_function('document.querySelector("video").currentTime >= 1')
            page.reload()
            assert choices.count() == 2
            for width, height in [(1920, 1080), (390, 844)]:
                page.set_viewport_size({'width': width, 'height': height})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            browser.close()
        assert (output / 'final.pt').read_bytes() == before
    finally:
        service.terminate()
        service.communicate(timeout=10)


def test_evaluation_recording_shows_fixed_snapshot_and_protocol(tmp_path):
    graph = prepared_graph(tmp_path)
    output = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph,
                 '--output', output, '--updates', 1, '--rollout-steps', 4)
    assert result.returncode == 0, result.stdout + result.stderr
    protocol = tmp_path / 'protocol.json'
    result = cli('evaluation-protocol', output, '--output', protocol, '--seeds', '1', '--max-frames', 120)
    assert result.returncode == 0, result.stdout + result.stderr
    result = cli('evaluate', output, '--protocol', protocol)
    assert result.returncode == 0, result.stdout + result.stderr
    evaluation = json.loads(result.stdout)
    service = subprocess.Popen([sys.executable, '-m', 'flycade', 'history', str(output), '--serve', '--port', '0'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        import select
        assert select.select([service.stdout], [], [], 15)[0]
        url = json.loads(service.stdout.readline())['url']
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.get_by_role('button', name='초기 모델', exact=False).click()
            assert page.get_by_text('확률적 평가 기록 영상', exact=True).is_visible()
            assert evaluation['protocol_id'] in page.locator('#detail').inner_text()
            assert 'initial' in page.locator('#detail').inner_text()
            page.locator('video').evaluate('(v) => v.play()')
            page.wait_for_function('document.querySelector("video").currentTime > 0')
            assert evaluation['evaluation_id'] in page.locator('video').get_attribute('src')
            browser.close()
    finally:
        service.terminate()
        service.communicate(timeout=10)
