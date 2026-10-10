"""Fixed-protocol comparison and recorded video selection over the local HTTP service."""
import json

from playwright.sync_api import expect, sync_playwright

from test_graph_cli import cli
from test_live_browser import start_service
from test_training_cli import prepared_graph


def test_compare_initial_latest_best_without_mixing_live_circuit(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    config = tmp_path / 'training.json'
    config.write_text(json.dumps({'updates': 2, 'rollout_steps': 4, 'evaluation_every_updates': 1}))
    protocol = tmp_path / 'evaluation.json'
    protocol.write_text(json.dumps({'seeds': [11,22,33], 'max_frames': 120, 'video_seconds': 1}))
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                 '--training-config', config, '--initial-evaluation-config', protocol)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(cli('evaluations', run).stdout)
    group = summary['protocols'][0]
    before = (run / 'final.pt').read_bytes()
    service, url = start_service(run)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={'width': 1280, 'height': 720})
            page.goto(url+'/compare')
            expect(page.get_by_role('heading', name='고정 정책 평가 비교')).to_be_visible()
            expect(page.locator('#left-identity')).to_contain_text('initial')
            page.locator('#right-choice').select_option('latest')
            expect(page.locator('#right-identity')).to_contain_text(group['latest_snapshot'])
            assert page.locator('#right-episodes tbody tr').count() == 3
            assert page.locator('#right-completion').inner_text().endswith('/ 3')
            assert page.locator('#circuit').count() == 0
            expect(page.locator('#right-circuit')).to_have_text('회로 관측 데이터 없음')
            video = page.locator('#right-video')
            video.evaluate('(v) => v.play()')
            page.wait_for_function('() => document.querySelector("#right-video").currentTime > 0')
            page.locator('#right-choice').select_option('best')
            expect(page.locator('#right-identity')).to_contain_text(group['best_snapshot'])
            for width in (1920, 1280, 390):
                page.set_viewport_size({'width': width, 'height': 844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            browser.close()
        assert (run / 'final.pt').read_bytes() == before
    finally:
        service.terminate()
        service.communicate(timeout=10)
