"""Neuron exploration at the real HTTP/browser and public observation artifact seam."""
import json

from playwright.sync_api import expect, sync_playwright

from test_graph_cli import cli
from test_live_browser import start_service
from test_training_cli import prepared_graph


def observed_run(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                 '--updates', 1, '--rollout-steps', 32)
    assert result.returncode == 0, result.stdout + result.stderr
    return run, json.loads((run / 'live' / 'latest.json').read_text())


def test_neuron_selection_matches_real_sample_and_keyboard_filters(tmp_path):
    run, envelope = observed_run(tmp_path)
    original = (run / 'final.pt').read_bytes()
    service, url = start_service(run)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={'width': 1280, 'height': 720})
            page.goto(url)
            page.wait_for_selector('[data-sample-id]')
            node = envelope['graph']['nodes'][0]
            mark = page.locator(f'#circuit [data-node="{node["index"]}"]')
            mark.focus()
            page.keyboard.press('Enter')
            expect(page.locator('#selected-neuron')).to_contain_text(node['root_id'])
            expected = sum(envelope['sample']['activity'][0]) / len(envelope['sample']['activity'][0])
            assert abs(float(page.locator('#selected-value').get_attribute('data-value')) - expected) < 1e-7
            assert '영역: 알 수 없음' in page.locator('#selected-neuron').inner_text()
            expect(page.locator(f'[data-neuron-row="{node["index"]}"]')).to_have_attribute('aria-selected', 'true')
            page.locator('#node-filter').select_option('output')
            assert page.locator('#circuit [data-group="input"]:visible').count() == 0
            assert page.locator('#selected-neuron').inner_text().find(node['root_id']) >= 0
            expect(page.locator('#filter-counts')).to_contain_text('정책 변경 없음')
            page.locator('#zoom-in').click()
            assert page.locator('#circuit').get_attribute('viewBox') != '0 0 340 350'
            page.locator('#frame-choice').select_option('2')
            assert page.locator('#selected-input').get_attribute('src') == envelope['sample']['pixels'][2]
            for width in (1920, 1280, 390):
                page.set_viewport_size({'width': width, 'height': 844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            browser.close()
        assert (run / 'final.pt').read_bytes() == original
    finally:
        service.terminate()
        service.communicate(timeout=10)


def test_short_history_breaks_gaps_and_keeps_selected_missing_neuron(tmp_path):
    import copy
    import time
    from test_live_browser import publish
    run, original = observed_run(tmp_path)
    service, url = start_service(run)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.wait_for_selector('[data-sample-id]')
            node = original['graph']['nodes'][0]
            page.locator('#neuron-details summary').click()
            page.locator(f'[data-node-select="{node["index"]}"]').click()
            base = time.time()
            for index in range(94):
                packet = copy.deepcopy(original)
                packet['sequence'] += index + 1
                packet['updated_unix'] = base + index
                packet['sample'].update(sample_id=f'controlled:{index}', step=1000+index,
                                        observed_unix=base+index*.2)
                packet['sample']['activity'][0] = None if index == 90 else [0.] * 8
                if index >= 92:
                    packet['sequence'] += 3
                publish(run, packet)
                expect(page.locator('#sample-id')).to_have_text(f'controlled:{index}')
                if index == 90:
                    expect(page.locator('#selected-value')).to_have_text('활성 누락')
                if index == 91:
                    assert float(page.locator('#selected-value').get_attribute('data-value')) == 0
            assert page.locator('#history-activity').get_attribute('data-count') == '90'
            assert page.locator('#history-activity path[data-series]').count() >= 2
            assert page.locator('#history-probability path[data-series]').count() >= 2
            assert float(page.locator('#history-probability circle').last.get_attribute('data-value')) == original['sample']['probabilities'][0]
            expect(page.locator('#history-note')).to_contain_text('최대 90개')
            browser.close()
    finally:
        service.terminate()
        service.communicate(timeout=10)


def test_run_worker_switch_discards_late_stream_and_clears_history(tmp_path):
    import copy
    import select
    import subprocess
    import sys
    from test_live_browser import publish
    run, original = observed_run(tmp_path)
    other = tmp_path / 'other'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', run / 'graph', '--output', other,
                 '--updates', 1, '--rollout-steps', 8, '--seed', 31)
    assert result.returncode == 0, result.stdout + result.stderr
    other_packet = json.loads((other / 'live' / 'latest.json').read_text())
    worker = run / 'live' / 'workers' / '1'
    worker.mkdir(parents=True)
    packet = copy.deepcopy(original)
    packet['sample'].update(worker=1, sample_id='controlled-worker-1')
    (worker / 'latest.json').write_text(json.dumps(packet))
    service = subprocess.Popen([sys.executable, '-m', 'flycade', 'live', str(run), '--run', str(other), '--port', '0'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([service.stdout], [], [], 15)[0]
        response = json.loads(service.stdout.readline())
        assert 'url' in response, response
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(response['url'])
            expect(page.locator('#sample-id')).to_have_text(original['sample']['sample_id'])
            held = []
            old_body = page.request.get(response['url']+'/api/latest?run='+original['run_id']+'&worker=0').text()
            def delay_old_worker(route):
                if 'worker=0' in route.request.url and original['run_id'] in route.request.url:
                    held.append(route)
                else:
                    route.continue_()
            page.route('**/api/latest?*', delay_old_worker)
            page.wait_for_timeout(450)
            assert held
            page.locator('#worker-choice').select_option('1')
            expect(page.locator('#sample-id')).to_have_text('controlled-worker-1')
            for route in held:
                try:
                    route.fulfill(status=200, content_type='application/json', body=old_body)
                except Exception as error:
                    # An explicitly aborted old HTTP request may already be gone.
                    assert 'closed' in str(error).lower() or 'invalid interception' in str(error).lower()
            page.unroute('**/api/latest?*', delay_old_worker)
            page.wait_for_timeout(250)
            expect(page.locator('#sample-id')).to_have_text('controlled-worker-1')
            assert page.locator('#history-activity').get_attribute('data-count') == '1'
            stale = copy.deepcopy(packet)
            stale['sequence'] += 1
            stale['sample'].update(worker=0, sample_id='late-worker-0')
            (worker / 'latest.json').write_text(json.dumps(stale))
            page.wait_for_timeout(450)
            expect(page.locator('#sample-id')).to_have_text('controlled-worker-1')
            page.locator('#run-choice').select_option(other_packet['run_id'])
            expect(page.locator('#sample-id')).to_have_text(other_packet['sample']['sample_id'])
            assert page.locator('#worker-choice option').count() == 1
            assert page.locator('#history-activity').get_attribute('data-count') == '1'
            publish(other, original)
            page.wait_for_timeout(450)
            expect(page.locator('#sample-id')).to_have_text(other_packet['sample']['sample_id'])
            browser.close()
    finally:
        service.terminate()
        service.communicate(timeout=10)


def test_top_ten_retains_selection_when_rank_drops(tmp_path):
    import copy
    import hashlib
    import numpy as np
    import pyarrow as pa
    import pyarrow.feather as feather
    from test_graph_cli import fixture
    from test_live_browser import publish
    cache, source, config = fixture(tmp_path)
    ids = [720575940000000001+i for i in range(12)]
    feather.write_feather(pa.table({'pre_pt_root_id': ids[:-1], 'post_pt_root_id': ids[1:],
        'neuropil': ['ME_L']*11, 'syn_count': [5]*11}), cache / 'connections.feather')
    np.save(cache / 'roots.npy', np.array(ids, dtype=np.uint64))
    (cache / 'annotations.tsv').write_text('root_id\tcell_type\tsuper_class\n'+''.join(
        f'{rid}\t{kind}\toptic\n' for rid, kind in zip(ids, ['input']+['middle']*10+['output'])))
    manifest = json.loads(source.read_text())
    for spec in manifest['files'].values():
        spec['sha256'] = hashlib.sha256((cache / spec['name']).read_bytes()).hexdigest()
    source.write_text(json.dumps(manifest))
    graph = tmp_path / 'graph'
    assert cli('prepare-graph', '--cache', cache, '--source-manifest', source, '--config', config, '--output', graph).returncode == 0
    settings = tmp_path / 'training.json'
    settings.write_text(json.dumps({'updates': 1, 'rollout_steps': 4, 'propagation_steps': 11, 'state_dim': 2}))
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run, '--training-config', settings)
    assert result.returncode == 0, result.stdout + result.stderr
    envelope = json.loads((run / 'live' / 'latest.json').read_text())
    service, url = start_service(run)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.wait_for_selector('[data-sample-id]')
            assert page.locator('#neurons tr').count() == 10
            page.locator('#circuit [data-node="0"]').focus()
            page.keyboard.press('Enter')
            packet = copy.deepcopy(envelope)
            packet['sequence'] += 1
            packet['updated_unix'] += 1
            packet['sample'].update(sample_id='controlled-rank-change', step=100)
            packet['sample']['activity'] = [[0.,0.]]+[[.5,.5]]*11
            publish(run, packet)
            expect(page.locator('#sample-id')).to_have_text('controlled-rank-change')
            assert page.locator('#neurons tr').count() == 11
            expect(page.locator('[data-neuron-row="0"]')).to_have_attribute('aria-selected', 'true')
            expect(page.locator('#selected-neuron')).to_contain_text(str(ids[0]))
            browser.close()
    finally:
        service.terminate()
        service.communicate(timeout=10)
