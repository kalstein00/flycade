"""Native video seeking selects coherent recorded observations, including missing intervals."""
import gzip
import hashlib
import json

from playwright.sync_api import expect, sync_playwright

from test_graph_cli import cli
from test_live_browser import start_service
from test_training_cli import prepared_graph


def test_replay_seek_pause_missing_sample_and_neuron_inspection(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    evaluation = tmp_path / 'evaluation.json'
    evaluation.write_text(json.dumps({'seeds': [11,22,33], 'max_frames': 120, 'video_seconds': 2}))
    trained = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                  '--updates', 1, '--rollout-steps', 4, '--initial-evaluation-config', evaluation)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    evaluation_id = json.loads(trained.stdout)['initial_evaluation_id']
    directory = run / 'evaluations' / evaluation_id
    report = json.loads((directory / 'report.json').read_text())
    path = directory / report['replay']['file']
    data = json.loads(gzip.decompress(path.read_bytes()))
    first = data['samples'][0]
    # A declared missing observation interval; preserve the real forward values around it.
    del data['samples'][1]
    encoded = gzip.compress(json.dumps(data).encode(), mtime=0)
    path.write_bytes(encoded)
    report['replay']['sha256'] = hashlib.sha256(encoded).hexdigest()
    (directory / 'report.json').write_text(json.dumps(report))
    before = (run / 'final.pt').read_bytes()
    service, url = start_service(run)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={'width':1280,'height':900})
            page.goto(url+'/replay?evaluation='+evaluation_id)
            expect(page.get_by_role('heading',name='평가 영상과 회로 이력')).to_be_visible()
            expect(page.locator('#sample-identity')).to_contain_text(first['sample_id'])
            node = data['graph']['nodes'][0]
            page.locator(f'#circuit [data-node="{node["index"]}"]').focus()
            page.keyboard.press('Enter')
            expected = sum(first['activity'][0])/len(first['activity'][0])
            assert abs(float(page.locator('#selected-value').get_attribute('data-value')) - expected) < 1e-7
            player = page.locator('#replay-video')
            player.evaluate('(v)=>{v.pause();v.currentTime=.4;}')
            expect(page.locator('#replay-status')).to_contain_text('해당 시점의 관측 데이터 없음')
            assert page.locator('#circuit').count() == 0
            player.evaluate('(v)=>{v.currentTime=.8;}')
            expect(page.locator('#sample-identity')).to_contain_text(data['samples'][1]['sample_id'])
            player.evaluate('(v)=>v.play()')
            page.wait_for_function('()=>document.querySelector("#replay-video").currentTime>.9')
            player.evaluate('(v)=>v.pause()')
            for width in (1920,1280,390):
                page.set_viewport_size({'width':width,'height':844})
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            browser.close()
        assert (run / 'final.pt').read_bytes() == before
    finally:
        service.terminate(); service.communicate(timeout=10)


def test_replay_discards_late_selection_and_shows_legacy_absence(tmp_path):
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    evaluation = tmp_path / 'evaluation.json'
    evaluation.write_text(json.dumps({'seeds':[11,22,33],'max_frames':120,'video_seconds':2}))
    trained = cli('train','--fixture','--device','cpu','--graph',graph,'--output',run,'--updates',1,
                  '--rollout-steps',4,'--initial-evaluation-config',evaluation)
    assert trained.returncode == 0
    initial = json.loads(trained.stdout)['initial_evaluation_id']
    later = cli('evaluate',run,'--snapshot','latest','--protocol',run/'initial-evaluation-protocol.json')
    assert later.returncode == 0, later.stdout+later.stderr
    latest = json.loads(later.stdout)['evaluation_id']
    directory = run/'evaluations'/latest
    report = json.loads((directory/'report.json').read_text())
    del report['replay']  # Legacy recording without observations.
    (directory/'report.json').write_text(json.dumps(report))
    service,url = start_service(run)
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch();page=browser.new_page()
            page.goto(url+'/replay?evaluation='+initial)
            expect(page.locator('#sample-identity')).to_contain_text(initial)
            pending=[]
            page.route('**/api/replay?**', lambda route: pending.append(route) if 'evaluation='+initial in route.request.url else route.continue_())
            page.locator('#evaluation-choice').select_option(latest)
            expect(page.locator('#replay-status')).to_contain_text('회로 관측 데이터 없음')
            page.locator('#evaluation-choice').select_option(initial)
            page.wait_for_timeout(100)
            assert pending
            response=pending[0].fetch()
            page.locator('#evaluation-choice').select_option(latest)
            expect(page.locator('#replay-status')).to_contain_text('회로 관측 데이터 없음')
            pending[0].fulfill(response=response)
            page.wait_for_timeout(100)
            assert page.locator('#circuit').count() == 0
            expect(page.locator('#replay-identity')).to_contain_text(latest)
            page.locator('#mode-choice').select_option('live')
            expect(page.locator('#replay-status')).to_contain_text('고정 평가 관측')
            expect(page.locator('#sample-identity')).to_contain_text(latest)
            assert page.locator('#replay-video').is_hidden()
            expect(page.locator('#live-image img')).to_be_visible()
            report['replay'] = {'status':'failed','error':'Disk full fixture'}
            (directory/'report.json').write_text(json.dumps(report))
            page.locator('#mode-choice').select_option('recorded')
            expect(page.locator('#replay-status')).to_contain_text('기록 실패 · Disk full fixture')
            browser.close()
    finally:
        service.terminate();service.communicate(timeout=10)


def test_running_fixed_evaluation_has_its_own_coherent_stream(tmp_path):
    import subprocess
    import sys
    import time
    graph=prepared_graph(tmp_path);run=tmp_path/'run'
    assert cli('train','--fixture','--device','cpu','--graph',graph,'--output',run,
               '--updates',1,'--rollout-steps',4).returncode==0
    before=(run/'final.pt').read_bytes();protocol=tmp_path/'protocol.json'
    assert cli('evaluation-protocol',run,'--output',protocol,'--mode','deterministic','--seeds','11',
               '--max-frames',600,'--video-seconds',3).returncode==0
    process=subprocess.Popen([sys.executable,'-m','flycade','evaluate',str(run),'--snapshot','initial',
        '--protocol',str(protocol),'--realtime'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    service,url=start_service(run)
    try:
        deadline=time.monotonic()+10
        while not list((run/'evaluations').glob('*/live.json')) and time.monotonic()<deadline:
            assert process.poll() is None,process.communicate()
            time.sleep(.02)
        with sync_playwright() as pw:
            browser=pw.chromium.launch();page=browser.new_page()
            page.goto(url+'/replay')
            page.locator('#mode-choice').select_option('live')
            expect(page.locator('#replay-status')).to_contain_text('고정 평가 관측 · 실행 중')
            expect(page.locator('#sample-identity')).to_contain_text('snapshot initial')
            expect(page.locator('#live-image img')).to_be_visible()
            assert page.locator('#circuit [data-node]').count()>0
            page.close();browser.close()
        stdout,stderr=process.communicate(timeout=20)
        assert process.returncode==0,stdout+stderr
        assert (run/'final.pt').read_bytes()==before
    finally:
        service.terminate();service.communicate(timeout=10)
        if process.poll() is None:
            process.kill();process.communicate(timeout=10)
