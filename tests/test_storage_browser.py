"""Protected playback and missing history after local retention."""
import json
from playwright.sync_api import expect, sync_playwright
from test_graph_cli import cli
from test_live_browser import start_service
from test_training_cli import prepared_graph


def test_storage_status_and_protected_playback_after_cleanup(tmp_path):
    graph=prepared_graph(tmp_path);run=tmp_path/'run'
    config=tmp_path/'train.json';config.write_text(json.dumps({'updates':2,'rollout_steps':4,'evaluation_every_updates':1}))
    evaluation=tmp_path/'eval.json';evaluation.write_text(json.dumps({'seeds':[11,22,33],'max_frames':120,'video_seconds':1}))
    result=cli('train','--fixture','--device','cpu','--graph',graph,'--output',run,'--training-config',config,'--initial-evaluation-config',evaluation)
    assert result.returncode==0,result.stdout+result.stderr
    before=json.loads(cli('evaluations',run).stdout)['protocols'][0]
    assert cli('storage',run,'--keep-evaluations',1,'--apply').returncode==0
    after=json.loads(cli('evaluations',run).stdout)['protocols'][0]
    removed={r['evaluation_id'] for r in before['results']}-{r['evaluation_id'] for r in after['results']}
    assert removed
    service,url=start_service(run)
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch();page=browser.new_page()
            page.goto(url)
            expect(page.locator('#storage-summary')).to_contain_text('남은 공간')
            page.locator('#storage-info summary').click()
            expect(page.locator('#storage-result')).to_contain_text('최근 정리')
            page.goto(url+'/compare')
            expect(page.locator('#left-identity')).to_contain_text('initial')
            page.locator('#right-video').evaluate('(v)=>v.play()')
            page.wait_for_function('()=>document.querySelector("#right-video").currentTime>0')
            page.goto(url+'/replay?evaluation='+after['initial'])
            expect(page.locator('#circuit')).to_be_visible()
            page.locator('#replay-video').evaluate('(v)=>v.play()')
            page.wait_for_function('()=>document.querySelector("#replay-video").currentTime>0')
            page.goto(url+'/replay?evaluation='+next(iter(removed)))
            expect(page.locator('#replay-status')).to_contain_text('평가 기록 없음')
            assert page.locator('#circuit').count()==0
            browser.close()
    finally:
        service.terminate();service.communicate(timeout=10)
