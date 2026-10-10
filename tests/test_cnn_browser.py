"""CNN results remain separate, with usable pixels/actions and no invented circuit."""
import json

from playwright.sync_api import expect, sync_playwright

from test_graph_cli import cli
from test_live_browser import start_service
from test_training_cli import prepared_graph


def test_cnn_live_comparison_and_replay_show_no_circuit(tmp_path):
    graph=prepared_graph(tmp_path);run=tmp_path/'cnn';evaluation=tmp_path/'evaluation.json'
    evaluation.write_text(json.dumps({'seeds':[11,22,33],'max_frames':120,'video_seconds':1}))
    trained=cli('train','--fixture','--device','cpu','--graph',graph,'--output',run,'--model-kind','cnn',
                '--updates',1,'--rollout-steps',4,'--initial-evaluation-config',evaluation)
    assert trained.returncode==0,trained.stdout+trained.stderr
    initial=json.loads(trained.stdout)['initial_evaluation_id']
    before=(run/'final.pt').read_bytes()
    service,url=start_service(run)
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch();page=browser.new_page()
            page.goto(url)
            expect(page.locator('#model-kind')).to_have_text('모델: CNN PPO')
            expect(page.get_by_role('heading',name='회로 해당 없음')).to_be_visible()
            assert page.locator('#circuit').count()==0
            assert page.locator('#actions .action-row').count()==7
            page.goto(url+'/compare')
            expect(page.locator('#comparison-model')).to_contain_text('CNN PPO')
            page.goto(url+'/replay?evaluation='+initial)
            expect(page.get_by_role('heading',name='회로 해당 없음')).to_be_visible()
            assert page.locator('#circuit').count()==0
            assert page.locator('#inputs img').count()==4
            assert page.locator('#history-probability').count()==1
            page.locator('#frame-choice').select_option('0')
            expect(page.locator('#selected-input')).to_be_visible()
            for width in (1920,1280,390):
                page.set_viewport_size({'width':width,'height':844})
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            browser.close()
        assert (run/'final.pt').read_bytes()==before
    finally:
        service.terminate();service.communicate(timeout=10)
