"""The CNN control uses the same public run/evaluation lifecycle and reports its own outcomes."""
import json

from test_graph_cli import cli
from test_training_cli import prepared_graph


def test_cnn_runs_resume_and_compare_under_the_same_protocol(tmp_path):
    graph=prepared_graph(tmp_path)
    evaluation=tmp_path/'evaluation.json'
    evaluation.write_text(json.dumps({'seeds':[11,22,33],'max_frames':120,'video_seconds':1}))
    runs=[tmp_path/'connectome',tmp_path/'cnn']
    for kind,run in zip(('connectome','cnn'),runs):
        game=tmp_path/f'{kind}-game.json'
        game.write_text(json.dumps({'max_frames':1000 if kind=='connectome' else 10000}))
        config=tmp_path/f'{kind}.json'
        config.write_text(json.dumps({'updates':2,'rollout_steps':4,'evaluation_every_updates':1,'model_kind':kind}))
        result=cli('train','--fixture','--device','cpu','--graph',graph,'--output',run,
                   '--config',game,'--training-config',config,'--initial-evaluation-config',evaluation,'--stop-after-updates',1)
        assert result.returncode==0,result.stdout+result.stderr
        initial=(run/'initial.pt').read_bytes()
        result=cli('resume',run)
        assert result.returncode==0,result.stdout+result.stderr
        report=json.loads(result.stdout)
        assert report['updates']==2 and report['optimizer_steps']==4
        assert (run/'initial.pt').read_bytes()==initial
        manifest=json.loads((run/'run.json').read_text())
        assert manifest['model']['kind']==kind and manifest['model']['pretrained_or_teacher'] is False
        assert manifest['model']['parameter_count']>0
        if kind=='cnn':
            assert report['graph_influence']['applicable'] is False
            observed=json.loads((run/'live/latest.json').read_text())
            assert observed['graph']['applicable'] is False
            assert observed['sample']['activity']==[]
    compared=cli('compare-runs',*runs)
    assert compared.returncode==0,compared.stdout+compared.stderr
    summary=json.loads(compared.stdout)
    assert summary['same_evaluation_protocol'] and summary['same_training_budget']
    assert summary['setting_differences']=={'model_kind':['connectome','cnn'],'game.max_frames':[1000,10000]}
    assert [row['model_kind'] for row in summary['runs']]==['connectome','cnn']
    for row in summary['runs']:
        assert row['initial']['snapshot_updates']==0 and row['latest']['snapshot_updates']==2
        assert row['initial']['evaluation_count']==3
        assert 'mean_distance' in row['change'] and 'completion_rate' in row['change'] and 'deaths' in row['change']

    extended=cli('extend-budget',runs[1],'--updates',3)
    assert extended.returncode==0,extended.stdout+extended.stderr
    changed=json.loads(cli('compare-runs',*runs).stdout)
    assert not changed['same_training_budget']
    assert [row['budget']['total_updates'] for row in changed['runs']]==[2,3]
    rejected=cli('compare-runs',*runs,'--protocol','not-a-shared-protocol')
    assert rejected.returncode==2 and 'protocol_incompatible' in rejected.stdout


def test_cnn_branch_and_weight_reuse_keep_model_identity(tmp_path):
    graph=prepared_graph(tmp_path);parent=tmp_path/'cnn'
    trained=cli('train','--fixture','--device','cpu','--graph',graph,'--output',parent,
                '--model-kind','cnn','--updates',3,'--rollout-steps',4,'--stop-after-updates',1)
    assert trained.returncode==0,trained.stdout+trained.stderr
    checkpoint=json.loads(trained.stdout)['checkpoint_id']
    branch=tmp_path/'branch'
    assert cli('branch',parent,'--checkpoint',checkpoint,'--output',branch).returncode==0
    resumed=cli('resume',branch,'--stop-after-updates',1)
    assert resumed.returncode==0,resumed.stdout+resumed.stderr
    assert json.loads(resumed.stdout)['updates']==2
    config=tmp_path/'warm.json';config.write_text(json.dumps({'updates':1,'rollout_steps':4}))
    warm=tmp_path/'warm'
    result=cli('warm-start',parent,'--checkpoint',checkpoint,'--output',warm,'--training-config',config)
    assert result.returncode==0,result.stdout+result.stderr
    manifest=json.loads((warm/'run.json').read_text())
    assert manifest['model']['kind']=='cnn'
    assert manifest['lineage']['kind']=='warm_start'
    assert json.loads(result.stdout)['updates']==1
