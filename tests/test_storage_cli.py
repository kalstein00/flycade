"""Retention and offline operation at the public process/artifact boundary."""
import json

from test_graph_cli import cli
from test_training_cli import prepared_graph


def test_storage_protects_evidence_and_bounds_unprotected_artifacts(tmp_path):
    graph=prepared_graph(tmp_path);run=tmp_path/'run'
    config=tmp_path/'train.json';config.write_text(json.dumps({'updates':5,'rollout_steps':4,'evaluation_every_updates':1}))
    evaluation=tmp_path/'eval.json';evaluation.write_text(json.dumps({'seeds':[11,22,33],'max_frames':120,'video_seconds':1}))
    result=cli('train','--fixture','--device','cpu','--graph',graph,'--output',run,
        '--training-config',config,'--initial-evaluation-config',evaluation,'--stop-after-updates',1)
    assert result.returncode==0,result.stdout+result.stderr
    catalog=json.loads(cli('evaluations',run).stdout)['protocols'][0]
    pinned=catalog['latest_snapshot']
    assert cli('pin-checkpoint',run,pinned).returncode==0
    protected=run/'evaluations'/catalog['latest']
    evidence={str(p.relative_to(run)):p.read_bytes() for p in protected.rglob('*') if p.is_file()}
    branch=tmp_path/'branch'
    assert cli('branch',run,'--checkpoint',pinned,'--output',branch).returncode==0
    origin=(branch/'origin/checkpoint.pt').read_bytes()
    result=cli('storage',run,'--keep-evaluations',1,'--keep-videos',1,'--log-bytes',4096)
    assert result.returncode==0,result.stdout+result.stderr
    result=cli('resume',run)
    assert result.returncode==0,result.stdout+result.stderr
    orphan=run/'snapshots'/'abandoned.partial';orphan.write_bytes(b'incomplete')
    result=cli('storage',run,'--apply')
    assert result.returncode==0,result.stdout+result.stderr
    storage=json.loads(result.stdout)
    assert storage['free_bytes']>0 and storage['cleanup']['status']=='complete'
    assert not orphan.exists()
    assert storage['policy']['keep_evaluations']==1
    catalog=json.loads(cli('evaluations',run).stdout)['protocols'][0]
    rows=catalog['results']
    assert {row['evaluation_id'] for row in rows}=={catalog['initial'],catalog['best'],catalog['latest'],protected.name}
    for name,data in evidence.items():
        assert (run/name).read_bytes()==data
    assert (branch/'origin/checkpoint.pt').read_bytes()==origin
    assert cli('inspect-graph',graph).returncode==0
    assert cli('resume',branch,'--stop-after-updates',1).returncode==0
    assert len(list((run/'videos').glob('*.webm')))<=1
    for name in ('transitions.jsonl','updates.jsonl'):
        path=run/name
        assert path.stat().st_size<=4096
        for line in path.read_text().splitlines():
            json.loads(line)
    history=json.loads(cli('checkpoints',run).stdout)
    assert any(row['checkpoint_id']==pinned for row in history['checkpoints'])
    assert (run/'initial.pt').exists()


def test_offline_new_process_resume_evaluate_and_missing_graph_failure(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    graph=prepared_graph(tmp_path);run=tmp_path/'run'
    trained=cli('train','--fixture','--device','cpu','--graph',graph,'--output',run,
                '--updates',3,'--rollout-steps',4,'--stop-after-updates',1)
    assert trained.returncode==0,trained.stdout+trained.stderr
    protocol=tmp_path/'protocol.json'
    assert cli('evaluation-protocol',run,'--output',protocol,'--max-frames',120,'--video-seconds',1).returncode==0
    library=tmp_path/'offline.so'
    subprocess.run(['cc','-shared','-fPIC',str(Path(__file__).parent/'faults/offline.c'),'-ldl','-o',str(library)],check=True)
    env={**os.environ,'LD_PRELOAD':str(library)}
    probe=subprocess.run([sys.executable,'-c',"import socket; socket.create_connection(('1.1.1.1',443),timeout=1)"],env=env,capture_output=True,text=True)
    assert probe.returncode!=0 and 'Network is unreachable' in probe.stderr
    for args in [('resume',run,'--stop-after-updates',1),('evaluate',run,'--snapshot','latest','--protocol',protocol),('history',run),('storage',run,'--apply')]:
        result=subprocess.run([sys.executable,'-m','flycade',*map(str,args)],env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((run/'report.json').read_text())['updates']==2
    (run/'graph').rename(run/'missing-graph')
    failed=subprocess.run([sys.executable,'-m','flycade','resume',str(run)],env=env,capture_output=True,text=True)
    assert failed.returncode==2
    assert not (run/'graph').exists()


def test_cleanup_failure_keeps_last_good_and_reports_problem(tmp_path):
    graph=prepared_graph(tmp_path);run=tmp_path/'run'
    result=cli('train','--fixture','--device','cpu','--graph',graph,'--output',run,'--updates',2,'--rollout-steps',4,'--stop-after-updates',1)
    assert result.returncode==0,result.stdout+result.stderr
    checkpoint=(run/'final.pt').read_bytes()
    orphan=run/'snapshots'/'orphan.partial';orphan.write_bytes(b'partial')
    (run/'snapshots').chmod(0o555)
    try:
        result=cli('storage',run,'--apply')
        report=json.loads(result.stdout)
        assert report['cleanup']['status']=='failed' and report['cleanup']['errors']
        assert report['last_recovery']['updates']==1
        assert (run/'final.pt').read_bytes()==checkpoint
    finally:
        (run/'snapshots').chmod(0o755)
    assert cli('storage',run,'--apply').returncode==0
    assert not orphan.exists()
    assert cli('resume',run).returncode==0


def test_repeated_initial_evaluations_keep_first_best_latest_not_all_duplicates(tmp_path):
    graph=prepared_graph(tmp_path);run=tmp_path/'run';protocol=tmp_path/'protocol.json'
    assert cli('train','--fixture','--device','cpu','--graph',graph,'--output',run,'--updates',1,'--rollout-steps',4).returncode==0
    assert cli('storage',run,'--keep-evaluations',1).returncode==0
    assert cli('evaluation-protocol',run,'--output',protocol,'--max-frames',120,'--video-seconds',1).returncode==0
    identities=[]
    for _ in range(3):
        result=cli('evaluate',run,'--snapshot','initial','--protocol',protocol)
        assert result.returncode==0,result.stdout+result.stderr
        identities.append(json.loads(result.stdout)['evaluation_id'])
    group=json.loads(cli('evaluations',run).stdout)['protocols'][0]
    assert group['initial']==identities[0] and group['best']==identities[0]
    assert {row['evaluation_id'] for row in group['results']}=={identities[0],identities[-1]}
    assert not (run/'evaluations'/identities[1]).exists()


def test_retry_collects_orphans_and_bounds_failed_evaluations_but_keeps_live_lease(tmp_path):
    import os
    import shutil
    import uuid
    graph=prepared_graph(tmp_path);run=tmp_path/'run';protocol=tmp_path/'protocol.json'
    assert cli('train','--fixture','--device','cpu','--graph',graph,'--output',run,'--updates',1,'--rollout-steps',4).returncode==0
    assert cli('evaluation-protocol',run,'--output',protocol,'--max-frames',120,'--video-seconds',1).returncode==0
    identities=[]
    for _ in range(3):
        result=cli('evaluate',run,'--snapshot','initial','--protocol',protocol)
        assert result.returncode==0,result.stdout+result.stderr
        identities.append(json.loads(result.stdout)['evaluation_id'])
    (run/'evaluations').chmod(0o555)
    try:
        result=cli('storage',run,'--keep-evaluations',1,'--apply')
        assert json.loads(result.stdout)['cleanup']['status']=='failed'
        assert (run/'evaluations'/identities[1]).exists()
    finally:
        (run/'evaluations').chmod(0o755)
    orphan=run/'videos'/'orphan.webm';orphan.write_bytes(b'interrupted video delete')
    assert cli('storage',run,'--apply').returncode==0
    assert not (run/'evaluations'/identities[1]).exists()
    assert not orphan.exists()
    failed=[]
    for i in range(3):
        identifier=str(uuid.uuid4());failed.append(identifier)
        directory=run/'evaluations'/identifier
        shutil.copytree(run/'evaluations'/identities[0],directory)
        path=directory/'report.json';row=json.loads(path.read_text())
        row.update(evaluation_id=identifier,status='failed' if i else 'running',created_unix=i)
        path.write_text(json.dumps(row))
    lease=run/'evaluation-leases'/f'{failed[0]}.json'
    lease.write_text(json.dumps({'pid':os.getpid(),'snapshot_id':'initial'}))
    assert cli('storage',run,'--apply').returncode==0
    assert (run/'evaluations'/failed[0]).exists()
    assert not (run/'evaluations'/failed[1]).exists()
    assert (run/'evaluations'/failed[2]).exists()
    lease.unlink()
    assert cli('storage',run,'--apply').returncode==0
    assert not (run/'evaluations'/failed[0]).exists()
