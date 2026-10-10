"""Experiment lineage and schedule changes through public CLI/artifacts."""
import hashlib
import json

import torch

from test_graph_cli import cli
from test_recovery_cli import saved_run


def tree_hashes(run):
    return {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in run.rglob('*') if p.is_file()}


def test_past_checkpoint_branches_full_state_without_changing_parent(tmp_path):
    parent, saves = saved_run(tmp_path, 2)
    before = tree_hashes(parent)
    child = tmp_path / 'child'
    result = cli('branch', parent, '--checkpoint', saves[0]['checkpoint_id'], '--output', child)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['lineage']['kind'] == 'full_state_branch'
    assert report['lineage']['parent_checkpoint_id'] == saves[0]['checkpoint_id']
    assert report['updates'] == 1 and report['transitions'] == 4
    old = torch.load(parent / saves[0]['file'], weights_only=False)
    new = torch.load(child / 'final.pt', weights_only=False)
    assert new['manifest']['run_id'] != old['manifest']['run_id']
    assert new['optimizer']['param_groups'] == old['optimizer']['param_groups']
    for key, value in old['model'].items():
        assert torch.equal(value.to_dense(), new['model'][key].to_dense())
    for key, state in old['optimizer']['state'].items():
        for name, value in state.items():
            assert torch.equal(value, new['optimizer']['state'][key][name])
    assert torch.equal(old['torch_rng'], new['torch_rng'])
    assert new['progress']['next_autosave_seconds'] == old['progress']['next_autosave_seconds']
    assert tree_hashes(parent) == before
    assert (child / 'origin' / 'checkpoint.pt').read_bytes() == (parent / saves[0]['file']).read_bytes()
    result = cli('resume', child, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['updates'] == 2
    assert tree_hashes(parent) == before


def test_warm_start_loads_all_weights_but_resets_experiment_state(tmp_path):
    parent, saves = saved_run(tmp_path, 2)
    before = tree_hashes(parent)
    child = tmp_path / 'warm'
    config = tmp_path / 'warm-config.json'
    config.write_text(json.dumps({'updates': 3, 'rollout_steps': 4, 'epochs': 1,
                                  'seed': 31, 'learning_rate': 0.001, 'autosave_seconds': 900}))
    result = cli('warm-start', parent, '--checkpoint', saves[-1]['checkpoint_id'],
                 '--output', child, '--training-config', config, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['lineage']['kind'] == 'warm_start'
    assert report['updates'] == 1 and report['transitions'] == 4 and report['optimizer_steps'] == 1
    assert report['next_autosave_seconds'] == 900
    manifest = json.loads((child / 'run.json').read_text())
    assert manifest['training']['seed'] == 31
    assert manifest['training']['learning_rate'] == 0.001
    assert 'optimizer' in manifest['lineage']['excluded']
    old = torch.load(parent / saves[-1]['file'], weights_only=False)
    initial = torch.load(child / 'initial.pt', weights_only=True)
    for key, value in old['model'].items():
        assert torch.equal(value.to_dense(), initial[key].to_dense())
    new = torch.load(child / 'final.pt', weights_only=False)
    assert all(v['step'].item() == 1 for v in new['optimizer']['state'].values())
    assert tree_hashes(parent) == before
    config.write_text(json.dumps({'state_dim': 3}))
    invalid = tmp_path / 'wrong-shape'
    result = cli('warm-start', parent, '--checkpoint', saves[-1]['checkpoint_id'],
                 '--output', invalid, '--training-config', config)
    assert result.returncode == 2 and not invalid.exists()
    assert tree_hashes(parent) == before


def test_completed_run_requires_explicit_budget_event_without_schedule_restart(tmp_path):
    from test_training_cli import prepared_graph
    graph = prepared_graph(tmp_path)
    run = tmp_path / 'run'
    result = cli('train', '--fixture', '--device', 'cpu', '--graph', graph, '--output', run,
                 '--updates', 1, '--rollout-steps', 4)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = (run / 'run.json').read_bytes()
    before = torch.load(run / 'final.pt', weights_only=False)
    assert cli('resume', run).returncode == 2
    result = cli('extend-budget', run, '--updates', 3)
    assert result.returncode == 0, result.stdout + result.stderr
    event = json.loads(result.stdout)
    assert event['previous_updates'] == 1 and event['total_updates'] == 3
    assert event['at_updates'] == 1 and event['schedule_rule'].startswith('constant learning rate')
    assert cli('extend-budget', run, '--updates', 2).returncode == 2
    result = cli('resume', run, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['updates'] == 2 and report['optimizer_steps'] == 4
    assert report['budget']['total_updates'] == 3
    after = torch.load(run / 'final.pt', weights_only=False)
    assert after['progress']['next_autosave_seconds'] == before['progress']['next_autosave_seconds']
    assert after['optimizer']['param_groups'] == before['optimizer']['param_groups']
    assert all(v['step'].item() == 4 for v in after['optimizer']['state'].values())
    assert (run / 'run.json').read_bytes() == manifest
    assert json.loads(cli('resume', run).stdout)['updates'] == 3
    assert cli('resume', run).returncode == 2


def test_extended_budget_is_inherited_by_branch_and_missing_ledger_is_rejected(tmp_path):
    parent, saves = saved_run(tmp_path, 1)
    assert cli('extend-budget', parent, '--updates', 25).returncode == 0
    child = tmp_path / 'branch'
    result = cli('branch', parent, '--checkpoint', saves[0]['checkpoint_id'], '--output', child)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((child / 'run.json').read_text())
    assert manifest['training']['updates'] == 25
    assert manifest['lineage']['parent_budget']['events'][0]['previous_updates'] == 20
    result = cli('resume', parent, '--stop-after-updates', 1)
    assert result.returncode == 0, result.stdout + result.stderr
    (parent / 'budget.json').unlink()
    result = cli('resume', parent, '--stop-after-updates', 1)
    assert result.returncode == 2 and 'ledger is older' in result.stdout
    assert cli('extend-budget', parent, '--updates', 30).returncode == 2
