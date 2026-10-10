"""Explicit small-sample outcome comparison for separately identified Runs."""
import json
from pathlib import Path
from typing import Any

from flycade.budget import budget_info
from flycade.errors import PreparationError
from flycade.evaluation_index import evaluation_catalog


def compare_runs(runs: list[Path], protocol_id: str | None = None) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError('Choose at least two Runs')
    manifests = [json.loads((run / 'run.json').read_text()) for run in runs]
    if len({manifest['run_id'] for manifest in manifests}) != len(runs):
        raise ValueError('Choose distinct Runs')
    catalogs = [evaluation_catalog(run) for run in runs]
    common = set.intersection(*({group['protocol_id'] for group in catalog['protocols']} for catalog in catalogs))
    if protocol_id is None and len(common) == 1:
        protocol_id = next(iter(common))
    if protocol_id not in common:
        raise PreparationError('protocol_incompatible', 'Choose --protocol shared by all Runs; different protocols are not identical conditions')
    rows = []
    budgets = []
    for run, manifest, catalog in zip(runs, manifests, catalogs):
        group = next(group for group in catalog['protocols'] if group['protocol_id'] == protocol_id)
        initial = next((row for row in group['results'] if row['evaluation_id'] == group['initial']), None)
        latest = next(row for row in group['results'] if row['evaluation_id'] == group['latest'])
        if initial is None:
            raise PreparationError('initial_evaluation_missing', f"Evaluate the initial snapshot with this protocol for Run {manifest['run_id']}")
        budget = budget_info(run, manifest)
        planned = {'total_updates': budget['total_updates'], 'rollout_steps': manifest['training']['rollout_steps'],
                   'epochs': manifest['training']['epochs'],
                   'planned_transitions': budget['total_updates'] * manifest['training']['rollout_steps']}
        budgets.append(planned)
        rows.append({'run_id': manifest['run_id'], 'model_kind': manifest['model'].get('kind', 'connectome'),
            'parameter_count': manifest['model'].get('parameter_count'), 'budget': planned,
            'initial': initial, 'latest': latest,
            'change': {'mean_distance': latest['mean_distance'] - initial['mean_distance'],
                       'completion_rate': latest['completion_rate'] - initial['completion_rate'],
                       'deaths': latest['termination_counts'].get('death', 0) - initial['termination_counts'].get('death', 0)},
            'evidence': latest['evidence']})
    keys = sorted(set().union(*(manifest['training'] for manifest in manifests)))
    differences = {key: [manifest['training'].get(key) for manifest in manifests] for key in keys
                   if any(manifest['training'].get(key) != manifests[0]['training'].get(key) for manifest in manifests[1:])}
    game_keys = sorted(set().union(*(manifest['game'] for manifest in manifests)))
    differences.update({f'game.{key}': [manifest['game'].get(key) for manifest in manifests] for key in game_keys
                        if any(manifest['game'].get(key) != manifests[0]['game'].get(key) for manifest in manifests[1:])})
    return {'protocol_id': protocol_id, 'same_evaluation_protocol': True,
            'same_training_budget': all(budget == budgets[0] for budget in budgets),
            'setting_differences': differences, 'runs': rows,
            'limitations': 'Small fixed-seed sample; model capacity differs. Weight/loss changes alone do not prove improved play or model superiority.',
            'investigation': ['ROM/state/integration identity', 'raw reward and termination traces',
                              'pixel-to-policy mapping', 'action probabilities and exploration',
                              'optimizer/loss/gradient logs and fixed-policy initial/latest results']}
