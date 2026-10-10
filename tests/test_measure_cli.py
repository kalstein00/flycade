"""Resource experiment's public process/report boundary, short CPU evidence only."""
import json
import subprocess
import sys
from pathlib import Path
from test_training_cli import prepared_graph


def test_short_measurement_exercises_save_resume_and_marks_non_acceptance(tmp_path):
    graph = prepared_graph(tmp_path)
    output = tmp_path / 'measurement'
    result = subprocess.run([sys.executable, 'scripts/measure_resources.py', '--graph', str(graph),
        '--output', str(output), '--fixture', '--phase-seconds', '1'], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / 'measurement.json').read_text())
    assert not report['a5_duration_met']
    assert report['evidence'] == 'synthetic CPU fixture'
    assert [p['mode'] for p in report['phases']] == ['off', 'live', 'concurrent_evaluation', 'sequential_evaluation']
    assert all(p['status'] == 'saved' and p['new_updates'] > 0 for p in report['phases'])
    assert len({p['session_id'] for p in report['phases']}) == 4
    assert report['evaluations'] and all(e['status'] == 'completed' for e in report['evaluations'])
    assert report['peak_process_tree_rss_bytes'] > 0
    assert report['orphan_children'] == []
    assert report['phases'][-1]['updates'] > report['phases'][0]['updates']


def test_failed_browser_setup_cleans_up_measured_process_tree(tmp_path):
    import os
    graph = prepared_graph(tmp_path)
    output = tmp_path / 'failed-measurement'
    environment = {**os.environ, 'PLAYWRIGHT_BROWSERS_PATH': str(tmp_path / 'missing-browser')}
    result = subprocess.run([sys.executable, 'scripts/measure_resources.py', '--graph', str(graph),
        '--output', str(output), '--fixture', '--phase-seconds', '.2'],
        env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert not (output / 'measurement.json').exists()
    pids = {p['pid'] for line in (output / 'resources.jsonl').read_text().splitlines()
            for p in json.loads(line)['processes']}
    for pid in pids:
        stat = Path(f'/proc/{pid}/stat')
        assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    assert (output / 'run' / 'latest.json').exists(), 'Graceful stop must retain a checkpoint'
