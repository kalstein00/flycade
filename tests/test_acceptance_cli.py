"""Acceptance guards must not turn a short run or same-boot resume into acceptance."""
import json
from pathlib import Path
import subprocess
import sys


def test_acceptance_rejects_short_duration_without_creating_a_run(tmp_path):
    output=tmp_path/'short'
    result=subprocess.run([sys.executable,'scripts/acceptance_run.py','--output',str(output),'--seconds','60'],capture_output=True,text=True)
    assert result.returncode==2 and 'at least 3600 seconds' in result.stderr
    assert not output.exists()


def test_restart_requires_a_real_changed_boot_before_touching_the_run(tmp_path):
    run=tmp_path/'run';run.mkdir();marker=run/'untouched';marker.write_text('original')
    handoff=tmp_path/'handoff.json'
    handoff.write_text(json.dumps({'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'run':str(run)}))
    result=subprocess.run([sys.executable,'scripts/verify_restart.py',str(handoff)],capture_output=True,text=True)
    assert result.returncode==2 and 'boot ID is unchanged' in result.stdout
    assert list(run.iterdir())==[marker] and marker.read_text()=='original'
    assert not handoff.with_name('restart-verification.json').exists()
