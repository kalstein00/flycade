"""Read-only host probes; absence and failed probes remain visible."""
import csv
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def command(argv: list[str]) -> tuple[str, str | None]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        if result.returncode:
            return '', result.stderr.strip() or f'exit {result.returncode}'
        return result.stdout.strip(), None
    except (OSError, subprocess.TimeoutExpired) as exc:
        return '', str(exc)


def diagnose(directory: Path) -> dict[str, Any]:
    issues: list[str] = []
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        mem[key] = int(value.strip().split()[0]) * 1024
    # Both cgroup versions, including the current process's nested membership.
    limits = []
    candidates = {Path('/sys/fs/cgroup/memory.max'),
                  Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')}
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, group = line.split(':', 2)
        base = Path('/sys/fs/cgroup')
        filename = 'memory.max'
        if 'memory' in controllers.split(','):
            base /= 'memory'
            filename = 'memory.limit_in_bytes'
        elif controllers:
            continue
        leaf = base / group.lstrip('/')
        for parent in [leaf, *leaf.parents]:
            if parent == base.parent:
                break
            candidates.add(parent / filename)
    for path in candidates:
        try:
            value = path.read_text().strip()
            if value.isdigit():
                limits.append(int(value))
        except OSError:
            pass
    effective = min([mem['MemTotal'], *limits])
    wsl = 'microsoft' in platform.release().lower()
    windows_ram = None
    if wsl:
        output, error = command(['powershell.exe', '-NoProfile', '-NonInteractive',
                                 '-Command', '(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory'])
        if not error and output.strip().isdigit():
            windows_ram = int(output.strip())
        else:
            issues.append(f'Windows RAM unavailable: {error or output}')
    smi = shutil.which('nvidia-smi')
    if smi is None and Path('/usr/lib/wsl/lib/nvidia-smi').exists():
        smi = '/usr/lib/wsl/lib/nvidia-smi'
    output, error = command([smi or 'nvidia-smi',
        '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits'])
    gpus = []
    if error:
        issues.append(f'GPU probe unavailable: {error}. WSL uses the Windows NVIDIA driver; do not install a Linux display driver.')
    else:
        for row in csv.reader(output.splitlines()):
            if len(row) == 4:
                name, total, free, driver = [v.strip() for v in row]
                gpus.append({'name': name, 'total_vram_mib': int(total) if total.isdigit() else None,
                             'free_vram_mib': int(free) if free.isdigit() else None, 'driver': driver})
    packages: dict[str, str | None] = {}
    for name in ['stable-retro', 'gymnasium', 'numpy', 'pillow', 'torch', 'stable-baselines3']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    _, import_error = command([sys.executable, '-c', 'import stable_retro, numpy; from PIL import Image'])
    if import_error:
        issues.append(f'NES dependencies unavailable: {import_error}. Run uv sync --locked.')
    disk = shutil.disk_usage(directory)
    if effective < 2 * 1024**3:
        issues.append('Less than 2 GiB effective Linux RAM: start with one short NES demo; training budget unmeasured.')
    if disk.free < 1024**3:
        issues.append('Less than 1 GiB free disk: free space before installation or recording observations.')
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.machine())
    return {'platform': platform.platform(), 'wsl': wsl, 'python': sys.version,
            'cpu': {'name': cpu, 'logical_count': os.cpu_count(),
                    'available_count': len(os.sched_getaffinity(0))},
            'memory': {'windows_total_bytes': windows_ram, 'linux_total_bytes': mem['MemTotal'],
                       'linux_available_bytes': mem['MemAvailable'], 'cgroup_limits_bytes': sorted(limits),
                       'effective_limit_bytes': effective},
            'disk': {'directory': str(directory.resolve()), 'free_bytes': disk.free, 'total_bytes': disk.total},
            'gpus': gpus, 'packages': packages, 'issues': issues,
            'missing_training_packages': [name for name in ['torch', 'stable-baselines3'] if packages[name] is None],
            'capabilities': {'diagnostics': 'available',
                             'nes_demo': 'dependencies import; ROM/state validation required' if not import_error else 'blocked: dependencies',
                             'gpu_training': 'unverified: follow-up ticket',
                             'connectome_training': 'unverified: follow-up ticket'}}
