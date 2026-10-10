"""Bounded reward experiment; queue a follow-up once, never poll from the agent."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--queue-thread')
    parser.add_argument('--resume-run', type=Path, help='Continue a saved Run; output must be a fresh segment directory')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600:
        parser.error('--seconds must be between 1 and 3600')
    run = args.resume_run if args.resume_run is not None else args.output / 'run'
    if args.resume_run is not None and not (run / 'report.json').is_file():
        parser.error('Resume Run is missing its saved report')
    if args.resume_run is None and run.exists():
        parser.error('Run already exists; use a fresh output directory')
    if any((args.output / name).exists() for name in ('ready.json', 'summary.json', 'trainer.log')):
        parser.error('Segment output already exists; use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = [sys.executable, '-m', 'flycade']

    def cli(*values: object) -> dict[str, Any]:
        result = subprocess.run([*prefix, *map(str, values)], capture_output=True,
                                text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        document: dict[str, Any] = json.loads(result.stdout)
        return document

    trainer: subprocess.Popen[str] | None = None
    try:
        if args.resume_run is None:
            # Initial evaluation and first update prove readiness before timing the session.
            first = cli('train', '--graph', '.flycade/graphs/visual-a2-final-001', '--output', run,
                        '--training-config', 'configs/training-reward-v2.json',
                        '--config', 'configs/game-reward-v2.json', '--device', 'cuda',
                        '--initial-evaluation-config', 'configs/evaluation-small.json',
                        '--stop-after-updates', 1)
        else:
            first = json.loads((run / 'report.json').read_text())
        starting_catalog = cli('evaluations', run)
        (args.output / 'ready.json').write_text(json.dumps(first, indent=2))
        started = time.monotonic()
        with (args.output / 'trainer.log').open('w') as log:
            trainer = subprocess.Popen([*prefix, 'resume', str(run)], stdout=log,
                                       stderr=subprocess.STDOUT, text=True)
            print(json.dumps({'status': 'learning_started', 'run': str(run),
                              'seconds': args.seconds, 'trainer_pid': trainer.pid}), flush=True)
            try:
                trainer.wait(timeout=args.seconds)
            except subprocess.TimeoutExpired:
                cli('save', run, '--stop')
                trainer.wait(timeout=30)
        if trainer.returncode:
            raise RuntimeError(f'Trainer exited {trainer.returncode}; inspect trainer.log')
        elapsed = time.monotonic() - started
        final = json.loads((run / 'report.json').read_text())
        evaluated = cli('evaluate', run, '--snapshot', 'latest', '--protocol',
                        run / 'initial-evaluation-protocol.json')
        catalog = cli('evaluations', run)
        summary = {'run': str(run.resolve()), 'starting_catalog': starting_catalog,
                   'requested_session_seconds': args.seconds, 'session_wall_seconds': elapsed,
                   'learning_seconds': final['training_seconds'] - first['training_seconds'],
                   'initial_update': first['updates'], 'final': final,
                   'final_evaluation': evaluated, 'catalog': catalog,
                   'interpretation': 'Compare distance/completion/endings within this protocol; '
                                     'reward totals differ from the old reward system.'}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({'output': str(args.output), 'updates': final['updates'],
                          'wall_seconds': elapsed}), flush=True)
    finally:
        try:
            if trainer is not None and trainer.poll() is None:
                try:
                    cli('save', run, '--stop')
                    trainer.wait(timeout=30)
                finally:
                    if trainer.poll() is None:
                        trainer.terminate()
                        try:
                            trainer.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            trainer.kill()
                            trainer.wait()
        finally:
            if args.queue_thread:
                message = (f'학습 {args.seconds}초 구간이 종료됐습니다. {args.output}/summary.json과 '
                           'trainer.log를 확인하고 초기 대비 이동 거리·완주·사망·정체·행동 분포를 분석하세요. '
                           '구간 시작 전 평가와 최종 평가를 비교하고 실패도 숨기지 말고 기록하세요. 사용자 요청은5분 단위 학습입니다. 임의로 장시간 연장하지 마세요. '
                           '반복 폴링 없이 작업하고, 실제 Windows 화면/WSL 재시작 미확인은 별도로 유지하세요.')
                delivered = subprocess.run(['codex', 'queue', '--thread', args.queue_thread,
                                             '--message', message], capture_output=True, text=True, timeout=30)
                (args.output / 'queue-delivery.json').write_text(json.dumps({
                    'returncode': delivered.returncode, 'stdout': delivered.stdout,
                    'stderr': delivered.stderr}, ensure_ascii=False, indent=2))



if __name__ == '__main__':
    main()
