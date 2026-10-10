"""Machine-readable local preparation CLI."""
import argparse
import json
from pathlib import Path

from flycade.diagnostics import diagnose
from flycade.errors import PreparationError
from flycade.rom import catalog, register_rom, inspect_registration


def main() -> int:
    parser = argparse.ArgumentParser(prog='flycade')
    sub = parser.add_subparsers(dest='command', required=True)
    diagnostic = sub.add_parser('diagnose')
    diagnostic.add_argument('--directory', type=Path, default=Path.cwd())
    sub.add_parser('catalog')
    register = sub.add_parser('register-rom')
    register.add_argument('rom', type=Path)
    register.add_argument('--home', type=Path, default=Path('.flycade'))
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--home', type=Path, default=Path('.flycade'))
    demo_parser = sub.add_parser('demo')
    demo_parser.add_argument('--home', type=Path, default=Path('.flycade'))
    demo_parser.add_argument('--output', type=Path, required=True)
    demo_parser.add_argument('--steps', type=int, default=120)
    action_args = demo_parser.add_mutually_exclusive_group()
    action_args.add_argument('--actions', default='0,1,2,3,4,5,6')
    action_args.add_argument('--actions-file', type=Path)
    demo_parser.add_argument('--config', type=Path)
    graph_parser = sub.add_parser('prepare-graph')
    graph_parser.add_argument('--cache', type=Path, default=Path('.flycade/data/v783'))
    graph_parser.add_argument('--source-manifest', type=Path)
    graph_parser.add_argument('--config', type=Path, required=True)
    graph_parser.add_argument('--output', type=Path, required=True)
    graph_inspect = sub.add_parser('inspect-graph')
    graph_inspect.add_argument('output', type=Path)
    graph_inspect.add_argument('--cache', type=Path)
    graph_fetch = sub.add_parser('fetch-graph-data')
    graph_fetch.add_argument('--cache', type=Path, default=Path('.flycade/data/v783'))
    graph_fetch.add_argument('--source-manifest', type=Path)
    training = sub.add_parser('train')
    training.add_argument('--home', type=Path, default=Path('.flycade'))
    training.add_argument('--graph', type=Path, required=True)
    training.add_argument('--output', type=Path, required=True)
    training.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    training.add_argument('--fixture', action='store_true', help='Synthetic CPU evidence only')
    training.add_argument('--updates', type=int)
    training.add_argument('--rollout-steps', type=int)
    training.add_argument('--seed', type=int)
    training.add_argument('--training-config', type=Path)
    training.add_argument('--config', type=Path, help='GameConfig JSON')
    training.add_argument('--stop-after-updates', type=int, help='Save and stop this session; preserve total budget')
    resume_parser = sub.add_parser('resume')
    resume_parser.add_argument('output', type=Path)
    resume_parser.add_argument('--home', type=Path, default=Path('.flycade'))
    resume_parser.add_argument('--stop-after-updates', type=int)
    history = sub.add_parser('history')
    history.add_argument('output', type=Path)
    history.add_argument('--serve', action='store_true', help='Read-only local browser playback')
    history.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    try:
        if args.command == 'diagnose':
            report = diagnose(args.directory)
        elif args.command == 'catalog':
            report = catalog()
        elif args.command == 'register-rom':
            report = register_rom(args.rom, args.home)
        elif args.command == 'fetch-graph-data':
            from flycade.graph import DEFAULT_SOURCE, fetch_graph_data
            report = fetch_graph_data(args.cache, args.source_manifest or DEFAULT_SOURCE)
        elif args.command == 'inspect-graph':
            from flycade.graph import inspect_graph
            report = inspect_graph(args.output, args.cache)
        elif args.command == 'prepare-graph':
            from flycade.graph import DEFAULT_SOURCE, prepare_graph
            report = prepare_graph(args.cache, args.source_manifest or DEFAULT_SOURCE, args.config, args.output)
        elif args.command == 'inspect':
            report = inspect_registration(args.home)
        elif args.command == 'history':
            from flycade.recording import recording_history
            if args.serve:
                from flycade.history import serve_history
                report = serve_history(args.output, args.port)
            else:
                report = recording_history(args.output)
        elif args.command == 'resume':
            from flycade.training import resume
            report = resume(args.output, args.home, args.stop_after_updates)
        elif args.command == 'train':
            from flycade.game import GameConfig
            from flycade.training import TrainingConfig, train
            settings = json.loads(args.training_config.read_text()) if args.training_config else {}
            settings.update({name: getattr(args, name) for name in ('updates', 'rollout_steps', 'seed')
                             if getattr(args, name) is not None})
            game = GameConfig(**json.loads(args.config.read_text())) if args.config else GameConfig()
            report = train(args.home, args.graph, args.output, TrainingConfig(**settings),
                           game, args.device, args.fixture, args.stop_after_updates)
        else:
            from flycade.demo import demo
            from flycade.game import GameConfig
            config = GameConfig(**json.loads(args.config.read_text())) if args.config else GameConfig()
            actions = json.loads(args.actions_file.read_text()) if args.actions_file else [int(a) for a in args.actions.split(',')]
            if not isinstance(actions, list):
                raise ValueError('actions-file must contain a JSON array of action IDs')
            report = demo(args.home, args.output, args.steps, actions, config)
    except PreparationError as exc:
        print(json.dumps({'error': {'code': exc.code, 'message': str(exc)}}))
        return 2
    except (OSError, ImportError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({'error': {'code': 'preparation_failed', 'message': str(exc)}}))
        return 2
    except KeyboardInterrupt:
        print(json.dumps({'error': {'code': 'interrupted', 'message': 'Emulator closed; partial report retained if execution started.'}}))
        return 130
    print(json.dumps(report, indent=2))
    return 0
