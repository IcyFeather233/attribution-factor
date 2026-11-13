"""ReCreate optimization over a fixed shared mini-swe-agent v2.4.6 task seed.

This is an explicit experimental adapter. Native runs and their scores are not
overwritten. The legacy optimizer and v2 task worker never share an interpreter.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from .shared_v2_protocol import RUN_ROOT, PROTOCOL, guard_run
from qwen_gpt55_experiments.common.udocker_runtime import persistent_environment
from qwen_gpt55_experiments.common.execution_protocol import record_execution_backend
from qwen_gpt55_experiments.common.analysis_payloads import GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Adaptation/final worker counts, sampling seed and smoke flags are forwarded to the native driver.')
    parser.add_argument('--ablation-group', choices=GROUPS, required=True)
    parser.add_argument('--run-name', default='shared_v246_round1')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-final', action='store_true')
    args, _ = parser.parse_known_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be one directory component')
    os.environ.update(persistent_environment())
    os.environ.setdefault('EXPERIMENT_CONTAINER_BACKEND', 'remote_docker')
    os.environ.setdefault('EXPERIMENT_REMOTE_HOST', 'server241')
    if os.environ['EXPERIMENT_CONTAINER_BACKEND'] != 'remote_docker':
        parser.error('Shared-v2 task worker currently supports the registered server241 backend')
    if args.prepare_only and (args.resume or args.resume_final):
        parser.error('--prepare-only creates a fresh seed; it cannot combine with resume')
    run = RUN_ROOT / args.ablation_group / args.run_name
    protocol = guard_run(run, resume=args.resume or args.resume_final)
    record_execution_backend(run)
    (run / 'shared_v2_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')

    # Import ReCreate only after selecting the process-local protocol. Its bundled
    # legacy mini-swe-agent is intentionally used for the optimizer alone.
    from . import run_native as legacy
    from .shared_v2_adapter import install
    install(legacy)
    legacy.RUN_ROOT = RUN_ROOT
    legacy.EVOLUTION_PROTOCOL = PROTOCOL
    legacy.NATIVE_UPSTREAM_FUNCTIONS = ['initialize_shared_mini_swe_seed', 'run_single_evolution', 'run_batch_synthesis']
    if args.prepare_only:
        adapter = legacy.UdockerSWEBenchAdapter(step_limit=250, analysis_payloads={})
        legacy.initialize_workspace(workspace=run / 'workspace', runs_recreate_dir=run / 'runs_recreate',
                                    adapter=adapter, smoke=False)
        (run / 'resume32_drained.json').write_text(json.dumps({
            'checkpoint_type': 'prepared fixed seed; no tasks or optimizer calls executed',
            'protocol': PROTOCOL, 'completed_cases': 0}, indent=2) + '\n')
        print(json.dumps({'state': 'prepared_only', 'run_dir': str(run), 'protocol': PROTOCOL,
                          'model_calls': 0, 'next': 'Use the same arguments with --resume instead of --prepare-only'}))
        return
    if '--run-name' not in sys.argv:
        sys.argv += ['--run-name', args.run_name]
    legacy.main()
    path = run / 'summary.json'
    summary = json.loads(path.read_text())
    summary.update(initialization='fixed_shared_mini_swe_seed', task_runtime='2.4.6',
                   optimizer_runtime='1.14.4', shared_seed_sha256=protocol['seed_sha256'],
                   task_trace_format='canonical v2; separate faithful compatibility view for legacy analysis')
    path.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
