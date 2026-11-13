"""Launch one native method with the validated server241 execution profile."""
import argparse
import os
from pathlib import Path
import subprocess

from .run_pipeline import METHODS, GROUPS, command, child_environment, load_sidecar_key
from .common.model_config import task_api_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--print-command', action='store_true')
    parser.add_argument('--resume-meta', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-analysis-proposal', action='store_true')
    parser.add_argument('--recreate-profile', choices=('native', 'shared-v246'), default='native')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('Use 1..16 workers; the shared profile reserves separate evaluation slots')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be a single directory name')
    cmd = command(args.method, args.group)
    if args.recreate_profile != 'native':
        if args.method != 'recreate':
            parser.error('--recreate-profile requires --method recreate')
        cmd[cmd.index('qwen_gpt55_experiments.recreate.run_native')] = 'qwen_gpt55_experiments.recreate.run_shared_v2'
    if args.resume_analysis_proposal:
        if args.method != 'meta_harness' or args.resume or args.resume_meta:
            parser.error('--resume-analysis-proposal requires meta_harness and cannot combine with other resume flags')
        cmd += ['--resume-analysis-proposal']
    if args.method == 'harnessfix':
        run_dir = Path(__file__).resolve().parent / args.method / args.group / args.run_name
        cmd += ['--run-dir', str(run_dir)]
    else:
        cmd += ['--run-name', args.run_name]
    for i, arg in enumerate(cmd[:-1]):
        if arg in ('--final-workers', '--benchmark-workers', '--validation-workers'):
            cmd[i + 1] = str(args.workers)
        elif arg in ('--adaptation-workers', '--evolution-workers', '--analysis-workers'):
            cmd[i + 1] = str(min(4, args.workers))
    if args.resume and args.method != 'meta_harness':
        run_dir = Path(__file__).resolve().parent / args.method / args.group / args.run_name
        if args.method == 'gepa' and (run_dir / 'evolution_summary.json').exists():
            cmd = [cmd[0], '-B', '-m', 'qwen_gpt55_experiments.gepa.resume_final',
                   '--ablation-group', args.group, '--run-name', args.run_name, '--workers', str(args.workers)]
        else:
            cmd += ['--resume']
            if args.method == 'gepa':
                cmd += ['--resume-metrics-from', str(run_dir)]
    if args.resume_meta or (args.resume and args.method == 'meta_harness'):
        if args.method != 'meta_harness':
            parser.error('--resume-meta requires --method meta_harness')
        # Recover the corpus actually saved in this run, including legacy 32-case runs.
        search = Path(__file__).resolve().parent / 'meta_harness' / args.group / args.run_name / 'search'
        adaptation_count = len(list((search / 'traces').glob('*.traj.json')))
        if not adaptation_count:
            parser.error('Meta-Harness resume requires an existing trace corpus')
        cmd = [cmd[0], '-B', '-m', 'qwen_gpt55_experiments.meta_harness.resume_native',
               '--ablation-group', args.group, '--run-name', args.run_name,
               '--adaptation-tasks', str(adaptation_count), '--adaptation-seed', '20260902', '--workers', str(args.workers)]
    if args.print_command:
        import shlex
        print(shlex.join(cmd))
        return
    env = child_environment(load_sidecar_key(), task_api_key())
    env.update(EXPERIMENT_CONTAINER_BACKEND='remote_docker', EXPERIMENT_REMOTE_HOST='server241',
               EXPERIMENT_REMOTE_MEMORY='8g', EXPERIMENT_REMOTE_CPUS='4',
               EXPERIMENT_REMOTE_MIN_DISK_GIB='100', EXPERIMENT_REMOTE_STOP_DISK_GIB='80',
               EXPERIMENT_REMOTE_MIN_MEMORY_GIB='64', UDOCKER_MAX_CONTAINERS='16',
               UDOCKER_MAX_EVALUATORS='4', UDOCKER_CREATE_CONCURRENCY='2', PYTHONDONTWRITEBYTECODE='1')
    # Deliberately a single foreground job: callers can supervise it without
    # starting the old three-group/five-method sweep or changing method budgets.
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == '__main__':
    main()
