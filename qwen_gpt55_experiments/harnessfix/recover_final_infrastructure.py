"""After the live baseline finishes, retry only infrastructure-invalid final tasks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from qwen_gpt55_experiments.common.result_validity import retryable_result, retryable_text
from qwen_gpt55_experiments.supervise_parallel import atomic_json


def alive(pid: int, expected: str) -> bool:
    proc = Path('/proc') / str(pid)
    try:
        args = proc.joinpath('cmdline').read_bytes().decode().split('\0')
        return expected in args and proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def gradeable(directory: Path) -> bool:
    try:
        result = json.loads((directory / 'result.json').read_text())
        report = json.loads((directory / 'evaluation/report.json').read_text())
    except (OSError, ValueError):
        return False
    value = (result.get('evaluation') or {}).get('resolved')
    return isinstance(value, bool) and report.get('resolved') == value and not retryable_result(result)


def resource_limit_record(log, instance_id):
    """Read the worker's own structured terminal record; never infer from test text."""
    decoder = json.JSONDecoder()
    for offset in reversed([i for i, char in enumerate(log) if char == '{']):
        try:
            record, _ = decoder.raw_decode(log[offset:])
        except ValueError:
            continue
        if (isinstance(record, dict) and record.get('instance_id') == instance_id
                and record.get('exit_status') == 'ContextWindowExceededError'
                and isinstance(record.get('api_calls'), int)):
            return record
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--source-summary', type=Path,
                        help='Archived incomplete summary; fresh summary is written only after recovery')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_relative_to('/volume/med'):
        raise ValueError('Recovery storage must be persistent')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    recovery = run / ('final_infrastructure_recovery_' + stamp)
    recovery.mkdir()
    state = {'state': 'waiting_for_original_final', 'wait_pid': args.wait_pid, 'targets': [], 'completed': [], 'errors': {}, 'resource_limit_failures': []}
    def save():
        atomic_json(recovery / 'status.json', state)
    save()
    while (alive(args.wait_pid, 'qwen_gpt55_experiments.harnessfix.run_native')
           or alive(args.wait_pid, 'qwen_gpt55_experiments.run_remote')):
        time.sleep(30)
    summary_path = run / 'summary.json'
    source_summary = args.source_summary or summary_path
    if not source_summary.is_file():
        raise RuntimeError('Original run ended without summary; do not guess selected candidate')
    summary = json.loads(source_summary.read_text())
    candidate = Path(summary['selected_agent']).resolve()
    if not candidate.is_relative_to(run):
        raise ValueError('Selected candidate outside the requested run')
    ids = [x.strip() for x in (run / 'instance_ids.txt').read_text().splitlines() if x.strip()]
    for iid in ids:
        directory = run / 'final' / iid
        if gradeable(directory):
            continue
        result = json.loads((directory / 'result.json').read_text()) if (directory / 'result.json').exists() else {}
        log = (directory / 'worker.log').read_text(errors='replace') if (directory / 'worker.log').exists() else ''
        terminal = resource_limit_record(log, iid)
        if terminal is not None:
            backup = recovery / 'resource_limits' / iid
            backup.mkdir(parents=True)
            shutil.copy2(directory / 'result.json', backup / 'result.before.json')
            if (directory / 'evaluation').exists():
                shutil.copytree(directory / 'evaluation', backup / 'evaluation.before')
            report = {'instance_id': iid, 'resolved': False, 'error': 'context_window_exceeded',
                      'patch_applied': False, 'terminal_resource_limit': True, 'official_tests_executed': False,
                      'reason': 'Task reached the fixed model context limit; no additional rollout sampled'}
            (directory / 'evaluation').mkdir(exist_ok=True)
            atomic_json(directory / 'evaluation/report.json', report)
            terminal.update(evaluation=report, terminal_result_recovered_from_worker_log=True)
            atomic_json(directory / 'result.json', terminal)
            state['resource_limit_failures'].append(iid)
            continue
        if retryable_result(result) or retryable_text(log) or 'RemoteInfrastructureError' in log:
            state['targets'].append(iid)
        else:
            state['errors'][iid] = 'Invalid final artifact without confirmed infrastructure cause'
    shutil.copy2(source_summary, recovery / 'summary.before.json')
    state['state'] = 'running'
    save()

    def one(iid):
        for attempt in range(2):
            output = recovery / 'attempts' / iid / str(attempt)
            output.mkdir(parents=True)
            cmd = [sys.executable, '-B', '-m', 'qwen_gpt55_experiments.harnessfix.candidate_worker',
                   '--candidate-dir', str(candidate), '--instance-id', iid, '--output-dir', str(output)]
            with (output / 'worker.log').open('x') as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        env=os.environ.copy(), start_new_session=True)
                try:
                    code = proc.wait(timeout=4200)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    code = -1
            if code == 0 and gradeable(output):
                existing = run / 'final' / iid
                archived = recovery / 'before' / iid
                archived.parent.mkdir(exist_ok=True)
                existing.rename(archived)
                output.rename(existing)
                return
            text = (output / 'worker.log').read_text(errors='replace')
            if not retryable_text(text) and 'RemoteInfrastructureError' not in text:
                raise RuntimeError('Retry did not produce a gradeable result; inspect saved worker log')
        raise RuntimeError('Infrastructure retries exhausted')

    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = {executor.submit(one, iid): iid for iid in state['targets']}
        for future in as_completed(tasks):
            iid = tasks[future]
            try:
                future.result()
                state['completed'].append(iid)
            except Exception as exc:
                state['errors'][iid] = str(exc)
            save()
    if state['errors']:
        state['state'] = 'needs_attention'
        save()
        raise RuntimeError('Final recovery incomplete: ' + str(state['errors']))
    if not all(gradeable(run / 'final' / iid) for iid in ids):
        raise RuntimeError('Final verification failed')
    results = [json.loads((run / 'final' / iid / 'result.json').read_text()) for iid in ids]
    resolved = [x['instance_id'] for x in results if x['evaluation']['resolved']]
    summary['final'] = {**summary.get('final', {}), 'status': 'complete', 'total': len(ids),
                        'resolved': len(resolved), 'resolved_ids': resolved, 'results': results}
    summary['final_infrastructure_recovery'] = {'path': str(recovery), 'reused': len(ids) - len(state['targets']), 'retried': state['targets']}
    atomic_json(summary_path, summary)
    state.update(state='complete', resolved=len(resolved), total=len(ids))
    save()
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
