"""Native HarnessFix with independently configured diagnosis and validation pools.

This is a bounded adapter, not the upstream train100/val50/test100 protocol.
It restores failed-only diagnosis, native reading budgets and native promotion;
a fresh original-agent validation reference removes the historical-all-failed gate.
"""
import ast
import argparse
import json
import os
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'baselines/harnessfix'


def analysis_ids(eval_path, selected_ids):
    data = json.loads(Path(eval_path).read_text())
    failed = set().union(*(set(data.get(key, [])) for key in ('unresolved_ids', 'empty_patch_ids', 'error_ids')))
    # A resolved task is never a regression merely because it was selected for analysis.
    failed -= set(data.get('resolved_ids', []))
    return sorted(failed & set(selected_ids)) if selected_ids is not None else sorted(failed)


def native_promotion(iteration_dir, gate, audit_passed=True):
    # Load only the upstream function, without importing CLI globals or dotenv.
    path = UPSTREAM / 'run_pipeline_swe.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'step_promotion_decision')
    namespace = {'json': json, 'FAILURE_ANALYSIS_DIR': Path(iteration_dir), 'log': lambda *a: None,
                 'run_label_suffix': lambda: '', '__builtins__': __builtins__}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['step_promotion_decision'](
        version=1, audit={'passed': audit_passed}, train_compare={}, val_compare=gate,
        min_improvement=1, min_target_metrics=1, max_error_rate_delta=.15,
        max_invalid_rate_delta=.15, dry_run=False)


def install(adapter, *, analysis_tasks=148):
    original_write = adapter.write_json
    def write_json(path, value):
        path = Path(path)
        if path.name == 'iteration_report.json' and isinstance(value, dict):
            receipt = path.parent / 'analysis_selection_failures.json'
            if receipt.exists():
                selection = json.loads(receipt.read_text())
                value = {**value, 'analysis_count': len(selection['actual_failed_ids']),
                         'analysis_input_eval': selection['input_eval']}
        return original_write(path, value)
    adapter.write_json = write_json
    original_benchmark = adapter.run_benchmark
    def benchmark(output_root, **kwargs):
        from attribution_experiments.common.result_validity import retryable_result, retryable_text
        from .recover_final_infrastructure import resource_limit_record
        root = Path(output_root)
        for attempt in range(3):
            rows = original_benchmark(root, **{**kwargs, 'resume': True})
            invalid = []
            for row in rows:
                iid = row.get('instance_id')
                if isinstance((row.get('evaluation') or {}).get('resolved'), bool) and not retryable_result(row):
                    continue
                directory = root / iid
                log_path = directory / 'worker.log'
                log = log_path.read_text(errors='replace') if log_path.exists() else ''
                terminal = resource_limit_record(log, iid)
                if terminal:
                    terminal['evaluation'] = {'resolved': False, 'error': 'context_window_exceeded',
                                              'terminal_resource_limit': True, 'official_tests_executed': False}
                    adapter.write_json(directory / 'result.json', terminal)
                    row.clear(); row.update(terminal)
                    continue
                if not retryable_result(row) and not retryable_text(log):
                    raise RuntimeError(f'Non-gradeable native benchmark case requires diagnosis: {iid}')
                invalid.append(directory)
            if not invalid:
                return rows
            if attempt == 2:
                raise RuntimeError('Native benchmark infrastructure retries exhausted')
            for directory in invalid:
                archive = root.parent / (root.name + '_infrastructure_archive') / (directory.name + '_' + str(time.time_ns()))
                archive.parent.mkdir(exist_ok=True)
                directory.rename(archive)
        raise AssertionError('unreachable')
    adapter.run_benchmark = benchmark
    original_prepare_config = adapter.prepare_analysis_config
    def prepare_config(run_dir, inject_analysis):
        path = original_prepare_config(run_dir, inject_analysis)
        config = yaml.safe_load(path.read_text())
        native = yaml.safe_load((UPSTREAM / 'failure_analysis/analysis_config_swe.yaml').read_text())
        template = config['agent']['instance_template']
        marker = '## Your Assignment'
        if marker not in template:
            raise RuntimeError('Cannot locate native analysis template boundary')
        config['agent']['instance_template'] = template[template.index(marker):]
        config['agent']['step_limit'] = native.get('agent', {}).get('step_limit', native.get('step_limit', 20))
        config['agent']['cost_limit'] = native.get('agent', {}).get('cost_limit', native.get('cost_limit', 1))
        scratch = Path(run_dir) / 'analysis_scratch'
        scratch.mkdir(exist_ok=True)
        for section in ('agent', 'model'):
            for key, value in list(config.get(section, {}).items()):
                if isinstance(value, str):
                    config[section][key] = value.replace('/tmp/', str(scratch) + '/')
        # Retain verified endpoint, temperature=0, observation formatting and persistent paths.
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        return path
    adapter.prepare_analysis_config = prepare_config

    original_analyze = adapter.run_analysis
    def analyze(iteration_dir, **kwargs):
        run = Path(iteration_dir).parent
        # Expand only diagnosis of the original failed traces. Current-candidate
        # failures/regressions retain their actual evaluation's IDs and source.
        if (kwargs.get('label', 'failures') == 'failures'
                and Path(kwargs['eval_path']).resolve() == (run / 'baseline/eval.json').resolve()):
            pool = run / 'analysis_ids.txt'
            if not pool.exists():
                raise RuntimeError('Missing independent HarnessFix analysis pool')
            kwargs['selected_ids'] = pool.read_text().split()
        chosen = analysis_ids(kwargs['eval_path'], kwargs.get('selected_ids'))
        audit = {'label': kwargs.get('label', 'failures'), 'input_eval': str(kwargs['eval_path']),
                 'requested_ids': kwargs.get('selected_ids'), 'actual_failed_ids': chosen}
        adapter.write_json(Path(iteration_dir) / ('analysis_selection_' + audit['label'] + '.json'), audit)
        kwargs['selected_ids'] = chosen
        if not chosen:
            output = Path(iteration_dir) / 'analysis' / (kwargs.get('label', 'failures') + '.jsonl')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('')
            return output
        return original_analyze(iteration_dir, **kwargs)
    adapter.run_analysis = analyze

    original_gate = adapter.run_gate
    def gate(iteration_dir, **kwargs):
        # Compare process metrics on exactly the validation IDs, not 148 old cases vs 32 new ones.
        ids = Path(kwargs['ids_file']).read_text().split()
        baseline = Path(kwargs['baseline_eval'])
        run = Path(iteration_dir).parent
        if baseline == run / 'baseline/eval.json':
            reference = run / 'fresh_validation_reference'
            kwargs['baseline_eval'] = reference / 'harnessfix_layout/eval.json'
            kwargs['baseline_traces'] = reference / 'harnessfix_layout/traces'
        comparison = original_gate(iteration_dir, **kwargs)
        promotion = native_promotion(iteration_dir, comparison)
        comparison['bounded_legacy_passed'] = comparison['passed']
        comparison.update(passed=promotion['promoted'], failure_reasons=promotion['failure_reasons'],
                          promotion_policy='upstream_step_promotion_decision', native_promotion=promotion)
        adapter.write_json(Path(iteration_dir) / 'native_promotion.json', comparison)
        return comparison
    adapter.run_gate = gate

    original_prepare = adapter.prepare_run
    def prepare(run_dir, **kwargs):
        run = Path(run_dir)
        if run.exists() and {p.name for p in run.iterdir()} == {'optimization_budget.json'}:
            # The native driver considers any file an existing run. This sole
            # registration file precedes all experiment work and is safe to prepare.
            kwargs = {**kwargs, 'resume': True}
        result = original_prepare(run_dir, **kwargs)
        run = Path(run_dir)
        all_ids = (run / 'instance_ids.txt').read_text().split()
        chosen = adapter.select_validation_ids(all_ids, limit=min(analysis_tasks, len(all_ids)),
                                                seed=kwargs.get('validation_seed', 20260902))
        (run / 'analysis_ids.txt').write_text('\n'.join(chosen) + '\n')
        result.update(analysis_task_count=len(chosen), analysis_ids=chosen,
                      adaptation_set=f'{len(chosen)} original failure traces for diagnosis; validation configured separately')
        result['paths']['analysis_ids'] = str(run / 'analysis_ids.txt')
        adapter.write_json(run / 'manifest.json', result)
        # prepare-only remains read/prepare-only. Actual run calibrates the gate once.
        if '--prepare-only' in sys.argv:
            return result
        ids = Path(result['paths']['validation_ids']).read_text().split()
        optimizer_key, task_key = adapter.optimizer_api_key(), adapter.task_api_key()
        env = adapter.optimizer_env(optimizer_key, task_key)
        reference = run / 'fresh_validation_reference'
        outputs = reference / 'rollouts'
        rows = adapter.run_benchmark(outputs, candidate_dir=run / 'original_agent', ids=ids,
                                     workers=4, env=env, api_key=task_key, resume=True)
        from attribution_experiments.common.result_validity import retryable_result
        if len(rows) != len(ids) or any(retryable_result(row) or not isinstance((row.get('evaluation') or {}).get('resolved'), bool) for row in rows):
            raise RuntimeError('Fresh native validation reference incomplete; do not promote against synthetic zero')
        adapter.stage_benchmark(outputs, reference / 'harnessfix_layout', ids)
        adapter.write_json(run / 'native_alignment_protocol.json', {
            'version': 'harnessfix_full_analysis_v3', 'diagnosis': 'failed IDs only',
            'source_reading_budget': 'native 20-step diagnosis and 40-step aggregate; no 5/8 command clamp',
            'promotion': 'upstream step_promotion_decision defaults', 'fresh_validation_reference_ids': ids,
            'diagnosis_source': 'provided historical failure traces initially; current candidate failures after promotion',
            'training_validation_overlap': bool(set(chosen) & set(ids)), 'adaptation_budget': len(chosen),
            'analysis_task_ids': chosen, 'validation_budget': len(ids),
            'not_full_upstream_split': 'historical diagnosis pool and candidate validation are independently configured; final 148 is a repair benchmark, not held-out testing',
            'optimizer_temperature': 0, 'task_temperature': 1,
            'metrics': 'unsupported targets rejected; tool observations supported; invalid submission = observed rejection episodes',
        })
        return result
    adapter.prepare_run = prepare
    adapter.PROMOTION_POLICY = 'upstream_step_promotion_decision'
    original_redact = adapter.redact_api_key
    adapter.redact_api_key = lambda root, key: None if not key or key == 'EMPTY' else original_redact(root, key)


def main():
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument('--analysis-tasks', type=int, default=148)
    profile_args, remaining = profile_parser.parse_known_args(sys.argv[1:])
    if profile_args.analysis_tasks < 1:
        profile_parser.error('--analysis-tasks must be positive')
    sys.argv = [sys.argv[0], *remaining]
    if '--help' in sys.argv:
        print('Native profile option: --analysis-tasks N (default 148); independent of --validation-tasks.')
    os.environ['HARNESSFIX_NATIVE_PROFILE'] = '1'
    def supplied(name):
        return any(arg == name or arg.startswith(name + '=') for arg in sys.argv)
    def argument(name, default):
        for index, arg in enumerate(sys.argv):
            if arg == name:
                return sys.argv[index + 1]
            if arg.startswith(name + '='):
                return arg.split('=', 1)[1]
        return default
    if not supplied('--max-iterations'):
        sys.argv += ['--max-iterations', '1']
    if not supplied('--max-promotion-failures'):
        sys.argv += ['--max-promotion-failures', '2']
    if supplied('--run-dir') and '--check-only' not in sys.argv and '--help' not in sys.argv:
        from attribution_experiments.common.optimization_budget import pin_budget
        run = Path(argument('--run-dir', '')).resolve()
        pin_budget(run, {'protocol': 'harnessfix_full_analysis_v3',
                         'analysis_tasks': profile_args.analysis_tasks,
                         'max_iterations': int(argument('--max-iterations', '1')),
                         'max_promotion_failures': int(argument('--max-promotion-failures', '2')),
                         'validation_tasks': int(argument('--validation-tasks', '32')),
                         'validation_seed': int(argument('--validation-seed', '20260902')),
                         'group': argument('--ablation-group', 'baseline')},
                   has_progress=any(run.glob('iteration_*')) or (run / 'summary.json').exists())
    from . import run_native as adapter
    install(adapter, analysis_tasks=profile_args.analysis_tasks)
    adapter.main()
    if '--prepare-only' in sys.argv or '--check-only' in sys.argv:
        return
    # The legacy adapter assigns 0/148 without a new rollout if no candidate is
    # promoted. A current-runtime comparison requires evaluating the retained original.
    if not supplied('--run-dir'):
        raise RuntimeError('Native alignment runs require an explicit new --run-dir')
    run = Path(argument('--run-dir', ''))
    summary_path = run / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['optimization_budget'] = json.loads((run / 'optimization_budget.json').read_text())
    summary['analysis_task_count'] = len((run / 'analysis_ids.txt').read_text().split())
    adapter.write_json(summary_path, summary)
    if summary.get('selected_iteration') == 0:
        ids = (run / 'instance_ids.txt').read_text().split()
        optimizer_key, task_key = adapter.optimizer_api_key(), adapter.task_api_key()
        env = adapter.optimizer_env(optimizer_key, task_key)
        summary.update(status='running_original_final', final_score_is_incomplete=True)
        adapter.write_json(summary_path, summary)
        rows = adapter.run_benchmark(run / 'final', candidate_dir=run / 'original_agent', ids=ids,
                                     workers=8, env=env, api_key=task_key, resume=True)
        from attribution_experiments.common.result_validity import retryable_result
        if any(retryable_result(row) or not isinstance((row.get('evaluation') or {}).get('resolved'), bool) for row in rows):
            raise RuntimeError('Retained-original final benchmark incomplete')
        resolved = [row['instance_id'] for row in rows if row['evaluation']['resolved']]
        summary.update(status='complete', final_score_is_incomplete=False,
                       final={'status': 'complete', 'total': len(rows), 'resolved': len(resolved), 'resolved_ids': resolved, 'results': rows},
                       retained_original_evaluated=True)
        adapter.write_json(summary_path, summary)


if __name__ == '__main__':
    main()
