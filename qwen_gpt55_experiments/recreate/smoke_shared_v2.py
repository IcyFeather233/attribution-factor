"""Bounded infrastructure smoke: one real task, at most three Qwen calls, no optimizer calls."""
from datetime import datetime, timezone
import json
import os

from .shared_v2_protocol import ROOT, PROTOCOL
from qwen_gpt55_experiments.common.udocker_runtime import persistent_environment


def main():
    os.environ.update(persistent_environment())
    os.environ.update(EXPERIMENT_CONTAINER_BACKEND='remote_docker', EXPERIMENT_REMOTE_HOST='server241')
    os.environ.setdefault('EXPERIMENT_REMOTE_CAPACITY_FILE', str(
        ROOT / 'qwen_gpt55_experiments/runtime/remote_sweeps/remote_round1_20260909T063057Z/capacity.json'))
    from . import run_native as legacy
    from .shared_v2_adapter import install
    from evolve_utils.scaffold_ops import load_scaffold
    install(legacy)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = ROOT / 'qwen_gpt55_experiments/runtime/shared_v2_smoke' / stamp
    root.mkdir(parents=True)
    adapter = legacy.UdockerSWEBenchAdapter(step_limit=3, analysis_payloads={})
    legacy.initialize_workspace(workspace=root / 'workspace', runs_recreate_dir=root / 'optimizer_logs',
                                adapter=adapter, smoke=True)
    instance = next(x for x in legacy.load_instances() if x.instance_id == 'psf__requests-2931')
    output = root / instance.instance_id
    output.mkdir()
    base = root / 'workspace/global_v000'
    legacy.prepare_final_task_workspace(base, output)
    wrapper = None
    try:
        status, patch, container, wrapper = adapter.run_agent(
            instance, load_scaffold(base), legacy.TASK_MODEL, output, output, 1.0)
        grade = adapter.evaluate(container, instance, output / 'evaluation')
        legacy.save_traj(wrapper.agent, output / 'trajectory.json', exit_status=status,
                         result=patch, instance_id=instance.instance_id)
        data = json.loads((output / 'trajectory.json').read_text())
        assert data['info']['mini_version'] == '2.4.6'
        assert 0 < data['info']['model_stats']['api_calls'] <= 3
        assert any(m.get('tool_calls') for m in data['messages'])
        assert isinstance(grade.success, bool)
        result = {'state': 'smoke_passed', 'protocol': PROTOCOL, 'task': instance.instance_id,
                  'task_runtime': data['info']['mini_version'], 'api_calls': data['info']['model_stats']['api_calls'],
                  'optimizer_calls': 0, 'exit_status': status, 'evaluator_returned': True,
                  'resolved': grade.success, 'scope': 'Interface smoke only; not an experimental score',
                  'output_dir': str(root)}
        (root / 'smoke_result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        if wrapper:
            wrapper.cleanup()


if __name__ == '__main__':
    main()
