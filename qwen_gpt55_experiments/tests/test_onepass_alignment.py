from qwen_gpt55_experiments.common.optimization_budget import adaptation_batches, committed_resume_boundary
from qwen_gpt55_experiments.gepa.protocol import one_pass_proposals, reflection_coverage, one_pass_metric_budget
from qwen_gpt55_experiments.plan_budget_repair import jobs
import pytest


def test_onepass_plan_covers_all_five_methods_with_native_update_units():
    rows = jobs(['baseline'])
    assert {r['method'] for r in rows} == {'harnessfix', 'meta_harness', 'gepa', 'scope', 'recreate'}
    for row in rows:
        cmd = row['command']
        if row['method'] == 'harnessfix':
            assert cmd[cmd.index('--analysis-tasks') + 1] == '148'
            assert cmd[cmd.index('--max-iterations') + 1] == '1'
        else:
            assert cmd[cmd.index('--adaptation-tasks') + 1] == '148'
        if row['method'] == 'gepa':
            assert '--full-trace-pass' in cmd
            assert cmd[cmd.index('--max-candidate-proposals') + 1] == '10'
            assert cmd[cmd.index('--reflection-batch-size') + 1] == '15'
        if row['method'] == 'recreate':
            assert cmd[cmd.index('--adaptation-repeats') + 1] == '1'
        if row['method'] in ('recreate', 'scope'):
            assert cmd[cmd.index('--adaptation-batch-size') + 1] == '15'
    batches = list(adaptation_batches(list(range(148)), 15, 1))
    assert len(batches) == 10
    assert [len(batch) for _, _, batch in batches] == [15] * 9 + [13]
    assert [iid for _, _, batch in batches for iid in batch] == list(range(148))
    from qwen_gpt55_experiments.run_pipeline import command
    for method in ('gepa', 'recreate', 'scope'):
        cmd = command(method, 'baseline')
        flag = '--reflection-batch-size' if method == 'gepa' else '--adaptation-batch-size'
        assert cmd[cmd.index(flag) + 1] == '15'


def test_completed_scope_final_partial_batch_does_not_rewind():
    assert committed_resume_boundary(148, 148, 8) == 148
    assert committed_resume_boundary(147, 148, 8) == 144
    assert committed_resume_boundary(32, 148, 8) == 32
    assert committed_resume_boundary(148, 148, 15) == 148
    assert committed_resume_boundary(147, 148, 15) == 135


@pytest.mark.parametrize('minibatch,expected_proposals,pause_after,improves',
                         [(3, 50, 20, False), (15, 10, 4, False), (15, 10, 9, True)])
def test_actual_gepa_epoch_covers_148_tasks_across_resume(tmp_path, minibatch, expected_proposals, pause_after, improves):
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
    from qwen_gpt55_experiments.gepa.single_pass_sampler import SinglePassBatchSampler
    ids = [f'task-{i}' for i in range(148)]
    reflections = []
    evaluation_calls = []
    def evaluate(candidate, example):
        evaluation_calls.append((candidate, example))
        score = int(candidate.split('_')[1]) / 100.0 if improves and candidate != 'seed' else 0.0
        return score, {'Inputs': {'Task ID': example}, 'Feedback': 'Inspect the trace and improve the scaffold'}
    def reflect(prompt):
        reflections.append(prompt)
        return f'```\ncandidate_{len(reflections)}\n```'
    proposals = one_pass_proposals(148, minibatch)
    assert proposals == expected_proposals
    def run(stop=None):
        return optimize_anything(seed_candidate='seed', evaluator=evaluate, dataset=ids, valset=ids[:22],
            config=GEPAConfig(
                engine=EngineConfig(run_dir=str(tmp_path / 'engine'), max_candidate_proposals=proposals,
                                    max_metric_calls=one_pass_metric_budget(22, 148, minibatch), parallel=False,
                                    write_agent_state=True),
                reflection=ReflectionConfig(reflection_lm=reflect, reflection_minibatch_size=minibatch,
                                            batch_sampler=SinglePassBatchSampler(minibatch),
                                            skip_perfect_score=False),
                stop_callbacks=stop, merge=None, refiner=None))
    run(lambda state: state.i >= pause_after - 1)
    partial = reflection_coverage(tmp_path / 'engine', ids)
    assert partial['unique_reflection_tasks'] == pause_after * minibatch
    assert len(partial['missing_task_ids']) == 148 - pause_after * minibatch
    result = run()
    covered = reflection_coverage(tmp_path / 'engine', ids)
    assert len(reflections) == expected_proposals
    assert covered['unique_reflection_tasks'] == 148
    assert covered['task_presentations'] == 148
    assert covered['batch_sizes'] == [minibatch] * (proposals - 1) + [148 - minibatch * (proposals - 1)]
    assert covered['repeated_task_ids'] == []
    assert covered['completed_reflection_proposals'] == expected_proposals
    assert not covered['missing_task_ids']
    assert result.best_candidate == (f'candidate_{proposals}' if improves else 'seed')
    if improves:
        assert result.total_metric_calls == len(evaluation_calls) == one_pass_metric_budget(22, 148, minibatch)


def test_unpadded_budget_accounts_for_smaller_final_batch():
    assert one_pass_metric_budget(22, 148, 15) == 538
