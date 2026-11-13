import pytest

from qwen_gpt55_experiments.supervise_remote import selected_groups_for_adoption


def test_adopt_uses_full_persisted_selection_including_not_started_groups():
    groups = ['group3_1_causing_flow_context', 'group3_2_inevitable_flow_context', 'group3_3_both_flow_context']
    state = {'selected_groups': groups, 'jobs': [{'group': groups[0]}]}
    assert selected_groups_for_adoption(None, state) == groups
    assert selected_groups_for_adoption(groups, state) == groups
    with pytest.raises(RuntimeError, match='do not match'):
        selected_groups_for_adoption(groups[:1], state)


def test_legacy_manifest_deduplicates_recorded_groups():
    assert selected_groups_for_adoption(None, {'jobs': [{'group': 'a'}, {'group': 'a'}, {'group': 'b'}]}) == ['a', 'b']
    with pytest.raises(RuntimeError, match='no recorded groups'):
        selected_groups_for_adoption(None, {'jobs': []})


def test_meta_benchmark_does_not_score_infrastructure_failure(monkeypatch, tmp_path):
    from qwen_gpt55_experiments.meta_harness import run_native as meta
    seen = []
    def task(**kwargs):
        iid = kwargs['instance']['instance_id']
        seen.append(iid)
        if iid == 'bad':
            return {'exit_status': 'RemoteInfrastructureError', 'exception': 'remote failed'}
        return {'evaluation': {'resolved': True}}
    monkeypatch.setattr(meta, 'run_task', task)
    with pytest.raises(RuntimeError, match='bad'):
        meta.benchmark_candidate(instances=[{'instance_id': x} for x in ['good', 'bad']],
                                 scaffold={'agent': {}, 'model': {}}, agent_class=None, assets_dir=None,
                                 output_dir=tmp_path, api_key='test', step_limit=250, workers=2)
    assert sorted(seen) == ['bad', 'good']
