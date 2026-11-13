from qwen_gpt55_experiments.launch_alignment_suite import jobs


def test_all_conditions_registered_before_scores_and_three_controls():
    planned = jobs()
    assert len(planned) == 20 and len({j['id'] for j in planned}) == 20
    hf = [j for j in planned if j['method'] == 'harnessfix']
    assert len(hf) == 11
    one = [j for j in hf if j['command'][j['command'].index('--max-iterations') + 1] == '1']
    assert len(one) == 10 and len({j['group'] for j in one}) == 10
    assert len([j for j in planned if j['method']=='unmodified_control']) == 3
    assert all(j['state']=='queued' for j in planned)
