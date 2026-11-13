import json

import pytest

from attribution_experiments.common.remote_capacity import agent_capacity
from attribution_experiments.supervise_parallel import pending_groups, legacy_running


def test_new_group_overlaps_without_exceeding_two():
    first = dict(group='group1_1_causing', state='running')
    state = {'jobs': [first]}
    assert pending_groups(state) == ['group1_2_inevitable']
    state['jobs'].append(dict(group='group1_2_inevitable', state='running', dynamic_capacity=True))
    assert pending_groups(state) == []
    assert legacy_running(state)
    first['state'] = 'complete'
    assert not legacy_running(state)
    assert pending_groups(state) == ['group1_3_both']


def test_failure_blocks_new_groups():
    assert pending_groups({'jobs': [dict(group='group1_1_causing', state='failed')]}) == []


def test_three_groups_start_only_the_third():
    state = {'group_concurrency': 3, 'jobs': [dict(group=g, state='running')
        for g in ('group1_1_causing', 'group1_2_inevitable')]}
    assert pending_groups(state) == ['group1_3_both']
    state['jobs'].append(dict(group='group1_3_both', state='running'))
    assert pending_groups(state) == []


def test_capacity_changes_without_restarting_worker(tmp_path, monkeypatch):
    source = tmp_path / 'capacity.json'
    monkeypatch.setenv('EXPERIMENT_CONTAINER_BACKEND', 'remote_docker')
    monkeypatch.setenv('EXPERIMENT_REMOTE_CAPACITY_FILE', str(source))
    source.write_text(json.dumps({'agent_containers': 16}))
    assert agent_capacity() == 16
    source.write_text(json.dumps({'agent_containers': 24}))
    assert agent_capacity() == 24
    source.write_text(json.dumps({'agent_containers': 48}))
    assert agent_capacity() == 48
    source.write_text(json.dumps({'agent_containers': 64}))
    assert agent_capacity() == 64
    source.write_text(json.dumps({'agent_containers': 65}))
    with pytest.raises(ValueError):
        agent_capacity()
    monkeypatch.setenv('EXPERIMENT_CONTAINER_BACKEND', 'udocker')
    assert agent_capacity(8) == 8
