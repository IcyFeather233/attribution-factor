"""Shared capacity for new remote workers during non-disruptive adoption."""

MAX_AGENT_CONTAINERS = 64
import json
import os
from pathlib import Path


def agent_capacity(default=16):
    path = os.environ.get('EXPERIMENT_REMOTE_CAPACITY_FILE')
    if os.environ.get('EXPERIMENT_CONTAINER_BACKEND') != 'remote_docker' or not path:
        return default
    source = Path(path).resolve()
    if not source.is_relative_to('/volume/med'):
        raise ValueError('Capacity configuration must use persistent storage')
    count = json.loads(source.read_text())['agent_containers']
    if type(count) is not int or not 1 <= count <= MAX_AGENT_CONTAINERS:
        raise ValueError(f'Remote agent capacity must be in 1..{MAX_AGENT_CONTAINERS}')
    return count
