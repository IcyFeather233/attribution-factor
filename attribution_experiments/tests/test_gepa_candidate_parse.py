"""Verify GEPA candidate parsing tolerates a trailing brace and fences."""

import pytest

from attribution_experiments.gepa.run_native import parse_candidate


def test_trailing_brace_is_stripped():
    candidate = "updates:\n  agent: {}\n  model: {}\n}"
    parsed = parse_candidate(candidate)
    assert parsed == {"agent": {}, "model": {}}


def test_fenced_block_is_stripped():
    candidate = "```yaml\nupdates:\n  agent: {}\n  model: {}\n```"
    parsed = parse_candidate(candidate)
    assert parsed == {"agent": {}, "model": {}}


def test_invalid_yaml_still_raises():
    with pytest.raises(ValueError):
        parse_candidate("updates: [not a mapping")


def test_unwrapped_agent_model_is_wrapped():
    candidate = (
        "agent:\n"
        "  instance_template: '{{task}}'\n"
        "model: {}\n"
    )
    parsed = parse_candidate(candidate)
    assert parsed == {"agent": {"instance_template": "{{task}}"}, "model": {}}


def test_top_level_model_is_merged_into_updates():
    candidate = (
        "updates:\n"
        "  agent:\n"
        "    instance_template: '{{task}}'\n"
        "model: {}\n"
    )
    parsed = parse_candidate(candidate)
    assert parsed == {"agent": {"instance_template": "{{task}}"}, "model": {}}
