import pytest
import yaml
from jinja2 import Environment, StrictUndefined, UndefinedError

from attribution_experiments.recreate.shared_v2_protocol import (
    seed_scaffold, escape_literal_probe_examples, validate_scaffold_file,
)


def test_django_cli_example_is_literal_and_idempotent(tmp_path):
    scaffold = seed_scaffold()
    scaffold['instance_template'] += "\n`python3 /workspace/agent_tools/testing/django_template_probe/main.py --template '{{ value }}' --context-json '{}'`\n"
    fixed = escape_literal_probe_examples(scaffold)
    assert escape_literal_probe_examples(fixed) == fixed
    rendered = Environment(undefined=StrictUndefined).from_string(fixed['instance_template']).render(task='TASK')
    assert "--template '{{ value }}'" in rendered
    path = tmp_path / 'scaffold.yaml';path.write_text(yaml.safe_dump(scaffold))
    validate_scaffold_file(path)
    assert path.with_name('scaffold.yaml.literal_escape.json').exists()
    validate_scaffold_file(path)


def test_unknown_real_template_variable_still_fails_without_rewriting(tmp_path):
    scaffold = seed_scaffold();scaffold['instance_template'] += '\n{{ unknown_runtime_variable }}'
    path = tmp_path / 'scaffold.yaml';path.write_text(yaml.safe_dump(scaffold));before=path.read_bytes()
    with pytest.raises(UndefinedError):validate_scaffold_file(path)
    assert path.read_bytes() == before
