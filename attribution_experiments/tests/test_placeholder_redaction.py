import importlib
import os

import pytest


TARGET = os.environ.get('REDACTION_TEST_TARGET', 'attribution_experiments.common.runner')


def redactor():
    module = importlib.import_module(TARGET)
    return getattr(module, 'redact_path', None) or module.redact_secret


@pytest.mark.parametrize('key', ['EMPTY', '', None])
def test_nonsecret_placeholder_leaves_test_identifiers_and_bytes_unchanged(tmp_path, key):
    path = tmp_path / 'result.json'
    original = '{"test":"EMPTY_CHANGELIST_VALUE", "api_key":"<REDACTED>"}\n'
    path.write_text(original)
    before = path.stat().st_mtime_ns
    redactor()(path, key)
    assert path.read_text() == original
    assert path.stat().st_mtime_ns == before


def test_real_secret_still_redacted_even_if_it_contains_placeholder_word(tmp_path):
    path = tmp_path / 'result.json'
    secret = 'test-secret-EMPTY-98765'
    path.write_text('{"test":"EMPTY_CHANGELIST_VALUE","api_key":"' + secret + '"}')
    redactor()(path, secret)
    assert path.read_text() == '{"test":"EMPTY_CHANGELIST_VALUE","api_key":"<REDACTED>"}'


def test_recreate_bulk_redaction_preserves_nonsecret_artifacts(tmp_path):
    if TARGET != 'attribution_experiments.recreate.run_native':
        pytest.skip('Bulk entry point belongs to ReCreate')
    module = importlib.import_module(TARGET)
    (tmp_path / 'nested').mkdir()
    path = tmp_path / 'nested/report.json'
    path.write_text('EMPTY_CHANGELIST_VALUE; actual-test-key')
    module.redact_api_key(tmp_path, 'EMPTY')
    assert path.read_text() == 'EMPTY_CHANGELIST_VALUE; actual-test-key'
    module.redact_api_key(tmp_path, 'actual-test-key')
    assert path.read_text() == 'EMPTY_CHANGELIST_VALUE; <REDACTED>'
