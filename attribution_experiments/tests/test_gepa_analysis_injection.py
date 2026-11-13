"""Verify GEPA treats attribution as trusted evidence in reflection."""

from attribution_experiments.gepa.run_native import build_reflection_prompt_template


def test_reflection_prompt_marks_attribution_as_trusted():
    prompt = build_reflection_prompt_template()
    assert "trusted pre-computed" in prompt
    assert "failure localization" in prompt
    assert "Prefer it over re-reading the raw trajectory" in prompt


def test_evaluator_info_puts_trusted_evidence_first():
    source = open("attribution_experiments/gepa/run_native.py").read()
    trusted = source.index("Provided Failure Analysis (Trusted Evidence)")
    inputs = source.index("info[\"Inputs\"] = {")
    assert trusted < inputs
