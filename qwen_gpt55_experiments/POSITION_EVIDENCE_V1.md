# Position evidence protocol v1

This protocol is implemented separately from all existing runs. Historical scores
remain position-plus-diagnosis experiments. No existing payload is overwritten.

## Model input

Each supplied anchor contains exactly `step` (one-based assistant step) and
`message_index` (zero-based original message). `historical_step_evidence` contains
a mechanical extraction of that assistant message, its reasoning/actions and
following observations, ending before the next assistant message. Metadata is
allowlisted: provider settings and unrelated runtime fields are not copied.

All conditions use the same 8,000-character per-anchor excerpt budget. Excerpts
carry original message indices, truncation flags, source/selection SHA-256, and an
absolute reader command for the full step. Both resolves its two anchors
independently. Shifted groups resolve the shifted positions, not the old target.
No annotated `role`, `what_went_wrong`, `evidence`, `recoverability` or
`why_inevitable_here` is included in an anchor. A raw message's conversational
`role` remains necessary to identify assistant/tool messages.

Baseline's provided information remains empty. Group3 adds the existing declared
control/data graph as an explicit additional evidence treatment, with stripped
position-only anchors; it is not claimed to be a pure-position condition.

## Native method boundaries

| Method | Delivery |
|---|---|
| GEPA | Native reflection side information includes the resolved historical step, separately from current candidate feedback; new reflection instructions do not call it a supplied diagnosis. |
| SCOPE | The native `on_step_complete` task context contains the historical step alongside separately supplied current actions/observations. |
| ReCreate shared-v246 | Dedicated result evidence reaches the native per-case prompt on both failure and success, and travels with local proposals into batch synthesis. |
| Meta-Harness | Per-task workspace JSON includes both indices and raw excerpts; native analysis instructions identify the schema and full-step reader. Full original traces remain available. |
| HarnessFix | Initial per-case diagnosis contains the resolved step. The first aggregate prompt contains compact native diagnoses and position references, so a read-and-submit command cannot bypass all diagnosis consumption. Full diagnosis/evidence files remain available. Aggregate context directories are private to each run. |

Delivery receipts attest to input transmission, not that the LLM attended to every
token, cited every hint, or that an intervention caused a score gain. Native
optimization, candidate promotion and task execution budgets remain unchanged.

## Preparation and future execution

```
native_baseline_runs/.venv/bin/python -B -m qwen_gpt55_experiments.common.position_evidence --generate
native_baseline_runs/.venv/bin/python -B -m qwen_gpt55_experiments.register_position_protocol
native_baseline_runs/.venv/bin/python -B -m qwen_gpt55_experiments.position_native --method gepa --group group2_1_causing_shifted --validate-only
```

Generation and registration refuse to overwrite frozen artifacts. Registration
only prepares a 50-cell plan; it does not start experiments. A future launch uses
`launch_registered_plan --plan runtime/.../plan.json --launch` (the actual plan is
under `qwen_gpt55_experiments/runtime/position_step_v1_20261019/plan.json`).
The explicit `position_native` entrypoint installs process-local adapters, uses
`persistent_environment`, and requires a `position_step_v1_` run name. A resume
requires a matching protocol marker and byte-equivalent JSON evidence. Old worker
entrypoints, source modules, snapshots and running experiments are unchanged.

The new protocol does not silently correct semantic anchor labels. Independent
review is still required, including the identified django-15098 last-responsibility
ambiguity. Mechanically correct indices are not proof of causal correctness.
