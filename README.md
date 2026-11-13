# Attribution Factor: harness-evolution baselines

This repository collects the code used to study failure-attribution evidence in five harness-evolution methods: GEPA, SCOPE, ReCreate, Meta-Harness and HarnessFix. It includes the modified upstream implementations, the experiment adapters, and the baseline plus nine attribution conditions. No experiment results, private traces, API keys, model weights, containers or benchmark datasets are included.

The adapters are research code, not a benchmark-independent turnkey framework. They were developed for SWE-bench-style coding tasks and mini-swe-agent v2.4.6 trajectories. A new attribution dataset must be converted to the schema below. A different task environment or agent harness also needs a new execution/evaluation adapter and a review of each method's update target and prompts. Do not interpret copying a dataset as supporting an arbitrary agent automatically.

## Layout

- `baselines/`: local upstream code, including project modifications. Individual projects retain their own license files. The HarnessFix repository has no top-level license in the source snapshot; check its upstream terms before redistribution or reuse.
- `agent_runtime/mini-swe-agent-v2.4.6/`: shared task agent used by the aligned experiment.
- `attribution_experiments/{gepa,scope,recreate,meta_harness,harnessfix}/`: method adapters and injection points.
- `attribution_experiments/common/analysis_payloads.py`: generation of the ten conditions.
- `attribution_experiments/position_native.py` and `common/position_evidence.py`: position-only protocol and native adapters. `POSITION_EVIDENCE_V1.md` describes the original run; paths and 148-task instructions there are historical.
- `attribution_experiments/tests/`: offline checks; some historical recovery tests assume the original project and data.

| Condition | Supplied evidence |
| --- | --- |
| baseline | none |
| 1.1 / 1.2 / 1.3 | causing / inevitable / both anchors |
| 2.1 / 2.2 / 2.3 | independently shifted causing / inevitable / both anchors |
| 3.1 / 3.2 / 3.3 | causing / inevitable / both anchors plus their control-flow and data-flow windows |

The legacy protocol includes annotation fields such as `what_went_wrong` and `evidence`. Shifted anchors retain these fields, so a shifted number does not remove the original diagnosis. The newer position protocol removes those annotations and resolves the selected position against the original trace. Flow conditions additionally include supplied graph evidence. These protocols answer different questions and their scores should not be silently merged.

## Prepare a dataset

Use Python 3.11 for HarnessFix: its preserved experiment adapter is distributed as Python 3.11 bytecode (`run_native_impl_bounded_validation.cpython-311.pyc`); the source of that exact implementation was not present in the original workspace. The included upstream HarnessFix source is separate from this adapter. Other packages have their own dependencies listed in the corresponding baseline's `pyproject.toml` or requirements file. Running benchmarks additionally requires SWE-bench, a compatible Docker backend and the original model/tool dependencies.

Provide the following external files, kept out of this repository:

1. A UTF-8 task-ID file, one unique ID per line. IDs must consist of letters, digits, `.`, `_` and `-` for the position protocol.
2. `<analysis-dir>/<id>.analysis.json` for each ID, with `schema_version: "2.0"`, `classification: "confirmed_agent_failure"`, a `trace.path` pointing to its trace JSON, and `step_attribution.failure_causing_wrong_step` and `step_attribution.failure_inevitable_step`. Each anchor must contain integer `step` and `message_index`; the legacy protocol can also contain diagnosis fields. Both anchors are required.
3. Each trace is a JSON object with a `messages` array. `step` counts assistant messages starting at 1; `message_index` is the zero-based index into the full array. The selected indices must agree. Tool/action fields must follow the mini-swe-agent message format for the corresponding agent adapter to consume them faithfully.
4. `<graph-dir>/<id>.graph.json`, even if initially generating only non-flow conditions. It must have `schema_version: "1.0"`, `task_id`, `canonical_scaffold_version`, and `windows` for both anchor names. Each window has `center_step`, contiguous `window_steps` spanning at most three steps either side, and `control_flow` and `data_flow` objects with `nodes` and `edges`. Node identifiers must be present as `node_id`. These graphs should be independently produced from the trace and actual agent scaffold; this repository does not infer or validate their semantic correctness.
5. For execution, a SWE-bench-compatible parquet dataset with `instance_id`, `repo`, `problem_statement` and the fields required by the benchmark/evaluator, plus matching container images. Merely generating evidence does not require parquet or a model API.
6. For execution, a reference mini-swe-agent trace containing `info.config.agent`, `info.config.model`, and `info.config.environment`, used to initialize the unmodified task scaffold. It must correspond to the agent being evaluated and should not expose credentials.

Example configuration (use your own persistent paths):

```bash
export ATTRIBUTION_TASK_IDS=/path/to/task_ids.txt
export ATTRIBUTION_ANALYSIS_ROOT=/path/to/analysis
export ATTRIBUTION_GRAPH_ROOT=/path/to/graphs
export ATTRIBUTION_PARQUET=/path/to/tasks.parquet
export ATTRIBUTION_REFERENCE_TRACE=/path/to/reference.traj.json
export ATTRIBUTION_PAYLOAD_ROOT="$PWD/attribution_experiments/payloads"
python -m attribution_experiments.generate_payloads
python -m attribution_experiments.common.position_evidence --generate
```

Generation of the legacy payloads overwrites files in its output directory; use a fresh output directory. The position generator refuses to overwrite its output. It reads legacy payloads from `ATTRIBUTION_PAYLOAD_ROOT` and verifies each index against the trace. The position manifest freezes hashes of traces and payloads, so do not move or alter inputs when resuming. The graph semantics and attribution labels still require human or independent review.

## Running experiments

`python -m attribution_experiments.position_native --method METHOD --group GROUP --validate-only` checks position evidence and prints the method arguments without calling a model. It requires at least three task IDs and a generated position manifest. For an actual run, configure `OPTIMIZER_API_KEY`, `TASK_MODEL_API_KEY`, `OPTIMIZER_BASE_URL`, `OPTIMIZER_MODEL`, `TASK_MODEL_BASE_URL`, and `TASK_MODEL`; install the dependencies for each upstream method and the SWE-bench backend. The default names and localhost endpoints are placeholders. For an endpoint that requires a placeholder key, explicitly set the corresponding key to `EMPTY`. Provider-specific request fields can be supplied as a JSON object in `TASK_MODEL_EXTRA_BODY`; they are not sent by default. The current ReCreate router identifies the two endpoints by model name, so use distinct model identifiers for its task and optimizer endpoints. Review the method-specific options and promotion rules before running.

The execution path still contains project-specific assumptions: `/volume/med` persistent storage, `server241` remote Docker, the `native_baseline_runs/.venv` layout in some workers, SWE-bench image naming, and mini-swe-agent scaffolds. The older scripts `plan_budget_repair`, `register_position_protocol`, and dated recovery scripts describe particular historical runs and are not portable launchers. Do not run them unchanged on a new dataset. An end-to-end new-dataset benchmark has not been validated in this extraction. Assess data splits and avoid treating overlapping adaptation/validation tasks as held-out test data.

The GEPA, SCOPE and ReCreate adapters use batches; HarnessFix and Meta-Harness use outer iterations. The 148-task historical configuration used ten batches for the former and one outer iteration for the latter. Budget comparability is about task coverage, not an identical number of model calls.

## Provenance

The local snapshots originated at GEPA `65130973`, SCOPE `4dc0da5e`, ReCreate `442c77f2`, Meta-Harness `44b99421`, and HarnessFix `831913a5`. They contain local experiment modifications and are not identical to those commits. See upstream licenses and citations in each baseline directory. The original author and upstream retain their respective rights.
