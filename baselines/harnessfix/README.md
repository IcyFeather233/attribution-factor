# HarnessFix

Trace-guided harness repair framework for long-horizon agents.

HarnessFix is a paper-aligned harness repair prototype. It keeps the original
closed-loop idea:

- run a task agent on a benchmark
- analyze failed trajectories
- consolidate recurring defects
- generate a scoped repair plan
- modify the harness
- validate with promotion decisions and best-so-far tracking

and extends it toward the design described in the draft paper:

- HTIR: harness-aware trace intermediate representation
- typed repair operators
- plan-to-diff audit
- promotion-oriented validation and best-so-far selection
- accepted/rejected repair memory
- three benchmark branches: SWE, GAIA, AppWorld

## Status

Implemented:

- SWE branch
- GAIA branch
- AppWorld branch
- HTIR + defect consolidation + operator registry + audit + memory + promotion loop

Validated locally:

- AppWorld official installation route works with:
  - `appworld==0.1.3.post1`
  - `APPWORLD_ROOT=/home/icyfeather/projects/appworld_root`
  - Docker image `appworld-agent-pypi:latest`
- Real AppWorld smoke runs:
  - single-task smoke: `1/1 solved`
  - 10-task pilot: `1/10 solved`, remaining tasks produce partial assertion progress rather than environment failures

## Repository Layout

- `run_pipeline_swe.py`: end-to-end closed loop for SWE-Bench style coding agents
- `run_pipeline_gaia.py`: end-to-end closed loop for GAIA research agents
- `run_pipeline_appworld.py`: end-to-end closed loop for AppWorld API agents
- `failure_analysis/`: HTIR, diagnosis, consolidation, operator registry, audit, validation, memory
- `task_agent/`: benchmark-specific task agents and runner scripts
- `data/`: sampled benchmark subsets and smoke subsets
- `eval/`: evaluation scripts
- `enhancement_implementation/`: modify-agent configs

## Recommended Models

- Weak task-agent model: `openai/gpt-4.1-mini`
- Stronger comparison model: `openai/gpt-4.1`
- Analysis / planning / modify model: `openai/claude-opus-4-5-20251101-thinking`

These defaults reflect the currently working gateway routes on this machine.

## Quick Start

### 1. Activate the repo environment

```bash
cd /home/icyfeather/projects/HarnessFix
source .venv/bin/activate
```

### 1.1. Install local benchmark dependencies

Install the vendored benchmark and task-agent packages in editable mode:

```bash
.venv/bin/python -m pip install -e task_agent/mini-swe-agent
.venv/bin/python -m pip install -e task_agent/open_deep_research
.venv/bin/python -m pip install -e eval/SWE-bench
```

### 2. Configure API keys

The repo loads `.env` from the project root.

Task-agent routes use:

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`

SWE task-agent routing is additionally controlled by:

- `task_agent/model_registry.json`

### 3. Prepare datasets

```bash
.venv/bin/python3 data/sample_train_test.py
.venv/bin/python3 data/sample_val.py
.venv/bin/python3 data/sample_gaia.py
.venv/bin/python3 data/sample_appworld.py
```

This creates:

- `data/verified_train_100`, `data/verified_val_50`, `data/verified_test_100`
- `data/gaia_train_100`, `data/gaia_val_50`, `data/gaia_test_100`
- `data/appworld_train_100`, `data/appworld_val_50`, `data/appworld_test_100`

The optional AppWorld smoke subset is generated on demand:

```bash
.venv/bin/python3 data/sample_appworld_smoke.py
```

GAIA split semantics:

- `gaia_train_100` and `gaia_val_50` are sampled from the official GAIA `validation` split
- `gaia_test_100` is sampled from the official GAIA `test` split
- this keeps the final GAIA test set held out from the improve-and-validate loop

### 4. Build the AppWorld image

```bash
docker build -t appworld-agent-pypi:latest \
  -f task_agent/appworld_agent/Dockerfile \
  task_agent/appworld_agent
```

### 5. Set AppWorld root

```bash
export APPWORLD_ROOT=/home/icyfeather/projects/appworld_root
```

## Running Benchmarks

### SWE

```bash
.venv/bin/python3 run_pipeline_swe.py \
  --model openai/gpt-4.1-mini \
  --analysis-model openai/claude-opus-4-5-20251101-thinking \
  --start-version 1 \
  --max-iterations 3 \
  --workers 4 \
  --cost-limit 1
```

### GAIA

```bash
.venv/bin/python3 run_pipeline_gaia.py \
  --model openai/gpt-4.1-mini \
  --analysis-model openai/claude-opus-4-5-20251101-thinking \
  --start-version 1 \
  --max-iterations 3 \
  --workers 2 \
  --concurrency 4
```

### AppWorld

```bash
APPWORLD_ROOT=/home/icyfeather/projects/appworld_root \
.venv/bin/python3 run_pipeline_appworld.py \
  --model openai/gpt-4.1-mini \
  --analysis-model openai/claude-opus-4-5-20251101-thinking \
  --start-version 1 \
  --max-iterations 3 \
  --workers 2 \
  --concurrency 4
```

When the AppWorld validation gate passes, the pipeline prints a final held-out
test command for `data/appworld_test_100`.

## Smoke Tests

### Single-task AppWorld smoke

```bash
APPWORLD_ROOT=/home/icyfeather/projects/appworld_root \
APPWORLD_AGENT_IMAGE=appworld-agent-pypi:latest \
./task_agent/run_appworld.sh \
  --subset "$(pwd)/data/appworld_val_50" \
  --filter d4e9306_3 \
  --model openai/gpt-4.1-mini \
  --workers 1 \
  --concurrency 1 \
  --output traces/appworld_smoke_d4e9306_3_v2
```

Observed local outcome:

- `d4e9306_3`: solved, `score=1.00`

### 10-task AppWorld pilot

```bash
APPWORLD_ROOT=/home/icyfeather/projects/appworld_root \
APPWORLD_AGENT_IMAGE=appworld-agent-pypi:latest \
./task_agent/run_appworld.sh \
  --subset "$(pwd)/data/appworld_pilot_10" \
  --model openai/gpt-4.1-mini \
  --workers 2 \
  --concurrency 2 \
  --output traces/appworld_pilot_10
```

Observed local outcome:

- `10/10` tasks completed
- `1/10` fully solved
- remaining tasks produced partial assertion satisfaction:
  - examples: `25%`, `33.3%`, `50%`, `62.5%`, `80%`

This is useful because it shows the AppWorld branch is no longer blocked on
environment setup; remaining issues are task-solving/prompt/harness quality.

## Documentation

- [HOW_TO_RUN.md](./HOW_TO_RUN.md): full command reference and detailed workflow
- [PAPER_ALIGNMENT.md](./PAPER_ALIGNMENT.md): what was changed to align the repo with the paper design
- [APPWORLD_SETUP_DEBUG.md](./APPWORLD_SETUP_DEBUG.md): AppWorld installation and verification debugging notes

## Notes

- The copied `.venv` originally had broken `pip` wrappers pointing to the old
  project path; this repo now fixes that.
- Installing `appworld` into the host `.venv` can conflict with `pydantic 2`
  based agent dependencies; the working runtime path for AppWorld tasks is the
  dedicated Docker image `appworld-agent-pypi:latest`.
- Some AppWorld outputs created inside Docker may be root-owned; use fresh
  output directories for new smoke runs if cleanup becomes noisy.
