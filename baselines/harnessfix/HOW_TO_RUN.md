# HarnessFix：完整操作手册

> 所有命令均从项目根目录 `/home/icyfeather/projects/HarnessFix` 执行。

---

## 目录

1. [前置条件](#0-前置条件)
2. [数据集准备](#1-数据集准备)
3. [运行 Task Agent（SWE-Bench 推理）](#2-运行-task-agentswe-bench-推理)
4. [评估结果](#3-评估结果)
5. [失败分析流水线（手动）](#4-失败分析流水线手动)
6. [闭环改进流水线（自动）— SWE-Bench](#5-闭环改进流水线自动--swe-bench)
7. [闭环改进流水线（自动）— GAIA](#6-闭环改进流水线自动--gaia)
8. [闭环改进流水线（自动）— AppWorld](#7-闭环改进流水线自动--appworld)
9. [调试单条实例](#8-调试单条实例)
10. [进阶：Singularity/Apptainer](#9-进阶singularityapptainer)
11. [费用和速率控制](#10-费用和速率控制)
12. [常用命令速查](#附常用命令速查)

---

## 0. 前置条件

### Docker

mini-swe-agent 的 SWE-Bench 模式默认使用 Docker 为每个任务提供隔离环境。

```bash
docker info
```

如果没有 Docker，可以改用 Singularity/Apptainer，见[进阶说明](#8-进阶singularityapptainer)。

### Python 虚拟环境

```bash
source .venv/bin/activate        # 激活
.venv/bin/python3 <script>       # 或直接用完整路径
```

### API Key

```bash
mini-extra config set OPENAI_API_KEY "your-key"
# 或写入 .env 文件
echo 'OPENAI_API_KEY=your-key' >> .env
```

> `OPENAI_API_BASE` 由 `run_swebench.sh` 根据 `task_agent/model_registry.json` 中的 `api_base` 字段自动设置，无需手动指定。

### 安装依赖

```bash
.venv/bin/python -m pip install -e task_agent/mini-swe-agent
.venv/bin/python -m pip install -e task_agent/open_deep_research
.venv/bin/python -m pip install -e eval/SWE-bench
```

### 支持的模型

| 用途 | 模型名 | 说明 |
|------|--------|------|
| Task Agent 推理（SWE / GAIA / AppWorld） | `openai/gpt-4.1-mini` | 当前验证通过、适合作为弱模型主线 |
| 对比用更强 Task Agent | `openai/gpt-4.1` | 与 `gpt-4.1-mini` 同家族，适合做强模型对照 |
| 失败分析 / 改进 | `openai/MODEL` | api_base: `http://localhost:8000/v1` |

---

## 1. 数据集准备

SWE-Bench Verified 共 500 条，分三组互不重叠：

| 数据集 | 大小 | 用途 |
|--------|------|------|
| `verified_train_100` | 100 条 | Task Agent 推理 + 失败分析 |
| `verified_val_50`    | 50 条  | 每轮改进后的 validation / promotion 判断 |
| `verified_test_100`  | 100 条 | 最终评测，全程不碰 |

```bash
# 生成 train_100 和 test_100（seed=42）
.venv/bin/python3 data/sample_train_test.py
# → data/verified_train_100/  data/verified_test_100/

# 生成 val_50（从剩余 300 条中采样，seed=123）
.venv/bin/python3 data/sample_val.py
# → data/verified_val_50/

# GAIA 数据集
.venv/bin/python3 data/sample_gaia.py
# → data/gaia_train_100/  data/gaia_val_50/  data/gaia_test_100/

> 说明：
> - `gaia_train_100` 和 `gaia_val_50` 来自官方 `validation` split
> - `gaia_test_100` 来自官方 `test` split
> - `gaia_test_100` 应作为最终 held-out test，不参与闭环改进

# AppWorld 数据集
.venv/bin/python3 data/sample_appworld.py
# → data/appworld_train_100/  data/appworld_val_50/  data/appworld_test_100/

# 预拉取 Docker 镜像（避免首次运行超时，每组只需一次）
./task_agent/pull_swebench_images.sh "$(pwd)/data/verified_train_100" test 4
./task_agent/pull_swebench_images.sh "$(pwd)/data/verified_val_50"    test 4
./task_agent/pull_swebench_images.sh "$(pwd)/data/verified_test_100"  test 4
```

---

## 2. 运行 Task Agent（SWE-Bench 推理）

### 基本用法

```bash
./task_agent/run_swebench.sh \
    --subset "$(pwd)/data/verified_train_100" \
    --split test \
    --model openai/Qwen3-Coder \
    --workers 4 \
    --output traces/swe_train_qwen3_coder_original \
    --cost-limit 1
```

| 参数 | 说明 |
|------|------|
| `--subset` | 数据集目录或 `verified` / `lite` |
| `--split` | 固定 `test` |
| `--model` | litellm 格式：`provider/model-name` |
| `--workers` | 并发数，建议 4-8 |
| `--output` | 输出目录（固定目录可断点续跑） |
| `--cost-limit` | 单实例费用上限（美元） |
| `--rerun-status` | 重跑指定退出状态（`Timeout` / `LimitsExceeded`） |

### 使用增强版 agent

通过 `PYTHONPATH` 切换到 enhanced 版本，无需重新安装：

```bash
PYTHONPATH=task_agent/enhanced_swe_vN/src \
./task_agent/run_swebench.sh \
    --subset "$(pwd)/data/verified_train_100" \
    --split test \
    --model openai/Qwen3-Coder \
    --workers 4 \
    --output traces/swe_train_qwen3_coder_swe_enhanced_vN \
    --cost-limit 1
```

### Val Baseline（只需跑一次）

```bash
./task_agent/run_swebench.sh \
    --subset "$(pwd)/data/verified_val_50" \
    --split test \
    --model openai/Qwen3-Coder \
    --workers 4 \
    --output traces/swe_val_qwen3_coder_original \
    --cost-limit 1
```

### 重跑未完成实例

```bash
./task_agent/run_swebench.sh \
    --subset "$(pwd)/data/verified_train_100" \
    --split test \
    --model openai/Qwen3-Coder \
    --workers 4 \
    --output traces/swe_train_qwen3_coder_original \
    --cost-limit 1 \
    --rerun-status Timeout
```

### 输出结构

```
traces/<run_name>/
├── preds.json                      # 所有预测（评测用）
├── minisweagent.log
├── exit_statuses_<timestamp>.yaml
└── <instance_id>/
    └── <instance_id>.traj.json     # 完整 trace
```

---

## 3. 评估结果

### 运行评测

```bash
mkdir -p eval/results_swe_train_qwen3_coder_original
python -m swebench.harness.run_evaluation \
    -d princeton-nlp/SWE-Bench_Verified \
    -s test \
    -p traces/swe_train_qwen3_coder_original/preds.json \
    --max_workers 4 \
    -id swe_train_qwen3_coder_original \
    --report_dir eval/results_swe_train_qwen3_coder_original \
    -i $(cat data/verified_train_100/instance_ids.txt | tr '\n' ' ')
```

| 参数 | 说明 |
|------|------|
| `-d` | 固定 `princeton-nlp/SWE-Bench_Verified` |
| `-p` | `preds.json` 路径 |
| `-id` | 评测标识符（决定日志目录名） |
| `--report_dir` | 报告写入目录（需提前 `mkdir -p`） |
| `-i` | 限定评测实例列表 |

### 查看结果

```bash
python3 -c "
import json, glob
subset = set(open('data/verified_train_100/instance_ids.txt').read().split())
r = json.load(open(glob.glob('eval/results_swe_train_qwen3_coder_original/*.json')[0]))
resolved = set(r.get('resolved_ids', [])) & subset
print(f'Resolved: {len(resolved)}/{len(subset)}')
print(f'Empty patch: {len(set(r.get(\"empty_patch_ids\",[])) & subset)}')
print(f'Unresolved:  {len(set(r.get(\"unresolved_ids\",[])) & subset)}')
"
```

### 对比两个版本

```bash
python3 -c "
import json, glob
subset = set(open('data/verified_test_100/instance_ids.txt').read().split())
r1 = json.load(open(glob.glob('eval/results_swe_train_qwen3_coder_original/*.json')[0]))
r2 = json.load(open(glob.glob('eval/results_swe_train_qwen3_coder_swe_enhanced_vN/*.json')[0]))
resolved_r1 = set(r1.get('resolved_ids',[])) & subset
resolved_r2 = set(r2.get('resolved_ids',[])) & subset
print(f'Original:    {len(resolved_r1)}/100')
print(f'Enhanced vN: {len(resolved_r2)}/100')
print(f'新解决: {resolved_r2 - resolved_r1}')
print(f'新退步: {resolved_r1 - resolved_r2}')
"
```

---

## 4. 失败分析流水线（手动）

在完成 train 推理和评测后，对失败实例进行结构化归因分析。

### 文件结构

```
failure_analysis/
├── explore_task_agent.py         # 一次性：探索 task agent 代码生成实现文档
├── run_analysis.py               # 批量分析失败 trace（SWE 和 GAIA 统一入口）
├── aggregate_results.py          # 聚类分析，生成 improvement_plan
├── check_val_gate.py             # 手动比较工具；自动 SWE pipeline 使用 promotion decision
├── extract_traj_summary.py       # 工具：压缩 traj.json（被分析 agent 调用）
├── analysis_config_swe.yaml      # SWE 分析 agent 配置（prompt、模型参数）
├── analysis_config_gaia.yaml     # GAIA 分析 agent 配置
├── task_agent_impl_doc.md        # explore_task_agent.py 生成的实现文档
└── results/
    ├── <run_id>_analysis.jsonl          # 增量写入，支持断点续跑
    ├── val_regression_analyses_vN.jsonl # 闭环流水线中的 val 回归分析
    ├── val_compare_swe_vN.json          # 各轮 val 对比结果
    ├── promotion_swe_vN.json            # 各轮 promotion 决策
    └── iteration_report_swe_vN.json     # 传给下一轮 aggregate 的完整上下文

improvement_plans/
├── swe_<model_slug>_vN.md        # 各轮 SWE 改进计划（人可读）
├── swe_<model_slug>_vN.json      # 各轮 SWE 改进计划（机器可读，供 modify 阶段消费）
├── gaia_<model_slug>_vN.md       # 各轮 GAIA 改进计划
└── gaia_<model_slug>_vN.json

enhancement_implementation/
├── config_swe.yaml               # 代码修改 agent 的配置（SWE）
├── config_gaia.yaml              # 代码修改 agent 的配置（GAIA）
└── results/
    ├── modify_agent_vN.traj.json       # SWE 修改过程 trace
    └── modify_agent_gaia_vN.traj.json  # GAIA 修改过程 trace
```

### Step 1：探索 Task Agent 代码（一次性）

```bash
.venv/bin/python3 failure_analysis/explore_task_agent.py
# 已存在时重新生成：
.venv/bin/python3 failure_analysis/explore_task_agent.py --force
```

输出 `failure_analysis/task_agent_impl_doc.md`，注入后续分析的 system prompt。

### Step 2：批量分析失败 trace

```bash
# SWE（指定路径，支持断点续跑）
.venv/bin/python3 failure_analysis/run_analysis.py \
    --mode swe \
    --traces-dir traces/swe_train_qwen3_coder_original \
    --eval-results eval/results_swe_train_qwen3_coder_original/<report>.json \
    --output-file failure_analysis/results/swe_train_qwen3_coder_original_analysis.jsonl \
    --workers 4 --resume

# GAIA
.venv/bin/python3 failure_analysis/run_analysis.py \
    --mode gaia \
    --traces-dir traces/gaia_train_gpt_4o_original \
    --eval-results eval/gaia_results_gaia_train_gpt_4o_original/results.json \
    --gaia-subset data/gaia_train_100 \
    --output-file failure_analysis/results/gaia_train_gpt_4o_original_analysis.jsonl \
    --workers 4 --resume
```

每条诊断输出字段：`instance_id` / `failure_category` / `exit_status` / `failure_manifestation` / `failure_reason` / `agent_design_issue` / `affected_component`

| 参数 | 说明 |
|------|------|
| `--mode` | `swe`（默认）或 `gaia` |
| `--model` / `-m` | 分析模型（默认 `openai/MODEL`） |
| `--traces-dir` | traces 目录 |
| `--eval-results` | 评测结果 JSON 文件 |
| `--gaia-subset` | GAIA 数据集目录（仅 `--mode gaia` 需要） |
| `--output-file` | 输出 JSONL 路径 |
| `--instance-id` / `-i` | 只分析指定实例 |
| `--instance-ids-file` | 从文件批量指定实例 ID（每行一个） |
| `--category` / `-c` | 按类别过滤 |
| `--resume` / `--no-resume` | 跳过已完成实例 / 强制重新分析 |
| `--workers` | 并行度 |

### Step 3：生成改进计划

```bash
# SWE（版本 v1）
.venv/bin/python3 failure_analysis/aggregate_results.py \
    --results-file failure_analysis/results/swe_train_qwen3_coder_original_analysis.jsonl \
    --output improvement_plans/swe_qwen3_coder_v1.md \
    --spec-output improvement_plans/swe_qwen3_coder_v1.json

# 后续轮次（带 val 回归反馈和上轮计划参考）
.venv/bin/python3 failure_analysis/aggregate_results.py \
    --results-file failure_analysis/results/swe_train_qwen3_coder_swe_enhanced_v1_analysis.jsonl \
    --output improvement_plans/swe_qwen3_coder_v2.md \
    --spec-output improvement_plans/swe_qwen3_coder_v2.json \
    --val-analyses failure_analysis/results/val_regression_analyses_v1.jsonl \
    --prev-plan improvement_plans/swe_qwen3_coder_v1.md \
    --force
```

| 参数 | 说明 |
|------|------|
| `--results-file` | 输入的 JSONL 分析结果 |
| `--output` | 输出改进计划 Markdown 路径 |
| `--spec-output` | 输出机器可读 JSON 规格路径 |
| `--val-analyses` | 上轮 val 回归分析（闭环时传入） |
| `--prev-plan` | 上轮改进计划（供 LLM 参考，避免重复错误） |
| `--force` | 覆盖已有文件 |

### Step 4：查看改动

改进计划由闭环流水线自动实施（见第 5 节），也可检查结果：

```bash
# 查看代码改动
diff -rq task_agent/mini-swe-agent/src task_agent/enhanced_swe_v1/src

# 查看修改过程 trace
cat enhancement_implementation/results/modify_agent_v1.traj.json | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('Exit:', d.get('exit_status'))
for s in d.get('trajectory',[]): print(s)
"
```

---

## 5. 闭环改进流水线（自动）— SWE-Bench

`run_pipeline_swe.py` 将 train推理 → train评测 → 失败分析 → aggregate → modify → enhanced train推理/评测/对比 → val推理 → val评测 → promotion decision 自动串联为闭环。candidate 只有通过 promotion decision 才会成为下一轮 base；失败时保留当前 base，分析 val 回归，并把完整 iteration report 传给下一轮。

### 流程图

```
BOOTSTRAP（仅首次）
  ├─ val_base_infer：原始 agent 跑 val_50
  └─ val_base_eval：swebench harness 评测

每轮迭代 (candidate vN，base = 当前 promoted 版本)
  ├─ Step 0: train_infer（原始或上一版 enhanced agent 跑 train_100）
  ├─ Step 1: train_eval（swebench harness 评测 train）
  ├─ Step 2: train_analysis → failure_analysis/results/<run_id>_analysis.jsonl
  ├─ Step 3: aggregate → improvement_plans/swe_<model_slug>_vN.md + .json
  │           首轮：仅用 train 失败分析
  │           后续：+ val 回归分析 + 上轮 plan + 上轮完整 iteration report
  ├─ Step 4: modify → task_agent/enhanced_swe_vN/
  ├─ Step 5: train_infer（enhanced_vN 跑 train_100）
  ├─ Step 6: train_eval（评测 enhanced train 结果）
  ├─ Step 6.5: train_compare（对比 base train vs enhanced train，仅记录）
  ├─ Step 7: val_infer（enhanced_vN 跑 val_50）
  ├─ Step 8: val_eval（评测 val 结果）
  ├─ Step 9: val_compare（对比当前 base vs candidate）
  ├─ Step 10: promotion decision
  │            ├─ ✓ PROMOTE → candidate 成为下一轮 base
  │            └─ ✗ NOT PROMOTED → 保留当前 base
  ├─ Step 11: iteration_report（完整记录 plan、diff、train/val 效果、promotion 结果）
  └─ Step 12: 分析 val 回归 → 下一轮；连续 promotion failure 达阈值则停止
```

### 运行闭环

```bash
.venv/bin/python3 run_pipeline_swe.py \
    --start-version 1 \
    --max-iterations 3 \
    --analysis-model openai/MODEL \
    --model openai/Qwen3-Coder \
    --workers 4 \
    --cost-limit 1
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--start-version` | 起始版本号 | 2 |
| `--max-iterations` | 最多循环轮次 | 3 |
| `--analysis-model` | 分析/改进用模型 | `openai/MODEL` |
| `--model` | task agent 推理模型 | `openai/gpt-5-mini` |
| `--workers` | 推理并发数 | 4 |
| `--cost-limit` | 单实例费用上限 | 1.0 |
| `--min-improvement` | promotion：val 净解决数最小提升 | 1 |
| `--min-target-metrics` | promotion：至少几个目标指标改善 | 1 |
| `--max-error-rate-delta` | promotion：允许 error rate 增量上限 | 0.15 |
| `--max-invalid-rate-delta` | promotion：允许 invalid submission rate 增量上限 | 0.15 |
| `--max-promotion-failures` | 连续几次未 promote 后停止 | 2 |
| `--force` | 强制重新生成所有步骤 | false |
| `--dry-run` | 打印计划不实际执行 | false |

### 手动 val 对比

不跑完整流水线时，可用兼容工具单独比较两个版本的 val 结果：

```bash
.venv/bin/python3 failure_analysis/check_val_gate.py \
    --baseline-eval eval/results_swe_val_qwen3_coder_original \
    --current-eval  eval/results_val_openai-Qwen3-Coder_swe_enhanced_vN \
    --val-ids       data/verified_val_50/instance_ids.txt \
    --output        failure_analysis/results/manual_val_compare_swe_vN.json
```

### Gate 通过后：最终 test_100 评测

```bash
# 推理
PYTHONPATH=task_agent/enhanced_swe_vN/src \
./task_agent/run_swebench.sh \
    --subset "$(pwd)/data/verified_test_100" \
    --split test \
    --model openai/Qwen3-Coder \
    --workers 4 \
    --output traces/test_qwen3_coder_swe_enhanced_vN \
    --cost-limit 1

# 评测
mkdir -p eval/results_test_qwen3_coder_swe_enhanced_vN
python -m swebench.harness.run_evaluation \
    -d princeton-nlp/SWE-Bench_Verified -s test \
    -p traces/test_qwen3_coder_swe_enhanced_vN/preds.json \
    --max_workers 4 -id test_qwen3_coder_swe_enhanced_vN \
    --report_dir eval/results_test_qwen3_coder_swe_enhanced_vN \
    -i $(cat data/verified_test_100/instance_ids.txt | tr '\n' ' ')
```

### 产物目录结构（SWE）

```
task_agent/
├── mini-swe-agent/              # 原始 task agent（只读参考）
├── enhanced_swe_v1/             # 第1轮改进版本（pipeline 自动生成）
└── enhanced_swe_v2/             # 第2轮改进版本

traces/
├── swe_train_qwen3_coder_original/
├── swe_val_qwen3_coder_original/
├── swe_train_qwen3_coder_enhanced_v1/
└── val_openai-Qwen3-Coder_swe_enhanced_v1/

eval/
├── results_swe_train_qwen3_coder_original/
├── results_swe_val_qwen3_coder_original/
├── results_swe_train_qwen3_coder_enhanced_v1/
└── results_val_openai-Qwen3-Coder_swe_enhanced_v1/

improvement_plans/
├── swe_qwen3_coder_v1.md
├── swe_qwen3_coder_v1.json
├── swe_qwen3_coder_v2.md
└── swe_qwen3_coder_v2.json

failure_analysis/results/
├── swe_train_qwen3_coder_original_analysis.jsonl
├── train_compare_swe_v1.json
├── val_compare_swe_v1.json
├── promotion_swe_v1.json
├── iteration_report_swe_v1.json
├── val_regression_analyses_v1.jsonl
└── val_regression_analyses_v2.jsonl

enhancement_implementation/results/
├── modify_agent_v1.traj.json
└── modify_agent_v2.traj.json
```

---

## 6. 闭环改进流水线（自动）— GAIA

`run_pipeline_gaia.py` 结构与 SWE 完全对应，使用 `open_deep_research` 作为 task agent。

### 主要差异

| 项目 | SWE | GAIA |
|------|-----|------|
| Task agent | `mini-swe-agent` | `open_deep_research` |
| 推理脚本 | `run_swebench.sh` | `run_gaia.sh` |
| Enhanced agent 切换 | `PYTHONPATH=...` | `AGENT_SRC_DIR=...` |
| 评测脚本 | `swebench.harness.run_evaluation` | `eval/eval_gaia.py` |
| 分析模式 | `--mode swe` | `--mode gaia` |

### 运行闭环

```bash
.venv/bin/python3 run_pipeline_gaia.py \
    --start-version 1 \
    --max-iterations 3 \
    --model openai/gpt-4.1-mini \
    --analysis-model openai/MODEL \
    --workers 2 \
    --concurrency 4
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--model` | GAIA task agent 推理模型 | `openai/gpt-4.1-mini` |
| `--workers` | 并行 task 数 | 2 |
| `--concurrency` | GAIA runner 内部并发 | 4 |
| 其余参数 | 同 SWE pipeline | — |

当 `val gate` 通过后，脚本会打印一段针对 `data/gaia_test_100` 的最终 held-out test 运行命令。

---

## 7. 闭环改进流水线（自动）— AppWorld

### AppWorld 前置环境

当前仓库的 AppWorld 分支使用：

- host 侧官方安装的 `appworld` Python 包
- `APPWORLD_ROOT=/home/icyfeather/projects/appworld_root`
- Docker 镜像 `appworld-agent-pypi:latest`

如果需要重建镜像：

```bash
docker build -t appworld-agent-pypi:latest \
  -f task_agent/appworld_agent/Dockerfile \
  task_agent/appworld_agent
```

### 运行闭环

```bash
APPWORLD_ROOT=/home/icyfeather/projects/appworld_root \
.venv/bin/python3 run_pipeline_appworld.py \
    --start-version 1 \
    --max-iterations 3 \
    --model openai/gpt-4.1-mini \
    --analysis-model openai/MODEL \
    --workers 2 \
    --concurrency 4
```

当 `val gate` 通过后，脚本会打印一段针对 `data/appworld_test_100` 的最终 held-out test 运行命令。

### 运行 5 任务 smoke

先按需生成 smoke 子集：

```bash
.venv/bin/python3 data/sample_appworld_smoke.py
```

```bash
APPWORLD_ROOT=/home/icyfeather/projects/appworld_root \
APPWORLD_AGENT_IMAGE=appworld-agent-pypi:latest \
./task_agent/run_appworld.sh \
    --subset "$(pwd)/data/appworld_smoke_5" \
    --model openai/gpt-4.1-mini \
    --workers 2 \
    --concurrency 2 \
    --output traces/appworld_smoke_batch_5
```

---

## 8. 调试单条实例

```bash
mini-extra swebench-single \
    --subset verified \
    --split test \
    --model openai/Qwen3-Coder \
    --config task_agent/mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml \
    --instance sqlfluff__sqlfluff-1625 \
    --exit-immediately

# 查看 trace
mini-extra inspector traces/<run_name>/<instance_id>/<instance_id>.traj.json
```

---

## 9. 进阶：Singularity/Apptainer

无 Docker 环境时使用：

```bash
mini-extra swebench \
    --subset verified \
    --split test \
    --model openai/Qwen3-Coder \
    --config task_agent/mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml \
    --workers 4 \
    --environment-class singularity \
    --output "$OUTPUT_DIR"
```

---

## 10. 费用和速率控制

```bash
export MSWEA_GLOBAL_COST_LIMIT=50.0   # 整体费用上限（美元）
export MSWEA_GLOBAL_CALL_LIMIT=10000  # API 调用次数上限
```

**费用估算：**

| 步骤 | 规模 | 预估费用 |
|------|------|----------|
| train_100 推理（gpt-4.1-mini） | 100 条 × $1 上限 | ~$30-50 |
| val_50 推理（每轮）| 50 条 × $1 上限 | ~$15-25 |
| 失败分析（claude-opus-thinking） | ~70 条 × $0.02 | ~$1-2 |
| aggregate + modify（claude-opus-thinking） | 1-2 次调用 | ~$0.5-1 |

---

## 附：常用命令速查

```bash
# ── 数据集准备 ────────────────────────────────────────────────────────
.venv/bin/python3 data/sample_train_test.py   # train_100 + test_100
.venv/bin/python3 data/sample_val.py          # val_50
.venv/bin/python3 data/sample_gaia.py         # gaia_train_100 + gaia_val_50 + gaia_test_100
.venv/bin/python3 data/sample_appworld.py     # appworld_train_100 + appworld_val_50 + appworld_test_100
.venv/bin/python3 data/sample_appworld_smoke.py  # optional appworld_smoke_5

# ── Task Agent 推理（SWE） ────────────────────────────────────────────
./task_agent/run_swebench.sh \
    --subset "$(pwd)/data/verified_train_100" --split test \
    --model openai/gpt-4.1-mini --workers 4 \
    --output traces/swe_train_gpt_4_1_mini_original --cost-limit 1

# ── 评测（SWE） ───────────────────────────────────────────────────────
mkdir -p eval/results_swe_train_qwen3_coder_original
python -m swebench.harness.run_evaluation \
    -d princeton-nlp/SWE-Bench_Verified -s test \
    -p traces/swe_train_qwen3_coder_original/preds.json \
    --max_workers 4 -id swe_train_qwen3_coder_original \
    --report_dir eval/results_swe_train_qwen3_coder_original \
    -i $(cat data/verified_train_100/instance_ids.txt | tr '\n' ' ')

# ── 失败分析 ──────────────────────────────────────────────────────────
.venv/bin/python3 failure_analysis/explore_task_agent.py        # 一次性
.venv/bin/python3 failure_analysis/run_analysis.py \
    --mode swe \
    --traces-dir traces/swe_train_qwen3_coder_original \
    --eval-results eval/results_swe_train_qwen3_coder_original/<report>.json \
    --output-file failure_analysis/results/swe_train_qwen3_coder_original_analysis.jsonl \
    --workers 4 --resume

# ── 闭环改进流水线（SWE） ─────────────────────────────────────────────
.venv/bin/python3 run_pipeline_swe.py \
    --start-version 1 --max-iterations 3 \
    --analysis-model openai/MODEL \
    --model openai/gpt-4.1-mini \
    --workers 4 --cost-limit 1

# ── 闭环改进流水线（GAIA） ────────────────────────────────────────────
.venv/bin/python3 run_pipeline_gaia.py \
    --start-version 1 --max-iterations 3 \
    --model openai/gpt-4.1-mini --workers 2

# ── 闭环改进流水线（AppWorld） ───────────────────────────────────────
APPWORLD_ROOT=/home/icyfeather/projects/appworld_root \
.venv/bin/python3 run_pipeline_appworld.py \
    --start-version 1 --max-iterations 3 \
    --model openai/gpt-4.1-mini \
    --analysis-model openai/MODEL \
    --workers 2 --concurrency 4

# ── 手动 val 对比（兼容工具） ─────────────────────────────────────────
.venv/bin/python3 failure_analysis/check_val_gate.py \
    --baseline-eval eval/results_swe_val_qwen3_coder_original \
    --current-eval  eval/results_val_openai-Qwen3-Coder_swe_enhanced_vN \
    --val-ids       data/verified_val_50/instance_ids.txt \
    --output        failure_analysis/results/manual_val_compare_swe_vN.json
```
