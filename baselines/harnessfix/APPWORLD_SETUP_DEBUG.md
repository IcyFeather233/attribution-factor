# AppWorld Setup and Debug Guide

这份文档记录当前仓库已经验证通过的 AppWorld 安装路线，以及排查失败时应该看哪些点。

当前推荐路线：

- host 环境：`/home/icyfeather/projects/HarnessFix/.venv`
- Python 包：`appworld==0.1.3.post1`
- root 目录：`/home/icyfeather/projects/appworld_root`
- Docker 运行镜像：`appworld-agent-pypi:latest`

最重要的三个验证目标：

1. `python -m appworld.cli install` 成功
2. `from appworld.environment import AppWorld` 成功
3. `verify tests` 和 `verify tasks` 成功

---

## 1. 进入当前项目目录

```bash
cd /home/icyfeather/projects/HarnessFix
pwd
```

预期输出：

```bash
/home/icyfeather/projects/HarnessFix
```

确认虚拟环境存在：

```bash
ls -la .venv
```

---

## 2. 检查 `python` 和 `pip` 是否指向同一个环境

```bash
.venv/bin/python -V
.venv/bin/python -m pip -V
```

预期：
- 两条命令都指向 `HarnessFix/.venv`

---

## 3. 安装官方 PyPI 版 AppWorld

从项目根目录执行：

```bash
.venv/bin/python -m pip install appworld
```

安装后确认：

```bash
.venv/bin/python -m pip show appworld
.venv/bin/python - <<'PY'
import appworld
print("APPWORLD_MODULE", appworld.__file__)
from appworld.environment import AppWorld
print("IMPORT_OK")
PY
```

---

## 4. 执行官方安装步骤

```bash
.venv/bin/python -m appworld.cli install
```

---

## 5. 准备 APPWORLD_ROOT

```bash
mkdir -p /home/icyfeather/projects/appworld_root
export APPWORLD_ROOT=/home/icyfeather/projects/appworld_root
```

---

## 6. 下载官方数据

注意：`appworld download data` 可能按当前工作目录落盘，所以建议直接在指定 root 下执行：

```bash
cd /home/icyfeather/projects/appworld_root
/home/icyfeather/projects/HarnessFix/.venv/bin/python -m appworld.cli download data
```

---

## 7. 验证 root 目录结构

```bash
ls -la /home/icyfeather/projects/appworld_root
ls -la /home/icyfeather/projects/appworld_root/data
```

你应该看到：

- `data/`
- 以及 `data/tasks/`, `data/api_docs/` 等子目录

---

## 8. 跑官方 verify

```bash
.venv/bin/python -m appworld.cli verify tests --root /home/icyfeather/projects/appworld_root
.venv/bin/python -m appworld.cli verify tasks --root /home/icyfeather/projects/appworld_root
```

如果这两步都通过，说明官方 AppWorld 安装和数据路线是完整可用的。

---

## 9. 构建当前项目使用的 AppWorld 镜像

当前仓库不再依赖旧的本地 bundle 路线，而是使用一个基于官方 PyPI 包的镜像：

```bash
docker build -t appworld-agent-pypi:latest \
  -f task_agent/appworld_agent/Dockerfile \
  task_agent/appworld_agent
```

---

## 10. 运行当前项目里的 AppWorld smoke

单任务 smoke：

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

10 任务 pilot：

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
