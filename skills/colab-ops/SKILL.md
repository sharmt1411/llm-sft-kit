---
name: colab-ops
description: >-
  Use this skill to operate Google Colab via the `sft-colab` MCP server
  (better_colab_MCP) for remote GPU training: browser/runtime connection,
  accelerator switching, Google Drive sync setup, Secrets security, foreground
  cell execution, cache isolation, 1/10-duration progress scheduling, and runtime
  shutdown.
---

# Colab Remote GPU Operations SOP (`sft-colab`)

Use this workflow whenever running, monitoring, or stopping training/evaluation jobs on Google Colab through `sft-colab` MCP tools.

## 0. 多项目并发冲突检测与端口隔离（仅在启用 Colab 且连接前检查）

当确认本项目使用 Colab、准备**调用 `open_colab_browser_connection` 之前**，先判定是否存在**真实的活跃浏览器/会话冲突**，避免误报或顶掉别的项目的 Colab 连接：

1. **检测colab-mcp多项目冲突**：
   - **检查 Edge CDP（默认 `9333`）与当前活跃标签页**：访问 `http://127.0.0.1:9333/json/list` 查看已打开的 `colab.research.google.com` 页面 URL：
     - 若 `9333` 未开启，或其中没有任何其他项目的活跃 Colab 页面：可直接使用默认配置连接，无需隔离端口。
     - 若 `9333` 已有**其他项目正在运行的另一份 Colab Notebook 页面**：`open_colab_browser_connection` 会**直接覆盖导航**掉已有页面！
     - 若 `9333` 已有另一个 `colab-mcp` 正连着**同一个** Notebook URL：后连的会踢断先连的（`Colab Frontend disconnected`）。
2. **仅在确认冲突时配置端口隔离**：
   - 先通过命令探测一对**未被占用**的空闲端口（如 WS 端口 `65037`、CDP 端口 `9334`）。
   - 修改当前项目所用插件的 `mcp_config.json`（`mcpServers.sft-colab.env`），**同时设置以下三个隔离变量**（缺一不可）：
     - `COLAB_MCP_PORT`: `"65037"`（隔离本地 WebSocket 端口）
     - `COLAB_MCP_EDGE_CDP_PORT`: `"9334"`（隔离 Edge 调试端口）
     - `COLAB_MCP_EDGE_PROFILE`: `"<独立目录>"`（如 `~/.llm-sft-kit/edge-profiles/<project-name>`；**必须更换 `--user-data-dir`**，否则 Edge 会直接复用旧窗口进程并忽略新的 CDP 端口）
   - 修改 `mcp_config.json` 后，提示用户**重载/重启 `sft-colab` MCP 服务**使新环境变量生效（新 Edge Profile 首次需登录一次 Google 账号）。

## 1. 训练环境与连接顺序
1. **浏览器层连接**：调用 `open_colab_browser_connection` 并传入插件 `rules/AGENTS.md` 中记录的完整 `COLAB_NOTEBOOK_URL`。**在连接成功前禁止调用其他 `sft-colab` 工具**；若连接失败应重连，多次失败须提示用户手动介入（如登录账号或打开目标 Notebook）。
2. **算力层连接与硬件切换**：
   - 切换运行时规格（如训练用 A100、推理评测/冒烟用 L4/T4）与挂载 Google Drive 均需由用户手动操作确认。
   - 用户就绪后调用 `connect_runtime` 并运行 `check_gpu` 校验显存与显卡型号。
   - **每次切换加速器或重建运行时后会话都会断开**，必须重新走一遍 `open_colab_browser_connection` -> `connect_runtime` -> `check_gpu` -> 依赖校验。
3. **运行时依赖校验与按需安装（每次新建/切换运行时必查）**：
   - Colab 默认镜像通常预装了 `torch`/`transformers`，但**缺少 `unsloth`、`trl`、`bitsandbytes` 等训练依赖**，且每次重建/切换 GPU 运行时都会清空非预装包。
   - `connect_runtime` 就绪后，先执行一次轻量 import 探测：
     ```bash
     python -c "import unsloth, trl, peft, bitsandbytes, datasets, accelerate, yaml, huggingface_hub; print('deps ok')"
     ```
   - 若缺包，再按需执行 Colab 兼容安装（注意避免破坏 Colab 原生 `torch`/CUDA 版本匹配，例如按 Unsloth 官方 Colab 方式安装或仅补装缺失库），验证 import 通过后再运行训练脚本。

## 2. Google Drive 文件同步设置与缓存隔离

1. **双向同步与云端路径自动探测**：
   - 本地电脑使用 Google Drive 桌面端同步项目目录（云端映射路径通常形如 `/content/drive/Othercomputers/<云端设备名>/<项目名>`，其中 `<云端设备名>` 常与当前 Windows `hostname` 不一致。
   - 用户在 Colab 中挂载云盘（`drive.mount('/content/drive')`）并连上运行时后，若 `COLAB_DRIVE_DIR` 尚未确定，直接通过 `run_shell_command` 在云端自动探测并校验 `configs/base.yaml`：
     ```bash
     find /content/drive -maxdepth 4 -type d -name "<项目目录名>" ! -path "*/.Trash*"
     ```
   - 确认目标目录下存在 `configs/base.yaml` 后，自动将结果回填至插件 `rules/AGENTS.md` 的 `COLAB_DRIVE_DIR` 与 `configs/base.yaml` 的 `project.colab_dir`（若发现多个同名目录再向用户确认）。
   - 脚本修改与数据准备可在本地完成并自动同步至云端；训练产出的 `runs/` 与日志会自动回传本地。
2. **严防同步盘污染与卡死**：
   - `HF_HOME`、`TORCH_HOME`、`UNSLOTH_COMPILE_LOCATION`、`TRITON_CACHE_DIR` 必须重定向到 `/tmp` 或 `/root/.cache`。
   - 训练中的高频中间 step checkpoint 必须先写入 `/content/temp_training`，训完再保存 `checkpoints/best` 与 `checkpoints/last` 到同步目录。
3. Google Drive 同笔记本每次连接新实例后都需要授权。因此建议放在单元格最上并在每次连接实例后提醒用户授权。

## 3. 授权与安全规范（含 `userdata` 子进程传递机制）

- **涉密凭据走 Colab Secrets**：`HF_TOKEN`、`WANDB_API_KEY`、`DASHSCOPE_API_KEY`、`NGROK_AUTHTOKEN` 等必须存放在 Colab 左侧栏 **Secrets（钥匙图标）** 中；严禁把密钥硬编码进 `.py` / `.yaml` 或输出到日志/对话。
- **重要机制限制（`userdata` 与 `!python` 子进程割裂）**：`google.colab.userdata.get(...)` 依赖 Notebook IPython 内核与 Colab 前端的 Jupyter Comms 通道；当通过 `!python scripts/train.py` 启动独立子进程时，子进程**无法直接访问 `userdata`**。因此必须在 Notebook 启动单元格的内核进程中先把需要的 Secret 注入 `os.environ`，子进程 `train.py` 即可通过 `os.environ` 自动继承。
- **公开配置与密钥隔离**：`HF_REPO_ID` 等公开仓库名可写在配置或非涉密环境变量中。
- **长耗时授权门禁**：发起长耗时远程训练前，必须先向用户确认授权（或由用户在 Colab 单元格直接执行）。

## 4. 前台执行（标准启动单元格）、1/10 进度跟踪与实例释放

- **单元格前台运行**：长耗时训练必须通过 `add_code_cell` / `patch_cell` + `run_code_cell` 在 Notebook 单元格前台执行（不要挂纯后台进程而让笔记本前端空闲，否则 Colab 会回收实例）。`run_shell_command` 仅用于短平快排查。
- **标准启动单元格模板**（先在内核注入 Secrets 到 `os.environ`，再切换目录并前台拉起 `!python`）：
  ```python
  import os
  from google.colab import userdata

  for _k in ("HF_TOKEN", "WANDB_API_KEY", "WAN_DB_ENTITY", "DASHSCOPE_API_KEY", "NGROK_AUTHTOKEN"):
      if not os.environ.get(_k):
          try:
              _v = userdata.get(_k)
              if _v:
                  os.environ[_k] = str(_v).strip()
          except Exception:
              pass

  os.environ["COLAB_PROJECT_DIR"] = "<COLAB_DRIVE_DIR>"
  %cd <COLAB_DRIVE_DIR>
  !python scripts/train.py --exp_config configs/exp_0001_smoke_test.yaml --exp_name exp0001_smoke_test
  ```
- **1/10 间隔巡检**：训练正常启动后，根据步数与单步耗时预估总时长，默认使用 `schedule` 按约总时长的 **1/10** 间隔设置定时任务检查进度，用 `read_cell_outputs` 读取最新约 20 行（或 `tail` 日志）向用户反馈状态，避免死循环轮询。
- **安全释放实例**：确认 `meta.json`、`NOTES.md`、`logs/train.log` 落盘且 HF Hub 上传完成后，通过 `runtime.unassign()`（`train.py --auto_disconnect` 已内置）或 `shutdown_runtime` 释放算力配额。
