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

## 0. 多项目并发冲突检测与端口隔离（必查）

当本机可能同时打开多个项目使用 Colab 时，**调用 `open_colab_browser_connection` 之前**，先用命令检测当前环境是否已有活跃实例，避免顶掉别的项目的 Colab 连接：

1. **检查端口与当前活跃标签页**：
   - 检查 WebSocket 端口（默认 `65036`）与 Edge CDP 端口（默认 `9333`）是否处于监听状态，查看 `%TEMP%\colab-mcp-current.json`（或 `$TMPDIR/colab-mcp-current.json`），并访问 `http://127.0.0.1:9333/json/list` 查看已打开的 Colab 页面 URL。
   - **冲突判定**：
     - 若 `9333` 已有其他项目不同的 Colab 页面在运行，而新目标页尚未在该 Edge 打开：`open_colab_browser_connection` 会**直接覆盖导航**掉已有页面！
     - 若 `9333` 已有另一个 `colab-mcp` 正连着**同一个** Notebook URL：后连的会踢断先连的（`Colab Frontend disconnected`）。
2. **如何自动避坑并修改配置隔离**：
   - 若需要与已有项目并发运行互不干扰，先通过脚本/命令探测一对**未被占用**的空闲端口（如 WS 端口 `65037`、CDP 端口 `9334`）。
   - 修改当前项目所用插件的 `mcp_config.json`（`mcpServers.sft-colab.env`），**同时设置以下三个隔离变量**（缺一不可）：
     - `COLAB_MCP_PORT`: `"65037"`（隔离本地 WebSocket 端口）
     - `COLAB_MCP_EDGE_CDP_PORT`: `"9334"`（隔离 Edge 调试端口）
     - `COLAB_MCP_EDGE_PROFILE`: `"<独立目录>"`（如 `~/.llm-sft-kit/edge-profiles/<project-name>`；**必须更换 `--user-data-dir`**，否则 Edge 会直接复用旧窗口进程并忽略新的 CDP 端口）
   - 修改 `mcp_config.json` 后，提示用户**重载/重启 `sft-colab` MCP 服务**使新环境变量生效（新 Edge Profile 首次需登录一次 Google 账号）。

## 1. 训练环境与连接顺序
0. 首次项目连接：当项目初始化时，由于未建立项目项目笔记本，使用`open_colab_browser_connection`打开默认网址，请求用户在该实例浏览器中登录账户创建笔记本，并引导配置相应的密钥（如果账号未配置需要配置，如果账号配置过需要点开密钥授权新笔记本。）。建立后将笔记本地址记录到`rules/AGENTS.md` 中的`COLAB_NOTEBOOK_URL`。（由于本mcp使用自定义profile配置浏览器，用户使用默认浏览器创建后，在连接时打开新的浏览器仍需要重复登录，所以直接通过连接打开默认网页等待用户操作。）
1. **浏览器层连接**：调用 `open_colab_browser_connection` 并传入插件 `rules/AGENTS.md` 中记录的完整 `COLAB_NOTEBOOK_URL`。**在连接成功前禁止调用其他 `sft-colab` 工具**；若连接失败应重连，多次失败须提示用户手动介入（如登录账号或打开目标 Notebook）。
2. **算力层连接与硬件切换**：
   - 切换运行时规格（如训练用 A100、推理评测切换 T4/L4 节省成本）与挂载 Google Drive 均需由用户手动操作确认。
   - 用户就绪后调用 `connect_runtime` 并运行 `check_gpu` 校验显存与显卡型号。
   - **每次切换加速器后会话都会断开**，必须重新走一遍 `open_colab_browser_connection` -> `connect_runtime` -> `check_gpu`。

## 2. Google Drive 文件同步设置与缓存隔离

1. **双向同步配置**：
   - 本地电脑使用 Google Drive 桌面端同步项目目录（云端映射路径通常形如 `/content/drive/Othercomputers/<主机名>/<项目名>`）。
   - 用户在 Colab 中挂载云盘（`drive.mount('/content/drive')`）后，将该挂载路径写入插件 `rules/AGENTS.md` 的 `COLAB_DRIVE_DIR` 与 `configs/base.yaml` 的 `project.colab_dir`。具体文件夹可以在云端搜索同名项目文件夹，如果未发现可以向用户确认。
   - 脚本修改与数据准备可在本地完成并自动同步至云端；训练产出的 `runs/` 与日志会自动回传本地。
2. **严防同步盘污染与卡死**：
   - `HF_HOME`、`TORCH_HOME`、`UNSLOTH_COMPILE_LOCATION`、`TRITON_CACHE_DIR` 必须重定向到 `/tmp` 或 `/root/.cache`。
   - 训练中的高频中间 step checkpoint 必须先写入 `/content/temp_training`，训完再保存 `checkpoints/best` 与 `checkpoints/last` 到同步目录。
3. Google drive 同笔记本每次连接新实例后都需要授权。因此建议放在单元格最上并在每次连接实例后提醒用户授权。

## 3. 授权与安全规范

- **涉密凭据走 Colab Secrets**：`HF_TOKEN`、`WANDB_API_KEY`、`DASHSCOPE_API_KEY`、`NGROK_AUTHTOKEN` 等必须存放在 Colab 左侧栏 **Secrets（钥匙图标）** 中，由脚本通过 `google.colab.userdata` 读取；严禁把密钥硬编码进 `.py` / `.yaml` 或提交到 Git。
- **公开配置与密钥隔离**：`HF_REPO_ID` 等公开仓库名可写在配置或非涉密环境变量中。
- **长耗时授权门禁**：发起长耗时远程训练前，必须先向用户确认授权（或由用户在 Colab 单元格直接执行）。

## 4. 前台执行、1/10 进度跟踪与实例释放

- **单元格前台运行**：长耗时训练必须通过 `add_code_cell` / `patch_cell` + `run_code_cell` 在 Notebook 单元格前台执行（不要挂纯后台进程而让笔记本前端空闲，否则 Colab 会回收实例）。`run_shell_command` 仅用于短平快排查。
- **1/10 间隔巡检**：训练正常启动后，根据步数与单步耗时预估总时长，默认使用 `schedule` 按约总时长的 **1/10** 间隔设置定时任务检查进度，用 `read_cell_outputs` 读取最新约 20 行（或 `tail` 日志）向用户反馈状态，避免死循环轮询。
- **安全释放实例**：确认 `meta.json`、`NOTES.md`、`logs/train.log` 落盘且 HF Hub 上传完成后，通过 `runtime.unassign()`（`train.py --auto_disconnect` 已内置）或 `shutdown_runtime` 释放算力配额。
