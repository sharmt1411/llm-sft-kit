# LLM SFT Kit

一套面向**对话类大模型 4-bit LoRA/SFT 微调**的可复用工程插件包（尤其适合多轮业务/客服对话微调场景），整合了标准实验脚手架、ChatML 数据清洗与纯回复 Loss 训练模板、Colab MCP 算力调度 SOP、OpenAI SSE 推理服务部署，以及可选的外置 [`eval-platform`](https://github.com/sharmt1411/eval-platform) 命令行对接。

依托本地 Coding Agent 与 Colab 算力调度（默认兼容 Antigravity 开箱即用，其他平台可通过 AI 辅助迁移），可在单卡（如 Colab L4）上完成 4-bit 模型（如 `Qwen3.8-27B` 等）的轻量微调与验证。

外部关联仓库：
- **Colab-mcp-tool (`better_colab_MCP`)**: https://github.com/sharmt1411/better_colab_MCP
- **eval-platform（可选外置评测后端）**: https://github.com/sharmt1411/eval-platform

## 适用范围与能力边界

为避免预期偏差，本项目各模块的定位与边界如下：

1. **插件本体专注领域（训练与部署）**
   - **适用**：基于 Unsloth 的 **4-bit QLoRA 纯回复 Loss 微调**（标准 ChatML `messages` 格式，内置多轮对话按回复回合切分展开、可选 `<think>` 剥离与按数据源权重过采样）、Colab 远程调度监控，以及单/双卡 OpenAI 兼容 SSE 推理部署。
   - **不涵盖**：全参数微调、多机多卡分布式训练、DPO/RLHF/GRPO 等强化学习对齐，以及多模态训练；插件自身也不内置评测代码。
2. **外置 `eval-platform` 能力边界（可选选装）**
   - **常规文本评测（`benchmark run`）**：适用于带有明确 `reference`、`rubric` 或 `regex_constraints` 的考题式数据集，做多模型横向对比与分歧诊断。
   - **模拟对弈、规则质检与飞轮修复（`simulation` / `flywheel`）**：采用“模拟客户对弈 + SOP 状态机 + 红线 + FAQ”机制，**主要针对流程型多轮客服/业务对话质检与数据修复**，并不适用于代码沙盒执行、数学符号推导或通用 Agent 工具链等普遍领域的模型质检。

## 目录结构

```text
llm-sft-kit/
├── plugin.json                 # 插件清单
├── mcp_config.json             # 接入 better_colab_MCP（服务名 sft-colab，已启用 --core-only）
├── hooks.json                  # SessionStart 环境探测：控制 eval-platform 技能门控
├── rules/AGENTS.md             # 实验目录、增量配置、训后记录与云端缓存隔离规范
├── scripts/
│   └── check_runtime_env.py    # 探测 evalctl 与外置评测平台后端连通状态
└── skills/
    ├── sft-init/               # 建项期使用：参数收集、目录生成、回填插件 rules/AGENTS.md 槽位与环境验收
    ├── sft-pipeline/           # 日常实验闭环：ChatML 数据准备、4-bit LoRA SFT 训练与归档复盘
    ├── sft-serving/            # 单卡/双卡 4-bit + LoRA OpenAI SSE 服务与隧道暴露
    ├── colab-ops/              # sft-colab MCP 调度、Drive/环境自检与长训练监控 SOP
    └── eval-platform/          # 外置 evalctl 对接技能（常规题库评测，以及面向 SOP 多轮对话的对弈/质检/飞轮）
```

## 前置依赖与选装说明

1. **基础环境**：Python >= 3.12；`sft-colab` MCP 通过 `uvx` 自动拉取 `better_colab_MCP`（已开启 `--core-only` 仅暴露 16 个核心工具），要求系统已安装 [`uv`](https://docs.astral.sh/uv/)。
2. **Colab 调度（可选）**：本机安装 Microsoft Edge，默认通过 `COLAB_MCP_EDGE_CDP_PORT=9333` 连接。
3. **外置评测平台 `eval-platform`（可选）**：
   - 插件仅内置 `evalctl` 操作指南与门控脚本，不包含评测后端源码；是否额外安装 `evalctl` CLI 并启动后端（默认 `:9021`）由使用方自行决定。
   - 会话启动时 `SessionStart` 钩子会自动探测环境：若 `evalctl` 与后端就绪则放行对接流程；若未安装、服务未启动，或在项目根目录放置了 `.eval_disabled`，插件会明确禁用 `eval-platform` 技能，且不会降级生成临时本地评测脚本。
4. **Google Drive（可选）**：推荐通过 Google Drive 桌面端同步本机与 Colab 云端工作目录，方便本地 Agent 直接修改代码并由 Notebook 单元格执行，减少传输与读取开销；如不使用 Drive，也可自行改用 Git 或 Hugging Face Hub 同步。

## 安装与挂载方式

- **方式一（推荐分发）：放入全局插件目录**
  ```bash
  git clone https://github.com/sharmt1411/llm-sft-kit.git ~/.gemini/config/plugins/llm-sft-kit
  ```
- **方式二：项目级挂载（在目标项目根目录执行，已有 `.agents/plugins` 目录直接复用，没有则新建）**
  - **macOS / Linux (`bash`)**：
    ```bash
    mkdir -p .agents/plugins
    if [ -d .agents/plugins/llm-sft-kit/.git ]; then
      git -C .agents/plugins/llm-sft-kit pull --ff-only
    elif [ ! -d .agents/plugins/llm-sft-kit ]; then
      git clone https://github.com/sharmt1411/llm-sft-kit.git .agents/plugins/llm-sft-kit
    fi
    ```
  - **Windows (`PowerShell`)**：
    ```powershell
    New-Item -ItemType Directory -Force .agents\plugins | Out-Null
    if (Test-Path .agents\plugins\llm-sft-kit\.git) {
      git -C .agents\plugins\llm-sft-kit pull --ff-only
    } elseif (-not (Test-Path .agents\plugins\llm-sft-kit)) {
      git clone https://github.com/sharmt1411/llm-sft-kit.git .agents\plugins\llm-sft-kit
    }
    ```
  - Windows 下请使用实目录，勿使用目录联接（`mklink /J`）。若后续不想把插件副本提交进目标项目 Git，可在目标项目 `.gitignore` 中添加 `.agents/plugins/`。

安装完毕后，让 Agent 读取插件技能即可开始项目初始化、数据准备、微调训练与服务部署；若你的场景同时涉及带评分标准的文本评测，或基于 SOP 规则的多轮对话质检与数据迭代，再按需接入外置 `eval-platform`。
