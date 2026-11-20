# LLM SFT Kit

一套面向大模型 LoRA/SFT 微调项目的可复用工程插件包，整合了标准实验脚手架、数据处理与纯回复 Loss 训练模板、Colab MCP 算力调度 SOP、OpenAI SSE 推理服务部署，以及可选的全功能 `eval-platform` 闭环对接。

可以让本地agent利用colab算力实现小参数模型如qwen3.8-27b的微调。
目前兼容antigravity，可以直接使用。其他平台需要ai辅助迁移。

其中 `eval-platform` 与 `Colab-mcp-tool`（`better_colab_MCP`）详见：
- **eval-platform**: https://github.com/sharmt1411/eval-platform
- **Colab-mcp-tool (`better_colab_MCP`)**: https://github.com/sharmt1411/better_colab_MCP

## 目录结构

```text
llm-sft-kit/
├── plugin.json                 # 插件清单
├── mcp_config.json             # 接入 better_colab_MCP（服务名 sft-colab，已启用 --core-only）
├── hooks.json                  # SessionStart 环境探测：控制 eval-platform 技能门控
├── rules/AGENTS.md             # 实验目录、增量配置、训后记录与云端缓存隔离规范
├── scripts/
│   └── check_runtime_env.py    # 探测 evalctl 与评测平台后端连通状态
└── skills/
    ├── sft-init/               # 仅建项期使用：参数收集、目录生成、回填插件 rules/AGENTS.md 槽位与环境验收
    ├── sft-pipeline/           # 日常实验闭环：数据准备、SFT 训练与归档复盘
    ├── sft-serving/            # 单卡/双卡 4-bit + LoRA OpenAI SSE 服务与隧道暴露
    ├── colab-ops/              # sft-colab MCP 调度与长训练监控 SOP
    └── eval-platform/          # 全功能 evalctl 评测、对弈与数据飞轮技能
```

## 前置依赖与选装说明

1. **基础环境**：Python >= 3.12；`sft-colab` MCP 通过 `uvx` 自动拉取 `better_colab_MCP`（已开启 `--core-only` 仅暴露 16 个核心工具），要求系统已安装 [`uv`](https://docs.astral.sh/uv/)。
2. **Colab 调度（可选）**：本机安装 Microsoft Edge，默认通过 `COLAB_MCP_EDGE_CDP_PORT=9333` 连接。
3. **评测平台 `eval-platform`（可选，由用户与 Agent 自行决定是否安装/复用）**：
   - 插件已内置完整的 `eval-platform` 技能文档，但是否安装 `evalctl` CLI 与启动评测后端（默认 `:9021`），由使用方及其 Agent 根据自身需求自行决定。
   - 会话启动时 `SessionStart` 钩子会自动探测环境：若 `evalctl` 与后端就绪则放行全功能闭环；若未安装或服务未启动（或在项目根目录放置了 `.eval_disabled`），插件会明确禁用该技能，且不会降级成临时本地评测脚本。
4. Google Drive（可选）：本项目使用googledrive同步本机与云端数据。方便本地agent直接编辑文件、同步文件。减少操作阻力。 同时单元格可直接运行脚本，节省agent操作读取单元格的token消耗。如果不使用drive同步，仍可使用git或huggingface同步。但体验后drive仍然是比较便捷配合colab调试的同步方案。

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

安装完毕后可以配合ai完成相关配置，即可进行你的微调了。
配合评测平台，可以实现模型评测，合成数据生成，质检对话修复等操作，实现飞轮闭环。
