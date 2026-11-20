# LLM SFT Kit

一套面向大模型 LoRA/SFT 微调项目的可复用工程插件包，整合了标准实验脚手架、数据处理与纯回复 Loss 训练模板、Colab MCP 算力调度 SOP、OpenAI SSE 推理服务部署，以及可选的全功能 `eval-platform` 闭环对接。

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

## 安装与挂载方式

- **方式一（推荐分发）：放入全局插件目录**
  ```bash
  git clone https://github.com/sharmt1411/llm-sft-kit.git ~/.gemini/config/plugins/llm-sft-kit
  ```
- **方式二：项目级挂载（适合单项目使用或团队共享）**
  将插件目录直接克隆或复制到目标项目的 `.agents/plugins/llm-sft-kit/` 下即可自动发现加载（Windows 下请使用实目录，勿使用目录联接）。
