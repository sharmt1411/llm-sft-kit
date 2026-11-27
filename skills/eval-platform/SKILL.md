---
name: eval-platform
description: 通过统一命令行工具 evalctl（或 python -m cli / python backend/cli.py）操纵通用大模型测试管理与数据飞轮平台：模型注册连通、规则SOP与Prompt调试实测、数据集导入过滤导出、常规评测对比与分歧分析、双Agent模拟对弈（含SSE流式）以及数据飞轮质检/微创修复/切题导出全闭环。
---

# Eval Platform Agent CLI 技能指南 (`evalctl`)

本指南面向本机 Coding Agent，采用**渐进式披露**方式按需探索并调用平台全部 8 大能力模块（60 条命令）。

## 一、环境检测、下载启动与路径语义

### 1. 检测本地 CLI 与服务状态（先查后动）
- **检测 CLI 命令**：运行 `evalctl --help`（若未全局安装但在源码根目录，可用 `python -m cli --help` 或 `python backend/cli.py --help` 替代）。
- **检测本地 `:9021` 后端服务**：
  ```bash
  evalctl system health
  ```
  - 返回 `{"status": "healthy", ...}`（退出码 `0`）：本地服务已运行，直接使用默认 HTTP 模式即可（与 Web 工作台共享 SQLite 数据库与产物）。
  - 返回退出码 `2`（连接失败）：说明本地 `:9021` 服务未启动。**纯 CLI 调测可直接加全局参数 `--in-process`**（如 `evalctl --in-process system health`）通过 ASGI 进程内直调后端，无需强行起服务；**需要长驻服务或联用 Web 工作台时**，按第 3 步提示用户启动服务。

### 2. 下载与安装（当前环境尚未部署时）
- **安装源仓库**：`https://github.com/sharmt1411/eval-platform.git`（要求 Python `>= 3.10`；若启动前端 Web 工作台另需 Node.js `>= 18`）。
  ```bash
  git clone https://github.com/sharmt1411/eval-platform.git
  cd eval-platform
  pip install -r backend/requirements.txt
  pip install -e .
  ```
- 若仅在外部项目中安装 CLI 客户端连接远端 HTTP 服务（不本地起后端或走 `--in-process`），也可单独安装：`pip install "git+https://github.com/sharmt1411/eval-platform.git"`。
- 若使用 Google Vertex AI 模型，需在本机先完成 ADC 鉴权（`gcloud auth application-default login`），并在环境变量或模型配置中指定 Project ID 与 Location。

### 3. 启动本地服务（长期运行服务需由用户手动运行或获批后启动）
```bash
# 1. 启动后端 FastAPI 服务（默认监听 127.0.0.1:9021，API 文档 http://127.0.0.1:9021/docs）
uvicorn backend.app.main:app --host 127.0.0.1 --port 9021 --reload

# 2. 可选：启动前端 Web 工作台（默认监听 http://localhost:5173，反向代理 /api 至 9021）
cd frontend && npm install && npm run dev
```
- **传输模式切换**：默认连接 `http://127.0.0.1:9021`，可通过全局 `--base-url` 或环境变量 `EVALCTL_BASE_URL` 指向其他后端地址；`--in-process` 仅建议在未启动 `:9021` 服务时使用，已有服务运行时优先走 HTTP 模式以免并发写库冲突。

### 4. 全局参数语法硬约束（必看）
```text
evalctl [全局参数] <group> <command> [子命令参数]
```
- **所有全局参数必须置于子命令之前**，包括：`--out`、`--format`、`--timeout`、`--profile`、`--config`、`--base-url`、`--in-process`、`--yes`。放在子命令之后会导致 Click 解析报错。
- 正确示例：`evalctl --out data/runs/sim.json --timeout 0 simulation stream ...`

### 5. 跨项目调用的路径语义
在外部业务仓库（如训练/微调项目目录）中运行 `evalctl` 时，请严格区分两类路径：
- **客户端当前路径**（相对当前终端工作目录）：`dataset import --file`、`--payload`、`--out`、`--save-jsonl`、`--dialogue-file`、`--messages-file`、`--template-file`、`--assembled-prompt-file`。
- **平台服务端路径**（相对平台仓库根目录或服务端绝对路径）：`benchmark run --dataset-path`、`flywheel batch --input-file-path`、`flywheel inspect-dataset --dataset-path`、`dataset import --source-file-path`。
- **外部项目标准导入做法**：先用 `evalctl dataset import --dataset-id <id> --dataset-type benchmark --file <本地路径>` 导入，平台会自动落盘为 `data/datasets/<id>.jsonl` 并返回 `file_path`；随后再使用该平台路径运行 `benchmark run --dataset-path ...`。

---

## 二、渐进式发现与权限策略（不要死记硬背参数）

1. **先探明当前放行边界**：
   ```bash
   evalctl --format compact system capabilities
   ```
   若执行命令返回退出码 `3`（`POLICY_DENIED`），再查具体拦截原因：
   ```bash
   evalctl system policy --check <cmd_id>
   ```
2. **按需查看子命令参数**（已被策略禁用的命令不会出现在 `--help` 中）：
   ```bash
   evalctl benchmark --help
   evalctl benchmark side-by-side --help
   ```
3. **Profile 边界速览**：
   - 内置 5 套 Profile：`full`（默认全开，60 条）、`safe`（禁用删除与重置）、`readonly`（仅只读查询与组装预览）、`eval-loop`（评测调优闭环，仅放行 31 条命令）、`flywheel-loop`（对弈修复闭环）。
   - ⚠️ **Profile 常见踩坑**：`eval-loop` **不包含** `model.create` / `model.update`（及 `simulation` / `flywheel`）。若需要注册模型或修改远程推理地址，必须使用 `full` 或 `safe`，否则会报 `POLICY_DENIED`（退出码 `3`）。
   - 可通过全局 `--profile <name>` 切换；开启破坏性操作二次确认时补加全局 `--yes`。

---

## 三、高频端到端闭环工作流

### 闭环 0：动态接入远程推理地址（Colab / Kaggle / 隧道评测必跑）
```bash
# 1. 更新已有模型或注册新模型
#    注意：openai / harmony 协议的 --api-url 必须填完整端点（含 /v1/chat/completions）
evalctl model update <model_id> --api-url https://<tunnel-host>/v1/chat/completions
# 或注册新模型：
evalctl model create --id qwen3-sft --name "Qwen3-SFT" --provider-type openai \
  --api-url https://<tunnel-host>/v1/chat/completions --model-name qwen3-8b

# 2. 连通性探测 + 单轮对话验真
evalctl model test <model_id>
evalctl model chat <model_id> --message "你好，请简要自我介绍"
```

### 闭环 1：外部导入 → 常规评测对比 → 分歧诊断
```bash
# 1. 紧凑查看模型，导入外部评测集并核对资产列表
evalctl --format table model list --brief
evalctl dataset import --dataset-id my_eval --dataset-type benchmark --file ./eval.jsonl
evalctl --format table dataset list

# 2. 发起多模型横向对比评测并等待完成
#    注：常规评测仅针对具备 reference/rubric/regex_constraints 的考题集；纯对话集会被默认拦截防假阳性 100% PASS
#    注：单卡/内网穿透串行服务（如 GPU_THREAD_LOCK）务必设 --workers 1 并调大 --wait-timeout；全局 --timeout 0 防读取超时
evalctl --timeout 0 benchmark run \
  --dataset-path data/datasets/my_eval.jsonl \
  --model-id qwen3-sft --model-id mock-a \
  --judge-model-id vertex-gemini-flash-judge \
  --workers 1 --wait --wait-timeout 3600

# 3. 先看分歧摘要，再把明细落盘做深度分析
evalctl benchmark side-by-side <task_id> --filter divergence --summary-only
evalctl --out data/runs/debug_sbs.json benchmark side-by-side <task_id> --filter fail
```

### 闭环 2：规则 / 提示词模板调优与在线实测
```bash
# 1. 紧凑列出提示词摘要或规则列表
evalctl --format table prompt list --summary
evalctl --format table rule list

# 2. 基于真实规则动态拼装调试 Prompt 并落盘，再直接用该报文实测
evalctl --out data/runs/debug_req.json rule debug-prompt --rule-id demo_after_sales --prompt-type judge
evalctl rule test-prompt --model-id vertex-gemini-flash-judge --payload data/runs/debug_req.json

# 3. 常规评测裁判 Prompt 调试同理
evalctl --out data/runs/bench_judge.json benchmark debug-judge --dataset-path data/datasets/demo_benchmark.jsonl
evalctl benchmark test-judge --model-id vertex-gemini-flash-judge --payload data/runs/bench_judge.json
```

### 闭环 3：双 Agent 模拟对弈（含大模型/思考模式采样造数）→ 飞轮质检/修复 → 切题/导出 SFT 与评测集

> **双用途提示**：模拟对弈既可用于**压测微调模型实战能力**，也常用于**冷启动/长尾造数**——把高性能老师模型或开启思考参数（`thinking_enabled: true`）的模型挂为 `--target-model-id` 直接对弈采样，后台会自动将每轮 `assistant` 的思考链 `thinking` 与正文剥离并一并入库，再经白盒质检与清洗导出高质量 SFT / CoT 蒸馏数据集。

```bash
# 0.（可选：老师模型/思考模式采样造数时）开启目标模型思考参数，模拟客户侧保持 NONE/LOW 以兼顾速度与口语化
evalctl model update <teacher_model_id> \
  --config-json '{"thinking_enabled": true, "thinking_level": "HIGH", "max_tokens": 4096}'

# 1. 流式运行对弈（进度打到 stderr；汇总 JSON 约 200KB+，务必加全局 --out 落盘）
#    测模型或老师模型采样均通过 --target-model-id 指定；全剧本批量采样传 --scenario-id all
evalctl --timeout 0 --out data/runs/sim_result.json simulation stream \
  --rule-id demo_after_sales \
  --target-model-id harmony-demo-target-model \
  --client-model-id vertex-gemini-flash-client \
  --judge-model-id vertex-gemini-flash-judge \
  --remediate-model-id vertex-gemini-flash-judge \
  --scenario-id case_01_standard_after_sales

# 2. 查看飞轮样本摘要（--brief + --format table 防爆上下文）
evalctl --format table flywheel samples --rule-id demo_after_sales --brief --limit 20

# 3. 按新规则重检（只检不修）或微创修复，再检查切题切片
evalctl --timeout 0 flywheel judge <sample_id> --rule-id demo_after_sales --judge-model-id vertex-gemini-flash-judge
evalctl --timeout 0 flywheel remediate <sample_id> --remediate-model-id vertex-gemini-flash-judge
evalctl flywheel slices-get <sample_id>

# 4. 导出黄金 SFT 训练集（仅允许达标/人工放行样本，按 active_version 导出，默认排除已导出评测集的留出同源对话并启用去重防稀释）或导出为 Benchmark 评测集
#    注：如需导出采集到的 assistant 思考链用于 CoT 蒸馏，显式加 --inject-thinking（默认不导出）
#    注：replace 修复仅改写 content 不会重生成 thinking，强 CoT 蒸馏建议优先筛选 GOLD_PASS 样本导出：
#        evalctl flywheel export-selected --sample-id <id1> --sample-id <id2> --inject-thinking
#    去重防稀释参数：--enable-dedup/--no-dedup、--cluster-quota <K>（训练默认20/评测默认5）、--sim-threshold <θ>（默认0.82）、--target-count <N>
#    SFT 导出还支持：--mode create|append、--target-dataset-id <id>、--no-exclude-benchmark-exported（显式放行时将把其既有评测题改为普通题）
#    评测切题导出支持：--reserve-for-training（预留后续进训练集，主动标为普通题）；已进过训练集的样本切题时将自动标为普通题
evalctl flywheel export-gold --rule-id demo_after_sales --version-tag v1.1 --cluster-quota 20 --sim-threshold 0.82
evalctl flywheel export-gold --rule-id demo_after_sales --version-tag v1.1_cot --inject-thinking
evalctl flywheel export-benchmark <sample_id> --dataset-id fw_eval_v1 --name "飞轮回归评测集" --cluster-quota 5

# 5. 批量飞轮治理（初检->AI修复->二检->gold/discarded；--input-file-path 为服务端路径；默认入池 Upsert 并对黄金集去重）
evalctl --timeout 0 flywheel batch --rule-id demo_after_sales --input-file-path data/runs/raw.jsonl \
  --judge-model-id vertex-gemini-flash-judge --workers 1 --wait --wait-timeout 3600

# 6. 纯对话数据集批量规则质检打分（只检不修，不触发 AI 修复，产出均分/通过率/高频扣分项看板）
#    适用场景：flywheel_gold / SFT 多轮对话语料按 SOP + 红线 + FAQ 规则批量打分验真
evalctl --timeout 0 flywheel inspect-dataset \
  --dataset-id <dataset_id> \
  --rule-id <rule_id> \
  --judge-model-id <judge_model_id> \
  --workers 4 --wait --brief
#    按任务 ID 查看未通过明细或落盘完整报告：
evalctl flywheel inspect-report <task_id> --only-fail --brief
evalctl --out data/runs/inspect_report.json flywheel inspect-report <task_id>
```

### 闭环 4：数据集智能抽样试算与条件过滤导出
```bash
# 1. 查看历史基准模型线与分类分布
evalctl dataset filter-options demo_benchmark

# 2. 试算并另存为聚焦薄弱错题的新评测集（append 模式自动开启四步去重，可选 --enable-dedup/--cluster-quota/--sim-threshold/--target-count）
evalctl dataset export-preview demo_benchmark --export-scope failed_only --model-line gpt-oss-20b
evalctl dataset export demo_benchmark \
  --export-scope failed_only --model-line gpt-oss-20b \
  --export-target save_dataset --target-dataset-id demo_failed_focus

# 3. 外部数据集追加导入（append 模式默认去重，同步维护 .dedup_cache.json；--no-dedup 可显式关闭）
evalctl dataset import --dataset-id ext_eval --dataset-type benchmark --mode append --file raw_eval.jsonl --cluster-quota 5
```

---

## 四、防爆上下文、超时/并发参数与退出码

### 1. 大载荷保护与紧凑格式（防爆上下文）
- **必用 `--brief` 或全局 `--out <path>` 的超大载荷命令**：
  - `simulation recent --brief`（不加 `--brief` 时 `--limit 3` 可达 ~292 KB；列表检索务必加 `--brief`，需看完整对话时配全局 `--out` 并收紧 `--limit`）
  - `simulation stream`（汇总输出约 ~219 KB，务必配全局 `--out`）
  - `benchmark tasks --brief`（不加 `--brief` 时 `--limit 10` 约 ~34 KB；列表检索务必加 `--brief`，需看完整配置时配 `--limit` 与全局 `--out`）
  - `benchmark side-by-side`、`flywheel samples`、`flywheel inspect-report`、`flywheel inspect-dataset --wait`、`rule get`（优先搭配 `--summary-only` / `--brief` 或全局 `--out`）
- **紧凑视图推荐**：在只读列表命令（如 `model list --brief`、`benchmark tasks --brief`、`simulation recent --brief`、`flywheel samples --brief`、`dataset list`、`system capabilities`、`prompt list --summary`）上使用全局 `--format table`（或 `--format compact`），大幅节省上下文 Token。

### 2. 慢推理 / 隧道防超时与裁判边界
- **HTTP 读取超时 `--timeout`**：CLI 默认 `600` 秒；对 `simulation stream`、`flywheel judge/remediate` 或慢推理隧道请求，传入全局 `--timeout 0`（不限）或 `--timeout 300`/`1200`（注：后端 `OpenAIAdapter` / `HarmonyAdapter` 已默认开启 `stream=True` SSE 流式保活与 `300s` 分片读超时，支持长生成穿越 Cloudflare 等隧道网关）。
- **异步等待 `--wait-timeout` 与串行并发 `--workers 1`**：`benchmark run --wait` 和 `flywheel batch --wait` 默认 `--wait-timeout 600` 秒，超过 30 题容易超时，应按需调大（如 `--wait-timeout 3600`）；面对单卡或带 `GPU_THREAD_LOCK` 的内网穿透串行服务，必须配合 `--workers 1`。
- **数据集类型与评测/质检分工边界（必看）**：
  - **常规对比评测 (`benchmark run`)**：仅适用于**具备 `reference`、`rubric` 或 `regex_constraints` 的考题集**；对无答案标准的纯对话集（如 `flywheel_gold` 多轮对话语料）默认会拦截返回 HTTP 400（防跳过校验制造 100% 假阳性通过）。仅在做纯连通性/吞吐测速时才显式传 `--allow-unlabeled`。
  - **规则质检打分 (`flywheel inspect-dataset`)**：专用于**纯对话数据集（只检不修）**，按指定 `--rule-id` 复用 SOP 状态机 + 红线 ∥ FAQ 双轨逐条打分，返回均分、通过率、拓扑连通率、高频扣分项与逐样本诊断；如需同步写入飞轮样本池复核，追加 `--save-to-flywheel`。

### 3. 协议约定与退出码
- **输出流分离**：成功时 `stdout` 仅输出合法 JSON（或 `--format table`）；轮询进度、SSE 逐轮日志与错误 JSON 均输出到 `stderr`。
- **复杂载荷传递**：写/运行类命令支持 `--payload <file.json|->`（`-` 从 stdin 读取）与 `--payload-json '<json>'`，再叠加具体 CLI 选项覆盖。

| 退出码 | 含义 | 常见处理建议 |
| :---: | :--- | :--- |
| `0` | 成功 | 解析 stdout JSON 结果 |
| `1` | 参数错误或后端 HTTP 4xx/5xx 业务异常 | 查看 stderr JSON `error.message`（如全局参数放错位置、缺必填项） |
| `2` | 后端连接失败或请求超时 | 检查 `:9021` 服务是否启动；长耗时请求加全局 `--timeout 0` 或调大 `--wait-timeout` |
| `3` | 命令被当前权限策略拦截 (`POLICY_DENIED`) | 运行 `evalctl system policy --check <cmd_id>`；如修改模型配置需切回 `full` 或 `safe` |
| `4` | 破坏性命令缺少确认 (`CONFIRMATION_REQUIRED`) | 确认无误后在全局参数补加 `--yes` |
