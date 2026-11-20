# LLM SFT 工程规范（插件内置规则）

<!-- llm-sft-kit:project-config:start -->
## 当前项目环境与变量配置（`sft-init` 收集后由 Agent 直接更新此处）

- **基础配置（非涉密）**
  - `MODEL_TARGET`: ``
  - `ENABLE_THINKING`: `false`
  - `COMPUTE_TARGET`: ``  <!-- colab | kaggle | local -->
  - `COLAB_NOTEBOOK_URL`: ``
  - `COLAB_DRIVE_DIR`: ``  <!-- 例如 /content/drive/Othercomputers/<主机名>/<项目名> -->
  - `HF_REPO_ID`: ``
  - `EVAL_PLATFORM`: `pending`  <!-- enabled | disabled | pending -->

- **涉密凭据清单（仅记录键名，通过 Colab Secrets / 环境变量读取，严禁填入明文值）**
  - `HF_TOKEN`: 拉取受限基座或推送 LoRA 到 Hugging Face Hub
  - `WANDB_API_KEY` / `WAN_DB_ENTITY`: 可选，实验指标可视化
  - `NGROK_AUTHTOKEN`: 可选，推理隧道（未配则自动回退 cloudflared）
  - `DASHSCOPE_API_KEY`: 可选，外部评测/API 调用
<!-- llm-sft-kit:project-config:end -->

## 1. 标准项目结构与增量配置
- 配置分层：`configs/base.yaml` 存放公共超参；实验配置 `configs/exp_XXXX_<desc>.yaml` 仅覆盖与 `base.yaml` 不同的字段，带四位序号方便排序。
- 数据版本化：`data/raw/` 只读；`data/processed/vN/` 存放带 `manifest.json` 的清洗切分产物；`data/eval/vN/` 存放跨实验恒定评测集；大文件与 `runs/*/checkpoints/` 不入 Git。
- 每次训练输出到独立目录 `runs/YYYY-MM-DD_<exp_name>/`，自动落盘 `config.yaml` 快照、`meta.json`（git commit、数据 sha256、库版本、GPU、起止时间）、`logs/train.log`、`checkpoints/{best,last}`、`eval_results.json`、`evidence/` 与 `NOTES.md`。
- **禁止修改用户项目自带的根目录 `AGENTS.md`**：本插件所需的项目参数与运行记录约定统一维护在插件自身的 `rules/AGENTS.md`（以及项目 `configs/base.yaml`）中。

## 2. 训后记录规则
- 每次训练完成后，必须同步更新：
  1. 对应运行目录下的 `NOTES.md`（假设、结果、结论、问题/异常、下一步）；
  2. 项目根目录 `experiments.md`（一行一条实验记录）；
  3. 项目根目录 `lessons.md`（跨实验沉淀的规律或踩坑）。

## 3. 训练环境与文件同步（Colab / Google Drive / Kaggle）
- **云端缓存隔离**：严禁将 `HF_HOME`、`TORCH_HOME`、`UNSLOTH_COMPILE_LOCATION`、`TRITON_CACHE_DIR` 等缓存写入 Google Drive 同步目录或 Kaggle 工作配额盘；云端运行必须重定向至 `/tmp` 或 `/root/.cache`，高频过程 checkpoint 写入本地临时盘（如 `/content/temp_training`）。
- **同步方式**：通过 Google Drive 桌面端与 Colab `drive.mount` 实现双向同步；本地修改代码或数据后同步云端，避免在云端同步目录内高频写入临时小文件。
- **前台调度与跟踪**：Colab 长时间训练必须在 Notebook 单元格前台运行（避免纯后台运行在浏览器空闲时被断连），并提前做好 `DualLogger` 落盘；训练启动后预估总时长，按约 1/10 间隔设定时任务跟踪进度，全部落盘同步后再释放实例（`runtime.unassign()`）。

## 4. 授权与安全
- 涉密变量（如 `HF_TOKEN`、`WANDB_API_KEY`、`DASHSCOPE_API_KEY`、`NGROK_AUTHTOKEN`）必须通过 **Colab Secrets** 或本机环境变量读取，严禁明文写入代码、配置文件、Git 或在日志/对话中输出。
- 涉及远程 GPU 长耗时训练、切换运行时或挂载 Google Drive，须先经用户授权或由用户手动操作配合。

## 5. 评测平台边界
- 若会话初始化提示 `eval-platform` 未就绪，禁止调用 `eval-platform` 技能，也禁止自造简陋本地评测替代脚本(除非用户坚持，否则推荐评测后台)；需要评测时提示用户开启完整评测平台或先部署推理端点。
