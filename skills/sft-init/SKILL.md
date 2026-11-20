---
name: sft-init
description: >-
  Use this skill ONLY when initializing a new LLM SFT project or setting up the
  workspace environment for the first time: probes local tools, collects user
  parameters (model, Colab URL, Google Drive sync path, HF repo, eval-platform
  choice), scaffolds the directory tree, populates blank config slots in the
  plugin's rules/AGENTS.md (never touching the user project's own AGENTS.md),
  guides manual Secrets/Drive setup, and verifies connectivity.
---

# LLM SFT Project & Environment Initialization

Use this skill **only during initial project/environment setup**. Once initialized, use `sft-pipeline` for daily data prep, training, and archival.

> [!IMPORTANT]
> Never create, overwrite, or modify the user project's own root `AGENTS.md`. All SFT environment variables and configuration slots collected during initialization must be recorded in the **plugin's own `rules/AGENTS.md`** (plus `configs/base.yaml`).

## Step 1: Preflight Check (Automatic)

Before asking the user anything, inspect the target workspace and local environment:
1. Check whether `configs/`, `data/`, or `scripts/` already exist in the target directory (never overwrite existing project data or configs without `--force`).
2. Check local binaries: `python`, `uv` (required for `sft-colab` MCP), `git`, and whether `evalctl` + `:9021` backend are available (align with `SessionStart` hook status).

## Step 2: Collect Required User Parameters

Inspect the blank slots in the plugin's `rules/AGENTS.md` (`<!-- llm-sft-kit:project-config:start -->`) and ask the user for any missing parameters:

| Parameter | Required? | Where Saved | Notes |
| :--- | :--- | :--- | :--- |
| Base model (`MODEL_TARGET`) | Yes | Plugin `rules/AGENTS.md`, `configs/base.yaml` | e.g. `Qwen/Qwen2.5-7B-Instruct` or pre-quantized 4-bit repo |
| Thinking mode (`ENABLE_THINKING`) | Default `false` | Plugin `rules/AGENTS.md`, `configs/base.yaml` | `false` strips `<think>...</think>` for pure reply loss |
| Compute target (`COMPUTE_TARGET`) | Yes | Plugin `rules/AGENTS.md` | `colab` / `kaggle` / `local` |
| Colab Notebook URL (`COLAB_NOTEBOOK_URL`) | If `colab` | Plugin `rules/AGENTS.md` | Full URL of the user's Launchpad notebook (can be created in Step 5 if none yet) |
| Google Drive sync / path (`COLAB_DRIVE_DIR`) | Optional (auto-detected) | Plugin `rules/AGENTS.md`, `configs/base.yaml` | Confirm whether Google Drive Desktop sync is enabled for this folder; leave blank to auto-detect after `connect_runtime` |
| Hugging Face repo (`HF_REPO_ID`) | Optional | Plugin `rules/AGENTS.md`, `configs/base.yaml` | Public adapter target (e.g. `org/model-lora`) |
| `eval-platform` usage | Optional | Plugin `rules/AGENTS.md` (or `.eval_disabled`) | If `evalctl`/backend missing, confirm whether user will start it or disable via `.eval_disabled` |
| Initial `data.source_weights` | Optional | `configs/base.yaml` | Filename substring -> upsampling multiplier (default `1`) |

## Step 3: Scaffold Project & Update Plugin `rules/AGENTS.md` Directly

1. Run [init_project.py](./scripts/init_project.py) to scaffold the workspace (`configs/`, `data/`, `scripts/`, `runs/`, `experiments.md`, `lessons.md`, `.gitignore`):

```bash
python "<PLUGIN_ROOT>/skills/sft-init/scripts/init_project.py" \
  --target-dir . \
  --model "<MODEL_TARGET>" \
  --hf-repo "<HF_REPO_ID>"
```

2. Directly edit the plugin's own `<PLUGIN_ROOT>/rules/AGENTS.md` (`<!-- llm-sft-kit:project-config:start -->` block) to fill in the collected non-secret fields (`MODEL_TARGET`, `ENABLE_THINKING`, `COMPUTE_TARGET`, `COLAB_NOTEBOOK_URL`, `COLAB_DRIVE_DIR`, `HF_REPO_ID`, `EVAL_PLATFORM`). If parameters arrive across multiple turns, edit only the newly provided lines—never wipe previously filled slots or write secret token values.

3. When training on Colab: Run the `colab-ops` browser/Edge CDP conflict check (check whether Edge CDP `http://127.0.0.1:9333/json/list` has an active conflicting Colab session), then connect via `sft-colab` (`open_colab_browser_connection`) to open a new browser page. If the user has not yet created a project notebook, call open_colab_browser_connection to open the default URL, then ask the user to sign in to their account in that browser instance and create a notebook there.
(This MCP launches the browser with its own custom profile. If the user creates the notebook in their default browser, the browser opened at connection time would still require signing in again. So open the default page through the connection and wait for the user to complete these steps.)

## Step 4: Guide Manual User Setup (Security & Sync Checklist)

Ask the user to complete the following manual steps and confirm once ready:

1. **Secrets Configuration (Never paste secret values into chat or code)**:
   - On **Colab**: open the left sidebar **Secrets (key icon)**, add required keys and toggle Notebook access ON:
     - `HF_TOKEN` (required for gated base models or pushing LoRA to Hugging Face)
     - `WANDB_API_KEY` / `WAN_DB_ENTITY` (optional, for W&B tracking)
     - `NGROK_AUTHTOKEN` (optional, for Ngrok serving tunnel; falls back to `cloudflared`)
     - `DASHSCOPE_API_KEY` (optional, if using external Qwen/DashScope APIs)
     - Each new notebook needs to be granted access to the secret once; after that, it’s not required again. 
   - On **Local / Kaggle**: set via OS environment variables or Kaggle Add-ons -> Secrets.
   - Note that only on Colab can you use the MCP tool to operate it; Kaggle is not supported by MCP.
2. **Google Drive Sync Setup (if using Colab)**:
   - Add the local project folder to **Google Drive for Desktop** and wait for initial sync.
   - Open the Colab Notebook in Edge (logged into Google), select GPU runtime (e.g., A100 for training, L4/T4 for inference/smoke), and run `from google.colab import drive; drive.mount('/content/drive')`.
3. **Optional `eval-platform` Backend**:
   - If using `eval-platform`, ensure `evalctl` is installed and the `:9021` backend is running (or `--in-process` environment is configured).

## Step 5: Connectivity, Dependencies & Readiness Verification

Once the user confirms Step 4 is complete:
- If `colab`:
  1. **Auto-detect / verify `COLAB_DRIVE_DIR`**: if `COLAB_DRIVE_DIR` is empty, search `/content/drive` for the synced project folder (e.g. `find /content/drive -maxdepth 4 -type d -name "<project_name>" ! -path "*/.Trash*"`) and confirm `configs/base.yaml` exists inside it, then backfill `COLAB_DRIVE_DIR` in plugin `rules/AGENTS.md` and `project.colab_dir` in `configs/base.yaml`.
  2. **Verify Colab runtime dependencies**: probe `unsloth`, `trl`, `peft`, `bitsandbytes`, `datasets`, `accelerate`, `pyyaml`, `huggingface_hub` inside the runtime and install any missing packages before running scripts(through cell code).
- If `eval-platform` enabled: verify `evalctl system health` (or `evalctl --format compact system capabilities`).
- Hand off to `sft-pipeline` to place raw `.jsonl` files into `data/raw/`, run `prepare_data.py --version v1`, and run a 2-step smoke test (`exp_0001_smoke_test.yaml`).
