"""Scaffold a reproducible LLM SFT project without touching any AGENTS.md files."""

import argparse
import shutil
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = SKILL_DIR.parent.parent


def _write_if_needed(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        print(f"[skip] {path} already exists")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[create] {path}")


def _copy_if_needed(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        print(f"[skip] {dst} already exists")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[copy] {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize a reproducible LLM SFT workspace.")
    parser.add_argument("--target-dir", type=str, default=".", help="Target project directory")
    parser.add_argument("--model", type=str, default="", help="Base model name or path")
    parser.add_argument("--enable-thinking", action="store_true", help="Default enable_thinking=true (default false)")
    parser.add_argument("--hf-repo", type=str, default="", help="Optional Hugging Face repo ID for LoRA push")
    parser.add_argument("--colab-drive-dir", type=str, default="", help="Optional Colab Drive mount path")
    parser.add_argument("--no-eval-platform", action="store_true", help="Create .eval_disabled marker")
    parser.add_argument("--force", action="store_true", help="Overwrite existing script/config templates")
    args = parser.parse_args()

    root = Path(args.target_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # 1. Data & runs directories
    for rel in ("data/raw/.gitkeep", "data/processed/.gitkeep", "data/eval/.gitkeep", "runs/.gitkeep"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("", encoding="utf-8")

    # 2. Copy standard scripts from plugin templates
    pipeline_scripts = PLUGIN_ROOT / "skills" / "sft-pipeline" / "scripts"
    serving_scripts = PLUGIN_ROOT / "skills" / "sft-serving" / "scripts"
    _copy_if_needed(pipeline_scripts / "prepare_data_tpl.py", root / "scripts" / "prepare_data.py", args.force)
    _copy_if_needed(pipeline_scripts / "train_sft_tpl.py", root / "scripts" / "train.py", args.force)
    _copy_if_needed(serving_scripts / "deploy_t4x2_tpl.py", root / "scripts" / "deploy_serving.py", args.force)

    # 3. Config templates
    base_yaml = (
        "project:\n"
        f'  colab_dir: "{args.colab_drive_dir}"\n'
        "\n"
        "model:\n"
        f'  name_or_path: "{args.model}"\n'
        f'  hf_repo_id: "{args.hf_repo}"\n'
        "  load_in_4bit: true\n"
        "  max_seq_length: 4096\n"
        f"  enable_thinking: {'true' if args.enable_thinking else 'false'}\n"
        '  reasoning_effort: "low"\n'
        "\n"
        "data:\n"
        '  version: "v1"\n'
        '  train_file: "data/processed/v1/train.jsonl"\n'
        '  val_file: "data/processed/v1/val.jsonl"\n'
        "  max_seq_length: 4096\n"
        "  default_weight: 1\n"
        "  source_weights: {}\n"
        "\n"
        "training:\n"
        "  learning_rate: 1.5e-4\n"
        "  num_train_epochs: 3\n"
        "  per_device_train_batch_size: 2\n"
        "  gradient_accumulation_steps: 4\n"
        "  warmup_ratio: 0.05\n"
        "  weight_decay: 0.01\n"
        "  logging_steps: 5\n"
        '  eval_strategy: "steps"\n'
        "  eval_steps: 100\n"
        '  save_strategy: "steps"\n'
        "  save_steps: 100\n"
        "  save_total_limit: 2\n"
        '  optim: "adamw_8bit"\n'
        '  lr_scheduler_type: "cosine"\n'
        "  seed: 3407\n"
        "  max_grad_norm: 0.3\n"
        "\n"
        "lora:\n"
        "  r: 32\n"
        "  lora_alpha: 64\n"
        "  lora_dropout: 0.0\n"
        '  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]\n'
    )
    _write_if_needed(root / "configs" / "base.yaml", base_yaml, args.force)

    smoke_yaml = (
        "# 2-step end-to-end smoke test overrides\n"
        "training:\n"
        "  max_steps: 2\n"
        "  logging_steps: 1\n"
        "  eval_steps: 1\n"
        "  save_steps: 1\n"
    )
    _write_if_needed(root / "configs" / "exp_0001_smoke_test.yaml", smoke_yaml, args.force)

    # 4. Archival & gitignore files
    gitignore = (
        "/data/\n"
        "!/data/**/.gitkeep\n"
        "runs/**/checkpoints/\n"
        "/runs/\n"
        "*.pt\n"
        "*.bin\n"
        "*.safetensors\n"
        "*.pth\n"
        "__pycache__/\n"
        "*.py[cod]\n"
        "unsloth_compiled_cache/\n"
        ".venv/\n"
        "venv/\n"
        ".DS_Store\n"
        "Thumbs.db\n"
        "desktop.ini\n"
    )
    _write_if_needed(root / ".gitignore", gitignore, args.force)

    experiments_md = (
        "| ID | 改动点 | 数据版本 | 训练/评测指标 | 推理性能 (TTFT / TPOT / TPS) | 结论 | 是否保留模型 |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    _write_if_needed(root / "experiments.md", experiments_md, args.force)
    _write_if_needed(root / "lessons.md", "# 跨实验规律与踩坑\n\n", args.force)

    if args.no_eval_platform:
        _write_if_needed(root / ".eval_disabled", "Disabled via init_project.py --no-eval-platform\n", args.force)

    print(f"SFT project scaffold ready at: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
