"""Generic LoRA SFT training script with config snapshots, pure-reply loss masking, and run archival.

Features:
1. Redirects Colab/cloud caches (/content) away from Drive sync directories.
2. Merges configs/base.yaml + configs/exp_*.yaml and snapshots config.yaml + meta.json.
3. DualLogger tees stdout/stderr to runs/<date>_<exp>/logs/train.log.
4. Supports Unsloth FastLanguageModel with graceful fallback to Transformers + BitsAndBytes.
5. Compiles pure-assistant-reply loss masks (non-assistant tokens set to -100).
6. Saves checkpoints/best and checkpoints/last, updates NOTES.md, optionally pushes to HF Hub,
   and releases Colab runtime if requested.
"""

import argparse
import hashlib
import inspect
import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if os.path.exists("/content"):
    os.environ["UNSLOTH_COMPILE_LOCATION"] = "/tmp/unsloth_compiled_cache"
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    os.environ["TORCH_HOME"] = "/root/.cache/torch"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
os.environ["TQDM_DISABLE_WIDGETS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import platform
import re
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
import yaml

try:
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        logging as hf_logging,
    )
except ImportError:
    transformers = None

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

try:
    import peft
    from peft import LoraConfig, get_peft_model
except ImportError:
    peft = None

try:
    import trl
    from trl import SFTConfig, SFTTrainer
except ImportError:
    trl = None
    SFTConfig, SFTTrainer = None, None

warnings.filterwarnings("ignore")
if transformers:
    hf_logging.set_verbosity_error()


def ensure_colab_working_dir(config: dict):
    """Align cwd on Colab to `project.colab_dir` or `COLAB_PROJECT_DIR` when running under /content."""
    colab_dir = os.environ.get("COLAB_PROJECT_DIR") or config.get("project", {}).get("colab_dir")
    if colab_dir and Path("/content").exists():
        target = Path(colab_dir)
        if target.exists() and Path.cwd() != target:
            os.chdir(target)
            print(f"[Colab] Switched working directory to: {target}")


def init_credentials(config: dict):
    """Load HF_TOKEN / WANDB_API_KEY from Colab Secrets (if available) or environment variables."""
    try:
        from google.colab import userdata

        for key in ("HF_TOKEN", "WANDB_API_KEY", "WAN_DB_ENTITY", "HF_REPO_ID"):
            if key not in os.environ:
                try:
                    val = userdata.get(key)
                    if val:
                        os.environ[key] = str(val).strip()
                except Exception:
                    pass
    except Exception:
        pass

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=True)
            print("Hugging Face authentication ready.")
        except Exception as exc:
            print(f"Hugging Face login warning: {exc}")

    cfg_repo = config.get("model", {}).get("hf_repo_id")
    if cfg_repo and not os.environ.get("HF_REPO_ID"):
        os.environ["HF_REPO_ID"] = str(cfg_repo).strip()


class DualLogger:
    """Tee stdout/stderr into runs/<date>_<exp>/logs/train.log."""

    def __init__(self, log_filepath: Path, stream):
        if getattr(stream, "_is_dual_logger", False):
            try:
                stream.close()
            except Exception:
                pass
            stream = stream.terminal
        self.terminal = stream
        self._is_dual_logger = True
        self.log_filepath = log_filepath
        log_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        if not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if not self.log_file.closed:
            self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.flush()
            self.log_file.close()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)


def flush_dual_loggers():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown_or_not_git"


def get_file_sha256(filepath: Path) -> str:
    if not filepath.exists() or not filepath.is_file():
        return "not_found"
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def collect_meta_info(config: dict, start_time: str) -> dict:
    """Collect hardware, library versions, git commit, and dataset SHA256 hashes."""
    gpu_info: str | list[str] = "cpu"
    if torch.cuda.is_available():
        gpu_info = [
            f"{torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f}GB)"
            for i in range(torch.cuda.device_count())
        ]

    data_cfg = config.get("data", {})
    data_ver = data_cfg.get("version", "v1")
    train_file = Path(data_cfg.get("train_file", f"data/processed/{data_ver}/train.jsonl"))
    val_file = Path(data_cfg.get("val_file", f"data/processed/{data_ver}/val.jsonl"))

    libs = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": getattr(transformers, "__version__", "not_installed") if transformers else "not_installed",
        "peft": getattr(peft, "__version__", "not_installed") if peft else "not_installed",
        "trl": getattr(trl, "__version__", "not_installed") if trl else "not_installed",
    }
    try:
        import unsloth

        libs["unsloth"] = getattr(unsloth, "__version__", "installed")
    except Exception:
        libs["unsloth"] = "not_installed"

    return {
        "git_commit": get_git_commit(),
        "model_target": config.get("model", {}).get("name_or_path", "unknown"),
        "data_version": data_ver,
        "data_hashes": {
            "train_file": train_file.as_posix(),
            "train_sha256": get_file_sha256(train_file),
            "val_file": val_file.as_posix(),
            "val_sha256": get_file_sha256(val_file),
        },
        "platform": platform.platform(),
        "libraries": libs,
        "gpu": gpu_info,
        "start_time": start_time,
        "end_time": None,
    }


def load_merged_config(base_config_path: str, exp_config_path: str | None, enable_thinking_override: bool | None) -> dict:
    config: dict = {}
    if Path(base_config_path).exists():
        with open(base_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    if exp_config_path and Path(exp_config_path).exists():
        with open(exp_config_path, "r", encoding="utf-8") as f:
            exp_cfg = yaml.safe_load(f) or {}
        config = deep_merge_dicts(config, exp_cfg)
    if enable_thinking_override is not None:
        config.setdefault("model", {})["enable_thinking"] = enable_thinking_override
    else:
        config.setdefault("model", {}).setdefault("enable_thinking", False)
    return config


def setup_experiment_run(config: dict, exp_name: str) -> Path:
    """Create runs/YYYY-MM-DD_<exp_name>/ tree, snapshots, and attach DualLogger."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path("runs") / f"{date_str}_{exp_name}"
    (run_dir / "logs" / "tensorboard").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "best").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "last").mkdir(parents=True, exist_ok=True)
    (run_dir / "evidence").mkdir(parents=True, exist_ok=True)

    train_log_file = run_dir / "logs" / "train.log"
    sys.stdout = DualLogger(train_log_file, sys.stdout)
    sys.stderr = DualLogger(train_log_file, sys.stderr)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    meta_info = collect_meta_info(config, datetime.now().isoformat())
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    notes_path = run_dir / "NOTES.md"
    if not notes_path.exists():
        model_name = config.get("model", {}).get("name_or_path", "unknown")
        data_ver = config.get("data", {}).get("version", "v1")
        notes_path.write_text(
            f"# {exp_name}\n\n"
            f"## 假设\n使用 `{model_name}` 在数据集 `{data_ver}` 上执行 LoRA SFT 微调。\n\n"
            f"## 结果\n\n\n"
            f"## 结论\n\n\n"
            f"## 问题 / 异常\n\n\n"
            f"## 下一步\n\n",
            encoding="utf-8",
        )

    print(f"Initialized run directory: {run_dir}")
    return run_dir


def load_model_and_tokenizer(config: dict):
    """Load base model & tokenizer via Unsloth or fallback to Transformers + BitsAndBytes."""
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name_or_path")
    if not model_name:
        raise ValueError("`model.name_or_path` must be set in configs/base.yaml or exp config.")

    max_seq_length = int(model_cfg.get("max_seq_length", 4096))
    load_in_4bit = bool(model_cfg.get("load_in_4bit", True))
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    print(f"Loading model: {model_name} (max_seq_length={max_seq_length}, 4bit={load_in_4bit})")
    try:
        from unsloth import FastLanguageModel

        offload_kwargs = {}
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0
        offload_layers = model_cfg.get("offload_layers")
        if offload_layers is not None:
            offload_kwargs["offload_layers"] = int(offload_layers)
        elif any(tag in str(model_name).lower() for tag in ("27b", "32b", "35b")) and total_vram_gb < 20.0:
            offload_kwargs["offload_layers"] = 30

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            full_finetuning=False,
            token=hf_token,
            **offload_kwargs,
        )
        model._is_unsloth = True
    except (ImportError, Exception) as err:
        print(f"Unsloth unavailable ({err}), falling back to Transformers + BitsAndBytes...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            if load_in_4bit and torch.cuda.is_available()
            else None
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            token=hf_token,
        )
        model._is_unsloth = False

    if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
        tokenizer = tokenizer.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|im_end|>"
    tokenizer.padding_side = "right"
    return model, tokenizer


def setup_lora(model, config: dict):
    lora_cfg = config.get("lora", {})
    lora_r = int(lora_cfg.get("r", 32))
    lora_alpha = int(lora_cfg.get("lora_alpha", 64))
    lora_dropout = float(lora_cfg.get("lora_dropout", 0.0))
    target_modules = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    if getattr(model, "_is_unsloth", False):
        from unsloth import FastLanguageModel

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=int(config.get("training", {}).get("seed", 3407)),
        )
    else:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)

    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False
    if hasattr(model.config, "router_aux_loss_coef"):
        model.config.router_aux_loss_coef = 0.0

    model.print_trainable_parameters()
    return model


def load_jsonl_messages(file_path: Path, enable_thinking: bool = False) -> list[dict]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {file_path}. Run `python scripts/prepare_data.py --version {file_path.parent.name}` first."
        )
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            msgs = item.get("messages", [])
            if not msgs or msgs[-1].get("role") != "assistant":
                continue
            if not any(m.get("role") == "user" for m in msgs[:-1]):
                continue
            if not enable_thinking:
                for m in msgs:
                    if m.get("role") == "assistant":
                        m["content"] = re.sub(r"<think>[\s\S]*?</think>", "", str(m.get("content", ""))).strip()
            records.append({"messages": msgs})
    return records


def load_and_prepare_datasets(config: dict) -> tuple[Dataset, Dataset]:
    data_cfg = config.get("data", {})
    enable_thinking = bool(config.get("model", {}).get("enable_thinking", False))
    data_ver = data_cfg.get("version", "v1")
    train_path = Path(data_cfg.get("train_file") or f"data/processed/{data_ver}/train.jsonl")
    val_path = Path(data_cfg.get("val_file") or f"data/processed/{data_ver}/val.jsonl")

    train_data = load_jsonl_messages(train_path, enable_thinking=enable_thinking)
    val_data = load_jsonl_messages(val_path, enable_thinking=enable_thinking)
    print(f"Loaded datasets: train={len(train_data)}, val={len(val_data)}")
    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def _unwrap_token_ids(rendered) -> list[int]:
    if hasattr(rendered, "get"):
        rendered = rendered.get("input_ids", rendered)
    elif hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    if rendered and isinstance(rendered, (list, tuple)) and isinstance(rendered[0], (list, tuple)):
        rendered = rendered[0]
    return list(rendered)


def formatting_and_masking_func(
    examples,
    tokenizer,
    max_seq_length: int = 4096,
    enable_thinking: bool = False,
    reasoning_effort: str = "low",
):
    """Mask prompt tokens with -100 so loss is computed exclusively on the final assistant reply."""
    batch_input_ids, batch_attention_mask, batch_labels = [], [], []
    chat_kwargs = {"enable_thinking": enable_thinking}
    if enable_thinking and reasoning_effort:
        chat_kwargs["reasoning_effort"] = reasoning_effort

    for convo in examples["messages"]:
        if not convo or convo[-1].get("role") != "assistant" or not any(m.get("role") == "user" for m in convo):
            continue

        prompt_convo = convo[:-1]
        try:
            try:
                p_raw = tokenizer.apply_chat_template(prompt_convo, tokenize=True, add_generation_prompt=True, **chat_kwargs)
                f_raw = tokenizer.apply_chat_template(convo, tokenize=True, add_generation_prompt=False, **chat_kwargs)
            except TypeError:
                try:
                    p_raw = tokenizer.apply_chat_template(prompt_convo, tokenize=True, add_generation_prompt=True, chat_template_kwargs=chat_kwargs)
                    f_raw = tokenizer.apply_chat_template(convo, tokenize=True, add_generation_prompt=False, chat_template_kwargs=chat_kwargs)
                except TypeError:
                    p_raw = tokenizer.apply_chat_template(prompt_convo, tokenize=True, add_generation_prompt=True)
                    f_raw = tokenizer.apply_chat_template(convo, tokenize=True, add_generation_prompt=False)
        except Exception:
            continue

        prompt_ids = _unwrap_token_ids(p_raw)
        full_ids = _unwrap_token_ids(f_raw)
        split_idx = len(prompt_ids)
        labels = [-100] * len(full_ids) if len(full_ids) < split_idx else ([-100] * split_idx + full_ids[split_idx:])
        attention_mask = [1] * len(full_ids)

        if len(full_ids) > max_seq_length:
            full_ids = full_ids[:max_seq_length]
            labels = labels[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]

        batch_input_ids.append(full_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask, "labels": batch_labels}


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, config: dict, run_dir: Path, temp_ckpt_dir: Path):
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    max_seq_length = int(data_cfg.get("max_seq_length", model_cfg.get("max_seq_length", 4096)))
    enable_thinking = bool(model_cfg.get("enable_thinking", False))
    reasoning_effort = str(model_cfg.get("reasoning_effort", "low"))

    tokenized_train = train_dataset.map(
        lambda x: formatting_and_masking_func(x, tokenizer, max_seq_length, enable_thinking, reasoning_effort),
        batched=True,
        remove_columns=["messages"],
    )
    tokenized_eval = eval_dataset.map(
        lambda x: formatting_and_masking_func(x, tokenizer, max_seq_length, enable_thinking, reasoning_effort),
        batched=True,
        remove_columns=["messages"],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    temp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_strategy = train_cfg.get("eval_strategy", "steps")
    save_strategy = train_cfg.get("save_strategy", "steps")

    has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = bool(train_cfg.get("bf16", has_bf16)) and has_bf16
    use_fp16 = bool(train_cfg.get("fp16", not use_bf16)) if not use_bf16 else False

    sft_kwargs = {
        "output_dir": str(temp_ckpt_dir),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 2)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 4)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 3)),
        "max_steps": int(train_cfg.get("max_steps", -1)),
        "learning_rate": float(train_cfg.get("learning_rate", 1.5e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "logging_steps": int(train_cfg.get("logging_steps", 5)),
        "optim": train_cfg.get("optim", "adamw_8bit" if torch.cuda.is_available() else "adamw_torch"),
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "seed": int(train_cfg.get("seed", 3407)),
        "bf16": use_bf16,
        "fp16": use_fp16,
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 0.3)),
        "save_strategy": save_strategy,
        "save_steps": int(train_cfg.get("save_steps", 100)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "eval_strategy": eval_strategy,
        "eval_steps": int(train_cfg.get("eval_steps", 100)),
        "load_best_model_at_end": eval_strategy == save_strategy,
        "metric_for_best_model": "eval_loss" if eval_strategy == save_strategy else None,
        "greater_is_better": False,
        "report_to": "tensorboard",
    }

    sft_sig = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sft_sig:
        sft_kwargs["max_length"] = max_seq_length
    elif "max_seq_length" in sft_sig:
        sft_kwargs["max_seq_length"] = max_seq_length
    if "dataset_kwargs" in sft_sig:
        sft_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

    sft_args = SFTConfig(**sft_kwargs)
    if hasattr(sft_args, "dataset_kwargs") and not sft_args.dataset_kwargs:
        sft_args.dataset_kwargs = {"skip_prepare_dataset": True}
    sft_args.logging_dir = str(run_dir / "logs" / "tensorboard")
    if "warmup_ratio" in train_cfg:
        sft_args.warmup_ratio = float(train_cfg["warmup_ratio"])

    trainer_kwargs = {
        "model": model,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_eval,
        "data_collator": data_collator,
        "args": sft_args,
    }
    sig_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in sig_params else "tokenizer"] = tokenizer
    return SFTTrainer(**trainer_kwargs)


def save_best_and_last_checkpoints(trainer, tokenizer, run_dir: Path, temp_dir: Path):
    best_dir = run_dir / "checkpoints" / "best"
    last_dir = run_dir / "checkpoints" / "last"
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    ckpt_dirs = sorted(
        [d for d in temp_dir.glob("checkpoint-*") if d.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    if ckpt_dirs:
        for item in ckpt_dirs[-1].iterdir():
            if item.is_file():
                shutil.copy2(item, last_dir / item.name)
    else:
        trainer.save_model(str(last_dir))
        tokenizer.save_pretrained(str(last_dir))

    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def finalize_run_records(run_dir: Path, config: dict, eval_payload: dict):
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["end_time"] = datetime.now().isoformat()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_loss = eval_payload.get("eval_loss", "N/A")
    eval_loss_str = f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else str(eval_loss)
    notes_path = run_dir / "NOTES.md"
    if notes_path.exists():
        notes = notes_path.read_text(encoding="utf-8")
        summary = (
            f"- 训练耗时: {eval_payload.get('train_runtime_min', 'N/A')} 分钟\n"
            f"- 最终 `val_loss`: {eval_loss_str}\n"
            f"- 若已配置 `eval-platform`，请部署 SSE 端点并运行 `evalctl` 补充业务评测结论。"
        )
        notes = notes.replace("## 结果\n\n\n", f"## 结果\n{summary}\n\n")
        notes_path.write_text(notes, encoding="utf-8")

    print(f"Finalized run records in {run_dir}. Remember to update NOTES.md, experiments.md, and lessons.md.")
    flush_dual_loggers()


def sync_to_hf_hub(run_dir: Path, hf_repo_id: str | None, version_tag: str = "v1.0.0"):
    target_repo = (hf_repo_id or "").strip()
    if not target_repo:
        print("Skipping Hugging Face Hub push (no hf_repo_id / HF_REPO_ID configured).")
        return

    flush_dual_loggers()
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN", "").strip() or None)
        api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True)
        best_dir = run_dir / "checkpoints" / "best"
        if best_dir.exists() and any(best_dir.iterdir()):
            api.upload_folder(
                folder_path=str(best_dir),
                repo_id=target_repo,
                repo_type="model",
                commit_message=f"Release {version_tag}: Best LoRA adapter from {run_dir.name}",
                ignore_patterns=["optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state*"],
            )
            print(f"Uploaded best adapter to https://huggingface.co/{target_repo}")
    except Exception as exc:
        print(f"Hugging Face upload skipped/failed: {exc}")


def release_colab_runtime(auto_disconnect: bool):
    flush_dual_loggers()
    if auto_disconnect and os.path.exists("/content"):
        try:
            from google.colab import runtime

            time.sleep(2)
            runtime.unassign()
        except Exception as exc:
            print(f"Colab runtime unassign skipped: {exc}")


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT training entrypoint")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--exp_config", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="exp_sft")
    parser.add_argument("--auto_disconnect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=None)
    args, _ = parser.parse_known_args()

    config = load_merged_config(args.base_config, args.exp_config, args.enable_thinking)
    ensure_colab_working_dir(config)
    init_credentials(config)

    run_dir = setup_experiment_run(config, args.exp_name)
    model, tokenizer = load_model_and_tokenizer(config)
    model = setup_lora(model, config)
    train_dataset, eval_dataset = load_and_prepare_datasets(config)

    temp_ckpt_dir = Path("/content/temp_training") if Path("/content").exists() else (run_dir / "checkpoints" / "temp_training")
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, config, run_dir, temp_ckpt_dir)

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    eval_payload = {
        "train_runtime_min": round(train_result.metrics.get("train_runtime", 0) / 60, 2),
        "train_loss": train_result.metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "metrics": eval_metrics,
    }
    (run_dir / "eval_results.json").write_text(json.dumps(eval_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    save_best_and_last_checkpoints(trainer, tokenizer, run_dir, temp_ckpt_dir)
    finalize_run_records(run_dir, config, eval_payload)

    repo_to_push = config.get("model", {}).get("hf_repo_id") or os.environ.get("HF_REPO_ID")
    if "smoke" in args.exp_name.lower():
        repo_to_push = None
    sync_to_hf_hub(run_dir, repo_to_push)
    release_colab_runtime(args.auto_disconnect)


if __name__ == "__main__":
    main()
