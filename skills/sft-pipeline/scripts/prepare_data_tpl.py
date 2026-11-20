"""SFT dataset preparation pipeline: data/raw -> data/processed/vN.

Features:
1. Standardize messages to ChatML (system/user/assistant), strip empty/invalid entries.
2. Optional <think>...</think> stripping when enable_thinking=False.
3. Optional multi-turn expansion per assistant turn.
4. Per-source deterministic stratified train/val split (anti-leakage).
5. Config-driven upsampling weights (data.source_weights in base.yaml) applied ONLY to train.jsonl.
6. Version overwrite guard and automatic manifest.json with SHA256 hashes.
"""

import argparse
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def get_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hex digest for a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_yaml_config(config_path: Path) -> dict:
    """Load YAML config file if available."""
    if config_path.exists() and yaml:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def clean_record_messages(record: dict, enable_thinking: bool = False) -> list[dict] | None:
    """Clean a raw JSONL record into a valid ChatML messages list."""
    raw_msgs = record.get("messages")
    if not isinstance(raw_msgs, list) or not raw_msgs:
        return None

    cleaned_msgs = []
    has_user = False
    has_assistant = False

    for msg in raw_msgs:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content", "")
        if role not in ("system", "user", "assistant") or content is None:
            continue

        content = str(content).strip()
        if not enable_thinking and role == "assistant":
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()

        if not content:
            continue

        if role == "user":
            has_user = True
        elif role == "assistant":
            has_assistant = True
        cleaned_msgs.append({"role": role, "content": content})

    return cleaned_msgs if (has_user and has_assistant) else None


def expand_dialogue_turns(messages: list[dict]) -> list[list[dict]]:
    """Expand multi-turn dialogues into progressive prefixes ending on each assistant turn."""
    if not messages:
        return []

    sys_prefix = [messages[0]] if messages[0].get("role") == "system" else []
    dialogue = messages[1:] if sys_prefix else messages

    expanded = []
    for idx, msg in enumerate(dialogue):
        if msg["role"] == "assistant":
            sub = dialogue[: idx + 1]
            if any(m.get("role") == "user" for m in sub):
                expanded.append(sys_prefix + sub)
    return expanded if expanded else [messages]


def determine_source_weight(filename: str, config: dict) -> int:
    """Resolve upsampling weight from `data.source_weights` mapping (substring match, case-insensitive)."""
    data_cfg = config.get("data", {})
    weights_map = data_cfg.get("source_weights", {}) or {}
    fn_lower = filename.lower()

    if isinstance(weights_map, dict):
        for pattern, weight in weights_map.items():
            if str(pattern).lower() in fn_lower:
                return max(1, int(weight))

    return max(1, int(data_cfg.get("default_weight", 1)))


def process_dataset(
    raw_dir: Path,
    output_dir: Path,
    version: str,
    config_path: Path = Path("configs/base.yaml"),
    val_ratio: float = 0.1,
    seed: int = 3407,
    expand_multiturn: bool = True,
    enable_thinking: bool | None = None,
    overwrite: bool = False,
):
    """Run cleaning, per-source stratified splitting, weighted train upsampling, and manifest generation."""
    version = str(version or "").strip()
    if not version:
        raise ValueError("Dataset --version is required (e.g. v1, v2).")

    train_out_path = output_dir / "train.jsonl"
    val_out_path = output_dir / "val.jsonl"
    if (train_out_path.exists() or val_out_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Target directory already contains train/val data: {output_dir}. "
            "Use a new --version or pass --overwrite."
        )

    config = load_yaml_config(config_path) if config_path.exists() else {}
    if enable_thinking is None:
        enable_thinking = bool(config.get("model", {}).get("enable_thinking", False))

    raw_files = sorted(raw_dir.glob("*.jsonl"))
    if not raw_files:
        raise FileNotFoundError(f"No .jsonl files found in {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    source_stats = {}
    all_train_samples = []
    all_val_samples = []

    print(f"Preparing dataset version={version} from {raw_dir} (enable_thinking={enable_thinking})")

    for file_path in raw_files:
        fname = file_path.name
        fhash = get_file_sha256(file_path)
        weight = determine_source_weight(fname, config)

        cleaned_records = []
        raw_line_count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                raw_line_count += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                cleaned_msgs = clean_record_messages(item, enable_thinking=enable_thinking)
                if not cleaned_msgs:
                    continue

                if expand_multiturn and len(cleaned_msgs) > 3:
                    for slice_msgs in expand_dialogue_turns(cleaned_msgs):
                        cleaned_records.append({"messages": slice_msgs})
                else:
                    cleaned_records.append({"messages": cleaned_msgs})

        total_cleaned = len(cleaned_records)
        if total_cleaned == 0:
            print(f"  [skip] {fname}: 0 valid records")
            continue

        rng.shuffle(cleaned_records)
        if total_cleaned == 1:
            val_count, train_count = 0, 1
        else:
            val_count = min(total_cleaned - 1, max(1, round(total_cleaned * val_ratio)))
            train_count = total_cleaned - val_count

        val_slice = cleaned_records[:val_count]
        train_slice = cleaned_records[val_count:]
        weighted_train_slice = train_slice * weight

        all_train_samples.extend(weighted_train_slice)
        all_val_samples.extend(val_slice)

        source_stats[fname] = {
            "sha256": fhash,
            "raw_lines": raw_line_count,
            "valid_samples": total_cleaned,
            "weight": weight,
            "train_distinct": train_count,
            "train_weighted": len(weighted_train_slice),
            "val_distinct": val_count,
        }
        print(
            f"  {fname}: raw={raw_line_count} valid={total_cleaned} "
            f"train={train_count}x{weight}={len(weighted_train_slice)} val={val_count}"
        )

    rng.shuffle(all_train_samples)
    rng.shuffle(all_val_samples)

    with open(train_out_path, "w", encoding="utf-8") as f:
        for sample in all_train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(val_out_path, "w", encoding="utf-8") as f:
        for sample in all_val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    train_sha256 = get_file_sha256(train_out_path)
    val_sha256 = get_file_sha256(val_out_path)

    manifest = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "val_ratio": val_ratio,
        "enable_thinking": enable_thinking,
        "expand_multiturn": expand_multiturn,
        "summary": {
            "total_raw_sources": len(raw_files),
            "total_train_samples": len(all_train_samples),
            "total_val_samples": len(all_val_samples),
        },
        "sources": source_stats,
        "files": {
            "train.jsonl": {"sha256": train_sha256, "count": len(all_train_samples), "path": train_out_path.as_posix()},
            "val.jsonl": {"sha256": val_sha256, "count": len(all_val_samples), "path": val_out_path.as_posix()},
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done: train={len(all_train_samples)}, val={len(all_val_samples)}, manifest={manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare versioned SFT datasets.")
    parser.add_argument("--version", type=str, required=True, help="Dataset version tag (e.g. v1, v2)")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw JSONL directory")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Base YAML config path")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--expand_multiturn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    process_dataset(
        raw_dir=Path(args.raw_dir),
        output_dir=Path("data/processed") / args.version,
        version=args.version,
        config_path=Path(args.config),
        val_ratio=args.val_ratio,
        seed=args.seed,
        expand_multiturn=args.expand_multiturn,
        enable_thinking=args.enable_thinking,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
