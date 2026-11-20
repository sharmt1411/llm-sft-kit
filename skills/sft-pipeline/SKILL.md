---
name: sft-pipeline
description: >-
  Use this skill to run daily LLM LoRA/SFT iterations on an initialized project:
  prepare versioned datasets (cleaning, anti-leakage stratified split,
  configurable source weights, manifest.json), launch pure-assistant-reply-loss
  SFT training, and archive runs (config.yaml, meta.json, NOTES.md,
  experiments.md, lessons.md). For initial project setup, use `sft-init`.
---

# LLM SFT Training & Archival Pipeline

> Prerequisite: if the workspace has not been scaffolded or configured yet (`configs/base.yaml`, `scripts/prepare_data.py`, `AGENTS.md` project config slots), use the **`sft-init`** skill first.

## 1. Prepare Versioned Datasets

Place source `.jsonl` files (each line `{"messages": [...]}`) into `data/raw/`. Configure per-source upsampling weights in `configs/base.yaml` under `data.source_weights` (substring match against filename).

Run:

```bash
python scripts/prepare_data.py --version v1 --config configs/base.yaml
```

Key guarantees built into `prepare_data.py`:
- Strips `<think>...</think>` blocks from assistant messages when `enable_thinking` is false.
- Runs deterministic per-source stratified train/val splitting (preventing cross-split leakage).
- Applies `source_weights` only to `train.jsonl` (keeping `val.jsonl` at 1x unweighted distribution).
- Refuses to overwrite an existing `data/processed/vN/` unless `--overwrite` is passed.
- Generates `manifest.json` with SHA256 hashes and sample statistics.

## 2. Configure & Run Experiments

Keep `configs/base.yaml` stable and write experiment diffs in `configs/exp_XXXX_<desc>.yaml`:

```bash
# Quick 2-step smoke test
python scripts/train.py --exp_config configs/exp_0001_smoke_test.yaml --exp_name exp0001_smoke_test

# Full run
python scripts/train.py --exp_config configs/exp_0002_baseline.yaml --exp_name exp0002_baseline
```

Refer to [hardware_pitfalls.md](./references/hardware_pitfalls.md) before sizing GPUs, `eval_steps`, or `max_seq_length`.

## 3. Mandatory Post-Run Archival

After every training run, verify `runs/YYYY-MM-DD_<exp_name>/` contains `config.yaml`, `meta.json`, `logs/train.log`, `checkpoints/best`, `checkpoints/last`, `eval_results.json`, and `NOTES.md`, then update:
1. `runs/.../NOTES.md` (Conclusions, Issues/Anomalies, Next Steps)
2. `experiments.md` (one summary row per run)
3. `lessons.md` (cross-experiment findings and pitfalls)

See [notes_and_schema.md](./references/notes_and_schema.md) for directory layout and exact templates.
