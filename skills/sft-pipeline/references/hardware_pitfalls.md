# Hardware, Precision & Training Pitfalls

Distilled cross-experiment rules for LLM LoRA/SFT workloads:

1. **T4 / L4 GPU & BF16 / `offload_layers`**:
   - T4 (Turing `sm_75`) does NOT support native `bfloat16`. Always auto-fallback to `float16` when `torch.cuda.is_bf16_supported()` is false.
   - Training >= 27B models on a free 15GB T4 with high CPU offload causes OOM/RAM deadlock. On L4 (~22GB VRAM) or 24GB GPUs, 27B–35B 4-bit models still exceed VRAM during loading/training and require `offload_layers: "auto"` (plus High-RAM system memory) to avoid bitsandbytes CPU-spill errors; use >= 40GB VRAM (A100) for full-speed runs.
   - Always `import unsloth` before `transformers`, `peft`, and `trl` so all memory/kernel patches take effect.
2. **Inference Memory-Bandwidth Wall**:
   - Autoregressive token generation at batch=1 is memory-bandwidth bound. For 4-bit 27B (~15GB weights), L4 (300 GB/s) physically caps around ~4–5 tok/s even with zero CPU offload. Do not misdiagnose bandwidth ceilings as CPU offloading without checking `hf_device_map` and VRAM allocation.
3. **TRL / Transformers Compatibility**:
   - Inspect `SFTTrainer.__init__` signature to pass `processing_class` (TRL >= 0.12 / 1.x) or `tokenizer` (older TRL).
   - `apply_chat_template(..., tokenize=True)` may return `BatchEncoding` or nested `[[...]]` in newer Transformers / multimodal Processors; always unwrap to `list[int]` before slicing labels.
   - Multimodal Processors (e.g., VL processors) must be called with `tokenizer(text=..., ...)` keyword argument or unwrapped via `getattr(processor, "tokenizer", processor)`.
4. **Validation Inflection & `eval_steps` Budget**:
   - Multi-epoch dense SFT often hits best `eval_loss` around epoch 2.5–3.5 while later epochs over-sharpen confidence; keep `load_best_model_at_end=True` and `metric_for_best_model="eval_loss"`.
   - With >= 500 validation samples, `eval_steps=50` can consume 30–40% of total wall time; prefer `100–150`.
5. **Cloud Cache Isolation**:
   - On Colab (`/content`), redirect `HF_HOME`, `TORCH_HOME`, `UNSLOTH_COMPILE_LOCATION`, `TRITON_CACHE_DIR`, and intermediate trainer `output_dir` checkpoints to `/tmp` or `/root/.cache` so Google Drive FUSE sync is not flooded.
   - On Kaggle (20GB `/kaggle/working` quota), redirect `HF_HOME` to `/tmp/hf_cache` and prefer pre-quantized 4-bit base checkpoints.
6. **Streaming & Tunnel Timeouts**:
   - Free Cloudflare tunnels (`trycloudflare.com`) enforce ~100s non-streaming request timeouts. Always use `stream=True` SSE for long generations (e.g., > 2000 tokens) and launch `cloudflared` with `start_new_session=True` so notebook cell interrupts do not kill the tunnel.
   - Suppress one-time C++/kernel stderr warnings (e.g., `causal_conv1d`) via a 3-token silent warmup before streaming output.
