---
name: sft-serving
description: >-
  Use this skill to serve a fine-tuned LoRA adapter over an OpenAI-compatible SSE
  streaming endpoint (/v1/chat/completions) on single or multi-GPU hosts (such as
  Kaggle T4x2), measure TTFT/TPOT/TPS, run single-instance base-vs-LoRA A/B requests
  via disable_adapter(), and expose a public tunnel for evalctl.
---

# SFT Model Serving & Tunneling

Deploy the trained LoRA adapter as an OpenAI-compatible SSE endpoint so external callers or `eval-platform` (`evalctl`) can benchmark and inspect it.

## 1. Key Runtime Patterns Built Into `deploy_t4x2_tpl.py`

- **Cache Redirection**: Sets `HF_HOME=/tmp/hf_cache` and `TORCH_HOME=/tmp/torch_cache` to protect 20GB Kaggle working quotas and Drive sync folders.
- **Multi-GPU Balanced Placement**: Uses `device_map="auto"` with explicit per-GPU `max_memory` (`MAX_GPU_MEMORY_GIB`, default `14.2`) so 4-bit large models stay 100% in VRAM across 2×16GB T4 GPUs without CPU offload.
- **FP16 Precision on Turing**: Enforces `torch.float16` on GPUs without native BF16 support (T4 `sm_75`).
- **Silent 3-Token Warmup**: Clears one-time kernel/C++ warnings before serving requests.
- **SSE Streaming (`/v1/chat/completions`, `stream=True`)**: Prevents ~100s gateway timeouts on free Cloudflare tunnels (`trycloudflare.com`) for long outputs.
- **Dynamic Base vs. LoRA A/B**: Pass `disable_adapter: true` (or `model: "base"`) in requests to toggle `MODEL.disable_adapter()` on a single loaded instance.
- **Detached Tunnel Process**: Launches `cloudflared` with `start_new_session=True` so notebook cell interrupts do not kill the tunnel.

## 2. Launch Serving Endpoint

```bash
BASE_MODEL_ID="unsloth/your-4bit-base" \
LORA_REPO_ID="your-org/your-lora-repo" \
python scripts/deploy_serving.py
```

Key environment variables (or CLI flags):
- `BASE_MODEL_ID`: Pre-quantized 4-bit base model ID or local path.
- `LORA_REPO_ID`: LoRA adapter repo ID or local checkpoint path (e.g., `runs/.../checkpoints/best`).
- `MAX_GPU_MEMORY_GIB`: Per-GPU cap (default `14.2` for 16GB T4).
- `PORT`: Local FastAPI port (default `8000`).
- `NGROK_AUTHTOKEN`: Optional Ngrok token; falls back to `cloudflared` if unset.

## 3. Hand Off to `eval-platform` (If Enabled)

Once the tunnel URL `https://<tunnel-host>` is printed, connect it to `evalctl` (with `--workers 1` for single-instance GPU lock serialization):
If you are not sure about <model_id>，ask user to confirm.

```bash
evalctl model update <model_id> --api-url https://<tunnel-host>/v1/chat/completions
evalctl model test <model_id>
```
