"""Generic 4-bit + LoRA OpenAI-compatible SSE serving & tunnel script (single/multi-GPU).

Supports:
- Multi-GPU VRAM balancing via device_map="auto" and max_memory
- Single-instance Base vs. LoRA A/B switching via PeftModel.disable_adapter()
- Streaming SSE (/v1/chat/completions) and /health endpoints
- TTFT / TPOT / TPS latency metrics
- Ngrok or Cloudflared (start_new_session=True) public tunneling
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import threading
import time
import uuid
from contextlib import nullcontext
from typing import Any, Optional

if os.path.exists("/kaggle"):
    os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
    os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
elif os.path.exists("/content"):
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("TORCH_HOME", "/root/.cache/torch")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    import transformers
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from peft import PeftModel
    from pydantic import BaseModel, Field
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        TextIteratorStreamer,
    )
    import uvicorn
    _SERVING_IMPORT_ERR = None
except ImportError as exc:
    torch = None
    transformers = None
    FastAPI = None
    HTTPException = Exception
    StreamingResponse = None
    PeftModel = None
    BaseModel = object
    Field = lambda default=None, **kw: default
    uvicorn = None
    _SERVING_IMPORT_ERR = exc


def get_secret(key: str) -> Optional[str]:
    val = os.environ.get(key, "").strip()
    if val:
        return val
    for mod_name, cls_or_fn in (("kaggle_secrets", "UserSecretsClient"), ("google.colab", "userdata")):
        try:
            mod = __import__(mod_name, fromlist=[cls_or_fn])
            if mod_name == "kaggle_secrets":
                secret_val = getattr(mod, cls_or_fn)().get_secret(key)
            else:
                secret_val = getattr(mod, cls_or_fn).get(key)
            if secret_val:
                os.environ[key] = str(secret_val).strip()
                return os.environ[key]
        except Exception:
            continue
    return None


def print_gpu_status(prefix: str = "GPU status") -> list[dict]:
    if not torch.cuda.is_available():
        print(f"[{prefix}] No CUDA GPU detected.")
        return []
    status_list = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info = {
            "gpu_id": i,
            "name": props.name,
            "allocated_gb": round(torch.cuda.memory_allocated(i) / (1024**3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved(i) / (1024**3), 2),
            "total_gb": round(props.total_memory / (1024**3), 2),
        }
        status_list.append(info)
        print(
            f"  cuda:{i} ({info['name']}): alloc={info['allocated_gb']}GB "
            f"res={info['reserved_gb']}GB total={info['total_gb']}GB"
        )
    return status_list


def apply_chat_template(
    tokenizer_or_processor,
    messages: list[dict],
    enable_thinking: bool = False,
    reasoning_effort: str = "low",
) -> str:
    chat_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
    if enable_thinking and reasoning_effort:
        chat_kwargs["reasoning_effort"] = reasoning_effort

    for tpl_obj in (tokenizer_or_processor, getattr(tokenizer_or_processor, "tokenizer", None)):
        if tpl_obj is None or not hasattr(tpl_obj, "apply_chat_template"):
            continue
        try:
            return tpl_obj.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_kwargs)
        except TypeError:
            try:
                return tpl_obj.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, chat_template_kwargs=chat_kwargs
                )
            except TypeError:
                return tpl_obj.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise RuntimeError("Tokenizer/Processor does not support apply_chat_template")


def load_model(base_model_id: str, lora_repo_id: Optional[str], max_gpu_gib: float = 14.2):
    hf_token = get_secret("HF_TOKEN")
    num_gpus = torch.cuda.device_count()
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    max_memory = {i: f"{max_gpu_gib}GiB" for i in range(num_gpus)} if num_gpus > 0 else {}
    if max_memory:
        max_memory["cpu"] = "24GiB"

    tok_source = lora_repo_id or base_model_id
    try:
        raw_tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True, token=hf_token)
    except Exception:
        raw_tokenizer = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True, token=hf_token)

    text_tokenizer = getattr(raw_tokenizer, "tokenizer", None) or raw_tokenizer
    if getattr(text_tokenizer, "pad_token", None) is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token or "<|im_end|>"
    text_tokenizer.padding_side = "left"

    cfg = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True, token=hf_token)
    model_cls = AutoModelForCausalLM
    arch_list = getattr(cfg, "architectures", None) or []
    if arch_list:
        model_cls = getattr(transformers, arch_list[0], None) or AutoModelForCausalLM

    base_model = model_cls.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto" if num_gpus > 0 else None,
        max_memory=max_memory or None,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
    )

    if lora_repo_id:
        model = PeftModel.from_pretrained(base_model, lora_repo_id, torch_dtype=compute_dtype, token=hf_token)
    else:
        model = base_model
    model.eval()

    print_gpu_status("Model loaded")
    # 3-token silent warmup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    warmup_prompt = apply_chat_template(raw_tokenizer, [{"role": "user", "content": "hi"}], enable_thinking=False)
    inputs = text_tokenizer(text=str(warmup_prompt), return_tensors="pt").to(device)
    with torch.inference_mode():
        model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=3,
            do_sample=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warmup completed.")
    return model, raw_tokenizer, text_tokenizer


class _DummyApp:
    def get(self, *_a, **_kw):
        return lambda fn: fn

    def post(self, *_a, **_kw):
        return lambda fn: fn


app = FastAPI(title="LLM SFT Inference API", version="1.0.0") if FastAPI is not None else _DummyApp()
MODEL = None
RAW_TOKENIZER = None
TEXT_TOKENIZER = None
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "")
LORA_REPO_ID = os.environ.get("LORA_REPO_ID", "")
DEFAULT_ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "false").lower() in ("1", "true", "yes")
GPU_THREAD_LOCK = threading.Lock()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "sft-lora"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)
    stream: bool = False
    enable_thinking: Optional[bool] = None
    disable_adapter: bool = False


@app.get("/health")
async def health_check():
    return {
        "status": "ok" if MODEL is not None else "loading",
        "base_model": BASE_MODEL_ID,
        "lora_adapter": LORA_REPO_ID or None,
        "supports_disable_adapter": hasattr(MODEL, "disable_adapter") if MODEL is not None else False,
        "gpus": print_gpu_status("API /health"),
    }


def _run_sync_generation(
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    enable_thinking: bool,
    disable_adapter: bool,
) -> tuple[str, dict]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prompt_text = apply_chat_template(RAW_TOKENIZER, messages, enable_thinking=enable_thinking)
    model_inputs = TEXT_TOKENIZER(text=str(prompt_text), return_tensors="pt").to(device)
    prompt_tokens = int(model_inputs["input_ids"].shape[1])

    streamer = TextIteratorStreamer(TEXT_TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    do_sample = temperature > 1e-4
    gen_kwargs: dict[str, Any] = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs.get("attention_mask"),
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "streamer": streamer,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    start_time = time.perf_counter()
    first_token_time = None
    chunks: list[str] = []

    def _target():
        ctx = MODEL.disable_adapter() if (disable_adapter and hasattr(MODEL, "disable_adapter")) else nullcontext()
        with GPU_THREAD_LOCK, torch.inference_mode(), ctx:
            MODEL.generate(**gen_kwargs)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    for piece in streamer:
        if first_token_time is None and piece:
            first_token_time = time.perf_counter()
        chunks.append(piece)
    thread.join()
    end_time = time.perf_counter()

    reply = "".join(chunks).strip()
    if not enable_thinking and "<think>" in reply:
        reply = re.sub(r"<think>[\s\S]*?</think>", "", reply).strip()

    output_tokens = len(TEXT_TOKENIZER.encode(reply, add_special_tokens=False))
    first_t = first_token_time or end_time
    gen_time_s = max(0.001, end_time - first_t)
    metrics = {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_time_s": round(max(0.001, end_time - start_time), 2),
        "ttft_ms": round(max(0.0, first_t - start_time) * 1000.0, 1),
        "tpot_ms": round((gen_time_s / max(1, output_tokens - 1)) * 1000.0, 1),
        "tps": round(output_tokens / gen_time_s, 2) if output_tokens > 0 else 0.0,
        "adapter_disabled": bool(disable_adapter),
    }
    return reply, metrics


@app.post("/v1/chat/completions")
async def openai_chat_completions(req: ChatCompletionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    enable_thinking = DEFAULT_ENABLE_THINKING if req.enable_thinking is None else req.enable_thinking
    use_disable_adapter = bool(req.disable_adapter or "base" in req.model.lower())
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if not req.stream:
        reply, metrics = await asyncio.to_thread(
            _run_sync_generation,
            raw_messages,
            req.max_tokens,
            req.temperature,
            req.top_p,
            req.repetition_penalty,
            enable_thinking,
            use_disable_adapter,
        )
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": metrics["output_tokens"],
                "total_tokens": metrics["prompt_tokens"] + metrics["output_tokens"],
            },
            "latency_metrics": metrics,
        }

    async def _sse_generator():
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        prompt_text = apply_chat_template(RAW_TOKENIZER, raw_messages, enable_thinking=enable_thinking)
        model_inputs = TEXT_TOKENIZER(text=str(prompt_text), return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(TEXT_TOKENIZER, skip_prompt=True, skip_special_tokens=True)
        do_sample = req.temperature > 1e-4
        gen_kwargs: dict[str, Any] = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs.get("attention_mask"),
            "max_new_tokens": req.max_tokens,
            "do_sample": do_sample,
            "repetition_penalty": req.repetition_penalty,
            "streamer": streamer,
        }
        if do_sample:
            gen_kwargs["temperature"] = req.temperature
            gen_kwargs["top_p"] = req.top_p

        def _run():
            ctx = MODEL.disable_adapter() if (use_disable_adapter and hasattr(MODEL, "disable_adapter")) else nullcontext()
            with GPU_THREAD_LOCK, torch.inference_mode(), ctx:
                MODEL.generate(**gen_kwargs)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        loop = asyncio.get_running_loop()
        it = iter(streamer)
        while True:
            piece = await loop.run_in_executor(None, next, it, None)
            if piece is None:
                break
            if not piece:
                continue
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        thread.join()
        end_chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_sse_generator(), media_type="text/event-stream")


def start_public_tunnel(port: int) -> str:
    ngrok_token = get_secret("NGROK_AUTHTOKEN")
    if ngrok_token:
        try:
            from pyngrok import ngrok

            ngrok.set_auth_token(ngrok_token)
            ngrok.kill()
            return ngrok.connect(port, "http").public_url
        except Exception as exc:
            print(f"Ngrok failed ({exc}), falling back to cloudflared...")

    cf_bin = "/tmp/cloudflared"
    if not os.path.exists(cf_bin):
        subprocess.run(
            ["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", cf_bin],
            check=True,
        )
        os.chmod(cf_bin, 0o755)

    proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            time.sleep(0.2)
            continue
        m = re.search(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)", line)
        if m:
            return m.group(1)
    return ""


def main():
    global MODEL, RAW_TOKENIZER, TEXT_TOKENIZER, BASE_MODEL_ID, LORA_REPO_ID, DEFAULT_ENABLE_THINKING
    parser = argparse.ArgumentParser(description="Serve 4-bit base + LoRA adapter over OpenAI SSE")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_ID)
    parser.add_argument("--lora-repo", type=str, default=LORA_REPO_ID)
    parser.add_argument("--max-gpu-gib", type=float, default=float(os.environ.get("MAX_GPU_MEMORY_GIB", "14.2")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=DEFAULT_ENABLE_THINKING)
    parser.add_argument("--no-tunnel", action="store_true")
    args = parser.parse_args()

    if _SERVING_IMPORT_ERR is not None:
        raise RuntimeError(f"Missing serving dependencies ({_SERVING_IMPORT_ERR}). Install transformers/peft/fastapi/uvicorn.")

    if not args.base_model:
        raise ValueError("--base-model or BASE_MODEL_ID is required.")
    BASE_MODEL_ID = args.base_model
    LORA_REPO_ID = args.lora_repo
    DEFAULT_ENABLE_THINKING = bool(args.enable_thinking)

    MODEL, RAW_TOKENIZER, TEXT_TOKENIZER = load_model(BASE_MODEL_ID, LORA_REPO_ID or None, args.max_gpu_gib)

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info"),
        daemon=True,
    )
    server_thread.start()
    time.sleep(2.0)

    public_url = "" if args.no_tunnel else start_public_tunnel(args.port)
    print(f"Local endpoint: http://127.0.0.1:{args.port}/v1/chat/completions")
    if public_url:
        print(f"Public endpoint: {public_url}/v1/chat/completions")

    try:
        while server_thread.is_alive():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Server stopped.")


if __name__ == "__main__":
    main()
