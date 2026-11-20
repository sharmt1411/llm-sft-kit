"""SessionStart hook: probe evalctl CLI and backend availability for eval-platform gating."""

import json
import os
import shutil
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _read_stdin_payload() -> dict:
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return {}
        raw = sys.stdin.read().strip()
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _has_eval_disabled_marker(workspace_paths: list[str]) -> bool:
    if os.environ.get("EVAL_PLATFORM_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        return True
    for wp in workspace_paths:
        try:
            p = Path(wp)
            if (p / ".eval_disabled").exists():
                return True
        except Exception:
            continue
    return False


def _probe_backend(base_url: str, timeout_s: float = 0.35) -> tuple[bool, str]:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            pass
    except OSError as exc:
        return False, f"backend port {host}:{port} unreachable ({exc.__class__.__name__})"

    health_url = base_url.rstrip("/") + "/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return (200 <= resp.status < 500), f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        # 404/401 still proves HTTP server is listening; only 5xx or connection errors fail
        return exc.code < 500, f"HTTP {exc.code}"
    except Exception as exc:
        return False, f"HTTP probe failed ({exc.__class__.__name__})"


def main() -> int:
    payload = _read_stdin_payload()
    workspace_paths = payload.get("workspacePaths") or []

    if _has_eval_disabled_marker(workspace_paths):
        msg = (
            "[llm-sft-kit] eval-platform is DISABLED for this workspace (.eval_disabled or env). "
            "Do NOT invoke the `eval-platform` skill and do NOT write ad-hoc local evaluation fallback scripts."
        )
        print(json.dumps({"injectSteps": [{"systemMessage": {"systemMessage": msg}}]}, ensure_ascii=False))
        return 0

    evalctl_bin = shutil.which("evalctl")
    if not evalctl_bin:
        msg = (
            "[llm-sft-kit] eval-platform is DISABLED (`evalctl` CLI not found in PATH). "
            "Do NOT invoke the `eval-platform` skill and do NOT write ad-hoc local evaluation fallback scripts. "
            "Focus on training scaffolding, Colab ops, and SSE serving."
        )
        print(json.dumps({"injectSteps": [{"systemMessage": {"systemMessage": msg}}]}, ensure_ascii=False))
        return 0

    in_process_mode = os.environ.get("EVALCTL_IN_PROCESS", "").strip().lower() in ("1", "true", "yes")
    base_url = os.environ.get("EVALCTL_BASE_URL", "http://127.0.0.1:9021")
    backend_ok, detail = (True, "in-process mode") if in_process_mode else _probe_backend(base_url)

    if backend_ok:
        msg = (
            f"[llm-sft-kit] eval-platform is READY (evalctl={evalctl_bin}, {detail}). "
            "Full `evalctl` benchmarking, simulation, and data-flywheel workflows are enabled."
        )
    else:
        msg = (
            f"[llm-sft-kit] eval-platform is DISABLED (`evalctl` found, but {detail}). "
            "Do NOT invoke `eval-platform` unless the user starts the backend or requests `--in-process` mode, "
            "and do NOT create ad-hoc local evaluation fallback scripts."
        )

    print(json.dumps({"injectSteps": [{"systemMessage": {"systemMessage": msg}}]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
