"""Auto-launch an external vLLM OpenAI-compatible server for --validate --vllm.

When ``embedded: false``, the trace sender normally expects a ``vllm serve``
already running at the configured endpoint.  To keep
``uv run python main.py --validate --vllm`` runnable as a single command, the
sender auto-starts the server when nothing answers at the endpoint, waits for
readiness, and shuts it down afterwards.

The serve argv is built from ``vllm.engine_args`` (the same fields the
embedded path passes to ``AsyncEngineArgs``) plus the optional server flags
``language_model_only`` / ``reasoning_parser`` / ``default_chat_template_kwargs``.

If a server is already running at the endpoint it is reused and left untouched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

# engine_args key -> vLLM serve CLI flag (scalar values; skipped when None).
_FLAG_MAP = {
    "tensor_parallel_size": "--tensor-parallel-size",
    "pipeline_parallel_size": "--pipeline-parallel-size",
    "data_parallel_size": "--data-parallel-size",
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "max_model_len": "--max-model-len",
    "dtype": "--dtype",
    "kv_cache_dtype": "--kv-cache-dtype",
    "seed": "--seed",
    "load_format": "--load-format",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "reasoning_parser": "--reasoning-parser",
}

# engine_args key -> vLLM serve CLI flag (booleans; passed only when True).
_BOOL_FLAGS = {
    "enable_prefix_caching": "--enable-prefix-caching",
    "enforce_eager": "--enforce-eager",
    "language_model_only": "--language-model-only",
}

# How long to wait for the server to become ready before giving up.
_READY_TIMEOUT_S = 600.0


def endpoint_reachable(endpoint: str, timeout: float = 3.0) -> bool:
    """Return True if an OpenAI-compatible server already answers /models."""
    try:
        r = httpx.get(f"{endpoint}/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _vllm_bin() -> str:
    """Locate the `vllm` console script (prefer the one next to sys.executable)."""
    candidate = Path(sys.executable).parent / "vllm"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("vllm")
    if found:
        return found
    raise RuntimeError(
        f"cannot find the `vllm` executable next to {sys.executable}"
    )


def build_serve_cmd(vllm_cfg: dict) -> list[str]:
    """Build the `vllm serve` argv from the config's vllm block."""
    engine = vllm_cfg.get("engine_args", {}) or {}
    model = vllm_cfg.get("model")
    if not model:
        raise ValueError("vllm.model is required to start a server")

    cmd = [_vllm_bin(), "serve", model]

    for key, flag in _FLAG_MAP.items():
        val = engine.get(key)
        if val is not None:
            cmd += [flag, str(val)]

    for key, flag in _BOOL_FLAGS.items():
        if engine.get(key, False):
            cmd.append(flag)

    dct = engine.get("default_chat_template_kwargs")
    if dct:
        cmd += ["--default-chat-template-kwargs", json.dumps(dct)]

    p = urlparse(vllm_cfg.get("endpoint", "http://localhost:8000/v1"))
    cmd += ["--host", p.hostname or "localhost", "--port", str(p.port or 8000)]

    return cmd


class SpawnedServer:
    """Handle to a server process we started ourselves (must be stopped)."""

    def __init__(self, proc: subprocess.Popen, log_path: Path):
        self.proc = proc
        self.log_path = log_path


def spawn_server(vllm_cfg: dict, output_dir) -> SpawnedServer:
    """Start `vllm serve` with the config's args, logging to output_dir."""
    cmd = build_serve_cmd(vllm_cfg)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "vllm_serve.log"
    log_f = open(log_path, "w")

    print("No vLLM server at the endpoint — starting one...")
    print("  " + " ".join(cmd))
    print(f"  log: {log_path}")

    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return SpawnedServer(proc, log_path)


def _tail(path, n: int = 30) -> str:
    if not path:
        return "(no log)"
    try:
        lines = Path(path).read_text().splitlines()
        return "\n".join(lines[-n:]) if lines else "(empty log)"
    except Exception as e:
        return f"(cannot read log {path}: {e})"


def wait_for_server(endpoint: str, *, proc=None, log_path=None,
                    timeout: float = _READY_TIMEOUT_S) -> None:
    """Block until the endpoint answers /models, or raise with the log tail."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if endpoint_reachable(endpoint, timeout=2.0):
            print("vLLM server ready.")
            return
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"vLLM serve exited early (rc={proc.returncode}). "
                f"Log tail:\n{_tail(log_path)}"
            )
        time.sleep(2.0)

    if proc is not None:
        proc.terminate()
    raise RuntimeError(
        f"Timed out after {timeout:.0f}s waiting for vLLM server at "
        f"{endpoint}. Log tail:\n{_tail(log_path)}"
    )


def stop_server(server: SpawnedServer) -> None:
    """Terminate a server we spawned (safe to call once)."""
    if server.proc.poll() is not None:
        return
    print("Stopping vLLM server...")
    try:
        server.proc.terminate()
        server.proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.proc.kill()
        server.proc.wait()
