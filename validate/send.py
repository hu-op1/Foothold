"""Send trace requests to vLLM via OpenAI-compatible API.

Decodes token IDs to text, sends requests to vLLM, and records timing.
Timing starts from the API call; tokenizer decode is not included.

Timeseries data is captured by polling vLLM's /metrics endpoint
in the background, giving real per-tick throughput from vLLM's
internal scheduler counters (same approach as LLMServingSim's
PD-disagg runner).

Output (matching sim format):
    <output_dir>/meta.json
    <output_dir>/requests.jsonl
    <output_dir>/timeseries.csv
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from sim.trace import load_trace

@dataclass
class _MetricsSample:
    t: float
    num_running: int
    num_waiting: int
    prompt_tokens_cumulative: float
    gen_tokens_cumulative: float
    kv_cache_pct: float

_METRIC_RE = re.compile(
    r"^vllm:(num_requests_running|num_requests_waiting|prompt_tokens_total|"
    r"generation_tokens_total|kv_cache_usage_perc)"
    r"(?:\{[^}]*\})?\s+([\d.e+\-]+)",
    re.MULTILINE,
)

# Metrics that should be summed across engines (counters and gauge counts).
_SUM_METRICS = frozenset({
    "num_requests_running",
    "num_requests_waiting",
    "prompt_tokens_total",
    "generation_tokens_total",
})

# Metrics that should take the maximum across engines.
_MAX_METRICS = frozenset({"kv_cache_usage_perc"})

def _parse_metrics(text: str) -> dict[str, float]:
    """Parse vLLM /metrics output, aggregating across engines.

    vLLM v1 exposes per-engine entries differentiated by the ``engine`` label,
    e.g. two engines produce two lines for the same metric name.  Counters
    and running/waiting-gauge counts are summed; KV cache usage takes the max.
    """
    accum: dict[str, float] = {}
    for m in _METRIC_RE.finditer(text):
        name = m.group(1)
        value = float(m.group(2))
        if name in _SUM_METRICS:
            accum[name] = accum.get(name, 0.0) + value
        elif name in _MAX_METRICS:
            accum[name] = max(accum.get(name, 0.0), value)
        else:
            accum[name] = value  # last-write (shouldn't happen for known keys)
    return accum

def _metrics_base_url(endpoint: str) -> str:
    p = urlparse(endpoint)
    return f"{p.scheme}://{p.netloc}"

async def _poll_metrics(
    metrics_url: str,
    tick_seconds: float,
    samples: list[_MetricsSample],
    stop: asyncio.Event,
) -> None:
    t0 = time.perf_counter()
    consecutive_errors = 0
    async with httpx.AsyncClient(timeout=5.0) as client:
        while not stop.is_set():
            t = round(time.perf_counter() - t0, 3)
            try:
                resp = await client.get(metrics_url)
                resp.raise_for_status()
                m = _parse_metrics(resp.text)
                samples.append(_MetricsSample(
                    t=t,
                    num_running=int(m.get("num_requests_running", 0)),
                    num_waiting=int(m.get("num_requests_waiting", 0)),
                    prompt_tokens_cumulative=m.get("prompt_tokens_total", 0.0),
                    gen_tokens_cumulative=m.get("generation_tokens_total", 0.0),
                    kv_cache_pct=m.get("kv_cache_usage_perc", 0.0),
                ))
                if consecutive_errors:
                    print(f"[metrics] recovered after {consecutive_errors} failures at t={t:.1f}s")
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors == 1:
                    print(f"[metrics] WARNING: poll failed ({e}), metrics data will be incomplete")
                elif consecutive_errors % 20 == 0:
                    print(f"[metrics] still failing after {consecutive_errors} attempts ({e})")
            await asyncio.sleep(tick_seconds)

def run_send(config: dict) -> None:
    vllm_cfg = config.get("vllm", {})
    endpoint = vllm_cfg.get("endpoint", "http://localhost:8000/v1")
    model = vllm_cfg.get("model")
    api_key = vllm_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "EMPTY")
    timeout = vllm_cfg.get("timeout", 600)
    max_concurrency = vllm_cfg.get("max_concurrency", 32)
    trace_path = vllm_cfg.get("trace_path")
    trace_format = vllm_cfg.get("trace_format", "sharegpt")
    max_requests = vllm_cfg.get("max_requests")
    output_dir = Path(vllm_cfg.get("output_dir", "vllm/output"))
    tick_seconds = vllm_cfg.get("tick_seconds", 0.5)
    tokenizer_id = vllm_cfg.get("tokenizer") or model

    if not model:
        print("vllm.model is required in config")
        return
    if not trace_path:
        print("vllm.trace_path is required in config")
        return

    requests = load_trace(trace_path, max_requests=max_requests, format=trace_format)
    total_requests = len(requests)
    print(f"Loaded {total_requests} requests from {trace_path} (format={trace_format})")

    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    print(f"Endpoint: {endpoint}")
    print(f"Model: {model}  Concurrency: {max_concurrency}")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    client = AsyncOpenAI(base_url=endpoint, api_key=api_key, timeout=timeout)

    metrics_url = f"{_metrics_base_url(endpoint)}/metrics"
    print(f"Metrics polling: {metrics_url}")

    request_records, metric_samples = asyncio.run(_send_all(
        requests, client, model, tokenizer, max_concurrency, trace_format,
        metrics_url, tick_seconds,
    ))

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_meta(output_dir, model, trace_path, started_at, finished_at, len(request_records))
    _write_requests(output_dir, request_records)
    _write_timeseries(output_dir, metric_samples, tick_seconds)

    print(f"Done → {output_dir}")

async def _send_all(
    requests,
    client,
    model,
    tokenizer,
    max_concurrency,
    trace_format,
    metrics_url,
    tick_seconds,
):
    semaphore = asyncio.Semaphore(max_concurrency)
    records: list[dict] = []
    metric_samples: list[_MetricsSample] = []

    if trace_format == "agentic":
        roots = [r for r in requests if r.sub_request_index == 0]
    else:
        roots = list(requests)

    roots.sort(key=lambda r: r.arrival_time)

    t_base = time.perf_counter()

    # Start metrics polling in background
    stop_poll = asyncio.Event()
    poll_task = asyncio.create_task(
        _poll_metrics(metrics_url, tick_seconds, metric_samples, stop_poll)
    )
    # Let the first poll happen before sending
    await asyncio.sleep(tick_seconds)

    async def _session_loop(root_req):
        current = root_req
        chain_arrival = root_req.arrival_time

        while current is not None:
            wait = chain_arrival - (time.perf_counter() - t_base)
            if wait > 0:
                await asyncio.sleep(wait)

            text = tokenizer.decode(current.prompt_token_ids, skip_special_tokens=True)
            messages = [{"role": "user", "content": text}]

            rec = await _send_one(current, client, model, messages, semaphore)

            rec["arrival_time"] = chain_arrival
            rec["input_toks"] = len(current.prompt_token_ids)
            records.append(rec)

            if current.next_sub_request is not None:
                await asyncio.sleep(current.tool_duration)
                chain_arrival = time.perf_counter() - t_base
                current = current.next_sub_request
            else:
                break

    tasks = [asyncio.create_task(_session_loop(r)) for r in roots]
    await asyncio.gather(*tasks)

    # One final poll, then stop
    stop_poll.set()
    await poll_task

    return records, metric_samples

async def _send_one(req, client, model, messages, semaphore):
    async with semaphore:
        t_start = time.perf_counter()
        first_ts = None
        n_out = req.max_output_len

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=n_out,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={
                    "min_tokens": n_out,
                    "ignore_eos": True,
                },
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_ts is None:
                        first_ts = time.perf_counter()

            t_end = time.perf_counter()

            return {
                "request_id": req.request_id,
                "status": "ok",
                "ttft": first_ts - t_start if first_ts is not None else None,
                "latency": t_end - t_start,
                "output_tokens": n_out,
            }

        except Exception as e:
            t_end = time.perf_counter()
            return {
                "request_id": req.request_id,
                "status": f"error: {e}",
                "ttft": None,
                "latency": t_end - t_start,
                "output_tokens": 0,
            }

def _write_meta(out, model, trace_path, started_at, finished_at, num_requests):
    payload = {
        "schema_version": 1,
        "model": model,
        "simulator": "foothold-vllm-send",
        "dataset_path": trace_path,
        "started_at": started_at,
        "finished_at": finished_at,
        "num_requests": num_requests,
    }
    (out / "meta.json").write_text(json.dumps(payload, indent=2))

def _write_requests(out, records):
    with (out / "requests.jsonl").open("w") as f:
        for r in records:
            arr = r.get("arrival_time", 0.0)
            ttft = r.get("ttft")
            latency = r.get("latency", 0.0)

            f.write(json.dumps({
                "request_id": r["request_id"],
                "input_toks": r.get("input_toks", 0),
                "output_toks": r.get("output_tokens", 0),
                "arrival_time": arr,
                "queued_ts": arr,
                "scheduled_ts": arr,
                "first_token_ts": arr + ttft if ttft is not None else None,
                "last_token_ts": arr + latency,
            }) + "\n")

def _write_timeseries(out, samples, tick_seconds):
    header = ["t", "prompt_throughput", "gen_throughput",
              "running", "waiting", "kv_cache_pct"]

    with (out / "timeseries.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        if len(samples) < 2:
            return

        prev = samples[0]
        w.writerow([
            prev.t, 0.0, 0.0,
            prev.num_running, prev.num_waiting,
            round(prev.kv_cache_pct, 3),
        ])

        for s in samples[1:]:
            dt = s.t - prev.t
            prompt_tput = max(0.0, (s.prompt_tokens_cumulative - prev.prompt_tokens_cumulative) / dt) if dt > 0 else 0.0
            gen_tput = max(0.0, (s.gen_tokens_cumulative - prev.gen_tokens_cumulative) / dt) if dt > 0 else 0.0

            w.writerow([
                s.t,
                round(prompt_tput, 1),
                round(gen_tput, 1),
                s.num_running,
                s.num_waiting,
                round(s.kv_cache_pct, 3),
            ])
            prev = s


# ── Embedded mode (in-process vLLM, no HTTP) ───────────────────────────


def run_send_embedded(config: dict) -> None:
    """Run benchmark by embedding vLLM's AsyncLLM directly in-process.

    Uses ``BenchStatLogger`` (a ``StatLoggerBase`` subclass) to capture
    per-iteration scheduler stats.  vLLM calls ``record()`` on every
    scheduling step with exact per-iteration token counts — no cumulative
    counter desync, no polling gaps.
    """
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from validate.stat_logger import BenchStatLogger

    vllm_cfg = config.get("vllm", {})
    engine_cfg = vllm_cfg.get("engine_args", {})

    model = vllm_cfg["model"]
    trace_path = vllm_cfg["trace_path"]
    trace_format = vllm_cfg.get("trace_format", "sharegpt")
    max_requests = vllm_cfg.get("max_requests")
    output_dir = Path(vllm_cfg["output_dir"])
    tick_seconds = vllm_cfg.get("tick_seconds", 0.5)

    requests = load_trace(trace_path, max_requests=max_requests, format=trace_format)
    print(f"Loaded {len(requests)} requests from {trace_path} (format={trace_format})")

    tp = engine_cfg.get("tensor_parallel_size", 1)
    print(f"Model: {model}  TP={tp}")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    engine_args = AsyncEngineArgs(
        model=model,
        tensor_parallel_size=tp,
        pipeline_parallel_size=engine_cfg.get("pipeline_parallel_size", 1),
        data_parallel_size=engine_cfg.get("data_parallel_size", 1),
        max_num_seqs=engine_cfg.get("max_num_seqs", 128),
        max_num_batched_tokens=engine_cfg.get("max_num_batched_tokens", 2048),
        max_model_len=engine_cfg.get("max_model_len"),
        dtype=engine_cfg.get("dtype", "bfloat16"),
        kv_cache_dtype=engine_cfg.get("kv_cache_dtype", "auto"),
        seed=engine_cfg.get("seed", 42),
        enable_prefix_caching=engine_cfg.get("enable_prefix_caching", True),
        load_format=engine_cfg.get("load_format", "auto"),
        enforce_eager=engine_cfg.get("enforce_eager", False),
        gpu_memory_utilization=engine_cfg.get("gpu_memory_utilization", 0.9),
    )

    BenchStatLogger.reset()
    print("Booting AsyncLLM (embedded mode)...")
    engine = AsyncLLM.from_engine_args(
        engine_args, stat_loggers=[BenchStatLogger],
    )

    records = asyncio.run(_send_all_embedded(
        engine, requests, trace_format,
        max_model_len=engine_args.max_model_len,
    ))

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_meta(output_dir, model, trace_path, started_at, finished_at, len(records))
    _write_requests_embedded(output_dir, records)
    header, rows = BenchStatLogger.downsample_to_csv_rows(tick_seconds)
    _write_timeseries_embedded(output_dir, header, rows)

    print("Shutting down...")
    engine.shutdown()
    print(f"Done -> {output_dir}")


async def _send_all_embedded(engine, requests, trace_format, *, max_model_len=None):
    """Schedule requests by arrival time, collect per-request metrics."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    records: list[dict] = []

    if trace_format == "agentic":
        roots = [r for r in requests if r.sub_request_index == 0]
    else:
        roots = list(requests)

    roots.sort(key=lambda r: r.arrival_time)

    loop = asyncio.get_event_loop()
    t0 = loop.time()

    # For agentic traces total counts all sub_requests so progress
    # (e.g. "5/39") is meaningful; sharegpt counts roots as before.
    total = len(requests) if trace_format == "agentic" else len(roots)
    counter = [0]

    async def _send_one(req):
        """Send a single request to the embedded engine; append to records."""
        i = counter[0]
        counter[0] += 1

        n_out = req.max_output_len
        tok_ids = list(req.prompt_token_ids)
        if max_model_len and len(tok_ids) + n_out > max_model_len:
            keep = max(1, max_model_len - n_out)
            tok_ids = tok_ids[-keep:]

        sp = SamplingParams(
            min_tokens=n_out,
            max_tokens=n_out,
            ignore_eos=True,
            temperature=0.0,
        )
        prompt = TokensPrompt(prompt_token_ids=tok_ids)
        request_id = f"bench-{i}"

        last_metrics = None
        async for output in engine.generate(prompt, sp, request_id):
            if output.metrics is not None:
                last_metrics = output.metrics

        if last_metrics is not None:
            # vLLM uses two different clocks:
            #   arrival_time  = time.time()       (Unix epoch)
            #   queued_ts, first_token_ts, ... = time.monotonic()
            # Normalize arrival_time into the monotonic domain so that
            # validate/plot.py compute_latencies() can subtract them.
            queued = getattr(last_metrics, "queued_ts", None)
            scheduled = getattr(last_metrics, "scheduled_ts", None)
            first = getattr(last_metrics, "first_token_ts", None)
            last = getattr(last_metrics, "last_token_ts", None)

            # Use queued_ts as the effective arrival_time — all monotonic.
            arrival_norm = queued

            records.append({
                "request_id": f"bench-{i}",
                "input_toks": len(req.prompt_token_ids),
                "output_toks": n_out,
                "arrival_time": arrival_norm,
                "queued_ts": queued,
                "scheduled_ts": scheduled,
                "first_token_ts": first,
                "last_token_ts": last,
            })
        else:
            records.append({
                "request_id": f"bench-{i}",
                "input_toks": len(req.prompt_token_ids),
                "output_toks": n_out,
                "arrival_time": None,
                "queued_ts": None,
                "scheduled_ts": None,
                "first_token_ts": None,
                "last_token_ts": None,
            })

        print(f"  {i + 1}/{total} requests done")

    async def _one(req):
        if trace_format == "agentic":
            # Walk the session chain: each sub_request follows the
            # previous after tool_duration.  Arrival_time only applies
            # to the root; chained sub_requests start immediately after
            # the previous finishes + its tool_duration.
            current = req
            first_arrival = req.arrival_time

            while current is not None:
                if current is req:
                    # Root: wait for arrival_time before dispatching
                    delay = (t0 + first_arrival) - loop.time()
                    if delay > 0:
                        await asyncio.sleep(delay)

                await _send_one(current)

                if current.next_sub_request is not None:
                    await asyncio.sleep(current.tool_duration)
                    current = current.next_sub_request
                else:
                    break
        else:
            # ShareGPT: single request, respect arrival_time
            delay = (t0 + req.arrival_time) - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

            await _send_one(req)

    tasks = [asyncio.create_task(_one(r)) for r in roots]
    await asyncio.gather(*tasks)
    return records


def _write_requests_embedded(out, records):
    """Write requests.jsonl with absolute timestamps from vLLM metrics."""
    with (out / "requests.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps({
                "request_id": r["request_id"],
                "input_toks": r.get("input_toks", 0),
                "output_toks": r.get("output_toks", 0),
                "arrival_time": r.get("arrival_time"),
                "queued_ts": r.get("queued_ts"),
                "scheduled_ts": r.get("scheduled_ts"),
                "first_token_ts": r.get("first_token_ts"),
                "last_token_ts": r.get("last_token_ts"),
            }) + "\n")


def _write_timeseries_embedded(out, header, rows):
    with (out / "timeseries.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
