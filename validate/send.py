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


def _parse_metrics(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for m in _METRIC_RE.finditer(text):
        result[m.group(1)] = float(m.group(2))
    return result


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
    async with httpx.AsyncClient(timeout=5.0) as client:
        while not stop.is_set():
            t = round(time.perf_counter() - t0, 3)
            try:
                resp = await client.get(metrics_url)
                m = _parse_metrics(resp.text)
                samples.append(_MetricsSample(
                    t=t,
                    num_running=int(m.get("num_requests_running", 0)),
                    num_waiting=int(m.get("num_requests_waiting", 0)),
                    prompt_tokens_cumulative=m.get("prompt_tokens_total", 0.0),
                    gen_tokens_cumulative=m.get("generation_tokens_total", 0.0),
                    kv_cache_pct=m.get("kv_cache_usage_perc", 0.0),
                ))
            except Exception:
                pass
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
    if len(samples) < 2:
        return

    header = ["t", "prompt_throughput", "gen_throughput",
              "running", "waiting", "kv_cache_pct"]

    with (out / "timeseries.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

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
