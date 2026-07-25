"""Send trace requests to vLLM via OpenAI-compatible API.

Decodes token IDs to text, sends requests to vLLM, and records timing.
Timing starts from the API call; tokenizer decode is not included.

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
import time
from pathlib import Path

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from sim.trace import load_trace


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
    output_dir = vllm_cfg.get("output_dir", "vllm/output")
    tick_seconds = vllm_cfg.get("tick_seconds", 0.5)
    tokenizer_id = vllm_cfg.get("tokenizer", model)

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
    request_records = asyncio.run(_send_all(
        requests, client, model, tokenizer, max_concurrency, trace_format,
    ))

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_meta(out, model, trace_path, started_at, finished_at, len(request_records))
    _write_requests(out, request_records)
    _write_timeseries(out, request_records, tick_seconds)

    print(f"Done → {out}")


async def _send_all(requests, client, model, tokenizer, max_concurrency, trace_format):
    semaphore = asyncio.Semaphore(max_concurrency)
    records: list[dict] = []

    if trace_format == "agentic":
        roots = [r for r in requests if r.sub_request_index == 0]
    else:
        roots = list(requests)

    roots.sort(key=lambda r: r.arrival_time)

    t_base = time.perf_counter()

    async def _session_loop(root_req):
        current = root_req
        chain_arrival = root_req.arrival_time
        decode_start = time.perf_counter() - t_base

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

    return records


async def _send_one(req, client, model, messages, semaphore):
    async with semaphore:
        t_start = time.perf_counter()
        first_ts = None
        output_tokens = 0

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=req.max_output_len,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_ts is None:
                        first_ts = time.perf_counter()
                    output_tokens += 1

            t_end = time.perf_counter()

            return {
                "request_id": req.request_id,
                "status": "ok",
                "ttft": first_ts - t_start if first_ts is not None else None,
                "latency": t_end - t_start,
                "output_tokens": output_tokens,
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


def _write_timeseries(out, records, tick_seconds):
    if not records:
        return

    max_t = max(
        r.get("arrival_time", 0.0) + r.get("latency", 0.0)
        for r in records
    )
    num_buckets = max(int(max_t / tick_seconds) + 1, 1)

    with (out / "timeseries.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "prompt_throughput", "gen_throughput",
                     "running", "waiting", "kv_cache_pct"])

        for b in range(num_buckets):
            t_start = b * tick_seconds
            t_end = (b + 1) * tick_seconds
            t_mid = round(t_end, 3)

            running = 0
            prompt_tokens = 0
            gen_tokens = 0

            for r in records:
                arr = r.get("arrival_time", 0.0)
                lat = r.get("latency", 0.0)
                finish = arr + lat

                if arr < t_end and finish >= t_end:
                    running += 1
                elif arr < t_end and finish < t_end and finish >= t_start:
                    running += 1

                if t_start <= arr < t_end:
                    prompt_tokens += r.get("input_toks", 0)

                if t_start <= finish < t_end:
                    gen_tokens += r.get("output_tokens", 0)

            w.writerow([
                t_mid,
                round(prompt_tokens / tick_seconds, 1),
                round(gen_tokens / tick_seconds, 1),
                running,
                0,  # waiting — not available from vLLM API
                0.0,  # kv_cache_pct — not available
            ])
