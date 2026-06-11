"""Trace loaders: ShareGPT, custom JSON, synthetic generation."""

import json
import random
from pathlib import Path

from pd_sim.request import Request


def load_trace(path, fmt="sharegpt", max_requests=None, model_name=None):
    """Load a request trace, normalizing to list[Request].

    Args:
        path: Path to trace file.
        fmt: "sharegpt" | "custom" | "synthetic"
        max_requests: Cap on number of requests (None = all).
        model_name: Model name for tokenizer selection (ShareGPT).

    Returns:
        list[Request] sorted by arrival_time.
    """
    if fmt == "sharegpt":
        requests = _load_sharegpt(path, max_requests)
    elif fmt == "custom":
        requests = _load_custom(path, max_requests)
    elif fmt == "synthetic":
        requests = _load_synthetic(path, max_requests)
    else:
        raise ValueError(f"Unknown trace format: {fmt}. Use sharegpt, custom, or synthetic.")
    requests.sort(key=lambda r: r.arrival_time)
    return requests


def _load_sharegpt(path, max_requests):
    """Load ShareGPT-format JSON trace.

    Each entry: {"conversations": [{"value": "..."}, ...]}
    Uses character-length heuristic for tokenization (chars / 3.5 ≈ tokens for English).
    Last turn is the assistant response → max_output_len.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    requests = []
    for i, entry in enumerate(data):
        if max_requests and i >= max_requests:
            break

        conversations = entry.get("conversations", [])
        if not conversations:
            continue

        # Build full prompt from all turns except last assistant response
        prompt_parts = []
        output_len = 0
        for j, turn in enumerate(conversations):
            value = turn.get("value", "")
            if turn.get("from") == "gpt" or turn.get("role") == "assistant":
                if j == len(conversations) - 1:
                    output_len = max(int(len(value) / 3.5), 1)
                else:
                    prompt_parts.append(value)
            else:
                prompt_parts.append(value)

        prompt_text = "".join(prompt_parts)
        prompt_len = max(int(len(prompt_text) / 3.5), 1)

        # Generate placeholder token IDs for prefix cache simulation
        token_ids = [hash(f"{i}-{t}") % 50000 for t in range(prompt_len)]

        req = Request(
            request_id=f"sharegpt-{i}",
            arrival_time=0.0 if "timestamp" not in entry else entry["timestamp"],
            prompt_token_ids=token_ids,
            max_output_len=output_len,
        )
        requests.append(req)

    return requests


def _load_custom(path, max_requests):
    """Load custom JSON trace format.

    Expected format: [{"prompt_len": int, "output_len": int, "timestamp": float}, ...]
    Or with explicit token IDs: [{"prompt_token_ids": [...], "output_len": int, ...}]
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    requests = []
    for i, entry in enumerate(data):
        if max_requests and i >= max_requests:
            break

        if "prompt_token_ids" in entry:
            token_ids = entry["prompt_token_ids"]
        else:
            prompt_len = entry.get("prompt_len", entry.get("input_len", 100))
            token_ids = [hash(f"{i}-{t}") % 50000 for t in range(prompt_len)]

        req = Request(
            request_id=entry.get("request_id", f"custom-{i}"),
            arrival_time=entry.get("timestamp", entry.get("arrival_time", 0.0)),
            prompt_token_ids=token_ids,
            max_output_len=entry.get("output_len", entry.get("max_output_len", 256)),
            priority=entry.get("priority", 0),
        )
        requests.append(req)

    return requests


def _load_synthetic(path, max_requests):
    """Load or generate synthetic trace from JSON config.

    Format: {"num_requests": 1000, "arrival_rate": 4.0,
             "prompt_len_mean": 1024, "prompt_len_std": 512,
             "output_len_mean": 256, "output_len_std": 128,
             "shared_prefix_len": 0, "vocab_size": 50000}
    """
    with open(path, encoding="utf-8") as f:
        params = json.load(f)

    num = min(params.get("num_requests", 1000), max_requests or 1000000)
    arrival_rate = params.get("arrival_rate", 4.0)
    pl_mean = params.get("prompt_len_mean", 1024)
    pl_std = params.get("prompt_len_std", 512)
    ol_mean = params.get("output_len_mean", 256)
    ol_std = params.get("output_len_std", 128)
    shared_prefix = params.get("shared_prefix_len", 0)
    vocab_size = params.get("vocab_size", 50000)
    seed = params.get("seed", 42)

    rng = random.Random(seed)
    shared_ids = [rng.randint(0, vocab_size - 1) for _ in range(shared_prefix)]

    requests = []
    clock = 0.0
    for i in range(num):
        prompt_len = max(1, int(rng.gauss(pl_mean, pl_std)))
        output_len = max(1, int(rng.gauss(ol_mean, ol_std)))

        unique_len = max(0, prompt_len - shared_prefix)
        unique_ids = [rng.randint(0, vocab_size - 1) for _ in range(unique_len)]
        token_ids = shared_ids + unique_ids

        req = Request(
            request_id=f"synth-{i}",
            arrival_time=clock,
            prompt_token_ids=token_ids[:prompt_len],
            max_output_len=output_len,
        )
        requests.append(req)

        # Poisson arrival interval
        clock += rng.expovariate(arrival_rate)

    return requests
