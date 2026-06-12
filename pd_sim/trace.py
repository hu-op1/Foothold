"""Trace loader — JSONL format only."""

import json

from pd_sim.request import Request


def load_trace(path, max_requests=None):
    """Load a JSONL request trace, normalizing to list[Request].

    Expected format per line:
      {"input_toks": int, "output_toks": int, "arrival_time_ns": int,
       "input_tok_ids": [...], "output_tok_ids": [...]}

    Args:
        path: Path to JSONL trace file.
        max_requests: Cap on number of requests (None = all).

    Returns:
        list[Request] sorted by arrival_time.
    """
    requests = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_requests and i >= max_requests:
                break

            entry = json.loads(line)

            token_ids = entry["input_tok_ids"]
            output_len = entry["output_toks"]

            if "arrival_time_ns" in entry:
                arr = entry["arrival_time_ns"] / 1e9
            elif "timestamp" in entry:
                arr = entry["timestamp"]
            else:
                arr = entry.get("arrival_time", 0.0)

            req = Request(
                request_id=entry.get("request_id", f"req-{i}"),
                arrival_time=arr,
                prompt_token_ids=token_ids,
                max_output_len=output_len,
            )
            requests.append(req)

    requests.sort(key=lambda r: r.arrival_time)
    return requests
