"""Trace loader — ShareGPT and Agentic JSONL formats."""

import json

from pd_sim.request import Request


def load_trace(path, max_requests=None, format="sharegpt"):
    """Load a JSONL request trace, normalizing to list[Request].

    Args:
        path: Path to JSONL trace file.
        max_requests: Cap on number of requests (sessions for agentic, lines for sharegpt).
        format: "sharegpt" (one line = one request) or "agentic" (one line = one session).

    Returns:
        list[Request] sorted by arrival_time.
    """
    if format == "agentic":
        return _load_agentic_trace(path, max_requests)
    return _load_sharegpt_trace(path, max_requests)


def _load_sharegpt_trace(path, max_requests=None):
    """Load ShareGPT-format trace: one request per line.

    Expected format per line:
      {"input_toks": int, "output_toks": int, "arrival_time_ns": int,
       "input_tok_ids": [...], "output_tok_ids": [...]}
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


def _load_agentic_trace(path, max_requests=None):
    """Load Agentic-format trace: one session per line with chained sub_requests.

    Expected format per line:
      {"session_id": ..., "arrival_time_ns": ..., "sub_requests": [
          {"input_toks": ..., "output_toks": ..., "tool_duration_ns": ...,
           "input_tok_ids": [...], "output_tok_ids": [...]},
          ...
      ]}

    Sub_requests within a session are causally dependent: sub_req N+1 cannot
    start until sub_req N finishes + tool_duration_ns.  This is modeled by
    linking them via Request.next_sub_request — the engine enqueues the next
    sub_request when the current one completes.

    max_requests applies to sessions (lines), not individual sub_requests.
    """
    requests = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_requests and i >= max_requests:
                break

            session = json.loads(line)
            sub_reqs = session.get("sub_requests", [])
            if not sub_reqs:
                continue

            session_id = str(session.get("session_id", f"session-{i}"))
            session_arrival = session.get("arrival_time_ns", 0) / 1e9

            prev_req = None
            for j, sub in enumerate(sub_reqs):
                req = Request(
                    request_id=f"{session_id}-sub{j}",
                    arrival_time=session_arrival if prev_req is None else 0.0,
                    prompt_token_ids=sub["input_tok_ids"],
                    max_output_len=sub["output_toks"],
                    session_id=session_id,
                    sub_request_index=j,
                    tool_duration=sub.get("tool_duration_ns", 0) / 1e9,
                )
                if prev_req is not None:
                    prev_req.next_sub_request = req
                prev_req = req
                requests.append(req)

    # Sort by arrival_time — only first sub_requests have real times;
    # chained sub_requests (time=0.0) sort early but are skipped at enqueue.
    requests.sort(key=lambda r: r.arrival_time)
    return requests
