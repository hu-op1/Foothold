---
description: "Use when adding or modifying trace generation tools in tools/. Covers the generator, its CLI args, output schema, and arrival-rate modeling."
applyTo: "tools/**/*.py"
---

# tools/ — Trace Generation

One standalone script converts real data into JSONL traces that `sim/trace.py` loads. Run with `uv run python tools/<script>.py`.

## generate_conversation_trace.py (agentic)

Reads an HF dataset ID, a local JSONL/parquet directory, or the local agent-session trace dir.

| Arg | Meaning |
|-----|---------|
| `--dataset` | default `reference/DeepSeek-v4-Pro-Agent` — local agent-session trace dir (one JSONL per session, trace-event format, read fully offline); HF ids and local JSONL/parquet dirs still supported |
| `--model` | required |
| `--sps` | required — sessions/sec |
| `--num-reqs` | default 0 = all |
| `--seed` | default 42 |
| `--thinking-time` | default 5 — max thinking sec (tool_duration cap) |
| `--max-kv-toks` | default 40960 — input+output token cap |
| `--max-model-len` | default 0 — truncate conversation when exceeded |
| `--output` | default `""` |
| `--split` | default `train` |
| `--no-stream` | disable streaming |

- Arrival times use a **Poisson process** (`expovariate`).
- Trace-event sessions (developer/user/assistant/toolResult) are converted to messages; assistant chains (incl. tool-call markers) become growing-context sub-requests.
- Parses `messages` / `conversations` / `instruction-output` formats for other datasets.

## Output schema (shared)

One JSONL line per session:

```json
{
  "session_id": 0,
  "arrival_time_ns": 123456789,
  "sub_requests": [
    {"input_toks": 128, "output_toks": 64, "tool_duration_ns": 0,
     "input_tok_ids": [...], "output_tok_ids": [...]}
  ]
}
```

Loaded by `sim/trace.py` with `format="agentic"` (session chains) or `format="sharegpt"` (one request per line). Pre-generated traces live in `traces/`.
