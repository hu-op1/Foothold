---
description: "Use when adding or modifying trace generation tools in tools/. Covers the two generators, their CLI args, output schema, and arrival-rate modeling."
applyTo: "tools/**/*.py"
---

# tools/ — Trace Generation

Two standalone scripts convert real data into JSONL traces that `sim/trace.py` loads. Run with `uv run python tools/<script>.py`.

## generate_agent_trace.py (agentic format)

Streams the HF dataset `ansulev/DeepSeek-v4-Pro-Agent` and rebuilds token IDs with the model's tokenizer (`apply_chat_template`).

| Arg | Meaning |
|-----|---------|
| `--model` | required — HF model ID for the tokenizer |
| `--sps` | required — sessions/sec arrival rate |
| `--output` | default `traces/agent_trace.jsonl` |
| `--max-sessions` | 0 = all |
| `--seed` | default 42 |

- Arrival times synthesized with **uniform spacing** `1/sps`.
- Last sub-request of each session has `tool_duration_ns=0`; others random 0–10 s.
- `session_id` renumbered 0,1,2…; sessions sorted by original arrival time.

## generate_conversation_trace.py (sharegpt/agentic)

Reads an HF dataset ID or a local JSONL/parquet directory.

| Arg | Meaning |
|-----|---------|
| `--dataset` | required |
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

- Arrival times use a **Poisson process** (`expovariate`) — differs from the agent generator's uniform spacing.
- Parses `messages` / `conversations` / `instruction-output` formats.
- Single-turn conversations are split into multiple sub-requests with growing context.

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
