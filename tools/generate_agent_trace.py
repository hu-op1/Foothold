"""Generate agentic-workload trace JSONL from real agent session traces.

Reads pi coding agent trace files from traces/DeepSeek-v4-Pro-Agent/ and produces
a single JSONL file where each line is one session with chained sub-requests.

Uses the specified model's HuggingFace tokenizer to generate real token IDs,
reconstructing the conversation text from the trace events.

Format per output line:
  {"session_id": "...", "arrival_time_ns": ..., "sub_requests": [
      {"input_toks": N, "output_toks": M, "tool_duration_ns": T,
       "input_tok_ids": [...], "output_tok_ids": [...]},
      ...,
      {"input_toks": N, "output_toks": M, "tool_duration_ns": 0,
       "input_tok_ids": [...], "output_tok_ids": [...]}
  ]}

Usage:
  uv run python tools/generate_agent_trace.py --model Qwen/Qwen3-8B --sps 0.05
  uv run python tools/generate_agent_trace.py --model Qwen/Qwen3-8B --sps 0.2 --max-sessions 500
  uv run python tools/generate_agent_trace.py --model Qwen/Qwen3-8B --sps 1.0 --output traces/my.jsonl
  uv run python tools/generate_agent_trace.py --model Qwen/Qwen3-8B --sps 0.1 --seed 123
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import tqdm

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
DEFAULT_TRACE_DIR = ROOT / "traces" / "DeepSeek-v4-Pro-Agent"
DEFAULT_OUTPUT = ROOT / "traces" / "agent_trace.jsonl"


# ── Helpers to reconstruct text from trace content blocks ──────────

def _get_text(content_blocks):
    """Join all thinking/text blocks into a single string."""
    parts = []
    for b in content_blocks:
        t = b.get("type", "")
        if t == "thinking":
            parts.append(b.get("thinking", ""))
        elif t == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def _get_tool_calls(content_blocks):
    """Extract tool calls in OpenAI-format dicts for chat-template use."""
    calls = []
    for b in content_blocks:
        if b.get("type") == "toolCall":
            tc = b.get("toolCall", {})
            calls.append({
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                },
            })
    return calls


def _build_output_text(content_blocks):
    """Build the full output text of an assistant message.

    Includes thinking, visible text, and tool-call formatting tokens.
    This text is fed to the tokenizer to get output_tok_ids.
    """
    parts = []
    for b in content_blocks:
        t = b.get("type", "")
        if t == "thinking":
            parts.append(b.get("thinking", ""))
        elif t == "text":
            parts.append(b.get("text", ""))
        elif t == "toolCall":
            tc = b.get("toolCall", {})
            # Format tool call as the model would generate it
            # Use <|tool_call|> marker (Qwen convention) — neutral, adds realistic tokens
            parts.append(
                f"<|tool_call|>\n"
                f'{{"name": "{tc.get("name", "")}", '
                f'"arguments": {json.dumps(tc.get("arguments", {}), ensure_ascii=False)}}}'
            )
    return "\n".join(parts)


# ── Tokenizer loading ─────────────────────────────────────────────

def load_tokenizer(model_name):
    """Load HuggingFace tokenizer with graceful error on missing deps."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "ERROR: 'transformers' not installed.\n"
            f"  Run: uv add transformers  (or pip install transformers)\n"
            "  Then retry."
        )
        raise SystemExit(1)

    print(f"Loading tokenizer: {model_name} ...", end=" ", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"vocab_size={tok.vocab_size}")
    return tok


# ── Session parsing ───────────────────────────────────────────────

def parse_session_with_tokenizer(filepath, tokenizer):
    """Parse one session, reconstruct conversation, tokenize with HF tokenizer.

    Returns dict with session_id, original_arrival_s, sub_requests.
    Each sub-request has real token IDs from the model's tokenizer.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
    except Exception:
        return None

    if not lines:
        return None

    # Session metadata
    session_evt = lines[0]
    if session_evt.get("type") != "session":
        return None
    session_id = session_evt["id"]
    ts_iso = session_evt["timestamp"]
    arr_dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    original_arrival_s = arr_dt.timestamp()

    # Walk events sequentially, building messages and sub-requests
    current_messages = []          # OpenAI-format message dicts
    sub_requests = []
    for evt in lines:
        if evt.get("type") != "message":
            continue
        msg = evt.get("message", {})
        role = msg.get("role")
        content_blocks = msg.get("content", [])

        if role == "developer":
            text = _get_text(content_blocks)
            current_messages.append({"role": "system", "content": text})

        elif role == "user":
            text = _get_text(content_blocks)
            current_messages.append({"role": "user", "content": text})

        elif role == "assistant":
            # --- INPUT: tokenize messages up to now ---
            encoded = tokenizer.apply_chat_template(
                current_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            # transformers 5.x may return BatchEncoding; normalize to list[int]
            if hasattr(encoded, "input_ids"):
                ids = encoded.input_ids
                input_ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            elif isinstance(encoded, (list, tuple)):
                input_ids = list(encoded)
            else:
                input_ids = list(encoded)

            # --- OUTPUT: tokenize the assistant's response ---
            output_text = _build_output_text(content_blocks)
            out_enc = tokenizer.encode(output_text)
            if hasattr(out_enc, "tolist"):
                output_ids = out_enc.tolist()
            elif isinstance(out_enc, (list, tuple)):
                output_ids = list(out_enc)
            else:
                output_ids = list(out_enc)

            # --- Add to current_messages for next turns ---
            text = _get_text(content_blocks)
            tool_calls = _get_tool_calls(content_blocks)
            asst_msg = {"role": "assistant", "content": text}
            if tool_calls:
                asst_msg["tool_calls"] = tool_calls
            current_messages.append(asst_msg)

            # --- Store sub-request (tool_duration filled later) ---
            sub_requests.append({
                "input_tok_ids": input_ids,
                "output_tok_ids": output_ids,
            })

        elif role == "toolResult":
            tool_call_id = msg.get("toolCallId", "")
            tool_text = _get_text(content_blocks)
            current_messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_text,
            })

    # Build final result with random tool durations (last = 0)
    result = []
    for i, sr in enumerate(sub_requests):
        is_last = (i == len(sub_requests) - 1)
        tool_dur = 0 if is_last else random.randint(0, 10_000_000_000)
        result.append({
            "input_toks": len(sr["input_tok_ids"]),
            "output_toks": len(sr["output_tok_ids"]),
            "tool_duration_ns": tool_dur,
            "input_tok_ids": sr["input_tok_ids"],
            "output_tok_ids": sr["output_tok_ids"],
        })

    if not result:
        return None

    return {
        "session_id": session_id,
        "original_arrival_s": original_arrival_s,
        "sub_requests": result,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate agentic trace from real agent sessions"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HF model name for tokenizer (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--sps", type=float, required=True,
                        help="Session arrival rate: sessions per second "
                             "(e.g. 0.05 = one every 20s)")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR),
                        help="Directory containing agent .jsonl session files")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output JSONL file path")
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="Cap number of sessions (0 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling session file selection")
    args = parser.parse_args()

    # ── Load tokenizer ──
    tokenizer = load_tokenizer(args.model)

    # ── Discover & shuffle trace files ──
    trace_dir = Path(args.trace_dir)
    if not trace_dir.is_dir():
        print(f"ERROR: trace directory not found: {trace_dir}")
        return 1

    all_files = sorted(trace_dir.glob("*.jsonl"))
    if not all_files:
        print(f"ERROR: no .jsonl files in {trace_dir}")
        return 1

    random.seed(args.seed)
    random.shuffle(all_files)
    if args.max_sessions and args.max_sessions < len(all_files):
        all_files = all_files[:args.max_sessions]
    files = all_files

    print(f"Processing {len(files)} session files from {trace_dir}")
    print(f"Arrival rate: {args.sps} sessions/s  "
          f"(interval: {1/args.sps:.1f}s per session)")

    # ── Parse sessions ──
    sessions = []
    for fp in tqdm.tqdm(files, desc="Parsing sessions"):
        s = parse_session_with_tokenizer(fp, tokenizer)
        if s and s["sub_requests"]:
            sessions.append(s)

    if not sessions:
        print("ERROR: no valid sessions found")
        return 1

    # Sort by original arrival time, then reassign session_id = 0,1,2,…
    sessions.sort(key=lambda s: s["original_arrival_s"])

    # Assign synthetic arrival times and sequential session ids
    interval_ns = 1.0 / args.sps * 1e9
    for i, s in enumerate(sessions):
        s["session_id"] = i
        s["arrival_time_ns"] = int(i * interval_ns)
        del s["original_arrival_s"]

    total_sessions = len(sessions)
    total_sub_requests = sum(len(s["sub_requests"]) for s in sessions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for s in tqdm.tqdm(sessions, desc="Writing output"):
            out.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nDone: {total_sessions} sessions, {total_sub_requests} sub-requests")
    print(f"Arrival range: {sessions[0]['arrival_time_ns']/1e9:.1f}s → "
          f"{sessions[-1]['arrival_time_ns']/1e9:.1f}s "
          f"(duration: {(sessions[-1]['arrival_time_ns'] - sessions[0]['arrival_time_ns'])/1e9:.1f}s)")
    print(f"Output: {output_path}")
    if output_path.exists():
        print(f"Size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    # Sample
    print("\n--- Sample output (first session) ---")
    first = sessions[0]
    print(f"session_id: {first['session_id']}")
    print(f"arrival_time_ns: {first['arrival_time_ns']} ({first['arrival_time_ns']/1e9:.3f}s)")
    print(f"sub_requests: {len(first['sub_requests'])}")
    for i, sr in enumerate(first["sub_requests"][:3]):
        print(f"  [{i}] input={sr['input_toks']} output={sr['output_toks']} "
              f"tool_dur={sr['tool_duration_ns']/1e9:.3f}s "
              f"tok_ids_in={len(sr['input_tok_ids'])} tok_ids_out={len(sr['output_tok_ids'])}")
    n = len(first["sub_requests"])
    if n > 3:
        sr_last = first["sub_requests"][-1]
        print(f"  [{n-1}] input={sr_last['input_toks']} output={sr_last['output_toks']} "
              f"tool_dur={sr_last['tool_duration_ns']} (last)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
