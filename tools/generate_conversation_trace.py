"""Generate conversation trace JSONL from a HF dataset or local file.

Usage:
  uv run tools/generate_conversation_trace.py --model Qwen/Qwen3-4B --sps 5 --num-reqs 20
  uv run tools/generate_conversation_trace.py --dataset heiheiha798/sharegpt-regen-qwen3-8b-non-thinking --model Qwen/Qwen3-4B --sps 5 --num-reqs 20
  uv run tools/generate_conversation_trace.py --dataset path/to/file.jsonl --model ... --sps 5 --num-reqs 20

Default dataset is the local agent-session trace dir reference/DeepSeek-v4-Pro-Agent
(one JSONL per session, trace-event format) — no HuggingFace network access needed.
The model tokenizer is still loaded from HuggingFace via --model.
"""

import argparse, json, os, random
from pathlib import Path

import tqdm
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATASET = str(ROOT / "reference" / "DeepSeek-v4-Pro-Agent")


def load_data(path, split="train", streaming=True):
    """Load from HF id or local JSONL/parquet dir."""
    if os.path.isdir(path) or os.path.isfile(path):
        p = Path(path)
        if p.is_dir():
            files = list(p.glob("*.jsonl")) + list(p.glob("*.json"))
            fmt = "json" if files else "parquet"
        else:
            fmt, p = "json", p.parent
        from datasets import load_dataset
        return load_dataset(fmt, data_dir=str(p), split=split, streaming=streaming)
    from datasets import load_dataset
    return load_dataset(path, split=split, streaming=streaming)


# ── Local agent-session trace dir (DeepSeek-v4-Pro-Agent) ─────────

def is_trace_dir(path):
    """True if `path` is a dir/file of trace-event JSONL (one session per line, "type" key)."""
    if not (os.path.isdir(path) or os.path.isfile(path)):
        return False
    p = Path(path)
    files = list(p.glob("*.jsonl")) if p.is_dir() else [p]
    if not files:
        return False
    try:
        with open(files[0], encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        return isinstance(first, dict) and "type" in first
    except (OSError, json.JSONDecodeError):
        return False


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


def _tool_call_dict(block):
    tc = block.get("toolCall")
    if isinstance(tc, dict):
        return tc
    return {"name": block.get("name", ""), "arguments": block.get("arguments", {})}


def _tool_call_text(tc):
    args = tc.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            args = {"raw": args}
    return f'<|tool_call|>\n{{"name": "{tc.get("name", "")}", "arguments": {json.dumps(args, ensure_ascii=False)}}}'


def _assistant_content(content_blocks):
    """Full assistant text: thinking + visible text + tool-call markers."""
    parts = []
    for b in content_blocks:
        t = b.get("type", "")
        if t == "thinking":
            parts.append(b.get("thinking", ""))
        elif t == "text":
            parts.append(b.get("text", ""))
        elif t == "toolCall":
            parts.append(_tool_call_text(_tool_call_dict(b)))
    return "\n".join(parts)


def trace_events_to_messages(events):
    """Convert one session's trace events into a list of (role, content) messages."""
    msgs = []
    for evt in events:
        if evt.get("type") != "message":
            continue
        msg = evt.get("message", {})
        role = msg.get("role")
        blocks = msg.get("content", [])
        if role == "developer":
            msgs.append(("system", _get_text(blocks)))
        elif role == "user":
            msgs.append(("user", _get_text(blocks)))
        elif role == "assistant":
            msgs.append(("assistant", _assistant_content(blocks)))
        elif role == "toolResult":
            msgs.append(("tool", _get_text(blocks)))
    if not any(r in ("user", "assistant") for r, _ in msgs):
        return []
    return msgs


def iter_trace_sessions(directory):
    """Yield parsed (role, content) message lists, one per session file, no network."""
    files = sorted(Path(directory).glob("*.jsonl"))
    for f in files:
        events = []
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            continue
        msgs = trace_events_to_messages(events)
        if msgs:
            yield msgs


def parse(example):
    """Extract list of (role, content) from a trace-session message list or any common dict format."""
    if isinstance(example, list):
        return example
    for key in ("messages", "conversations"):
        if key in example and isinstance(example[key], list):
            msgs = example[key]
            if msgs and isinstance(msgs[0], dict):
                first = msgs[0]
                if "role" in first and "content" in first:
                    return [(m["role"], m["content"]) for m in msgs]
                rm = {"human": "user", "gpt": "assistant", "assistant": "assistant",
                      "user": "user", "system": "system"}
                return [(rm.get(m.get("from", "").lower(), m.get("from", "")),
                         m.get("value", "")) for m in msgs]
    for u, a in [("instruction", "output"), ("prompt", "response"),
                  ("question", "answer"), ("input", "output")]:
        if u in example and a in example:
            return [("user", example[u]), ("assistant", example[a])]
    return None


def make_turns(msgs, tok, max_model_len=0):
    """Turn messages into list of (input_ids, output_ids) with growing context.

    Every assistant message in the chain yields one sub-request whose input is
    the full accumulated user/assistant history (tool messages are skipped, so
    this also handles agent-style user → assistant → tool → assistant … chains).
    """
    msgs = [m for m in msgs if m[0] in ("user", "assistant")]
    if not msgs or msgs[0][0] != "user":
        return []
    ctx, out = [], []
    for role, content in msgs:
        if role == "user":
            ctx.append({"role": "user", "content": content})
            continue
        ids = tok.apply_chat_template(ctx, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
        out_ids = tok.encode(content)
        if hasattr(out_ids, "tolist"):
            out_ids = out_ids.tolist()

        total = len(ids) + len(out_ids)
        if max_model_len > 0 and total > max_model_len:
            break

        out.append((list(ids), out_ids))
        ctx.append({"role": "assistant", "content": content})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="HF dataset id, local JSONL/parquet dir, or local agent-session trace dir "
                        f"(default: {DEFAULT_DATASET})")
    p.add_argument("--model", required=True)
    p.add_argument("--sps", type=float, required=True, help="sessions/sec")
    p.add_argument("--num-reqs", type=int, default=0, help="num sessions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thinking-time", type=float, default=5, help="max thinking sec")
    p.add_argument("--max-kv-toks", type=int, default=40960, help="max input+output tokens")
    p.add_argument("--max-model-len", type=int, default=0, help="max total seq len per turn; truncate conversation when exceeded")
    p.add_argument("--output", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--no-stream", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if is_trace_dir(args.dataset):
        print(f"Using local trace dir: {args.dataset}")
        ds = list(iter_trace_sessions(args.dataset))
        random.shuffle(ds)
    else:
        ds = load_data(args.dataset, args.split, streaming=not args.no_stream)
        if args.no_stream:
            ds = ds.shuffle(seed=args.seed)
        else:
            ds = ds.shuffle(buffer_size=1000, seed=args.seed)

    sessions, seen = [], 0

    for ex in tqdm.tqdm(ds, desc="Parsing"):
        if args.num_reqs and len(sessions) >= args.num_reqs:
            break
        seen += 1
        msgs = parse(ex)
        if not msgs:
            continue
        trs = make_turns(msgs, tok, max_model_len=args.max_model_len)
        if not trs:
            continue
        if args.max_kv_toks:
            last = trs[-1]
            if len(last[0]) + len(last[1]) > args.max_kv_toks:
                continue
        sessions.append(trs)

    if not sessions:
        print("No valid conversations found.")
        return 1

    # Poisson arrival
    t_ns = 0.0
    for s in sessions:
        s.insert(0, {"arrival": int(round(t_ns))})
        t_ns += random.expovariate(args.sps) * 1e9 if args.sps > 0 else 0

    sessions.sort(key=lambda s: s[0]["arrival"])

    out_path = Path(args.output) if args.output else Path(
        f"traces/conversation-{args.model.split('/')[-1]}-n{len(sessions)}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sid, s in enumerate(sessions):
            sub = []
            for i, (inp, out) in enumerate(s[1:]):
                last = (i == len(s) - 2)
                dur = 0 if last else (
                    random.randint(0, int(max(1, args.thinking_time * 1e9))) if args.thinking_time > 0 else 0)
                sub.append({"input_toks": len(inp), "output_toks": len(out),
                            "tool_duration_ns": dur,
                            "input_tok_ids": inp, "output_tok_ids": out})
            obj = {"session_id": sid, "arrival_time_ns": s[0]["arrival"],
                   "sub_requests": sub}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    n_sub = sum(len(s) - 1 for s in sessions)
    print(f"\nDone: {written} sessions, {n_sub} sub-requests")
    print(f"Output: {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    raise SystemExit(main())