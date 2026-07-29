"""Generate conversation trace JSONL from a HF dataset or local file.

Usage:
  uv run tools/generate_conversation_trace.py --dataset heiheiha798/sharegpt-regen-qwen3-8b-non-thinking --model Qwen/Qwen3-4B --sps 5 --num-reqs 20
  uv run tools/generate_conversation_trace.py --dataset path/to/file.jsonl --model ... --sps 5 --num-reqs 20
"""

import argparse, json, os, random, sys
from pathlib import Path

import tqdm
from transformers import AutoTokenizer


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


def parse(example):
    """Extract list of (role, content) from any common format."""
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
    """Turn messages into list of (input_ids, output_ids) with context."""
    msgs = [m for m in msgs if m[0] in ("user", "assistant")]
    if not msgs or msgs[0][0] != "user":
        return []
    ctx, out = [], []
    i = 0
    while i < len(msgs):
        if msgs[i][0] == "user":
            ctx.append({"role": "user", "content": msgs[i][1]})
            j = i + 1
            while j < len(msgs) and msgs[j][0] == "assistant":
                j += 1
            if j > i + 1:
                ids = tok.apply_chat_template(ctx, tokenize=True, add_generation_prompt=True)
                if hasattr(ids, "input_ids"):
                    ids = ids.input_ids
                    if hasattr(ids, "tolist"):
                        ids = ids.tolist()
                out_ids = tok.encode(msgs[i + 1][1])
                if hasattr(out_ids, "tolist"):
                    out_ids = out_ids.tolist()

                total = len(ids) + len(out_ids)
                if max_model_len > 0 and total > max_model_len:
                    break

                out.append((list(ids), out_ids))
                ctx.append({"role": "assistant", "content": msgs[i + 1][1]})
            i = j
        else:
            i += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
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

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ds = load_data(args.dataset, args.split, streaming=not args.no_stream)
    if args.no_stream:
        ds = ds.shuffle(seed=args.seed)
    else:
        ds = ds.shuffle(buffer_size=1000, seed=args.seed)

    random.seed(args.seed)
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