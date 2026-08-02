---
description: "Use when working on the validate module. Covers sim visualization, vLLM trace send (API + embedded), sim-vs-vLLM comparison, and the validate.yaml schema."
applyTo: "validate/**/*.py"
---

# validate/ — Sim Output Validation

Entry point: `uv run python main.py --validate [--vllm] [--compare]`, dispatched by `validate/cli.py:dispatch()`.

## Modes

| Flags | Behavior |
|-------|----------|
| `--validate` | Visualize sim output (CDF plots, throughput) |
| `--validate --vllm` | Send trace to vLLM, produce same output format as sim |
| `--validate --compare` | Compare foothold sim vs LLMServingSim vs vLLM |

`--vllm` and `--compare` require `--validate`.

## plot.py

- Loads sim output via `load_requests`, `load_timeseries`, `compute_latencies`.
- Plots: `plot_throughput`, `plot_requests`, `plot_latency_cdfs`.
- `write_summary` writes Mean/Median/P90/P95/P99 + relative error + curve-area comparison.
- Contains **LLMServingSim-specific parsers**: `load_sim_log` (regex over `[Ns]` log lines), `load_sim_csv`, `sim_latencies`.
- Colors come from the `colors:` section of `config/validate.yaml`.

## send.py — vLLM trace send

Two modes:

- **API mode**: OpenAI-compatible endpoint (`vllm.endpoint`), polls `/metrics` in the background (`_poll_metrics`; `_SUM_METRICS`/`_MAX_METRICS` cross-engine aggregation).
- **Embedded mode** (`vllm.embedded: true`): spawns vLLM `AsyncLLM` in-process with a custom `StatLoggerBase` for per-iteration stats — accurate, no polling gaps. Requires dedicated GPUs (cannot coexist with a running vLLM server).

Output matches sim format: `meta.json`, `requests.jsonl`, `timeseries.csv`. Agentic session chains handled by `_send_all` → `_session_loop`.

## stat_logger.py

`BenchStatLogger(StatLoggerBase)` — records per-iteration exact token counts (no polling gaps), `downsample_to_csv_rows(tick_seconds)` buckets into timeseries rows. One instance per DP engine; samples share a class-level list.

## compare.py

`run_compare(config, vllm_dir_override=)`: requires foothold sim output; optionally LLMServingSim (`sim_csv` + `sim_log`) and vLLM (`vllm_dir`). Writes comparison charts + `summary.txt`.

## config/validate.yaml schema

Key sections (see `config/validate.yaml` for the full example):
- `sim_dir` (required) — sim output dir.
- Comparison sources: `sim_csv`, `sim_log`, `vllm_dir`, `compare_dir` (null excludes that source).
- `colors` — hex per source (foothold / llmservingsim / vllm).
- `title`, `prefix` — plot title suffix / output filename prefix.
- `vllm:` block — `embedded`, `endpoint`, `model`, `api_key`, `timeout`, `max_concurrency`, `trace_path`, `trace_format` ("sharegpt" | "agentic"), `max_requests`, `tokenizer`, `output_dir`, `tick_seconds`, and `engine_args` (used only when `embedded: true`).
