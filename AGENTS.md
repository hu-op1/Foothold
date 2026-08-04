# AGENTS.md

AI coding agent guidance for this repository. See [CLAUDE.md](CLAUDE.md) for full architecture and commands.

## Environment

- **Python ≥ 3.12** — modern features (walrus, match/case, `pathlib.Path`) are fine
- **Package manager**: `uv` only — never `pip install` or `python -m pip`
- **Package index**: `https://pypi.tuna.tsinghua.edu.cn/simple` (default). PyTorch from `https://download.pytorch.org/whl/cu128`
- **GPU required**: All operations need a CUDA-capable NVIDIA GPU. Benchmarks need ≥6 GB VRAM.

## Key conventions

- **`main.py` is the only CLI entry point** — subcommands: `bench`, `fit`, `search`, `sim`, `validate`
- Config files by stage: `config/bench.yaml` (bench/fit), `config/search.yaml` (search), `config/sim.yaml` (sim), `config/validate.yaml` (validate)
- Results saved as CSV (`bench/results/<gpu>/`, `sim/output/`) or JSON (`fit/results/<gpu>.json`)
- Use `pathlib.Path` for filesystem paths (some legacy uses of `os.path` exist in `main.py` and `bench/`)
- Use `torch.cuda.Event` for GPU timing — see `CudaTimer`, `warmup()`, `benchmark()` in `bench/utils.py:8`
- Call `check_memory()` before allocating large tensors to avoid OOM
- Model specs defined in `models/*.py` — each file exports a `SPEC` dict + `build_graph(spec)`, loaded via `sim/config.py:load_model_spec()`/`load_model_graph()`. YAML `model:` values (full HF IDs, short names, or paths) resolve to these files; no HuggingFace Hub or local `config.json` needed.
- `sim/` uses a computation-graph executor: `sim/graph.py` (`OpSpec`, `ModelGraph`, TP/EP/PP transforms) + `sim/layers/` builders; per-model graphs come from `models/*.py`
- Trace format: `sharegpt` (one JSONL line per request, optional `input_tok_ids`/`output_tok_ids`) or `agentic` (session chains with sub-requests, `tool_duration`)
- Trace generation tools in `tools/`: `generate_conversation_trace.py` (agentic format, default dataset = local `reference/DeepSeek-v4-Pro-Agent` trace dir; tokenizer still from HF)

## External repos (standalone — not part of this project, do not modify or import)

`LLMServingSim/`, `SimAI/`, `astra-sim/`, `DistServe/`, `Frontier/`, `vidur/`, `apex_plus/`, `vllm-0.19.0/`, `PDD/`

## Testing

Tests live in `test/` (gitignored — local-only). Run with:
```bash
uv run python -m pytest test/                # pytest tests (test_sim.py)
uv run python test/<script>.py               # standalone scripts (analyze.py, etc.)
```
All tests require a CUDA GPU and benchmark/fit result data.

## Module-specific instructions

Granular instruction files live in `.github/instructions/` — auto-attached when editing files in the matching module:

| File | Applies To | What It Covers |
|------|-----------|----------------|
| `.github/instructions/bench.instructions.md` | `bench/**/*.py` | CUDA timing, checkpoint/resume, OOM handling, benchmark conventions (incl. memcpy, gateddelta) |
| `.github/instructions/fit.instructions.md` | `fit/**/*.py` | Roofline fitting, matmul prefill/decode split, elementwise proxy map, memcpy LUT, gateddelta `ls_*` params |
| `.github/instructions/sim.instructions.md` | `sim/**/*.py` | Event-driven engine, scheduler, computation graph, executor, memory pool, request lifecycle |
| `.github/instructions/config.instructions.md` | `config/*.yaml` | Config schemas (bench/search/sim/validate), model spec resolution, common pitfalls (communication via memcpy LUT) |
| `.github/instructions/test.instructions.md` | `test/**/*.py` | Pytest conventions, fixtures, trace generation, sim test patterns |
| `.github/instructions/models.instructions.md` | `models/*.py` | `SPEC` dict + `build_graph(spec)` contract, model name resolution, hybrid architectures |
| `.github/instructions/tools.instructions.md` | `tools/**/*.py` | Trace generators (agentic/sharegpt), CLI args, arrival-rate modeling |
| `.github/instructions/validate.instructions.md` | `validate/**/*.py` | Visualization, vLLM send (API + embedded), sim-vs-vLLM comparison, validate.yaml schema |

## Design docs

- `docs/vllm-simulator-gaps.md` — vLLM v0.19.0 vs simulator gap analysis (CUDA Graph, kernel launch overhead, async scheduling, etc.)
- `docs/accuracy-improvements.md` — Simulation accuracy improvement tracking & regression log
- `docs/foothold-competitive-analysis.md` — Full feature comparison matrix against 7 competing simulators
