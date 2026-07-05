# AGENTS.md

AI coding agent guidance for this repository. See [CLAUDE.md](CLAUDE.md) for full architecture and commands.

## Environment

- **Python ≥ 3.14** — modern features (walrus, match/case, `pathlib.Path`) are fine
- **Package manager**: `uv` only — never `pip install` or `python -m pip`
- **Package index**: `https://pypi.tuna.tsinghua.edu.cn/simple` (default). PyTorch from `https://download.pytorch.org/whl/cu128`
- **GPU required**: All operations need a CUDA-capable NVIDIA GPU. Benchmarks need ≥6 GB VRAM.

## Key conventions

- **`main.py` is the only CLI entry point** — subcommands: `bench`, `fit`, `search`, `sim`, `validate`
- Config files by stage: `config/bench.yaml` (bench/fit), `config/search.yaml` (search), `config/sim.yaml` (sim)
- Results saved as CSV (`bench/results/<gpu>/`, `sim/output/`) or JSON (`fit/results/<gpu>.json`)
- Use `pathlib.Path` for filesystem paths (some legacy uses of `os.path` exist in `main.py` and `bench/`)
- Use `torch.cuda.Event` for GPU timing — see `CudaTimer`, `warmup()`, `benchmark()` in `bench/utils.py:8`
- Call `check_memory()` before allocating large tensors to avoid OOM
- Model specs loaded from **HuggingFace Hub** via `transformers.AutoConfig.from_pretrained()` — no local `config.json` files needed. Any HF model ID works in YAML configs.
- Trace format: `sharegpt` (one JSONL line per request) or `agentic` (session chains with sub-requests, `tool_duration`)

## External repos (standalone — not part of this project, do not modify or import)

`InferSim/`, `LLMServingSim/`, `SimAI/`, `apex_plus/`, `vllm-0.19.0/`

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
| `.github/instructions/bench.instructions.md` | `bench/**/*.py` | CUDA timing, checkpoint/resume, OOM handling, benchmark conventions |
| `.github/instructions/fit.instructions.md` | `fit/**/*.py` | Roofline fitting, matmul prefill/decode split, elementwise proxy map |
| `.github/instructions/sim.instructions.md` | `sim/**/*.py` | Event-driven engine, scheduler, executor, memory pool, request lifecycle |
| `.github/instructions/config.instructions.md` | `config/*.yaml` | Config schemas, model spec loading, common pitfalls |
| `.github/instructions/test.instructions.md` | `test/**/*.py` | Pytest conventions, fixtures, trace generation, sim test patterns |

## Design docs

- `docs/vllm-simulator-gaps.md` — vLLM v0.19.0 vs simulator gap analysis (CUDA Graph, kernel launch overhead, async scheduling, etc.)
- `docs/accuracy-improvements.md` — Simulation accuracy improvement tracking & regression log
