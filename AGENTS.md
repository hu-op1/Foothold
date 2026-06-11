# AGENTS.md

AI coding agent guidance for this repository. See [CLAUDE.md](CLAUDE.md) for full architecture and commands.

## Environment

- **Python ≥ 3.14** — many modern features assumed (e.g., `pathlib.Path`, walrus operator, match/case are fine; don't use older compatibility patterns)
- **Package manager**: `uv` only — never use `pip install` or `python -m pip`
- **Package index**: Tsinghua mirror (`https://pypi.tuna.tsinghua.edu.cn/simple`). PyTorch wheels from `https://download.pytorch.org/whl/cu128`
- **GPU required**: All benchmarks need a CUDA-capable NVIDIA GPU with ≥6 GB VRAM. Never try to run benchmarks on a CPU-only machine

## Key conventions

- All modules are library-first — `main.py` is the only CLI entry point
- Use `pathlib.Path` for all filesystem paths (not `os.path`)
- Use `torch.cuda.Event` for GPU timing (see `CudaTimer` in `bench/utils.py`)
- Memory guard: always call `check_memory()` before allocating large tensors to avoid OOM
- Warmup before timing: `warmup(fn, iters)` before `benchmark(fn, iters)` — both in [bench/utils.py](bench/utils.py)
- Results are always saved as `.xlsx` via `openpyxl` (not CSV)

## External repos

`InferSim/`, `LLMServingSim/`, `SimAI/`, `apex_plus/` are standalone reference projects — **not** part of this codebase. Do not modify them and do not expect their imports to work in this project.

## Design docs

- [docs/superpowers/plans/](docs/superpowers/plans/) — implementation plans for past refactors
- [docs/superpowers/specs/](docs/superpowers/specs/) — detailed design specs for past refactors
- [docs/simulators_comparison.md](docs/simulators_comparison.md) — comparison of external simulators

## Testing

Test scripts live in `test/` — they are standalone scripts, not a pytest suite. Run them with `uv run python test/<name>.py`. They rely on `results/` data and a CUDA GPU.
