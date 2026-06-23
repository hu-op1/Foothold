# AGENTS.md

AI coding agent guidance for this repository. See [CLAUDE.md](CLAUDE.md) for full architecture and commands.

## Environment

- **Python ≥ 3.14** — modern features (walrus, match/case, `pathlib.Path`) are fine
- **Package manager**: `uv` only — never `pip install` or `python -m pip`
- **Package index**: `https://pypi.tuna.tsinghua.edu.cn/simple` (default). PyTorch from `https://download.pytorch.org/whl/cu128`
- **GPU required**: All operations need a CUDA-capable NVIDIA GPU. Benchmarks need ≥6 GB VRAM.

## Key conventions

- **`main.py` is the only CLI entry point** — subcommands: `bench`, `fit`, `search`, `sim`. Legacy `--bench`/`--fit`/`--search` flags also work
- Config files by stage: `config/default.yaml` (bench/fit), `config/predict.yaml`, `config/search.yaml`, `config/sim.yaml`
- Results saved as `.xlsx` via `openpyxl` (not CSV)
- Use `pathlib.Path` for filesystem paths (some legacy uses of `os.path` exist in `main.py` and `bench/`)
- Use `torch.cuda.Event` for GPU timing — see `CudaTimer`, `warmup()`, `benchmark()` in `bench/utils.py:8`
- Call `check_memory()` before allocating large tensors to avoid OOM
- Model specs auto-discovered from `models/<vendor>/<family>/<model>/config.json` (HF format). YAML fallback at `config/model_specs.yaml`

## External repos (standalone — not part of this project, do not modify or import)

`InferSim/`, `LLMServingSim/`, `SimAI/`, `apex_plus/`, `vllm-0.19.0/`

## Testing

Tests live in `test/` (gitignored — local-only). Run with:
```bash
uv run python -m pytest test/                # pytest tests (test_sim.py)
uv run python test/<script>.py               # standalone scripts (analyze.py, etc.)
```
All tests require a CUDA GPU and benchmark/fit result data.

## Design docs

`docs/architecture.md` — comprehensive Chinese-language architecture doc.
