---
description: "Use when writing or modifying tests in test/. Covers pytest conventions, fixtures, trace generation, and simulation test patterns."
applyTo: "test/**/*.py"
---

# test/ — Testing Conventions

## Running Tests

```bash
uv run python -m pytest test/                    # All tests
uv run python -m pytest test/test_sim.py -v    # Sim tests verbose
uv run python test/<script>.py                  # Standalone scripts (no pytest)
```

All tests require a CUDA GPU and benchmark/fit result data.

## Pytest Tests (`test_sim.py`)

### Fixtures

Three module-level fixtures, defined once and reused:

```python
@pytest.fixture
def model():
    from sim.config import load_model_spec
    return load_model_spec("meta-llama/Llama-2-7b-hf")

@pytest.fixture
def hw():
    import json
    with open("fit/results/3090.json") as f:
        return json.load(f)

@pytest.fixture
def config():
    from sim.config import load_config
    return load_config()
```

### Trace Generation

Use `_make_jsonl_trace()` helper for test traces:

```python
def _make_jsonl_trace(path, num=5, shared_prefix_ids=None):
    """Generate a local JSONL trace file for testing."""
```

Parameters:
- `num`: number of requests
- `shared_prefix_ids`: optional list of shared prefix token IDs (for caching tests)
- Arrival time spacing: `0.5s` between requests
- Timestamps in nanoseconds (`arrival_time_ns`)

### Test Patterns

Import modules with **lazy imports** inside test functions (some sim imports are expensive):

```python
def test_colocated_simulation(model, hw, config):
    from sim.engine import SimulationEngine  # lazy import
    engine = SimulationEngine(config, model, hw)
    metrics = engine.run(reqs, mode="colocated")
    assert metrics.num_requests == N
    assert metrics.throughput() > 0
```

### Assertion Patterns

- Positive throughput: `assert metrics.throughput() > 0`
- Positive latency: `assert metrics.p99_latency() > 0`
- Correct counts: `assert metrics.num_requests == N`
- Cache hits: second request should have lower TTFT with shared prefix
- Strategy search: `assert len(results) >= N`, `assert best["score"] > 0`

### Temp File Cleanup

Use `tempfile.NamedTemporaryFile(delete=False)` with `try/finally`:

```python
try:
    _make_jsonl_trace(tmp_path, num=10)
    # ... test ...
finally:
    os.unlink(tmp_path)
```

## Standalone Scripts

Scripts like `test/analyze.py`, `test/compare.py`, `test/gemm.py` are standalone (no pytest). Run directly with `uv run python test/<script>.py`. They have no fixed structure — inspect the file before editing.
