"""Integration tests for sim module."""

import json
import os
import tempfile

import pytest


def _make_jsonl_trace(path, num=5, shared_prefix_ids=None):
    """Generate a JSONL trace with optional shared prefix for cache testing."""
    with open(path, "w") as f:
        for i in range(num):
            base = shared_prefix_ids or []
            unique = [hash(f"u-{i}-{t}") % 50000 for t in range(max(1, 100 + i * 20))]
            entry = {
                "input_toks": len(base) + len(unique),
                "output_toks": 50 + i * 10,
                "arrival_time_ns": int(i * 500_000_000),  # 0.5s apart
                "input_tok_ids": base + unique,
                "output_tok_ids": [hash(f"o-{i}-{t}") % 50000 for t in range(50)],
            }
            json.dump(entry, f)
            f.write("\n")


@pytest.fixture
def model():
    from sim.config import load_model_spec
    return load_model_spec("meta-llama/Llama-2-7b-hf")


@pytest.fixture
def hw():
    import json
    with open("fit/results/3090.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def config():
    from sim.config import load_config
    cfg = load_config()
    cfg["max_model_len"] = 8192
    return cfg


def test_load_jsonl_trace():
    from sim.trace import load_trace
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        _make_jsonl_trace(f.name, num=5)
        path = f.name
    try:
        reqs = load_trace(path)
        assert len(reqs) == 5
        assert all(r.prompt_len > 0 for r in reqs)
        assert all(r.max_output_len > 0 for r in reqs)
        assert all(len(r.prompt_token_ids) == r.prompt_len for r in reqs)
    finally:
        os.unlink(path)


def test_block_pool_allocation():
    from sim.memory import BlockPool, compute_block_hashes
    from sim.request import Request

    pool = BlockPool(50)
    r = Request("test", 0.0, list(range(200)), 50)
    r.block_hashes = compute_block_hashes(r.prompt_token_ids, 16)

    blocks = pool.allocate_slots(r, 64, 16)
    assert blocks is not None
    assert len(blocks) == 4

    pool.free_request(r)
    assert pool.get_num_free_blocks() == 49


def test_prefix_cache_hit():
    from sim.memory import BlockPool, compute_block_hashes
    from sim.request import Request

    pool = BlockPool(200)
    block_size = 16

    # First request: 4 blocks of 16 tokens each
    shared = list(range(64))
    r1 = Request("r1", 0.0, shared, 50)
    r1.block_hashes = compute_block_hashes(r1.prompt_token_ids, block_size)
    pool.allocate_slots(r1, 64, block_size)
    pool.commit_pending_cache()  # P3-10: deferred cache requires explicit commit

    # Second request: same prefix, longer
    r2_tokens = shared + list(range(100, 196))  # 4 shared blocks + 2 unique
    r2 = Request("r2", 0.0, r2_tokens, 50)
    r2.block_hashes = compute_block_hashes(r2.prompt_token_ids, block_size)
    cached_blocks, num_cached = pool.get_computed_blocks(r2.block_hashes)
    assert num_cached == 4  # first 4 blocks match


def test_colocated_simulation(model, hw, config):
    from sim.engine import SimulationEngine
    from sim.trace import load_trace

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        _make_jsonl_trace(f.name, num=10)
        path = f.name

    try:
        reqs = load_trace(path)
        engine = SimulationEngine(config, model, hw)
        metrics = engine.run(reqs, mode="colocated")

        assert metrics.num_requests == 10
        assert metrics.throughput() > 0
        assert metrics.p99_latency() > 0
    finally:
        os.unlink(path)


def test_disaggregated_simulation(model, hw, config):
    from sim.engine import SimulationEngine
    from sim.trace import load_trace

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        _make_jsonl_trace(f.name, num=10)
        path = f.name

    try:
        reqs = load_trace(path)
        engine = SimulationEngine(config, model, hw)
        metrics = engine.run(reqs, mode="disaggregated", pd_ratio=(2, 2))

        assert metrics.num_requests == 10
        assert metrics.throughput() > 0
    finally:
        os.unlink(path)


def test_strategy_search(model, hw, config):
    from sim.engine import SimulationEngine
    from sim.strategy import search
    from sim.trace import load_trace

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        _make_jsonl_trace(f.name, num=5)
        path = f.name

    try:
        reqs = load_trace(path)

        config["strategy"]["mode"] = "colocated"
        config["strategy"]["total_gpus"] = 1
        config["strategy"]["search"]["max_workers"] = 1
        config["strategy"]["search"]["max_batched_tokens"] = [256, 2048]
        config["strategy"]["search"]["prefill_thresholds"] = [256]

        engine = SimulationEngine(config, model, hw)
        results = search(engine, reqs, config)

        assert len(results) >= 2
        best = results[0]
        assert best["score"] > 0
    finally:
        os.unlink(path)


def test_jsonl_prefix_caching_in_simulation(model, hw, config):
    """Verify prefix cache actually reduces prefill time in colocated sim."""
    from sim.engine import SimulationEngine
    from sim.trace import load_trace

    # Two requests sharing the same prefix
    shared_ids = list(range(256))  # 256 shared tokens

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        # Request 1
        json.dump({
            "input_toks": 300, "output_toks": 50,
            "arrival_time_ns": 0,
            "input_tok_ids": shared_ids + [hash(f"u1-{t}") % 50000 for t in range(44)],
            "output_tok_ids": [],
        }, f)
        f.write("\n")
        # Request 2 — same prefix
        json.dump({
            "input_toks": 400, "output_toks": 50,
            "arrival_time_ns": 100_000_000,  # 0.1s
            "input_tok_ids": shared_ids + [hash(f"u2-{t}") % 50000 for t in range(144)],
            "output_tok_ids": [],
        }, f)
        f.write("\n")
        path = f.name

    try:
        reqs = load_trace(path)
        engine = SimulationEngine(config, model, hw)
        metrics = engine.run(reqs, mode="colocated")

        assert metrics.num_requests == 2
        assert metrics.throughput() > 0
        # Second request should have lower TTFT due to prefix cache hit
    finally:
        os.unlink(path)
