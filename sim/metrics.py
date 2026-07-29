"""Per-request metrics collection and aggregate statistics."""

import statistics


class MetricsCollector:
    def __init__(self):
        self.records: list[dict] = []
        self.cache_hit_rate: float | None = None  # set by engine after run
        self.time_breakdown: dict[str, float] | None = None  # set by engine after run

    def record(self, request) -> None:
        """Record metrics for a finished request."""
        if request.finish_time is None or request.arrival_time is None:
            return

        ttft = request.ttft if request.ttft is not None else 0.0
        total_lat = request.finish_time - request.arrival_time
        num_out = max(request.num_output_tokens, 1)

        self.records.append({
            "request_id": request.request_id,
            "arrival_time": request.arrival_time,
            "first_token_time": request.arrival_time + ttft,
            "completion_time": request.finish_time,
            "ttft": ttft,
            "total_latency": total_lat,
            "tpot": (total_lat - ttft) / num_out,
            "num_output_tokens": request.num_output_tokens,
            "prompt_len": request.prompt_len,
        })

    # ── aggregated metrics ──────────────────────────────────────────

    @property
    def num_requests(self) -> int:
        return len(self.records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r["prompt_len"] for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r["num_output_tokens"] for r in self.records)

    @property
    def total_time(self) -> float:
        if not self.records:
            return 0.0
        return max(r["completion_time"] for r in self.records) - min(
            r["arrival_time"] for r in self.records
        )

    def throughput(self) -> float:
        """Output tokens per second."""
        tt = self.total_time
        return self.total_output_tokens / tt if tt > 0 else 0.0

    def input_throughput(self) -> float:
        """Input (prompt) tokens per second."""
        tt = self.total_time
        return self.total_input_tokens / tt if tt > 0 else 0.0

    def total_throughput(self) -> float:
        """Input + output tokens per second."""
        tt = self.total_time
        return (self.total_input_tokens + self.total_output_tokens) / tt if tt > 0 else 0.0

    # ── TTFT ──

    def mean_ttft(self) -> float:
        vals = [r["ttft"] for r in self.records]
        return statistics.mean(vals) if vals else 0.0

    def p50_ttft(self) -> float:
        return self._percentile("ttft", 50)

    def p90_ttft(self) -> float:
        return self._percentile("ttft", 90)

    def p99_ttft(self) -> float:
        return self._percentile("ttft", 99)

    # ── TPOT ──

    def mean_tpot(self) -> float:
        vals = [r["tpot"] for r in self.records if r["tpot"] > 0]
        return statistics.mean(vals) if vals else 0.0

    def p50_tpot(self) -> float:
        return self._percentile("tpot", 50)

    def p90_tpot(self) -> float:
        return self._percentile("tpot", 90)

    def p99_tpot(self) -> float:
        return self._percentile("tpot", 99)

    # ── total latency percentiles ──

    def p50_latency(self) -> float:
        return self._percentile("total_latency", 50)

    def p90_latency(self) -> float:
        return self._percentile("total_latency", 90)

    def p95_latency(self) -> float:
        return self._percentile("total_latency", 95)

    def p99_latency(self) -> float:
        return self._percentile("total_latency", 99)

    def _percentile(self, key: str, pct: float) -> float:
        vals = sorted(r[key] for r in self.records)
        if not vals:
            return 0.0
        idx = int(len(vals) * pct / 100)
        return vals[min(idx, len(vals) - 1)]

    def slo_compliance(self, ttft_ms, tpot_ms) -> dict:
        """Check if p90 TTFT and p90 TPOT meet SLO thresholds.

        SLO is a binary gate: both p90 TTFT and p90 TPOT must be ≤ their
        respective thresholds.  Per-request pass rates are not used — p90
        is the right aggregate because a few tail requests exceeding the
        threshold are acceptable; what matters is that 90% of users have
        a good experience.

        Total latency is intentionally NOT an SLO metric — it scales
        linearly with output token count.
        """
        if not self.records:
            return {"p90_ttft_ms": 0.0, "p90_tpot_ms": 0.0, "slo_pass": False}

        p90_ttft = self.p90_ttft() * 1000
        p90_tpot = self.p90_tpot() * 1000
        slo_pass = p90_ttft <= ttft_ms and p90_tpot <= tpot_ms

        return {
            "p90_ttft_ms": p90_ttft,
            "p90_tpot_ms": p90_tpot,
            "slo_pass": slo_pass,
        }

    def time_breakdown_pct(self) -> dict[str, float]:
        """Return each time component as % of total accumulated time."""
        if not self.time_breakdown:
            return {}
        total = sum(self.time_breakdown.values())
        if total == 0:
            return {}
        return {f"{k}_pct": v / total * 100 for k, v in self.time_breakdown.items()}


# ── Raw-op-name → CSV category name aggregation ───────────────────────
# graph.evaluate() produces raw op names (qkv_proj, o_proj, …) but the
# CSV expects aggregated category names (attn_proj, ffn_proj, …).
# aggregate_breakdown() maps raw → category before percentages are computed.

_AGGREGATION_MAP: dict[str, set[str]] = {
    "attn_proj": {"qkv_proj", "o_proj", "qk_proj", "v_proj",
                  "out_proj", "attn_gate_proj", "attn_gate_mul"},
    "ffn_proj": {"gate_up_proj", "down_proj"},
    "attn_prefill": {"flash_attn", "linear_scan"},
    "attn_decode": set(),
    "fused_add_norm": {"fused_residual_norm", "rmsnorm", "residual_add"},
    "rope": {"rope_q", "rope_kv"},
    "swiglu": {"swiglu"},
    "lm_head": {"lm_head"},
    "all_reduce": {"all_reduce"},
    "all_to_all": {"all_to_all"},
    "expert_proj": {"expert_gate_up", "expert_down"},
    "router_proj": {"router"},
    "kv_transfer": {"kv_transfer"},
    "swap": {"swap"},
}


def aggregate_breakdown(raw_acc: dict[str, float]) -> dict[str, float]:
    """Map raw op names from graph evaluation to the category names the
    CSV expects.  Any raw key not listed in the mapping passes through
    unchanged, so existing direct matches (swiglu, lm_head, …) are fine."""
    aggregated: dict[str, float] = {}
    unmapped = set(raw_acc)
    for cat, raw_names in _AGGREGATION_MAP.items():
        total = sum(raw_acc.get(k, 0.0) for k in raw_names)
        if total > 0:
            aggregated[cat] = total
        unmapped -= raw_names
    for k in unmapped:
        aggregated[k] = raw_acc[k]
    return aggregated
