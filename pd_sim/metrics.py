"""Per-request metrics collection and aggregate statistics."""

import statistics


class MetricsCollector:
    def __init__(self):
        self.records: list[dict] = []

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

    def mean_ttft(self) -> float:
        vals = [r["ttft"] for r in self.records]
        return statistics.mean(vals) if vals else 0.0

    def mean_tpot(self) -> float:
        vals = [r["tpot"] for r in self.records if r["tpot"] > 0]
        return statistics.mean(vals) if vals else 0.0

    def p50_latency(self) -> float:
        return self._percentile("total_latency", 50)

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

    def slo_compliance(self, ttft_ms, tpot_ms, p99_ms) -> dict:
        """Return SLO compliance fraction and per-metric pass rates."""
        if not self.records:
            return {"score": 0.0, "ttft_pass": 0.0, "tpot_pass": 0.0, "p99_pass": False}

        ttft_pass = sum(1 for r in self.records if r["ttft"] * 1000 <= ttft_ms)
        tpot_pass = sum(1 for r in self.records if r["tpot"] * 1000 <= tpot_ms)
        p99_pass = self.p99_latency() * 1000 <= p99_ms

        n = len(self.records)
        compliance = (ttft_pass / n) * (tpot_pass / n) * (1.0 if p99_pass else 0.0)

        return {
            "score": compliance,
            "ttft_pass": ttft_pass / n,
            "tpot_pass": tpot_pass / n,
            "p99_pass": p99_pass,
        }

    def score(self, ttft_ms, tpot_ms, p99_ms) -> float:
        """Combined throughput × SLO compliance score."""
        slo = self.slo_compliance(ttft_ms, tpot_ms, p99_ms)
        return self.throughput() * slo["score"]
