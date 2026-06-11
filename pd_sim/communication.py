"""KV cache transfer model for disaggregated prefill→decode communication."""

from pd_sim.request import Request


DTYPE_BYTES = 2  # fp16


def kv_bytes_per_layer(kv_len, num_kv_heads, head_dim):
    """Bytes for one layer's KV cache at a given sequence length."""
    return 2 * kv_len * num_kv_heads * head_dim * DTYPE_BYTES


def raw_transfer_time(kv_len, num_layers, num_kv_heads, head_dim,
                      bandwidth_gb_s, latency_us):
    """Raw transfer time for all KV layers without overlap.

    Args:
        kv_len: sequence length of KV cache to transfer.
        num_layers: number of attention layers.
        num_kv_heads: KV heads (accounting for GQA).
        head_dim: dimension per head.
        bandwidth_gb_s: inter-node bandwidth in GB/s.
        latency_us: fixed per-transfer latency in microseconds.

    Returns:
        Transfer time in seconds.
    """
    per_layer = kv_bytes_per_layer(kv_len, num_kv_heads, head_dim)
    total_bytes = num_layers * per_layer
    return total_bytes / (bandwidth_gb_s * 1e9) + latency_us * 1e-6


def effective_xfer_overhead(kv_len, num_layers, num_kv_heads, head_dim,
                            bandwidth_gb_s, latency_us, prefill_step_time_s):
    """Effective overhead after overlap with prefill compute.

    KV saves start per-layer during prefill forward.
    Overlap ≈ prefill_time * (num_layers - 1) / num_layers
    """
    raw = raw_transfer_time(kv_len, num_layers, num_kv_heads, head_dim,
                            bandwidth_gb_s, latency_us)
    if prefill_step_time_s <= 0 or num_layers <= 1:
        return raw
    overlap = prefill_step_time_s * (num_layers - 1) / num_layers
    return max(0.0, raw - overlap)


def transfer_blocks(request: Request, pool_d, bytes_per_block: int,
                    bandwidth_gb_s: float, latency_us: float) -> float:
    """Compute transfer time for a request's KV blocks from P to D.

    Only blocks not already cached on D side need to be sent.
    Cached blocks on D side are touched (ref_cnt++).

    Args:
        request: Request with populated block_table and block_hashes.
        pool_d: Decode-side BlockPool.
        bytes_per_block: Bytes per KV cache block.
        bandwidth_gb_s: Inter-node bandwidth (GB/s).
        latency_us: Fixed per-transfer latency.

    Returns:
        Transfer time in seconds for non-cached blocks.
    """
    blocks_to_send = 0
    for i, bid in enumerate(request.block_table):
        if i < len(request.block_hashes):
            bh = request.block_hashes[i]
            cached_id = pool_d.get_cached_block(bh)
            if cached_id is not None:
                pool_d.touch([cached_id])
                continue
        blocks_to_send += 1

    if blocks_to_send == 0:
        return 0.0

    total_bytes = blocks_to_send * bytes_per_block
    return total_bytes / (bandwidth_gb_s * 1e9) + latency_us * 1e-6
