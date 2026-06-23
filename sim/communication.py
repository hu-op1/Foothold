"""KV cache transfer model for disaggregated prefill→decode communication."""

from sim.request import Request


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
                    bandwidth_gb_s: float, latency_us: float,
                    block_size: int = 16) -> float:
    """Allocate D-side blocks for a request's KV cache and compute transfer time.

    Allocates blocks in pool_d for each block in the request's block_table.
    Cached blocks on D side are reused (touched).
    Updates request.block_table with D-side block IDs.

    Args:
        request: Request with populated block_table and block_hashes.
        pool_d: Decode-side BlockPool.
        bytes_per_block: Bytes per KV cache block.
        bandwidth_gb_s: Inter-node bandwidth (GB/s).
        latency_us: Fixed per-transfer latency.
        block_size: Number of tokens per block.

    Returns:
        Transfer time in seconds for non-cached blocks.
    """
    blocks_to_send = 0
    new_table: list[int] = []

    for i, bid in enumerate(request.block_table):
        if bid < 0:
            new_table.append(-1)
            continue

        if i < len(request.block_hashes):
            bh = request.block_hashes[i]
            cached_id = pool_d.get_cached_block(bh)
            if cached_id is not None:
                pool_d.touch([cached_id])
                new_table.append(cached_id)
                continue

        # Allocate a fresh block on the D side
        block = pool_d.get_new_block()
        if block.is_null:
            # Rollback: free blocks already allocated in new_table
            pool_d.free_blocks([b for b in new_table if b >= 0])
            return float("inf")

        new_table.append(block.block_id)
        blocks_to_send += 1

        # Cache the block on D side if it covers a full prompt block
        if i < len(request.block_hashes):
            block_start = i * block_size
            block_end = min(block_start + block_size, request.num_prompt_tokens)
            if block_end - block_start == block_size:
                block.block_hash = request.block_hashes[i]
                pool_d._cache_block(block)

    # Replace request's block_table with D-side blocks
    request.block_table = new_table

    if blocks_to_send == 0:
        return 0.0

    total_bytes = blocks_to_send * bytes_per_block
    return total_bytes / (bandwidth_gb_s * 1e9) + latency_us * 1e-6
