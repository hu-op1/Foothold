"""KV cache transfer model for disaggregated prefill→decode communication."""

import numpy as np

from sim.request import Request


def _lut_lookup(bytes_val, lut_bytes, lut_time_s):
    """Linear interpolation on a sorted (bytes, time) lookup table."""
    if bytes_val <= lut_bytes[0]:
        return lut_time_s[0]
    if bytes_val >= lut_bytes[-1]:
        slope = ((lut_time_s[-1] - lut_time_s[-2])
                 / (lut_bytes[-1] - lut_bytes[-2]))
        extra = bytes_val - lut_bytes[-1]
        return lut_time_s[-1] + slope * extra
    idx = int(np.searchsorted(lut_bytes, bytes_val))
    if idx == 0:
        return lut_time_s[0]
    x0, x1 = lut_bytes[idx - 1], lut_bytes[idx]
    t0, t1 = lut_time_s[idx - 1], lut_time_s[idx]
    return t0 + (bytes_val - x0) * (t1 - t0) / (x1 - x0)


def all_to_all_time(total_bytes: float, lut_bytes, lut_time_s,
                    ep_size: int = 1) -> float:
    """Predicted all-to-all communication time for MoE token dispatch/combine.

    In EP, each GPU sends tokens to all other GPUs in the EP group.
    For uniform routing, each GPU sends total_bytes / ep_size bytes
    to each other GPU.  The total transfer for one GPU is
    total_bytes * (ep_size - 1) / ep_size (one hop each).

    When ep_size <= 1, returns 0.0.

    Uses the memcpy LUT as an approximation for GPU-to-GPU transfer.
    """
    if ep_size <= 1:
        return 0.0
    if lut_bytes is None or lut_time_s is None or len(lut_bytes) == 0:
        return 0.0

    per_gpu_bytes = total_bytes * (ep_size - 1) / ep_size

    if per_gpu_bytes <= 0:
        return 0.0

    import numpy as np
    return _lut_lookup(per_gpu_bytes,
                       np.asarray(lut_bytes, dtype=np.float64),
                       np.asarray(lut_time_s, dtype=np.float64))


def memcpy_time(bytes_val, lut_bytes, lut_time_s):
    """Predicted transfer time for *bytes_val* across a PCIe/NVLink link.

    Uses linear interpolation on the measured lookup table.
    Raises ValueError if LUT is not available.
    """
    if lut_bytes is None or lut_time_s is None or len(lut_bytes) == 0:
        raise ValueError("memcpy LUT not available — run --bench then --fit first")
    return _lut_lookup(bytes_val, np.asarray(lut_bytes, dtype=np.float64),
                       np.asarray(lut_time_s, dtype=np.float64))


def transfer_blocks(request: Request, pool_d, bytes_per_block: int,
                    lut_bytes, lut_time_s,
                    block_size: int = 16) -> float:
    """Allocate D-side blocks for a request's KV cache and compute transfer time.

    Allocates blocks in pool_d for each block in the request's block_table.
    Cached blocks on D side are reused (touched).
    Updates request.block_table with D-side block IDs.

    Args:
        request: Request with populated block_table and block_hashes.
        pool_d: Decode-side BlockPool.
        bytes_per_block: Bytes per KV cache block.
        lut_bytes: LUT byte-size array.
        lut_time_s: LUT transfer-time array.
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
                pool_d.cached_block_hash_to_block.insert(request.block_hashes[i], block)

    # Replace request's block_table with D-side blocks
    request.block_table = new_table

    if blocks_to_send == 0:
        return 0.0

    total_bytes = blocks_to_send * bytes_per_block
    return memcpy_time(total_bytes, lut_bytes, lut_time_s)
