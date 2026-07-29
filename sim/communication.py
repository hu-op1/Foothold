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


def all_to_all_time(total_bytes: float, ep_size: int = 1,
                    comm_model: str = "lut",
                    intra_bw_gb_s: float = 9.7,
                    intra_latency_us: float = 2.0,
                    lut_bytes=None, lut_time_s=None) -> float:
    """Predicted all-to-all communication time for MoE token dispatch/combine.

    In EP, each GPU sends tokens to all other GPUs in the EP group.
    For uniform routing, each GPU sends total_bytes / ep_size bytes
    to each other GPU.  The total transfer for one GPU is
    total_bytes * (ep_size - 1) / ep_size (one hop each).

    When ep_size <= 1, returns 0.0.
    """
    if ep_size <= 1:
        return 0.0

    per_gpu_bytes = total_bytes * (ep_size - 1) / ep_size
    if per_gpu_bytes <= 0:
        return 0.0

    if comm_model == "bw_latency":
        bw_bytes_s = intra_bw_gb_s * 1e9
        lat_s = intra_latency_us * 1e-6
        return per_gpu_bytes / bw_bytes_s + lat_s

    if lut_bytes is None or lut_time_s is None or len(lut_bytes) == 0:
        return 0.0
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


def ring_all_reduce_time(per_hop_bytes: float, tp: int,
                         comm_model: str = "lut",
                         intra_bw_gb_s: float = 9.7,
                         intra_latency_us: float = 2.0,
                         lut_bytes=None, lut_time_s=None) -> float:
    """Predicted ring all-reduce time for one projection output.

    Ring all-reduce with *tp* GPUs:
      - reduce-scatter:  tp−1 hops
      - all-gather:      tp−1 hops
      - total:           2·(tp−1) hops
      - per-hop transfer: total_bytes / tp (= per_hop_bytes)

    Two models:

    ``"bw_latency"``
        Bandwidth + latency model:
        ``2·(tp−1) · (per_hop_bytes / BW + latency)``

    ``"lut"`` (default)
        Uses the measured D2H memcpy lookup table to model each hop.
        ``2·(tp−1) · lut_lookup(per_hop_bytes)``
    """
    if tp <= 1:
        return 0.0
    steps = 2 * (tp - 1)

    if comm_model == "bw_latency":
        bw_bytes_s = intra_bw_gb_s * 1e9
        lat_s = intra_latency_us * 1e-6
        return steps * (per_hop_bytes / bw_bytes_s + lat_s)

    # LUT model (default)
    t = memcpy_time(per_hop_bytes, lut_bytes, lut_time_s)
    return steps * t


def _transfer_time(total_bytes: float,
                   comm_model: str = "lut",
                   bw_gb_s: float = 9.7,
                   lut_bytes=None, lut_time_s=None) -> float:
    """Single-transfer time for *total_bytes* bytes.

    ``"bw_latency"``: total_bytes / (bw_gb_s * 1e9)
    ``"lut"``: interpolate from measured LUT
    """
    if total_bytes <= 0:
        return 0.0
    if comm_model == "bw_latency":
        return total_bytes / (bw_gb_s * 1e9)
    if lut_bytes is None or lut_time_s is None or len(lut_bytes) == 0:
        return 0.0
    return memcpy_time(total_bytes, lut_bytes, lut_time_s)


def transfer_blocks(request: Request, pool_d, bytes_per_block: int,
                    comm_model: str = "lut",
                    bw_gb_s: float = 9.7,
                    lut_bytes=None, lut_time_s=None,
                    block_size: int = 16) -> float:
    """Allocate D-side blocks for a request's KV cache and compute transfer time.

    Allocates blocks in pool_d for each block in the request's block_table.
    Cached blocks on D side are reused (touched).
    Updates request.block_table with D-side block IDs.

    Args:
        request: Request with populated block_table and block_hashes.
        pool_d: Decode-side BlockPool.
        bytes_per_block: Bytes per KV cache block.
        comm_model: "lut" or "bw_latency"
        bw_gb_s: bandwidth for transfer (GB/s), used when comm_model="bw_latency"
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
    return _transfer_time(total_bytes, comm_model=comm_model,
                          bw_gb_s=bw_gb_s,
                          lut_bytes=lut_bytes, lut_time_s=lut_time_s)
