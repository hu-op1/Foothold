"""Block-level KV cache with prefix caching — mirrors vllm BlockPool."""

import hashlib
from dataclasses import dataclass


DTYPE_BYTES = 2  # fp16


@dataclass
class KVCacheBlock:
    block_id: int
    block_hash: bytes | None = None
    ref_cnt: int = 0
    is_null: bool = False
    last_accessed: float = 0.0

    def reset_hash(self):
        self.block_hash = None


def hash_block(token_ids: list[int]) -> bytes:
    """Hash a block of token IDs for prefix cache lookup."""
    return hashlib.sha256(bytes(str(token_ids), "utf-8")).digest()


def compute_block_hashes(token_ids: list[int], block_size: int) -> list[bytes]:
    """Compute block hashes for a request's prompt token IDs."""
    hashes = []
    for i in range(0, len(token_ids), block_size):
        chunk = token_ids[i:i + block_size]
        hashes.append(hash_block(chunk))
    return hashes


class BlockPool:
    """Manages KVCacheBlock allocation with prefix caching via block hash lookup.

    Supports GPU↔CPU swap for disaggregated D-side preemption:
    when the block pool is exhausted, running requests can be swapped out
    to CPU memory (keeping their computed tokens) and swapped back in
    later when space frees up, avoiding recomputation.
    """

    def __init__(self, num_blocks: int, enable_caching: bool = True,
                 cpu_swap_bw_bytes_per_s: float | None = None):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.num_blocks = num_blocks
        self.enable_caching = enable_caching

        # GPU↔CPU swap bandwidth (bytes/s).  None disables swap.
        self.cpu_swap_bw = cpu_swap_bw_bytes_per_s

        # Swapped-out requests: request_id → num_blocks
        self._swapped_out: dict[str, int] = {}

        # Prefix cache statistics
        self._cache_queries: int = 0
        self._cache_hits: int = 0

        # All blocks. Block 0 is the null block (placeholder).
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_blocks)
        ]
        self.blocks[0].is_null = True
        self.blocks[0].ref_cnt = 1  # never freed

        # Free blocks in LRU order — head pops first (next to allocate).
        self.free_block_ids: list[int] = list(range(1, num_blocks))

        # Block hash → block_id for prefix cache lookup.
        self.cached_block_hash_to_block: dict[bytes, int] = {}

        self.clock: float = 0.0

    # ── query ──────────────────────────────────────────────────────────

    def get_num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def get_usage(self) -> float:
        total = self.num_blocks - 1  # exclude null block
        return 1.0 - (self.get_num_free_blocks() / total) if total > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Prefix cache hit rate (0–1). Returns 0 if caching disabled or no queries."""
        if self._cache_queries == 0:
            return 0.0
        return self._cache_hits / self._cache_queries

    # ── prefix cache ───────────────────────────────────────────────────

    def get_cached_block(self, block_hash: bytes) -> int | None:
        """Return block_id if block_hash is cached, else None."""
        if not self.enable_caching:
            return None
        return self.cached_block_hash_to_block.get(block_hash)

    def get_computed_blocks(self, block_hashes: list[bytes]) -> tuple[list[int], int]:
        """Return (cached_block_ids, num_computed_tokens) for longest prefix match.

        Walks block_hashes from start, returns consecutive cache hits.
        Stops at first miss. num_computed_tokens is in cache-hit blocks (not tokens).
        """
        cached: list[int] = []
        for bh in block_hashes:
            block_id = self.get_cached_block(bh)
            if block_id is None:
                break
            cached.append(block_id)
        if self.enable_caching:
            self._cache_queries += len(block_hashes)
            self._cache_hits += len(cached)
        return cached, len(cached)

    def _cache_block(self, block: KVCacheBlock) -> None:
        """Register a full block in the hash cache."""
        if not self.enable_caching or block.block_hash is None:
            return
        self.cached_block_hash_to_block[block.block_hash] = block.block_id

    def _evict_block(self, block: KVCacheBlock) -> None:
        """Remove a block from the hash cache."""
        if block.block_hash is not None:
            self.cached_block_hash_to_block.pop(block.block_hash, None)
            block.reset_hash()

    # ── allocation ─────────────────────────────────────────────────────

    def touch(self, block_ids: list[int]) -> None:
        """Increment ref_cnt for cached blocks being reused.

        If ref_cnt was 0, the block should be in the free queue — remove it.
        Guard against double-touch (ref_cnt already > 0, or already removed).
        """
        for bid in block_ids:
            block = self.blocks[bid]
            if block.ref_cnt == 0 and not block.is_null:
                if bid in self.free_block_ids:
                    self.free_block_ids.remove(bid)
            block.ref_cnt += 1
            block.last_accessed = self.clock

    def get_new_block(self) -> KVCacheBlock:
        """Allocate one block from the free pool. Evicts cached block if needed.

        Returns null_block (block 0) only as fallback — callers should check is_null.
        """
        if self.free_block_ids:
            bid = self.free_block_ids.pop(0)
            block = self.blocks[bid]
            self._evict_block(block)
            block.ref_cnt += 1
            block.last_accessed = self.clock
            return block
        return self.blocks[0]  # null block — signals OOM

    def allocate_slots(
        self, request: "Request", num_new_tokens: int, block_size: int
    ) -> list[int] | None:
        """Allocate blocks for num_new_tokens beyond request.num_computed_tokens.

        Returns list of newly allocated block_ids in order,
        or None if allocation fails (OOM).
        """
        from sim.request import Request

        new_block_ids: list[int] = []
        new_start = request.num_computed_tokens
        new_end = new_start + num_new_tokens

        first_block_idx = new_start // block_size
        last_block_idx = (new_end - 1) // block_size

        for bi in range(first_block_idx, last_block_idx + 1):
            block_start = bi * block_size
            block_end = min(block_start + block_size, request.num_prompt_tokens)

            # Reuse existing block at this position if already owned
            already_owned = (
                bi < len(request.block_table)
                and request.block_table[bi] >= 0
            )
            if already_owned:
                new_block_ids.append(request.block_table[bi])
                continue

            if block_end - block_start == block_size and bi < len(request.block_hashes):
                # Full block — check prefix cache
                bh = request.block_hashes[bi]
                cached_id = self.get_cached_block(bh)
                if self.enable_caching:
                    self._cache_queries += 1
                if cached_id is not None:
                    if self.enable_caching:
                        self._cache_hits += 1
                    self.touch([cached_id])
                    new_block_ids.append(cached_id)
                    continue

            # Need a fresh block
            block = self.get_new_block()
            if block.is_null:
                # Allocation failed — rollback
                self.free_blocks(new_block_ids)
                return None
            new_block_ids.append(block.block_id)

            # If this block is now full, cache it
            if block_end - block_start == block_size and bi < len(request.block_hashes):
                block.block_hash = request.block_hashes[bi]
                self._cache_block(block)

        # Extend request's block table, filling gaps with empty slots.
        # Free any old block at each position before overwriting.
        for i, bid in enumerate(new_block_ids):
            idx = first_block_idx + i
            while len(request.block_table) <= idx:
                request.block_table.append(-1)
            old_bid = request.block_table[idx]
            if old_bid >= 0 and old_bid != bid:
                self.free_blocks([old_bid])
            request.block_table[idx] = bid

        return new_block_ids

    def free_blocks(self, block_ids: list[int]) -> None:
        """Decrement ref_cnt for blocks. Move to free queue if ref_cnt hits 0."""
        for bid in block_ids:
            if bid < 0:
                continue
            block = self.blocks[bid]
            if block.is_null:
                continue
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self.free_block_ids.append(bid)

    def free_request(self, request: "Request") -> None:
        """Free all blocks allocated to a request."""
        self.free_blocks([b for b in request.block_table if b >= 0])
        request.block_table.clear()

    # ── GPU ↔ CPU swap ────────────────────────────────────────────────

    def is_swapped(self, request_id: str) -> bool:
        """Return True if the request's blocks are currently swapped out."""
        return request_id in self._swapped_out

    def swap_out(self, request: "Request", block_size: int) -> float:
        """Swap a request's KV cache blocks to CPU.

        Frees physical blocks and records the count for later swap-in.
        Returns the swap-out time in seconds, or 0 if swap is disabled.
        """
        if self.cpu_swap_bw is None:
            return 0.0

        blocks = [b for b in request.block_table if b >= 0]
        if not blocks:
            return 0.0

        total_bytes = len(blocks) * self._block_bytes

        # Save for swap-in
        self._swapped_out[request.request_id] = len(blocks)

        # Free physical blocks
        self.free_blocks(blocks)
        request.block_table.clear()

        return total_bytes / self.cpu_swap_bw

    def swap_in(self, request: "Request", block_size: int) -> float:
        """Restore a request's KV cache blocks from CPU.

        Allocates fresh blocks; the data transfer is modelled as time only.
        Returns the swap-in time in seconds, or float('inf') if allocation
        fails (not enough free blocks).
        """
        if self.cpu_swap_bw is None:
            return 0.0

        num_blocks = self._swapped_out.get(request.request_id)
        if num_blocks is None:
            return 0.0

        total_bytes = num_blocks * self._block_bytes
        swap_time = total_bytes / self.cpu_swap_bw

        # Allocate fresh blocks
        new_blocks: list[int] = []
        for _ in range(num_blocks):
            block = self.get_new_block()
            if block.is_null:
                # Rollback
                self.free_blocks(new_blocks)
                return float("inf")
            new_blocks.append(block.block_id)

        del self._swapped_out[request.request_id]
        request.block_table = new_blocks
        return swap_time

    @property
    def _block_bytes(self) -> int:
        """Bytes per block. Set externally by the engine after construction."""
        return getattr(self, "bytes_per_block", 8_388_608)  # 8 MiB default
