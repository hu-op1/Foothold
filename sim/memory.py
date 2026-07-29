"""Block-level KV cache with prefix caching — mirrors vllm BlockPool.

Data structures aligned with vLLM 0.19.0:

- Chain hashing (parent_block_hash dependency) prevents false prefix matches.
- FreeKVCacheBlockQueue (doubly linked list) provides O(1) LRU eviction.
- BlockHashToBlockMap allows multiple physical blocks per hash (no dedup).
"""

import hashlib
from dataclasses import dataclass

from sim.communication import memcpy_time


# ── Chain-hash sentinel ──────────────────────────────────────────────────
# Root of the hash chain for the first block in a sequence.
NONE_HASH: bytes = hashlib.sha256(b"vllm-none-hash").digest()


def hash_block_tokens(parent_block_hash: bytes | None,
                       curr_block_token_ids: list[int]) -> bytes:
    """Chain hash: depends on parent block so each block ties to its prefix.

    Matches vLLM 0.19.0's ``hash_block_tokens()`` in ``kv_cache_utils.py``.
    Simplified — no extra_keys (simulator does not model LoRA/multimodal).
    """
    if parent_block_hash is None:
        parent_block_hash = NONE_HASH
    # Encode token IDs as big-endian 4-byte integers for deterministic hashing.
    token_bytes = b"".join(t.to_bytes(4, "big", signed=False)
                           for t in curr_block_token_ids)
    return hashlib.sha256(parent_block_hash + token_bytes).digest()


def compute_block_hashes(token_ids: list[int], block_size: int) -> list[bytes]:
    """Compute block hashes using chain hashing (vLLM 0.19.0 style)."""
    hashes: list[bytes] = []
    parent_hash: bytes | None = None
    for i in range(0, len(token_ids), block_size):
        chunk = token_ids[i:i + block_size]
        bh = hash_block_tokens(parent_hash, chunk)
        hashes.append(bh)
        parent_hash = bh
    return hashes


def extend_block_hashes_from_output(request: "Request", block_size: int) -> None:
    """Recompute block_hashes from input + generated output tokens.

    Called during decode (from ``update_from_output``) after new output tokens
    have been generated.  Recomputing from scratch ensures:
    - Correct chain hashes for output token blocks
    - Correct hash update when a prefill partial block becomes full
    """
    total = request.num_computed_tokens
    if total == 0:
        return
    tokens = request.prompt_token_ids + request.output_tok_ids[:request.num_output_tokens]
    actual = tokens[:total]
    new_hashes = compute_block_hashes(actual, block_size)

    old_len = len(request.block_hashes)
    if len(new_hashes) <= old_len:
        # Only the last block changed (partial → full). Update in-place.
        for i in range(len(new_hashes)):
            if i < old_len and request.block_hashes[i] != new_hashes[i]:
                request.block_hashes[i] = new_hashes[i]
    else:
        request.block_hashes = new_hashes


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class KVCacheBlock:
    """A single KV cache block.

    prev_free_block / next_free_block are *only* valid when the block is
    in the free queue (FreeKVCacheBlockQueue); they are None otherwise.
    """
    block_id: int
    block_hash: bytes | None = None
    ref_cnt: int = 0
    is_null: bool = False

    # Doubly linked list pointers for FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    def reset_hash(self):
        self.block_hash = None


class FreeKVCacheBlockQueue:
    """Doubly linked list of free KVCacheBlocks (vLLM 0.19.0).

    Uses prev_free_block / next_free_block on each block object.
    Fake head/tail sentinel blocks eliminate edge-case branches.

    Operations are all O(1):

    - popleft() / popleft_n(n): pop from head (LRU end)
    - remove(block):           remove from middle (on cache hit touch)
    - append(block):           push to tail (most recently freed)
    - append_n(blocks):        batch push to tail
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Link consecutive blocks.
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Fake sentinel nodes (never popped, never freed).
        self._head = KVCacheBlock(block_id=-1)
        self._tail = KVCacheBlock(block_id=-1)

        if self.num_free_blocks > 0:
            self._head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self._head
            self._tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self._tail
        else:
            self._head.next_free_block = self._tail
            self._tail.prev_free_block = self._head

    # ── helpers ───────────────────────────────────────────────────────

    def _remove_real(self, block: KVCacheBlock) -> None:
        """Detach *block* from the linked list (caller ensures it's real)."""
        prev_b = block.prev_free_block
        next_b = block.next_free_block
        prev_b.next_free_block = next_b     # type: ignore[union-attr]
        next_b.prev_free_block = prev_b     # type: ignore[union-attr]
        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

    # ── public API ────────────────────────────────────────────────────

    def popleft(self) -> KVCacheBlock:
        """Remove and return the block at the head (LRU end)."""
        if self.num_free_blocks == 0:
            raise ValueError("Free queue is empty")
        block = self._head.next_free_block
        if block is self._tail:  # sanity — should not happen when count > 0
            raise ValueError("Free queue is empty")
        self._remove_real(block)
        return block

    def remove(self, block: KVCacheBlock) -> None:
        """Remove *block* from the middle of the queue.

        Raises RuntimeError if block is not in the free list
        (prev/next pointers are None).
        """
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(
                f"Block {block.block_id} is not in the free queue"
            )
        self._remove_real(block)

    def append(self, block: KVCacheBlock) -> None:
        """Insert *block* at the tail (most recently freed)."""
        prev_tail = self._tail.prev_free_block
        prev_tail.next_free_block = block    # type: ignore[union-attr]
        block.prev_free_block = prev_tail
        block.next_free_block = self._tail
        self._tail.prev_free_block = block
        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Insert *blocks* at the tail in order, batch O(1)."""
        if not blocks:
            return

        # Chain the incoming blocks together.
        for i in range(len(blocks)):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < len(blocks) - 1:
                blocks[i].next_free_block = blocks[i + 1]

        first = blocks[0]
        last = blocks[-1]

        prev_tail = self._tail.prev_free_block
        prev_tail.next_free_block = first   # type: ignore[union-attr]
        first.prev_free_block = prev_tail
        last.next_free_block = self._tail
        self._tail.prev_free_block = last

        self.num_free_blocks += len(blocks)

class BlockHashToBlockMap:
    """Multi-block-aware prefix cache map (vLLM 0.19.0).

    Stores either a single ``KVCacheBlock`` or a ``dict[int, KVCacheBlock]``
    per hash key.  Two different blocks can share the same hash without
    collapsing — vLLM deliberately does not deduplicate prefix blocks.
    """

    def __init__(self) -> None:
        self._cache: dict[bytes, KVCacheBlock | dict[int, KVCacheBlock]] = {}

    def get_one_block(self, key: bytes) -> KVCacheBlock | None:
        """Return any block for *key*, or None."""
        blocks = self._cache.get(key)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            return blocks
        return next(iter(blocks.values()))

    def insert(self, key: bytes, block: KVCacheBlock) -> None:
        """Register *block* under *key*."""
        blocks = self._cache.get(key)
        if blocks is None:
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # Upgrade to dict.
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        else:
            blocks[block.block_id] = block

    def pop(self, key: bytes, block_id: int) -> KVCacheBlock | None:
        """Remove and return the block matching *block_id* under *key*.

        Returns None if the key or the specific block_id is not found.
        When the last block under a key is removed, the key is deleted.
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # Wrong block — put back.
            self._cache[key] = blocks
            return None
        # dict case.
        block = blocks.pop(block_id, None)
        if blocks:
            self._cache[key] = blocks  # reinsert remaining dict
        return block

    def __len__(self) -> int:
        return len(self._cache)


# ── BlockPool ────────────────────────────────────────────────────────────

class BlockPool:
    """Manages KVCacheBlock allocation with prefix caching via block hash lookup.

    Supports GPU↔CPU swap for disaggregated D-side preemption:
    when the block pool is exhausted, running requests can be swapped out
    to CPU memory (keeping their computed tokens) and swapped back in
    later when space frees up, avoiding recomputation.

    Aligned with vLLM 0.19.0:
    - Chain hashing prevents false prefix matches.
    - FreeKVCacheBlockQueue (doubly linked list) for O(1) LRU eviction.
    - BlockHashToBlockMap for multi-block hash entries.
    - Deferred caching (``_pending_cache`` + ``commit_pending_cache()``)
      matches vLLM's post-GPU-execution ``cache_blocks()``.
    """

    def __init__(self, num_blocks: int, enable_caching: bool = True,
                 comm_model: str = "lut",
                 cpu_swap_bw_gb_s: float = 9.7,
                 comm_lut_bytes: list[int] | None = None,
                 comm_lut_time_s: list[float] | None = None):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.num_blocks = num_blocks
        self.enable_caching = enable_caching

        self._swap_comm_model = comm_model
        self._swap_cpu_bw_gb_s = cpu_swap_bw_gb_s
        self._swap_lut_bytes = comm_lut_bytes
        self._swap_lut_time_s = comm_lut_time_s

        # Swapped-out requests: request_id → num_blocks
        self._swapped_out: dict[str, int] = {}

        # Prefix cache statistics
        self._cache_queries: int = 0
        self._cache_hits: int = 0

        # All blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_blocks)
        ]

        # Free block queue (doubly linked list, LRU order).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Reserve one block as the null/placeholder block (never freed).
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        self.null_block.ref_cnt = 1

        # Prefix cache: block hash → KVCacheBlock(s).
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Deferred cache: blocks allocated this step whose hashes are
        # committed only after GPU execution completes (P3-10).
        self._pending_cache: list[KVCacheBlock] = []

        self.clock: float = 0.0

    # ── query ──────────────────────────────────────────────────────────

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

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
        block = self.cached_block_hash_to_block.get_one_block(block_hash)
        return block.block_id if block else None

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
        self.cached_block_hash_to_block.insert(block.block_hash, block)

    def cache_new_full_blocks(self, request: "Request", block_size: int) -> None:
        """Cache blocks that became full after new tokens were generated.

        During decode, a previously-partial block (e.g., the last block of a
        prefill that was not full) may become full as output tokens accumulate.
        This method detects those blocks and adds them to ``_pending_cache``
        so they become visible via ``commit_pending_cache()``.
        """
        if not self.enable_caching:
            return
        for bi, bid in enumerate(request.block_table):
            if bid < 0:
                continue
            block = self.blocks[bid]
            if block.block_hash is not None or block.is_null:
                continue
            # Block is full if (block index + 1) * block_size <= computed tokens
            if (bi + 1) * block_size <= request.num_computed_tokens:
                if bi < len(request.block_hashes):
                    block.block_hash = request.block_hashes[bi]
                    self._pending_cache.append(block)

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """Remove *block* from the hash cache, if present.

        Returns True if the block was actually evicted from the cache.
        """
        block_hash = block.block_hash
        if block_hash is None:
            return False
        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            return False
        block.reset_hash()
        return True

    def commit_pending_cache(self) -> None:
        """Commit deferred block hashes into the prefix cache.

        Called after GPU execution completes (via scheduler's
        ``update_from_output()``) so that blocks allocated in this step
        become visible to subsequent scheduling rounds only AFTER their
        content has been written by the GPU.  Matches vLLM's
        ``cache_blocks()`` called from ``update_from_output()``.
        """
        for block in self._pending_cache:
            self._cache_block(block)
        self._pending_cache.clear()

    # ── allocation ─────────────────────────────────────────────────────

    def touch(self, block_ids: list[int]) -> None:
        """Increment ref_cnt for cached blocks being reused.

        If ref_cnt was 0, the block is in the free queue — remove it
        via the doubly linked list (O(1)).  Guard against double-touch
        (ref_cnt already > 0, or already removed).
        """
        for bid in block_ids:
            block = self.blocks[bid]
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def get_new_block(self) -> KVCacheBlock:
        """Allocate one block from the free pool. Evicts cached block if needed.

        Returns null_block if the pool is exhausted — callers should check is_null.
        """
        try:
            block = self.free_block_queue.popleft()
        except ValueError:
            return self.null_block  # OOM
        self._maybe_evict_cached_block(block)
        block.ref_cnt += 1
        return block

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
            # Use new_end (tokens computed after this step) so that output
            # token positions beyond prompt_len are correctly sized.
            block_end = min(block_start + block_size, new_end)

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
                if cached_id is not None:
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

            # If this block is now full, defer caching until GPU step completes.
            # vLLM caches after execution (update_from_output), not at allocation
            # time.  Premature caching allows same-step requests to falsely hit.
            if block_end - block_start == block_size and bi < len(request.block_hashes):
                block.block_hash = request.block_hashes[bi]
                self._pending_cache.append(block)

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
        """Decrement ref_cnt for blocks.  Move to free queue if ref_cnt hits 0."""
        to_free: list[KVCacheBlock] = []
        for bid in block_ids:
            if bid < 0:
                continue
            block = self.blocks[bid]
            if block.is_null:
                continue
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                to_free.append(block)
        if to_free:
            self.free_block_queue.append_n(to_free)

    def free_request(self, request: "Request") -> None:
        """Free all blocks allocated to a request."""
        self.free_blocks([b for b in request.block_table if b >= 0])
        request.block_table.clear()

    # ── GPU ↔ CPU swap ────────────────────────────────────────────────

    def is_swapped(self, request_id: str) -> bool:
        """Return True if the request's blocks are currently swapped out."""
        return request_id in self._swapped_out

    def _swap_time(self, total_bytes: int) -> float:
        """Predicted transfer time for GPU↔CPU swap."""
        if self._swap_comm_model == "bw_latency":
            if self._swap_cpu_bw_gb_s <= 0:
                return 0.0
            return total_bytes / (self._swap_cpu_bw_gb_s * 1e9)
        # LUT model
        if self._swap_lut_bytes is None or self._swap_lut_time_s is None:
            return 0.0
        return memcpy_time(total_bytes,
                           lut_bytes=self._swap_lut_bytes,
                           lut_time_s=self._swap_lut_time_s)

    def swap_out(self, request: "Request", block_size: int) -> float:
        """Swap a request's KV cache blocks to CPU.

        Frees physical blocks and records the count for later swap-in.
        Returns the swap-out time in seconds, or 0 if swap is disabled.
        """
        if self._swap_comm_model == "bw_latency":
            if self._swap_cpu_bw_gb_s <= 0:
                return 0.0
        elif self._swap_lut_bytes is None:
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

        return self._swap_time(total_bytes)

    def swap_in(self, request: "Request", block_size: int) -> float:
        """Restore a request's KV cache blocks from CPU.

        Allocates fresh blocks; the data transfer is modelled as time only.
        Returns the swap-in time in seconds, or float('inf') if allocation
        fails (not enough free blocks).
        """
        if self._swap_comm_model == "bw_latency":
            if self._swap_cpu_bw_gb_s <= 0:
                return 0.0
        elif self._swap_lut_bytes is None:
            return 0.0

        num_blocks = self._swapped_out.get(request.request_id)
        if num_blocks is None:
            return 0.0

        total_bytes = num_blocks * self._block_bytes
        swap_time = self._swap_time(total_bytes)

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
