"""Request model and status enums for PD simulation."""

import enum
from dataclasses import dataclass, field


class RequestStatus(enum.IntEnum):
    WAITING = 0
    WAITING_FOR_REMOTE_KVS = 1
    RUNNING = 2
    PREEMPTED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status >= RequestStatus.PREEMPTED


class FinishReason(enum.Enum):
    LENGTH = "length"
    ABORT = "abort"


@dataclass
class Request:
    request_id: str
    arrival_time: float
    prompt_token_ids: list[int]
    max_output_len: int
    priority: int = 0

    # Runtime state
    num_computed_tokens: int = 0
    num_output_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    finish_reason: FinishReason | None = None
    finish_time: float | None = None
    ttft: float | None = None
    is_prefill_chunk: bool = True
    scheduled_ts: float | None = None  # clock when first admitted to running

    # Output token IDs (for full-sequence hash computation in prefix cache).
    # Agentic traces: set from sub-request's output_tok_ids at load time.
    # ShareGPT traces: also set from trace if available.
    output_tok_ids: list[int] = field(default_factory=list)

    # Prefix caching
    block_hashes: list[bytes] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)  # ordered list of block_ids

    # Disaggregation: prefill→decode transfer tracking
    kv_transfer_start: float | None = None
    kv_transfer_end: float | None = None

    # Agentic trace: session chaining
    session_id: str | None = None
    sub_request_index: int = 0
    next_sub_request: 'Request | None' = None
    tool_duration: float = 0.0  # seconds, pause after this sub_request completes

    @property
    def num_tokens(self) -> int:
        """Total tokens for this request (prompt + generated output)."""
        return len(self.prompt_token_ids) + self.num_output_tokens

    @property
    def num_tokens_with_spec(self) -> int:
        """Tokens including any spec tokens (none in our simulator)."""
        return self.num_tokens

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)


