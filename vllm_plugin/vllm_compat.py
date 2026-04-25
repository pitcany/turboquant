"""Small compatibility layer for importing the vLLM attention interfaces."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from vllm.v1.attention.backend import (
        AttentionBackend,
        AttentionImpl,
        AttentionMetadataBuilder,
        AttentionType,
        CommonAttentionMetadata,
        MultipleOf,
    )
    VLLM_AVAILABLE = True
except ModuleNotFoundError:
    VLLM_AVAILABLE = False

    class AttentionBackend:  # pragma: no cover - exercised when vllm is absent.
        pass

    class AttentionImpl:  # pragma: no cover - exercised when vllm is absent.
        pass

    class AttentionMetadataBuilder:  # pragma: no cover - exercised when vllm is absent.
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    class AttentionType:  # pragma: no cover - exercised when vllm is absent.
        DECODER = "DECODER"

    @dataclass
    class CommonAttentionMetadata:  # pragma: no cover - exercised when vllm is absent.
        num_actual_tokens: int
        max_query_len: int
        query_start_loc: object
        seq_lens: object
        block_table_tensor: object
        slot_mapping: object
        causal: bool = True

    class MultipleOf:  # pragma: no cover - exercised when vllm is absent.
        def __init__(self, value: int) -> None:
            self.value = value
