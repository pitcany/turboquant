"""
TurboQuant KV Cache Spec — module-level class for pickling compatibility.

vLLM serializes KV cache specs between processes (TP workers). The spec
class must be importable by fully-qualified name, so it lives here rather
than inside a closure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from vllm.v1.kv_cache_interface import FullAttentionSpec, get_dtype_size
from vllm_plugin.attention import _compressed_fp16_elems


@dataclass(frozen=True, kw_only=True)
class TurboQuantSpec(FullAttentionSpec):
    """FullAttentionSpec with compressed ``real_page_size_bytes``.

    Tells vLLM's block allocator to use ~4.3x less memory per KV cache
    page, matching the TurboQuant compressed layout.

    The page size is snapshotted once in ``__post_init__`` and baked into
    the pickled state, so every TP worker sees an identical value —
    re-reading ``TQ_B_MSE`` / ``TQ_B_QJL`` on each access would let a worker
    with a divergent environment compute a different page size than the
    driver that owns the allocator.
    """

    def __post_init__(self) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        b_mse = int(os.environ.get("TQ_B_MSE", "2"))
        b_qjl = int(os.environ.get("TQ_B_QJL", "1"))
        val_bits = b_mse + b_qjl
        fp16 = _compressed_fp16_elems(self.head_size, b_mse, b_qjl, val_bits)
        size = (
            self.block_size * self.num_kv_heads
            * fp16 * get_dtype_size(self.dtype)
        )
        object.__setattr__(self, "_cached_page_size_bytes", size)

    @property
    def real_page_size_bytes(self) -> int:
        return self._cached_page_size_bytes
