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
    """

    @property
    def real_page_size_bytes(self) -> int:
        b_mse = int(os.environ.get("TQ_B_MSE", "2"))
        b_qjl = int(os.environ.get("TQ_B_QJL", "1"))
        val_bits = b_mse + b_qjl
        fp16 = _compressed_fp16_elems(self.head_size, b_mse, b_qjl, val_bits)
        return self.block_size * self.num_kv_heads * fp16 * get_dtype_size(self.dtype)
