"""
TurboQuant KV Cache Spec — module-level class for pickling compatibility.

vLLM serializes KV cache specs between processes (TP workers). The spec
class must be importable by fully-qualified name, so it lives here rather
than inside a closure.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.kv_cache_interface import FullAttentionSpec, get_dtype_size
from vllm_plugin.attention import _compressed_fp16_elems


@dataclass(frozen=True, kw_only=True)
class TurboQuantSpec(FullAttentionSpec):
    """FullAttentionSpec with compressed ``real_page_size_bytes``.

    Tells vLLM's block allocator to use ~4.3x less memory per KV cache
    page, matching the TurboQuant compressed layout.

    The page size is snapshotted once in ``__post_init__`` (via
    ``TurboQuantConfig`` so env *and* GGUF overrides are honoured) and
    baked into the pickled state, so every TP worker sees an identical
    value regardless of its own environment.
    """

    def __post_init__(self) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Route through TurboQuantConfig so b_mse / b_qjl honour both env
        # vars and GGUF metadata (TQ_GGUF_PATH).  Reading os.environ directly
        # would skip GGUF overrides and produce a page size that disagrees
        # with what TurboQuantAttentionImpl writes — silent buffer overrun.
        # Pass num_heads = num_kv_heads to trivially satisfy the config's
        # divisibility validation when env/GGUF leave num_heads at its default.
        from vllm_plugin.config import TurboQuantConfig
        cfg = TurboQuantConfig(
            num_heads=self.num_kv_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
        )
        fp16 = _compressed_fp16_elems(
            self.head_size, cfg.b_mse, cfg.b_qjl, cfg.b_total)
        size = (
            self.block_size * self.num_kv_heads
            * fp16 * get_dtype_size(self.dtype)
        )
        object.__setattr__(self, "_cached_page_size_bytes", size)

    @property
    def real_page_size_bytes(self) -> int:
        return self._cached_page_size_bytes
