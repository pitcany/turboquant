"""
TurboQuant vLLM Plugin — General Plugin

Registers the TurboQuant attention backend and patches the KV cache spec
to allocate compressed (4.3x smaller) cache pages.

Discovered by vLLM via the ``vllm.general_plugins`` entry point.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_turboquant() -> None:
    """Register TurboQuant backend + patch KV cache allocation."""
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "vllm_plugin.attention.TurboQuantAttentionBackend",
        )
        _patch_kv_cache_spec()
        logger.info("[TurboQuant] Registered as CUSTOM attention backend")
    except ImportError:
        logger.warning(
            "[TurboQuant] vLLM not available — skipping backend registration"
        )
    except Exception:
        logger.exception("[TurboQuant] Failed to register attention backend")


def _patch_kv_cache_spec() -> None:
    """Monkey-patch Attention.get_kv_cache_spec for compressed pages."""
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backend import AttentionType
    from vllm_plugin.kv_spec import TurboQuantSpec

    _original = Attention.get_kv_cache_spec

    def _patched(self, vllm_config):
        if self.attn_type != AttentionType.DECODER:
            return _original(self, vllm_config)
        if self.sliding_window is not None:
            return _original(self, vllm_config)

        block_size = vllm_config.cache_config.block_size
        return TurboQuantSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
        )

    Attention.get_kv_cache_spec = _patched

    # Register TurboQuantSpec in the KV cache manager's spec→manager map
    from vllm.v1.core.single_type_kv_cache_manager import (
        spec_manager_map,
        FullAttentionManager,
    )
    spec_manager_map[TurboQuantSpec] = FullAttentionManager

    logger.info("[TurboQuant] Patched KV cache spec for compressed page sizes")
