"""
TurboQuant vLLM Plugin — General Plugin

Registers the TurboQuant attention backend and patches the KV cache spec
to allocate compressed (4.3x smaller) cache pages.

Discovered by vLLM via the ``vllm.general_plugins`` entry point.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def _custom_backend_selected() -> bool:
    """Best-effort detection of whether the user actually picked CUSTOM/TQ.

    The plugin is loaded by vLLM on every startup via the
    ``vllm.general_plugins`` entry point, so we cannot install a global
    monkey-patch unconditionally — that would clobber the KV cache spec for
    runs using FlashAttention or other non-TQ backends.

    We treat the user as having picked CUSTOM if either:
      * ``VLLM_ATTENTION_BACKEND`` env var is ``CUSTOM`` (case-insensitive), or
      * the CLI was invoked with ``--attention-backend CUSTOM`` /
        ``--attention-backend=CUSTOM``.
    """
    if os.environ.get("VLLM_ATTENTION_BACKEND", "").upper() == "CUSTOM":
        return True
    argv = getattr(sys, "argv", []) or []
    for i, arg in enumerate(argv):
        if arg == "--attention-backend" and i + 1 < len(argv):
            if argv[i + 1].upper() == "CUSTOM":
                return True
        elif arg.startswith("--attention-backend="):
            if arg.split("=", 1)[1].upper() == "CUSTOM":
                return True
    return False


def register_turboquant() -> None:
    """Register TurboQuant backend + patch KV cache allocation.

    Set TQ_HYBRID=1 to use the hybrid backend (TQ compressed storage +
    SDPA attention compute) instead of the default pure TQ backend.
    """
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
        use_hybrid = os.environ.get("TQ_HYBRID", "0") == "1"
        if use_hybrid:
            register_backend(
                AttentionBackendEnum.CUSTOM,
                "vllm_plugin.attention_hybrid.HybridTQAttentionBackend",
            )
            logger.info("[TurboQuant] Registered HYBRID (TQ storage + SDPA) as CUSTOM backend")
        else:
            register_backend(
                AttentionBackendEnum.CUSTOM,
                "vllm_plugin.attention.TurboQuantAttentionBackend",
            )
            logger.info("[TurboQuant] Registered as CUSTOM attention backend")

        # Patch KV cache allocation only when the CUSTOM (TQ) backend is
        # actually selected for this run. Every TQ backend variant reads/
        # writes the compressed byte layout, so the allocator must size pages
        # to match — but installing the monkey-patch globally would force
        # TurboQuantSpec on FlashAttention runs too, causing the allocator to
        # under-size pages relative to what FlashAttention writes.
        #
        # TQ_PATCH_KV controls the gate:
        #   "auto" (default) — patch iff CUSTOM is selected (env or CLI)
        #   "1"              — force-patch (use when auto-detect misses your
        #                       config, e.g. backend chosen via Python API)
        #   "0"              — never patch
        patch_kv = os.environ.get("TQ_PATCH_KV", "auto").lower()
        if patch_kv == "1":
            do_patch = True
        elif patch_kv == "0":
            do_patch = False
        else:
            do_patch = _custom_backend_selected()

        if do_patch:
            _patch_kv_cache_spec()
        else:
            logger.info(
                "[TurboQuant] KV cache patch skipped "
                "(CUSTOM backend not selected; set TQ_PATCH_KV=1 to force)"
            )
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
