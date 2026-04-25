"""Shared quantizer initialization and KV packing helpers for vLLM backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from turboquant import TurboQuantMSE, TurboQuantProd
from vllm_plugin.triton_kernels import TRITON_AVAILABLE

if TYPE_CHECKING:
    from vllm_plugin.attention import _CompressedLayout

_NORM_EPS = 1e-8


def initialize_quantizers(
    head_size: int,
    total_bits: int,
    layer_idx: int,
    device: torch.device,
    rotation: str = "wht",
) -> dict[str, torch.Tensor | TurboQuantProd | TurboQuantMSE]:
    key_q = TurboQuantProd(
        head_size,
        total_bits,
        seed=layer_idx * 1000,
        device=str(device),
        rotation=rotation,
    )
    val_q = TurboQuantMSE(
        head_size,
        total_bits,
        seed=layer_idx * 1000 + 500,
        device=str(device),
        rotation=rotation,
    )
    return {
        "key_q": key_q,
        "val_q": val_q,
        "key_pi": key_q.mse.Pi.half(),
        "key_centroids": key_q.mse.centroids.half(),
        "val_pi": val_q.Pi.half(),
        "val_centroids": val_q.centroids.half(),
        "s_t": key_q.S.T.half(),
        "rotation": rotation,
        "key_sigma": getattr(key_q.mse, "sigma", None),
        "val_sigma": getattr(val_q, "sigma", None),
    }


def _can_fuse(key: torch.Tensor, key_q: TurboQuantProd,
              val_q: "TurboQuantMSE | None" = None) -> bool:
    """Check whether the fused Triton compress path is available."""
    return (
        TRITON_AVAILABLE
        and key.is_cuda
        and key_q.rotation == "wht"
        and (val_q is None or getattr(val_q, "rotation", "wht") == "wht")
        and key_q.qjl_dim == key_q.d
        and key_q.mse.bits == 2
        and key_q.d <= 256  # FWHT unrolled to 8 butterfly steps max
    )


def _can_fuse_pack(key: torch.Tensor, key_q: TurboQuantProd,
                    val_q: TurboQuantMSE,
                    layout: "_CompressedLayout") -> bool:
    """Check whether the single-kernel compress+pack path is available."""
    return (
        _can_fuse(key, key_q, val_q)
        and layout.key_mse_bits == 2
        and layout.val_mse_bits <= 4
    )


def store_compressed_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    layout: "_CompressedLayout",
    key_q: TurboQuantProd,
    val_q: TurboQuantMSE,
) -> None:
    """Compress K/V tensors and write the packed bytes into ``kv_cache``."""

    num_tokens = key.shape[0]
    block_indices = slot_mapping // block_size
    block_offsets = slot_mapping % block_size

    k_flat = key.reshape(num_tokens * num_kv_heads, head_size).float()
    v_flat = value.reshape(num_tokens * num_kv_heads, head_size).float()

    if _can_fuse_pack(key, key_q, val_q, layout):
        packed = _compress_fused_pack(k_flat, v_flat, layout, key_q, val_q)
    elif _can_fuse(key, key_q, val_q):
        packed = _compress_fused(k_flat, v_flat, layout, key_q, val_q)
    else:
        packed = _compress_torch(k_flat, v_flat, layout, key_q, val_q)

    # The packed layout is always fp16-aligned (total_bytes is even,
    # fp16_elems = total_bytes // 2).  Always view as fp16 regardless
    # of the kv_cache tensor's dtype to avoid misalignment.
    packed_fp16 = packed.view(torch.float16).reshape(
        num_tokens,
        num_kv_heads,
        layout.fp16_elems,
    )

    if kv_cache.dtype == torch.bfloat16:
        # Preserve raw packed bytes when kv_cache uses bf16 storage.
        packed_view = packed_fp16.view(torch.bfloat16)
    else:
        packed_view = packed_fp16.view(kv_cache.dtype)

    kv_cache[block_indices, block_offsets] = packed_view


def _compress_torch(
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    layout: "_CompressedLayout",
    key_q: TurboQuantProd,
    val_q: TurboQuantMSE,
) -> torch.Tensor:
    """Reference Python compress path."""
    key_norms = torch.norm(k_flat, dim=-1).clamp_min(_NORM_EPS)
    value_norms = torch.norm(v_flat, dim=-1).clamp_min(_NORM_EPS)
    key_units = k_flat / key_norms.unsqueeze(-1)
    value_units = v_flat / value_norms.unsqueeze(-1)

    compressed_keys = key_q.quantize(key_units)
    value_indices = val_q.quantize(value_units)

    return layout.pack(
        compressed_keys["mse_indices"],
        compressed_keys["qjl_signs"],
        compressed_keys["residual_norm"],
        key_norms,
        value_indices,
        value_norms,
    )


def _compress_fused_pack(
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    layout: "_CompressedLayout",
    key_q: TurboQuantProd,
    val_q: TurboQuantMSE,
) -> torch.Tensor:
    """Single-kernel compress+pack — raw K/V to packed bytes in one launch."""
    from vllm_plugin.triton_kernels import _fused_compress_pack_triton

    return _fused_compress_pack_triton(
        k_flat,
        v_flat,
        key_sigma=key_q.mse.sigma,
        val_sigma=val_q.sigma,
        key_boundaries=key_q.mse.boundaries,
        key_centroids=key_q.mse.centroids,
        val_boundaries=val_q.boundaries,
        s_matrix=key_q.S,
        layout=layout,
    )


def _compress_fused(
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    layout: "_CompressedLayout",
    key_q: TurboQuantProd,
    val_q: TurboQuantMSE,
) -> torch.Tensor:
    """Fused Triton compress path — two kernels (compress + pack)."""
    from vllm_plugin.triton_kernels import _fused_compress_triton, _pack_triton

    raw = _fused_compress_triton(
        k_flat,
        v_flat,
        key_sigma=key_q.mse.sigma,
        val_sigma=val_q.sigma,
        key_boundaries=key_q.mse.boundaries,
        key_centroids=key_q.mse.centroids,
        val_boundaries=val_q.boundaries,
        s_matrix=key_q.S,
        head_dim=key_q.d,
        qjl_dim=key_q.qjl_dim,
    )

    return _pack_triton(
        raw["key_mse_indices"],
        raw["qjl_signs"],
        raw["key_residual_norm"],
        raw["key_norm"],
        raw["val_mse_indices"],
        raw["val_norm"],
        layout,
    )
