"""Shared quantizer initialization and KV packing helpers for vLLM backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from turboquant import TurboQuantMSE, TurboQuantProd

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
    }


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

    key_norms = torch.norm(k_flat, dim=-1).clamp_min(_NORM_EPS)
    value_norms = torch.norm(v_flat, dim=-1).clamp_min(_NORM_EPS)
    key_units = k_flat / key_norms.unsqueeze(-1)
    value_units = v_flat / value_norms.unsqueeze(-1)

    compressed_keys = key_q.quantize(key_units)
    value_indices = val_q.quantize(value_units)

    packed = layout.pack(
        compressed_keys["mse_indices"],
        compressed_keys["qjl_signs"],
        compressed_keys["residual_norm"],
        key_norms,
        value_indices,
        value_norms,
    )
    packed_fp16 = packed.view(kv_cache.dtype).reshape(
        num_tokens,
        num_kv_heads,
        layout.fp16_elems,
    )

    kv_cache[block_indices, block_offsets] = packed_fp16
