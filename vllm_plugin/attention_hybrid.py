"""
Hybrid TurboQuant + SDPA Attention Backend

Stores KV cache compressed with TurboQuant (same ~4x memory reduction),
but dequantizes on-the-fly and uses torch.nn.functional.scaled_dot_product_attention
(which dispatches to FlashAttention/memory-efficient kernels) for the actual
attention computation.

Compared to the pure TurboQuant backend:
  - Same memory footprint (compressed KV cache)
  - Uses highly optimized fused attention kernels for compute
  - Requires full dequantization of K and V per step (extra work)
  - No asymmetric score estimation — uses standard dot-product on dequantized keys

Set TQ_HYBRID=1 in the environment to select this backend over the default
TurboQuant backend when --attention-backend CUSTOM is used.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F

from turboquant import TurboQuantProd, TurboQuantMSE, wht_unrotate
from vllm_plugin.attention import (
    _CompressedLayout,
    _compressed_fp16_elems,
    TurboQuantMetadata,
    TurboQuantMetadataBuilder,
)
from vllm_plugin.compress_utils import initialize_quantizers, store_compressed_kv
from vllm_plugin.config import TurboQuantConfig
from vllm_plugin.vllm_compat import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec


class HybridTQAttentionBackend(AttentionBackend):
    """TurboQuant compressed storage + SDPA attention compute."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True
    # Not CUDAGraph-safe: the forward loop iterates over requests in Python
    # and calls SDPA per-request with dynamic shapes.
    use_cudagraph: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16,
    ]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["HybridTQAttentionImpl"]:
        return HybridTQAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int,
                           num_kv_heads: int, head_size: int,
                           cache_dtype_str: str = "auto") -> tuple[int, ...]:
        cfg = TurboQuantConfig(
            num_heads=num_kv_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
        )
        fp16 = _compressed_fp16_elems(
            head_size, cfg.b_mse, cfg.b_qjl, cfg.b_total)
        return (num_blocks, block_size, num_kv_heads, fp16)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


class HybridTQAttentionImpl(AttentionImpl):
    """Compressed KV storage (TurboQuant) with SDPA attention compute.

    Keys and values are dequantized to fp16 before being passed to
    torch.nn.functional.scaled_dot_product_attention, which dispatches
    to the best available fused kernel (FlashAttention, memory-efficient,
    or math fallback).
    """

    _layer_counter: int = 0

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap

        # Deterministic seed per layer (resolved lazily).
        self.layer_idx: int | None = None

        cfg = TurboQuantConfig(
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=head_size,
        )
        self._b_mse = cfg.b_mse
        self._b_qjl = cfg.b_qjl
        self._b_total = cfg.b_total
        self._rotation = cfg.rotation
        self._heads_per_kv = self.num_heads // self.num_kv_heads

        self._layout = _CompressedLayout(
            head_size, cfg.b_mse, cfg.b_qjl, cfg.b_total)

        self._key_q: TurboQuantProd | None = None
        self._val_q: TurboQuantMSE | None = None
        self._init_device: torch.device | None = None

    def _resolve_layer_idx(self, layer: Any) -> int:
        if self.layer_idx is not None:
            return self.layer_idx

        layer_name = getattr(layer, "layer_name", None)
        if isinstance(layer_name, str):
            for part in layer_name.split("."):
                try:
                    self.layer_idx = int(part)
                    break
                except ValueError:
                    continue

        if self.layer_idx is None:
            self.layer_idx = HybridTQAttentionImpl._layer_counter
            HybridTQAttentionImpl._layer_counter += 1

        return self.layer_idx

    def _ensure_quantizers(self, device: torch.device, layer: Any) -> None:
        if self._key_q is not None and self._init_device == device:
            return
        layer_idx = self._resolve_layer_idx(layer)
        quantizers = initialize_quantizers(
            self.head_size,
            self._b_total,
            layer_idx,
            device,
            rotation=self._rotation,
        )
        self._key_q = quantizers["key_q"]
        self._val_q = quantizers["val_q"]
        self._init_device = device

        self._key_Pi = quantizers["key_pi"]
        self._key_centroids = quantizers["key_centroids"]
        self._val_Pi = quantizers["val_pi"]
        self._val_centroids = quantizers["val_centroids"]
        self._S_T = quantizers["s_t"]
        self._key_sigma = quantizers["key_sigma"]
        self._val_sigma = quantizers["val_sigma"]

    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = query.device

        if attn_metadata is None:
            if output is None:
                output = torch.zeros(
                    query.shape[0], self.num_heads, self.head_size,
                    dtype=query.dtype, device=device)
            return output

        self._ensure_quantizers(device, layer)

        N = attn_metadata.num_actual_tokens
        num_reqs = attn_metadata.seq_lens.shape[0]
        block_size = kv_cache.shape[1]

        if output is None:
            output = torch.zeros(
                query.shape[0], self.num_heads, self.head_size,
                dtype=query.dtype, device=device)

        # 1) Compress and store new KV tokens (identical to pure TQ)
        if key is not None and value is not None:
            self._store_compressed(
                key[:N], value[:N], kv_cache,
                attn_metadata.slot_mapping[:N], block_size)

        # 2) Compute attention per request using SDPA
        for ri in range(num_reqs):
            qs = attn_metadata.query_start_loc[ri].item()
            qe = attn_metadata.query_start_loc[ri + 1].item()
            q_len = qe - qs
            seq_len = attn_metadata.seq_lens[ri].item()
            if q_len == 0 or seq_len == 0:
                continue

            # Gather compressed KV
            n_blk = (seq_len + block_size - 1) // block_size
            blk_ids = attn_metadata.block_table[ri, :n_blk]
            comp = kv_cache[blk_ids].reshape(
                -1, self.num_kv_heads, self._layout.fp16_elems)[:seq_len]
            comp_bytes = comp.contiguous().view(torch.uint8).reshape(
                seq_len, self.num_kv_heads, self._layout.total_bytes)

            q = query[qs:qe]  # (q_len, nh, hd)
            pos_offset = seq_len - q_len

            self._attn_sdpa(
                q, comp_bytes, seq_len, q_len, pos_offset,
                output, qs, attn_metadata.causal)

        return output

    def _store_compressed(
        self, key: torch.Tensor, value: torch.Tensor,
        kv_cache: torch.Tensor, slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Compress K/V tensors and write the packed bytes into ``kv_cache``."""
        store_compressed_kv(
            key,
            value,
            kv_cache,
            slot_mapping,
            block_size,
            self.num_kv_heads,
            self.head_size,
            self._layout,
            self._key_q,
            self._val_q,
        )

    def _dequantize_kv(
        self, comp_bytes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize compressed KV back to fp16 keys and values.

        Args:
            comp_bytes: (seq_len, num_kv_heads, total_bytes) uint8

        Returns:
            keys:   (seq_len, num_kv_heads, head_dim) fp16
            values: (seq_len, num_kv_heads, head_dim) fp16
        """
        seq_len, nkh = comp_bytes.shape[:2]
        hd = self.head_size

        flat = comp_bytes.reshape(seq_len * nkh, self._layout.total_bytes)
        km_idx, k_signs, k_rnorm, k_norm, vm_idx, v_norm = self._layout.unpack(flat)

        # Dequantize keys: codebook lookup → inverse rotation → scale by norm
        k_rotated = self._key_centroids[km_idx.reshape(-1, hd)]  # (S*nkh, D)
        if self._rotation == "wht":
            k_deq = wht_unrotate(k_rotated.float(), self._key_sigma).half()
        else:
            k_deq = k_rotated @ self._key_Pi
        k_deq = k_deq.reshape(seq_len, nkh, hd)  # inverse rotation
        k_deq = k_deq * k_norm.half().reshape(seq_len, nkh, 1)

        # Dequantize values: codebook lookup → inverse rotation → scale by norm
        v_rotated = self._val_centroids[vm_idx.reshape(-1, hd)]
        if self._rotation == "wht":
            v_deq = wht_unrotate(v_rotated.float(), self._val_sigma).half()
        else:
            v_deq = v_rotated @ self._val_Pi
        v_deq = v_deq.reshape(seq_len, nkh, hd)
        v_deq = v_deq * v_norm.half().reshape(seq_len, nkh, 1)

        return k_deq, v_deq

    def _attn_manual_softcap(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        q_len: int, seq_len: int, pos_offset: int, causal: bool,
    ) -> torch.Tensor:
        """Manual attention with Gemma-style tanh logit soft-capping.

        SDPA has no hook for soft-capping, so when ``self.logits_soft_cap``
        is set we compute scores → tanh-cap → (optional causal mask) →
        softmax → @V ourselves.  Inputs are already shaped for SDPA:
        ``(1, nh, *, hd)``.
        """
        cap = self.logits_soft_cap
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = (scores.float() / cap).tanh() * cap
        if causal and q_len > 1:
            qpos = torch.arange(q_len, device=q.device) + pos_offset
            kpos = torch.arange(seq_len, device=q.device)
            mask = kpos[None, :] > qpos[:, None]            # (q_len, s_len)
            scores = scores.masked_fill(
                mask[None, None, :, :], float("-inf"))
        weights = F.softmax(scores, dim=-1).to(v.dtype)
        return torch.matmul(weights, v)

    def _attn_sdpa(
        self, queries: torch.Tensor, comp_bytes: torch.Tensor,
        seq_len: int, q_len: int, pos_offset: int,
        output: torch.Tensor, out_offset: int, causal: bool,
    ) -> None:
        """Attention via dequantized KV + scaled_dot_product_attention."""
        nkh = self.num_kv_heads
        hpkv = self._heads_per_kv
        hd = self.head_size

        # Dequantize all KV for this request
        k_deq, v_deq = self._dequantize_kv(comp_bytes)
        # k_deq, v_deq: (seq_len, nkh, hd)

        # Expand KV heads for GQA: (seq_len, nkh, hd) → (seq_len, nh, hd)
        if hpkv > 1:
            k_deq = k_deq.unsqueeze(2).expand(-1, -1, hpkv, -1).reshape(
                seq_len, nkh * hpkv, hd)
            v_deq = v_deq.unsqueeze(2).expand(-1, -1, hpkv, -1).reshape(
                seq_len, nkh * hpkv, hd)

        # Reshape for SDPA: (batch=1, num_heads, seq_len, head_dim)
        # SDPA expects: query (1, nh, q_len, hd), key (1, nh, s_len, hd), value same
        q = queries.half().permute(1, 0, 2).unsqueeze(0)   # (1, nh, q_len, hd)
        k = k_deq.half().permute(1, 0, 2).unsqueeze(0)     # (1, nh, s_len, hd)
        v = v_deq.half().permute(1, 0, 2).unsqueeze(0)     # (1, nh, s_len, hd)

        if self.logits_soft_cap is not None:
            # SDPA has no soft-cap option — do the attention manually so we
            # can apply tanh(x / cap) * cap to the pre-softmax logits.
            out = self._attn_manual_softcap(
                q, k, v, q_len, seq_len, pos_offset, causal)
        elif causal and q_len > 1 and pos_offset == 0 and q_len == seq_len:
            # Full prefill — use is_causal=True (efficient fused path)
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale)
        elif causal and q_len > 1:
            # Partial prefill with offset — need explicit mask
            qpos = torch.arange(q_len, device=queries.device) + pos_offset
            kpos = torch.arange(seq_len, device=queries.device)
            mask = kpos[None, :] <= qpos[:, None]  # (q_len, s_len)
            mask = mask.unsqueeze(0).unsqueeze(0)   # (1, 1, q_len, s_len)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=self.scale)
        else:
            # Decode or non-causal
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=self.scale)

        # (1, nh, q_len, hd) → (q_len, nh, hd)
        out = out.squeeze(0).permute(1, 0, 2)

        if output.dim() == 3:
            output[out_offset:out_offset + q_len] = out.to(output.dtype)
        else:
            output[out_offset:out_offset + q_len] = out.reshape(
                q_len, nkh * hpkv * hd).to(output.dtype)
