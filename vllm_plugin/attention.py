"""
TurboQuant vLLM v1 Attention Backend

Implements KV cache compression using TurboQuant's two-stage algorithm:
  Stage 1: PolarQuant — random rotation + Lloyd-Max MSE quantization (2-bit)
  Stage 2: QJL — 1-bit residual correction for unbiased inner products

Attention scores use the asymmetric estimator — keys are NEVER fully
dequantized for scoring. Values are dequantized for the weighted sum.

The compressed data is stored directly in vLLM's KV cache tensor using a
custom byte layout, providing ~4x memory reduction vs FP16.

Reference: https://arxiv.org/abs/2504.19874
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F

from turboquant import TurboQuantProd, TurboQuantMSE, wht_rotate, wht_unrotate
from vllm_plugin.compress_utils import initialize_quantizers, store_compressed_kv
from vllm_plugin.config import TurboQuantConfig
from vllm_plugin.triton_wrapper import (
    fused_decode_attention,
    turboquant_decode_attention,
)
from vllm_plugin.vllm_compat import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

# ═══════════════════════════════════════════════════════════════════════════
# Bit-packing utilities
# ═══════════════════════════════════════════════════════════════════════════


def _pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit indices (0-3) into uint8.  (n, d) → (n, d//4)."""
    n, d = indices.shape
    idx = indices.to(torch.uint8).reshape(n, d // 4, 4)
    return idx[..., 0] | (idx[..., 1] << 2) | (idx[..., 2] << 4) | (idx[..., 3] << 6)


def _unpack_2bit(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack uint8 → 2-bit indices (0-3).  (n, d//4) → (n, d)."""
    # Expand each byte into 4 values using shifts + mask, fully vectorized
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
    expanded = (packed.unsqueeze(-1) >> shifts) & 0x03  # (n, d//4, 4)
    return expanded.reshape(packed.shape[0], d).long()


def _pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices (0-15) into uint8 nibbles.  (n, d) → (n, d//2)."""
    n, d = indices.shape
    idx = indices.to(torch.uint8).reshape(n, d // 2, 2)
    return idx[..., 0] | (idx[..., 1] << 4)


def _unpack_4bit(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack uint8 nibbles → 4-bit indices.  (n, d//2) → (n, d)."""
    shifts = torch.tensor([0, 4], device=packed.device, dtype=torch.uint8)
    expanded = (packed.unsqueeze(-1) >> shifts) & 0x0F  # (n, d//2, 2)
    return expanded.reshape(packed.shape[0], d).long()


def _pack_bitplane(indices: torch.Tensor, num_planes: int) -> torch.Tensor:
    """Pack indices using bitplane packing.  (n, d) → (n, num_planes*ceil(d/8))."""
    n, d = indices.shape
    idx = indices.to(torch.uint8)
    padded_d = (d + 7) // 8 * 8
    planes = []
    for plane in range(num_planes):
        bits = (idx >> plane) & 1  # (n, d)
        if padded_d > d:
            bits = F.pad(bits, (0, padded_d - d))
        bits = bits.reshape(n, padded_d // 8, 8)
        shifts = torch.arange(8, device=bits.device, dtype=torch.uint8)
        packed_plane = (bits << shifts).sum(dim=-1).to(torch.uint8)  # (n, d/8)
        planes.append(packed_plane)
    return torch.cat(planes, dim=-1)  # (n, num_planes * ceil(d/8))


def _unpack_bitplane(packed: torch.Tensor, d: int, num_planes: int) -> torch.Tensor:
    """Unpack bitplane-packed indices.  (n, num_planes*ceil(d/8)) → (n, d)."""
    bytes_per_plane = (d + 7) // 8
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    result = torch.zeros(packed.shape[0], d, device=packed.device, dtype=torch.long)
    for plane in range(num_planes):
        plane_bytes = packed[:, plane * bytes_per_plane:(plane + 1) * bytes_per_plane]
        bits = ((plane_bytes.unsqueeze(-1) >> shifts) & 1)  # (n, bytes, 8)
        bits = bits.reshape(packed.shape[0], -1)[:, :d]  # (n, d)
        result |= bits.long() << plane
    return result


def _pack_nbits(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack n-bit indices into uint8 bytes. Dispatches to specialized packers."""
    if bits == 2:
        return _pack_2bit(indices)
    if bits == 4:
        return _pack_4bit(indices)
    if bits in (3, 5):
        return _pack_bitplane(indices, bits)
    raise ValueError(f"Unsupported bit width: {bits}")


def _unpack_nbits(packed: torch.Tensor, d: int, bits: int) -> torch.Tensor:
    """Unpack n-bit indices from uint8 bytes. Dispatches to specialized unpackers."""
    if bits == 2:
        return _unpack_2bit(packed, d)
    if bits == 4:
        return _unpack_4bit(packed, d)
    if bits in (3, 5):
        return _unpack_bitplane(packed, d, bits)
    raise ValueError(f"Unsupported bit width: {bits}")


def _pack_1bit(signs: torch.Tensor) -> torch.Tensor:
    """Pack ±1 signs into uint8.  (n, d) → (n, d//8)."""
    n, d = signs.shape
    bits = (signs > 0).to(torch.uint8).reshape(n, d // 8, 8)
    shifts = torch.arange(8, device=bits.device, dtype=torch.uint8)
    return (bits << shifts).sum(dim=-1).to(torch.uint8)


def _unpack_1bit(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack uint8 → ±1 signs.  (n, d//8) → (n, d) float."""
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    # packed: (n, d//8) → (n, d//8, 1), shifts: (8,) → broadcast
    bits = ((packed.unsqueeze(-1) >> shifts) & 1)  # (n, d//8, 8)
    return (bits.reshape(packed.shape[0], d).float() * 2 - 1)


# ═══════════════════════════════════════════════════════════════════════════
# Compressed cache byte layout
# ═══════════════════════════════════════════════════════════════════════════


def _packed_byte_len(d: int, bits: int) -> int:
    """Compute the packed byte length for *d* coordinates at *bits* per coord."""
    if bits in (1, 2, 4):
        return d * bits // 8
    if bits in (3, 5):
        # Bitplane packing: N planes, each ceil(d/8) bytes
        return bits * ((d + 7) // 8)
    raise ValueError(f"Unsupported bit width: {bits}")


class _CompressedLayout:
    """Byte-level layout of compressed KV data for one token, one KV head.

    Supports configurable bit-widths:
      - key_mse_bits ∈ {2, 3, 4}: MSE quantization indices
      - key_qjl_bits = 1: QJL sign bits
      - val_mse_bits ∈ {3, 4, 5}: value MSE indices (= key_mse_bits + key_qjl_bits)

    For head_dim=128, b_mse=2, b_qjl=1 (val_bits=3) the layout is::

        [0..31]    key MSE indices   (128×2 bit = 32 bytes)
        [32..47]   key QJL signs     (128×1 bit = 16 bytes)
        [48..49]   key residual norm (float16)
        [50..51]   key original norm (float16)
        [52..99]   val MSE indices   (128×3 bit = 48 bytes, bitplane)
        [100..101] val original norm (float16)
        ─────────  102 bytes = 51 float16 elements
    """

    def __init__(
        self,
        head_dim: int,
        key_mse_bits: int,
        key_qjl_bits: int,
        val_mse_bits: int,
    ):
        self.head_dim = head_dim
        self.key_mse_bits = key_mse_bits
        self.val_mse_bits = val_mse_bits

        # Key MSE indices
        self.km_off = 0
        self.km_len = _packed_byte_len(head_dim, key_mse_bits)
        # Key QJL signs: 1-bit packed
        self.kq_off = self.km_off + self.km_len
        self.kq_len = head_dim * key_qjl_bits // 8
        # Key residual norm (fp16)
        self.kr_off = self.kq_off + self.kq_len
        # Key original norm (fp16)
        self.kn_off = self.kr_off + 2
        # Value MSE indices.  For val_mse_bits <= 4, use 4-bit nibble packing
        # (wastes a few bits but keeps the Triton kernel compatible and matches
        # the original wire format).  For val_mse_bits == 5, use bitplane.
        self.vm_off = self.kn_off + 2
        if val_mse_bits <= 4:
            self._val_pack_bits = 4
            self.vm_len = head_dim * 4 // 8
        else:
            self._val_pack_bits = val_mse_bits
            self.vm_len = _packed_byte_len(head_dim, val_mse_bits)
        # Value norm (fp16)
        self.vn_off = self.vm_off + self.vm_len
        # Total
        self.total_bytes = self.vn_off + 2
        self.total_bytes += self.total_bytes % 2  # pad to even
        self.fp16_elems = self.total_bytes // 2

    # ── pack / unpack ────────────────────────────────────────────────

    def pack(self, k_mse: torch.Tensor, k_signs: torch.Tensor,
             k_rnorm: torch.Tensor, k_norm: torch.Tensor,
             v_mse: torch.Tensor, v_norm: torch.Tensor) -> torch.Tensor:
        """Pack all compressed components → (n, total_bytes) uint8."""
        n = k_mse.shape[0]
        dev = k_mse.device
        buf = torch.zeros(n, self.total_bytes, dtype=torch.uint8, device=dev)
        buf[:, self.km_off:self.km_off + self.km_len] = _pack_nbits(
            k_mse, self.key_mse_bits)
        buf[:, self.kq_off:self.kq_off + self.kq_len] = _pack_1bit(k_signs)
        buf[:, self.kr_off:self.kr_off + 2] = (
            k_rnorm.to(torch.float16).unsqueeze(-1).view(torch.uint8))
        buf[:, self.kn_off:self.kn_off + 2] = (
            k_norm.to(torch.float16).unsqueeze(-1).view(torch.uint8))
        buf[:, self.vm_off:self.vm_off + self.vm_len] = _pack_nbits(
            v_mse, self._val_pack_bits)
        buf[:, self.vn_off:self.vn_off + 2] = (
            v_norm.to(torch.float16).unsqueeze(-1).view(torch.uint8))
        return buf

    def unpack(self, buf: torch.Tensor):
        """Unpack (n, total_bytes) uint8 → component tensors."""
        d = self.head_dim
        k_mse = _unpack_nbits(
            buf[:, self.km_off:self.km_off + self.km_len], d,
            self.key_mse_bits)
        k_signs = _unpack_1bit(
            buf[:, self.kq_off:self.kq_off + self.kq_len], d)
        k_rnorm = buf[:, self.kr_off:self.kr_off + 2].contiguous().view(
            torch.float16).squeeze(-1).float()
        k_norm = buf[:, self.kn_off:self.kn_off + 2].contiguous().view(
            torch.float16).squeeze(-1).float()
        v_mse = _unpack_nbits(
            buf[:, self.vm_off:self.vm_off + self.vm_len], d,
            self._val_pack_bits)
        v_norm = buf[:, self.vn_off:self.vn_off + 2].contiguous().view(
            torch.float16).squeeze(-1).float()
        return k_mse, k_signs, k_rnorm, k_norm, v_mse, v_norm


def _compressed_fp16_elems(head_dim: int, b_mse: int, b_qjl: int,
                           val_bits: int) -> int:
    layout = _CompressedLayout(head_dim, b_mse, b_qjl, val_bits)
    return layout.fp16_elems


# ═══════════════════════════════════════════════════════════════════════════
# Metadata
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TurboQuantMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor   # [num_reqs + 1]
    seq_lens: torch.Tensor          # [num_reqs]
    block_table: torch.Tensor       # [num_reqs, max_blocks]
    slot_mapping: torch.Tensor      # [num_actual_tokens]
    causal: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# Metadata builder
# ═══════════════════════════════════════════════════════════════════════════


class TurboQuantMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self, kv_cache_spec: "AttentionSpec",
                 layer_names: list[str], vllm_config: "VllmConfig",
                 device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> TurboQuantMetadata:
        m = common_attn_metadata
        return TurboQuantMetadata(
            num_actual_tokens=m.num_actual_tokens,
            max_query_len=m.max_query_len,
            query_start_loc=m.query_start_loc,
            seq_lens=m.seq_lens,
            block_table=m.block_table_tensor,
            slot_mapping=m.slot_mapping,
            causal=m.causal,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Attention backend (registered as AttentionBackendEnum.CUSTOM)
# ═══════════════════════════════════════════════════════════════════════════


class TurboQuantAttentionBackend(AttentionBackend):
    """vLLM v1 attention backend with TurboQuant KV cache compression.

    Stores ~4× less KV cache memory than FP16 by packing compressed
    indices and norms into the cache tensor directly.

    CUDAGraph support: the fused decode path uses Triton kernels with
    pre-computed grids and no Python-level data-dependent branching,
    making it safe for CUDAGraph capture.  During capture, KV storage
    is skipped (key/value are None) and the fused decode kernel runs
    with the captured seq_lens/block_table buffers that vLLM updates
    in-place between replays.
    """

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True
    # CUDAGraph-safe: centroid scalars are cached as Python floats at
    # init (no .item()), BLOCK_N is fixed at 64 (no seq_lens.max()),
    # and the per-request fallback loop (prefill) is never captured.
    use_cudagraph: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16,
    ]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

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


# ═══════════════════════════════════════════════════════════════════════════
# Attention implementation
# ═══════════════════════════════════════════════════════════════════════════


class TurboQuantAttentionImpl(AttentionImpl):
    """Attention with TurboQuant-compressed KV cache.

    Keys use TurboQuantProd  (2-bit MSE + 1-bit QJL → unbiased IP).
    Values use TurboQuantMSE (3-bit MSE → errors average under softmax).

    All compressed data lives in the ``kv_cache`` tensor provided by vLLM.
    No auxiliary buffers are needed.
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

        # TQ params
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

        # Triton decode dispatch
        self._use_triton = os.environ.get("TQ_USE_TRITON", "0") == "1"
        self._num_kv_splits = int(os.environ.get("TQ_NUM_KV_SPLITS", "8"))

        # Lazy-init quantizers (need CUDA device)
        self._key_q: TurboQuantProd | None = None
        self._val_q: TurboQuantMSE | None = None
        self._init_device: torch.device | None = None

    # ── lazy quantizer creation ──────────────────────────────────────

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
            self.layer_idx = TurboQuantAttentionImpl._layer_counter
            TurboQuantAttentionImpl._layer_counter += 1

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

        # Cache centroid scalars as plain Python floats so kernel launchers
        # never call .item() (which forces GPU→CPU sync and breaks CUDAGraph).
        kc = self._key_centroids.float()
        vc = self._val_centroids.float()
        self._key_centroid_scalars = tuple(float(kc[i]) for i in range(len(kc)))
        self._val_centroid_scalars = tuple(float(vc[i]) for i in range(len(vc)))

    # ── forward ──────────────────────────────────────────────────────

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

        # During profile/dummy runs, attn_metadata is None — return zeros
        if attn_metadata is None:
            if output is None:
                output = torch.zeros(
                    query.shape[0], self.num_heads, self.head_size,
                    dtype=query.dtype, device=device)
            return output

        self._ensure_quantizers(device, layer)

        N = attn_metadata.num_actual_tokens
        num_reqs = attn_metadata.seq_lens.shape[0]
        # kv_cache: (num_blocks, block_size, num_kv_heads, fp16_elems)
        block_size = kv_cache.shape[1]

        if output is None:
            output = torch.zeros(
                query.shape[0], self.num_heads, self.head_size,
                dtype=query.dtype, device=device)

        # 1) Compress and store new KV tokens
        #    During CUDAGraph replay key/value are None — skip storage.
        has_raw_kv = key is not None and value is not None
        if has_raw_kv:
            self._store_compressed(
                key[:N], value[:N], kv_cache,
                attn_metadata.slot_mapping[:N], block_size)

        # 2) Fused decode fast-path (all requests are single-token decode)
        #    This path is CUDAGraph-safe: no .item() calls, no Python
        #    branching on tensor values, deterministic kernel grid.
        if attn_metadata.max_query_len == 1 and self._use_triton:
            fused = self._fused_decode(
                query[:N], kv_cache, attn_metadata, output)
            if fused is not None:
                return fused

        # 3) Compute attention per request (prefill / fallback)
        #    NOT CUDAGraph-safe (uses .item()).  Only reached during
        #    prefill or when fused decode is unavailable.
        for ri in range(num_reqs):
            qs = attn_metadata.query_start_loc[ri].item()
            qe = attn_metadata.query_start_loc[ri + 1].item()
            q_len = qe - qs
            seq_len = attn_metadata.seq_lens[ri].item()
            if q_len == 0 or seq_len == 0:
                continue

            q = query[qs:qe]  # (q_len, nh, hd)

            # 3a) SDPA prefill: use raw K/V with memory-efficient attention.
            #     FlashAttention uses O(1) memory vs O(Q*S) for the manual
            #     path, preventing OOM on long contexts.
            #     Falls back to manual path for logit soft-capping (Gemma)
            #     or when raw K/V are unavailable (CUDAGraph replay).
            if has_raw_kv and q_len > 1 and self.logits_soft_cap is None:
                k_new = key[qs:qe]    # (q_len, nkh, hd)
                v_new = value[qs:qe]  # (q_len, nkh, hd)

                cached_len = seq_len - q_len
                if cached_len > 0:
                    k_cached, v_cached = self._decompress_kv_from_cache(
                        kv_cache, attn_metadata.block_table[ri],
                        cached_len, block_size)
                    k_full = torch.cat([k_cached, k_new], dim=0)
                    v_full = torch.cat([v_cached, v_new], dim=0)
                else:
                    k_full = k_new
                    v_full = v_new

                out = self._sdpa_prefill(
                    q, k_full, v_full, causal=attn_metadata.causal)

                if output.dim() == 3:
                    output[qs:qe] = out.to(output.dtype)
                else:
                    output[qs:qe] = out.reshape(q_len, -1).to(output.dtype)
                continue

            # 3b) Fallback: TQ decompress + manual attention
            n_blk = (seq_len + block_size - 1) // block_size
            blk_ids = attn_metadata.block_table[ri, :n_blk]
            comp = kv_cache[blk_ids]   # (n_blk, bs, nkh, fp16)
            comp = comp.reshape(-1, self.num_kv_heads, self._layout.fp16_elems)
            comp = comp[:seq_len]      # (seq_len, nkh, fp16)

            # Byte view: (seq_len, nkh, total_bytes)
            comp_bytes = comp.contiguous().view(torch.uint8).reshape(
                seq_len, self.num_kv_heads, self._layout.total_bytes)

            pos_offset = seq_len - q_len  # first query's position in sequence

            self._attn_one_request(
                q, comp_bytes, seq_len, q_len, pos_offset,
                output, qs, attn_metadata.causal)

        return output

    # ── fused decode (all requests q_len=1) ───────────────────────────

    def _fused_decode(
        self, queries: torch.Tensor, kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        output: torch.Tensor,
    ) -> "torch.Tensor | None":
        """Try the fused Triton decode path.  Returns None on fallback."""

        out = fused_decode_attention(
            queries, kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self._layout,
            key_centroids=self._key_centroids.float(),
            val_centroids=self._val_centroids.float(),
            key_pi_t=self._key_q.mse.Pi.T,
            val_pi=self._val_q.Pi,
            s_t=self._key_q.S.T,
            heads_per_kv=self._heads_per_kv,
            qjl_dim=self._key_q.qjl_dim,
            sm_scale=self.scale,
            rotation=self._rotation,
            key_sigma=self._key_sigma,
            val_sigma=self._val_sigma,
            key_centroid_scalars=self._key_centroid_scalars,
            val_centroid_scalars=self._val_centroid_scalars,
        )
        if out is None:
            return None

        N = queries.shape[0]
        if output.dim() == 3:
            output[:N] = out.to(output.dtype)
        else:
            output[:N] = out.reshape(N, -1).to(output.dtype)
        return output

    # ── compress & store ─────────────────────────────────────────────

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

    # ── SDPA prefill (memory-efficient) ────────────────────────────────

    def _sdpa_prefill(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Prefill attention via F.scaled_dot_product_attention.

        Uses FlashAttention or the memory-efficient SDPA backend which
        requires O(1) extra memory for the attention matrix — versus
        O(Q*S*H) for the manual einsum path.  This prevents OOM on long
        contexts where the manual path would materialise a huge score
        tensor.

        When Q < S (prefix-cache hit / chunked prefill), PyTorch SDPA's
        ``is_causal=True`` applies a **top-left** aligned causal mask,
        which is wrong — the queries correspond to the *last* Q
        positions.  In that case we construct an explicit bottom-right
        aligned boolean mask instead.

        Args:
            q: (Q, num_heads, head_dim)
            k: (S, num_kv_heads, head_dim)
            v: (S, num_kv_heads, head_dim)
            causal: whether to apply causal mask

        Returns:
            (Q, num_heads, head_dim)
        """
        Q_len = q.shape[0]
        S_len = k.shape[0]

        # SDPA expects (batch, heads, seq, dim)
        q_sdpa = q.permute(1, 0, 2).unsqueeze(0)   # (1, NH, Q, HD)
        k_sdpa = k.permute(1, 0, 2).unsqueeze(0)   # (1, NKH, S, HD)
        v_sdpa = v.permute(1, 0, 2).unsqueeze(0)   # (1, NKH, S, HD)

        # Expand KV heads for GQA
        if self._heads_per_kv > 1:
            k_sdpa = k_sdpa.repeat_interleave(self._heads_per_kv, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(self._heads_per_kv, dim=1)

        use_causal = causal and Q_len > 1

        if use_causal and Q_len < S_len:
            # Bottom-right aligned causal mask for chunked prefill.
            # Query i sits at sequence position (S-Q+i) and may attend
            # to KV positions 0 .. (S-Q+i).
            pos_offset = S_len - Q_len
            q_pos = torch.arange(Q_len, device=q.device) + pos_offset
            k_pos = torch.arange(S_len, device=q.device)
            mask = k_pos[None, :] <= q_pos[:, None]   # (Q, S) bool
            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=mask,
                scale=self.scale,
            )
        else:
            # Q == S (full prefill) — is_causal top-left == bottom-right
            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                is_causal=use_causal,
                scale=self.scale,
            )
        # (1, NH, Q, HD) → (Q, NH, HD)
        return out.squeeze(0).permute(1, 0, 2)

    def _decompress_kv_from_cache(
        self, kv_cache: torch.Tensor, block_table_row: torch.Tensor,
        num_tokens: int, block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress prefix-cached KV tokens to dense tensors for SDPA.

        Uses MSE reconstruction only (no QJL correction, which is
        query-dependent and cannot be pre-applied).  MSE at 2-bit gives
        a good approximation — prefill quality on the cached prefix is
        not as critical as decode accuracy.

        Returns:
            (keys, values) each shaped (num_tokens, num_kv_heads, head_dim)
        """
        n_blk = (num_tokens + block_size - 1) // block_size
        blk_ids = block_table_row[:n_blk]
        comp = kv_cache[blk_ids]
        comp = comp.reshape(-1, self.num_kv_heads, self._layout.fp16_elems)
        comp = comp[:num_tokens]

        comp_bytes = comp.contiguous().view(torch.uint8).reshape(
            num_tokens, self.num_kv_heads, self._layout.total_bytes)

        S = num_tokens
        nkh = self.num_kv_heads
        hd = self.head_size

        flat_bytes = comp_bytes.reshape(S * nkh, self._layout.total_bytes)
        (km_idx, _, _,
         k_norm, vm_idx, v_norm) = self._layout.unpack(flat_bytes)
        km_idx = km_idx.reshape(S, nkh, hd)
        k_norm = k_norm.reshape(S, nkh)
        vm_idx = vm_idx.reshape(S, nkh, hd)
        v_norm = v_norm.reshape(S, nkh)

        # Keys: k ≈ k_norm * inverse_rotate(centroids[mse_idx])
        k_rotated = self._key_centroids[km_idx.reshape(-1, hd)]
        if self._rotation == "wht":
            k_dir = wht_unrotate(k_rotated.float(), self._key_sigma).half()
        else:
            k_dir = (k_rotated @ self._key_Pi).half()
        k_full = k_dir.reshape(S, nkh, hd) * k_norm.half().unsqueeze(-1)

        # Values: v ≈ v_norm * inverse_rotate(centroids[val_idx])
        v_rotated = self._val_centroids[vm_idx.reshape(-1, hd)]
        if self._rotation == "wht":
            v_dir = wht_unrotate(v_rotated.float(), self._val_sigma).half()
        else:
            v_dir = (v_rotated @ self._val_Pi).half()
        v_full = v_dir.reshape(S, nkh, hd) * v_norm.half().unsqueeze(-1)

        return k_full, v_full

    # ── attention for one request (batched across all KV heads) ─────

    def _attn_one_request(
        self, queries: torch.Tensor, comp_bytes: torch.Tensor,
        seq_len: int, q_len: int, pos_offset: int,
        output: torch.Tensor, out_offset: int, causal: bool,
    ) -> None:
        # Triton decode path for single-token generation
        if q_len == 1 and self._use_triton:
            out = turboquant_decode_attention(
                queries, comp_bytes, self._layout,
                key_centroids=self._key_centroids.float(),
                val_centroids=self._val_centroids.float(),
                key_pi=self._key_q.mse.Pi,
                key_pi_t=self._key_q.mse.Pi.T,
                val_pi=self._val_q.Pi,
                s_t=self._key_q.S.T,
                heads_per_kv=self._heads_per_kv,
                qjl_dim=self._key_q.qjl_dim,
                sm_scale=self.scale,
                causal=causal,
                pos_offset=pos_offset,
                num_kv_splits=self._num_kv_splits,
                use_triton=True,
                rotation=self._rotation,
                key_sigma=self._key_sigma,
                val_sigma=self._val_sigma,
            )
            if output.dim() == 3:
                output[out_offset:out_offset + 1] = out.to(output.dtype)
            else:
                output[out_offset:out_offset + 1] = out.reshape(
                    1, self.num_kv_heads * self._heads_per_kv * self.head_size
                ).to(output.dtype)
            return

        hd = self.head_size
        nkh = self.num_kv_heads
        hpkv = self._heads_per_kv
        S, Q = seq_len, q_len
        dev = queries.device

        # ── Unpack ALL heads at once ──
        flat_bytes = comp_bytes.reshape(S * nkh, self._layout.total_bytes)
        (km_idx, k_signs, k_rnorm,
         k_norm, vm_idx, v_norm) = self._layout.unpack(flat_bytes)
        km_idx = km_idx.reshape(S, nkh, hd)
        k_signs = k_signs.reshape(S, nkh, hd)
        k_rnorm = k_rnorm.reshape(S, nkh)
        k_norm = k_norm.reshape(S, nkh)
        vm_idx = vm_idx.reshape(S, nkh, hd)
        v_norm = v_norm.reshape(S, nkh)

        # ── Dequantize ALL heads in FP16 ──
        # codebook lookup + inverse rotation, all in half precision
        k_rotated = self._key_centroids[km_idx.reshape(-1, hd)]   # (S*nkh, D)
        if self._rotation == "wht":
            k_mse = wht_unrotate(k_rotated.float(), self._key_sigma).half().reshape(S, nkh, hd)
        else:
            k_mse = (k_rotated @ self._key_Pi).reshape(S, nkh, hd)
        v_rotated = self._val_centroids[vm_idx.reshape(-1, hd)]
        if self._rotation == "wht":
            v_full = wht_unrotate(v_rotated.float(), self._val_sigma).half().reshape(S, nkh, hd)
        else:
            v_full = (v_rotated @ self._val_Pi).reshape(S, nkh, hd)
        v_full = v_full * v_norm.half().unsqueeze(-1)

        # ── Batched attention scores via einsum ──
        # queries: (Q, NH, D) → (Q, nkh, hpkv, D) for GQA
        # Use half precision for speed — sufficient for attention scores
        q_f = queries.half().reshape(Q, nkh, hpkv, hd)
        k_mse = k_mse.half()
        k_signs = k_signs.half()

        # Term 1: <q, k_mse>  → (Q, nkh, hpkv, S)
        t1 = torch.einsum("qghd,sgd->qghs", q_f, k_mse)

        # Term 2: QJL correction
        q_proj = q_f @ self._S_T                               # (Q, nkh, hpkv, m)
        t2_raw = torch.einsum("qghm,sgm->qghs", q_proj, k_signs)
        corr = math.sqrt(math.pi / 2) / self._key_q.qjl_dim
        t2 = corr * t2_raw * k_rnorm.half().permute(1, 0)[None, :, None, :]

        scores = (t1 + t2) * k_norm.half().permute(1, 0)[None, :, None, :]
        scores = scores * self.scale                           # (Q, nkh, hpkv, S)

        # Logit soft capping (Gemma-style)
        if self.logits_soft_cap is not None:
            scores = (scores.float() / self.logits_soft_cap).tanh() * self.logits_soft_cap

        # Causal mask
        if causal and Q > 1:
            pos = torch.arange(S, device=dev)
            qpos = torch.arange(Q, device=dev) + pos_offset
            mask = pos[None, :] > qpos[:, None]                # (Q, S)
            scores.masked_fill_(mask[:, None, None, :], float("-inf"))

        # Softmax in float32 for numerical stability, then back to half
        w = F.softmax(scores.float(), dim=-1).half()           # (Q, nkh, hpkv, S)
        out = torch.einsum("qghs,sgd->qghd", w, v_full.half())  # (Q, nkh, hpkv, D)
        out = out.reshape(Q, nkh * hpkv, hd)                  # (Q, NH, D)

        # Write
        if output.dim() == 3:
            output[out_offset:out_offset + Q] = out.to(output.dtype)
        else:
            output[out_offset:out_offset + Q] = out.reshape(
                Q, nkh * hpkv * hd).to(output.dtype)
