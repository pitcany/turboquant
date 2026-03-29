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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure turboquant modules are importable
# ---------------------------------------------------------------------------
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from turboquant import TurboQuantProd, TurboQuantMSE
from vllm_plugin.config import TurboQuantConfig
from vllm_plugin.triton_wrapper import turboquant_decode_attention

# ---------------------------------------------------------------------------
# vLLM v1 imports
# ---------------------------------------------------------------------------
from vllm.v1.attention.backend import (
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


class _CompressedLayout:
    """Byte-level layout of compressed KV data for one token, one KV head.

    For head_dim=128, b_mse=2, b_qjl=1, val_bits=3 the layout is::

        [0..31]    key MSE indices   (128×2 bit = 32 bytes, 2-bit packed)
        [32..47]   key QJL signs     (128×1 bit = 16 bytes, 1-bit packed)
        [48..49]   key residual norm (float16)
        [50..51]   key original norm (float16)
        [52..115]  val MSE indices   (128×4 bit = 64 bytes, nibble packed)
        [116..117] val original norm (float16)
        ─────────  118 bytes total = 59 float16 elements
    """

    def __init__(self, head_dim: int, key_mse_bits: int, key_qjl_bits: int,
                 val_mse_bits: int):
        self.head_dim = head_dim
        self.key_mse_bits = key_mse_bits
        self.val_mse_bits = val_mse_bits

        # Key MSE: 2-bit packed
        self.km_off = 0
        self.km_len = head_dim * key_mse_bits // 8
        # Key QJL signs: 1-bit packed
        self.kq_off = self.km_off + self.km_len
        self.kq_len = head_dim * key_qjl_bits // 8
        # Key residual norm (fp16)
        self.kr_off = self.kq_off + self.kq_len
        # Key original norm (fp16)
        self.kn_off = self.kr_off + 2
        # Value MSE: nibble packed (val_mse_bits ≤ 4)
        self.vm_off = self.kn_off + 2
        self.vm_len = head_dim * 4 // 8  # 4-bit nibbles
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
        buf[:, self.km_off:self.km_off + self.km_len] = _pack_2bit(k_mse)
        buf[:, self.kq_off:self.kq_off + self.kq_len] = _pack_1bit(k_signs)
        buf[:, self.kr_off:self.kr_off + 2] = (
            k_rnorm.to(torch.float16).unsqueeze(-1).view(torch.uint8))
        buf[:, self.kn_off:self.kn_off + 2] = (
            k_norm.to(torch.float16).unsqueeze(-1).view(torch.uint8))
        buf[:, self.vm_off:self.vm_off + self.vm_len] = _pack_4bit(v_mse)
        buf[:, self.vn_off:self.vn_off + 2] = (
            v_norm.to(torch.float16).unsqueeze(-1).view(torch.uint8))
        return buf

    def unpack(self, buf: torch.Tensor):
        """Unpack (n, total_bytes) uint8 → component tensors."""
        d = self.head_dim
        k_mse = _unpack_2bit(buf[:, self.km_off:self.km_off + self.km_len], d)
        k_signs = _unpack_1bit(buf[:, self.kq_off:self.kq_off + self.kq_len], d)
        k_rnorm = buf[:, self.kr_off:self.kr_off + 2].contiguous().view(
            torch.float16).squeeze(-1).float()
        k_norm = buf[:, self.kn_off:self.kn_off + 2].contiguous().view(
            torch.float16).squeeze(-1).float()
        v_mse = _unpack_4bit(buf[:, self.vm_off:self.vm_off + self.vm_len], d)
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
    """

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True
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
        b_mse = int(os.environ.get("TQ_B_MSE", "2"))
        b_qjl = int(os.environ.get("TQ_B_QJL", "1"))
        val_bits = b_mse + b_qjl
        fp16 = _compressed_fp16_elems(head_size, b_mse, b_qjl, val_bits)
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

        # Deterministic seed per layer
        self.layer_idx = TurboQuantAttentionImpl._layer_counter
        TurboQuantAttentionImpl._layer_counter += 1

        # TQ params
        cfg = TurboQuantConfig(
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=head_size,
        )
        self._b_mse = cfg.b_mse
        self._b_qjl = cfg.b_qjl
        self._b_total = cfg.b_total
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

    def _ensure_quantizers(self, device: torch.device) -> None:
        if self._key_q is not None and self._init_device == device:
            return
        self._key_q = TurboQuantProd(
            self.head_size, self._b_total,
            seed=self.layer_idx * 1000, device=str(device))
        self._val_q = TurboQuantMSE(
            self.head_size, self._b_total,
            seed=self.layer_idx * 1000 + 500, device=str(device))
        self._init_device = device

        # Cache half-precision copies of rotation matrices for faster dequant
        self._key_Pi = self._key_q.mse.Pi.half()
        self._key_centroids = self._key_q.mse.centroids.half()
        self._val_Pi = self._val_q.Pi.half()
        self._val_centroids = self._val_q.centroids.half()
        # Cache half-precision QJL projection
        self._S_T = self._key_q.S.T.half()

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

        self._ensure_quantizers(device)

        N = attn_metadata.num_actual_tokens
        num_reqs = attn_metadata.seq_lens.shape[0]
        # kv_cache: (num_blocks, block_size, num_kv_heads, fp16_elems)
        block_size = kv_cache.shape[1]

        if output is None:
            output = torch.zeros(
                query.shape[0], self.num_heads, self.head_size,
                dtype=query.dtype, device=device)

        # 1) Compress and store new KV tokens
        if key is not None and value is not None:
            self._store_compressed(
                key[:N], value[:N], kv_cache,
                attn_metadata.slot_mapping[:N], block_size)

        # 2) Compute attention per request
        for ri in range(num_reqs):
            qs = attn_metadata.query_start_loc[ri].item()
            qe = attn_metadata.query_start_loc[ri + 1].item()
            q_len = qe - qs
            seq_len = attn_metadata.seq_lens[ri].item()
            if q_len == 0 or seq_len == 0:
                continue

            # Gather compressed KV for this request
            n_blk = (seq_len + block_size - 1) // block_size
            blk_ids = attn_metadata.block_table[ri, :n_blk]
            comp = kv_cache[blk_ids]   # (n_blk, bs, nkh, fp16)
            comp = comp.reshape(-1, self.num_kv_heads, self._layout.fp16_elems)
            comp = comp[:seq_len]      # (seq_len, nkh, fp16)

            # Byte view: (seq_len, nkh, total_bytes)
            comp_bytes = comp.contiguous().view(torch.uint8).reshape(
                seq_len, self.num_kv_heads, self._layout.total_bytes)

            q = query[qs:qe]  # (q_len, nh, hd)
            pos_offset = seq_len - q_len  # first query's position in sequence

            self._attn_one_request(
                q, comp_bytes, seq_len, q_len, pos_offset,
                output, qs, attn_metadata.causal)

        return output

    # ── compress & store ─────────────────────────────────────────────

    def _store_compressed(
        self, key: torch.Tensor, value: torch.Tensor,
        kv_cache: torch.Tensor, slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Compress K,V and write packed bytes into kv_cache.

        kv_cache shape: (num_blocks, block_size, num_kv_heads, fp16_elems)
        All KV heads processed in one batch — no Python loop.
        """
        n = key.shape[0]
        nkh = self.num_kv_heads
        blk = slot_mapping // block_size
        off = slot_mapping % block_size

        # Flatten heads: (n, nkh, D) → (n*nkh, D)
        k_flat = key.reshape(n * nkh, self.head_size).float()
        v_flat = value.reshape(n * nkh, self.head_size).float()

        # Normalize
        kn = torch.norm(k_flat, dim=-1)
        k_unit = k_flat / (kn.unsqueeze(-1) + 1e-8)
        vn = torch.norm(v_flat, dim=-1)
        v_unit = v_flat / (vn.unsqueeze(-1) + 1e-8)

        # Compress all at once
        ck = self._key_q.quantize(k_unit)
        vi = self._val_q.quantize(v_unit)

        # Pack → (n*nkh, total_bytes)
        packed = self._layout.pack(
            ck["mse_indices"], ck["qjl_signs"],
            ck["residual_norm"], kn, vi, vn)
        packed_fp16 = packed.view(kv_cache.dtype).reshape(
            n, nkh, self._layout.fp16_elems)

        # Scatter all heads at once
        kv_cache[blk, off] = packed_fp16

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
                sm_scale=1.0 / math.sqrt(self.head_size),
                causal=causal,
                pos_offset=pos_offset,
                num_kv_splits=self._num_kv_splits,
                use_triton=True,
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
        k_mse = (k_rotated @ self._key_Pi).reshape(S, nkh, hd)   # inverse rotation
        v_rotated = self._val_centroids[vm_idx.reshape(-1, hd)]
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
        scores = scores / math.sqrt(hd)                        # (Q, nkh, hpkv, S)

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
