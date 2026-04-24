"""TurboQuant decode wrapper.

The wrapper exposes the rotated-domain attention path independently from the
vLLM backend so it can be tested directly. Decode mode uses the two-stage
split-KV contract from the Triton plan; prefill keeps a straightforward
PyTorch path.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional, Sequence

import torch
import torch.nn.functional as F

from turboquant import wht_rotate, wht_unrotate
from vllm_plugin.triton_kernels import (
    TRITON_AVAILABLE,
    _tq_decode_stage1,
    _tq_decode_stage2,
    _tq_fused_decode,
    _tq_fused_decode_prerot,
)

if TYPE_CHECKING:
    from vllm_plugin.attention import _CompressedLayout


def prerotate_queries(
    queries: torch.Tensor,
    key_pi_t: torch.Tensor,
    s_t: torch.Tensor,
    rotation: str = "haar",
    key_sigma: Optional[torch.Tensor] = None,
    *,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rotated queries and QJL sketches for the decode kernels."""

    head_dim = queries.shape[-1]
    qjl_dim = s_t.shape[-1]
    # Fast path: fused Triton kernel for WHT rotation + QJL sketch.
    # Requires qjl_dim == head_dim (kernel allocates sketch at head_dim width)
    # and head_dim <= 256 (FWHT butterfly is unrolled to 8 steps max).
    if (
        use_triton
        and rotation == "wht"
        and key_sigma is not None
        and TRITON_AVAILABLE
        and queries.is_cuda
        and qjl_dim == head_dim
        and head_dim <= 256
    ):
        from vllm_plugin.triton_kernels import _prerotate_triton

        return _prerotate_triton(queries, key_sigma, s_t)

    q_float = queries.float()
    if rotation == "wht":
        q_rot = wht_rotate(q_float, key_sigma)
    else:
        q_rot = q_float @ key_pi_t.float()
    return q_rot, q_float @ s_t.float()


def turboquant_decode_attention_pytorch(
    queries: torch.Tensor,
    comp_bytes: torch.Tensor,
    layout: "_CompressedLayout",
    *,
    key_centroids: torch.Tensor,
    val_centroids: torch.Tensor,
    key_pi: torch.Tensor,
    key_pi_t: torch.Tensor,
    val_pi: torch.Tensor,
    s_t: torch.Tensor,
    heads_per_kv: int,
    qjl_dim: int,
    sm_scale: float,
    causal: bool,
    pos_offset: int,
    rotation: str = "haar",
    key_sigma: Optional[torch.Tensor] = None,
    val_sigma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference attention path using rotated-domain score/value math."""

    del key_pi

    seq_len, num_kv_heads = comp_bytes.shape[:2]
    q_len = queries.shape[0]
    head_dim = layout.head_dim

    flat_bytes = comp_bytes.reshape(seq_len * num_kv_heads, layout.total_bytes)
    (km_idx, k_signs, k_rnorm,
     k_norm, vm_idx, v_norm) = layout.unpack(flat_bytes)
    km_idx = km_idx.reshape(seq_len, num_kv_heads, head_dim)
    k_signs = k_signs.reshape(seq_len, num_kv_heads, head_dim)
    k_rnorm = k_rnorm.reshape(seq_len, num_kv_heads)
    k_norm = k_norm.reshape(seq_len, num_kv_heads)
    vm_idx = vm_idx.reshape(seq_len, num_kv_heads, head_dim)
    v_norm = v_norm.reshape(seq_len, num_kv_heads)

    k_rot = key_centroids[km_idx.reshape(-1, head_dim)].reshape(
        seq_len, num_kv_heads, head_dim)
    v_rot = val_centroids[vm_idx.reshape(-1, head_dim)].reshape(
        seq_len, num_kv_heads, head_dim)
    v_rot = v_rot * v_norm.unsqueeze(-1)

    q_view = queries.reshape(q_len, num_kv_heads, heads_per_kv, head_dim)
    q_rot, q_sketch = prerotate_queries(
        q_view,
        key_pi_t,
        s_t,
        rotation=rotation,
        key_sigma=key_sigma,
        use_triton=False,
    )

    mse_scores = torch.einsum("qghd,sgd->qghs", q_rot, k_rot.float())
    qjl_scores = torch.einsum("qghd,sgd->qghs", q_sketch, k_signs.float())
    qjl_corr = math.sqrt(math.pi / 2.0) / qjl_dim
    scores = mse_scores + qjl_corr * qjl_scores * k_rnorm.float().permute(
        1, 0)[None, :, None, :]
    scores = scores * k_norm.float().permute(1, 0)[None, :, None, :]
    scores = scores * sm_scale

    if causal and q_len > 1:
        pos = torch.arange(seq_len, device=queries.device)
        qpos = torch.arange(q_len, device=queries.device) + pos_offset
        mask = pos[None, :] > qpos[:, None]
        scores.masked_fill_(mask[:, None, None, :], float("-inf"))

    weights = F.softmax(scores, dim=-1)
    out_rot = torch.einsum("qghs,sgd->qghd", weights, v_rot.float())
    if rotation == "wht":
        out = wht_unrotate(out_rot.float(), val_sigma)
    else:
        out = out_rot @ val_pi.float()
    return out.reshape(q_len, num_kv_heads * heads_per_kv, head_dim).to(
        dtype=queries.dtype)


def turboquant_decode_attention(
    queries: torch.Tensor,
    comp_bytes: torch.Tensor,
    layout: "_CompressedLayout",
    *,
    key_centroids: torch.Tensor,
    val_centroids: torch.Tensor,
    key_pi: torch.Tensor,
    key_pi_t: torch.Tensor,
    val_pi: torch.Tensor,
    s_t: torch.Tensor,
    heads_per_kv: int,
    qjl_dim: int,
    sm_scale: float,
    causal: bool,
    pos_offset: int,
    num_kv_splits: int = 8,
    use_triton: bool = False,
    rotation: str = "haar",
    key_sigma: Optional[torch.Tensor] = None,
    val_sigma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode wrapper with WHT/Haar rotation support."""

    del key_pi, causal, pos_offset

    if queries.shape[0] != 1:
        raise ValueError("turboquant_decode_attention only supports decode (q_len=1)")

    num_kv_heads = comp_bytes.shape[1]
    head_dim = layout.head_dim
    q_view = queries.reshape(1, num_kv_heads, heads_per_kv, head_dim)
    q_rot, q_sketch = prerotate_queries(
        q_view,
        key_pi_t,
        s_t,
        rotation=rotation,
        key_sigma=key_sigma,
        use_triton=use_triton,
    )

    partial_acc, partial_lse = _tq_decode_stage1(
        q_rot.squeeze(0),
        q_sketch.squeeze(0),
        comp_bytes,
        layout,
        key_centroids=key_centroids,
        val_centroids=val_centroids,
        qjl_corr=math.sqrt(math.pi / 2.0) / qjl_dim,
        sm_scale=sm_scale,
        num_kv_splits=num_kv_splits,
        use_triton=use_triton,
    )
    out = _tq_decode_stage2(
        partial_acc, partial_lse, val_pi, use_triton=use_triton,
        rotation=rotation, val_sigma=val_sigma)
    return out.reshape(1, num_kv_heads * heads_per_kv, head_dim).to(
        dtype=queries.dtype)


def fused_decode_attention(
    queries: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    layout: "_CompressedLayout",
    *,
    key_centroids: torch.Tensor | Sequence[float],
    val_centroids: torch.Tensor | Sequence[float],
    key_pi_t: torch.Tensor,
    val_pi: torch.Tensor,
    s_t: torch.Tensor,
    heads_per_kv: int,
    qjl_dim: int,
    sm_scale: float,
    rotation: str = "wht",
    key_sigma: Optional[torch.Tensor] = None,
    val_sigma: Optional[torch.Tensor] = None,
    max_seq_len: Optional[int] = None,
) -> "torch.Tensor | None":
    """Fused decode: pre-rotate queries + score/accumulate/unrotate in one launch.

    Returns the output tensor ``(num_reqs, num_heads, head_dim)`` on success,
    or ``None`` when the Triton fused path is unavailable (caller should fall
    back to the per-request loop).
    """

    num_reqs = queries.shape[0]
    num_kv_heads = kv_cache.shape[2]
    head_dim = layout.head_dim
    qjl_corr = math.sqrt(math.pi / 2.0) / qjl_dim
    if max_seq_len is None:
        max_seq_len = int(os.environ.get("TQ_MAX_SEQ_LEN", "4096"))
    if isinstance(key_centroids, torch.Tensor):
        key_centroids_values = key_centroids.float().tolist()
    else:
        key_centroids_values = list(float(x) for x in key_centroids)
    if isinstance(val_centroids, torch.Tensor):
        val_centroids_values = val_centroids.float().tolist()
    else:
        val_centroids_values = list(float(x) for x in val_centroids)

    q_view = queries.reshape(num_reqs, num_kv_heads, heads_per_kv, head_dim)

    # Fast path: fused prerotation + decode in one kernel (no intermediates)
    out = _tq_fused_decode_prerot(
        q_view, key_sigma, s_t,
        kv_cache, block_table, seq_lens, layout,
        key_centroids=key_centroids_values,
        val_centroids=val_centroids_values,
        val_pi=val_pi,
        val_sigma=val_sigma,
        qjl_corr=qjl_corr,
        sm_scale=sm_scale,
        rotation=rotation,
        block_n=64 if max_seq_len >= 64 else 32,
    )

    # Fallback: separate prerotation + fused decode
    if out is None:
        q_rot, q_sketch = prerotate_queries(
            q_view, key_pi_t, s_t, rotation=rotation, key_sigma=key_sigma)
        out = _tq_fused_decode(
            q_rot, q_sketch, kv_cache, block_table, seq_lens, layout,
            key_centroids=key_centroids_values,
            val_centroids=val_centroids_values,
            val_pi=val_pi,
            val_sigma=val_sigma,
            qjl_corr=qjl_corr,
            sm_scale=sm_scale,
            rotation=rotation,
            block_n=64 if max_seq_len >= 64 else 32,
        )

    if out is None:
        return None

    return out.reshape(num_reqs, num_kv_heads * heads_per_kv, head_dim).to(
        dtype=queries.dtype)
