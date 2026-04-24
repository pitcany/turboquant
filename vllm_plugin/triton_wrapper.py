"""TurboQuant decode wrapper.

The wrapper exposes the rotated-domain attention path independently from the
vLLM backend so it can be tested directly. Decode mode uses the two-stage
split-KV contract from the Triton plan; prefill keeps a straightforward
PyTorch path.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm_plugin.triton_kernels import _tq_decode_stage1, _tq_decode_stage2

if TYPE_CHECKING:
    from vllm_plugin.attention import _CompressedLayout


def prerotate_queries(
    queries: torch.Tensor,
    key_pi_t: torch.Tensor,
    s_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rotated queries and QJL sketches for the decode kernels."""

    q_float = queries.float()
    return q_float @ key_pi_t.float(), q_float @ s_t.float()


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
    q_rot, q_sketch = prerotate_queries(q_view, key_pi_t, s_t)

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
) -> torch.Tensor:
    """Decode wrapper.

    ``use_triton`` is reserved for the future specialized kernel path. The
    current implementation always executes the split-KV contract in torch,
    which keeps the entry point stable and exercises the same integration
    surface.
    """

    del key_pi, causal, pos_offset

    if queries.shape[0] != 1:
        raise ValueError("turboquant_decode_attention only supports decode (q_len=1)")

    num_kv_heads = comp_bytes.shape[1]
    head_dim = layout.head_dim
    q_view = queries.reshape(1, num_kv_heads, heads_per_kv, head_dim)
    q_rot, q_sketch = prerotate_queries(q_view, key_pi_t, s_t)

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
    out = _tq_decode_stage2(partial_acc, partial_lse, val_pi, use_triton=use_triton)
    return out.reshape(1, num_kv_heads * heads_per_kv, head_dim).to(
        dtype=queries.dtype)
