"""TurboQuant decode kernels.

This module provides two-stage split-KV decode kernels with Triton-backed CUDA
implementations and torch fallbacks for CPU or unsupported environments.
"""

from __future__ import annotations

import importlib.util
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_plugin.attention import _CompressedLayout


_HAS_TRITON = importlib.util.find_spec("triton") is not None
if _HAS_TRITON:
    import triton
    import triton.language as tl
else:
    triton = None
    tl = None


TRITON_AVAILABLE = _HAS_TRITON


def _stage1_torch(
    q_rot: torch.Tensor,
    q_sketch: torch.Tensor,
    comp_bytes: torch.Tensor,
    layout: "_CompressedLayout",
    *,
    key_centroids: torch.Tensor,
    val_centroids: torch.Tensor,
    qjl_corr: float,
    sm_scale: float,
    num_kv_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len, num_kv_heads = comp_bytes.shape[:2]
    head_dim = layout.head_dim
    num_splits = max(1, min(int(num_kv_splits), seq_len))

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

    split_size = math.ceil(seq_len / num_splits)
    partial_acc: list[torch.Tensor] = []
    partial_lse: list[torch.Tensor] = []

    for split_idx in range(num_splits):
        start = split_idx * split_size
        end = min(start + split_size, seq_len)
        k_rot_split = k_rot[start:end].float()
        k_signs_split = k_signs[start:end].float()
        k_rnorm_split = k_rnorm[start:end].float()
        k_norm_split = k_norm[start:end].float()
        v_rot_split = v_rot[start:end].float()

        mse_scores = torch.einsum("ghd,sgd->ghs", q_rot.float(), k_rot_split)
        qjl_scores = torch.einsum("ghd,sgd->ghs", q_sketch.float(), k_signs_split)
        scores = mse_scores + qjl_corr * qjl_scores * k_rnorm_split.permute(
            1, 0)[:, None, :]
        scores = scores * k_norm_split.permute(1, 0)[:, None, :]
        scores = scores * sm_scale

        split_max = scores.amax(dim=-1, keepdim=True)
        split_probs = torch.exp(scores - split_max)
        split_sum = split_probs.sum(dim=-1)
        split_out = torch.einsum(
            "ghs,sgd->ghd",
            split_probs / split_sum.unsqueeze(-1),
            v_rot_split,
        )
        partial_acc.append(split_out)
        partial_lse.append(split_max.squeeze(-1) + torch.log(split_sum))

    return torch.stack(partial_acc, dim=0), torch.stack(partial_lse, dim=0)


def _stage2_torch(
    partial_acc: torch.Tensor,
    partial_lse: torch.Tensor,
    val_pi: torch.Tensor,
    rotation: str = "haar",
    val_sigma: "torch.Tensor | None" = None,
) -> torch.Tensor:
    max_lse = partial_lse.amax(dim=0, keepdim=True)
    weights = torch.exp(partial_lse - max_lse)
    denom = weights.sum(dim=0)
    merged_rot = (partial_acc * weights.unsqueeze(-1)).sum(dim=0) / denom.unsqueeze(-1)
    if rotation == "wht":
        from turboquant import wht_unrotate
        return wht_unrotate(merged_rot.float(), val_sigma)
    return merged_rot @ val_pi.float()


if _HAS_TRITON:

    @triton.jit
    def _lookup_4(idx, c0, c1, c2, c3):
        return tl.where(idx == 0, c0,
                        tl.where(idx == 1, c1,
                                 tl.where(idx == 2, c2, c3)))


    @triton.jit
    def _lookup_8(idx, c0, c1, c2, c3, c4, c5, c6, c7):
        return tl.where(
            idx == 0, c0,
            tl.where(
                idx == 1, c1,
                tl.where(
                    idx == 2, c2,
                    tl.where(
                        idx == 3, c3,
                        tl.where(
                            idx == 4, c4,
                            tl.where(
                                idx == 5, c5,
                                tl.where(idx == 6, c6, c7),
                            ),
                        ),
                    ),
                ),
            ),
        )


    @triton.jit
    def _tq_decode_stage1_kernel(
        Q_ROT,
        Q_SKETCH,
        COMP_BYTES,
        COMP_FP16,
        KEY_CENTROIDS,
        VAL_CENTROIDS,
        PARTIAL_ACC,
        PARTIAL_LSE,
        stride_qrot_h,
        stride_qrot_g,
        stride_qsk_h,
        stride_qsk_g,
        stride_comp_s,
        stride_comp_h,
        stride_compfp_s,
        stride_compfp_h,
        stride_pacc_s,
        stride_pacc_h,
        stride_pacc_g,
        stride_plse_s,
        stride_plse_h,
        stride_plse_g,
        seq_len,
        qjl_corr,
        sm_scale,
        c0,
        c1,
        c2,
        c3,
        vc0,
        vc1,
        vc2,
        vc3,
        vc4,
        vc5,
        vc6,
        vc7,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
        KM_OFF: tl.constexpr,
        KQ_OFF: tl.constexpr,
        KR_FP16_OFF: tl.constexpr,
        KN_FP16_OFF: tl.constexpr,
        VM_OFF: tl.constexpr,
        VN_FP16_OFF: tl.constexpr,
        KM_BYTES: tl.constexpr,
        KQ_BYTES: tl.constexpr,
        VM_BYTES: tl.constexpr,
    ):
        kv_head = tl.program_id(0)
        q_group = tl.program_id(1)
        split_kv_id = tl.program_id(2)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM

        q_rot = tl.load(
            Q_ROT + kv_head * stride_qrot_h + q_group * stride_qrot_g + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        q_sketch = tl.load(
            Q_SKETCH + kv_head * stride_qsk_h + q_group * stride_qsk_g + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        kv_len_per_split = tl.cdiv(seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, seq_len)

        e_max = -float("inf")
        e_sum = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        has_tokens = split_kv_end > split_kv_start

        if has_tokens:
            offs_km = tl.arange(0, KM_BYTES)
            offs_kq = tl.arange(0, KQ_BYTES)
            offs_vm = tl.arange(0, VM_BYTES)

            for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < split_kv_end

                km_ptrs = (
                    COMP_BYTES
                    + offs_n[:, None] * stride_comp_s
                    + kv_head * stride_comp_h
                    + (KM_OFF + offs_km)[None, :]
                )
                packed_km = tl.load(
                    km_ptrs,
                    mask=mask_n[:, None],
                    other=0,
                )

                score_mse = tl.zeros([BLOCK_N], dtype=tl.float32)
                for shift_idx in range(4):
                    q_coords = offs_km * 4 + shift_idx
                    q_vals = tl.load(
                        Q_ROT + kv_head * stride_qrot_h + q_group * stride_qrot_g + q_coords,
                        mask=q_coords < HEAD_DIM,
                        other=0.0,
                    ).to(tl.float32)
                    idx = ((packed_km >> (shift_idx * 2)) & 0x3).to(tl.int32)
                    vals = _lookup_4(idx, c0, c1, c2, c3).to(tl.float32)
                    score_mse += tl.sum(vals * q_vals[None, :], axis=1)

                qq_ptrs = (
                    COMP_BYTES
                    + offs_n[:, None] * stride_comp_s
                    + kv_head * stride_comp_h
                    + (KQ_OFF + offs_kq)[None, :]
                )
                packed_qjl = tl.load(
                    qq_ptrs,
                    mask=mask_n[:, None],
                    other=0,
                )

                score_qjl = tl.zeros([BLOCK_N], dtype=tl.float32)
                for bit_idx in range(8):
                    q_coords = offs_kq * 8 + bit_idx
                    q_vals = tl.load(
                        Q_SKETCH + kv_head * stride_qsk_h + q_group * stride_qsk_g + q_coords,
                        mask=q_coords < HEAD_DIM,
                        other=0.0,
                    ).to(tl.float32)
                    bits = ((packed_qjl >> bit_idx) & 1).to(tl.float32)
                    signs = bits * 2.0 - 1.0
                    score_qjl += tl.sum(signs * q_vals[None, :], axis=1)

                residual_norm = tl.load(
                    COMP_FP16
                    + offs_n * stride_compfp_s
                    + kv_head * stride_compfp_h
                    + KR_FP16_OFF,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)
                key_norm = tl.load(
                    COMP_FP16
                    + offs_n * stride_compfp_s
                    + kv_head * stride_compfp_h
                    + KN_FP16_OFF,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)

                scores = (score_mse + qjl_corr * score_qjl * residual_norm) * key_norm
                scores = scores * sm_scale
                scores = tl.where(mask_n, scores, -float("inf"))

                vm_ptrs = (
                    COMP_BYTES
                    + offs_n[:, None] * stride_comp_s
                    + kv_head * stride_comp_h
                    + (VM_OFF + offs_vm)[None, :]
                )
                packed_vm = tl.load(
                    vm_ptrs,
                    mask=mask_n[:, None],
                    other=0,
                )
                value_norm = tl.load(
                    COMP_FP16
                    + offs_n * stride_compfp_s
                    + kv_head * stride_compfp_h
                    + VN_FP16_OFF,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)

                n_e_max = tl.maximum(tl.max(scores, axis=0), e_max)
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(scores - n_e_max)
                p = tl.where(mask_n, p, 0.0)

                acc *= re_scale

                block_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
                for nibble_idx in range(2):
                    coords = offs_vm * 2 + nibble_idx
                    idx = ((packed_vm >> (nibble_idx * 4)) & 0xF).to(tl.int32)
                    vals = _lookup_8(idx, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7).to(
                        tl.float32)
                    vals *= value_norm[:, None]
                    updates = tl.sum(p[:, None] * vals, axis=0)
                    coord_mask = coords[:, None] == offs_d[None, :]
                    block_acc += tl.sum(
                        tl.where(coord_mask, updates[:, None], 0.0),
                        axis=0,
                    )

                acc += block_acc
                e_sum = e_sum * re_scale + tl.sum(p, axis=0)
                e_max = n_e_max

        pacc_ptrs = (
            PARTIAL_ACC
            + split_kv_id * stride_pacc_s
            + kv_head * stride_pacc_h
            + q_group * stride_pacc_g
            + offs_d
        )
        tl.store(
            pacc_ptrs,
            tl.where(mask_d & has_tokens, acc / e_sum, 0.0),
            mask=mask_d,
        )

        plse_ptr = (
            PARTIAL_LSE
            + split_kv_id * stride_plse_s
            + kv_head * stride_plse_h
            + q_group * stride_plse_g
        )
        tl.store(
            plse_ptr,
            tl.where(has_tokens, e_max + tl.log(e_sum), -float("inf")),
        )


    @triton.jit
    def _tq_decode_stage2_kernel(
        PARTIAL_ACC,
        PARTIAL_LSE,
        VAL_PI,
        OUT,
        stride_pacc_s,
        stride_pacc_h,
        stride_pacc_g,
        stride_plse_s,
        stride_plse_h,
        stride_plse_g,
        stride_val_pi_row,
        stride_val_pi_col,
        stride_out_h,
        stride_out_g,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
    ):
        kv_head = tl.program_id(0)
        q_group = tl.program_id(1)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM

        e_max = -float("inf")
        e_sum = 0.0
        merged_rot = tl.zeros([BLOCK_D], dtype=tl.float32)

        for split_kv_id in range(NUM_KV_SPLITS):
            lse = tl.load(
                PARTIAL_LSE
                + split_kv_id * stride_plse_s
                + kv_head * stride_plse_h
                + q_group * stride_plse_g
            ).to(tl.float32)
            valid = lse != -float("inf")
            n_e_max = tl.where(valid, tl.maximum(lse, e_max), e_max)
            old_scale = tl.exp(e_max - n_e_max)
            split_weight = tl.where(valid, tl.exp(lse - n_e_max), 0.0)
            split_acc = tl.load(
                PARTIAL_ACC
                + split_kv_id * stride_pacc_s
                + kv_head * stride_pacc_h
                + q_group * stride_pacc_g
                + offs_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)

            merged_rot = merged_rot * old_scale + split_acc * split_weight
            e_sum = e_sum * old_scale + split_weight
            e_max = n_e_max

        merged_rot = tl.where(mask_d, merged_rot / e_sum, 0.0)
        val_pi = tl.load(
            VAL_PI + offs_d[:, None] * stride_val_pi_row + offs_d[None, :] * stride_val_pi_col,
            mask=mask_d[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(merged_rot[:, None] * val_pi, axis=0)
        tl.store(
            OUT + kv_head * stride_out_h + q_group * stride_out_g + offs_d,
            out,
            mask=mask_d,
        )


def _stage1_triton(
    q_rot: torch.Tensor,
    q_sketch: torch.Tensor,
    comp_bytes: torch.Tensor,
    layout: "_CompressedLayout",
    *,
    key_centroids: torch.Tensor,
    val_centroids: torch.Tensor,
    qjl_corr: float,
    sm_scale: float,
    num_kv_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len, num_kv_heads = comp_bytes.shape[:2]
    heads_per_kv = q_rot.shape[1]
    num_splits = max(1, min(int(num_kv_splits), seq_len))
    block_d = triton.next_power_of_2(layout.head_dim)
    block_n = 64 if seq_len >= 64 else 32

    comp_bytes = comp_bytes.contiguous()
    comp_fp16 = comp_bytes.view(torch.float16).reshape(
        seq_len, num_kv_heads, layout.fp16_elems).contiguous()
    q_rot_f = q_rot.float().contiguous()
    q_sketch_f = q_sketch.float().contiguous()
    key_centroids_f = key_centroids.float().contiguous()
    val_centroids_f = val_centroids.float().contiguous()

    partial_acc = torch.empty(
        (num_splits, num_kv_heads, heads_per_kv, layout.head_dim),
        dtype=torch.float32,
        device=comp_bytes.device,
    )
    partial_lse = torch.empty(
        (num_splits, num_kv_heads, heads_per_kv),
        dtype=torch.float32,
        device=comp_bytes.device,
    )

    grid = (num_kv_heads, heads_per_kv, num_splits)
    _tq_decode_stage1_kernel[grid](
        q_rot_f,
        q_sketch_f,
        comp_bytes,
        comp_fp16,
        key_centroids_f,
        val_centroids_f,
        partial_acc,
        partial_lse,
        q_rot_f.stride(0),
        q_rot_f.stride(1),
        q_sketch_f.stride(0),
        q_sketch_f.stride(1),
        comp_bytes.stride(0),
        comp_bytes.stride(1),
        comp_fp16.stride(0),
        comp_fp16.stride(1),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        seq_len,
        qjl_corr,
        sm_scale,
        float(key_centroids_f[0].item()),
        float(key_centroids_f[1].item()),
        float(key_centroids_f[2].item()),
        float(key_centroids_f[3].item()),
        float(val_centroids_f[0].item()),
        float(val_centroids_f[1].item()),
        float(val_centroids_f[2].item()),
        float(val_centroids_f[3].item()),
        float(val_centroids_f[4].item()),
        float(val_centroids_f[5].item()),
        float(val_centroids_f[6].item()),
        float(val_centroids_f[7].item()),
        HEAD_DIM=layout.head_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        NUM_KV_SPLITS=num_splits,
        KM_OFF=layout.km_off,
        KQ_OFF=layout.kq_off,
        KR_FP16_OFF=layout.kr_off // 2,
        KN_FP16_OFF=layout.kn_off // 2,
        VM_OFF=layout.vm_off,
        VN_FP16_OFF=layout.vn_off // 2,
        KM_BYTES=layout.km_len,
        KQ_BYTES=layout.kq_len,
        VM_BYTES=layout.vm_len,
        num_warps=4,
        num_stages=2,
    )
    return partial_acc, partial_lse


def _stage2_triton(
    partial_acc: torch.Tensor,
    partial_lse: torch.Tensor,
    val_pi: torch.Tensor,
) -> torch.Tensor:
    num_splits, num_kv_heads, heads_per_kv, head_dim = partial_acc.shape
    block_d = triton.next_power_of_2(head_dim)
    out = torch.empty(
        (num_kv_heads, heads_per_kv, head_dim),
        dtype=torch.float32,
        device=partial_acc.device,
    )
    val_pi_f = val_pi.float().contiguous()

    grid = (num_kv_heads, heads_per_kv)
    _tq_decode_stage2_kernel[grid](
        partial_acc.contiguous(),
        partial_lse.contiguous(),
        val_pi_f,
        out,
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        val_pi_f.stride(0),
        val_pi_f.stride(1),
        out.stride(0),
        out.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        NUM_KV_SPLITS=num_splits,
        num_warps=4,
        num_stages=2,
    )
    return out


def _tq_decode_stage1(
    q_rot: torch.Tensor,
    q_sketch: torch.Tensor,
    comp_bytes: torch.Tensor,
    layout: "_CompressedLayout",
    *,
    key_centroids: torch.Tensor,
    val_centroids: torch.Tensor,
    qjl_corr: float,
    sm_scale: float,
    num_kv_splits: int,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The Triton kernel currently hardcodes 2-bit key / 4-bit value unpacking.
    # For other bit-widths, fall back to the generic torch path.
    triton_compatible = (
        layout.key_mse_bits == 2
        and layout.val_mse_bits <= 4
        and len(key_centroids) == 4
        and len(val_centroids) == 8
    )
    if use_triton and triton_compatible and TRITON_AVAILABLE and comp_bytes.is_cuda and q_rot.is_cuda:
        return _stage1_triton(
            q_rot,
            q_sketch,
            comp_bytes,
            layout,
            key_centroids=key_centroids,
            val_centroids=val_centroids,
            qjl_corr=qjl_corr,
            sm_scale=sm_scale,
            num_kv_splits=num_kv_splits,
        )
    return _stage1_torch(
        q_rot,
        q_sketch,
        comp_bytes,
        layout,
        key_centroids=key_centroids,
        val_centroids=val_centroids,
        qjl_corr=qjl_corr,
        sm_scale=sm_scale,
        num_kv_splits=num_kv_splits,
    )


def _tq_decode_stage2(
    partial_acc: torch.Tensor,
    partial_lse: torch.Tensor,
    val_pi: torch.Tensor,
    *,
    use_triton: bool = True,
    rotation: str = "haar",
    val_sigma: "torch.Tensor | None" = None,
) -> torch.Tensor:
    # The Triton stage2 kernel hardcodes Pi as a dense matrix multiply,
    # so WHT must use the torch path.
    if rotation == "haar" and use_triton and TRITON_AVAILABLE and partial_acc.is_cuda:
        return _stage2_triton(partial_acc, partial_lse, val_pi)
    return _stage2_torch(
        partial_acc, partial_lse, val_pi,
        rotation=rotation, val_sigma=val_sigma)
