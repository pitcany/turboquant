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

    # ═══════════════════════════════════════════════════════════════════
    # FWHT helpers
    # ═══════════════════════════════════════════════════════════════════

    @triton.jit
    def _tq_fwht_full(x, BLOCK_D: tl.constexpr):
        """Full in-register Fast Walsh-Hadamard Transform (unrolled butterfly)."""
        if BLOCK_D >= 2:
            x = _tq_fwht_step(x, 1, BLOCK_D)
        if BLOCK_D >= 4:
            x = _tq_fwht_step(x, 2, BLOCK_D)
        if BLOCK_D >= 8:
            x = _tq_fwht_step(x, 4, BLOCK_D)
        if BLOCK_D >= 16:
            x = _tq_fwht_step(x, 8, BLOCK_D)
        if BLOCK_D >= 32:
            x = _tq_fwht_step(x, 16, BLOCK_D)
        if BLOCK_D >= 64:
            x = _tq_fwht_step(x, 32, BLOCK_D)
        if BLOCK_D >= 128:
            x = _tq_fwht_step(x, 64, BLOCK_D)
        if BLOCK_D >= 256:
            x = _tq_fwht_step(x, 128, BLOCK_D)
        return x

    # ═══════════════════════════════════════════════════════════════════
    # Decode kernels
    # ═══════════════════════════════════════════════════════════════════

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

        # Vectorized address mappings: map each head dimension to its
        # packed byte offset and bit-shift within that byte.
        km_byte = offs_d // 4           # 2-bit key: 4 indices per byte
        km_shift = (offs_d % 4) * 2
        kq_byte = offs_d // 8           # 1-bit QJL: 8 signs per byte
        kq_shift = offs_d % 8
        vm_byte = offs_d // 2           # 4-bit value: 2 indices per byte
        vm_shift = (offs_d % 2) * 4

        if has_tokens:
            for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < split_kv_end
                mask_nd = mask_n[:, None] & mask_d[None, :]

                # ── Key MSE: vectorized unpack → single dot product ──
                # Load the packed byte for each (token, dim) pair directly.
                # Adjacent dims share a byte (4 dims/byte); L1 handles
                # the duplicate reads.
                km_ptrs = (
                    COMP_BYTES
                    + offs_n[:, None] * stride_comp_s
                    + kv_head * stride_comp_h
                    + KM_OFF + km_byte[None, :]
                )
                packed_km = tl.load(km_ptrs, mask=mask_nd, other=0)
                k_idx = ((packed_km >> km_shift[None, :]) & 0x3).to(tl.int32)
                k_vals = _lookup_4(k_idx, c0, c1, c2, c3).to(tl.float32)
                score_mse = tl.sum(k_vals * q_rot[None, :], axis=1)

                # ── QJL: vectorized unpack → single dot product ──
                kq_ptrs = (
                    COMP_BYTES
                    + offs_n[:, None] * stride_comp_s
                    + kv_head * stride_comp_h
                    + KQ_OFF + kq_byte[None, :]
                )
                packed_qjl = tl.load(kq_ptrs, mask=mask_nd, other=0)
                signs = ((packed_qjl >> kq_shift[None, :]) & 1).to(tl.float32) * 2.0 - 1.0
                score_qjl = tl.sum(signs * q_sketch[None, :], axis=1)

                # ── Norms ──
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

                # ── Values: vectorized unpack (no scatter!) ──
                # Same strategy as keys: load the packed byte for each
                # (token, dim) directly, extract the 4-bit nibble, lookup.
                vm_ptrs = (
                    COMP_BYTES
                    + offs_n[:, None] * stride_comp_s
                    + kv_head * stride_comp_h
                    + VM_OFF + vm_byte[None, :]
                )
                packed_vm = tl.load(vm_ptrs, mask=mask_nd, other=0)
                v_idx = ((packed_vm >> vm_shift[None, :]) & 0xF).to(tl.int32)
                v_vals = _lookup_8(
                    v_idx, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7,
                ).to(tl.float32)

                value_norm = tl.load(
                    COMP_FP16
                    + offs_n * stride_compfp_s
                    + kv_head * stride_compfp_h
                    + VN_FP16_OFF,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)
                v_vals *= value_norm[:, None]

                # ── Online softmax + direct accumulation ──
                n_e_max = tl.maximum(tl.max(scores, axis=0), e_max)
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(scores - n_e_max)
                p = tl.where(mask_n, p, 0.0)

                acc = acc * re_scale + tl.sum(p[:, None] * v_vals, axis=0)
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
    def _tq_fwht_step(x, h: tl.constexpr, BLOCK_D: tl.constexpr):
        """One butterfly step of the Fast Walsh-Hadamard Transform."""
        offs = tl.arange(0, BLOCK_D)
        # For each index, determine if it's in the "even" half of its group
        # Group = offs // h, is_even = (group & 1) == 0
        is_even = ((offs // h) & 1) == 0
        # Partner: flip the h-bit (even→+h, odd→-h)
        partner_offs = tl.where(is_even, offs + h, offs - h)
        # Gather partner values using the standard Triton pattern
        partner_val = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_D)[None, :] == partner_offs[:, None],
                x[None, :], 0.0,
            ),
            axis=1,
        )
        return tl.where(is_even, x + partner_val, partner_val - x)

    @triton.jit
    def _tq_decode_stage2_kernel(
        PARTIAL_ACC,
        PARTIAL_LSE,
        VAL_PI,
        VAL_SIGMA,
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
        USE_WHT: tl.constexpr,
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

        if USE_WHT:
            # Inverse WHT rotation: out = sigma * (1/sqrt(d)) * FWHT(merged_rot)
            # Unrolled butterfly: up to 8 steps for BLOCK_D <= 256
            transformed = merged_rot
            if BLOCK_D >= 2:
                transformed = _tq_fwht_step(transformed, 1, BLOCK_D)
            if BLOCK_D >= 4:
                transformed = _tq_fwht_step(transformed, 2, BLOCK_D)
            if BLOCK_D >= 8:
                transformed = _tq_fwht_step(transformed, 4, BLOCK_D)
            if BLOCK_D >= 16:
                transformed = _tq_fwht_step(transformed, 8, BLOCK_D)
            if BLOCK_D >= 32:
                transformed = _tq_fwht_step(transformed, 16, BLOCK_D)
            if BLOCK_D >= 64:
                transformed = _tq_fwht_step(transformed, 32, BLOCK_D)
            if BLOCK_D >= 128:
                transformed = _tq_fwht_step(transformed, 64, BLOCK_D)
            if BLOCK_D >= 256:
                transformed = _tq_fwht_step(transformed, 128, BLOCK_D)
            rsqrt_d = 1.0 / tl.sqrt(float(HEAD_DIM))
            sigma = tl.load(VAL_SIGMA + offs_d, mask=mask_d, other=1.0).to(tl.float32)
            out = sigma * transformed * rsqrt_d
        else:
            # Dense Haar rotation: out = merged_rot @ val_pi
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

    # ── Fused decode kernel: score + softmax + accumulate + unrotate ──
    # One program per (request, kv_head, query_group).  Walks the block
    # table directly so the Python host never gathers KV data.

    @triton.jit
    def _tq_fused_decode_kernel(
        # Pre-rotated queries: (num_reqs, num_kv_heads, heads_per_kv, D)
        Q_ROT, Q_SKETCH,
        # KV cache – two typed views of the same memory
        KV_U8,       # uint8 view  (num_blocks, block_size, nkh, total_bytes)
        KV_FP16,     # fp16  view  (num_blocks, block_size, nkh, fp16_elems)
        # Indirection
        BLOCK_TABLE,  # (num_reqs, max_blocks)  int32
        SEQ_LENS,     # (num_reqs,)             int32
        # Unrotation
        VAL_SIGMA,    # (D,) float32   – WHT path
        VAL_PI,       # (D, D) float32 – Haar path
        # Output: (num_reqs, nkh, hpkv, D)
        OUT,
        # ── strides (element counts in each pointer's dtype) ──
        stride_qr_r, stride_qr_h, stride_qr_g,
        stride_qs_r, stride_qs_h, stride_qs_g,
        stride_u8_b, stride_u8_t, stride_u8_h,
        stride_fp16_b, stride_fp16_t, stride_fp16_h,
        stride_bt_r,
        stride_vp_row, stride_vp_col,
        stride_out_r, stride_out_h, stride_out_g,
        # ── scalar params ──
        qjl_corr, sm_scale,
        # Key centroids (2-bit → 4 values)
        c0, c1, c2, c3,
        # Value centroids (4-bit packing → 8 values)
        vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7,
        # ── compile-time constants ──
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CACHE_BLOCK_SIZE: tl.constexpr,
        USE_WHT: tl.constexpr,
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
        req_id = tl.program_id(0)
        kv_head = tl.program_id(1)
        q_group = tl.program_id(2)

        seq_len = tl.load(SEQ_LENS + req_id).to(tl.int32)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM

        qr_base = (Q_ROT + req_id * stride_qr_r
                    + kv_head * stride_qr_h + q_group * stride_qr_g)
        qs_base = (Q_SKETCH + req_id * stride_qs_r
                    + kv_head * stride_qs_h + q_group * stride_qs_g)

        # Online-softmax accumulators
        e_max = -float("inf")
        e_sum = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        offs_km = tl.arange(0, KM_BYTES)
        offs_kq = tl.arange(0, KQ_BYTES)
        offs_vm = tl.arange(0, VM_BYTES)

        # ── iterate over KV tokens via block table ──────────────────
        for start_n in range(0, seq_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len

            # Block-table indirection
            blk_indices = (offs_n // CACHE_BLOCK_SIZE).to(tl.int32)
            tok_offsets = (offs_n % CACHE_BLOCK_SIZE).to(tl.int64)
            block_ids = tl.load(
                BLOCK_TABLE + req_id * stride_bt_r + blk_indices,
                mask=mask_n, other=0,
            ).to(tl.int64)

            u8_bases = (block_ids * stride_u8_b
                        + tok_offsets * stride_u8_t
                        + kv_head * stride_u8_h)
            fp16_bases = (block_ids * stride_fp16_b
                          + tok_offsets * stride_fp16_t
                          + kv_head * stride_fp16_h)

            # ── MSE scoring ──
            km_ptrs = (KV_U8 + u8_bases[:, None]
                       + (KM_OFF + offs_km.to(tl.int64))[None, :])
            packed_km = tl.load(km_ptrs, mask=mask_n[:, None], other=0)

            score_mse = tl.zeros([BLOCK_N], dtype=tl.float32)
            for shift_idx in range(4):
                q_coords = offs_km * 4 + shift_idx
                q_vals = tl.load(
                    qr_base + q_coords,
                    mask=q_coords < HEAD_DIM, other=0.0,
                ).to(tl.float32)
                idx = ((packed_km >> (shift_idx * 2)) & 0x3).to(tl.int32)
                vals = _lookup_4(idx, c0, c1, c2, c3).to(tl.float32)
                score_mse += tl.sum(vals * q_vals[None, :], axis=1)

            # ── QJL scoring ──
            qq_ptrs = (KV_U8 + u8_bases[:, None]
                       + (KQ_OFF + offs_kq.to(tl.int64))[None, :])
            packed_qjl = tl.load(qq_ptrs, mask=mask_n[:, None], other=0)

            score_qjl = tl.zeros([BLOCK_N], dtype=tl.float32)
            for bit_idx in range(8):
                q_coords = offs_kq * 8 + bit_idx
                q_vals = tl.load(
                    qs_base + q_coords,
                    mask=q_coords < HEAD_DIM, other=0.0,
                ).to(tl.float32)
                bits = ((packed_qjl >> bit_idx) & 1).to(tl.float32)
                signs = bits * 2.0 - 1.0
                score_qjl += tl.sum(signs * q_vals[None, :], axis=1)

            # ── norms (fp16 view) ──
            residual_norm = tl.load(
                KV_FP16 + fp16_bases + KR_FP16_OFF,
                mask=mask_n, other=0.0,
            ).to(tl.float32)
            key_norm = tl.load(
                KV_FP16 + fp16_bases + KN_FP16_OFF,
                mask=mask_n, other=0.0,
            ).to(tl.float32)

            scores = ((score_mse + qjl_corr * score_qjl * residual_norm)
                      * key_norm * sm_scale)
            scores = tl.where(mask_n, scores, -float("inf"))

            # ── online softmax ──
            n_e_max = tl.maximum(tl.max(scores, axis=0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(scores - n_e_max)
            p = tl.where(mask_n, p, 0.0)
            acc *= re_scale

            # ── value accumulation ──
            vm_ptrs = (KV_U8 + u8_bases[:, None]
                       + (VM_OFF + offs_vm.to(tl.int64))[None, :])
            packed_vm = tl.load(vm_ptrs, mask=mask_n[:, None], other=0)
            value_norm = tl.load(
                KV_FP16 + fp16_bases + VN_FP16_OFF,
                mask=mask_n, other=0.0,
            ).to(tl.float32)

            block_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            for nibble_idx in range(2):
                coords = offs_vm * 2 + nibble_idx
                idx = ((packed_vm >> (nibble_idx * 4)) & 0xF).to(tl.int32)
                vals = _lookup_8(
                    idx, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7,
                ).to(tl.float32)
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

        # ── normalise (safe for seq_len==0) ──
        safe_denom = tl.where(e_sum > 0, e_sum, 1.0)
        result = tl.where(mask_d, acc / safe_denom, 0.0)

        # ── unrotation (fused stage-2) ──
        if USE_WHT:
            transformed = result
            if BLOCK_D >= 2:
                transformed = _tq_fwht_step(transformed, 1, BLOCK_D)
            if BLOCK_D >= 4:
                transformed = _tq_fwht_step(transformed, 2, BLOCK_D)
            if BLOCK_D >= 8:
                transformed = _tq_fwht_step(transformed, 4, BLOCK_D)
            if BLOCK_D >= 16:
                transformed = _tq_fwht_step(transformed, 8, BLOCK_D)
            if BLOCK_D >= 32:
                transformed = _tq_fwht_step(transformed, 16, BLOCK_D)
            if BLOCK_D >= 64:
                transformed = _tq_fwht_step(transformed, 32, BLOCK_D)
            if BLOCK_D >= 128:
                transformed = _tq_fwht_step(transformed, 64, BLOCK_D)
            if BLOCK_D >= 256:
                transformed = _tq_fwht_step(transformed, 128, BLOCK_D)
            rsqrt_d = 1.0 / tl.sqrt(float(HEAD_DIM))
            sigma = tl.load(
                VAL_SIGMA + offs_d, mask=mask_d, other=1.0,
            ).to(tl.float32)
            result = sigma * transformed * rsqrt_d
        else:
            val_pi = tl.load(
                VAL_PI
                + offs_d[:, None] * stride_vp_row
                + offs_d[None, :] * stride_vp_col,
                mask=mask_d[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            result = tl.sum(result[:, None] * val_pi, axis=0)

        # ── store ──
        out_base = (OUT + req_id * stride_out_r
                    + kv_head * stride_out_h + q_group * stride_out_g)
        tl.store(out_base + offs_d, result, mask=mask_d)


    # ═══════════════════════════════════════════════════════════════════
    # Fused KV compression kernel
    # ═══════════════════════════════════════════════════════════════════

    @triton.jit
    def _tq_fused_compress_kernel(
        # Inputs — flattened (N, D) vectors
        KEY, VALUE,
        # WHT sign vectors
        KEY_SIGMA, VAL_SIGMA,
        # Quantizer tables (small 1-D arrays)
        KEY_BOUNDARIES, KEY_CENTROIDS, VAL_BOUNDARIES,
        # QJL projection matrix (QJL_DIM, D)
        S_MATRIX,
        # Outputs — raw indices / signs / norms
        KEY_MSE_OUT, QJL_SIGNS_OUT,
        KEY_RNORM_OUT, KEY_NORM_OUT,
        VAL_MSE_OUT, VAL_NORM_OUT,
        # Strides
        stride_kv,            # row stride for KEY / VALUE
        stride_s,             # row stride for S_MATRIX
        stride_idx,           # row stride for KEY_MSE_OUT / VAL_MSE_OUT
        stride_qjl,          # row stride for QJL_SIGNS_OUT
        # Scalar
        NUM_ROWS,
        # Constexpr
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        QJL_DIM: tl.constexpr,
        BLOCK_QJL: tl.constexpr,
        NUM_KEY_BOUNDS: tl.constexpr,
        NUM_VAL_BOUNDS: tl.constexpr,
    ):
        """Fused WHT-rotate + MSE-quantize + QJL-project + sign for KV compression.

        Each program handles one row = one (token, kv_head) pair.
        Replaces ~1500 CUDA kernel launches (Python-loop FWHT) with one Triton launch.
        """
        pid = tl.program_id(0)
        if pid >= NUM_ROWS:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM
        rsqrt_d = 1.0 / tl.sqrt(float(HEAD_DIM))

        # ── Load key & value ───────────────────────────────────────────
        key = tl.load(
            KEY + pid * stride_kv + offs_d, mask=mask_d, other=0.0,
        ).to(tl.float32)
        val = tl.load(
            VALUE + pid * stride_kv + offs_d, mask=mask_d, other=0.0,
        ).to(tl.float32)

        # ── Compute norms & normalize ──────────────────────────────────
        k_norm = tl.sqrt(tl.sum(key * key, axis=0))
        k_norm = tl.maximum(k_norm, 1e-8)
        v_norm = tl.sqrt(tl.sum(val * val, axis=0))
        v_norm = tl.maximum(v_norm, 1e-8)
        key_unit = key / k_norm
        val_unit = val / v_norm

        # ── Load WHT sign vectors ──────────────────────────────────────
        sigma_k = tl.load(
            KEY_SIGMA + offs_d, mask=mask_d, other=1.0,
        ).to(tl.float32)
        sigma_v = tl.load(
            VAL_SIGMA + offs_d, mask=mask_d, other=1.0,
        ).to(tl.float32)

        # ── Key: WHT forward rotation ──────────────────────────────────
        key_rot = _tq_fwht_full(key_unit * sigma_k, BLOCK_D) * rsqrt_d

        # ── Key: MSE bucketize ─────────────────────────────────────────
        k_idx = tl.zeros([BLOCK_D], dtype=tl.int32)
        for i in range(NUM_KEY_BOUNDS):
            b = tl.load(KEY_BOUNDARIES + i).to(tl.float32)
            k_idx += (key_rot > b).to(tl.int32)

        # ── Key: centroid dequant (rotated space) ──────────────────────
        k_hat_rot = tl.load(
            KEY_CENTROIDS + k_idx, mask=mask_d, other=0.0,
        ).to(tl.float32)

        # ── Key: WHT inverse rotation of dequantized ──────────────────
        k_hat = sigma_k * _tq_fwht_full(k_hat_rot, BLOCK_D) * rsqrt_d

        # ── Residual & its norm ────────────────────────────────────────
        residual = key_unit - k_hat
        r_norm = tl.sqrt(tl.sum(residual * residual, axis=0))
        r_norm = tl.maximum(r_norm, 1e-8)

        # ── QJL projection: sign(residual @ S^T) ──────────────────────
        qjl_offs = tl.arange(0, BLOCK_QJL)
        for j_start in range(0, QJL_DIM, BLOCK_QJL):
            j = j_start + qjl_offs
            j_mask = j < QJL_DIM
            s_block = tl.load(
                S_MATRIX + j[:, None] * stride_s + offs_d[None, :],
                mask=j_mask[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            dots = tl.sum(s_block * residual[None, :], axis=1)
            signs = tl.where(dots >= 0, 1.0, -1.0)
            tl.store(
                QJL_SIGNS_OUT + pid * stride_qjl + j, signs, mask=j_mask,
            )

        # ── Value: WHT forward rotation ────────────────────────────────
        val_rot = _tq_fwht_full(val_unit * sigma_v, BLOCK_D) * rsqrt_d

        # ── Value: MSE bucketize ───────────────────────────────────────
        v_idx = tl.zeros([BLOCK_D], dtype=tl.int32)
        for i in range(NUM_VAL_BOUNDS):
            b = tl.load(VAL_BOUNDARIES + i).to(tl.float32)
            v_idx += (val_rot > b).to(tl.int32)

        # ── Store all outputs ──────────────────────────────────────────
        tl.store(KEY_MSE_OUT + pid * stride_idx + offs_d, k_idx, mask=mask_d)
        tl.store(KEY_RNORM_OUT + pid, r_norm)
        tl.store(KEY_NORM_OUT + pid, k_norm)
        tl.store(VAL_MSE_OUT + pid * stride_idx + offs_d, v_idx, mask=mask_d)
        tl.store(VAL_NORM_OUT + pid, v_norm)


def _fused_compress_triton(
    key_flat: torch.Tensor,
    val_flat: torch.Tensor,
    key_sigma: torch.Tensor,
    val_sigma: torch.Tensor,
    key_boundaries: torch.Tensor,
    key_centroids: torch.Tensor,
    val_boundaries: torch.Tensor,
    s_matrix: torch.Tensor,
    head_dim: int,
    qjl_dim: int,
) -> dict[str, torch.Tensor]:
    """Launch the fused compress kernel; return raw indices/signs/norms."""
    num_rows = key_flat.shape[0]
    device = key_flat.device
    block_d = triton.next_power_of_2(head_dim)
    block_qjl = min(16, triton.next_power_of_2(qjl_dim))

    key_f = key_flat.float().contiguous()
    val_f = val_flat.float().contiguous()
    key_sigma_f = key_sigma.float().contiguous()
    val_sigma_f = val_sigma.float().contiguous()
    key_bounds_f = key_boundaries.float().contiguous()
    key_cents_f = key_centroids.float().contiguous()
    val_bounds_f = val_boundaries.float().contiguous()
    s_f = s_matrix.float().contiguous()

    key_mse_out = torch.empty((num_rows, head_dim), dtype=torch.int32, device=device)
    qjl_signs_out = torch.empty((num_rows, qjl_dim), dtype=torch.float32, device=device)
    key_rnorm_out = torch.empty(num_rows, dtype=torch.float32, device=device)
    key_norm_out = torch.empty(num_rows, dtype=torch.float32, device=device)
    val_mse_out = torch.empty((num_rows, head_dim), dtype=torch.int32, device=device)
    val_norm_out = torch.empty(num_rows, dtype=torch.float32, device=device)

    grid = (num_rows,)
    _tq_fused_compress_kernel[grid](
        key_f, val_f,
        key_sigma_f, val_sigma_f,
        key_bounds_f, key_cents_f, val_bounds_f,
        s_f,
        key_mse_out, qjl_signs_out,
        key_rnorm_out, key_norm_out,
        val_mse_out, val_norm_out,
        key_f.stride(0), s_f.stride(0),
        key_mse_out.stride(0), qjl_signs_out.stride(0),
        num_rows,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        QJL_DIM=qjl_dim,
        BLOCK_QJL=block_qjl,
        NUM_KEY_BOUNDS=len(key_bounds_f),
        NUM_VAL_BOUNDS=len(val_bounds_f),
        num_warps=4,
        num_stages=1,
    )

    return {
        "key_mse_indices": key_mse_out.long(),
        "qjl_signs": qjl_signs_out,
        "key_residual_norm": key_rnorm_out,
        "key_norm": key_norm_out,
        "val_mse_indices": val_mse_out.long(),
        "val_norm": val_norm_out,
    }


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
    rotation: str = "haar",
    val_sigma: "torch.Tensor | None" = None,
) -> torch.Tensor:
    num_splits, num_kv_heads, heads_per_kv, head_dim = partial_acc.shape
    block_d = triton.next_power_of_2(head_dim)
    out = torch.empty(
        (num_kv_heads, heads_per_kv, head_dim),
        dtype=torch.float32,
        device=partial_acc.device,
    )
    val_pi_f = val_pi.float().contiguous()
    use_wht = rotation == "wht"
    # For WHT, we need sigma on device; for Haar, pass a dummy pointer
    if use_wht and val_sigma is not None:
        val_sigma_f = val_sigma.float().contiguous().to(partial_acc.device)
    else:
        val_sigma_f = torch.ones(head_dim, dtype=torch.float32, device=partial_acc.device)

    grid = (num_kv_heads, heads_per_kv)
    _tq_decode_stage2_kernel[grid](
        partial_acc.contiguous(),
        partial_lse.contiguous(),
        val_pi_f,
        val_sigma_f,
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
        USE_WHT=use_wht,
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
    if use_triton and TRITON_AVAILABLE and partial_acc.is_cuda:
        return _stage2_triton(
            partial_acc, partial_lse, val_pi,
            rotation=rotation, val_sigma=val_sigma)
    return _stage2_torch(
        partial_acc, partial_lse, val_pi,
        rotation=rotation, val_sigma=val_sigma)


# ═══════════════════════════════════════════════════════════════════════
# Fused decode — single-launch path with block-table indirection
# ═══════════════════════════════════════════════════════════════════════


def _fused_decode_triton(
    q_rot: torch.Tensor,
    q_sketch: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    layout: "_CompressedLayout",
    *,
    key_centroids: torch.Tensor,
    val_centroids: torch.Tensor,
    val_pi: torch.Tensor,
    val_sigma: "torch.Tensor | None",
    qjl_corr: float,
    sm_scale: float,
    rotation: str = "wht",
) -> torch.Tensor:
    """Launch the fused decode kernel for all requests in one shot."""

    num_reqs, num_kv_heads, heads_per_kv, head_dim = q_rot.shape
    if num_reqs == 0:
        return torch.empty(
            (0, num_kv_heads, heads_per_kv, head_dim),
            dtype=torch.float32, device=kv_cache.device)
    block_d = triton.next_power_of_2(head_dim)
    block_size = kv_cache.shape[1]
    block_n = 64 if int(seq_lens.max().item()) >= 64 else 32

    # Two typed views of the same contiguous memory
    kv_fp16 = kv_cache.contiguous()
    kv_u8 = kv_fp16.view(torch.uint8)

    out = torch.empty(
        (num_reqs, num_kv_heads, heads_per_kv, head_dim),
        dtype=torch.float32,
        device=kv_cache.device,
    )

    use_wht = rotation == "wht"
    if use_wht and val_sigma is not None:
        val_sigma_f = val_sigma.float().contiguous().to(kv_cache.device)
    else:
        val_sigma_f = torch.ones(
            head_dim, dtype=torch.float32, device=kv_cache.device)
    val_pi_f = val_pi.float().contiguous().to(kv_cache.device)

    kc = key_centroids.float().contiguous()
    vc = val_centroids.float().contiguous()

    block_table_i = block_table.int().contiguous()
    seq_lens_i = seq_lens.int().contiguous()

    q_rot_c = q_rot.contiguous()
    q_sketch_c = q_sketch.contiguous()
    grid = (num_reqs, num_kv_heads, heads_per_kv)
    _tq_fused_decode_kernel[grid](
        q_rot_c,
        q_sketch_c,
        kv_u8,
        kv_fp16,
        block_table_i,
        seq_lens_i,
        val_sigma_f,
        val_pi_f,
        out,
        # Q_ROT strides
        q_rot_c.stride(0), q_rot_c.stride(1), q_rot_c.stride(2),
        # Q_SKETCH strides
        q_sketch_c.stride(0), q_sketch_c.stride(1), q_sketch_c.stride(2),
        # KV_U8 strides (bytes)
        kv_u8.stride(0), kv_u8.stride(1), kv_u8.stride(2),
        # KV_FP16 strides (fp16 elements)
        kv_fp16.stride(0), kv_fp16.stride(1), kv_fp16.stride(2),
        # Block-table stride
        block_table_i.stride(0),
        # Val_pi strides
        val_pi_f.stride(0), val_pi_f.stride(1),
        # Output strides
        out.stride(0), out.stride(1), out.stride(2),
        # Scalars
        qjl_corr,
        sm_scale,
        # Key centroids
        float(kc[0].item()), float(kc[1].item()),
        float(kc[2].item()), float(kc[3].item()),
        # Val centroids
        float(vc[0].item()), float(vc[1].item()),
        float(vc[2].item()), float(vc[3].item()),
        float(vc[4].item()), float(vc[5].item()),
        float(vc[6].item()), float(vc[7].item()),
        # Constexprs
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        CACHE_BLOCK_SIZE=block_size,
        USE_WHT=use_wht,
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
    return out


def _tq_fused_decode(
    q_rot: torch.Tensor,
    q_sketch: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    layout: "_CompressedLayout",
    *,
    key_centroids: torch.Tensor,
    val_centroids: torch.Tensor,
    val_pi: torch.Tensor,
    val_sigma: "torch.Tensor | None",
    qjl_corr: float,
    sm_scale: float,
    rotation: str = "wht",
) -> "torch.Tensor | None":
    """Dispatch fused decode.  Returns None when Triton is unavailable."""

    triton_ok = (
        TRITON_AVAILABLE
        and q_rot.is_cuda
        and layout.key_mse_bits == 2
        and layout.val_mse_bits <= 4
        and len(key_centroids) == 4
        and len(val_centroids) == 8
    )
    if not triton_ok:
        return None

    return _fused_decode_triton(
        q_rot, q_sketch, kv_cache, block_table, seq_lens, layout,
        key_centroids=key_centroids,
        val_centroids=val_centroids,
        val_pi=val_pi,
        val_sigma=val_sigma,
        qjl_corr=qjl_corr,
        sm_scale=sm_scale,
        rotation=rotation,
    )
