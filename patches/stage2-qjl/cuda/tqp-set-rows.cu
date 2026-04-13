// TQ4P SET_ROWS CUDA dispatch.
//
// SET_ROWS scatters source rows into destination rows indexed by src1.
// For TQ4P destinations, each row (= one head_dim vector) must be
// quantized via the full Stage-1 + Stage-2 pipeline. This can't use the
// generic `k_set_rows_quant` template because TQ4P quantization needs
// CTA-cooperative execution (shared memory, syncthreads, GEMV).
//
// Instead, we launch one CTA per row with D threads (128 or 256),
// reusing `tqp_quantize_block_device` from tqp-quantize.cu.

#include "tqp-constants-cuda.cuh"
#include "tqp-kernels.cuh"   // provides tqp_quantize_block_device
#include "ggml-tq-paper.h"

// ── SET_ROWS kernel for TQ4P_D128 ─────────────────────────────────────
//
// Grid:  n_rows CTAs  (one per source row being scattered)
// Block: 128 threads  (one per element in the head-dim vector)
//
// src0:  F32, shape [ne00=128, ne01=n_rows, ...]  — contiguous source
// src1:  I64/I32, shape [ne10=n_rows, ...]         — destination row indices
// dst:   TQ4P_D128, shape [ne0=128, ne1=n_kv, ...] — KV cache
//
// Each CTA reads one source row from src0, looks up the destination row
// index from src1, and quantizes into dst at that index.

template<uint8_t ROT, typename idx_t>
__global__ static void k_set_rows_tq4p_d128(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tq4p_d128 * __restrict__ dst,
        uint8_t layer_byte_val,
        uint8_t layer,
        const float * __restrict__ pi,
        const float * __restrict__ s,
        int64_t n_rows,
        int64_t src0_stride_row,    // nb01 / sizeof(float)
        int64_t dst_stride_row) {   // nb1  (bytes per dest row)

    const int64_t row = (int64_t)blockIdx.x;
    if (row >= n_rows) return;

    // Source: contiguous F32 row
    const float * src_row = src0 + row * src0_stride_row;

    // Destination: scattered via index
    const int64_t dst_row_idx = (int64_t)src1[row];
    block_tq4p_d128 * dst_block = (block_tq4p_d128 *)((char *)dst + dst_row_idx * dst_stride_row);

    tqp_quantize_block_device<QK_TQ4P_D128, ROT>(
        src_row,
        &dst_block->orig_norm,
        &dst_block->res_d,
        &dst_block->layer_idx,
        layer_byte_val,
        dst_block->qs,
        dst_block->qjl_signs,
        &c_tqp_sigma_d128[layer][0],
        pi + (size_t)layer * QK_TQ4P_D128 * QK_TQ4P_D128,
        s  + (size_t)layer * QK_TQ4P_D128 * QK_TQ4P_D128,
        c_tqp_centroids_d128,
        c_tqp_boundaries_d128);
}

// ── SET_ROWS kernel for TQ4P_D256 ─────────────────────────────────────

template<uint8_t ROT, typename idx_t>
__global__ static void k_set_rows_tq4p_d256(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tq4p_d256 * __restrict__ dst,
        uint8_t layer_byte_val,
        uint8_t layer,
        const float * __restrict__ pi,
        const float * __restrict__ s,
        int64_t n_rows,
        int64_t src0_stride_row,
        int64_t dst_stride_row) {

    const int64_t row = (int64_t)blockIdx.x;
    if (row >= n_rows) return;

    const float * src_row = src0 + row * src0_stride_row;
    const int64_t dst_row_idx = (int64_t)src1[row];
    block_tq4p_d256 * dst_block = (block_tq4p_d256 *)((char *)dst + dst_row_idx * dst_stride_row);

    tqp_quantize_block_device<QK_TQ4P_D256, ROT>(
        src_row,
        &dst_block->orig_norm,
        &dst_block->res_d,
        &dst_block->layer_idx,
        layer_byte_val,
        dst_block->qs,
        dst_block->qjl_signs,
        &c_tqp_sigma_d256[layer][0],
        pi + (size_t)layer * QK_TQ4P_D256 * QK_TQ4P_D256,
        s  + (size_t)layer * QK_TQ4P_D256 * QK_TQ4P_D256,
        c_tqp_centroids_d256,
        c_tqp_boundaries_d256);
}

// ── Host dispatch (extern "C" for cross-TU visibility) ────────────────

extern "C" void ggml_cuda_set_rows_tq4p_d128(
        const float * src0_d, const void * src1_d, void * dst_d,
        uint8_t layer_byte, bool idx_i64,
        int64_t n_rows, int64_t src0_stride_row, int64_t dst_stride_row,
        cudaStream_t stream) {

    if (tqp_cuda_init(QK_TQ4P_D128) != cudaSuccess) return;

    const uint8_t layer = TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS;
    const uint8_t rot   = TQP_EXTRACT_ROT(layer_byte);
    const uint8_t byte_stored = TQP_LAYER_BYTE(layer, rot);

    #define LAUNCH_D128(ROT_VAL, IDX_T) \
        k_set_rows_tq4p_d128<ROT_VAL, IDX_T> \
            <<<(unsigned int)n_rows, QK_TQ4P_D128, 0, stream>>>( \
                src0_d, (const IDX_T *)src1_d, (block_tq4p_d128 *)dst_d, \
                byte_stored, layer, d_tqp_pi_d128, d_tqp_s_d128, \
                n_rows, src0_stride_row, dst_stride_row)

    if (rot == TQP_ROT_WHT) {
        if (idx_i64) { LAUNCH_D128(TQP_ROT_WHT, int64_t); }
        else         { LAUNCH_D128(TQP_ROT_WHT, int32_t); }
    } else {
        if (idx_i64) { LAUNCH_D128(TQP_ROT_HAAR, int64_t); }
        else         { LAUNCH_D128(TQP_ROT_HAAR, int32_t); }
    }
    #undef LAUNCH_D128
}

extern "C" void ggml_cuda_set_rows_tq4p_d256(
        const float * src0_d, const void * src1_d, void * dst_d,
        uint8_t layer_byte, bool idx_i64,
        int64_t n_rows, int64_t src0_stride_row, int64_t dst_stride_row,
        cudaStream_t stream) {

    if (tqp_cuda_init(QK_TQ4P_D256) != cudaSuccess) return;

    const uint8_t layer = TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS;
    const uint8_t rot   = TQP_EXTRACT_ROT(layer_byte);
    const uint8_t byte_stored = TQP_LAYER_BYTE(layer, rot);

    #define LAUNCH_D256(ROT_VAL, IDX_T) \
        k_set_rows_tq4p_d256<ROT_VAL, IDX_T> \
            <<<(unsigned int)n_rows, QK_TQ4P_D256, 0, stream>>>( \
                src0_d, (const IDX_T *)src1_d, (block_tq4p_d256 *)dst_d, \
                byte_stored, layer, d_tqp_pi_d256, d_tqp_s_d256, \
                n_rows, src0_stride_row, dst_stride_row)

    if (rot == TQP_ROT_WHT) {
        if (idx_i64) { LAUNCH_D256(TQP_ROT_WHT, int64_t); }
        else         { LAUNCH_D256(TQP_ROT_WHT, int32_t); }
    } else {
        if (idx_i64) { LAUNCH_D256(TQP_ROT_HAAR, int64_t); }
        else         { LAUNCH_D256(TQP_ROT_HAAR, int32_t); }
    }
    #undef LAUNCH_D256
}
