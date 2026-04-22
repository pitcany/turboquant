#include "tqp-constants-cuda.cuh"
#include "tqp-kernels.cuh"

#include <cstddef>

template<int D, int BITS, uint8_t ROT, typename idx_t, typename Block>
__global__ static void k_set_rows_tqp(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        void * __restrict__ dst,
        uint8_t layer_byte_val,
        uint8_t layer,
        const float * __restrict__ pi,
        const float * __restrict__ s,
        const float * __restrict__ sigma,
        const float * __restrict__ centroids,
        const float * __restrict__ boundaries,
        int64_t n_total_blocks,
        int64_t n_blocks_per_row,
        int64_t src0_stride_row,
        int64_t dst_stride_row) {
    const int64_t bid = (int64_t)blockIdx.x;
    if (bid >= n_total_blocks) {
        return;
    }

    const int64_t row = bid / n_blocks_per_row;
    const int64_t bi = bid % n_blocks_per_row;

    const float * src_block = src0 + row * src0_stride_row + bi * D;
    const int64_t dst_row_idx = (int64_t)src1[row];
    char * dst_blk_ptr = (char *)dst + dst_row_idx * dst_stride_row + bi * sizeof(Block);

    tqp_quantize_block_device<D, ROT, BITS>(
        src_block,
        (uint16_t *)(dst_blk_ptr + offsetof(Block, orig_norm)),
        (uint16_t *)(dst_blk_ptr + offsetof(Block, res_d)),
        (uint8_t *)(dst_blk_ptr + offsetof(Block, layer_idx)),
        layer_byte_val,
        (uint8_t *)(dst_blk_ptr + offsetof(Block, qs)),
        (uint8_t *)(dst_blk_ptr + offsetof(Block, qjl_signs)),
        sigma + layer * D,
        pi + (size_t)layer * D * D,
        s + (size_t)layer * D * D,
        centroids,
        boundaries);
}

template<int D, int BITS, typename Block>
static void ggml_cuda_set_rows_tqp_impl(
        const float * src0_d,
        const void * src1_d,
        void * dst_d,
        uint8_t layer_byte,
        bool idx_i64,
        int64_t n_rows,
        int64_t src0_stride_row,
        int64_t dst_stride_row,
        cudaStream_t stream) {
    if (tqp_cuda_init(D, BITS) != cudaSuccess) {
        return;
    }
    const TqpDeviceState * state = tqp_cuda_current_device_state();
    if (!state) {
        return;
    }

    const uint8_t layer = TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS;
    const uint8_t rot = TQP_EXTRACT_ROT(layer_byte);
    const uint8_t byte_stored = TQP_LAYER_BYTE(layer, rot);

    const int64_t n_blocks_per_row = dst_stride_row / (int64_t)sizeof(Block);
    const int64_t n_total_blocks = n_rows * n_blocks_per_row;

    const float * pi = tqp_cuda_pi_ptr(state, D);
    const float * s = tqp_cuda_s_ptr(state, D);
    const float * sigma = tqp_cuda_sigma_ptr(state, D);
    const float * centroids = tqp_cuda_centroids_ptr(state, D, BITS);
    const float * boundaries = tqp_cuda_boundaries_ptr(state, D, BITS);
    if (!pi || !s || !sigma || !centroids || !boundaries) {
        return;
    }

#define TQP_SET_ROWS_LAUNCH(ROT_VAL, IDX_T)                                                                                      \
    k_set_rows_tqp<D, BITS, ROT_VAL, IDX_T, Block><<<(unsigned int)n_total_blocks, D, 0, stream>>>(                            \
        src0_d, (const IDX_T *)src1_d, dst_d, byte_stored, layer, pi, s, sigma, centroids, boundaries,                         \
        n_total_blocks, n_blocks_per_row, src0_stride_row, dst_stride_row)

    if (rot == TQP_ROT_WHT) {
        if (idx_i64) {
            TQP_SET_ROWS_LAUNCH(TQP_ROT_WHT, int64_t);
        } else {
            TQP_SET_ROWS_LAUNCH(TQP_ROT_WHT, int32_t);
        }
    } else {
        if (idx_i64) {
            TQP_SET_ROWS_LAUNCH(TQP_ROT_HAAR, int64_t);
        } else {
            TQP_SET_ROWS_LAUNCH(TQP_ROT_HAAR, int32_t);
        }
    }

#undef TQP_SET_ROWS_LAUNCH
}

#define TQP_DEFINE_SET_ROWS(D, BITS, BLOCK)                                                                                      \
    extern "C" void ggml_cuda_set_rows_tqp_d##D##_b##BITS(                                                                       \
            const float * src0_d, const void * src1_d, void * dst_d,                                                             \
            uint8_t layer_byte, bool idx_i64,                                                                                    \
            int64_t n_rows, int64_t src0_stride_row, int64_t dst_stride_row,                                                     \
            cudaStream_t stream) {                                                                                                \
        ggml_cuda_set_rows_tqp_impl<QK_TQP_D##D, BITS, BLOCK>(                                                                   \
            src0_d, src1_d, dst_d, layer_byte, idx_i64, n_rows, src0_stride_row, dst_stride_row, stream);                      \
    }

TQP_DEFINE_SET_ROWS(128, 2, block_tqp_d128_b2)
TQP_DEFINE_SET_ROWS(128, 3, block_tqp_d128_b3)
TQP_DEFINE_SET_ROWS(128, 4, block_tqp_d128_b4)
TQP_DEFINE_SET_ROWS(256, 2, block_tqp_d256_b2)
TQP_DEFINE_SET_ROWS(256, 3, block_tqp_d256_b3)
TQP_DEFINE_SET_ROWS(256, 4, block_tqp_d256_b4)

#undef TQP_DEFINE_SET_ROWS

extern "C" void ggml_cuda_set_rows_tq4p_d128(
        const float * src0_d, const void * src1_d, void * dst_d,
        uint8_t layer_byte, bool idx_i64,
        int64_t n_rows, int64_t src0_stride_row, int64_t dst_stride_row,
        cudaStream_t stream) {
    ggml_cuda_set_rows_tqp_d128_b3(
        src0_d, src1_d, dst_d, layer_byte, idx_i64, n_rows, src0_stride_row, dst_stride_row, stream);
}

extern "C" void ggml_cuda_set_rows_tq4p_d256(
        const float * src0_d, const void * src1_d, void * dst_d,
        uint8_t layer_byte, bool idx_i64,
        int64_t n_rows, int64_t src0_stride_row, int64_t dst_stride_row,
        cudaStream_t stream) {
    ggml_cuda_set_rows_tqp_d256_b3(
        src0_d, src1_d, dst_d, layer_byte, idx_i64, n_rows, src0_stride_row, dst_stride_row, stream);
}
