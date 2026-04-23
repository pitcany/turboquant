#include "tqp-constants-cuda.cuh"

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define QK_Q8K 256
#pragma pack(push, 1)
typedef struct {
    float d;
    int8_t qs[QK_Q8K];
    int16_t bsums[QK_Q8K / 16];
} block_q8k_cuda;
#pragma pack(pop)

__global__ static void tqp_dequantize_q8k_kernel(
        const block_q8k_cuda * __restrict__ src,
        float * __restrict__ dst,
        int64_t n_blocks) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) {
        return;
    }
    const int tid = threadIdx.x;
    const float d = src[b].d;
    dst[b * QK_Q8K + tid] = d * (float)src[b].qs[tid];
}

extern "C" void ggml_cuda_tqp_prepare_query_d64(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d128(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d256(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d64_b2(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d64_b3(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d64_b4(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d128_b2(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d128_b3(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d128_b4(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d256_b2(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d256_b3(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d256_b4(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d128(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d256(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d64_b2(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d64_b3(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d64_b4(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d128_b2(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d128_b3(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d128_b4(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d256_b2(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d256_b3(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d256_b4(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);

template<int D, int BITS, typename Block>
__device__ static inline float tqp_vec_dot_block_device(
        const Block * __restrict__ blk,
        const float * __restrict__ Sq,
        const float * __restrict__ q_rot,
        const float * __restrict__ centroids) {
    __shared__ uint8_t smem_qs[TQP_QS_BYTES(QK_TQP_D256, 4)];
    __shared__ uint8_t smem_signs[TQP_SIGN_BYTES(QK_TQP_D256)];
    __shared__ float partial_t1[8];
    __shared__ float partial_t2[8];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int IDX_BYTES = TQP_QS_BYTES(D, BITS);
    constexpr int SIGN_BYTES = TQP_SIGN_BYTES(D);

    for (int i = tid; i < IDX_BYTES; i += blockDim.x) {
        smem_qs[i] = blk->qs[i];
    }
    for (int i = tid; i < SIGN_BYTES; i += blockDim.x) {
        smem_signs[i] = blk->qjl_signs[i];
    }
    __syncthreads();

    float t1 = 0.0f;
    float t2 = 0.0f;
    const int base = tid * 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int elem = base + e;
        const uint8_t idx = tqp_unpack_index_bitplane<BITS>(smem_qs, elem);
        const float sign_val = tqp_unpack_sign_pm1(smem_signs, elem);
        t1 += q_rot[elem] * centroids[idx];
        t2 += Sq[elem] * sign_val;
    }

    t1 = tqp_warp_reduce_sum(t1);
    t2 = tqp_warp_reduce_sum(t2);

    if (lane == 0) {
        partial_t1[warp] = t1;
        partial_t2[warp] = t2;
    }
    __syncthreads();

    if (tid == 0) {
        constexpr int nwarps = (D / 4 + 31) / 32;
        float total_t1 = 0.0f;
        float total_t2 = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            total_t1 += partial_t1[w];
            total_t2 += partial_t2[w];
        }
        const float orig_norm = tqp_fp16_to_fp32_device(blk->orig_norm);
        const float res_d = tqp_fp16_to_fp32_device(blk->res_d);
        total_t1 *= orig_norm;
        total_t2 *= orig_norm * res_d * (TQP_SQRT_PI_OVER_2 / (float)D);
        return total_t1 + total_t2;
    }

    return 0.0f;
}

template<int D, int BITS, typename Block>
__global__ static void tqp_vec_dot_kernel(
        const Block * __restrict__ blocks,
        const float * __restrict__ Sq,
        const float * __restrict__ q_rot,
        float * __restrict__ out,
        int64_t n_blocks,
        const float * __restrict__ centroids) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) {
        return;
    }
    const float result = tqp_vec_dot_block_device<D, BITS>(&blocks[b], Sq, q_rot, centroids);
    if (threadIdx.x == 0) {
        out[b] = result;
    }
}

template<int D, int BITS, typename Block>
__global__ static void tqp_vec_dot_ggml_kernel(
        const Block * __restrict__ blocks,
        const float * __restrict__ Sq,
        const float * __restrict__ q_rot,
        float * __restrict__ dst,
        int64_t ne11,
        int64_t ne2,
        int64_t s01,
        int64_t s02,
        int64_t s03,
        int64_t d_s1,
        int64_t d_s2,
        int64_t d_s3,
        int64_t channel_ratio,
        int64_t sample_ratio,
        const float * __restrict__ centroids) {
    const int64_t row = (int64_t)blockIdx.x;
    const int64_t col = (int64_t)blockIdx.y;
    const int64_t channel_dst = (int64_t)blockIdx.z % ne2;
    const int64_t sample_dst = (int64_t)blockIdx.z / ne2;
    const int64_t channel_x = channel_dst / channel_ratio;
    const int64_t sample_x = sample_dst / sample_ratio;
    const int64_t query_index = ((sample_dst * ne2 + channel_dst) * ne11 + col);

    const Block * blk = blocks + sample_x * s03 + channel_x * s02 + row * s01;
    const float * Sq_i = Sq + query_index * D;
    const float * q_rot_i = q_rot + query_index * D;

    const float result = tqp_vec_dot_block_device<D, BITS>(blk, Sq_i, q_rot_i, centroids);
    if (threadIdx.x == 0) {
        dst[sample_dst * d_s3 + channel_dst * d_s2 + col * d_s1 + row] = result;
    }
}

template<int D, int BITS, typename Block>
static void ggml_cuda_tqp_vec_dot_blocks_impl(
        const void * blocks,
        const float * Sq,
        const float * q_rot,
        float * out,
        int64_t n_blocks,
        cudaStream_t stream) {
    if (tqp_cuda_init(D, BITS) != cudaSuccess) {
        return;
    }
    const TqpDeviceState * state = tqp_cuda_current_device_state();
    if (!state) {
        return;
    }
    const float * centroids = tqp_cuda_centroids_ptr(state, D, BITS);
    if (!centroids) {
        return;
    }
    tqp_vec_dot_kernel<D, BITS, Block><<<(unsigned int)n_blocks, D / 4, 0, stream>>>(
        (const Block *)blocks, Sq, q_rot, out, n_blocks, centroids);
}

template<typename Block>
static int tqp_cuda_vec_dot_row_host(
        int d,
        const float * q_host,
        const Block * blocks_host,
        float * out_host,
        int64_t n_blocks,
        void (*prepare_fn)(const float *, float *, float *, uint8_t, cudaStream_t),
        void (*vec_dot_fn)(const void *, const float *, const float *, float *, int64_t, cudaStream_t)) {
    if (n_blocks <= 0) {
        return 1;
    }

    const uint8_t layer_byte = blocks_host[0].layer_idx;

    float * q_dev = nullptr;
    float * Sq_dev = nullptr;
    float * q_rot_dev = nullptr;
    Block * blocks_dev = nullptr;
    float * out_dev = nullptr;

    const size_t q_bytes = (size_t)d * sizeof(float);
    const size_t block_bytes = (size_t)n_blocks * sizeof(Block);
    const size_t out_bytes = (size_t)n_blocks * sizeof(float);

    cudaError_t err = cudaMalloc((void **)&q_dev, q_bytes);
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void **)&Sq_dev, q_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&q_rot_dev, q_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&blocks_dev, block_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&out_dev, out_bytes);
    if (err != cudaSuccess) goto done;

    err = cudaMemcpy(q_dev, q_host, q_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto done;
    err = cudaMemcpy(blocks_dev, blocks_host, block_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto done;

    prepare_fn(q_dev, Sq_dev, q_rot_dev, layer_byte, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto done;

    vec_dot_fn(blocks_dev, Sq_dev, q_rot_dev, out_dev, n_blocks, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto done;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto done;

    err = cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost);

done:
    cudaFree(out_dev);
    cudaFree(blocks_dev);
    cudaFree(q_rot_dev);
    cudaFree(Sq_dev);
    cudaFree(q_dev);
    return (int)err;
}

template<typename Block>
static int tqp_cuda_vec_dot_q8k_host(
        int d,
        const void * q8k_host,
        const Block * blocks_host,
        float * out_host,
        int64_t n_blocks,
        void (*prepare_fn)(const float *, float *, float *, uint8_t, cudaStream_t),
        void (*vec_dot_fn)(const void *, const float *, const float *, float *, int64_t, cudaStream_t)) {
    if (n_blocks <= 0) return 1;

    const uint8_t layer_byte = blocks_host[0].layer_idx;
    const int64_t tqp_per_q8k = QK_Q8K / d;
    const int64_t n_q8k_blocks = (n_blocks + tqp_per_q8k - 1) / tqp_per_q8k;
    const size_t q8k_bytes = (size_t)n_q8k_blocks * sizeof(block_q8k_cuda);
    const size_t q_fp32_bytes = (size_t)n_q8k_blocks * QK_Q8K * sizeof(float);

    float * q_fp32_dev = nullptr;
    block_q8k_cuda * q8k_dev = nullptr;
    float * Sq_dev = nullptr;
    float * q_rot_dev = nullptr;
    Block * blocks_dev = nullptr;
    float * out_dev = nullptr;

    const size_t q_bytes = (size_t)d * sizeof(float);
    const size_t block_bytes = (size_t)n_blocks * sizeof(Block);
    const size_t out_bytes = (size_t)n_blocks * sizeof(float);

    cudaError_t err = cudaMalloc((void **)&q8k_dev, q8k_bytes);
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void **)&q_fp32_dev, q_fp32_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&Sq_dev, q_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&q_rot_dev, q_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&blocks_dev, block_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&out_dev, out_bytes);
    if (err != cudaSuccess) goto done;

    err = cudaMemcpy(q8k_dev, q8k_host, q8k_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto done;
    err = cudaMemcpy(blocks_dev, blocks_host, block_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto done;

    tqp_dequantize_q8k_kernel<<<(unsigned int)n_q8k_blocks, QK_Q8K, 0, 0>>>(
        q8k_dev, q_fp32_dev, n_q8k_blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto done;

    for (int64_t tqp_b = 0; tqp_b < n_blocks; ++tqp_b) {
        const int64_t qb = tqp_b / tqp_per_q8k;
        const int64_t sub = tqp_b % tqp_per_q8k;
        const float * q_slice = q_fp32_dev + qb * QK_Q8K + sub * d;

        prepare_fn(q_slice, Sq_dev, q_rot_dev, layer_byte, 0);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto done;
        vec_dot_fn(blocks_dev + tqp_b, Sq_dev, q_rot_dev, out_dev + tqp_b, 1, 0);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto done;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto done;

    err = cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost);

done:
    cudaFree(out_dev);
    cudaFree(blocks_dev);
    cudaFree(q_rot_dev);
    cudaFree(Sq_dev);
    cudaFree(q_fp32_dev);
    cudaFree(q8k_dev);
    return (int)err;
}

#define TQP_DEFINE_VEC_DOT(D, BITS, BLOCK)                                                                                       \
    extern "C" void ggml_cuda_tqp_vec_dot_blocks_d##D##_b##BITS(                                                                 \
            const void * blocks, const float * Sq, const float * q_rot,                                                          \
            float * out, int64_t n_blocks, cudaStream_t stream) {                                                                \
        ggml_cuda_tqp_vec_dot_blocks_impl<QK_TQP_D##D, BITS, BLOCK>(blocks, Sq, q_rot, out, n_blocks, stream);                 \
    }                                                                                                                             \
    extern "C" int tqp_cuda_vec_dot_row_d##D##_b##BITS(                                                                          \
            const float * q_host, const BLOCK * blocks_host, float * out_host, int64_t n_blocks) {                              \
        return tqp_cuda_vec_dot_row_host<BLOCK>(                                                                                 \
            QK_TQP_D##D, q_host, blocks_host, out_host, n_blocks,                                                                \
            ggml_cuda_tqp_prepare_query_d##D##_b##BITS,                                                                          \
            ggml_cuda_tqp_vec_dot_blocks_d##D##_b##BITS);                                                                        \
    }                                                                                                                             \
    extern "C" float tqp_cuda_vec_dot_block_d##D##_b##BITS(                                                                      \
            const float * q_host, const BLOCK * block_host) {                                                                    \
        float out = 0.0f;                                                                                                        \
        const int err = tqp_cuda_vec_dot_row_d##D##_b##BITS(q_host, block_host, &out, 1);                                       \
        return err == 0 ? out : 0.0f;                                                                                            \
    }                                                                                                                             \
    extern "C" int tqp_cuda_vec_dot_q8k_d##D##_b##BITS(                                                                          \
            const void * q8k_host, const BLOCK * blocks_host, float * out_host, int64_t n_blocks) {                             \
        return tqp_cuda_vec_dot_q8k_host<BLOCK>(                                                                                 \
            QK_TQP_D##D, q8k_host, blocks_host, out_host, n_blocks,                                                              \
            ggml_cuda_tqp_prepare_query_d##D##_b##BITS,                                                                          \
            ggml_cuda_tqp_vec_dot_blocks_d##D##_b##BITS);                                                                        \
    }

TQP_DEFINE_VEC_DOT(64, 2, block_tqp_d64_b2)
TQP_DEFINE_VEC_DOT(64, 3, block_tqp_d64_b3)
TQP_DEFINE_VEC_DOT(64, 4, block_tqp_d64_b4)
TQP_DEFINE_VEC_DOT(128, 2, block_tqp_d128_b2)
TQP_DEFINE_VEC_DOT(128, 3, block_tqp_d128_b3)
TQP_DEFINE_VEC_DOT(128, 4, block_tqp_d128_b4)
TQP_DEFINE_VEC_DOT(256, 2, block_tqp_d256_b2)
TQP_DEFINE_VEC_DOT(256, 3, block_tqp_d256_b3)
TQP_DEFINE_VEC_DOT(256, 4, block_tqp_d256_b4)

#undef TQP_DEFINE_VEC_DOT

extern "C" void ggml_cuda_tqp_vec_dot_blocks_d64(
        const void * blocks, const float * Sq, const float * q_rot,
        float * out, int64_t n_blocks, cudaStream_t stream) {
    ggml_cuda_tqp_vec_dot_blocks_d64_b3(blocks, Sq, q_rot, out, n_blocks, stream);
}

extern "C" void ggml_cuda_tqp_vec_dot_blocks_d128(
        const void * blocks, const float * Sq, const float * q_rot,
        float * out, int64_t n_blocks, cudaStream_t stream) {
    ggml_cuda_tqp_vec_dot_blocks_d128_b3(blocks, Sq, q_rot, out, n_blocks, stream);
}

extern "C" void ggml_cuda_tqp_vec_dot_blocks_d256(
        const void * blocks, const float * Sq, const float * q_rot,
        float * out, int64_t n_blocks, cudaStream_t stream) {
    ggml_cuda_tqp_vec_dot_blocks_d256_b3(blocks, Sq, q_rot, out, n_blocks, stream);
}

extern "C" int tqp_cuda_vec_dot_row_d64(
        const float * q_host, const block_tq4p_d64 * blocks_host, float * out_host, int64_t n_blocks) {
    return tqp_cuda_vec_dot_row_d64_b3(q_host, blocks_host, out_host, n_blocks);
}

extern "C" int tqp_cuda_vec_dot_row_d128(
        const float * q_host, const block_tq4p_d128 * blocks_host, float * out_host, int64_t n_blocks) {
    return tqp_cuda_vec_dot_row_d128_b3(q_host, blocks_host, out_host, n_blocks);
}

extern "C" int tqp_cuda_vec_dot_row_d256(
        const float * q_host, const block_tq4p_d256 * blocks_host, float * out_host, int64_t n_blocks) {
    return tqp_cuda_vec_dot_row_d256_b3(q_host, blocks_host, out_host, n_blocks);
}

extern "C" float tqp_cuda_vec_dot_block_d64(const float * q_host, const block_tq4p_d64 * block_host) {
    return tqp_cuda_vec_dot_block_d64_b3(q_host, block_host);
}

extern "C" float tqp_cuda_vec_dot_block_d128(const float * q_host, const block_tq4p_d128 * block_host) {
    return tqp_cuda_vec_dot_block_d128_b3(q_host, block_host);
}

extern "C" float tqp_cuda_vec_dot_block_d256(const float * q_host, const block_tq4p_d256 * block_host) {
    return tqp_cuda_vec_dot_block_d256_b3(q_host, block_host);
}

extern "C" int tqp_cuda_vec_dot_q8k_d64(
        const void * q8k_host, const block_tq4p_d64 * blocks_host, float * out_host, int64_t n_blocks) {
    return tqp_cuda_vec_dot_q8k_d64_b3(q8k_host, blocks_host, out_host, n_blocks);
}

extern "C" int tqp_cuda_vec_dot_q8k_d128(
        const void * q8k_host, const block_tq4p_d128 * blocks_host, float * out_host, int64_t n_blocks) {
    return tqp_cuda_vec_dot_q8k_d128_b3(q8k_host, blocks_host, out_host, n_blocks);
}

extern "C" int tqp_cuda_vec_dot_q8k_d256(
        const void * q8k_host, const block_tq4p_d256 * blocks_host, float * out_host, int64_t n_blocks) {
    return tqp_cuda_vec_dot_q8k_d256_b3(q8k_host, blocks_host, out_host, n_blocks);
}

#if __has_include("common.cuh")
#include "common.cuh"

static inline bool tqp_ggml_type_to_dbits(ggml_type type, int * d, int * bits) {
    switch (type) {
#ifdef GGML_TYPE_TQ4P_D64
        case GGML_TYPE_TQ4P_D64: *d = QK_TQP_D64; *bits = 3; return true;
#endif
        case GGML_TYPE_TQ4P_D128: *d = QK_TQP_D128; *bits = 3; return true;
        case GGML_TYPE_TQ4P_D256: *d = QK_TQP_D256; *bits = 3; return true;
#ifdef GGML_TYPE_TQP_D64_B2
        case GGML_TYPE_TQP_D64_B2: *d = QK_TQP_D64; *bits = 2; return true;
#endif
#ifdef GGML_TYPE_TQP_D64_B4
        case GGML_TYPE_TQP_D64_B4: *d = QK_TQP_D64; *bits = 4; return true;
#endif
#ifdef GGML_TYPE_TQP_D128_B2
        case GGML_TYPE_TQP_D128_B2: *d = QK_TQP_D128; *bits = 2; return true;
#endif
#ifdef GGML_TYPE_TQP_D128_B4
        case GGML_TYPE_TQP_D128_B4: *d = QK_TQP_D128; *bits = 4; return true;
#endif
#ifdef GGML_TYPE_TQP_D256_B2
        case GGML_TYPE_TQP_D256_B2: *d = QK_TQP_D256; *bits = 2; return true;
#endif
#ifdef GGML_TYPE_TQP_D256_B4
        case GGML_TYPE_TQP_D256_B4: *d = QK_TQP_D256; *bits = 4; return true;
#endif
        default: return false;
    }
}

extern "C" void ggml_cuda_op_tqp_vec_dot(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_Q8_K);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    int d = 0;
    int bits = 0;
    GGML_ASSERT(tqp_ggml_type_to_dbits(src0->type, &d, &bits));

    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(ne00 == d);
    GGML_ASSERT(ne10 == d);
    GGML_ASSERT(nb0 == (int64_t)sizeof(float));
    GGML_ASSERT(ne12 == ne2);
    GGML_ASSERT(ne13 == ne3);
    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(tqp_cuda_init(d, bits));

    const int64_t n_queries = ne11 * ne12 * ne13;
    ggml_cuda_pool_alloc<float> Sq_alloc(ctx.pool(), n_queries * d);
    ggml_cuda_pool_alloc<float> q_rot_alloc(ctx.pool(), n_queries * d);
    float * Sq = Sq_alloc.get();
    float * q_rot = q_rot_alloc.get();

    ggml_cuda_pool_alloc<float> q8k_fp32_alloc;
    const float * src1_d;
    int64_t q_s11;
    int64_t q_s12;
    int64_t q_s13;

    if (src1->type == GGML_TYPE_Q8_K) {
        const int64_t n_elements = ggml_nelements(src1);
        q8k_fp32_alloc.alloc(ctx.pool(), n_elements);
        float * fp32_buf = q8k_fp32_alloc.get();

        const int64_t n_q8k_blocks = n_elements / QK_Q8K;
        tqp_dequantize_q8k_kernel<<<(unsigned int)n_q8k_blocks, QK_Q8K, 0, stream>>>(
            (const block_q8k_cuda *)src1->data, fp32_buf, n_q8k_blocks);
        CUDA_CHECK(cudaGetLastError());

        src1_d = fp32_buf;
        q_s11 = (int64_t)d;
        q_s12 = ne11 * (int64_t)d;
        q_s13 = ne11 * ne12 * (int64_t)d;
    } else {
        GGML_ASSERT(nb10 == (int64_t)sizeof(float));
        src1_d = (const float *)src1->data;
        q_s11 = nb11 / (int64_t)sizeof(float);
        q_s12 = nb12 / (int64_t)sizeof(float);
        q_s13 = nb13 / (int64_t)sizeof(float);
    }

    uint8_t layer_byte = 0;
    {
        const uint8_t * first_block = (const uint8_t *)src0->data;
        const size_t layer_byte_offset = offsetof(block_tqp_d128_b2, layer_idx);
        CUDA_CHECK(cudaMemcpyAsync(&layer_byte, first_block + layer_byte_offset, 1, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    switch ((d << 8) | bits) {
        case (QK_TQP_D64 << 8) | 2:
            ggml_cuda_tqp_prepare_query_batch_d64_b2(src1_d, Sq, q_rot, ne11, ne12, ne13, q_s11, q_s12, q_s13, layer_byte, stream);
            break;
        case (QK_TQP_D64 << 8) | 3:
            ggml_cuda_tqp_prepare_query_batch_d64_b3(src1_d, Sq, q_rot, ne11, ne12, ne13, q_s11, q_s12, q_s13, layer_byte, stream);
            break;
        case (QK_TQP_D64 << 8) | 4:
            ggml_cuda_tqp_prepare_query_batch_d64_b4(src1_d, Sq, q_rot, ne11, ne12, ne13, q_s11, q_s12, q_s13, layer_byte, stream);
            break;
        case (QK_TQP_D128 << 8) | 2:
            ggml_cuda_tqp_prepare_query_batch_d128_b2(src1_d, Sq, q_rot, ne11, ne12, ne13, q_s11, q_s12, q_s13, layer_byte, stream);
            break;
        case (QK_TQP_D128 << 8) | 3:
            ggml_cuda_tqp_prepare_query_batch_d128_b3(src1_d, Sq, q_rot, ne11, ne12, ne13, q_s11, q_s12, q_s13, layer_byte, stream);
            break;
        case (QK_TQP_D128 << 8) | 4:
            ggml_cuda_tqp_prepare_query_batch_d128_b4(src1_d, Sq, q_rot, ne11, ne12, ne13, q_s11, q_s12, q_s13, layer_byte, stream);
            break;
        case (QK_TQP_D256 << 8) | 2:
            ggml_cuda_tqp_prepare_query_batch_d256_b2(src1_d, Sq, q_rot, ne11, ne12, ne13, q_s11, q_s12, q_s13, layer_byte, stream);
            break;
        case (QK_TQP_D256 << 8) | 3:
            ggml_cuda_tqp_prepare_query_batch_d256_b3(src1_d, Sq, q_rot, ne11, ne12, ne13, q_s11, q_s12, q_s13, layer_byte, stream);
            break;
        case (QK_TQP_D256 << 8) | 4:
            ggml_cuda_tqp_prepare_query_batch_d256_b4(src1_d, Sq, q_rot, ne11, ne12, ne13, q_s11, q_s12, q_s13, layer_byte, stream);
            break;
        default:
            GGML_ABORT("unsupported TQP CUDA vec-dot type");
    }
    CUDA_CHECK(cudaGetLastError());

    const size_t ts_src0 = ggml_type_size(src0->type);
    const int64_t s01 = nb01 / (int64_t)ts_src0;
    const int64_t s02 = nb02 / (int64_t)ts_src0;
    const int64_t s03 = nb03 / (int64_t)ts_src0;
    const int64_t d_s1 = nb1 / (int64_t)sizeof(float);
    const int64_t d_s2 = nb2 / (int64_t)sizeof(float);
    const int64_t d_s3 = nb3 / (int64_t)sizeof(float);
    const int64_t channel_ratio = ne2 / ne02;
    const int64_t sample_ratio = ne3 / ne03;
    const dim3 grid((unsigned int)ne01, (unsigned int)ne11, (unsigned int)(ne2 * ne3));
    float * dst_d = (float *)dst->data;

    const TqpDeviceState * state = tqp_cuda_current_device_state();
    const float * centroids = tqp_cuda_centroids_ptr(state, d, bits);
    GGML_ASSERT(centroids != nullptr);

    switch ((d << 8) | bits) {
        case (QK_TQP_D64 << 8) | 2:
            tqp_vec_dot_ggml_kernel<QK_TQP_D64, 2, block_tqp_d64_b2><<<grid, QK_TQP_D64 / 4, 0, stream>>>(
                (const block_tqp_d64_b2 *)src0->data, Sq, q_rot, dst_d,
                ne11, ne2, s01, s02, s03, d_s1, d_s2, d_s3, channel_ratio, sample_ratio, centroids);
            break;
        case (QK_TQP_D64 << 8) | 3:
            tqp_vec_dot_ggml_kernel<QK_TQP_D64, 3, block_tqp_d64_b3><<<grid, QK_TQP_D64 / 4, 0, stream>>>(
                (const block_tqp_d64_b3 *)src0->data, Sq, q_rot, dst_d,
                ne11, ne2, s01, s02, s03, d_s1, d_s2, d_s3, channel_ratio, sample_ratio, centroids);
            break;
        case (QK_TQP_D64 << 8) | 4:
            tqp_vec_dot_ggml_kernel<QK_TQP_D64, 4, block_tqp_d64_b4><<<grid, QK_TQP_D64 / 4, 0, stream>>>(
                (const block_tqp_d64_b4 *)src0->data, Sq, q_rot, dst_d,
                ne11, ne2, s01, s02, s03, d_s1, d_s2, d_s3, channel_ratio, sample_ratio, centroids);
            break;
        case (QK_TQP_D128 << 8) | 2:
            tqp_vec_dot_ggml_kernel<QK_TQP_D128, 2, block_tqp_d128_b2><<<grid, QK_TQP_D128 / 4, 0, stream>>>(
                (const block_tqp_d128_b2 *)src0->data, Sq, q_rot, dst_d,
                ne11, ne2, s01, s02, s03, d_s1, d_s2, d_s3, channel_ratio, sample_ratio, centroids);
            break;
        case (QK_TQP_D128 << 8) | 3:
            tqp_vec_dot_ggml_kernel<QK_TQP_D128, 3, block_tqp_d128_b3><<<grid, QK_TQP_D128 / 4, 0, stream>>>(
                (const block_tqp_d128_b3 *)src0->data, Sq, q_rot, dst_d,
                ne11, ne2, s01, s02, s03, d_s1, d_s2, d_s3, channel_ratio, sample_ratio, centroids);
            break;
        case (QK_TQP_D128 << 8) | 4:
            tqp_vec_dot_ggml_kernel<QK_TQP_D128, 4, block_tqp_d128_b4><<<grid, QK_TQP_D128 / 4, 0, stream>>>(
                (const block_tqp_d128_b4 *)src0->data, Sq, q_rot, dst_d,
                ne11, ne2, s01, s02, s03, d_s1, d_s2, d_s3, channel_ratio, sample_ratio, centroids);
            break;
        case (QK_TQP_D256 << 8) | 2:
            tqp_vec_dot_ggml_kernel<QK_TQP_D256, 2, block_tqp_d256_b2><<<grid, QK_TQP_D256 / 4, 0, stream>>>(
                (const block_tqp_d256_b2 *)src0->data, Sq, q_rot, dst_d,
                ne11, ne2, s01, s02, s03, d_s1, d_s2, d_s3, channel_ratio, sample_ratio, centroids);
            break;
        case (QK_TQP_D256 << 8) | 3:
            tqp_vec_dot_ggml_kernel<QK_TQP_D256, 3, block_tqp_d256_b3><<<grid, QK_TQP_D256 / 4, 0, stream>>>(
                (const block_tqp_d256_b3 *)src0->data, Sq, q_rot, dst_d,
                ne11, ne2, s01, s02, s03, d_s1, d_s2, d_s3, channel_ratio, sample_ratio, centroids);
            break;
        case (QK_TQP_D256 << 8) | 4:
            tqp_vec_dot_ggml_kernel<QK_TQP_D256, 4, block_tqp_d256_b4><<<grid, QK_TQP_D256 / 4, 0, stream>>>(
                (const block_tqp_d256_b4 *)src0->data, Sq, q_rot, dst_d,
                ne11, ne2, s01, s02, s03, d_s1, d_s2, d_s3, channel_ratio, sample_ratio, centroids);
            break;
        default:
            GGML_ABORT("unsupported TQP CUDA vec-dot type");
    }

    CUDA_CHECK(cudaGetLastError());
}
#endif
