#include "tqp-constants-cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>

template<typename dst_t>
__device__ static inline dst_t tqp_cast_from_float(float x);

template<>
__device__ inline float tqp_cast_from_float<float>(float x) {
    return x;
}

template<>
__device__ inline half tqp_cast_from_float<half>(float x) {
    return __float2half_rn(x);
}

template<int D, int BITS, typename Block, typename dst_t>
__device__ static inline void tqp_dequantize_block_device(
        const Block * __restrict__ blk,
        dst_t * __restrict__ y,
        const float * __restrict__ pi,
        const float * __restrict__ centroids,
        const float * __restrict__ sigma) {
    __shared__ float smem_vec[QK_TQP_D256];

    const int tid = threadIdx.x;
    uint16_t raw_norm;
    memcpy(&raw_norm, &blk->orig_norm, sizeof(uint16_t));
    const uint8_t layer = TQP_EXTRACT_LAYER(blk->layer_idx) % TQP_MAX_LAYERS;
    const uint8_t rot = TQP_EXTRACT_ROT(blk->layer_idx);
    const float orig_norm = tqp_fp16_to_fp32_device(raw_norm);

    smem_vec[tid] = centroids[tqp_unpack_index_bitplane<BITS>(blk->qs, tid)];
    __syncthreads();

    float x_hat_unit = 0.0f;
    if (rot == TQP_ROT_WHT) {
        tqp_wht_shared<D>(smem_vec);
        x_hat_unit = smem_vec[tid] * sigma[layer * D + tid] * rsqrtf((float)D);
    } else {
        const float * pi_layer = pi + (size_t)layer * D * D;
#pragma unroll 1
        for (int i = 0; i < D; ++i) {
            x_hat_unit = __fadd_rn(x_hat_unit, __fmul_rn(__ldg(&pi_layer[i * D + tid]), smem_vec[i]));
        }
    }

    float result = orig_norm * x_hat_unit;
    if (!isfinite(result)) {
        result = 0.0f;
    }
    y[tid] = tqp_cast_from_float<dst_t>(result);
}

template<int D, int BITS, typename Block, typename dst_t>
__global__ static void tqp_dequantize_row_kernel(
        const Block * __restrict__ x,
        dst_t * __restrict__ y,
        int64_t n_blocks,
        const float * __restrict__ pi,
        const float * __restrict__ centroids,
        const float * __restrict__ sigma) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) {
        return;
    }

    tqp_dequantize_block_device<D, BITS>(&x[b], y + b * D, pi, centroids, sigma);
}

template<int D, int BITS, typename Block, typename dst_t>
__global__ static void tqp_dequantize_row_nc_kernel(
        const Block * __restrict__ x,
        dst_t * __restrict__ y,
        int64_t ne00,
        int64_t ne01,
        int64_t ne02,
        int64_t s01,
        int64_t s02,
        int64_t s03,
        const float * __restrict__ pi,
        const float * __restrict__ centroids,
        const float * __restrict__ sigma) {
    const int64_t b0 = (int64_t)blockIdx.x;
    const int64_t i01 = (int64_t)blockIdx.y;
    const int64_t i02 = (int64_t)blockIdx.z % ne02;
    const int64_t i03 = (int64_t)blockIdx.z / ne02;

    const int64_t n_blocks_0 = ne00 / D;
    if (b0 >= n_blocks_0) {
        return;
    }

    const int64_t ibx = i03 * s03 + i02 * s02 + i01 * s01 + b0;
    const int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00 + b0 * D;
    tqp_dequantize_block_device<D, BITS>(&x[ibx], y + iy, pi, centroids, sigma);
}

template<int D, int BITS, typename Block, typename dst_t>
static void tqp_dequantize_row_cuda(const void * x, dst_t * y, int64_t k, cudaStream_t stream) {
    if (k <= 0 || k % D != 0) {
        return;
    }
    if (tqp_cuda_init(D, BITS) != cudaSuccess) {
        return;
    }

    const TqpDeviceState * state = tqp_cuda_current_device_state();
    if (!state) {
        return;
    }

    const float * pi = tqp_cuda_pi_ptr(state, D);
    const float * centroids = tqp_cuda_centroids_ptr(state, D, BITS);
    const float * sigma = tqp_cuda_sigma_ptr(state, D);
    if (!pi || !centroids || !sigma) {
        return;
    }

    const int64_t n_blocks = k / D;
    tqp_dequantize_row_kernel<D, BITS, Block><<<(unsigned int)n_blocks, D, 0, stream>>>(
        (const Block *)x, y, n_blocks, pi, centroids, sigma);
}

template<int D, int BITS, typename Block, typename dst_t>
static void tqp_dequantize_row_nc_cuda(
        const void * x,
        dst_t * y,
        int64_t ne00,
        int64_t ne01,
        int64_t ne02,
        int64_t ne03,
        int64_t s01,
        int64_t s02,
        int64_t s03,
        cudaStream_t stream) {
    if (ne00 <= 0 || ne00 % D != 0) {
        return;
    }
    if (tqp_cuda_init(D, BITS) != cudaSuccess) {
        return;
    }

    const TqpDeviceState * state = tqp_cuda_current_device_state();
    if (!state) {
        return;
    }

    const float * pi = tqp_cuda_pi_ptr(state, D);
    const float * centroids = tqp_cuda_centroids_ptr(state, D, BITS);
    const float * sigma = tqp_cuda_sigma_ptr(state, D);
    if (!pi || !centroids || !sigma) {
        return;
    }

    const dim3 grid((unsigned int)(ne00 / D), (unsigned int)ne01, (unsigned int)(ne02 * ne03));
    tqp_dequantize_row_nc_kernel<D, BITS, Block><<<grid, D, 0, stream>>>(
        (const Block *)x, y, ne00, ne01, ne02, s01, s02, s03, pi, centroids, sigma);
}

template<typename Block, typename dst_t>
static int tqp_cuda_dequantize_row_host(
        int d,
        const Block * x_host,
        dst_t * y_host,
        int64_t k,
        void (*device_fn)(const void *, dst_t *, int64_t, cudaStream_t)) {
    if (k <= 0 || k % d != 0) {
        return 1;
    }

    Block * x_dev = nullptr;
    dst_t * y_dev = nullptr;
    const size_t x_bytes = (size_t)(k / d) * sizeof(Block);
    const size_t y_bytes = (size_t)k * sizeof(dst_t);

    cudaError_t err = cudaMalloc((void **)&x_dev, x_bytes);
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void **)&y_dev, y_bytes);
    if (err != cudaSuccess) {
        cudaFree(x_dev);
        return (int)err;
    }

    err = cudaMemcpy(x_dev, x_host, x_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
        device_fn(x_dev, y_dev, k, 0);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(y_host, y_dev, y_bytes, cudaMemcpyDeviceToHost);
    }

    cudaFree(y_dev);
    cudaFree(x_dev);
    return (int)err;
}

#define TQP_DEFINE_DEQUANTIZE(D, BITS, BLOCK)                                                                                    \
    extern "C" void dequantize_row_tqp_d##D##_b##BITS##_cuda(                                                                    \
            const void * x, half * y, int64_t k, cudaStream_t stream) {                                                          \
        tqp_dequantize_row_cuda<QK_TQP_D##D, BITS, BLOCK>(x, y, k, stream);                                                     \
    }                                                                                                                             \
    extern "C" void dequantize_row_tqp_d##D##_b##BITS##_f32_cuda(                                                                \
            const void * x, float * y, int64_t k, cudaStream_t stream) {                                                         \
        tqp_dequantize_row_cuda<QK_TQP_D##D, BITS, BLOCK>(x, y, k, stream);                                                     \
    }                                                                                                                             \
    extern "C" void dequantize_row_tqp_d##D##_b##BITS##_nc_cuda(                                                                 \
            const void * x, half * y,                                                                                            \
            int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,                                                              \
            int64_t s01, int64_t s02, int64_t s03,                                                                               \
            cudaStream_t stream) {                                                                                                \
        tqp_dequantize_row_nc_cuda<QK_TQP_D##D, BITS, BLOCK>(x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);            \
    }                                                                                                                             \
    static void dequantize_row_tqp_d##D##_b##BITS##_u16_cuda(                                                                    \
            const void * x, uint16_t * y, int64_t k, cudaStream_t stream) {                                                      \
        dequantize_row_tqp_d##D##_b##BITS##_cuda(x, (half *)y, k, stream);                                                      \
    }                                                                                                                             \
    extern "C" int tqp_cuda_dequantize_row_d##D##_b##BITS##_f32(                                                                 \
            const BLOCK * x_host, float * y_host, int64_t k) {                                                                   \
        return tqp_cuda_dequantize_row_host<BLOCK, float>(                                                                       \
            QK_TQP_D##D, x_host, y_host, k, dequantize_row_tqp_d##D##_b##BITS##_f32_cuda);                                      \
    }                                                                                                                             \
    extern "C" int tqp_cuda_dequantize_row_d##D##_b##BITS##_f16(                                                                 \
            const BLOCK * x_host, uint16_t * y_host, int64_t k) {                                                                \
        return tqp_cuda_dequantize_row_host<BLOCK, uint16_t>(                                                                    \
            QK_TQP_D##D, x_host, y_host, k, dequantize_row_tqp_d##D##_b##BITS##_u16_cuda);                                      \
    }

TQP_DEFINE_DEQUANTIZE(64, 2, block_tqp_d64_b2)
TQP_DEFINE_DEQUANTIZE(64, 3, block_tqp_d64_b3)
TQP_DEFINE_DEQUANTIZE(64, 4, block_tqp_d64_b4)
TQP_DEFINE_DEQUANTIZE(128, 2, block_tqp_d128_b2)
TQP_DEFINE_DEQUANTIZE(128, 3, block_tqp_d128_b3)
TQP_DEFINE_DEQUANTIZE(128, 4, block_tqp_d128_b4)
TQP_DEFINE_DEQUANTIZE(256, 2, block_tqp_d256_b2)
TQP_DEFINE_DEQUANTIZE(256, 3, block_tqp_d256_b3)
TQP_DEFINE_DEQUANTIZE(256, 4, block_tqp_d256_b4)

#undef TQP_DEFINE_DEQUANTIZE

extern "C" void dequantize_row_tq4p_d64_cuda(const void * x, half * y, int64_t k, cudaStream_t stream) {
    dequantize_row_tqp_d64_b3_cuda(x, y, k, stream);
}

extern "C" void dequantize_row_tq4p_d128_cuda(const void * x, half * y, int64_t k, cudaStream_t stream) {
    dequantize_row_tqp_d128_b3_cuda(x, y, k, stream);
}

extern "C" void dequantize_row_tq4p_d256_cuda(const void * x, half * y, int64_t k, cudaStream_t stream) {
    dequantize_row_tqp_d256_b3_cuda(x, y, k, stream);
}

extern "C" void dequantize_row_tq4p_d64_f32_cuda(const void * x, float * y, int64_t k, cudaStream_t stream) {
    dequantize_row_tqp_d64_b3_f32_cuda(x, y, k, stream);
}

extern "C" void dequantize_row_tq4p_d128_f32_cuda(const void * x, float * y, int64_t k, cudaStream_t stream) {
    dequantize_row_tqp_d128_b3_f32_cuda(x, y, k, stream);
}

extern "C" void dequantize_row_tq4p_d256_f32_cuda(const void * x, float * y, int64_t k, cudaStream_t stream) {
    dequantize_row_tqp_d256_b3_f32_cuda(x, y, k, stream);
}

extern "C" void dequantize_row_tq4p_d64_nc_cuda(
        const void * x, half * y,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t s01, int64_t s02, int64_t s03,
        cudaStream_t stream) {
    dequantize_row_tqp_d64_b3_nc_cuda(x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

extern "C" void dequantize_row_tq4p_d128_nc_cuda(
        const void * x, half * y,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t s01, int64_t s02, int64_t s03,
        cudaStream_t stream) {
    dequantize_row_tqp_d128_b3_nc_cuda(x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

extern "C" void dequantize_row_tq4p_d256_nc_cuda(
        const void * x, half * y,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t s01, int64_t s02, int64_t s03,
        cudaStream_t stream) {
    dequantize_row_tqp_d256_b3_nc_cuda(x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

extern "C" int tqp_cuda_dequantize_row_d64_f32(const block_tq4p_d64 * x_host, float * y_host, int64_t k) {
    return tqp_cuda_dequantize_row_d64_b3_f32(x_host, y_host, k);
}

extern "C" int tqp_cuda_dequantize_row_d128_f32(const block_tq4p_d128 * x_host, float * y_host, int64_t k) {
    return tqp_cuda_dequantize_row_d128_b3_f32(x_host, y_host, k);
}

extern "C" int tqp_cuda_dequantize_row_d256_f32(const block_tq4p_d256 * x_host, float * y_host, int64_t k) {
    return tqp_cuda_dequantize_row_d256_b3_f32(x_host, y_host, k);
}

extern "C" int tqp_cuda_dequantize_row_d64_f16(const block_tq4p_d64 * x_host, uint16_t * y_host, int64_t k) {
    return tqp_cuda_dequantize_row_d64_b3_f16(x_host, y_host, k);
}

extern "C" int tqp_cuda_dequantize_row_d128_f16(const block_tq4p_d128 * x_host, uint16_t * y_host, int64_t k) {
    return tqp_cuda_dequantize_row_d128_b3_f16(x_host, y_host, k);
}

extern "C" int tqp_cuda_dequantize_row_d256_f16(const block_tq4p_d256 * x_host, uint16_t * y_host, int64_t k) {
    return tqp_cuda_dequantize_row_d256_b3_f16(x_host, y_host, k);
}
