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

template<int D, typename Block, typename dst_t>
__device__ static inline void tqp_dequantize_block_device(
        const Block * __restrict__ blk,
        dst_t * __restrict__ y,
        const float * __restrict__ pi_d128,
        const float * __restrict__ pi_d256) {
    __shared__ float smem_vec[QK_TQ4P_D256];

    const int tid = threadIdx.x;
    const uint8_t layer = TQP_EXTRACT_LAYER(blk->layer_idx) % TQP_MAX_LAYERS;
    const uint8_t rot = TQP_EXTRACT_ROT(blk->layer_idx);
    const float orig_norm = tqp_fp16_to_fp32_device(blk->orig_norm);

    if constexpr (D == QK_TQ4P_D128) {
        smem_vec[tid] = c_tqp_centroids_d128[tqp_unpack_index_bitplane(blk->qs, tid)];
    } else {
        smem_vec[tid] = c_tqp_centroids_d256[tqp_unpack_index_bitplane(blk->qs, tid)];
    }
    __syncthreads();

    float x_hat_unit = 0.0f;
    if (rot == TQP_ROT_WHT) {
        tqp_wht_shared<D>(smem_vec);
        if constexpr (D == QK_TQ4P_D128) {
            x_hat_unit = smem_vec[tid] * c_tqp_sigma_d128[layer][tid] * rsqrtf((float)D);
        } else {
            x_hat_unit = smem_vec[tid] * c_tqp_sigma_d256[layer][tid] * rsqrtf((float)D);
        }
    } else {
        const float * pi_layer = nullptr;
        if constexpr (D == QK_TQ4P_D128) {
            pi_layer = pi_d128 + (size_t)layer * QK_TQ4P_D128 * QK_TQ4P_D128;
        } else {
            pi_layer = pi_d256 + (size_t)layer * QK_TQ4P_D256 * QK_TQ4P_D256;
        }
#pragma unroll 1
        for (int i = 0; i < D; ++i) {
            x_hat_unit = __fadd_rn(x_hat_unit, __fmul_rn(__ldg(&pi_layer[i * D + tid]), smem_vec[i]));
        }
    }

    y[tid] = tqp_cast_from_float<dst_t>(orig_norm * x_hat_unit);
}

template<int D, typename Block, typename dst_t>
__global__ static void tqp_dequantize_row_kernel(
        const Block * __restrict__ x,
        dst_t * __restrict__ y,
        int64_t n_blocks,
        const float * __restrict__ pi_d128,
        const float * __restrict__ pi_d256) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) {
        return;
    }

    tqp_dequantize_block_device<D>(&x[b], y + b * D, pi_d128, pi_d256);
}

template<int D, typename Block, typename dst_t>
__global__ static void tqp_dequantize_row_nc_kernel(
        const Block * __restrict__ x,
        dst_t * __restrict__ y,
        int64_t ne00,
        int64_t ne01,
        int64_t ne02,
        int64_t s01,
        int64_t s02,
        int64_t s03,
        const float * __restrict__ pi_d128,
        const float * __restrict__ pi_d256) {
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
    tqp_dequantize_block_device<D>(&x[ibx], y + iy, pi_d128, pi_d256);
}

template<int D, typename Block, typename dst_t>
static void tqp_dequantize_row_cuda(const void * x, dst_t * y, int64_t k, cudaStream_t stream) {
    if (k <= 0 || k % D != 0) {
        return;
    }
    if (tqp_cuda_init(D) != cudaSuccess) {
        return;
    }

    const int64_t n_blocks = k / D;
    tqp_dequantize_row_kernel<D, Block><<<(unsigned int)n_blocks, D, 0, stream>>>(
        (const Block *)x, y, n_blocks, d_tqp_pi_d128, d_tqp_pi_d256);
}

template<int D, typename Block>
static void tqp_dequantize_row_nc_cuda(
        const void * x,
        half * y,
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
    if (tqp_cuda_init(D) != cudaSuccess) {
        return;
    }

    const dim3 grid((unsigned int)(ne00 / D), (unsigned int)ne01, (unsigned int)(ne02 * ne03));
    tqp_dequantize_row_nc_kernel<D, Block><<<grid, D, 0, stream>>>(
        (const Block *)x, y, ne00, ne01, ne02, s01, s02, s03, d_tqp_pi_d128, d_tqp_pi_d256);
}

extern "C" void dequantize_row_tq4p_d128_cuda(const void * x, half * y, int64_t k, cudaStream_t stream) {
    tqp_dequantize_row_cuda<QK_TQ4P_D128, block_tq4p_d128>(x, y, k, stream);
}

extern "C" void dequantize_row_tq4p_d256_cuda(const void * x, half * y, int64_t k, cudaStream_t stream) {
    tqp_dequantize_row_cuda<QK_TQ4P_D256, block_tq4p_d256>(x, y, k, stream);
}

extern "C" void dequantize_row_tq4p_d128_f32_cuda(const void * x, float * y, int64_t k, cudaStream_t stream) {
    tqp_dequantize_row_cuda<QK_TQ4P_D128, block_tq4p_d128>(x, y, k, stream);
}

extern "C" void dequantize_row_tq4p_d256_f32_cuda(const void * x, float * y, int64_t k, cudaStream_t stream) {
    tqp_dequantize_row_cuda<QK_TQ4P_D256, block_tq4p_d256>(x, y, k, stream);
}

extern "C" void dequantize_row_tq4p_d128_nc_cuda(
        const void * x, half * y,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t s01, int64_t s02, int64_t s03,
        cudaStream_t stream) {
    tqp_dequantize_row_nc_cuda<QK_TQ4P_D128, block_tq4p_d128>(
        x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

extern "C" void dequantize_row_tq4p_d256_nc_cuda(
        const void * x, half * y,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t s01, int64_t s02, int64_t s03,
        cudaStream_t stream) {
    tqp_dequantize_row_nc_cuda<QK_TQ4P_D256, block_tq4p_d256>(
        x, y, ne00, ne01, ne02, ne03, s01, s02, s03, stream);
}

static void dequantize_row_tq4p_d128_u16_cuda(const void * x, uint16_t * y, int64_t k, cudaStream_t stream) {
    dequantize_row_tq4p_d128_cuda(x, (half *)y, k, stream);
}

static void dequantize_row_tq4p_d256_u16_cuda(const void * x, uint16_t * y, int64_t k, cudaStream_t stream) {
    dequantize_row_tq4p_d256_cuda(x, (half *)y, k, stream);
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

extern "C" int tqp_cuda_dequantize_row_d128_f32(const block_tq4p_d128 * x_host, float * y_host, int64_t k) {
    return tqp_cuda_dequantize_row_host<block_tq4p_d128, float>(
        QK_TQ4P_D128, x_host, y_host, k, dequantize_row_tq4p_d128_f32_cuda);
}

extern "C" int tqp_cuda_dequantize_row_d256_f32(const block_tq4p_d256 * x_host, float * y_host, int64_t k) {
    return tqp_cuda_dequantize_row_host<block_tq4p_d256, float>(
        QK_TQ4P_D256, x_host, y_host, k, dequantize_row_tq4p_d256_f32_cuda);
}

extern "C" int tqp_cuda_dequantize_row_d128_f16(const block_tq4p_d128 * x_host, uint16_t * y_host, int64_t k) {
    return tqp_cuda_dequantize_row_host<block_tq4p_d128, uint16_t>(
        QK_TQ4P_D128, x_host, y_host, k, dequantize_row_tq4p_d128_u16_cuda);
}

extern "C" int tqp_cuda_dequantize_row_d256_f16(const block_tq4p_d256 * x_host, uint16_t * y_host, int64_t k) {
    return tqp_cuda_dequantize_row_host<block_tq4p_d256, uint16_t>(
        QK_TQ4P_D256, x_host, y_host, k, dequantize_row_tq4p_d256_u16_cuda);
}
