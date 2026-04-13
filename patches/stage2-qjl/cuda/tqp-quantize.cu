#include "tqp-constants-cuda.cuh"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>

extern "C" int tqp_cuda_device_count() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return -(int)err;
    }
    return count;
}

template<int D>
__device__ static inline void tqp_quantize_block_device(
        const float * __restrict__ x,
        uint16_t * orig_norm_out,
        uint16_t * res_d_out,
        uint8_t * __restrict__ qs_out,
        uint8_t * __restrict__ signs_out,
        const float * __restrict__ pi,
        const float * __restrict__ s,
        const float * __restrict__ centroids,
        const float * __restrict__ bounds) {
    __shared__ float smem_vec[QK_TQ4P_D256];
    __shared__ uint8_t smem_idx[QK_TQ4P_D256];
    __shared__ float smem_scalars[2];

    const int tid = threadIdx.x;

    smem_vec[tid] = x[tid];
    __syncthreads();

    if (tid == 0) {
        float sq = 0.0f;
#pragma unroll 1
        for (int i = 0; i < D; ++i) {
            sq = __fadd_rn(sq, __fmul_rn(smem_vec[i], smem_vec[i]));
        }
        float orig_norm = sqrtf(sq);
        if (orig_norm < 1e-8f) {
            orig_norm = 1e-8f;
        }
        smem_scalars[0] = orig_norm;
        smem_scalars[1] = 1.0f / orig_norm;
    }
    __syncthreads();

    smem_vec[tid] = __fmul_rn(smem_vec[tid], smem_scalars[1]);
    __syncthreads();

    float acc = 0.0f;
#pragma unroll 1
    for (int j = 0; j < D; ++j) {
        acc = __fadd_rn(acc, __fmul_rn(__ldg(&pi[tid * D + j]), smem_vec[j]));
    }
    smem_idx[tid] = tqp_bucketize_d3(acc, bounds);
    __syncthreads();

    float x_hat = 0.0f;
#pragma unroll 1
    for (int i = 0; i < D; ++i) {
        x_hat = __fadd_rn(x_hat, __fmul_rn(__ldg(&pi[i * D + tid]), centroids[smem_idx[i]]));
    }
    smem_vec[tid] = __fadd_rn(smem_vec[tid], -x_hat);
    __syncthreads();

    if (tid == 0) {
        float r_sq = 0.0f;
#pragma unroll 1
        for (int i = 0; i < D; ++i) {
            r_sq = __fadd_rn(r_sq, __fmul_rn(smem_vec[i], smem_vec[i]));
        }
        smem_scalars[1] = sqrtf(r_sq);
    }
    __syncthreads();

    float proj = 0.0f;
#pragma unroll 1
    for (int j = 0; j < D; ++j) {
        proj = __fadd_rn(proj, __fmul_rn(__ldg(&s[tid * D + j]), smem_vec[j]));
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const uint8_t idx = smem_idx[tid];

    const uint32_t ballot_lo  = __ballot_sync(0xffffffffu, (idx & 1u) != 0);
    const uint32_t ballot_mid = __ballot_sync(0xffffffffu, (idx & 2u) != 0);
    const uint32_t ballot_hi  = __ballot_sync(0xffffffffu, (idx & 4u) != 0);
    const uint32_t sign_mask  = __ballot_sync(0xffffffffu, proj < 0.0f);

    if (lane == 0) {
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const int group = warp * 4 + sub;
            qs_out[group * 3 + 0] = (uint8_t)((ballot_lo  >> (8 * sub)) & 0xffu);
            qs_out[group * 3 + 1] = (uint8_t)((ballot_mid >> (8 * sub)) & 0xffu);
            qs_out[group * 3 + 2] = (uint8_t)((ballot_hi  >> (8 * sub)) & 0xffu);
            signs_out[group] = (uint8_t)((sign_mask >> (8 * sub)) & 0xffu);
        }
    }

    if (tid == 0) {
        *orig_norm_out = tqp_fp32_to_fp16_device(smem_scalars[0]);
        *res_d_out = tqp_fp32_to_fp16_device(smem_scalars[1]);
    }
}

__global__ static void tqp_quantize_kernel_d128(
        const float * __restrict__ x,
        block_tq4p_d128 * __restrict__ y,
        const float * __restrict__ pi,
        const float * __restrict__ s,
        int64_t n_blocks) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) {
        return;
    }

    tqp_quantize_block_device<QK_TQ4P_D128>(
        x + b * QK_TQ4P_D128,
        &y[b].orig_norm,
        &y[b].res_d,
        y[b].qs,
        y[b].qjl_signs,
        pi,
        s,
        c_tqp_centroids_d128,
        c_tqp_boundaries_d128);
}

__global__ static void tqp_quantize_kernel_d256(
        const float * __restrict__ x,
        block_tq4p_d256 * __restrict__ y,
        const float * __restrict__ pi,
        const float * __restrict__ s,
        int64_t n_blocks) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) {
        return;
    }

    tqp_quantize_block_device<QK_TQ4P_D256>(
        x + b * QK_TQ4P_D256,
        &y[b].orig_norm,
        &y[b].res_d,
        y[b].qs,
        y[b].qjl_signs,
        pi,
        s,
        c_tqp_centroids_d256,
        c_tqp_boundaries_d256);
}

extern "C" void ggml_cuda_tqp_quantize_row_d128(const float * x, void * y, int64_t k, cudaStream_t stream) {
    if (k % QK_TQ4P_D128 != 0) {
        return;
    }
    if (tqp_cuda_init(QK_TQ4P_D128) != cudaSuccess) {
        return;
    }
    const int64_t n_blocks = k / QK_TQ4P_D128;
    tqp_quantize_kernel_d128<<<(unsigned int)n_blocks, QK_TQ4P_D128, 0, stream>>>(
        x, (block_tq4p_d128 *)y, d_tqp_pi_d128, d_tqp_s_d128, n_blocks);
}

extern "C" void ggml_cuda_tqp_quantize_row_d256(const float * x, void * y, int64_t k, cudaStream_t stream) {
    if (k % QK_TQ4P_D256 != 0) {
        return;
    }
    if (tqp_cuda_init(QK_TQ4P_D256) != cudaSuccess) {
        return;
    }
    const int64_t n_blocks = k / QK_TQ4P_D256;
    tqp_quantize_kernel_d256<<<(unsigned int)n_blocks, QK_TQ4P_D256, 0, stream>>>(
        x, (block_tq4p_d256 *)y, d_tqp_pi_d256, d_tqp_s_d256, n_blocks);
}

template<typename Block>
static int tqp_cuda_quantize_row_host(
        int d,
        const float * x_host,
        void * y_host,
        int64_t k,
        void (*device_fn)(const float *, void *, int64_t, cudaStream_t)) {
    if (k <= 0 || k % d != 0) {
        return 1;
    }

    float * x_dev = nullptr;
    void * y_dev = nullptr;
    const size_t x_bytes = (size_t)k * sizeof(float);
    const size_t y_bytes = (size_t)(k / d) * sizeof(Block);

    cudaError_t err = cudaMalloc((void **)&x_dev, x_bytes);
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc(&y_dev, y_bytes);
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

extern "C" int tqp_cuda_quantize_row_d128(const float * x_host, void * y_host, int64_t k) {
    return tqp_cuda_quantize_row_host<block_tq4p_d128>(
        QK_TQ4P_D128, x_host, y_host, k, ggml_cuda_tqp_quantize_row_d128);
}

extern "C" int tqp_cuda_quantize_row_d256(const float * x_host, void * y_host, int64_t k) {
    return tqp_cuda_quantize_row_host<block_tq4p_d256>(
        QK_TQ4P_D256, x_host, y_host, k, ggml_cuda_tqp_quantize_row_d256);
}
