#include "tqp-constants-cuda.cuh"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>

// Prepare-query kernel:
//   Sq    = S · q                                (rotation-agnostic)
//   q_rot:
//     TQP_ROT_WHT : (1/√d) · WHT(σ ⊙ q)
//     TQP_ROT_HAAR: Π · q                        (dense GEMV)
//
// One CTA per query; blockDim.x == D, one thread per element.
template<int D, uint8_t ROT>
__device__ static inline void tqp_prepare_query_device(
        const float * __restrict__ q_in,
        float * __restrict__ Sq_out,
        float * __restrict__ q_rot_out,
        const float * __restrict__ s,
        const float * __restrict__ sigma,
        const float * __restrict__ pi) {
    __shared__ float q_smem[QK_TQ4P_D256];

    const int tid = threadIdx.x;
    const float rsqrt_d = (ROT == TQP_ROT_WHT) ? rsqrtf((float)D) : 0.0f;

    // Load q once; we'll read it from smem for both S·q and the rotation.
    const float q_tid = q_in[tid];
    q_smem[tid] = q_tid;
    __syncthreads();

    // Sq = S · q (rotation-agnostic).
    float acc_s = 0.0f;
#pragma unroll 1
    for (int j = 0; j < D; ++j) {
        acc_s += __ldg(&s[tid * D + j]) * q_smem[j];
    }
    Sq_out[tid] = acc_s;

    if constexpr (ROT == TQP_ROT_WHT) {
        // Reuse q_smem for the RHT butterfly.
        __syncthreads();
        q_smem[tid] = q_tid * sigma[tid];
        tqp_wht_shared<D>(q_smem);
        q_rot_out[tid] = q_smem[tid] * rsqrt_d;
    } else {
        // Dense Haar GEMV: q_rot[tid] = Σ_j Π[tid, j] · q[j].
        // q_smem is still valid from the initial load — nothing has overwritten it.
        float acc_pi = 0.0f;
#pragma unroll 1
        for (int j = 0; j < D; ++j) {
            acc_pi += __ldg(&pi[tid * D + j]) * q_smem[j];
        }
        q_rot_out[tid] = acc_pi;
    }
}

// Per-layer sigma is read from __constant__ memory inside the kernel; Π
// is passed via device pointer. ROT is a template parameter so the dead
// branch of tqp_prepare_query_device is compiled out.
template<uint8_t ROT>
__global__ static void tqp_prepare_query_kernel_d128(
        const float * __restrict__ q,
        float * __restrict__ Sq,
        float * __restrict__ q_rot,
        const float * __restrict__ s,
        const float * __restrict__ pi_layer,
        int layer) {
    tqp_prepare_query_device<QK_TQ4P_D128, ROT>(
        q, Sq, q_rot, s, &c_tqp_sigma_d128[layer][0], pi_layer);
}

template<uint8_t ROT>
__global__ static void tqp_prepare_query_kernel_d256(
        const float * __restrict__ q,
        float * __restrict__ Sq,
        float * __restrict__ q_rot,
        const float * __restrict__ s,
        const float * __restrict__ pi_layer,
        int layer) {
    tqp_prepare_query_device<QK_TQ4P_D256, ROT>(
        q, Sq, q_rot, s, &c_tqp_sigma_d256[layer][0], pi_layer);
}

template<int D, uint8_t ROT>
__device__ static inline void tqp_prepare_query_batch_device(
        const float * __restrict__ q,
        float * __restrict__ Sq,
        float * __restrict__ q_rot,
        int64_t ne11,
        int64_t ne12,
        int64_t s11,
        int64_t s12,
        int64_t s13,
        const float * __restrict__ s,
        const float * __restrict__ sigma,
        const float * __restrict__ pi) {
    const int64_t col = (int64_t)blockIdx.x;
    const int64_t channel = (int64_t)blockIdx.y;
    const int64_t sample = (int64_t)blockIdx.z;
    const int64_t query_index = (sample * ne12 + channel) * ne11 + col;
    const float * q_i = q + col * s11 + channel * s12 + sample * s13;

    tqp_prepare_query_device<D, ROT>(q_i, Sq + query_index * D, q_rot + query_index * D, s, sigma, pi);
}

template<uint8_t ROT>
__global__ static void tqp_prepare_query_batch_kernel_d128(
        const float * __restrict__ q,
        float * __restrict__ Sq,
        float * __restrict__ q_rot,
        int64_t ne11,
        int64_t ne12,
        int64_t s11,
        int64_t s12,
        int64_t s13,
        const float * __restrict__ s,
        const float * __restrict__ pi_layer,
        int layer) {
    tqp_prepare_query_batch_device<QK_TQ4P_D128, ROT>(
        q, Sq, q_rot, ne11, ne12, s11, s12, s13, s, &c_tqp_sigma_d128[layer][0], pi_layer);
}

template<uint8_t ROT>
__global__ static void tqp_prepare_query_batch_kernel_d256(
        const float * __restrict__ q,
        float * __restrict__ Sq,
        float * __restrict__ q_rot,
        int64_t ne11,
        int64_t ne12,
        int64_t s11,
        int64_t s12,
        int64_t s13,
        const float * __restrict__ s,
        const float * __restrict__ pi_layer,
        int layer) {
    tqp_prepare_query_batch_device<QK_TQ4P_D256, ROT>(
        q, Sq, q_rot, ne11, ne12, s11, s12, s13, s, &c_tqp_sigma_d256[layer][0], pi_layer);
}

extern "C" void ggml_cuda_tqp_prepare_query_d128(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream) {
    if (tqp_cuda_init(QK_TQ4P_D128) != cudaSuccess) {
        return;
    }
    const TqpDeviceState * tqp_state = tqp_cuda_current_device_state();
    if (!tqp_state) {
        return;
    }
    const int layer = (int)(TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS);
    const uint8_t rot = TQP_EXTRACT_ROT(layer_byte);
    const float * pi_layer = tqp_state->pi_d128 + (size_t)layer * QK_TQ4P_D128 * QK_TQ4P_D128;
    const float * s_layer  = tqp_state->s_d128  + (size_t)layer * QK_TQ4P_D128 * QK_TQ4P_D128;
    if (rot == TQP_ROT_WHT) {
        tqp_prepare_query_kernel_d128<TQP_ROT_WHT><<<1, QK_TQ4P_D128, 0, stream>>>(
            q, Sq, q_rot, s_layer, pi_layer, layer);
    } else {
        tqp_prepare_query_kernel_d128<TQP_ROT_HAAR><<<1, QK_TQ4P_D128, 0, stream>>>(
            q, Sq, q_rot, s_layer, pi_layer, layer);
    }
}

extern "C" void ggml_cuda_tqp_prepare_query_d256(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream) {
    if (tqp_cuda_init(QK_TQ4P_D256) != cudaSuccess) {
        return;
    }
    const TqpDeviceState * tqp_state = tqp_cuda_current_device_state();
    if (!tqp_state) {
        return;
    }
    const int layer = (int)(TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS);
    const uint8_t rot = TQP_EXTRACT_ROT(layer_byte);
    const float * pi_layer = tqp_state->pi_d256 + (size_t)layer * QK_TQ4P_D256 * QK_TQ4P_D256;
    const float * s_layer  = tqp_state->s_d256  + (size_t)layer * QK_TQ4P_D256 * QK_TQ4P_D256;
    if (rot == TQP_ROT_WHT) {
        tqp_prepare_query_kernel_d256<TQP_ROT_WHT><<<1, QK_TQ4P_D256, 0, stream>>>(
            q, Sq, q_rot, s_layer, pi_layer, layer);
    } else {
        tqp_prepare_query_kernel_d256<TQP_ROT_HAAR><<<1, QK_TQ4P_D256, 0, stream>>>(
            q, Sq, q_rot, s_layer, pi_layer, layer);
    }
}

extern "C" void ggml_cuda_tqp_prepare_query_batch_d128(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream) {
    if (tqp_cuda_init(QK_TQ4P_D128) != cudaSuccess) {
        return;
    }
    const TqpDeviceState * tqp_state = tqp_cuda_current_device_state();
    if (!tqp_state) {
        return;
    }
    const int layer = (int)(TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS);
    const uint8_t rot = TQP_EXTRACT_ROT(layer_byte);
    const float * pi_layer = tqp_state->pi_d128 + (size_t)layer * QK_TQ4P_D128 * QK_TQ4P_D128;
    const float * s_layer  = tqp_state->s_d128  + (size_t)layer * QK_TQ4P_D128 * QK_TQ4P_D128;
    const dim3 grid((unsigned int)ne11, (unsigned int)ne12, (unsigned int)ne13);
    if (rot == TQP_ROT_WHT) {
        tqp_prepare_query_batch_kernel_d128<TQP_ROT_WHT><<<grid, QK_TQ4P_D128, 0, stream>>>(
            q, Sq, q_rot, ne11, ne12, s11, s12, s13, s_layer, pi_layer, layer);
    } else {
        tqp_prepare_query_batch_kernel_d128<TQP_ROT_HAAR><<<grid, QK_TQ4P_D128, 0, stream>>>(
            q, Sq, q_rot, ne11, ne12, s11, s12, s13, s_layer, pi_layer, layer);
    }
}

extern "C" void ggml_cuda_tqp_prepare_query_batch_d256(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream) {
    if (tqp_cuda_init(QK_TQ4P_D256) != cudaSuccess) {
        return;
    }
    const TqpDeviceState * tqp_state = tqp_cuda_current_device_state();
    if (!tqp_state) {
        return;
    }
    const int layer = (int)(TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS);
    const uint8_t rot = TQP_EXTRACT_ROT(layer_byte);
    const float * pi_layer = tqp_state->pi_d256 + (size_t)layer * QK_TQ4P_D256 * QK_TQ4P_D256;
    const float * s_layer  = tqp_state->s_d256  + (size_t)layer * QK_TQ4P_D256 * QK_TQ4P_D256;
    const dim3 grid((unsigned int)ne11, (unsigned int)ne12, (unsigned int)ne13);
    if (rot == TQP_ROT_WHT) {
        tqp_prepare_query_batch_kernel_d256<TQP_ROT_WHT><<<grid, QK_TQ4P_D256, 0, stream>>>(
            q, Sq, q_rot, ne11, ne12, s11, s12, s13, s_layer, pi_layer, layer);
    } else {
        tqp_prepare_query_batch_kernel_d256<TQP_ROT_HAAR><<<grid, QK_TQ4P_D256, 0, stream>>>(
            q, Sq, q_rot, ne11, ne12, s11, s12, s13, s_layer, pi_layer, layer);
    }
}

static int tqp_cuda_prepare_query_host(
        int d,
        const float * q_host,
        float * Sq_host,
        float * q_rot_host,
        uint8_t layer_byte,
        void (*device_fn)(const float *, float *, float *, uint8_t, cudaStream_t)) {
    float * q_dev = nullptr;
    float * Sq_dev = nullptr;
    float * q_rot_dev = nullptr;
    const size_t bytes = (size_t)d * sizeof(float);

    cudaError_t err = cudaMalloc((void **)&q_dev, bytes);
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void **)&Sq_dev, bytes);
    if (err != cudaSuccess) {
        cudaFree(q_dev);
        return (int)err;
    }
    err = cudaMalloc((void **)&q_rot_dev, bytes);
    if (err != cudaSuccess) {
        cudaFree(Sq_dev);
        cudaFree(q_dev);
        return (int)err;
    }

    err = cudaMemcpy(q_dev, q_host, bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
        device_fn(q_dev, Sq_dev, q_rot_dev, layer_byte, 0);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(Sq_host, Sq_dev, bytes, cudaMemcpyDeviceToHost);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(q_rot_host, q_rot_dev, bytes, cudaMemcpyDeviceToHost);
    }

    cudaFree(q_rot_dev);
    cudaFree(Sq_dev);
    cudaFree(q_dev);
    return (int)err;
}

extern "C" int tqp_cuda_prepare_query_d128(const float * q_host, float * Sq_host, float * q_rot_host, uint8_t layer_byte) {
    return tqp_cuda_prepare_query_host(
        QK_TQ4P_D128, q_host, Sq_host, q_rot_host, layer_byte, ggml_cuda_tqp_prepare_query_d128);
}

extern "C" int tqp_cuda_prepare_query_d256(const float * q_host, float * Sq_host, float * q_rot_host, uint8_t layer_byte) {
    return tqp_cuda_prepare_query_host(
        QK_TQ4P_D256, q_host, Sq_host, q_rot_host, layer_byte, ggml_cuda_tqp_prepare_query_d256);
}
