#include "tqp-constants-cuda.cuh"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>

template<int D, uint8_t ROT>
__device__ static inline void tqp_prepare_query_device(
        const float * __restrict__ q_in,
        float * __restrict__ Sq_out,
        float * __restrict__ q_rot_out,
        const float * __restrict__ s,
        const float * __restrict__ sigma,
        const float * __restrict__ pi) {
    __shared__ float q_smem[QK_TQP_D256];

    const int tid = threadIdx.x;
    const float rsqrt_d = (ROT == TQP_ROT_WHT) ? rsqrtf((float)D) : 0.0f;

    const float q_tid = q_in[tid];
    q_smem[tid] = q_tid;
    __syncthreads();

    float acc_s = 0.0f;
#pragma unroll 1
    for (int j = 0; j < D; ++j) {
        acc_s += __ldg(&s[tid * D + j]) * q_smem[j];
    }
    Sq_out[tid] = acc_s;

    if constexpr (ROT == TQP_ROT_WHT) {
        __syncthreads();
        q_smem[tid] = q_tid * sigma[tid];
        tqp_wht_shared<D>(q_smem);
        q_rot_out[tid] = q_smem[tid] * rsqrt_d;
    } else {
        float acc_pi = 0.0f;
#pragma unroll 1
        for (int j = 0; j < D; ++j) {
            acc_pi += __ldg(&pi[tid * D + j]) * q_smem[j];
        }
        q_rot_out[tid] = acc_pi;
    }
}

template<int D, uint8_t ROT>
__global__ static void tqp_prepare_query_kernel(
        const float * __restrict__ q,
        float * __restrict__ Sq,
        float * __restrict__ q_rot,
        const float * __restrict__ s,
        const float * __restrict__ sigma_layer,
        const float * __restrict__ pi_layer) {
    tqp_prepare_query_device<D, ROT>(q, Sq, q_rot, s, sigma_layer, pi_layer);
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

template<int D, uint8_t ROT>
__global__ static void tqp_prepare_query_batch_kernel(
        const float * __restrict__ q,
        float * __restrict__ Sq,
        float * __restrict__ q_rot,
        int64_t ne11,
        int64_t ne12,
        int64_t s11,
        int64_t s12,
        int64_t s13,
        const float * __restrict__ s,
        const float * __restrict__ sigma_layer,
        const float * __restrict__ pi_layer) {
    tqp_prepare_query_batch_device<D, ROT>(
        q, Sq, q_rot, ne11, ne12, s11, s12, s13, s, sigma_layer, pi_layer);
}

template<int D>
static void ggml_cuda_tqp_prepare_query_impl(
        const float * q,
        float * Sq,
        float * q_rot,
        uint8_t layer_byte,
        cudaStream_t stream) {
    if (tqp_cuda_init(D, 3) != cudaSuccess) {
        return;
    }
    const TqpDeviceState * state = tqp_cuda_current_device_state();
    if (!state) {
        return;
    }

    const int layer = (int)(TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS);
    const uint8_t rot = TQP_EXTRACT_ROT(layer_byte);
    const float * pi_layer = tqp_cuda_pi_ptr(state, D) + (size_t)layer * D * D;
    const float * s_layer = tqp_cuda_s_ptr(state, D) + (size_t)layer * D * D;
    const float * sigma_layer = tqp_cuda_sigma_ptr(state, D) + (size_t)layer * D;

    if (rot == TQP_ROT_WHT) {
        tqp_prepare_query_kernel<D, TQP_ROT_WHT><<<1, D, 0, stream>>>(
            q, Sq, q_rot, s_layer, sigma_layer, pi_layer);
    } else {
        tqp_prepare_query_kernel<D, TQP_ROT_HAAR><<<1, D, 0, stream>>>(
            q, Sq, q_rot, s_layer, sigma_layer, pi_layer);
    }
}

template<int D>
static void ggml_cuda_tqp_prepare_query_batch_impl(
        const float * q,
        float * Sq,
        float * q_rot,
        int64_t ne11,
        int64_t ne12,
        int64_t ne13,
        int64_t s11,
        int64_t s12,
        int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream) {
    if (tqp_cuda_init(D, 3) != cudaSuccess) {
        return;
    }
    const TqpDeviceState * state = tqp_cuda_current_device_state();
    if (!state) {
        return;
    }

    const int layer = (int)(TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS);
    const uint8_t rot = TQP_EXTRACT_ROT(layer_byte);
    const float * pi_layer = tqp_cuda_pi_ptr(state, D) + (size_t)layer * D * D;
    const float * s_layer = tqp_cuda_s_ptr(state, D) + (size_t)layer * D * D;
    const float * sigma_layer = tqp_cuda_sigma_ptr(state, D) + (size_t)layer * D;
    const dim3 grid((unsigned int)ne11, (unsigned int)ne12, (unsigned int)ne13);

    if (rot == TQP_ROT_WHT) {
        tqp_prepare_query_batch_kernel<D, TQP_ROT_WHT><<<grid, D, 0, stream>>>(
            q, Sq, q_rot, ne11, ne12, s11, s12, s13, s_layer, sigma_layer, pi_layer);
    } else {
        tqp_prepare_query_batch_kernel<D, TQP_ROT_HAAR><<<grid, D, 0, stream>>>(
            q, Sq, q_rot, ne11, ne12, s11, s12, s13, s_layer, sigma_layer, pi_layer);
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

#define TQP_DEFINE_PREPARE_QUERY(D, BITS)                                                                                      \
    extern "C" void ggml_cuda_tqp_prepare_query_d##D##_b##BITS(                                                                \
            const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream) {                            \
        ggml_cuda_tqp_prepare_query_impl<QK_TQP_D##D>(q, Sq, q_rot, layer_byte, stream);                                      \
    }                                                                                                                           \
    extern "C" void ggml_cuda_tqp_prepare_query_batch_d##D##_b##BITS(                                                          \
            const float * q, float * Sq, float * q_rot,                                                                        \
            int64_t ne11, int64_t ne12, int64_t ne13,                                                                          \
            int64_t s11, int64_t s12, int64_t s13,                                                                             \
            uint8_t layer_byte,                                                                                                \
            cudaStream_t stream) {                                                                                              \
        ggml_cuda_tqp_prepare_query_batch_impl<QK_TQP_D##D>(                                                                   \
            q, Sq, q_rot, ne11, ne12, ne13, s11, s12, s13, layer_byte, stream);                                               \
    }                                                                                                                           \
    extern "C" int tqp_cuda_prepare_query_d##D##_b##BITS(                                                                      \
            const float * q_host, float * Sq_host, float * q_rot_host, uint8_t layer_byte) {                                  \
        return tqp_cuda_prepare_query_host(                                                                                     \
            QK_TQP_D##D, q_host, Sq_host, q_rot_host, layer_byte, ggml_cuda_tqp_prepare_query_d##D##_b##BITS);               \
    }

TQP_DEFINE_PREPARE_QUERY(128, 2)
TQP_DEFINE_PREPARE_QUERY(128, 3)
TQP_DEFINE_PREPARE_QUERY(128, 4)
TQP_DEFINE_PREPARE_QUERY(256, 2)
TQP_DEFINE_PREPARE_QUERY(256, 3)
TQP_DEFINE_PREPARE_QUERY(256, 4)

#undef TQP_DEFINE_PREPARE_QUERY

extern "C" void ggml_cuda_tqp_prepare_query_d128(
        const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream) {
    ggml_cuda_tqp_prepare_query_d128_b3(q, Sq, q_rot, layer_byte, stream);
}

extern "C" void ggml_cuda_tqp_prepare_query_d256(
        const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream) {
    ggml_cuda_tqp_prepare_query_d256_b3(q, Sq, q_rot, layer_byte, stream);
}

extern "C" void ggml_cuda_tqp_prepare_query_batch_d128(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream) {
    ggml_cuda_tqp_prepare_query_batch_d128_b3(
        q, Sq, q_rot, ne11, ne12, ne13, s11, s12, s13, layer_byte, stream);
}

extern "C" void ggml_cuda_tqp_prepare_query_batch_d256(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream) {
    ggml_cuda_tqp_prepare_query_batch_d256_b3(
        q, Sq, q_rot, ne11, ne12, ne13, s11, s12, s13, layer_byte, stream);
}

extern "C" int tqp_cuda_prepare_query_d128(
        const float * q_host, float * Sq_host, float * q_rot_host, uint8_t layer_byte) {
    return tqp_cuda_prepare_query_d128_b3(q_host, Sq_host, q_rot_host, layer_byte);
}

extern "C" int tqp_cuda_prepare_query_d256(
        const float * q_host, float * Sq_host, float * q_rot_host, uint8_t layer_byte) {
    return tqp_cuda_prepare_query_d256_b3(q_host, Sq_host, q_rot_host, layer_byte);
}
