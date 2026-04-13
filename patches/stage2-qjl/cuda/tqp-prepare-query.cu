#include "tqp-constants-cuda.cuh"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>

template<int D>
__global__ static void tqp_prepare_query_kernel(
        const float * __restrict__ q,
        float * __restrict__ Sq,
        float * __restrict__ q_rot,
        const float * __restrict__ s,
        const float * __restrict__ pi) {
    __shared__ float q_smem[QK_TQ4P_D256];

    const int tid = threadIdx.x;
    q_smem[tid] = q[tid];
    __syncthreads();

    float acc_s = 0.0f;
    float acc_pi = 0.0f;
#pragma unroll 1
    for (int j = 0; j < D; ++j) {
        const float qj = q_smem[j];
        acc_s += __ldg(&s[tid * D + j]) * qj;
        acc_pi += __ldg(&pi[tid * D + j]) * qj;
    }

    Sq[tid] = acc_s;
    q_rot[tid] = acc_pi;
}

template<int D>
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
        const float * __restrict__ pi) {
    __shared__ float q_smem[QK_TQ4P_D256];

    const int tid = threadIdx.x;
    const int64_t col = (int64_t)blockIdx.x;
    const int64_t channel = (int64_t)blockIdx.y;
    const int64_t sample = (int64_t)blockIdx.z;
    const int64_t query_index = (sample * ne12 + channel) * ne11 + col;
    const float * q_i = q + col * s11 + channel * s12 + sample * s13;

    q_smem[tid] = q_i[tid];
    __syncthreads();

    float acc_s = 0.0f;
    float acc_pi = 0.0f;
#pragma unroll 1
    for (int j = 0; j < D; ++j) {
        const float qj = q_smem[j];
        acc_s += __ldg(&s[tid * D + j]) * qj;
        acc_pi += __ldg(&pi[tid * D + j]) * qj;
    }

    Sq[query_index * D + tid] = acc_s;
    q_rot[query_index * D + tid] = acc_pi;
}

extern "C" void ggml_cuda_tqp_prepare_query_d128(const float * q, float * Sq, float * q_rot, uint8_t layer_idx, cudaStream_t stream) {
    if (tqp_cuda_init(QK_TQ4P_D128) != cudaSuccess) {
        return;
    }
    const uint8_t li = TQP_LAYER_WRAP(layer_idx);
    tqp_prepare_query_kernel<QK_TQ4P_D128><<<1, QK_TQ4P_D128, 0, stream>>>(
        q, Sq, q_rot, tqp_s_d128_for_layer(li), tqp_pi_d128_for_layer(li));
}

extern "C" void ggml_cuda_tqp_prepare_query_d256(const float * q, float * Sq, float * q_rot, uint8_t layer_idx, cudaStream_t stream) {
    if (tqp_cuda_init(QK_TQ4P_D256) != cudaSuccess) {
        return;
    }
    const uint8_t li = TQP_LAYER_WRAP(layer_idx);
    tqp_prepare_query_kernel<QK_TQ4P_D256><<<1, QK_TQ4P_D256, 0, stream>>>(
        q, Sq, q_rot, tqp_s_d256_for_layer(li), tqp_pi_d256_for_layer(li));
}

extern "C" void ggml_cuda_tqp_prepare_query_batch_d128(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_idx,
        cudaStream_t stream) {
    if (tqp_cuda_init(QK_TQ4P_D128) != cudaSuccess) {
        return;
    }
    const uint8_t li = TQP_LAYER_WRAP(layer_idx);
    tqp_prepare_query_batch_kernel<QK_TQ4P_D128><<<dim3((unsigned int)ne11, (unsigned int)ne12, (unsigned int)ne13), QK_TQ4P_D128, 0, stream>>>(
        q, Sq, q_rot, ne11, ne12, s11, s12, s13,
        tqp_s_d128_for_layer(li), tqp_pi_d128_for_layer(li));
}

extern "C" void ggml_cuda_tqp_prepare_query_batch_d256(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_idx,
        cudaStream_t stream) {
    if (tqp_cuda_init(QK_TQ4P_D256) != cudaSuccess) {
        return;
    }
    const uint8_t li = TQP_LAYER_WRAP(layer_idx);
    tqp_prepare_query_batch_kernel<QK_TQ4P_D256><<<dim3((unsigned int)ne11, (unsigned int)ne12, (unsigned int)ne13), QK_TQ4P_D256, 0, stream>>>(
        q, Sq, q_rot, ne11, ne12, s11, s12, s13,
        tqp_s_d256_for_layer(li), tqp_pi_d256_for_layer(li));
}

static int tqp_cuda_prepare_query_host(
        int d,
        const float * q_host,
        float * Sq_host,
        float * q_rot_host,
        uint8_t layer_idx,
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
        device_fn(q_dev, Sq_dev, q_rot_dev, layer_idx, 0);
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

extern "C" int tqp_cuda_prepare_query_d128(const float * q_host, float * Sq_host, float * q_rot_host, uint8_t layer_idx) {
    return tqp_cuda_prepare_query_host(
        QK_TQ4P_D128, q_host, Sq_host, q_rot_host, layer_idx, ggml_cuda_tqp_prepare_query_d128);
}

extern "C" int tqp_cuda_prepare_query_d256(const float * q_host, float * Sq_host, float * q_rot_host, uint8_t layer_idx) {
    return tqp_cuda_prepare_query_host(
        QK_TQ4P_D256, q_host, Sq_host, q_rot_host, layer_idx, ggml_cuda_tqp_prepare_query_d256);
}
