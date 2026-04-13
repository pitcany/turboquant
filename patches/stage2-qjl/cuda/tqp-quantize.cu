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

// tqp_quantize_block_device is now in tqp-kernels.cuh so it can be shared
// with tqp-set-rows.cu. The kernel wrappers below call it directly.

// Per-layer sigma lives in __constant__ memory; the kernel reads it by
// symbol rather than via a kernel-argument pointer (host-side addresses of
// __constant__ symbols are not usable as device pointers). Π lives in
// device global memory via `pi`. ROT is a template parameter so the dead
// branch is compiled out.
template<uint8_t ROT>
__global__ static void tqp_quantize_kernel_d128(
        const float * __restrict__ x,
        block_tq4p_d128 * __restrict__ y,
        uint8_t layer_byte_val,
        uint8_t layer,
        const float * __restrict__ pi,
        const float * __restrict__ s,
        int64_t n_blocks) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) {
        return;
    }

    tqp_quantize_block_device<QK_TQ4P_D128, ROT>(
        x + b * QK_TQ4P_D128,
        &y[b].orig_norm,
        &y[b].res_d,
        &y[b].layer_idx,
        layer_byte_val,
        y[b].qs,
        y[b].qjl_signs,
        &c_tqp_sigma_d128[layer][0],
        pi + (size_t)layer * QK_TQ4P_D128 * QK_TQ4P_D128,
        s  + (size_t)layer * QK_TQ4P_D128 * QK_TQ4P_D128,
        c_tqp_centroids_d128,
        c_tqp_boundaries_d128);
}

template<uint8_t ROT>
__global__ static void tqp_quantize_kernel_d256(
        const float * __restrict__ x,
        block_tq4p_d256 * __restrict__ y,
        uint8_t layer_byte_val,
        uint8_t layer,
        const float * __restrict__ pi,
        const float * __restrict__ s,
        int64_t n_blocks) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) {
        return;
    }

    tqp_quantize_block_device<QK_TQ4P_D256, ROT>(
        x + b * QK_TQ4P_D256,
        &y[b].orig_norm,
        &y[b].res_d,
        &y[b].layer_idx,
        layer_byte_val,
        y[b].qs,
        y[b].qjl_signs,
        &c_tqp_sigma_d256[layer][0],
        pi + (size_t)layer * QK_TQ4P_D256 * QK_TQ4P_D256,
        s  + (size_t)layer * QK_TQ4P_D256 * QK_TQ4P_D256,
        c_tqp_centroids_d256,
        c_tqp_boundaries_d256);
}

extern "C" void ggml_cuda_tqp_quantize_row_d128(const float * x, void * y, int64_t k, uint8_t layer_byte, cudaStream_t stream) {
    if (k % QK_TQ4P_D128 != 0) {
        return;
    }
    if (tqp_cuda_init(QK_TQ4P_D128) != cudaSuccess) {
        return;
    }
    const uint8_t layer = TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS;
    const uint8_t rot   = TQP_EXTRACT_ROT(layer_byte);
    const uint8_t byte_stored = TQP_STORED_BYTE(layer, rot);
    const int64_t n_blocks = k / QK_TQ4P_D128;
    if (rot == TQP_ROT_WHT) {
        tqp_quantize_kernel_d128<TQP_ROT_WHT><<<(unsigned int)n_blocks, QK_TQ4P_D128, 0, stream>>>(
            x, (block_tq4p_d128 *)y, byte_stored, layer, d_tqp_pi_d128, d_tqp_s_d128, n_blocks);
    } else {
        tqp_quantize_kernel_d128<TQP_ROT_HAAR><<<(unsigned int)n_blocks, QK_TQ4P_D128, 0, stream>>>(
            x, (block_tq4p_d128 *)y, byte_stored, layer, d_tqp_pi_d128, d_tqp_s_d128, n_blocks);
    }
}

extern "C" void ggml_cuda_tqp_quantize_row_d256(const float * x, void * y, int64_t k, uint8_t layer_byte, cudaStream_t stream) {
    if (k % QK_TQ4P_D256 != 0) {
        return;
    }
    if (tqp_cuda_init(QK_TQ4P_D256) != cudaSuccess) {
        return;
    }
    const uint8_t layer = TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS;
    const uint8_t rot   = TQP_EXTRACT_ROT(layer_byte);
    const uint8_t byte_stored = TQP_STORED_BYTE(layer, rot);
    const int64_t n_blocks = k / QK_TQ4P_D256;
    if (rot == TQP_ROT_WHT) {
        tqp_quantize_kernel_d256<TQP_ROT_WHT><<<(unsigned int)n_blocks, QK_TQ4P_D256, 0, stream>>>(
            x, (block_tq4p_d256 *)y, byte_stored, layer, d_tqp_pi_d256, d_tqp_s_d256, n_blocks);
    } else {
        tqp_quantize_kernel_d256<TQP_ROT_HAAR><<<(unsigned int)n_blocks, QK_TQ4P_D256, 0, stream>>>(
            x, (block_tq4p_d256 *)y, byte_stored, layer, d_tqp_pi_d256, d_tqp_s_d256, n_blocks);
    }
}

template<typename Block>
static int tqp_cuda_quantize_row_host(
        int d,
        const float * x_host,
        void * y_host,
        int64_t k,
        uint8_t layer_byte,
        void (*device_fn)(const float *, void *, int64_t, uint8_t, cudaStream_t)) {
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
        device_fn(x_dev, y_dev, k, layer_byte, 0);
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

extern "C" int tqp_cuda_quantize_row_d128(const float * x_host, void * y_host, int64_t k, uint8_t layer_byte) {
    return tqp_cuda_quantize_row_host<block_tq4p_d128>(
        QK_TQ4P_D128, x_host, y_host, k, layer_byte, ggml_cuda_tqp_quantize_row_d128);
}

extern "C" int tqp_cuda_quantize_row_d256(const float * x_host, void * y_host, int64_t k, uint8_t layer_byte) {
    return tqp_cuda_quantize_row_host<block_tq4p_d256>(
        QK_TQ4P_D256, x_host, y_host, k, layer_byte, ggml_cuda_tqp_quantize_row_d256);
}
