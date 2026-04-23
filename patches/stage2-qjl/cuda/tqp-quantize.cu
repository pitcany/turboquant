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

template<int D, int BITS, typename Block, uint8_t ROT>
__global__ static void tqp_quantize_kernel(
        const float * __restrict__ x,
        Block * __restrict__ y,
        uint8_t layer_byte_val,
        uint8_t layer,
        const float * __restrict__ pi,
        const float * __restrict__ s,
        const float * __restrict__ sigma,
        const float * __restrict__ centroids,
        const float * __restrict__ boundaries,
        int64_t n_blocks) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) {
        return;
    }

    tqp_quantize_block_device<D, ROT, BITS>(
        x + b * D,
        &y[b].orig_norm,
        &y[b].res_d,
        &y[b].layer_idx,
        layer_byte_val,
        y[b].qs,
        y[b].qjl_signs,
        sigma + (size_t)layer * D,
        pi + (size_t)layer * D * D,
        s + (size_t)layer * D * D,
        centroids,
        boundaries);
}

template<int D, int BITS, typename Block>
static void ggml_cuda_tqp_quantize_row_impl(
        const float * x,
        void * y,
        int64_t k,
        uint8_t layer_byte,
        cudaStream_t stream) {
    if (k % D != 0) {
        return;
    }
    if (tqp_cuda_init(D, BITS) != cudaSuccess) {
        return;
    }

    const TqpDeviceState * state = tqp_cuda_current_device_state();
    if (!state) {
        return;
    }

    const uint8_t layer = TQP_EXTRACT_LAYER(layer_byte) % TQP_MAX_LAYERS;
    const uint8_t rot = TQP_EXTRACT_ROT(layer_byte);
    const uint8_t byte_stored = TQP_STORED_BYTE(layer, rot);
    const int64_t n_blocks = k / D;

    const float * pi = tqp_cuda_pi_ptr(state, D);
    const float * s = tqp_cuda_s_ptr(state, D);
    const float * sigma = tqp_cuda_sigma_ptr(state, D);
    const float * centroids = tqp_cuda_centroids_ptr(state, D, BITS);
    const float * boundaries = tqp_cuda_boundaries_ptr(state, D, BITS);
    if (!pi || !s || !sigma || !centroids || !boundaries) {
        return;
    }

    if (rot == TQP_ROT_WHT) {
        tqp_quantize_kernel<D, BITS, Block, TQP_ROT_WHT><<<(unsigned int)n_blocks, D, 0, stream>>>(
            x,
            (Block *)y,
            byte_stored,
            layer,
            pi,
            s,
            sigma,
            centroids,
            boundaries,
            n_blocks);
    } else {
        tqp_quantize_kernel<D, BITS, Block, TQP_ROT_HAAR><<<(unsigned int)n_blocks, D, 0, stream>>>(
            x,
            (Block *)y,
            byte_stored,
            layer,
            pi,
            s,
            sigma,
            centroids,
            boundaries,
            n_blocks);
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

#define TQP_DEFINE_QUANTIZE(D, BITS)                                                                                             \
    extern "C" void ggml_cuda_tqp_quantize_row_d##D##_b##BITS(                                                                   \
            const float * x, void * y, int64_t k, uint8_t layer_byte, cudaStream_t stream) {                                    \
        ggml_cuda_tqp_quantize_row_impl<QK_TQP_D##D, BITS, block_tqp_d##D##_b##BITS>(x, y, k, layer_byte, stream);             \
    }                                                                                                                            \
    extern "C" int tqp_cuda_quantize_row_d##D##_b##BITS(                                                                         \
            const float * x_host, void * y_host, int64_t k, uint8_t layer_byte) {                                               \
        return tqp_cuda_quantize_row_host<block_tqp_d##D##_b##BITS>(                                                             \
            QK_TQP_D##D, x_host, y_host, k, layer_byte, ggml_cuda_tqp_quantize_row_d##D##_b##BITS);                            \
    }

TQP_DEFINE_QUANTIZE(64, 2)
TQP_DEFINE_QUANTIZE(64, 3)
TQP_DEFINE_QUANTIZE(64, 4)
TQP_DEFINE_QUANTIZE(128, 2)
TQP_DEFINE_QUANTIZE(128, 3)
TQP_DEFINE_QUANTIZE(128, 4)
TQP_DEFINE_QUANTIZE(256, 2)
TQP_DEFINE_QUANTIZE(256, 3)
TQP_DEFINE_QUANTIZE(256, 4)

#undef TQP_DEFINE_QUANTIZE

extern "C" void ggml_cuda_tqp_quantize_row_d64(
        const float * x, void * y, int64_t k, uint8_t layer_byte, cudaStream_t stream) {
    ggml_cuda_tqp_quantize_row_d64_b3(x, y, k, layer_byte, stream);
}

extern "C" void ggml_cuda_tqp_quantize_row_d128(
        const float * x, void * y, int64_t k, uint8_t layer_byte, cudaStream_t stream) {
    ggml_cuda_tqp_quantize_row_d128_b3(x, y, k, layer_byte, stream);
}

extern "C" void ggml_cuda_tqp_quantize_row_d256(
        const float * x, void * y, int64_t k, uint8_t layer_byte, cudaStream_t stream) {
    ggml_cuda_tqp_quantize_row_d256_b3(x, y, k, layer_byte, stream);
}

extern "C" int tqp_cuda_quantize_row_d64(
        const float * x_host, void * y_host, int64_t k, uint8_t layer_byte) {
    return tqp_cuda_quantize_row_d64_b3(x_host, y_host, k, layer_byte);
}

extern "C" int tqp_cuda_quantize_row_d128(
        const float * x_host, void * y_host, int64_t k, uint8_t layer_byte) {
    return tqp_cuda_quantize_row_d128_b3(x_host, y_host, k, layer_byte);
}

extern "C" int tqp_cuda_quantize_row_d256(
        const float * x_host, void * y_host, int64_t k, uint8_t layer_byte) {
    return tqp_cuda_quantize_row_d256_b3(x_host, y_host, k, layer_byte);
}
