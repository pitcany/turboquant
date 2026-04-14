#pragma once

// TurboQuant(-ish) CUDA constants.
//
// Per block, the rotation mode Π is user-selectable:
//   TQP_ROT_WHT : (1/√d)·H·diag(σ) with σ_i in device global memory.
//   TQP_ROT_HAAR: dense d×d Haar Π_i in device global memory (paper-exact).
//
// All per-layer arrays (σ_i, Π_i, S_i, centroids, boundaries) live in
// device global memory via TqpDeviceState pointers, allocated once per
// device. The block's layer_idx byte packs (rotation << 7) | layer.

#include "tqp-kernels.cuh"

#include "tqp_centroids_d128.h"
#include "tqp_centroids_d256.h"
#include "tqp_constants_d128.h"
#include "tqp_constants_d256.h"

#include <cuda_runtime.h>

#include <stddef.h>
#include <array>
#include <mutex>

struct TqpDeviceState {
    float * pi_d128 = nullptr;
    float * s_d128 = nullptr;
    float * pi_d256 = nullptr;
    float * s_d256 = nullptr;
    // Formerly static __constant__, now device global memory to avoid
    // per-TU symbol divergence with whole-program CUDA compilation.
    float * sigma_d128 = nullptr;      // [TQP_MAX_LAYERS][QK_TQ4P_D128]
    float * sigma_d256 = nullptr;      // [TQP_MAX_LAYERS][QK_TQ4P_D256]
    float * centroids_d128 = nullptr;  // [8]
    float * boundaries_d128 = nullptr; // [7]
    float * centroids_d256 = nullptr;  // [8]
    float * boundaries_d256 = nullptr; // [7]
    bool init_d128 = false;
    bool init_d256 = false;
};

inline constexpr int TQP_MAX_CUDA_DEVICES = 8;
inline std::array<TqpDeviceState, TQP_MAX_CUDA_DEVICES> g_tqp_devices{};
inline std::mutex g_tqp_init_mutex;

static inline cudaError_t tqp_cuda_current_device(int * device_id) {
    const cudaError_t err = cudaGetDevice(device_id);
    if (err != cudaSuccess) {
        return err;
    }
    if (*device_id < 0 || *device_id >= TQP_MAX_CUDA_DEVICES) {
        return cudaErrorInvalidDevice;
    }
    return cudaSuccess;
}

static inline TqpDeviceState * tqp_cuda_current_device_state() {
    int device_id = 0;
    if (tqp_cuda_current_device(&device_id) != cudaSuccess) {
        return nullptr;
    }
    return &g_tqp_devices[(size_t)device_id];
}

static inline cudaError_t tqp_cuda_init(int head_dim) {
    int device_id = 0;
    cudaError_t err = tqp_cuda_current_device(&device_id);
    if (err != cudaSuccess) {
        return err;
    }

    std::lock_guard<std::mutex> lock(g_tqp_init_mutex);
    TqpDeviceState & state = g_tqp_devices[(size_t)device_id];

    if (head_dim == QK_TQ4P_D128) {
        if (!state.init_d128) {
            float * pi = nullptr;
            float * s  = nullptr;
            float * sigma = nullptr;
            float * centroids = nullptr;
            float * boundaries = nullptr;

            err = cudaMalloc((void **)&pi, sizeof(TQP_PI_D128));
            if (err != cudaSuccess) return err;
            err = cudaMalloc((void **)&s, sizeof(TQP_S_D128));
            if (err != cudaSuccess) { cudaFree(pi); return err; }
            err = cudaMalloc((void **)&sigma, sizeof(TQP_SIGMA_D128));
            if (err != cudaSuccess) { cudaFree(s); cudaFree(pi); return err; }
            err = cudaMalloc((void **)&centroids, sizeof(TQP_CENTROIDS_D128));
            if (err != cudaSuccess) { cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }
            err = cudaMalloc((void **)&boundaries, sizeof(TQP_BOUNDARIES_D128));
            if (err != cudaSuccess) { cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }

            err = cudaMemcpy(pi, TQP_PI_D128, sizeof(TQP_PI_D128), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(boundaries); cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }
            err = cudaMemcpy(s, TQP_S_D128, sizeof(TQP_S_D128), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(boundaries); cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }
            err = cudaMemcpy(sigma, TQP_SIGMA_D128, sizeof(TQP_SIGMA_D128), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(boundaries); cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }
            err = cudaMemcpy(centroids, TQP_CENTROIDS_D128, sizeof(TQP_CENTROIDS_D128), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(boundaries); cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }
            err = cudaMemcpy(boundaries, TQP_BOUNDARIES_D128, sizeof(TQP_BOUNDARIES_D128), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(boundaries); cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }

            state.pi_d128 = pi;
            state.s_d128  = s;
            state.sigma_d128 = sigma;
            state.centroids_d128 = centroids;
            state.boundaries_d128 = boundaries;
            state.init_d128 = true;
        }
        return cudaSuccess;
    }

    if (head_dim == QK_TQ4P_D256) {
        if (!state.init_d256) {
            float * pi = nullptr;
            float * s  = nullptr;
            float * sigma = nullptr;
            float * centroids = nullptr;
            float * boundaries = nullptr;

            err = cudaMalloc((void **)&pi, sizeof(TQP_PI_D256));
            if (err != cudaSuccess) return err;
            err = cudaMalloc((void **)&s, sizeof(TQP_S_D256));
            if (err != cudaSuccess) { cudaFree(pi); return err; }
            err = cudaMalloc((void **)&sigma, sizeof(TQP_SIGMA_D256));
            if (err != cudaSuccess) { cudaFree(s); cudaFree(pi); return err; }
            err = cudaMalloc((void **)&centroids, sizeof(TQP_CENTROIDS_D256));
            if (err != cudaSuccess) { cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }
            err = cudaMalloc((void **)&boundaries, sizeof(TQP_BOUNDARIES_D256));
            if (err != cudaSuccess) { cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }

            err = cudaMemcpy(pi, TQP_PI_D256, sizeof(TQP_PI_D256), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(boundaries); cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }
            err = cudaMemcpy(s, TQP_S_D256, sizeof(TQP_S_D256), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(boundaries); cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }
            err = cudaMemcpy(sigma, TQP_SIGMA_D256, sizeof(TQP_SIGMA_D256), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(boundaries); cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }
            err = cudaMemcpy(centroids, TQP_CENTROIDS_D256, sizeof(TQP_CENTROIDS_D256), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(boundaries); cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }
            err = cudaMemcpy(boundaries, TQP_BOUNDARIES_D256, sizeof(TQP_BOUNDARIES_D256), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(boundaries); cudaFree(centroids); cudaFree(sigma); cudaFree(s); cudaFree(pi); return err; }

            state.pi_d256 = pi;
            state.s_d256  = s;
            state.sigma_d256 = sigma;
            state.centroids_d256 = centroids;
            state.boundaries_d256 = boundaries;
            state.init_d256 = true;
        }
        return cudaSuccess;
    }

    return cudaErrorInvalidValue;
}

static inline void tqp_cuda_cleanup() {
    int device_id = 0;
    if (tqp_cuda_current_device(&device_id) != cudaSuccess) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_tqp_init_mutex);
    TqpDeviceState & state = g_tqp_devices[(size_t)device_id];

    if (state.pi_d128)         cudaFree(state.pi_d128);
    if (state.s_d128)          cudaFree(state.s_d128);
    if (state.sigma_d128)      cudaFree(state.sigma_d128);
    if (state.centroids_d128)  cudaFree(state.centroids_d128);
    if (state.boundaries_d128) cudaFree(state.boundaries_d128);
    if (state.pi_d256)         cudaFree(state.pi_d256);
    if (state.s_d256)          cudaFree(state.s_d256);
    if (state.sigma_d256)      cudaFree(state.sigma_d256);
    if (state.centroids_d256)  cudaFree(state.centroids_d256);
    if (state.boundaries_d256) cudaFree(state.boundaries_d256);

    state = TqpDeviceState{};
}

static inline void tqp_cuda_cleanup_all() {
    int original_device = 0;
    const cudaError_t original_err = cudaGetDevice(&original_device);

    std::lock_guard<std::mutex> lock(g_tqp_init_mutex);
    for (int device_id = 0; device_id < TQP_MAX_CUDA_DEVICES; ++device_id) {
        TqpDeviceState & state = g_tqp_devices[(size_t)device_id];
        if (!state.init_d128 && !state.init_d256) {
            continue;
        }

        if (cudaSetDevice(device_id) != cudaSuccess) {
            continue;
        }
        if (state.pi_d128)         cudaFree(state.pi_d128);
        if (state.s_d128)          cudaFree(state.s_d128);
        if (state.sigma_d128)      cudaFree(state.sigma_d128);
        if (state.centroids_d128)  cudaFree(state.centroids_d128);
        if (state.boundaries_d128) cudaFree(state.boundaries_d128);
        if (state.pi_d256)         cudaFree(state.pi_d256);
        if (state.s_d256)          cudaFree(state.s_d256);
        if (state.sigma_d256)      cudaFree(state.sigma_d256);
        if (state.centroids_d256)  cudaFree(state.centroids_d256);
        if (state.boundaries_d256) cudaFree(state.boundaries_d256);

        state = TqpDeviceState{};
    }

    if (original_err == cudaSuccess) {
        cudaSetDevice(original_device);
    }
}
