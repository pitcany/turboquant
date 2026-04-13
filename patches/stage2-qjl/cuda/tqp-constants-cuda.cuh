#pragma once

// TurboQuant(-ish) CUDA constants.
//
// Per block, the rotation mode Π is user-selectable:
//   TQP_ROT_WHT : (1/√d)·H·diag(σ) with σ_i in __constant__ memory (cheap).
//   TQP_ROT_HAAR: dense d×d Haar Π_i in device global memory (paper-exact).
//
// Both σ_i and Π_i are generated per layer (32 layers × ...). The S_i
// Gaussian QJL matrix is shared between rotations and lives in device
// global memory. The block's layer_idx byte packs (rotation << 7) | layer.

#include "tqp-kernels.cuh"

#include "tqp_centroids_d128.h"
#include "tqp_centroids_d256.h"
#include "tqp_constants_d128.h"
#include "tqp_constants_d256.h"

#include <cuda_runtime.h>

#include <stddef.h>
#include <array>
#include <mutex>

static __constant__ float c_tqp_centroids_d128[8];
static __constant__ float c_tqp_boundaries_d128[7];
static __constant__ float c_tqp_centroids_d256[8];
static __constant__ float c_tqp_boundaries_d256[7];

// Per-layer ±1 sign vectors σ_i for TQP_ROT_WHT. Size: 32 × d × 4 B.
static __constant__ float c_tqp_sigma_d128[TQP_MAX_LAYERS][QK_TQ4P_D128];
static __constant__ float c_tqp_sigma_d256[TQP_MAX_LAYERS][QK_TQ4P_D256];

struct TqpDeviceState {
    float * pi_d128 = nullptr;
    float * s_d128 = nullptr;
    float * pi_d256 = nullptr;
    float * s_d256 = nullptr;
    bool init_d128 = false;
    bool init_d256 = false;
};

inline constexpr int TQP_MAX_CUDA_DEVICES = 8;
inline std::array<TqpDeviceState, TQP_MAX_CUDA_DEVICES> g_tqp_devices{};
inline std::mutex g_tqp_init_mutex;

static std::array<bool, TQP_MAX_CUDA_DEVICES> g_tqp_cuda_constants_init_d128{};
static std::array<bool, TQP_MAX_CUDA_DEVICES> g_tqp_cuda_constants_init_d256{};

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
        if (!g_tqp_cuda_constants_init_d128[(size_t)device_id]) {
            err = cudaMemcpyToSymbol(c_tqp_centroids_d128, TQP_CENTROIDS_D128, sizeof(TQP_CENTROIDS_D128));
            if (err != cudaSuccess) return err;
            err = cudaMemcpyToSymbol(c_tqp_boundaries_d128, TQP_BOUNDARIES_D128, sizeof(TQP_BOUNDARIES_D128));
            if (err != cudaSuccess) return err;
            err = cudaMemcpyToSymbol(c_tqp_sigma_d128, TQP_SIGMA_D128, sizeof(TQP_SIGMA_D128));
            if (err != cudaSuccess) return err;
            g_tqp_cuda_constants_init_d128[(size_t)device_id] = true;
        }
        if (!state.init_d128) {
            float * pi = nullptr;
            float * s  = nullptr;
            err = cudaMalloc((void **)&pi, sizeof(TQP_PI_D128));
            if (err != cudaSuccess) return err;
            err = cudaMalloc((void **)&s,  sizeof(TQP_S_D128));
            if (err != cudaSuccess) { cudaFree(pi); return err; }
            err = cudaMemcpy(pi, TQP_PI_D128, sizeof(TQP_PI_D128), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(s); cudaFree(pi); return err; }
            err = cudaMemcpy(s, TQP_S_D128, sizeof(TQP_S_D128), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(s); cudaFree(pi); return err; }
            state.pi_d128 = pi;
            state.s_d128  = s;
            state.init_d128 = true;
        }
        return cudaSuccess;
    }

    if (head_dim == QK_TQ4P_D256) {
        if (!g_tqp_cuda_constants_init_d256[(size_t)device_id]) {
            err = cudaMemcpyToSymbol(c_tqp_centroids_d256, TQP_CENTROIDS_D256, sizeof(TQP_CENTROIDS_D256));
            if (err != cudaSuccess) return err;
            err = cudaMemcpyToSymbol(c_tqp_boundaries_d256, TQP_BOUNDARIES_D256, sizeof(TQP_BOUNDARIES_D256));
            if (err != cudaSuccess) return err;
            err = cudaMemcpyToSymbol(c_tqp_sigma_d256, TQP_SIGMA_D256, sizeof(TQP_SIGMA_D256));
            if (err != cudaSuccess) return err;
            g_tqp_cuda_constants_init_d256[(size_t)device_id] = true;
        }
        if (!state.init_d256) {
            float * pi = nullptr;
            float * s  = nullptr;
            err = cudaMalloc((void **)&pi, sizeof(TQP_PI_D256));
            if (err != cudaSuccess) return err;
            err = cudaMalloc((void **)&s,  sizeof(TQP_S_D256));
            if (err != cudaSuccess) { cudaFree(pi); return err; }
            err = cudaMemcpy(pi, TQP_PI_D256, sizeof(TQP_PI_D256), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(s); cudaFree(pi); return err; }
            err = cudaMemcpy(s, TQP_S_D256, sizeof(TQP_S_D256), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(s); cudaFree(pi); return err; }
            state.pi_d256 = pi;
            state.s_d256  = s;
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

    if (state.pi_d128) cudaFree(state.pi_d128);
    if (state.s_d128)  cudaFree(state.s_d128);
    if (state.pi_d256) cudaFree(state.pi_d256);
    if (state.s_d256)  cudaFree(state.s_d256);

    state = TqpDeviceState{};
    g_tqp_cuda_constants_init_d128[(size_t)device_id] = false;
    g_tqp_cuda_constants_init_d256[(size_t)device_id] = false;
}

static inline void tqp_cuda_cleanup_all() {
    int original_device = 0;
    const cudaError_t original_err = cudaGetDevice(&original_device);

    std::lock_guard<std::mutex> lock(g_tqp_init_mutex);
    for (int device_id = 0; device_id < TQP_MAX_CUDA_DEVICES; ++device_id) {
        TqpDeviceState & state = g_tqp_devices[(size_t)device_id];
        if (!state.init_d128 && !state.init_d256
                && !g_tqp_cuda_constants_init_d128[(size_t)device_id]
                && !g_tqp_cuda_constants_init_d256[(size_t)device_id]) {
            continue;
        }

        if (cudaSetDevice(device_id) != cudaSuccess) {
            continue;
        }
        if (state.pi_d128) cudaFree(state.pi_d128);
        if (state.s_d128)  cudaFree(state.s_d128);
        if (state.pi_d256) cudaFree(state.pi_d256);
        if (state.s_d256)  cudaFree(state.s_d256);

        state = TqpDeviceState{};
        g_tqp_cuda_constants_init_d128[(size_t)device_id] = false;
        g_tqp_cuda_constants_init_d256[(size_t)device_id] = false;
    }

    if (original_err == cudaSuccess) {
        cudaSetDevice(original_device);
    }
}
