#pragma once

// TurboQuant(-ish) CUDA constants.
//
// Per block, the rotation mode Pi is user-selectable:
//   TQP_ROT_WHT : (1/sqrt(d)) * H * diag(sigma)
//   TQP_ROT_HAAR: dense dxd Haar Pi_i
//
// All per-layer arrays (sigma_i, Pi_i, S_i) live in device global memory
// via TqpDeviceState pointers, allocated once per device and per head_dim.
// Stage-1 centroids/boundaries are allocated per (head_dim, bits).

#include "tqp-kernels.cuh"

#include "tqp_centroids_d128_b2.h"
#include "tqp_centroids_d128_b3.h"
#include "tqp_centroids_d128_b4.h"
#include "tqp_centroids_d256_b2.h"
#include "tqp_centroids_d256_b3.h"
#include "tqp_centroids_d256_b4.h"
#include "tqp_constants_d128.h"
#include "tqp_constants_d256.h"

#include <cuda_runtime.h>

#include <array>
#include <mutex>
#include <stddef.h>

inline constexpr int TQP_CUDA_BITS_MIN = 2;
inline constexpr int TQP_CUDA_BITS_MAX = 4;
inline constexpr int TQP_CUDA_BITS_VARIANTS = TQP_CUDA_BITS_MAX - TQP_CUDA_BITS_MIN + 1;

struct TqpDeviceState {
    float * pi_d128 = nullptr;
    float * s_d128 = nullptr;
    float * pi_d256 = nullptr;
    float * s_d256 = nullptr;
    float * sigma_d128 = nullptr; // [TQP_MAX_LAYERS][QK_TQP_D128]
    float * sigma_d256 = nullptr; // [TQP_MAX_LAYERS][QK_TQP_D256]
    std::array<float *, TQP_CUDA_BITS_VARIANTS> centroids_d128{};
    std::array<float *, TQP_CUDA_BITS_VARIANTS> boundaries_d128{};
    std::array<float *, TQP_CUDA_BITS_VARIANTS> centroids_d256{};
    std::array<float *, TQP_CUDA_BITS_VARIANTS> boundaries_d256{};
    bool init_common_d128 = false;
    bool init_common_d256 = false;
    std::array<bool, TQP_CUDA_BITS_VARIANTS> init_bits_d128{};
    std::array<bool, TQP_CUDA_BITS_VARIANTS> init_bits_d256{};
};

inline constexpr int TQP_MAX_CUDA_DEVICES = 8;
inline std::array<TqpDeviceState, TQP_MAX_CUDA_DEVICES> g_tqp_devices{};
inline std::mutex g_tqp_init_mutex;

static inline int tqp_cuda_bits_slot(int bits) {
    if (bits < TQP_CUDA_BITS_MIN || bits > TQP_CUDA_BITS_MAX) {
        return -1;
    }
    return bits - TQP_CUDA_BITS_MIN;
}

static inline size_t tqp_cuda_centroid_count(int bits) {
    return (size_t)(1u << bits);
}

static inline size_t tqp_cuda_boundary_count(int bits) {
    return tqp_cuda_centroid_count(bits) - 1;
}

static inline const float * tqp_cuda_host_centroids(int head_dim, int bits) {
    switch (head_dim) {
        case QK_TQP_D128:
            switch (bits) {
                case 2: return TQP_CENTROIDS_D128_B2;
                case 3: return TQP_CENTROIDS_D128_B3;
                case 4: return TQP_CENTROIDS_D128_B4;
            }
            break;
        case QK_TQP_D256:
            switch (bits) {
                case 2: return TQP_CENTROIDS_D256_B2;
                case 3: return TQP_CENTROIDS_D256_B3;
                case 4: return TQP_CENTROIDS_D256_B4;
            }
            break;
    }
    return nullptr;
}

static inline const float * tqp_cuda_host_boundaries(int head_dim, int bits) {
    switch (head_dim) {
        case QK_TQP_D128:
            switch (bits) {
                case 2: return TQP_BOUNDARIES_D128_B2;
                case 3: return TQP_BOUNDARIES_D128_B3;
                case 4: return TQP_BOUNDARIES_D128_B4;
            }
            break;
        case QK_TQP_D256:
            switch (bits) {
                case 2: return TQP_BOUNDARIES_D256_B2;
                case 3: return TQP_BOUNDARIES_D256_B3;
                case 4: return TQP_BOUNDARIES_D256_B4;
            }
            break;
    }
    return nullptr;
}

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

static inline float * tqp_cuda_centroids_ptr(const TqpDeviceState * state, int head_dim, int bits) {
    const int slot = tqp_cuda_bits_slot(bits);
    if (!state || slot < 0) {
        return nullptr;
    }
    switch (head_dim) {
        case QK_TQP_D128: return state->centroids_d128[(size_t)slot];
        case QK_TQP_D256: return state->centroids_d256[(size_t)slot];
        default: return nullptr;
    }
}

static inline float * tqp_cuda_boundaries_ptr(const TqpDeviceState * state, int head_dim, int bits) {
    const int slot = tqp_cuda_bits_slot(bits);
    if (!state || slot < 0) {
        return nullptr;
    }
    switch (head_dim) {
        case QK_TQP_D128: return state->boundaries_d128[(size_t)slot];
        case QK_TQP_D256: return state->boundaries_d256[(size_t)slot];
        default: return nullptr;
    }
}

static inline float * tqp_cuda_pi_ptr(const TqpDeviceState * state, int head_dim) {
    if (!state) {
        return nullptr;
    }
    return head_dim == QK_TQP_D128 ? state->pi_d128 : head_dim == QK_TQP_D256 ? state->pi_d256 : nullptr;
}

static inline float * tqp_cuda_s_ptr(const TqpDeviceState * state, int head_dim) {
    if (!state) {
        return nullptr;
    }
    return head_dim == QK_TQP_D128 ? state->s_d128 : head_dim == QK_TQP_D256 ? state->s_d256 : nullptr;
}

static inline float * tqp_cuda_sigma_ptr(const TqpDeviceState * state, int head_dim) {
    if (!state) {
        return nullptr;
    }
    return head_dim == QK_TQP_D128 ? state->sigma_d128 : head_dim == QK_TQP_D256 ? state->sigma_d256 : nullptr;
}

static inline cudaError_t tqp_cuda_init_common_d128(TqpDeviceState & state) {
    if (state.init_common_d128) {
        return cudaSuccess;
    }

    float * pi = nullptr;
    float * s = nullptr;
    float * sigma = nullptr;
    cudaError_t err = cudaMalloc((void **)&pi, sizeof(TQP_PI_D128));
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void **)&s, sizeof(TQP_S_D128));
    if (err != cudaSuccess) {
        cudaFree(pi);
        return err;
    }
    err = cudaMalloc((void **)&sigma, sizeof(TQP_SIGMA_D128));
    if (err != cudaSuccess) {
        cudaFree(s);
        cudaFree(pi);
        return err;
    }

    err = cudaMemcpy(pi, TQP_PI_D128, sizeof(TQP_PI_D128), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(sigma);
        cudaFree(s);
        cudaFree(pi);
        return err;
    }
    err = cudaMemcpy(s, TQP_S_D128, sizeof(TQP_S_D128), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(sigma);
        cudaFree(s);
        cudaFree(pi);
        return err;
    }
    err = cudaMemcpy(sigma, TQP_SIGMA_D128, sizeof(TQP_SIGMA_D128), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(sigma);
        cudaFree(s);
        cudaFree(pi);
        return err;
    }

    state.pi_d128 = pi;
    state.s_d128 = s;
    state.sigma_d128 = sigma;
    state.init_common_d128 = true;
    return cudaSuccess;
}

static inline cudaError_t tqp_cuda_init_common_d256(TqpDeviceState & state) {
    if (state.init_common_d256) {
        return cudaSuccess;
    }

    float * pi = nullptr;
    float * s = nullptr;
    float * sigma = nullptr;
    cudaError_t err = cudaMalloc((void **)&pi, sizeof(TQP_PI_D256));
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void **)&s, sizeof(TQP_S_D256));
    if (err != cudaSuccess) {
        cudaFree(pi);
        return err;
    }
    err = cudaMalloc((void **)&sigma, sizeof(TQP_SIGMA_D256));
    if (err != cudaSuccess) {
        cudaFree(s);
        cudaFree(pi);
        return err;
    }

    err = cudaMemcpy(pi, TQP_PI_D256, sizeof(TQP_PI_D256), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(sigma);
        cudaFree(s);
        cudaFree(pi);
        return err;
    }
    err = cudaMemcpy(s, TQP_S_D256, sizeof(TQP_S_D256), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(sigma);
        cudaFree(s);
        cudaFree(pi);
        return err;
    }
    err = cudaMemcpy(sigma, TQP_SIGMA_D256, sizeof(TQP_SIGMA_D256), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(sigma);
        cudaFree(s);
        cudaFree(pi);
        return err;
    }

    state.pi_d256 = pi;
    state.s_d256 = s;
    state.sigma_d256 = sigma;
    state.init_common_d256 = true;
    return cudaSuccess;
}

static inline cudaError_t tqp_cuda_init_bits_d128(TqpDeviceState & state, int bits) {
    const int slot = tqp_cuda_bits_slot(bits);
    if (slot < 0) {
        return cudaErrorInvalidValue;
    }
    if (state.init_bits_d128[(size_t)slot]) {
        return cudaSuccess;
    }

    const float * host_centroids = tqp_cuda_host_centroids(QK_TQP_D128, bits);
    const float * host_boundaries = tqp_cuda_host_boundaries(QK_TQP_D128, bits);
    if (!host_centroids || !host_boundaries) {
        return cudaErrorInvalidValue;
    }

    float * centroids = nullptr;
    float * boundaries = nullptr;
    cudaError_t err = cudaMalloc((void **)&centroids, tqp_cuda_centroid_count(bits) * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void **)&boundaries, tqp_cuda_boundary_count(bits) * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(centroids);
        return err;
    }

    err = cudaMemcpy(
        centroids,
        host_centroids,
        tqp_cuda_centroid_count(bits) * sizeof(float),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(boundaries);
        cudaFree(centroids);
        return err;
    }
    err = cudaMemcpy(
        boundaries,
        host_boundaries,
        tqp_cuda_boundary_count(bits) * sizeof(float),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(boundaries);
        cudaFree(centroids);
        return err;
    }

    state.centroids_d128[(size_t)slot] = centroids;
    state.boundaries_d128[(size_t)slot] = boundaries;
    state.init_bits_d128[(size_t)slot] = true;
    return cudaSuccess;
}

static inline cudaError_t tqp_cuda_init_bits_d256(TqpDeviceState & state, int bits) {
    const int slot = tqp_cuda_bits_slot(bits);
    if (slot < 0) {
        return cudaErrorInvalidValue;
    }
    if (state.init_bits_d256[(size_t)slot]) {
        return cudaSuccess;
    }

    const float * host_centroids = tqp_cuda_host_centroids(QK_TQP_D256, bits);
    const float * host_boundaries = tqp_cuda_host_boundaries(QK_TQP_D256, bits);
    if (!host_centroids || !host_boundaries) {
        return cudaErrorInvalidValue;
    }

    float * centroids = nullptr;
    float * boundaries = nullptr;
    cudaError_t err = cudaMalloc((void **)&centroids, tqp_cuda_centroid_count(bits) * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc((void **)&boundaries, tqp_cuda_boundary_count(bits) * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(centroids);
        return err;
    }

    err = cudaMemcpy(
        centroids,
        host_centroids,
        tqp_cuda_centroid_count(bits) * sizeof(float),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(boundaries);
        cudaFree(centroids);
        return err;
    }
    err = cudaMemcpy(
        boundaries,
        host_boundaries,
        tqp_cuda_boundary_count(bits) * sizeof(float),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(boundaries);
        cudaFree(centroids);
        return err;
    }

    state.centroids_d256[(size_t)slot] = centroids;
    state.boundaries_d256[(size_t)slot] = boundaries;
    state.init_bits_d256[(size_t)slot] = true;
    return cudaSuccess;
}

// No default for `bits` — all call sites must pass an explicit bit width.
static inline cudaError_t tqp_cuda_init(int head_dim, int bits) {
    int device_id = 0;
    cudaError_t err = tqp_cuda_current_device(&device_id);
    if (err != cudaSuccess) {
        return err;
    }

    std::lock_guard<std::mutex> lock(g_tqp_init_mutex);
    TqpDeviceState & state = g_tqp_devices[(size_t)device_id];

    switch (head_dim) {
        case QK_TQP_D128:
            err = tqp_cuda_init_common_d128(state);
            if (err != cudaSuccess) return err;
            return tqp_cuda_init_bits_d128(state, bits);
        case QK_TQP_D256:
            err = tqp_cuda_init_common_d256(state);
            if (err != cudaSuccess) return err;
            return tqp_cuda_init_bits_d256(state, bits);
        default:
            return cudaErrorInvalidValue;
    }
}

static inline void tqp_cuda_cleanup() {
    int device_id = 0;
    if (tqp_cuda_current_device(&device_id) != cudaSuccess) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_tqp_init_mutex);
    TqpDeviceState & state = g_tqp_devices[(size_t)device_id];

    if (state.pi_d128) cudaFree(state.pi_d128);
    if (state.s_d128) cudaFree(state.s_d128);
    if (state.sigma_d128) cudaFree(state.sigma_d128);
    if (state.pi_d256) cudaFree(state.pi_d256);
    if (state.s_d256) cudaFree(state.s_d256);
    if (state.sigma_d256) cudaFree(state.sigma_d256);

    for (size_t i = 0; i < TQP_CUDA_BITS_VARIANTS; ++i) {
        if (state.centroids_d128[i]) cudaFree(state.centroids_d128[i]);
        if (state.boundaries_d128[i]) cudaFree(state.boundaries_d128[i]);
        if (state.centroids_d256[i]) cudaFree(state.centroids_d256[i]);
        if (state.boundaries_d256[i]) cudaFree(state.boundaries_d256[i]);
    }

    state = TqpDeviceState{};
}

static inline void tqp_cuda_cleanup_all() {
    int original_device = 0;
    const cudaError_t original_err = cudaGetDevice(&original_device);

    std::lock_guard<std::mutex> lock(g_tqp_init_mutex);
    for (int device_id = 0; device_id < TQP_MAX_CUDA_DEVICES; ++device_id) {
        TqpDeviceState & state = g_tqp_devices[(size_t)device_id];
        if (!state.init_common_d128 && !state.init_common_d256) {
            bool has_bits = false;
            for (size_t i = 0; i < TQP_CUDA_BITS_VARIANTS; ++i) {
                has_bits = has_bits || state.init_bits_d128[i] || state.init_bits_d256[i];
            }
            if (!has_bits) {
                continue;
            }
        }

        if (cudaSetDevice(device_id) != cudaSuccess) {
            continue;
        }

        if (state.pi_d128) cudaFree(state.pi_d128);
        if (state.s_d128) cudaFree(state.s_d128);
        if (state.sigma_d128) cudaFree(state.sigma_d128);
        if (state.pi_d256) cudaFree(state.pi_d256);
        if (state.s_d256) cudaFree(state.s_d256);
        if (state.sigma_d256) cudaFree(state.sigma_d256);

        for (size_t i = 0; i < TQP_CUDA_BITS_VARIANTS; ++i) {
            if (state.centroids_d128[i]) cudaFree(state.centroids_d128[i]);
            if (state.boundaries_d128[i]) cudaFree(state.boundaries_d128[i]);
            if (state.centroids_d256[i]) cudaFree(state.centroids_d256[i]);
            if (state.boundaries_d256[i]) cudaFree(state.boundaries_d256[i]);
        }

        state = TqpDeviceState{};
    }

    if (original_err == cudaSuccess) {
        cudaSetDevice(original_device);
    }
}
