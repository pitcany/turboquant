#pragma once

// TurboQuant-ish CUDA constants — WHT variant.
//
// Branch: the paper's Haar rotation Π is replaced with the Randomized
// Hadamard Transform Π = (1/√d) · H · diag(σ). The per-layer d×d rotation
// matrix is replaced with a per-layer ±1 sign vector σ of length d, stored
// in __constant__ memory (all 32 layers × d × 4 B ≤ 32 KB — fits cleanly
// in the constant cache for both d=128 and d=256).
//
// Per-layer σ_i and S_i are selected at launch time via the `layer_idx`
// byte in the block header (matching the CPU path). The block struct in
// tqp-kernels.cuh is size-aligned with the CPU header (69/133 B).

#include "tqp-kernels.cuh"

#include "tqp_centroids_d128.h"
#include "tqp_centroids_d256.h"
#include "tqp_constants_d128.h"
#include "tqp_constants_d256.h"

#include <cuda_runtime.h>

#include <stddef.h>

static __constant__ float c_tqp_centroids_d128[8];
static __constant__ float c_tqp_boundaries_d128[7];
static __constant__ float c_tqp_centroids_d256[8];
static __constant__ float c_tqp_boundaries_d256[7];

// Per-layer ±1 sign vectors σ_i. Size: 32 × d × 4 B.
static __constant__ float c_tqp_sigma_d128[TQP_MAX_LAYERS][QK_TQ4P_D128];
static __constant__ float c_tqp_sigma_d256[TQP_MAX_LAYERS][QK_TQ4P_D256];

// Gaussian QJL matrices S_i stay in device global memory (too large for
// __constant__: 32 × d² × 4 B = up to 8 MB for d=256).
inline float * d_tqp_s_d128  = nullptr;
inline float * d_tqp_s_d256  = nullptr;

inline bool g_tqp_cuda_init_d128 = false;
inline bool g_tqp_cuda_init_d256 = false;
static bool g_tqp_cuda_constants_init_d128 = false;
static bool g_tqp_cuda_constants_init_d256 = false;

static inline cudaError_t tqp_cuda_init(int head_dim) {
    cudaError_t err = cudaSuccess;

    if (head_dim == QK_TQ4P_D128) {
        if (!g_tqp_cuda_constants_init_d128) {
            err = cudaMemcpyToSymbol(c_tqp_centroids_d128, TQP_CENTROIDS_D128, sizeof(TQP_CENTROIDS_D128));
            if (err != cudaSuccess) return err;
            err = cudaMemcpyToSymbol(c_tqp_boundaries_d128, TQP_BOUNDARIES_D128, sizeof(TQP_BOUNDARIES_D128));
            if (err != cudaSuccess) return err;
            err = cudaMemcpyToSymbol(c_tqp_sigma_d128, TQP_SIGMA_D128, sizeof(TQP_SIGMA_D128));
            if (err != cudaSuccess) return err;
            g_tqp_cuda_constants_init_d128 = true;
        }
        if (!g_tqp_cuda_init_d128) {
            float * s = nullptr;
            err = cudaMalloc((void **)&s, sizeof(TQP_S_D128));
            if (err != cudaSuccess) return err;
            err = cudaMemcpy(s, TQP_S_D128, sizeof(TQP_S_D128), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(s);
                return err;
            }
            d_tqp_s_d128 = s;
            g_tqp_cuda_init_d128 = true;
        }
        return cudaSuccess;
    }

    if (head_dim == QK_TQ4P_D256) {
        if (!g_tqp_cuda_constants_init_d256) {
            err = cudaMemcpyToSymbol(c_tqp_centroids_d256, TQP_CENTROIDS_D256, sizeof(TQP_CENTROIDS_D256));
            if (err != cudaSuccess) return err;
            err = cudaMemcpyToSymbol(c_tqp_boundaries_d256, TQP_BOUNDARIES_D256, sizeof(TQP_BOUNDARIES_D256));
            if (err != cudaSuccess) return err;
            err = cudaMemcpyToSymbol(c_tqp_sigma_d256, TQP_SIGMA_D256, sizeof(TQP_SIGMA_D256));
            if (err != cudaSuccess) return err;
            g_tqp_cuda_constants_init_d256 = true;
        }
        if (!g_tqp_cuda_init_d256) {
            float * s = nullptr;
            err = cudaMalloc((void **)&s, sizeof(TQP_S_D256));
            if (err != cudaSuccess) return err;
            err = cudaMemcpy(s, TQP_S_D256, sizeof(TQP_S_D256), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(s);
                return err;
            }
            d_tqp_s_d256 = s;
            g_tqp_cuda_init_d256 = true;
        }
        return cudaSuccess;
    }

    return cudaSuccess;
}

static inline void tqp_cuda_cleanup() {
    if (d_tqp_s_d128)  cudaFree(d_tqp_s_d128);
    if (d_tqp_s_d256)  cudaFree(d_tqp_s_d256);

    d_tqp_s_d128  = nullptr;
    d_tqp_s_d256  = nullptr;
    g_tqp_cuda_init_d128 = false;
    g_tqp_cuda_init_d256 = false;
    g_tqp_cuda_constants_init_d128 = false;
    g_tqp_cuda_constants_init_d256 = false;
}
