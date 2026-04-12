#pragma once

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

static float * d_tqp_pi_d128 = nullptr;
static float * d_tqp_s_d128  = nullptr;
static float * d_tqp_pi_d256 = nullptr;
static float * d_tqp_s_d256  = nullptr;

static bool g_tqp_cuda_init_d128 = false;
static bool g_tqp_cuda_init_d256 = false;

static inline cudaError_t tqp_cuda_init(int head_dim) {
    cudaError_t err = cudaSuccess;

    if (head_dim == QK_TQ4P_D128 && !g_tqp_cuda_init_d128) {
        err = cudaMemcpyToSymbol(c_tqp_centroids_d128, TQP_CENTROIDS_D128, sizeof(TQP_CENTROIDS_D128));
        if (err != cudaSuccess) return err;
        err = cudaMemcpyToSymbol(c_tqp_boundaries_d128, TQP_BOUNDARIES_D128, sizeof(TQP_BOUNDARIES_D128));
        if (err != cudaSuccess) return err;
        err = cudaMalloc((void **)&d_tqp_pi_d128, sizeof(TQP_PI_D128));
        if (err != cudaSuccess) return err;
        err = cudaMalloc((void **)&d_tqp_s_d128, sizeof(TQP_S_D128));
        if (err != cudaSuccess) return err;
        err = cudaMemcpy(d_tqp_pi_d128, TQP_PI_D128, sizeof(TQP_PI_D128), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;
        err = cudaMemcpy(d_tqp_s_d128, TQP_S_D128, sizeof(TQP_S_D128), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;
        g_tqp_cuda_init_d128 = true;
        return cudaSuccess;
    }

    if (head_dim == QK_TQ4P_D256 && !g_tqp_cuda_init_d256) {
        err = cudaMemcpyToSymbol(c_tqp_centroids_d256, TQP_CENTROIDS_D256, sizeof(TQP_CENTROIDS_D256));
        if (err != cudaSuccess) return err;
        err = cudaMemcpyToSymbol(c_tqp_boundaries_d256, TQP_BOUNDARIES_D256, sizeof(TQP_BOUNDARIES_D256));
        if (err != cudaSuccess) return err;
        err = cudaMalloc((void **)&d_tqp_pi_d256, sizeof(TQP_PI_D256));
        if (err != cudaSuccess) return err;
        err = cudaMalloc((void **)&d_tqp_s_d256, sizeof(TQP_S_D256));
        if (err != cudaSuccess) return err;
        err = cudaMemcpy(d_tqp_pi_d256, TQP_PI_D256, sizeof(TQP_PI_D256), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;
        err = cudaMemcpy(d_tqp_s_d256, TQP_S_D256, sizeof(TQP_S_D256), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;
        g_tqp_cuda_init_d256 = true;
        return cudaSuccess;
    }

    return cudaSuccess;
}

static inline void tqp_cuda_cleanup() {
    if (d_tqp_pi_d128) cudaFree(d_tqp_pi_d128);
    if (d_tqp_s_d128)  cudaFree(d_tqp_s_d128);
    if (d_tqp_pi_d256) cudaFree(d_tqp_pi_d256);
    if (d_tqp_s_d256)  cudaFree(d_tqp_s_d256);

    d_tqp_pi_d128 = nullptr;
    d_tqp_s_d128  = nullptr;
    d_tqp_pi_d256 = nullptr;
    d_tqp_s_d256  = nullptr;
    g_tqp_cuda_init_d128 = false;
    g_tqp_cuda_init_d256 = false;
}
