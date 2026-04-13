#pragma once

#include <cuda_runtime.h>

#include <stdint.h>

#define QK_TQ4P_D128 128
#define QK_TQ4P_D256 256

typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  qs[48];
    uint8_t  qjl_signs[16];
} block_tq4p_d128;

typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  qs[96];
    uint8_t  qjl_signs[32];
} block_tq4p_d256;

static_assert(sizeof(block_tq4p_d128) == 68, "block_tq4p_d128 size");
static_assert(sizeof(block_tq4p_d256) == 132, "block_tq4p_d256 size");

static constexpr float TQP_SQRT_PI_OVER_2 = 1.2533141373155001f;

__host__ __device__ static inline uint16_t tqp_fp32_to_fp16_bits(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t x = v.u;
    uint32_t sign = (x >> 31) & 0x1u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xffu);
    uint32_t mant = x & 0x7fffffu;

    if (exp == 0xff) {
        return (uint16_t)((sign << 15) | 0x7c00u | (mant ? 0x200u : 0u));
    }

    int32_t e = exp - 127 + 15;
    if (e >= 0x1f) {
        return (uint16_t)((sign << 15) | 0x7c00u);
    }
    if (e <= 0) {
        if (e < -10) {
            return (uint16_t)(sign << 15);
        }
        mant = (mant | 0x800000u) >> (1 - e);
        mant += 0x0FFFu + ((mant >> 13) & 1u);
        return (uint16_t)((sign << 15) | (mant >> 13));
    }

    mant += 0x0FFFu + ((mant >> 13) & 1u);
    if (mant & 0x800000u) {
        mant = 0;
        e += 1;
        if (e >= 0x1f) {
            return (uint16_t)((sign << 15) | 0x7c00u);
        }
    }

    return (uint16_t)((sign << 15) | ((uint32_t)e << 10) | (mant >> 13));
}

__host__ __device__ static inline float tqp_fp16_bits_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;

    if (exp == 0) {
        if (mant == 0) {
            out = sign << 31;
        } else {
            int shift = 0;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                shift++;
            }
            mant &= 0x3ffu;
            out = (sign << 31) | (((uint32_t)(127 - 15 - shift + 1)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        out = (sign << 31) | (((uint32_t)(exp - 15 + 127)) << 23) | (mant << 13);
    }

    union { uint32_t u; float f; } v = { out };
    return v.f;
}

__device__ static inline uint16_t tqp_fp32_to_fp16_device(float f) {
    return tqp_fp32_to_fp16_bits(f);
}

__device__ static inline float tqp_fp16_to_fp32_device(uint16_t h) {
    return tqp_fp16_bits_to_fp32(h);
}

__device__ static inline uint8_t tqp_bucketize_d3(float x, const float * bounds) {
    uint8_t b = 0;
    b += (x > bounds[0]);
    b += (x > bounds[1]);
    b += (x > bounds[2]);
    b += (x > bounds[3]);
    b += (x > bounds[4]);
    b += (x > bounds[5]);
    b += (x > bounds[6]);
    return b;
}

__device__ static inline float tqp_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffffu, val, 16);
    val += __shfl_xor_sync(0xffffffffu, val, 8);
    val += __shfl_xor_sync(0xffffffffu, val, 4);
    val += __shfl_xor_sync(0xffffffffu, val, 2);
    val += __shfl_xor_sync(0xffffffffu, val, 1);
    return val;
}

__device__ static inline uint8_t tqp_unpack_index_bitplane(const uint8_t * qs, int elem) {
    const int group = elem >> 3;
    const int bit = elem & 7;
    return (uint8_t)(((qs[group * 3 + 0] >> bit) & 1u)
        | (((qs[group * 3 + 1] >> bit) & 1u) << 1)
        | (((qs[group * 3 + 2] >> bit) & 1u) << 2));
}

__device__ static inline float tqp_unpack_sign_pm1(const uint8_t * signs, int elem) {
    const uint8_t bit = (uint8_t)((signs[elem >> 3] >> (elem & 7)) & 1u);
    return bit ? -1.0f : 1.0f;
}

// In-place Fast Walsh-Hadamard Transform on `smem[0..D-1]`, unnormalized
// (caller must multiply by 1/√d for the orthogonal version).
//
// Requires: blockDim.x == D, D is a power of 2, one thread per element.
// Thread `tid` owns `smem[tid]` throughout. Each butterfly stage combines
// pairs (i, i^h) with a ±1 sum; low-index thread writes (a+b), high-index
// thread writes (b-a) where a=smem[tid] and b=smem[tid^h].
template<int D>
__device__ static inline void tqp_wht_shared(float * smem) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int h = 1; h < D; h <<= 1) {
        __syncthreads();
        const float a = smem[tid];
        const float b = smem[tid ^ h];
        __syncthreads();
        smem[tid] = ((tid & h) == 0) ? (a + b) : (b - a);
    }
    __syncthreads();
}
