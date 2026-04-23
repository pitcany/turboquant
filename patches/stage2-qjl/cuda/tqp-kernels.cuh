#pragma once

#include <cuda_runtime.h>

#include <stdint.h>
#include <string.h>

#define QK_TQP_D64 64
#define QK_TQP_D128 128
#define QK_TQP_D256 256

#define QK_TQ4P_D64 QK_TQP_D64
#define QK_TQ4P_D128 QK_TQP_D128
#define QK_TQ4P_D256 QK_TQP_D256

#define TQP_QS_BYTES(D, BITS) (((D) * (BITS)) / 8)
#define TQP_SIGN_BYTES(D) ((D) / 8)
#define TQP_BLOCK_SIZE(D, BITS) (2 + 2 + 1 + TQP_QS_BYTES((D), (BITS)) + TQP_SIGN_BYTES(D))

// Must match patches/stage2-qjl/c/ggml-tq-paper.h
//
// TQP_LAYER_BYTE: the quantize-call layout with bit 6 = 1 ("explicit
// override"). Use when calling quantize functions to force a specific
// rotation, bypassing the runtime resolver.
//
// TQP_STORED_BYTE: the block-storage layout with bit 6 = 0. Used inside
// quantize kernels when writing the resolved rotation into a block
// header. NEVER use TQP_LAYER_BYTE for block storage — it would set
// bit 6 in the stored byte and corrupt downstream extract_rotation.
#define TQP_ROT_WHT  0u
#define TQP_ROT_HAAR 1u
#define TQP_BIT6_EXPLICIT            (1u << 6)
#define TQP_LAYER_BYTE(layer, rot)   ((uint8_t)(TQP_BIT6_EXPLICIT | (((uint32_t)(rot) & 1u) << 7) | ((uint32_t)(layer) & 0x1fu)))
#define TQP_STORED_BYTE(layer, rot)  ((uint8_t)((((uint32_t)(rot) & 1u) << 7) | ((uint32_t)(layer) & 0x1fu)))
#define TQP_EXTRACT_LAYER(byte)      ((uint8_t)((byte) & 0x1fu))
#define TQP_EXTRACT_ROT(byte)        ((uint8_t)(((byte) >> 7) & 1u))
#define TQP_EXTRACT_EXPLICIT(byte)   ((uint8_t)(((byte) >> 6) & 1u))

#pragma pack(push, 1)
typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  layer_idx;
    uint8_t  qs[TQP_QS_BYTES(QK_TQP_D64, 2)];
    uint8_t  qjl_signs[TQP_SIGN_BYTES(QK_TQP_D64)];
} block_tqp_d64_b2;

typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  layer_idx;
    uint8_t  qs[TQP_QS_BYTES(QK_TQP_D64, 3)];
    uint8_t  qjl_signs[TQP_SIGN_BYTES(QK_TQP_D64)];
} block_tqp_d64_b3;

typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  layer_idx;
    uint8_t  qs[TQP_QS_BYTES(QK_TQP_D64, 4)];
    uint8_t  qjl_signs[TQP_SIGN_BYTES(QK_TQP_D64)];
} block_tqp_d64_b4;

typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  layer_idx;
    uint8_t  qs[TQP_QS_BYTES(QK_TQP_D128, 2)];
    uint8_t  qjl_signs[TQP_SIGN_BYTES(QK_TQP_D128)];
} block_tqp_d128_b2;

typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  layer_idx;
    uint8_t  qs[TQP_QS_BYTES(QK_TQP_D128, 3)];
    uint8_t  qjl_signs[TQP_SIGN_BYTES(QK_TQP_D128)];
} block_tqp_d128_b3;

typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  layer_idx;
    uint8_t  qs[TQP_QS_BYTES(QK_TQP_D128, 4)];
    uint8_t  qjl_signs[TQP_SIGN_BYTES(QK_TQP_D128)];
} block_tqp_d128_b4;

typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  layer_idx;
    uint8_t  qs[TQP_QS_BYTES(QK_TQP_D256, 2)];
    uint8_t  qjl_signs[TQP_SIGN_BYTES(QK_TQP_D256)];
} block_tqp_d256_b2;

typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  layer_idx;
    uint8_t  qs[TQP_QS_BYTES(QK_TQP_D256, 3)];
    uint8_t  qjl_signs[TQP_SIGN_BYTES(QK_TQP_D256)];
} block_tqp_d256_b3;

typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  layer_idx;
    uint8_t  qs[TQP_QS_BYTES(QK_TQP_D256, 4)];
    uint8_t  qjl_signs[TQP_SIGN_BYTES(QK_TQP_D256)];
} block_tqp_d256_b4;
#pragma pack(pop)

typedef block_tqp_d64_b3 block_tq4p_d64;
typedef block_tqp_d128_b3 block_tq4p_d128;
typedef block_tqp_d256_b3 block_tq4p_d256;

static_assert(sizeof(block_tqp_d64_b2) == TQP_BLOCK_SIZE(QK_TQP_D64, 2), "block_tqp_d64_b2 size");
static_assert(sizeof(block_tqp_d64_b3) == TQP_BLOCK_SIZE(QK_TQP_D64, 3), "block_tqp_d64_b3 size");
static_assert(sizeof(block_tqp_d64_b4) == TQP_BLOCK_SIZE(QK_TQP_D64, 4), "block_tqp_d64_b4 size");
static_assert(sizeof(block_tqp_d128_b2) == TQP_BLOCK_SIZE(QK_TQP_D128, 2), "block_tqp_d128_b2 size");
static_assert(sizeof(block_tqp_d128_b3) == TQP_BLOCK_SIZE(QK_TQP_D128, 3), "block_tqp_d128_b3 size");
static_assert(sizeof(block_tqp_d128_b4) == TQP_BLOCK_SIZE(QK_TQP_D128, 4), "block_tqp_d128_b4 size");
static_assert(sizeof(block_tqp_d256_b2) == TQP_BLOCK_SIZE(QK_TQP_D256, 2), "block_tqp_d256_b2 size");
static_assert(sizeof(block_tqp_d256_b3) == TQP_BLOCK_SIZE(QK_TQP_D256, 3), "block_tqp_d256_b3 size");
static_assert(sizeof(block_tqp_d256_b4) == TQP_BLOCK_SIZE(QK_TQP_D256, 4), "block_tqp_d256_b4 size");
static_assert(sizeof(block_tq4p_d64) == 37, "block_tq4p_d64 size");
static_assert(sizeof(block_tq4p_d128) == 69, "block_tq4p_d128 size");
static_assert(sizeof(block_tq4p_d256) == 133, "block_tq4p_d256 size");

template<int D, int BITS>
struct tqp_cuda_block;

template<> struct tqp_cuda_block<QK_TQP_D64, 2> { using type = block_tqp_d64_b2; };
template<> struct tqp_cuda_block<QK_TQP_D64, 3> { using type = block_tqp_d64_b3; };
template<> struct tqp_cuda_block<QK_TQP_D64, 4> { using type = block_tqp_d64_b4; };
template<> struct tqp_cuda_block<QK_TQP_D128, 2> { using type = block_tqp_d128_b2; };
template<> struct tqp_cuda_block<QK_TQP_D128, 3> { using type = block_tqp_d128_b3; };
template<> struct tqp_cuda_block<QK_TQP_D128, 4> { using type = block_tqp_d128_b4; };
template<> struct tqp_cuda_block<QK_TQP_D256, 2> { using type = block_tqp_d256_b2; };
template<> struct tqp_cuda_block<QK_TQP_D256, 3> { using type = block_tqp_d256_b3; };
template<> struct tqp_cuda_block<QK_TQP_D256, 4> { using type = block_tqp_d256_b4; };

template<int D, int BITS>
using tqp_cuda_block_t = typename tqp_cuda_block<D, BITS>::type;

static constexpr float TQP_SQRT_PI_OVER_2 = 1.2533141373155001f;

__host__ __device__ static inline uint16_t tqp_fp32_to_fp16_bits(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t x = v.u;
    uint32_t sign = (x >> 31) & 0x1u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu);
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
    uint32_t exp = (h >> 10) & 0x1fu;
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

template<int BITS>
__device__ static inline uint8_t tqp_bucketize(float x, const float * bounds) {
    constexpr int N_BOUNDS = (1 << BITS) - 1;
    uint8_t bucket = 0;
#pragma unroll
    for (int i = 0; i < N_BOUNDS; ++i) {
        bucket += (x > bounds[i]);
    }
    return bucket;
}

__device__ static inline float tqp_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffffu, val, 16);
    val += __shfl_xor_sync(0xffffffffu, val, 8);
    val += __shfl_xor_sync(0xffffffffu, val, 4);
    val += __shfl_xor_sync(0xffffffffu, val, 2);
    val += __shfl_xor_sync(0xffffffffu, val, 1);
    return val;
}

template<int BITS>
__device__ static inline uint8_t tqp_unpack_index_bitplane(const uint8_t * qs, int elem) {
    const int group = elem >> 3;
    const int bit = elem & 7;
    uint8_t idx = 0;
#pragma unroll
    for (int bitplane = 0; bitplane < BITS; ++bitplane) {
        idx |= (uint8_t)(((qs[group * BITS + bitplane] >> bit) & 1u) << bitplane);
    }
    return idx;
}

__device__ static inline float tqp_unpack_sign_pm1(const uint8_t * signs, int elem) {
    const uint8_t bit = (uint8_t)((signs[elem >> 3] >> (elem & 7)) & 1u);
    return bit ? -1.0f : 1.0f;
}

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

template<int D, uint8_t ROT, int BITS>
__device__ static inline void tqp_quantize_block_device(
        const float * __restrict__ x,
        uint16_t * orig_norm_out,
        uint16_t * res_d_out,
        uint8_t * layer_idx_out,
        uint8_t layer_idx_val,
        uint8_t * __restrict__ qs_out,
        uint8_t * __restrict__ signs_out,
        const float * __restrict__ sigma,
        const float * __restrict__ pi,
        const float * __restrict__ s,
        const float * __restrict__ centroids,
        const float * __restrict__ bounds) {
    static_assert(D == QK_TQP_D64 || D == QK_TQP_D128 || D == QK_TQP_D256, "unsupported TQP CUDA head dim");
    static_assert(BITS >= 2 && BITS <= 4, "unsupported TQP CUDA bits");
    static_assert((D % 32) == 0, "head dim must align to warp width");

    __shared__ float smem_vec[QK_TQP_D256];
    __shared__ uint8_t smem_idx[QK_TQP_D256];
    __shared__ float smem_scalars[2];

    const int tid = threadIdx.x;
    const float rsqrt_d = (ROT == TQP_ROT_WHT) ? rsqrtf((float)D) : 0.0f;
    const float sigma_tid = (ROT == TQP_ROT_WHT) ? sigma[tid] : 0.0f;

    smem_vec[tid] = x[tid];
    __syncthreads();

    if (tid == 0) {
        float sq = 0.0f;
#pragma unroll 1
        for (int i = 0; i < D; ++i) {
            sq = __fadd_rn(sq, __fmul_rn(smem_vec[i], smem_vec[i]));
        }
        float orig_norm = sqrtf(sq);
        if (orig_norm < 1e-8f) {
            orig_norm = 1e-8f;
        }
        smem_scalars[0] = orig_norm;
        smem_scalars[1] = 1.0f / orig_norm;
    }
    __syncthreads();

    const float x_unit_reg = __fmul_rn(smem_vec[tid], smem_scalars[1]);

    float x_rot_tid;
    if constexpr (ROT == TQP_ROT_WHT) {
        smem_vec[tid] = __fmul_rn(x_unit_reg, sigma_tid);
        __syncthreads();
        tqp_wht_shared<D>(smem_vec);
        x_rot_tid = __fmul_rn(smem_vec[tid], rsqrt_d);
    } else {
        smem_vec[tid] = x_unit_reg;
        __syncthreads();
        float acc = 0.0f;
#pragma unroll 1
        for (int j = 0; j < D; ++j) {
            acc = __fadd_rn(acc, __fmul_rn(__ldg(&pi[tid * D + j]), smem_vec[j]));
        }
        x_rot_tid = acc;
    }
    smem_idx[tid] = tqp_bucketize<BITS>(x_rot_tid, bounds);
    __syncthreads();

    float x_hat_tid;
    if constexpr (ROT == TQP_ROT_WHT) {
        smem_vec[tid] = centroids[smem_idx[tid]];
        __syncthreads();
        tqp_wht_shared<D>(smem_vec);
        x_hat_tid = __fmul_rn(smem_vec[tid], __fmul_rn(sigma_tid, rsqrt_d));
    } else {
        smem_vec[tid] = centroids[smem_idx[tid]];
        __syncthreads();
        float acc = 0.0f;
#pragma unroll 1
        for (int i = 0; i < D; ++i) {
            acc = __fadd_rn(acc, __fmul_rn(__ldg(&pi[i * D + tid]), smem_vec[i]));
        }
        x_hat_tid = acc;
    }

    __syncthreads();
    smem_vec[tid] = __fadd_rn(x_unit_reg, -x_hat_tid);
    __syncthreads();

    if (tid == 0) {
        float r_sq = 0.0f;
#pragma unroll 1
        for (int i = 0; i < D; ++i) {
            r_sq = __fadd_rn(r_sq, __fmul_rn(smem_vec[i], smem_vec[i]));
        }
        smem_scalars[1] = sqrtf(r_sq);
    }
    __syncthreads();

    float proj = 0.0f;
#pragma unroll 1
    for (int j = 0; j < D; ++j) {
        proj = __fadd_rn(proj, __fmul_rn(__ldg(&s[tid * D + j]), smem_vec[j]));
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const uint8_t idx = smem_idx[tid];

    uint32_t bit_ballots[BITS];
#pragma unroll
    for (int bitplane = 0; bitplane < BITS; ++bitplane) {
        bit_ballots[bitplane] = __ballot_sync(0xffffffffu, (idx & (1u << bitplane)) != 0);
    }
    const uint32_t sign_mask = __ballot_sync(0xffffffffu, proj < 0.0f);

    if (lane == 0) {
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const int group = warp * 4 + sub;
#pragma unroll
            for (int bitplane = 0; bitplane < BITS; ++bitplane) {
                qs_out[group * BITS + bitplane] = (uint8_t)((bit_ballots[bitplane] >> (8 * sub)) & 0xffu);
            }
            signs_out[group] = (uint8_t)((sign_mask >> (8 * sub)) & 0xffu);
        }
    }

    if (tid == 0) {
        uint16_t norm_val = tqp_fp32_to_fp16_device(smem_scalars[0]);
        uint16_t resd_val = tqp_fp32_to_fp16_device(smem_scalars[1]);
        memcpy(orig_norm_out, &norm_val, sizeof(uint16_t));
        memcpy(res_d_out, &resd_val, sizeof(uint16_t));
        *layer_idx_out = layer_idx_val;
    }
}
