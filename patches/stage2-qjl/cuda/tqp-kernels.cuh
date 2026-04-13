#pragma once

#include <cuda_runtime.h>

#include <stdint.h>

#define QK_TQ4P_D128 128
#define QK_TQ4P_D256 256

// Must match patches/stage2-qjl/c/ggml-tq-paper.h
#define TQP_ROT_WHT  0u
#define TQP_ROT_HAAR 1u
#define TQP_LAYER_BYTE(layer, rot) ((uint8_t)((((uint32_t)(rot) & 1u) << 7) | ((uint32_t)(layer) & 0x1fu)))
#define TQP_EXTRACT_LAYER(byte)    ((uint8_t)((byte) & 0x1fu))
#define TQP_EXTRACT_ROT(byte)      ((uint8_t)(((byte) >> 7) & 1u))

// Block layout matches the CPU path (patches/stage2-qjl/c/ggml-tq-paper.h):
//   offset 0..1  orig_norm (fp16)
//   offset 2..3  res_d     (fp16)
//   offset 4     layer_idx (uint8) — selects per-layer σ and S
//   offset 5..   qs        (3-bit bitplane-packed indices)
//   then         qjl_signs (1-bit per coord)
// pragma pack(1) so the odd total sizes (69, 133) match CPU; without it
// the trailing uint16_t on the outer struct would force 2-byte alignment.
#pragma pack(push, 1)
typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  layer_idx;
    uint8_t  qs[48];
    uint8_t  qjl_signs[16];
} block_tq4p_d128;

typedef struct {
    uint16_t orig_norm;
    uint16_t res_d;
    uint8_t  layer_idx;
    uint8_t  qs[96];
    uint8_t  qjl_signs[32];
} block_tq4p_d256;
#pragma pack(pop)

static_assert(sizeof(block_tq4p_d128) == 69, "block_tq4p_d128 size");
static_assert(sizeof(block_tq4p_d256) == 133, "block_tq4p_d256 size");

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

// Full-vector TQ4P quantization (Stage-1 Lloyd-Max + Stage-2 QJL).
//
// One CTA per vector, D threads (one per element). Requires shared memory
// for intermediate vector, indices, and scalar reductions.
//
// Used by both tqp-quantize.cu (contiguous quantize) and tqp-set-rows.cu
// (index-scattered KV cache writes).
template<int D, uint8_t ROT>
__device__ static inline void tqp_quantize_block_device(
        const float * __restrict__ x,
        uint16_t * orig_norm_out,
        uint16_t * res_d_out,
        uint8_t * layer_idx_out,
        uint8_t   layer_idx_val,
        uint8_t * __restrict__ qs_out,
        uint8_t * __restrict__ signs_out,
        const float * __restrict__ sigma,
        const float * __restrict__ pi,
        const float * __restrict__ s,
        const float * __restrict__ centroids,
        const float * __restrict__ bounds) {
    __shared__ float smem_vec[QK_TQ4P_D256];
    __shared__ uint8_t smem_idx[QK_TQ4P_D256];
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

    // x_unit = x / ||x||, kept in a per-thread register for later residual.
    const float x_unit_reg = __fmul_rn(smem_vec[tid], smem_scalars[1]);

    float x_rot_tid;
    if constexpr (ROT == TQP_ROT_WHT) {
        // smem_vec[tid] = σ ⊙ x_unit, then in-place FWHT, then scale.
        smem_vec[tid] = __fmul_rn(x_unit_reg, sigma_tid);
        __syncthreads();
        tqp_wht_shared<D>(smem_vec);
        x_rot_tid = __fmul_rn(smem_vec[tid], rsqrt_d);
    } else {
        // Dense Haar GEMV: x_rot[tid] = Σ_j Π[tid, j] · x_unit[j].
        smem_vec[tid] = x_unit_reg;
        __syncthreads();
        float acc = 0.0f;
#pragma unroll 1
        for (int j = 0; j < D; ++j) {
            acc = __fadd_rn(acc, __fmul_rn(__ldg(&pi[tid * D + j]), smem_vec[j]));
        }
        x_rot_tid = acc;
    }
    smem_idx[tid] = tqp_bucketize_d3(x_rot_tid, bounds);
    __syncthreads();

    float x_hat_tid;
    if constexpr (ROT == TQP_ROT_WHT) {
        // x_hat_unit = (1/√d) · σ ⊙ WHT(centroids[idx]).
        smem_vec[tid] = centroids[smem_idx[tid]];
        __syncthreads();
        tqp_wht_shared<D>(smem_vec);
        x_hat_tid = __fmul_rn(smem_vec[tid], __fmul_rn(sigma_tid, rsqrt_d));
    } else {
        // Dense Haar GEMV (transposed): x_hat[tid] = Σ_i Π[i, tid] · c[idx[i]].
        smem_vec[tid] = centroids[smem_idx[tid]];
        __syncthreads();
        float acc = 0.0f;
#pragma unroll 1
        for (int i = 0; i < D; ++i) {
            acc = __fadd_rn(acc, __fmul_rn(__ldg(&pi[i * D + tid]), smem_vec[i]));
        }
        x_hat_tid = acc;
    }

    // residual = x_unit - x_hat_unit
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

    const uint32_t ballot_lo  = __ballot_sync(0xffffffffu, (idx & 1u) != 0);
    const uint32_t ballot_mid = __ballot_sync(0xffffffffu, (idx & 2u) != 0);
    const uint32_t ballot_hi  = __ballot_sync(0xffffffffu, (idx & 4u) != 0);
    const uint32_t sign_mask  = __ballot_sync(0xffffffffu, proj < 0.0f);

    if (lane == 0) {
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const int group = warp * 4 + sub;
            qs_out[group * 3 + 0] = (uint8_t)((ballot_lo  >> (8 * sub)) & 0xffu);
            qs_out[group * 3 + 1] = (uint8_t)((ballot_mid >> (8 * sub)) & 0xffu);
            qs_out[group * 3 + 2] = (uint8_t)((ballot_hi  >> (8 * sub)) & 0xffu);
            signs_out[group] = (uint8_t)((sign_mask >> (8 * sub)) & 0xffu);
        }
    }

    if (tid == 0) {
        *orig_norm_out = tqp_fp32_to_fp16_device(smem_scalars[0]);
        *res_d_out = tqp_fp32_to_fp16_device(smem_scalars[1]);
        *layer_idx_out = layer_idx_val;
    }
}
