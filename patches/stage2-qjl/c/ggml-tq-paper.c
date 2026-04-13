// TurboQuant-ish quantization — CPU reference implementation.
//
// Byte-exact mirror of patches/stage2-qjl/python/tq_paper_reference.py.
//
// Variant: this branch replaces the paper's Haar random orthogonal rotation
// Π with a Randomized Hadamard Transform, i.e. Π := (1/√d) · H · diag(σ)
// where H is the d×d Walsh-Hadamard matrix and σ ∈ {±1}^d is a per-layer
// random sign vector. This is still orthogonal but O(d log d) to apply, and
// keeps only a d-entry sign vector per layer instead of a d×d matrix.
//
// The algorithm is otherwise TurboQuant-ish: 3-bit Lloyd-Max scalar
// quantization on the rotated unit vector, then a 1-bit Gaussian QJL sign
// estimator on the residual. The QJL projection S is unchanged (still a
// d×d Gaussian matrix, per-layer). The Lloyd-Max codebook is regenerated
// with the same Gaussian approximation — for uniformly random unit vectors
// the RHT marginals are identical to Haar's, so the codebook comes out
// numerically the same.
//
// This file is deliberately simple: no SIMD, no unrolling, no vectorization.
// Correctness first; CUDA kernels come in a follow-up commit where perf
// matters.
//
// Per-layer sign vector σ_i and QJL projection S_i are stored as fp32 in
// the generated headers, indexed by layer_idx (seeds: σ_i=42+i, S_i=43+i).

#include "ggml-tq-paper.h"

#include <math.h>
#include <string.h>
#include <assert.h>

// Generated constants. Regenerate via
//   python3 patches/stage2-qjl/python/generate_constants.py
#include "tqp_centroids_d128.h"
#include "tqp_centroids_d256.h"
#include "tqp_constants_d128.h"
#include "tqp_constants_d256.h"

static inline uint8_t tqp_layer_idx(uint8_t layer_idx) {
    return (uint8_t)(layer_idx % TQP_MAX_LAYERS);
}

// ---------- fp16 conversion ----------
//
// When integrating into the fork, replace these with ggml_fp32_to_fp16 /
// ggml_fp16_to_fp32 from ggml-common.h. The implementations here are
// portable but slow; they match torch's round-to-nearest-even behavior which
// is what tq_paper_reference.py uses (via struct.pack('<e', ...)).

static inline uint16_t tqp_fp32_to_fp16(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t x = v.u;
    uint32_t sign = (x >> 31) & 0x1u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xffu);
    uint32_t mant = x & 0x7fffffu;

    if (exp == 0xff) {                     // inf or nan
        return (uint16_t)((sign << 15) | 0x7c00u | (mant ? 0x200u : 0u));
    }
    int32_t e = exp - 127 + 15;
    if (e >= 0x1f) {                       // overflow -> inf
        return (uint16_t)((sign << 15) | 0x7c00u);
    }
    if (e <= 0) {                          // subnormal or underflow
        if (e < -10) return (uint16_t)(sign << 15);
        mant = (mant | 0x800000u) >> (1 - e);
        mant += 0x0FFFu + ((mant >> 13) & 1u);  // round to nearest even
        return (uint16_t)((sign << 15) | (mant >> 13));
    }
    mant += 0x0FFFu + ((mant >> 13) & 1u);  // round to nearest even
    if (mant & 0x800000u) { mant = 0; e += 1; if (e >= 0x1f) return (uint16_t)((sign << 15) | 0x7c00u); }
    return (uint16_t)((sign << 15) | ((uint32_t)e << 10) | (mant >> 13));
}

static inline float tqp_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign << 31;
        } else {
            int shift = 0;
            while (!(mant & 0x400u)) { mant <<= 1; shift++; }
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

// ---------- bit pack / unpack ----------

// Pack 8 x 3-bit values into 3 bytes using the bitplane layout:
//   byte[0].bit_i = (v_i >> 0) & 1
//   byte[1].bit_i = (v_i >> 1) & 1
//   byte[2].bit_i = (v_i >> 2) & 1
static inline void tqp_pack_indices_bitplane(const uint8_t * idx, uint8_t * out, int n_groups) {
    for (int g = 0; g < n_groups; ++g) {
        uint32_t lo = 0, mid = 0, hi = 0;
        for (int i = 0; i < 8; ++i) {
            uint32_t v = idx[g * 8 + i];
            lo  |= ((v >> 0) & 1u) << i;
            mid |= ((v >> 1) & 1u) << i;
            hi  |= ((v >> 2) & 1u) << i;
        }
        out[g * 3 + 0] = (uint8_t)lo;
        out[g * 3 + 1] = (uint8_t)mid;
        out[g * 3 + 2] = (uint8_t)hi;
    }
}

static inline void tqp_unpack_indices_bitplane(const uint8_t * in, uint8_t * idx, int n_groups) {
    for (int g = 0; g < n_groups; ++g) {
        uint32_t lo = in[g * 3 + 0];
        uint32_t mid = in[g * 3 + 1];
        uint32_t hi = in[g * 3 + 2];
        for (int i = 0; i < 8; ++i) {
            idx[g * 8 + i] = (uint8_t)(((lo >> i) & 1u)
                                    | (((mid >> i) & 1u) << 1)
                                    | (((hi  >> i) & 1u) << 2));
        }
    }
}

// ---------- Walsh-Hadamard transform ----------
//
// Fast in-place Walsh-Hadamard transform, natural (Hadamard) ordering.
// Caller must divide by sqrt(d) for the orthogonal version.
// d must be a power of 2 (128 and 256 are).

static void tqp_wht_inplace(float * x, int d) {
    for (int h = 1; h < d; h <<= 1) {
        for (int i = 0; i < d; i += h << 1) {
            for (int j = i; j < i + h; ++j) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

// y = Π · v = (1/sqrt(d)) · H · diag(σ) · v
static inline void tqp_rht_apply(int d, const float * sigma, const float * v, float * y) {
    for (int i = 0; i < d; ++i) y[i] = sigma[i] * v[i];
    tqp_wht_inplace(y, d);
    const float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; ++i) y[i] *= inv_sqrt_d;
}

// y = Πᵀ · v = (1/sqrt(d)) · diag(σ) · H · v   (H is symmetric, self-transposed)
static inline void tqp_rht_apply_t(int d, const float * sigma, const float * v, float * y) {
    for (int i = 0; i < d; ++i) y[i] = v[i];
    tqp_wht_inplace(y, d);
    const float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; ++i) y[i] *= sigma[i] * inv_sqrt_d;
}

// ---------- Lloyd-Max bucketize ----------
//
// Linear search over 7 boundaries; branchless via comparisons and adds.
// For bits=3 this is always 7 comparisons — short enough not to warrant
// binary search. Matches torch.bucketize exactly: returns the number of
// boundaries strictly less than x (i.e., the left-most bin whose left edge
// is >= x).

static inline uint8_t tqp_bucketize_d3(float x, const float * bounds) {
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

// ---------- core quantize/dequantize (d-parametric helpers) ----------

static void tqp_quantize_block(
        int d,
        const float * sigma,      // (d,) ±1 sign vector, per-layer
        const float * s,          // (d x d) row-major QJL matrix, per-layer
        const float * centroids,  // (8,)
        const float * bounds,     // (7,)
        const float * x,          // (d,) input
        uint8_t * qs_out,         // (d * 3 / 8) bytes
        uint8_t * signs_out,      // (d / 8) bytes
        float * res_d_out,        // scalar output
        float * orig_norm_out)    // scalar output
{
    // orig_norm = ||x||
    float sq = 0.0f;
    for (int i = 0; i < d; ++i) sq += x[i] * x[i];
    float orig_norm = sqrtf(sq);
    if (orig_norm < 1e-8f) orig_norm = 1e-8f;
    float inv_norm = 1.0f / orig_norm;

    // x_unit = x / orig_norm
    float x_unit[QK_TQ4P_D256];
    for (int i = 0; i < d; ++i) x_unit[i] = x[i] * inv_norm;

    // x_rot = Π · x_unit = (1/√d) · H · diag(σ) · x_unit
    float x_rot[QK_TQ4P_D256];
    tqp_rht_apply(d, sigma, x_unit, x_rot);

    // indices[i] = bucketize(x_rot[i])
    uint8_t idx[QK_TQ4P_D256];
    for (int i = 0; i < d; ++i) idx[i] = tqp_bucketize_d3(x_rot[i], bounds);

    // x_hat_unit = Πᵀ · centroids[idx]
    float x_hat_rot[QK_TQ4P_D256];
    for (int i = 0; i < d; ++i) x_hat_rot[i] = centroids[idx[i]];
    float x_hat_unit[QK_TQ4P_D256];
    tqp_rht_apply_t(d, sigma, x_hat_rot, x_hat_unit);

    // residual = x_unit - x_hat_unit
    float residual[QK_TQ4P_D256];
    float r_sq = 0.0f;
    for (int i = 0; i < d; ++i) {
        residual[i] = x_unit[i] - x_hat_unit[i];
        r_sq += residual[i] * residual[i];
    }
    float res_d = sqrtf(r_sq);

    // QJL signs: proj = S . residual; sign bit = (proj < 0)
    uint8_t signs[QK_TQ4P_D256 / 8];
    memset(signs, 0, (size_t)(d / 8));
    for (int i = 0; i < d; ++i) {
        float acc = 0.0f;
        const float * s_row = s + (size_t)i * d;
        for (int j = 0; j < d; ++j) acc += s_row[j] * residual[j];
        if (acc < 0.0f) signs[i / 8] |= (uint8_t)(1u << (i % 8));
    }

    // pack indices (bitplane)
    tqp_pack_indices_bitplane(idx, qs_out, d / 8);
    memcpy(signs_out, signs, (size_t)(d / 8));
    *res_d_out = res_d;
    *orig_norm_out = orig_norm;
}

static void tqp_dequantize_block(
        int d,
        const float * sigma,
        const float * centroids,
        float orig_norm,
        const uint8_t * qs,
        float * y_out)
{
    uint8_t idx[QK_TQ4P_D256];
    tqp_unpack_indices_bitplane(qs, idx, d / 8);

    // x_hat_unit = Πᵀ · centroids[idx] (same as quantize path)
    float x_hat_rot[QK_TQ4P_D256];
    for (int i = 0; i < d; ++i) x_hat_rot[i] = centroids[idx[i]];
    float x_hat_unit[QK_TQ4P_D256];
    tqp_rht_apply_t(d, sigma, x_hat_rot, x_hat_unit);

    // Scale back to original magnitude
    for (int i = 0; i < d; ++i) y_out[i] = orig_norm * x_hat_unit[i];
}

// ---------- inner product ----------

static void tqp_prepare_query(int d, const float * s, const float * q, float * Sq) {
    for (int i = 0; i < d; ++i) {
        float acc = 0.0f;
        const float * s_row = s + (size_t)i * d;
        for (int j = 0; j < d; ++j) acc += s_row[j] * q[j];
        Sq[i] = acc;
    }
}

static float tqp_vec_dot_block(
        int d,
        const float * sigma,
        const float * centroids,
        const float * q,
        const float * Sq,
        const uint8_t * qs,
        const uint8_t * signs,
        float orig_norm,
        float res_d)
{
    // Stage 1: term1 = orig_norm · Σ_i (Π·q)[i] · centroids[idx[i]]
    float q_rot[QK_TQ4P_D256];
    tqp_rht_apply(d, sigma, q, q_rot);

    uint8_t idx[QK_TQ4P_D256];
    tqp_unpack_indices_bitplane(qs, idx, d / 8);

    float term1 = 0.0f;
    for (int i = 0; i < d; ++i) term1 += q_rot[i] * centroids[idx[i]];
    term1 *= orig_norm;

    // Stage 2: term2 = orig_norm . res_d . sqrt(pi/2)/d . sum_i Sq[i] . sign(residual projection)
    float term2 = 0.0f;
    for (int i = 0; i < d; ++i) {
        float sign_val = (signs[i / 8] & (1u << (i % 8))) ? -1.0f : 1.0f;
        term2 += Sq[i] * sign_val;
    }
    const float sqrt_pi_over_2 = 1.2533141373155001f;  // sqrt(pi/2)
    term2 *= orig_norm * res_d * sqrt_pi_over_2 / (float)d;

    return term1 + term2;
}

// ---------- per-d public entry points ----------
//
// The macros index into the per-layer 2D / 3D constant arrays:
//   SIGMA_ARR[layer_idx] = pointer to d fp32 ±1 values for the layer's σ
//   S_ARR[layer_idx]     = pointer to (d*d) floats for the layer's S

#define TQP_DEFINE_ROW_FUNCS(D, SIGMA_ARR, S_ARR, CENTROIDS, BOUNDS)                          \
    void ggml_quantize_row_tq4p_d##D(const float * x, block_tq4p_d##D * y,                    \
                                     int64_t k, uint8_t layer_idx) {                           \
        assert(k % D == 0);                                                                   \
        const uint8_t layer_idx_norm = tqp_layer_idx(layer_idx);                              \
        const float * sigma = SIGMA_ARR[layer_idx_norm];                                      \
        const float * s     = S_ARR[layer_idx_norm];                                          \
        const int64_t nb = k / D;                                                             \
        for (int64_t b = 0; b < nb; ++b) {                                                    \
            float res_d, orig_norm;                                                           \
            tqp_quantize_block(D, sigma, s, CENTROIDS, BOUNDS,                                \
                               x + b * D, y[b].qs, y[b].qjl_signs,                            \
                               &res_d, &orig_norm);                                           \
            y[b].orig_norm = tqp_fp32_to_fp16(orig_norm);                                     \
            y[b].res_d     = tqp_fp32_to_fp16(res_d);                                         \
            y[b].layer_idx = layer_idx_norm;                                                  \
        }                                                                                     \
    }                                                                                         \
                                                                                              \
    void ggml_dequantize_row_tq4p_d##D(const block_tq4p_d##D * x, float * y, int64_t k) {     \
        assert(k % D == 0);                                                                   \
        const int64_t nb = k / D;                                                             \
        for (int64_t b = 0; b < nb; ++b) {                                                    \
            const uint8_t layer_idx_norm = tqp_layer_idx(x[b].layer_idx);                     \
            const float * sigma = SIGMA_ARR[layer_idx_norm];                                  \
            float orig_norm = tqp_fp16_to_fp32(x[b].orig_norm);                               \
            tqp_dequantize_block(D, sigma, CENTROIDS, orig_norm, x[b].qs, y + b * D);         \
        }                                                                                     \
    }                                                                                         \
                                                                                              \
    void ggml_tqp_prepare_query_d##D(const float * q, float * Sq, uint8_t layer_idx) {        \
        tqp_prepare_query(D, S_ARR[tqp_layer_idx(layer_idx)], q, Sq);                         \
    }                                                                                         \
                                                                                              \
    float ggml_tqp_vec_dot_block_d##D(const float * q, const float * Sq,                      \
                                       const block_tq4p_d##D * blk) {                         \
        const uint8_t layer_idx_norm = tqp_layer_idx(blk->layer_idx);                         \
        return tqp_vec_dot_block(D, SIGMA_ARR[layer_idx_norm], CENTROIDS, q, Sq,              \
                                 blk->qs, blk->qjl_signs,                                    \
                                 tqp_fp16_to_fp32(blk->orig_norm),                            \
                                 tqp_fp16_to_fp32(blk->res_d));                               \
    }

TQP_DEFINE_ROW_FUNCS(128, TQP_SIGMA_D128, TQP_S_D128, TQP_CENTROIDS_D128, TQP_BOUNDARIES_D128)
TQP_DEFINE_ROW_FUNCS(256, TQP_SIGMA_D256, TQP_S_D256, TQP_CENTROIDS_D256, TQP_BOUNDARIES_D256)

// ---------- ggml dispatch wrappers ----------
//
// The vec_dot wrappers read layer_idx from the first K-block's header.
// In a well-formed attention pass, all blocks in a row share the same layer.

#define TQP_DEFINE_VEC_DOT(D)                                                                  \
    void ggml_vec_dot_tq4p_d##D##_f32(int n, float * s, size_t bs,                             \
                                      const void * vx, size_t bx,                              \
                                      const void * vy, size_t by, int nrc) {                   \
        assert(nrc == 1);                                                                      \
        assert(n % D == 0);                                                                    \
        (void)bs; (void)bx; (void)by;                                                          \
        const block_tq4p_d##D * blk = (const block_tq4p_d##D *)vx;                             \
        const float * q             = (const float *)vy;                                       \
        const int64_t nb = n / D;                                                              \
                                                                                               \
        float acc = 0.0f;                                                                      \
        for (int64_t b = 0; b < nb; ++b) {                                                     \
            float Sq[D];                                                                       \
            ggml_tqp_prepare_query_d##D(q + b * D, Sq, blk[b].layer_idx);                      \
            acc += ggml_tqp_vec_dot_block_d##D(q + b * D, Sq, &blk[b]);                        \
        }                                                                                      \
        *s = acc;                                                                              \
    }

TQP_DEFINE_VEC_DOT(128)
TQP_DEFINE_VEC_DOT(256)

// ---------- Q8_K query dequantize ----------
//
// Converts one Q8_K block (256 int8 values + scale) to fp32.
// This is the only extra work vs. the fp32 path; after dequantization the
// same Π rotation and QJL estimator run on the resulting fp32 query.

static inline void tqp_dequant_q8k(const block_q8k_compat * blk, float * out) {
    const float d = blk->d;
    for (int i = 0; i < QK_Q8K; ++i) {
        out[i] = d * (float)blk->qs[i];
    }
}

// ---------- Q8_K query dispatch wrappers ----------
//
// Same loop structure as the fp32 wrappers, but vy points to Q8_K blocks.
// We dequantize one Q8_K block (256 elts) at a time and feed the resulting
// fp32 values into the per-TQ4P-block inner product.
//
// n must be a multiple of QK_Q8K (256).  For D=128 each Q8_K block feeds
// 2 TQ4P blocks; for D=256 it feeds exactly 1.

#define TQP_DEFINE_VEC_DOT_Q8K(D)                                                                 \
    void ggml_vec_dot_tq4p_d##D##_q8k(int n, float * s, size_t bs,                                \
                                       const void * vx, size_t bx,                                 \
                                       const void * vy, size_t by, int nrc) {                      \
        assert(nrc == 1);                                                                          \
        assert(n % QK_Q8K == 0);                                                                   \
        (void)bs; (void)bx; (void)by;                                                              \
        const block_tq4p_d##D * blk = (const block_tq4p_d##D *)vx;                                \
        const block_q8k_compat * q8k = (const block_q8k_compat *)vy;                              \
        const int64_t nb_q8k = n / QK_Q8K;                                                        \
                                                                                                   \
        float acc = 0.0f;                                                                          \
        float q_buf[QK_Q8K];                                                                       \
        int64_t tqp_b = 0;                                                                        \
        for (int64_t qb = 0; qb < nb_q8k; ++qb) {                                                 \
            tqp_dequant_q8k(&q8k[qb], q_buf);                                                     \
            for (int sub = 0; sub < QK_Q8K / D; ++sub, ++tqp_b) {                                 \
                const float * q = q_buf + sub * D;                                                 \
                float Sq[D];                                                                       \
                ggml_tqp_prepare_query_d##D(q, Sq, blk[tqp_b].layer_idx);                         \
                acc += ggml_tqp_vec_dot_block_d##D(q, Sq, &blk[tqp_b]);                           \
            }                                                                                      \
        }                                                                                          \
        *s = acc;                                                                                  \
    }

TQP_DEFINE_VEC_DOT_Q8K(128)
TQP_DEFINE_VEC_DOT_Q8K(256)
