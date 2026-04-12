// TurboQuant paper-faithful quantization — CPU reference implementation.
//
// Byte-exact port of patches/stage2-qjl/python/tq_paper_reference.py, which in
// turn is verified byte-exact against turboquant.py::TurboQuantProd, which in
// turn reproduces the paper's reported MSE and inner-product correlation.
//
// This file is deliberately simple: no SIMD, no unrolling, no vectorization.
// Correctness first; CUDA kernels come in a follow-up commit where perf
// matters. On a single core this implementation is ~5-10× slower than the
// fork's existing TQ3_0 vec_dot.
//
// The rotation matrix Π and QJL projection matrix S are stored as fp32 in
// the generated headers, which keeps byte-exact agreement with the Python
// oracle. A later optimization may store them as fp16 to halve constant
// memory footprint, at the cost of loosening the byte-exact test.

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
        mant += 0x1000u;                   // round to nearest even
        return (uint16_t)((sign << 15) | (mant >> 13));
    }
    mant += 0x1000u;                       // round to nearest even
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

// Pack 8 × 3-bit values into 3 bytes using the bitplane layout:
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

// ---------- Lloyd-Max bucketize ----------
//
// Linear search over 7 boundaries; branchless via comparisons and adds.
// For bits=3 this is always 7 comparisons — short enough not to warrant
// binary search. Matches torch.bucketize exactly: returns the number of
// boundaries strictly less than or equal to x (i.e., the right-most bin
// whose left edge is ≤ x).

static inline uint8_t tqp_bucketize_d3(float x, const float * bounds) {
    uint8_t b = 0;
    b += (x >= bounds[0]);
    b += (x >= bounds[1]);
    b += (x >= bounds[2]);
    b += (x >= bounds[3]);
    b += (x >= bounds[4]);
    b += (x >= bounds[5]);
    b += (x >= bounds[6]);
    return b;
}

// ---------- core quantize/dequantize (d-parametric helpers) ----------

static void tqp_quantize_block(
        int d,
        const float * pi,         // (d × d) row-major
        const float * s,          // (d × d) row-major
        const float * centroids,  // (8,)
        const float * bounds,     // (7,)
        const float * x,          // (d,) input
        uint8_t * qs_out,         // (d * 3 / 8) bytes
        uint8_t * signs_out,      // (d / 8) bytes
        float * res_d_out,        // scalar output
        float * orig_norm_out)    // scalar output
{
    // orig_norm = ‖x‖
    float sq = 0.0f;
    for (int i = 0; i < d; ++i) sq += x[i] * x[i];
    float orig_norm = sqrtf(sq);
    if (orig_norm < 1e-8f) orig_norm = 1e-8f;
    float inv_norm = 1.0f / orig_norm;

    // x_unit = x / orig_norm
    float x_unit[QK_TQ4P_D256];
    for (int i = 0; i < d; ++i) x_unit[i] = x[i] * inv_norm;

    // x_rot = Π · x_unit
    float x_rot[QK_TQ4P_D256];
    for (int i = 0; i < d; ++i) {
        float acc = 0.0f;
        const float * pi_row = pi + (size_t)i * d;
        for (int j = 0; j < d; ++j) acc += pi_row[j] * x_unit[j];
        x_rot[i] = acc;
    }

    // indices[i] = bucketize(x_rot[i])
    uint8_t idx[QK_TQ4P_D256];
    for (int i = 0; i < d; ++i) idx[i] = tqp_bucketize_d3(x_rot[i], bounds);

    // x_hat_unit = Π^T · centroids[idx]
    float x_hat_unit[QK_TQ4P_D256];
    for (int j = 0; j < d; ++j) x_hat_unit[j] = 0.0f;
    for (int i = 0; i < d; ++i) {
        float c = centroids[idx[i]];
        const float * pi_row = pi + (size_t)i * d;
        for (int j = 0; j < d; ++j) x_hat_unit[j] += pi_row[j] * c;   // Π^T · c = Σ_i Π[i][:] · c[i]
    }

    // residual = x_unit - x_hat_unit
    float residual[QK_TQ4P_D256];
    float r_sq = 0.0f;
    for (int i = 0; i < d; ++i) {
        residual[i] = x_unit[i] - x_hat_unit[i];
        r_sq += residual[i] * residual[i];
    }
    float res_d = sqrtf(r_sq);

    // QJL signs: proj = S · residual; sign bit = (proj < 0)
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
        const float * pi,
        const float * centroids,
        float orig_norm,
        const uint8_t * qs,
        float * y_out)
{
    uint8_t idx[QK_TQ4P_D256];
    tqp_unpack_indices_bitplane(qs, idx, d / 8);

    // x_hat_unit = Π^T · centroids[idx] (same as quantize path)
    float x_hat_unit[QK_TQ4P_D256];
    for (int j = 0; j < d; ++j) x_hat_unit[j] = 0.0f;
    for (int i = 0; i < d; ++i) {
        float c = centroids[idx[i]];
        const float * pi_row = pi + (size_t)i * d;
        for (int j = 0; j < d; ++j) x_hat_unit[j] += pi_row[j] * c;
    }

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
        const float * pi,
        const float * centroids,
        const float * q,
        const float * Sq,
        const uint8_t * qs,
        const uint8_t * signs,
        float orig_norm,
        float res_d)
{
    // Stage 1: term1 = orig_norm · Σ_i (Π·q)[i] · centroids[idx[i]]
    //                = orig_norm · ⟨q, Π^T · centroids[idx]⟩
    // We compute via q_rot = Π · q.
    float q_rot[QK_TQ4P_D256];
    for (int i = 0; i < d; ++i) {
        float acc = 0.0f;
        const float * pi_row = pi + (size_t)i * d;
        for (int j = 0; j < d; ++j) acc += pi_row[j] * q[j];
        q_rot[i] = acc;
    }

    uint8_t idx[QK_TQ4P_D256];
    tqp_unpack_indices_bitplane(qs, idx, d / 8);

    float term1 = 0.0f;
    for (int i = 0; i < d; ++i) term1 += q_rot[i] * centroids[idx[i]];
    term1 *= orig_norm;

    // Stage 2: term2 = orig_norm · res_d · √(π/2)/d · Σ_i Sq[i] · sign(residual projection)
    //   sign bit: 1 → -1, 0 → +1
    float term2 = 0.0f;
    for (int i = 0; i < d; ++i) {
        float sign_val = (signs[i / 8] & (1u << (i % 8))) ? -1.0f : 1.0f;
        term2 += Sq[i] * sign_val;
    }
    const float sqrt_pi_over_2 = 1.2533141373155001f;  // √(π/2)
    term2 *= orig_norm * res_d * sqrt_pi_over_2 / (float)d;

    return term1 + term2;
}

// ---------- per-d public entry points ----------

#define TQP_DEFINE_ROW_FUNCS(D, PI, S, CENTROIDS, BOUNDS)                                 \
    void ggml_quantize_row_tq4p_d##D(const float * x, block_tq4p_d##D * y, int64_t k) {   \
        assert(k % D == 0);                                                               \
        const int64_t nb = k / D;                                                         \
        for (int64_t b = 0; b < nb; ++b) {                                                \
            float res_d, orig_norm;                                                       \
            tqp_quantize_block(D, PI, S, CENTROIDS, BOUNDS,                               \
                               x + b * D, y[b].qs, y[b].qjl_signs,                        \
                               &res_d, &orig_norm);                                       \
            y[b].orig_norm = tqp_fp32_to_fp16(orig_norm);                                 \
            y[b].res_d     = tqp_fp32_to_fp16(res_d);                                     \
        }                                                                                 \
    }                                                                                     \
                                                                                          \
    void ggml_dequantize_row_tq4p_d##D(const block_tq4p_d##D * x, float * y, int64_t k) { \
        assert(k % D == 0);                                                               \
        const int64_t nb = k / D;                                                         \
        for (int64_t b = 0; b < nb; ++b) {                                                \
            float orig_norm = tqp_fp16_to_fp32(x[b].orig_norm);                           \
            tqp_dequantize_block(D, PI, CENTROIDS, orig_norm, x[b].qs, y + b * D);        \
        }                                                                                 \
    }                                                                                     \
                                                                                          \
    void ggml_tqp_prepare_query_d##D(const float * q, float * Sq) {                       \
        tqp_prepare_query(D, S, q, Sq);                                                   \
    }                                                                                     \
                                                                                          \
    float ggml_tqp_vec_dot_block_d##D(const float * q, const float * Sq,                  \
                                       const block_tq4p_d##D * blk) {                     \
        return tqp_vec_dot_block(D, PI, CENTROIDS, q, Sq, blk->qs, blk->qjl_signs,        \
                                 tqp_fp16_to_fp32(blk->orig_norm),                        \
                                 tqp_fp16_to_fp32(blk->res_d));                           \
    }

TQP_DEFINE_ROW_FUNCS(128, TQP_PI_D128, TQP_S_D128, TQP_CENTROIDS_D128, TQP_BOUNDARIES_D128)
TQP_DEFINE_ROW_FUNCS(256, TQP_PI_D256, TQP_S_D256, TQP_CENTROIDS_D256, TQP_BOUNDARIES_D256)

// ---------- ggml dispatch wrappers ----------

#define TQP_DEFINE_VEC_DOT(D)                                                              \
    void ggml_vec_dot_tq4p_d##D##_f32(int n, float * s, size_t bs,                         \
                                      const void * vx, size_t bx,                          \
                                      const void * vy, size_t by, int nrc) {               \
        assert(nrc == 1);                                                                  \
        assert(n % D == 0);                                                                \
        (void)bs; (void)bx; (void)by;                                                      \
        /* ggml convention: vx = quantized blocks (K side), vy = converted query           \
         * in vec_dot_type (here GGML_TYPE_F32). */                                        \
        const block_tq4p_d##D * blk = (const block_tq4p_d##D *)vx;                         \
        const float * q             = (const float *)vy;                                   \
        const int64_t nb = n / D;                                                          \
                                                                                           \
        /* Amortize S·q across all blocks. */                                              \
        float Sq[D];                                                                       \
        ggml_tqp_prepare_query_d##D(q, Sq);                                                \
                                                                                           \
        float acc = 0.0f;                                                                  \
        for (int64_t b = 0; b < nb; ++b) {                                                 \
            acc += ggml_tqp_vec_dot_block_d##D(q + b * D, Sq, &blk[b]);                    \
        }                                                                                  \
        *s = acc;                                                                          \
    }

TQP_DEFINE_VEC_DOT(128)
TQP_DEFINE_VEC_DOT(256)
