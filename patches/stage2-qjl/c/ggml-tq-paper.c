// Configurable-bit-width TurboQuant(-ish) quantization — CPU reference.
//
// Byte-exact mirror of patches/stage2-qjl/python/tq_paper_reference.py.
// Supports Stage-1 Lloyd-Max bit-widths 2/3/4 via explicit TQP_D{d}_B{bits}
// block families. The legacy TQ4P entry points remain B3 compatibility
// wrappers.

#include "ggml-tq-paper.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

// ---------- runtime rotation selector ----------

#define TQP_ROT_UNSET 0xffu

static _Thread_local uint8_t g_tqp_thread_rotation = TQP_ROT_UNSET;

static uint8_t g_tqp_process_rotation = TQP_ROT_UNSET;
static pthread_once_t g_tqp_env_once = PTHREAD_ONCE_INIT;

static void tqp_init_env_rotation(void) {
    const char * val = getenv("OLLAMA_TQP_ROTATION");
    if (!val) {
        return;
    }
    if (val[0] == 'h' || val[0] == 'H') {
        g_tqp_process_rotation = TQP_ROT_HAAR;
    } else if (val[0] == 'w' || val[0] == 'W') {
        g_tqp_process_rotation = TQP_ROT_WHT;
    }
}

void tqp_set_default_rotation(uint8_t rot) {
    pthread_once(&g_tqp_env_once, tqp_init_env_rotation);
    g_tqp_process_rotation = rot;
}

void tqp_set_thread_rotation(uint8_t rot) {
    g_tqp_thread_rotation = rot;
}

void tqp_clear_thread_rotation(void) {
    g_tqp_thread_rotation = TQP_ROT_UNSET;
}

uint8_t tqp_resolve_rotation(uint8_t layer_byte) {
    pthread_once(&g_tqp_env_once, tqp_init_env_rotation);

    if (layer_byte & TQP_BIT6_EXPLICIT) {
        return (uint8_t)(layer_byte & ~TQP_BIT6_EXPLICIT);
    }

    uint8_t rot;
    if (g_tqp_thread_rotation != TQP_ROT_UNSET) {
        rot = g_tqp_thread_rotation & 1u;
    } else if (g_tqp_process_rotation != TQP_ROT_UNSET) {
        rot = g_tqp_process_rotation & 1u;
    } else {
        rot = TQP_ROT_WHT;
    }

    return TQP_STORED_BYTE(TQP_EXTRACT_LAYER(layer_byte), rot);
}

// Generated constants. Regenerate via:
//   python3 patches/stage2-qjl/python/generate_constants.py --bits 2,3,4 --dims 64,128,256
#include "tqp_centroids_d64_b2.h"
#include "tqp_centroids_d64_b3.h"
#include "tqp_centroids_d64_b4.h"
#include "tqp_centroids_d128_b2.h"
#include "tqp_centroids_d128_b3.h"
#include "tqp_centroids_d128_b4.h"
#include "tqp_centroids_d256_b2.h"
#include "tqp_centroids_d256_b3.h"
#include "tqp_centroids_d256_b4.h"
#include "tqp_constants_d64.h"
#include "tqp_constants_d128.h"
#include "tqp_constants_d256.h"

static inline uint8_t tqp_layer_idx(uint8_t layer_idx) {
    return (uint8_t)(layer_idx % TQP_MAX_LAYERS);
}

// ---------- fp16 conversion ----------

static inline uint16_t tqp_fp32_to_fp16(float f) {
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

static inline float tqp_bf16_to_fp32(uint16_t h) {
    union { uint32_t u; float f; } v = { (uint32_t)h << 16 };
    return v.f;
}

// ---------- bit pack / unpack ----------

static inline void tqp_pack_indices_bitplane(const uint8_t * idx, uint8_t * out, int n_groups, int bits) {
    for (int g = 0; g < n_groups; ++g) {
        for (int plane = 0; plane < bits; ++plane) {
            uint32_t packed = 0;
            for (int i = 0; i < 8; ++i) {
                const uint32_t v = idx[g * 8 + i];
                packed |= ((v >> plane) & 1u) << i;
            }
            out[g * bits + plane] = (uint8_t)packed;
        }
    }
}

static inline void tqp_unpack_indices_bitplane(const uint8_t * in, uint8_t * idx, int n_groups, int bits) {
    for (int g = 0; g < n_groups; ++g) {
        for (int i = 0; i < 8; ++i) {
            uint8_t value = 0;
            for (int plane = 0; plane < bits; ++plane) {
                value |= (uint8_t)(((in[g * bits + plane] >> i) & 1u) << plane);
            }
            idx[g * 8 + i] = value;
        }
    }
}

// ---------- Walsh-Hadamard transform ----------

static void tqp_wht_inplace(float * x, int d) {
    for (int h = 1; h < d; h <<= 1) {
        for (int i = 0; i < d; i += h << 1) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

static inline void tqp_rht_apply(int d, const float * sigma, const float * v, float * y) {
    for (int i = 0; i < d; ++i) {
        y[i] = sigma[i] * v[i];
    }
    tqp_wht_inplace(y, d);
    const float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; ++i) {
        y[i] *= inv_sqrt_d;
    }
}

static inline void tqp_rht_apply_t(int d, const float * sigma, const float * v, float * y) {
    for (int i = 0; i < d; ++i) {
        y[i] = v[i];
    }
    tqp_wht_inplace(y, d);
    const float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; ++i) {
        y[i] *= sigma[i] * inv_sqrt_d;
    }
}

static inline void tqp_pi_apply(int d, const float * pi, const float * v, float * y) {
    for (int i = 0; i < d; ++i) {
        float acc = 0.0f;
        const float * pi_row = pi + (size_t)i * d;
        for (int j = 0; j < d; ++j) {
            acc += pi_row[j] * v[j];
        }
        y[i] = acc;
    }
}

static inline void tqp_pi_apply_t(int d, const float * pi, const float * v, float * y) {
    for (int j = 0; j < d; ++j) {
        y[j] = 0.0f;
    }
    for (int i = 0; i < d; ++i) {
        const float vi = v[i];
        const float * pi_row = pi + (size_t)i * d;
        for (int j = 0; j < d; ++j) {
            y[j] += pi_row[j] * vi;
        }
    }
}

static inline void tqp_rot_apply(int d, uint8_t rot, const float * sigma, const float * pi,
                                 const float * v, float * y) {
    if (rot == TQP_ROT_HAAR) {
        tqp_pi_apply(d, pi, v, y);
    } else {
        tqp_rht_apply(d, sigma, v, y);
    }
}

static inline void tqp_rot_apply_t(int d, uint8_t rot, const float * sigma, const float * pi,
                                   const float * v, float * y) {
    if (rot == TQP_ROT_HAAR) {
        tqp_pi_apply_t(d, pi, v, y);
    } else {
        tqp_rht_apply_t(d, sigma, v, y);
    }
}

// ---------- Lloyd-Max bucketize ----------

static inline uint8_t tqp_bucketize(float x, const float * bounds, int n_bounds) {
    uint8_t bucket = 0;
    for (int i = 0; i < n_bounds; ++i) {
        bucket += (uint8_t)(x > bounds[i]);
    }
    return bucket;
}

// ---------- core quantize/dequantize ----------

static void tqp_quantize_block(
        int d,
        int bits,
        uint8_t rotation,
        const float * sigma,
        const float * pi,
        const float * s,
        const float * centroids,
        const float * bounds,
        const float * x,
        uint8_t * qs_out,
        uint8_t * signs_out,
        float * res_d_out,
        float * orig_norm_out) {
    float sq = 0.0f;
    for (int i = 0; i < d; ++i) {
        sq += x[i] * x[i];
    }
    float orig_norm = sqrtf(sq);
    if (orig_norm < 1e-8f) {
        orig_norm = 1e-8f;
    }
    const float inv_norm = 1.0f / orig_norm;

    float x_unit[QK_TQP_D256];
    for (int i = 0; i < d; ++i) {
        x_unit[i] = x[i] * inv_norm;
    }

    float x_rot[QK_TQP_D256];
    tqp_rot_apply(d, rotation, sigma, pi, x_unit, x_rot);

    uint8_t idx[QK_TQP_D256];
    const int n_bounds = (1 << bits) - 1;
    for (int i = 0; i < d; ++i) {
        idx[i] = tqp_bucketize(x_rot[i], bounds, n_bounds);
    }

    float x_hat_rot[QK_TQP_D256];
    for (int i = 0; i < d; ++i) {
        x_hat_rot[i] = centroids[idx[i]];
    }
    float x_hat_unit[QK_TQP_D256];
    tqp_rot_apply_t(d, rotation, sigma, pi, x_hat_rot, x_hat_unit);

    float residual[QK_TQP_D256];
    float r_sq = 0.0f;
    for (int i = 0; i < d; ++i) {
        residual[i] = x_unit[i] - x_hat_unit[i];
        r_sq += residual[i] * residual[i];
    }
    const float res_d = sqrtf(r_sq);

    uint8_t signs[QK_TQP_D256 / 8];
    memset(signs, 0, (size_t)(d / 8));
    for (int i = 0; i < d; ++i) {
        float acc = 0.0f;
        const float * s_row = s + (size_t)i * d;
        for (int j = 0; j < d; ++j) {
            acc += s_row[j] * residual[j];
        }
        if (acc < 0.0f) {
            signs[i / 8] |= (uint8_t)(1u << (i % 8));
        }
    }

    tqp_pack_indices_bitplane(idx, qs_out, d / 8, bits);
    memcpy(signs_out, signs, (size_t)(d / 8));
    *res_d_out = res_d;
    *orig_norm_out = orig_norm;
}

static void tqp_dequantize_block(
        int d,
        int bits,
        uint8_t rotation,
        const float * sigma,
        const float * pi,
        const float * centroids,
        float orig_norm,
        const uint8_t * qs,
        float * y_out) {
    uint8_t idx[QK_TQP_D256];
    tqp_unpack_indices_bitplane(qs, idx, d / 8, bits);

    float x_hat_rot[QK_TQP_D256];
    for (int i = 0; i < d; ++i) {
        x_hat_rot[i] = centroids[idx[i]];
    }
    float x_hat_unit[QK_TQP_D256];
    tqp_rot_apply_t(d, rotation, sigma, pi, x_hat_rot, x_hat_unit);

    for (int i = 0; i < d; ++i) {
        y_out[i] = orig_norm * x_hat_unit[i];
    }
}

// ---------- inner product ----------

static void tqp_prepare_query(int d, const float * s, const float * q, float * Sq) {
    for (int i = 0; i < d; ++i) {
        float acc = 0.0f;
        const float * s_row = s + (size_t)i * d;
        for (int j = 0; j < d; ++j) {
            acc += s_row[j] * q[j];
        }
        Sq[i] = acc;
    }
}

static float tqp_vec_dot_block(
        int d,
        int bits,
        uint8_t rotation,
        const float * sigma,
        const float * pi,
        const float * centroids,
        const float * q,
        const float * Sq,
        const uint8_t * qs,
        const uint8_t * signs,
        float orig_norm,
        float res_d) {
    float q_rot[QK_TQP_D256];
    tqp_rot_apply(d, rotation, sigma, pi, q, q_rot);

    uint8_t idx[QK_TQP_D256];
    tqp_unpack_indices_bitplane(qs, idx, d / 8, bits);

    float term1 = 0.0f;
    for (int i = 0; i < d; ++i) {
        term1 += q_rot[i] * centroids[idx[i]];
    }
    term1 *= orig_norm;

    float term2 = 0.0f;
    for (int i = 0; i < d; ++i) {
        const float sign_val = (signs[i / 8] & (1u << (i % 8))) ? -1.0f : 1.0f;
        term2 += Sq[i] * sign_val;
    }
    const float sqrt_pi_over_2 = 1.2533141373155001f;
    term2 *= orig_norm * res_d * sqrt_pi_over_2 / (float)d;

    return term1 + term2;
}

// ---------- per-(d,bits) entry points ----------

#define TQP_DEFINE_ROW_FUNCS(D, BITS, SIGMA_ARR, PI_ARR, S_ARR, CENTROIDS, BOUNDS)                        \
    void ggml_quantize_row_tqp_d##D##_b##BITS(const float * x, block_tqp_d##D##_b##BITS * y,              \
                                              int64_t k, uint8_t layer_byte) {                             \
        assert(k % D == 0);                                                                                 \
        const uint8_t resolved = tqp_resolve_rotation(layer_byte);                                          \
        const uint8_t layer_idx_norm = tqp_layer_idx(TQP_EXTRACT_LAYER(resolved));                          \
        const uint8_t rotation = TQP_EXTRACT_ROT(resolved);                                                 \
        const uint8_t stored = TQP_STORED_BYTE(layer_idx_norm, rotation);                                   \
        const float * sigma = SIGMA_ARR[layer_idx_norm];                                                    \
        const float * pi = PI_ARR[layer_idx_norm];                                                          \
        const float * s = S_ARR[layer_idx_norm];                                                            \
        const int64_t nb = k / D;                                                                           \
        for (int64_t b = 0; b < nb; ++b) {                                                                  \
            float res_d, orig_norm;                                                                         \
            tqp_quantize_block(D, BITS, rotation, sigma, pi, s, CENTROIDS, BOUNDS,                         \
                               x + b * D, y[b].qs, y[b].qjl_signs, &res_d, &orig_norm);                    \
            y[b].orig_norm = tqp_fp32_to_fp16(orig_norm);                                                   \
            y[b].res_d = tqp_fp32_to_fp16(res_d);                                                           \
            y[b].layer_idx = stored;                                                                        \
        }                                                                                                   \
    }                                                                                                       \
                                                                                                            \
    void ggml_dequantize_row_tqp_d##D##_b##BITS(const block_tqp_d##D##_b##BITS * x,                        \
                                                float * y, int64_t k) {                                     \
        assert(k % D == 0);                                                                                 \
        const int64_t nb = k / D;                                                                           \
        for (int64_t b = 0; b < nb; ++b) {                                                                  \
            const uint8_t layer_idx_norm = tqp_layer_idx(TQP_EXTRACT_LAYER(x[b].layer_idx));               \
            const uint8_t rotation = TQP_EXTRACT_ROT(x[b].layer_idx);                                       \
            const float * sigma = SIGMA_ARR[layer_idx_norm];                                                \
            const float * pi = PI_ARR[layer_idx_norm];                                                      \
            const float orig_norm = tqp_fp16_to_fp32(x[b].orig_norm);                                       \
            tqp_dequantize_block(D, BITS, rotation, sigma, pi, CENTROIDS, orig_norm, x[b].qs, y + b * D); \
        }                                                                                                   \
    }                                                                                                       \
                                                                                                            \
    void ggml_tqp_prepare_query_d##D##_b##BITS(const float * q, float * Sq, uint8_t layer_byte) {          \
        tqp_prepare_query(D, S_ARR[tqp_layer_idx(TQP_EXTRACT_LAYER(layer_byte))], q, Sq);                  \
    }                                                                                                       \
                                                                                                            \
    float ggml_tqp_vec_dot_block_d##D##_b##BITS(const float * q, const float * Sq,                         \
                                                const block_tqp_d##D##_b##BITS * blk) {                    \
        const uint8_t layer_idx_norm = tqp_layer_idx(TQP_EXTRACT_LAYER(blk->layer_idx));                   \
        const uint8_t rotation = TQP_EXTRACT_ROT(blk->layer_idx);                                           \
        return tqp_vec_dot_block(D, BITS, rotation,                                                         \
                                 SIGMA_ARR[layer_idx_norm], PI_ARR[layer_idx_norm], CENTROIDS,             \
                                 q, Sq, blk->qs, blk->qjl_signs,                                            \
                                 tqp_fp16_to_fp32(blk->orig_norm),                                          \
                                 tqp_fp16_to_fp32(blk->res_d));                                             \
    }

TQP_DEFINE_ROW_FUNCS(64, 2, TQP_SIGMA_D64, TQP_PI_D64, TQP_S_D64, TQP_CENTROIDS_D64_B2, TQP_BOUNDARIES_D64_B2)
TQP_DEFINE_ROW_FUNCS(64, 3, TQP_SIGMA_D64, TQP_PI_D64, TQP_S_D64, TQP_CENTROIDS_D64_B3, TQP_BOUNDARIES_D64_B3)
TQP_DEFINE_ROW_FUNCS(64, 4, TQP_SIGMA_D64, TQP_PI_D64, TQP_S_D64, TQP_CENTROIDS_D64_B4, TQP_BOUNDARIES_D64_B4)
TQP_DEFINE_ROW_FUNCS(128, 2, TQP_SIGMA_D128, TQP_PI_D128, TQP_S_D128, TQP_CENTROIDS_D128_B2, TQP_BOUNDARIES_D128_B2)
TQP_DEFINE_ROW_FUNCS(128, 3, TQP_SIGMA_D128, TQP_PI_D128, TQP_S_D128, TQP_CENTROIDS_D128_B3, TQP_BOUNDARIES_D128_B3)
TQP_DEFINE_ROW_FUNCS(128, 4, TQP_SIGMA_D128, TQP_PI_D128, TQP_S_D128, TQP_CENTROIDS_D128_B4, TQP_BOUNDARIES_D128_B4)
TQP_DEFINE_ROW_FUNCS(256, 2, TQP_SIGMA_D256, TQP_PI_D256, TQP_S_D256, TQP_CENTROIDS_D256_B2, TQP_BOUNDARIES_D256_B2)
TQP_DEFINE_ROW_FUNCS(256, 3, TQP_SIGMA_D256, TQP_PI_D256, TQP_S_D256, TQP_CENTROIDS_D256_B3, TQP_BOUNDARIES_D256_B3)
TQP_DEFINE_ROW_FUNCS(256, 4, TQP_SIGMA_D256, TQP_PI_D256, TQP_S_D256, TQP_CENTROIDS_D256_B4, TQP_BOUNDARIES_D256_B4)

#define TQP_DEFINE_ROW_FUNCS_DTYPE(D, BITS, SUFFIX, INTYPE, CONVERTER)                                      \
    void ggml_quantize_row_tqp_d##D##_b##BITS##_##SUFFIX(const INTYPE * x,                                  \
                                                         block_tqp_d##D##_b##BITS * y,                       \
                                                         int64_t k, uint8_t layer_byte) {                    \
        assert(k % D == 0);                                                                                  \
        const uint8_t resolved = tqp_resolve_rotation(layer_byte);                                           \
        const uint8_t layer_idx_norm = tqp_layer_idx(TQP_EXTRACT_LAYER(resolved));                           \
        const uint8_t rotation = TQP_EXTRACT_ROT(resolved);                                                  \
        const uint8_t stored = TQP_STORED_BYTE(layer_idx_norm, rotation);                                    \
        float buf[QK_TQP_D256];                                                                              \
        const int64_t nb = k / D;                                                                            \
        for (int64_t b = 0; b < nb; ++b) {                                                                   \
            for (int i = 0; i < D; ++i) {                                                                    \
                buf[i] = CONVERTER(x[b * D + i]);                                                            \
            }                                                                                                \
            float res_d, orig_norm;                                                                          \
            tqp_quantize_block(D, BITS, rotation,                                                            \
                               TQP_SIGMA_D##D[layer_idx_norm],                                               \
                               TQP_PI_D##D[layer_idx_norm],                                                  \
                               TQP_S_D##D[layer_idx_norm],                                                   \
                               TQP_CENTROIDS_D##D##_B##BITS,                                                 \
                               TQP_BOUNDARIES_D##D##_B##BITS,                                                \
                               buf, y[b].qs, y[b].qjl_signs, &res_d, &orig_norm);                            \
            y[b].orig_norm = tqp_fp32_to_fp16(orig_norm);                                                    \
            y[b].res_d = tqp_fp32_to_fp16(res_d);                                                            \
            y[b].layer_idx = stored;                                                                         \
        }                                                                                                    \
    }

TQP_DEFINE_ROW_FUNCS_DTYPE(64, 2, bf16, uint16_t, tqp_bf16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(64, 3, bf16, uint16_t, tqp_bf16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(64, 4, bf16, uint16_t, tqp_bf16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(128, 2, bf16, uint16_t, tqp_bf16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(128, 3, bf16, uint16_t, tqp_bf16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(128, 4, bf16, uint16_t, tqp_bf16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(256, 2, bf16, uint16_t, tqp_bf16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(256, 3, bf16, uint16_t, tqp_bf16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(256, 4, bf16, uint16_t, tqp_bf16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(64, 2, f16, uint16_t, tqp_fp16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(64, 3, f16, uint16_t, tqp_fp16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(64, 4, f16, uint16_t, tqp_fp16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(128, 2, f16, uint16_t, tqp_fp16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(128, 3, f16, uint16_t, tqp_fp16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(128, 4, f16, uint16_t, tqp_fp16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(256, 2, f16, uint16_t, tqp_fp16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(256, 3, f16, uint16_t, tqp_fp16_to_fp32)
TQP_DEFINE_ROW_FUNCS_DTYPE(256, 4, f16, uint16_t, tqp_fp16_to_fp32)

// ---------- ggml dispatch wrappers ----------

#define TQP_DEFINE_VEC_DOT(D, BITS)                                                                          \
    void ggml_vec_dot_tqp_d##D##_b##BITS##_f32(int n, float * s, size_t bs,                                 \
                                               const void * vx, size_t bx,                                   \
                                               const void * vy, size_t by, int nrc) {                        \
        assert(nrc == 1);                                                                                    \
        assert(n % D == 0);                                                                                  \
        (void)bs; (void)bx; (void)by;                                                                        \
        const block_tqp_d##D##_b##BITS * blk = (const block_tqp_d##D##_b##BITS *)vx;                        \
        const float * q = (const float *)vy;                                                                 \
        const int64_t nb = n / D;                                                                            \
        float acc = 0.0f;                                                                                    \
        for (int64_t b = 0; b < nb; ++b) {                                                                   \
            float Sq[D];                                                                                     \
            ggml_tqp_prepare_query_d##D##_b##BITS(q + b * D, Sq, blk[b].layer_idx);                         \
            acc += ggml_tqp_vec_dot_block_d##D##_b##BITS(q + b * D, Sq, &blk[b]);                           \
        }                                                                                                    \
        *s = acc;                                                                                            \
    }

TQP_DEFINE_VEC_DOT(64, 2)
TQP_DEFINE_VEC_DOT(64, 3)
TQP_DEFINE_VEC_DOT(64, 4)
TQP_DEFINE_VEC_DOT(128, 2)
TQP_DEFINE_VEC_DOT(128, 3)
TQP_DEFINE_VEC_DOT(128, 4)
TQP_DEFINE_VEC_DOT(256, 2)
TQP_DEFINE_VEC_DOT(256, 3)
TQP_DEFINE_VEC_DOT(256, 4)

static inline void tqp_dequant_q8k(const block_q8k_compat * blk, float * out) {
    const float d = blk->d;
    for (int i = 0; i < QK_Q8K; ++i) {
        out[i] = d * (float)blk->qs[i];
    }
}

#define TQP_DEFINE_VEC_DOT_Q8K(D, BITS)                                                                      \
    void ggml_vec_dot_tqp_d##D##_b##BITS##_q8k(int n, float * s, size_t bs,                                 \
                                               const void * vx, size_t bx,                                   \
                                               const void * vy, size_t by, int nrc) {                        \
        assert(nrc == 1);                                                                                    \
        assert(n % QK_Q8K == 0);                                                                             \
        (void)bs; (void)bx; (void)by;                                                                        \
        const block_tqp_d##D##_b##BITS * blk = (const block_tqp_d##D##_b##BITS *)vx;                        \
        const block_q8k_compat * q8k = (const block_q8k_compat *)vy;                                         \
        const int64_t nb_q8k = n / QK_Q8K;                                                                   \
        float acc = 0.0f;                                                                                    \
        float q_buf[QK_Q8K];                                                                                 \
        int64_t tqp_b = 0;                                                                                   \
        for (int64_t qb = 0; qb < nb_q8k; ++qb) {                                                            \
            tqp_dequant_q8k(&q8k[qb], q_buf);                                                                \
            for (int sub = 0; sub < QK_Q8K / D; ++sub, ++tqp_b) {                                            \
                const float * q = q_buf + sub * D;                                                           \
                float Sq[D];                                                                                 \
                ggml_tqp_prepare_query_d##D##_b##BITS(q, Sq, blk[tqp_b].layer_idx);                         \
                acc += ggml_tqp_vec_dot_block_d##D##_b##BITS(q, Sq, &blk[tqp_b]);                           \
            }                                                                                                \
        }                                                                                                    \
        *s = acc;                                                                                            \
    }

TQP_DEFINE_VEC_DOT_Q8K(64, 2)
TQP_DEFINE_VEC_DOT_Q8K(64, 3)
TQP_DEFINE_VEC_DOT_Q8K(64, 4)
TQP_DEFINE_VEC_DOT_Q8K(128, 2)
TQP_DEFINE_VEC_DOT_Q8K(128, 3)
TQP_DEFINE_VEC_DOT_Q8K(128, 4)
TQP_DEFINE_VEC_DOT_Q8K(256, 2)
TQP_DEFINE_VEC_DOT_Q8K(256, 3)
TQP_DEFINE_VEC_DOT_Q8K(256, 4)

// ---------- Legacy B3 wrappers ----------

#define TQP_DEFINE_LEGACY_WRAPPERS(D)                                                                       \
    void ggml_quantize_row_tq4p_d##D(const float * x, block_tq4p_d##D * y, int64_t k, uint8_t layer_byte) { \
        ggml_quantize_row_tqp_d##D##_b3(x, y, k, layer_byte);                                               \
    }                                                                                                       \
    void ggml_quantize_row_tq4p_d##D##_bf16(const ggml_bf16_t * x, block_tq4p_d##D * y,                    \
                                            int64_t k, uint8_t layer_byte) {                                \
        ggml_quantize_row_tqp_d##D##_b3_bf16(x, y, k, layer_byte);                                          \
    }                                                                                                       \
    void ggml_quantize_row_tq4p_d##D##_f16(const ggml_fp16_t * x, block_tq4p_d##D * y,                     \
                                           int64_t k, uint8_t layer_byte) {                                 \
        ggml_quantize_row_tqp_d##D##_b3_f16(x, y, k, layer_byte);                                           \
    }                                                                                                       \
    void ggml_dequantize_row_tq4p_d##D(const block_tq4p_d##D * x, float * y, int64_t k) {                  \
        ggml_dequantize_row_tqp_d##D##_b3(x, y, k);                                                         \
    }                                                                                                       \
    void ggml_tqp_prepare_query_d##D(const float * q, float * Sq, uint8_t layer_byte) {                    \
        ggml_tqp_prepare_query_d##D##_b3(q, Sq, layer_byte);                                                \
    }                                                                                                       \
    float ggml_tqp_vec_dot_block_d##D(const float * q, const float * Sq, const block_tq4p_d##D * blk) {    \
        return ggml_tqp_vec_dot_block_d##D##_b3(q, Sq, blk);                                                \
    }                                                                                                       \
    void ggml_vec_dot_tq4p_d##D##_f32(int n, float * s, size_t bs,                                          \
                                      const void * vx, size_t bx,                                           \
                                      const void * vy, size_t by, int nrc) {                                \
        ggml_vec_dot_tqp_d##D##_b3_f32(n, s, bs, vx, bx, vy, by, nrc);                                      \
    }                                                                                                       \
    void ggml_vec_dot_tq4p_d##D##_q8k(int n, float * s, size_t bs,                                          \
                                      const void * vx, size_t bx,                                           \
                                      const void * vy, size_t by, int nrc) {                                \
        ggml_vec_dot_tqp_d##D##_b3_q8k(n, s, bs, vx, bx, vy, by, nrc);                                      \
    }

TQP_DEFINE_LEGACY_WRAPPERS(64)
TQP_DEFINE_LEGACY_WRAPPERS(128)
TQP_DEFINE_LEGACY_WRAPPERS(256)
