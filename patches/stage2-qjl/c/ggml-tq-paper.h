// Configurable-bit-width TurboQuant(-ish) quantization types for ggml.
//
// Adds explicit TQP_D{dim}_B{bits} block families for 2/3/4-bit Stage-1
// Lloyd-Max quantization plus the 1-bit QJL residual estimator.
//
// Legacy compatibility:
//   TQ4P_D128 / TQ4P_D256 remain the historical B3 aliases.
//   block_tq4p_d* and ggml_*tq4p* entry points forward to the B3 path.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----- Block sizes -----

#define QK_TQP_D64 64
#define QK_TQP_D128 128
#define QK_TQP_D256 256

#define QK_TQ4P_D64  QK_TQP_D64
#define QK_TQ4P_D128 QK_TQP_D128
#define QK_TQ4P_D256 QK_TQP_D256

#define TQP_QS_BYTES(D, BITS) (((D) * (BITS)) / 8)
#define TQP_SIGN_BYTES(D)     ((D) / 8)
#define TQP_BLOCK_SIZE(D, BITS) (5 + TQP_QS_BYTES((D), (BITS)) + TQP_SIGN_BYTES(D))

// ----- Rotation modes (packed in the high bit of the layer byte) -----

#define TQP_ROT_WHT  0u
#define TQP_ROT_HAAR 1u

#define TQP_BIT6_EXPLICIT           (1u << 6)
#define TQP_LAYER_BYTE(layer, rot)  ((uint8_t)(TQP_BIT6_EXPLICIT | (((uint32_t)(rot) & 1u) << 7) | ((uint32_t)(layer) & 0x1fu)))
#define TQP_STORED_BYTE(layer, rot) ((uint8_t)((((uint32_t)(rot) & 1u) << 7) | ((uint32_t)(layer) & 0x1fu)))
#define TQP_EXTRACT_LAYER(byte)     ((uint8_t)((byte) & 0x1fu))
#define TQP_EXTRACT_ROT(byte)       ((uint8_t)(((byte) >> 7) & 1u))
#define TQP_EXTRACT_EXPLICIT(byte)  ((uint8_t)(((byte) >> 6) & 1u))

void    tqp_set_default_rotation(uint8_t rot);
void    tqp_set_thread_rotation(uint8_t rot);
void    tqp_clear_thread_rotation(void);
uint8_t tqp_resolve_rotation(uint8_t layer_byte);

// ----- Block structs -----

#pragma pack(push, 1)

#define TQP_DECLARE_BLOCK(D, BITS)                                              \
    typedef struct {                                                            \
        uint16_t orig_norm;                                                     \
        uint16_t res_d;                                                         \
        uint8_t  layer_idx;                                                     \
        uint8_t  qs[TQP_QS_BYTES(D, BITS)];                                     \
        uint8_t  qjl_signs[TQP_SIGN_BYTES(D)];                                  \
    } block_tqp_d##D##_b##BITS;                                                \
    _Static_assert(sizeof(block_tqp_d##D##_b##BITS) == TQP_BLOCK_SIZE(D, BITS), \
                   "block_tqp_d" #D "_b" #BITS " size")

TQP_DECLARE_BLOCK(64, 2);
TQP_DECLARE_BLOCK(64, 3);
TQP_DECLARE_BLOCK(64, 4);
TQP_DECLARE_BLOCK(128, 2);
TQP_DECLARE_BLOCK(128, 3);
TQP_DECLARE_BLOCK(128, 4);
TQP_DECLARE_BLOCK(256, 2);
TQP_DECLARE_BLOCK(256, 3);
TQP_DECLARE_BLOCK(256, 4);

#pragma pack(pop)

typedef block_tqp_d64_b3 block_tq4p_d64;
typedef block_tqp_d128_b3 block_tq4p_d128;
typedef block_tqp_d256_b3 block_tq4p_d256;

// ----- BF16 / FP16 types -----

#ifndef GGML_FILE_MAGIC
typedef uint16_t ggml_bf16_t;
typedef uint16_t ggml_fp16_t;
#endif

// ----- Generic TQP entry points -----

#define TQP_DECLARE_API(D, BITS)                                                                 \
    void ggml_quantize_row_tqp_d##D##_b##BITS(const float * x, block_tqp_d##D##_b##BITS * y,    \
                                              int64_t k, uint8_t layer_byte);                     \
    void ggml_quantize_row_tqp_d##D##_b##BITS##_bf16(const ggml_bf16_t * x,                      \
                                                     block_tqp_d##D##_b##BITS * y,               \
                                                     int64_t k, uint8_t layer_byte);             \
    void ggml_quantize_row_tqp_d##D##_b##BITS##_f16(const ggml_fp16_t * x,                       \
                                                    block_tqp_d##D##_b##BITS * y,                \
                                                    int64_t k, uint8_t layer_byte);              \
    static inline void ggml_quantize_row_tqp_d##D##_b##BITS##_default(const float * x,           \
                                                                       void * y, int64_t k) {    \
        ggml_quantize_row_tqp_d##D##_b##BITS(x, (block_tqp_d##D##_b##BITS *)y, k, 0x00);        \
    }                                                                                             \
    void ggml_dequantize_row_tqp_d##D##_b##BITS(const block_tqp_d##D##_b##BITS * x,              \
                                                float * y, int64_t k);                            \
    void ggml_tqp_prepare_query_d##D##_b##BITS(const float * q, float * Sq, uint8_t layer_byte); \
    float ggml_tqp_vec_dot_block_d##D##_b##BITS(const float * q, const float * Sq,               \
                                                const block_tqp_d##D##_b##BITS * blk);           \
    void ggml_vec_dot_tqp_d##D##_b##BITS##_f32(int n, float * s, size_t bs,                      \
                                               const void * vx, size_t bx,                        \
                                               const void * vy, size_t by, int nrc);             \
    void ggml_vec_dot_tqp_d##D##_b##BITS##_q8k(int n, float * s, size_t bs,                      \
                                               const void * vx, size_t bx,                        \
                                               const void * vy, size_t by, int nrc)

TQP_DECLARE_API(64, 2);
TQP_DECLARE_API(64, 3);
TQP_DECLARE_API(64, 4);
TQP_DECLARE_API(128, 2);
TQP_DECLARE_API(128, 3);
TQP_DECLARE_API(128, 4);
TQP_DECLARE_API(256, 2);
TQP_DECLARE_API(256, 3);
TQP_DECLARE_API(256, 4);

// ----- Legacy B3 aliases -----

void ggml_quantize_row_tq4p_d64(const float * x, block_tq4p_d64 * y, int64_t k, uint8_t layer_byte);
void ggml_quantize_row_tq4p_d128(const float * x, block_tq4p_d128 * y, int64_t k, uint8_t layer_byte);
void ggml_quantize_row_tq4p_d256(const float * x, block_tq4p_d256 * y, int64_t k, uint8_t layer_byte);

void ggml_quantize_row_tq4p_d64_bf16(const ggml_bf16_t * x, block_tq4p_d64 * y, int64_t k, uint8_t layer_byte);
void ggml_quantize_row_tq4p_d128_bf16(const ggml_bf16_t * x, block_tq4p_d128 * y, int64_t k, uint8_t layer_byte);
void ggml_quantize_row_tq4p_d256_bf16(const ggml_bf16_t * x, block_tq4p_d256 * y, int64_t k, uint8_t layer_byte);
void ggml_quantize_row_tq4p_d64_f16(const ggml_fp16_t * x, block_tq4p_d64 * y, int64_t k, uint8_t layer_byte);
void ggml_quantize_row_tq4p_d128_f16(const ggml_fp16_t * x, block_tq4p_d128 * y, int64_t k, uint8_t layer_byte);
void ggml_quantize_row_tq4p_d256_f16(const ggml_fp16_t * x, block_tq4p_d256 * y, int64_t k, uint8_t layer_byte);

static inline void ggml_quantize_row_tq4p_d64_default(const float * x, void * y, int64_t k) {
    ggml_quantize_row_tq4p_d64(x, (block_tq4p_d64 *)y, k, 0x00);
}
static inline void ggml_quantize_row_tq4p_d128_default(const float * x, void * y, int64_t k) {
    ggml_quantize_row_tq4p_d128(x, (block_tq4p_d128 *)y, k, 0x00);
}
static inline void ggml_quantize_row_tq4p_d256_default(const float * x, void * y, int64_t k) {
    ggml_quantize_row_tq4p_d256(x, (block_tq4p_d256 *)y, k, 0x00);
}

void ggml_dequantize_row_tq4p_d64(const block_tq4p_d64 * x, float * y, int64_t k);
void ggml_dequantize_row_tq4p_d128(const block_tq4p_d128 * x, float * y, int64_t k);
void ggml_dequantize_row_tq4p_d256(const block_tq4p_d256 * x, float * y, int64_t k);

void ggml_tqp_prepare_query_d64(const float * q, float * Sq, uint8_t layer_byte);
void ggml_tqp_prepare_query_d128(const float * q, float * Sq, uint8_t layer_byte);
void ggml_tqp_prepare_query_d256(const float * q, float * Sq, uint8_t layer_byte);

float ggml_tqp_vec_dot_block_d64(const float * q, const float * Sq, const block_tq4p_d64 * blk);
float ggml_tqp_vec_dot_block_d128(const float * q, const float * Sq, const block_tq4p_d128 * blk);
float ggml_tqp_vec_dot_block_d256(const float * q, const float * Sq, const block_tq4p_d256 * blk);

void ggml_vec_dot_tq4p_d64_f32(int n, float * s, size_t bs,
                               const void * vx, size_t bx,
                               const void * vy, size_t by, int nrc);
void ggml_vec_dot_tq4p_d128_f32(int n, float * s, size_t bs,
                                const void * vx, size_t bx,
                                const void * vy, size_t by, int nrc);
void ggml_vec_dot_tq4p_d256_f32(int n, float * s, size_t bs,
                                const void * vx, size_t bx,
                                const void * vy, size_t by, int nrc);

// ----- Q8_K-compatible query block -----

#define QK_Q8K 256

#pragma pack(push, 1)
typedef struct {
    float   d;
    int8_t  qs[QK_Q8K];
    int16_t bsums[QK_Q8K / 16];
} block_q8k_compat;
#pragma pack(pop)

_Static_assert(sizeof(block_q8k_compat) == 292, "block_q8k_compat size");

void ggml_vec_dot_tq4p_d64_q8k(int n, float * s, size_t bs,
                               const void * vx, size_t bx,
                               const void * vy, size_t by, int nrc);
void ggml_vec_dot_tq4p_d128_q8k(int n, float * s, size_t bs,
                                const void * vx, size_t bx,
                                const void * vy, size_t by, int nrc);
void ggml_vec_dot_tq4p_d256_q8k(int n, float * s, size_t bs,
                                const void * vx, size_t bx,
                                const void * vy, size_t by, int nrc);

#define TQP_MAX_LAYERS 32

#ifdef __cplusplus
}
#endif
