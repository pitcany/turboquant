#pragma once
// TQ4P prefill staging — eliminates quantize→dequantize round-trip.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Register a new staging buffer. Caller allocates from pool.
void tqp_staging_put(int device, const void * cache_data,
                     half * staging, size_t staging_bytes, int64_t n_elements);

// Look up staging WITHOUT consuming. Returns staging pointer or nullptr.
half * tqp_staging_find(int device, const void * cache_data);

// Add to the element count of an existing entry (for multi-call accumulation).
void tqp_staging_add_elements(int device, const void * cache_data,
                               int64_t additional);

// Look up and consume. Returns staging pointer if n_elements >= min_elements.
// On success, removes entry. Caller owns buffer and must free staging_bytes.
half * tqp_staging_take(int device, const void * cache_data,
                        int64_t min_elements, size_t * out_staging_bytes);

// Contiguous fp32→fp16 (for CPY staging).
void tqp_staging_f32_to_f16(const float * src, half * dst,
                             int64_t n, cudaStream_t stream);

// Scatter fp32→fp16 (for SET_ROWS staging).
void tqp_staging_f32_to_f16_scatter_i64(
    const float * src, const int64_t * idx, half * dst,
    int64_t n_rows, int64_t src_stride, int d, cudaStream_t stream);
void tqp_staging_f32_to_f16_scatter_i32(
    const float * src, const int32_t * idx, half * dst,
    int64_t n_rows, int64_t src_stride, int d, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
