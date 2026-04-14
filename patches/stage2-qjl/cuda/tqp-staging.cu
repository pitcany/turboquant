#include "tqp-staging.cuh"
#include <cassert>
#include <cstdio>

// ---------- staging cache ----------

#define TQP_STAGING_MAX_DEVICES 16
#define TQP_STAGING_MAX_ENTRIES 4

struct TqpStagingEntry {
    const void * cache_data;
    half * staging;
    size_t staging_bytes;
    int64_t n_elements;
};

struct TqpStagingCache {
    TqpStagingEntry entries[TQP_STAGING_MAX_ENTRIES];
    int n;
};

static TqpStagingCache g_tqp_staging[TQP_STAGING_MAX_DEVICES] = {};

static TqpStagingEntry * tqp_find_entry(int device, const void * cache_data) {
    if (device < 0 || device >= TQP_STAGING_MAX_DEVICES) return nullptr;
    TqpStagingCache & c = g_tqp_staging[device];
    for (int i = 0; i < c.n; ++i) {
        if (c.entries[i].cache_data == cache_data) return &c.entries[i];
    }
    return nullptr;
}

extern "C" void tqp_staging_put(int device, const void * cache_data,
                                half * staging, size_t staging_bytes,
                                int64_t n_elements) {
    if (device < 0 || device >= TQP_STAGING_MAX_DEVICES) return;
    TqpStagingCache & c = g_tqp_staging[device];
    // Should not already exist (caller checks with find first)
    // NOTE: eviction here drops the reference to a pool-allocated GPU
    // buffer without freeing it. The pool reclaims it on reset, but under
    // sustained pressure this can temporarily balloon pool usage. Callers
    // that drive the cache past TQP_STAGING_MAX_ENTRIES should add a
    // real free/reuse path before this code is enabled in production.
    if (c.n >= TQP_STAGING_MAX_ENTRIES) {
        for (int i = 0; i < c.n - 1; ++i) c.entries[i] = c.entries[i + 1];
        c.n--;
    }
    c.entries[c.n++] = {cache_data, staging, staging_bytes, n_elements};
}

extern "C" half * tqp_staging_find(int device, const void * cache_data) {
    TqpStagingEntry * e = tqp_find_entry(device, cache_data);
    return e ? e->staging : nullptr;
}

extern "C" void tqp_staging_add_elements(int device, const void * cache_data,
                                          int64_t additional) {
    TqpStagingEntry * e = tqp_find_entry(device, cache_data);
    if (e) {
        e->n_elements += additional;
    }
}

extern "C" half * tqp_staging_take(int device, const void * cache_data,
                                   int64_t min_elements,
                                   size_t * out_staging_bytes) {
    if (device < 0 || device >= TQP_STAGING_MAX_DEVICES) return nullptr;
    TqpStagingCache & c = g_tqp_staging[device];
    for (int i = 0; i < c.n; ++i) {
        if (c.entries[i].cache_data == cache_data) {
            if (c.entries[i].n_elements >= min_elements) {
                half * result = c.entries[i].staging;
                if (out_staging_bytes) *out_staging_bytes = c.entries[i].staging_bytes;
                for (int j = i; j < c.n - 1; ++j) c.entries[j] = c.entries[j + 1];
                c.n--;
                return result;
            }
            // Insufficient coverage — discard
            for (int j = i; j < c.n - 1; ++j) c.entries[j] = c.entries[j + 1];
            c.n--;
            return nullptr;
        }
    }
    return nullptr;
}

// ---------- fp32 → fp16 kernels ----------

__global__ static void k_f32_to_f16(const float * __restrict__ src,
                                     half * __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

extern "C" void tqp_staging_f32_to_f16(const float * src, half * dst,
                                        int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    const int bs = 256;
    k_f32_to_f16<<<(int)((n + bs - 1) / bs), bs, 0, stream>>>(src, dst, n);
}

template<typename idx_t>
__global__ static void k_f32_to_f16_scatter(
        const float * __restrict__ src,
        const idx_t * __restrict__ idx,
        half * __restrict__ dst,
        int64_t n_rows, int64_t src_stride, int d) {
    const int64_t row = (int64_t)blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= n_rows || tid >= d) return;
    const int64_t dst_row = (int64_t)idx[row];
    dst[dst_row * d + tid] = __float2half_rn(src[row * src_stride + tid]);
}

// CUDA silently fails kernel launches when blockDim > 1024. Current TQ4P
// head dims (64/128/256) are safe; guard against a future caller passing
// a larger dimension so the failure is visible rather than a silent no-op.
#define TQP_STAGING_MAX_D 1024

extern "C" void tqp_staging_f32_to_f16_scatter_i64(
        const float * src, const int64_t * idx, half * dst,
        int64_t n_rows, int64_t src_stride, int d, cudaStream_t stream) {
    if (n_rows <= 0) return;
    assert(d > 0 && d <= TQP_STAGING_MAX_D);
    k_f32_to_f16_scatter<<<(int)n_rows, d, 0, stream>>>(src, idx, dst, n_rows, src_stride, d);
}

extern "C" void tqp_staging_f32_to_f16_scatter_i32(
        const float * src, const int32_t * idx, half * dst,
        int64_t n_rows, int64_t src_stride, int d, cudaStream_t stream) {
    if (n_rows <= 0) return;
    assert(d > 0 && d <= TQP_STAGING_MAX_D);
    k_f32_to_f16_scatter<<<(int)n_rows, d, 0, stream>>>(src, idx, dst, n_rows, src_stride, d);
}
