#include "tqp-constants-cuda.cuh"

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// Q8_K block layout (matches ggml block_q8_K / block_q8k_compat).
#define QK_Q8K 256
#pragma pack(push, 1)
typedef struct {
    float   d;
    int8_t  qs[QK_Q8K];
    int16_t bsums[QK_Q8K / 16];
} block_q8k_cuda;
#pragma pack(pop)

// Dequantize Q8_K blocks to contiguous fp32.
// Grid: (n_blocks,), Block: (256,).
__global__ static void tqp_dequantize_q8k_kernel(
        const block_q8k_cuda * __restrict__ src,
        float * __restrict__ dst,
        int64_t n_blocks) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) return;
    const int tid = threadIdx.x;
    const float d = src[b].d;
    dst[b * QK_Q8K + tid] = d * (float)src[b].qs[tid];
}

extern "C" void ggml_cuda_tqp_prepare_query_d128(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_d256(const float * q, float * Sq, float * q_rot, uint8_t layer_byte, cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d128(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);
extern "C" void ggml_cuda_tqp_prepare_query_batch_d256(
        const float * q, float * Sq, float * q_rot,
        int64_t ne11, int64_t ne12, int64_t ne13,
        int64_t s11, int64_t s12, int64_t s13,
        uint8_t layer_byte,
        cudaStream_t stream);

template<int D, typename Block>
__device__ static inline float tqp_vec_dot_block_device(
        const Block * __restrict__ blk,
        const float * __restrict__ Sq,
        const float * __restrict__ q_rot,
        const float * __restrict__ centroids) {
    __shared__ uint8_t smem_qs[96];
    __shared__ uint8_t smem_signs[32];
    __shared__ float partial_t1[8];
    __shared__ float partial_t2[8];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int idx_bytes = D * 3 / 8;
    const int sign_bytes = D / 8;

    for (int i = tid; i < idx_bytes; i += blockDim.x) {
        smem_qs[i] = blk->qs[i];
    }
    for (int i = tid; i < sign_bytes; i += blockDim.x) {
        smem_signs[i] = blk->qjl_signs[i];
    }
    __syncthreads();

    float t1 = 0.0f;
    float t2 = 0.0f;
    const int base = tid * 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int elem = base + e;
        const uint8_t idx = tqp_unpack_index_bitplane(smem_qs, elem);
        const float sign_val = tqp_unpack_sign_pm1(smem_signs, elem);
        t1 += q_rot[elem] * centroids[idx];
        t2 += Sq[elem] * sign_val;
    }

    t1 = tqp_warp_reduce_sum(t1);
    t2 = tqp_warp_reduce_sum(t2);

    if (lane == 0) {
        partial_t1[warp] = t1;
        partial_t2[warp] = t2;
    }
    __syncthreads();

    if (tid == 0) {
        const int nwarps = D / 128;
        float total_t1 = 0.0f;
        float total_t2 = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            total_t1 += partial_t1[w];
            total_t2 += partial_t2[w];
        }
        const float orig_norm = tqp_fp16_to_fp32_device(blk->orig_norm);
        const float res_d = tqp_fp16_to_fp32_device(blk->res_d);
        total_t1 *= orig_norm;
        total_t2 *= orig_norm * res_d * (TQP_SQRT_PI_OVER_2 / (float)D);
        return total_t1 + total_t2;
    }

    return 0.0f;
}

__global__ static void tqp_vec_dot_kernel_d128(
        const block_tq4p_d128 * __restrict__ blocks,
        const float * __restrict__ Sq,
        const float * __restrict__ q_rot,
        float * __restrict__ out,
        int64_t n_blocks) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) {
        return;
    }
    const float result = tqp_vec_dot_block_device<QK_TQ4P_D128>(&blocks[b], Sq, q_rot, c_tqp_centroids_d128);
    if (threadIdx.x == 0) {
        out[b] = result;
    }
}

__global__ static void tqp_vec_dot_kernel_d256(
        const block_tq4p_d256 * __restrict__ blocks,
        const float * __restrict__ Sq,
        const float * __restrict__ q_rot,
        float * __restrict__ out,
        int64_t n_blocks) {
    const int64_t b = (int64_t)blockIdx.x;
    if (b >= n_blocks) {
        return;
    }
    const float result = tqp_vec_dot_block_device<QK_TQ4P_D256>(&blocks[b], Sq, q_rot, c_tqp_centroids_d256);
    if (threadIdx.x == 0) {
        out[b] = result;
    }
}

template<int D, typename Block>
__global__ static void tqp_vec_dot_ggml_kernel(
        const Block * __restrict__ blocks,
        const float * __restrict__ Sq,
        const float * __restrict__ q_rot,
        float * __restrict__ dst,
        int64_t ne11,
        int64_t ne2,
        int64_t s01,
        int64_t s02,
        int64_t s03,
        int64_t d_s1,
        int64_t d_s2,
        int64_t d_s3,
        int64_t channel_ratio,
        int64_t sample_ratio,
        const float * __restrict__ centroids) {
    const int64_t row = (int64_t)blockIdx.x;
    const int64_t col = (int64_t)blockIdx.y;
    const int64_t channel_dst = (int64_t)blockIdx.z % ne2;
    const int64_t sample_dst = (int64_t)blockIdx.z / ne2;
    const int64_t channel_x = channel_dst / channel_ratio;
    const int64_t sample_x = sample_dst / sample_ratio;
    const int64_t query_index = ((sample_dst * ne2 + channel_dst) * ne11 + col);

    const Block * blk = blocks + sample_x * s03 + channel_x * s02 + row * s01;
    const float * Sq_i = Sq + query_index * D;
    const float * q_rot_i = q_rot + query_index * D;

    const float result = tqp_vec_dot_block_device<D>(blk, Sq_i, q_rot_i, centroids);
    if (threadIdx.x == 0) {
        dst[sample_dst * d_s3 + channel_dst * d_s2 + col * d_s1 + row] = result;
    }
}

extern "C" void ggml_cuda_tqp_vec_dot_blocks_d128(
        const void * blocks, const float * Sq, const float * q_rot,
        float * out, int64_t n_blocks, cudaStream_t stream) {
    if (tqp_cuda_init(QK_TQ4P_D128) != cudaSuccess) {
        return;
    }
    tqp_vec_dot_kernel_d128<<<(unsigned int)n_blocks, 32, 0, stream>>>(
        (const block_tq4p_d128 *)blocks, Sq, q_rot, out, n_blocks);
}

extern "C" void ggml_cuda_tqp_vec_dot_blocks_d256(
        const void * blocks, const float * Sq, const float * q_rot,
        float * out, int64_t n_blocks, cudaStream_t stream) {
    if (tqp_cuda_init(QK_TQ4P_D256) != cudaSuccess) {
        return;
    }
    tqp_vec_dot_kernel_d256<<<(unsigned int)n_blocks, 64, 0, stream>>>(
        (const block_tq4p_d256 *)blocks, Sq, q_rot, out, n_blocks);
}

template<typename Block>
static int tqp_cuda_vec_dot_row_host(
        int d,
        const float * q_host,
        const Block * blocks_host,
        float * out_host,
        int64_t n_blocks,
        void (*prepare_fn)(const float *, float *, float *, uint8_t, cudaStream_t),
        void (*vec_dot_fn)(const void *, const float *, const float *, float *, int64_t, cudaStream_t)) {
    if (n_blocks <= 0) {
        return 1;
    }

    // All blocks in a single launch are assumed to share a layer_byte
    // (layer + rotation) per the ggml per-layer tensor convention. Read
    // it from the first host block.
    const uint8_t layer_byte = blocks_host[0].layer_idx;

    float * q_dev = nullptr;
    float * Sq_dev = nullptr;
    float * q_rot_dev = nullptr;
    Block * blocks_dev = nullptr;
    float * out_dev = nullptr;

    const size_t q_bytes = (size_t)d * sizeof(float);
    const size_t block_bytes = (size_t)n_blocks * sizeof(Block);
    const size_t out_bytes = (size_t)n_blocks * sizeof(float);

    cudaError_t err = cudaMalloc((void **)&q_dev, q_bytes);
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void **)&Sq_dev, q_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&q_rot_dev, q_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&blocks_dev, block_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&out_dev, out_bytes);
    if (err != cudaSuccess) goto done;

    err = cudaMemcpy(q_dev, q_host, q_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto done;
    err = cudaMemcpy(blocks_dev, blocks_host, block_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto done;

    prepare_fn(q_dev, Sq_dev, q_rot_dev, layer_byte, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto done;

    vec_dot_fn(blocks_dev, Sq_dev, q_rot_dev, out_dev, n_blocks, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto done;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto done;

    err = cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost);

done:
    cudaFree(out_dev);
    cudaFree(blocks_dev);
    cudaFree(q_rot_dev);
    cudaFree(Sq_dev);
    cudaFree(q_dev);
    return (int)err;
}

extern "C" int tqp_cuda_vec_dot_row_d128(
        const float * q_host,
        const block_tq4p_d128 * blocks_host,
        float * out_host,
        int64_t n_blocks) {
    return tqp_cuda_vec_dot_row_host<block_tq4p_d128>(
        QK_TQ4P_D128, q_host, blocks_host, out_host, n_blocks,
        ggml_cuda_tqp_prepare_query_d128,
        ggml_cuda_tqp_vec_dot_blocks_d128);
}

extern "C" int tqp_cuda_vec_dot_row_d256(
        const float * q_host,
        const block_tq4p_d256 * blocks_host,
        float * out_host,
        int64_t n_blocks) {
    return tqp_cuda_vec_dot_row_host<block_tq4p_d256>(
        QK_TQ4P_D256, q_host, blocks_host, out_host, n_blocks,
        ggml_cuda_tqp_prepare_query_d256,
        ggml_cuda_tqp_vec_dot_blocks_d256);
}

extern "C" float tqp_cuda_vec_dot_block_d128(
        const float * q_host,
        const block_tq4p_d128 * block_host) {
    float out = 0.0f;
    const int err = tqp_cuda_vec_dot_row_d128(q_host, block_host, &out, 1);
    return err == 0 ? out : 0.0f;
}

extern "C" float tqp_cuda_vec_dot_block_d256(
        const float * q_host,
        const block_tq4p_d256 * block_host) {
    float out = 0.0f;
    const int err = tqp_cuda_vec_dot_row_d256(q_host, block_host, &out, 1);
    return err == 0 ? out : 0.0f;
}

// Q8_K-aware test wrappers: dequantize Q8_K query to fp32 on device,
// then run the existing prepare_query + vec_dot pipeline.
template<typename Block>
static int tqp_cuda_vec_dot_q8k_host(
        int d,
        const void * q8k_host,    // block_q8k_cuda blocks
        const Block * blocks_host,
        float * out_host,
        int64_t n_blocks,
        void (*prepare_fn)(const float *, float *, float *, uint8_t, cudaStream_t),
        void (*vec_dot_fn)(const void *, const float *, const float *, float *, int64_t, cudaStream_t)) {
    if (n_blocks <= 0) return 1;

    const uint8_t layer_byte = blocks_host[0].layer_idx;
    // Q8_K has 256 elements per block. For d=128, d fits in 1/2 Q8K block;
    // for d=256, d fits in 1 Q8K block. We still need to allocate a full
    // n_q8k_blocks * QK_Q8K fp32 scratch because the dequantize kernel
    // launches with QK_Q8K (=256) threads per block and each writes its
    // slot — if we only sized for d floats, threads d..QK_Q8K-1 would run
    // past the buffer end. Only the first d entries are consumed by
    // prepare_query; the tail is harmlessly unused.
    const int64_t n_q8k_blocks = ((int64_t)d + QK_Q8K - 1) / QK_Q8K;
    const size_t q8k_bytes = (size_t)n_q8k_blocks * sizeof(block_q8k_cuda);
    const size_t q_fp32_bytes = (size_t)n_q8k_blocks * QK_Q8K * sizeof(float);

    float * q_fp32_dev = nullptr;
    block_q8k_cuda * q8k_dev = nullptr;
    float * Sq_dev = nullptr;
    float * q_rot_dev = nullptr;
    Block * blocks_dev = nullptr;
    float * out_dev = nullptr;

    const size_t q_bytes = (size_t)d * sizeof(float);
    const size_t block_bytes = (size_t)n_blocks * sizeof(Block);
    const size_t out_bytes = (size_t)n_blocks * sizeof(float);

    cudaError_t err = cudaMalloc((void **)&q8k_dev, q8k_bytes);
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void **)&q_fp32_dev, q_fp32_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&Sq_dev, q_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&q_rot_dev, q_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&blocks_dev, block_bytes);
    if (err != cudaSuccess) goto done;
    err = cudaMalloc((void **)&out_dev, out_bytes);
    if (err != cudaSuccess) goto done;

    err = cudaMemcpy(q8k_dev, q8k_host, q8k_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto done;
    err = cudaMemcpy(blocks_dev, blocks_host, block_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto done;

    // Dequantize Q8_K to fp32 on device.
    tqp_dequantize_q8k_kernel<<<(unsigned int)n_q8k_blocks, QK_Q8K, 0, 0>>>(
        q8k_dev, q_fp32_dev, n_q8k_blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto done;

    // Run prepare_query + vec_dot with the fp32 query.
    prepare_fn(q_fp32_dev, Sq_dev, q_rot_dev, layer_byte, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto done;
    vec_dot_fn(blocks_dev, Sq_dev, q_rot_dev, out_dev, n_blocks, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto done;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto done;

    err = cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost);

done:
    cudaFree(out_dev);
    cudaFree(blocks_dev);
    cudaFree(q_rot_dev);
    cudaFree(Sq_dev);
    cudaFree(q_fp32_dev);
    cudaFree(q8k_dev);
    return (int)err;
}

extern "C" int tqp_cuda_vec_dot_q8k_d128(
        const void * q8k_host,
        const block_tq4p_d128 * blocks_host,
        float * out_host,
        int64_t n_blocks) {
    return tqp_cuda_vec_dot_q8k_host<block_tq4p_d128>(
        QK_TQ4P_D128, q8k_host, blocks_host, out_host, n_blocks,
        ggml_cuda_tqp_prepare_query_d128,
        ggml_cuda_tqp_vec_dot_blocks_d128);
}

extern "C" int tqp_cuda_vec_dot_q8k_d256(
        const void * q8k_host,
        const block_tq4p_d256 * blocks_host,
        float * out_host,
        int64_t n_blocks) {
    return tqp_cuda_vec_dot_q8k_host<block_tq4p_d256>(
        QK_TQ4P_D256, q8k_host, blocks_host, out_host, n_blocks,
        ggml_cuda_tqp_prepare_query_d256,
        ggml_cuda_tqp_vec_dot_blocks_d256);
}

#if __has_include("common.cuh")
#include "common.cuh"

extern "C" void ggml_cuda_op_tqp_vec_dot(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_Q8_K);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4P_D128 || src0->type == GGML_TYPE_TQ4P_D256);
    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int d = src0->type == GGML_TYPE_TQ4P_D128 ? QK_TQ4P_D128 : QK_TQ4P_D256;
    GGML_ASSERT(ne00 == d);
    GGML_ASSERT(ne10 == d);
    GGML_ASSERT(nb0 == (int64_t)sizeof(float));
    GGML_ASSERT(ne12 == ne2);
    GGML_ASSERT(ne13 == ne3);
    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(tqp_cuda_init(d));

    const int64_t n_queries = ne11 * ne12 * ne13;
    ggml_cuda_pool_alloc<float> Sq_alloc(ctx.pool(), n_queries * d);
    ggml_cuda_pool_alloc<float> q_rot_alloc(ctx.pool(), n_queries * d);
    float * Sq = Sq_alloc.get();
    float * q_rot = q_rot_alloc.get();

    // If src1 is Q8_K, dequantize to a contiguous fp32 buffer first.
    ggml_cuda_pool_alloc<float> q8k_fp32_alloc;
    const float * src1_d;
    int64_t q_s11, q_s12, q_s13;

    if (src1->type == GGML_TYPE_Q8_K) {
        const int64_t n_elements = ggml_nelements(src1);
        q8k_fp32_alloc.alloc(ctx.pool(), n_elements);
        float * fp32_buf = q8k_fp32_alloc.get();

        const int64_t n_q8k_blocks = n_elements / QK_Q8K;
        tqp_dequantize_q8k_kernel<<<(unsigned int)n_q8k_blocks, QK_Q8K, 0, stream>>>(
            (const block_q8k_cuda *)src1->data, fp32_buf, n_q8k_blocks);
        CUDA_CHECK(cudaGetLastError());

        src1_d = fp32_buf;
        // After dequantize, data is contiguous: stride = d per column.
        q_s11 = (int64_t)d;
        q_s12 = ne11 * (int64_t)d;
        q_s13 = ne11 * ne12 * (int64_t)d;
    } else {
        GGML_ASSERT(nb10 == (int64_t)sizeof(float));
        src1_d = (const float *)src1->data;
        q_s11 = nb11 / (int64_t)sizeof(float);
        q_s12 = nb12 / (int64_t)sizeof(float);
        q_s13 = nb13 / (int64_t)sizeof(float);
    }

    // K-cache tensors are homogeneously one (layer, rotation) pair's worth
    // of blocks; read the packed layer_byte from the first block via a
    // 1-byte synchronous d2h copy and reuse it across all prepare_query
    // and vec_dot launches in this op.
    uint8_t layer_byte = 0;
    {
        const uint8_t * first_block = (const uint8_t *)src0->data;
        const size_t layer_byte_offset = offsetof(block_tq4p_d128, layer_idx);
        CUDA_CHECK(cudaMemcpyAsync(&layer_byte, first_block + layer_byte_offset, 1,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (d == QK_TQ4P_D128) {
        ggml_cuda_tqp_prepare_query_batch_d128(src1_d, Sq, q_rot, ne11, ne12, ne13, q_s11, q_s12, q_s13, layer_byte, stream);
    } else {
        ggml_cuda_tqp_prepare_query_batch_d256(src1_d, Sq, q_rot, ne11, ne12, ne13, q_s11, q_s12, q_s13, layer_byte, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    const size_t ts_src0 = ggml_type_size(src0->type);
    const int64_t s01 = nb01 / (int64_t)ts_src0;
    const int64_t s02 = nb02 / (int64_t)ts_src0;
    const int64_t s03 = nb03 / (int64_t)ts_src0;
    const int64_t d_s1 = nb1 / (int64_t)sizeof(float);
    const int64_t d_s2 = nb2 / (int64_t)sizeof(float);
    const int64_t d_s3 = nb3 / (int64_t)sizeof(float);
    const int64_t channel_ratio = ne2 / ne02;
    const int64_t sample_ratio = ne3 / ne03;

    const dim3 grid((unsigned int)ne01, (unsigned int)ne11, (unsigned int)(ne2 * ne3));
    float * dst_d = (float *)dst->data;

    if (d == QK_TQ4P_D128) {
        tqp_vec_dot_ggml_kernel<QK_TQ4P_D128, block_tq4p_d128><<<grid, 32, 0, stream>>>(
            (const block_tq4p_d128 *)src0->data, Sq, q_rot, dst_d,
            ne11, ne2, s01, s02, s03, d_s1, d_s2, d_s3,
            channel_ratio, sample_ratio, c_tqp_centroids_d128);
    } else {
        tqp_vec_dot_ggml_kernel<QK_TQ4P_D256, block_tq4p_d256><<<grid, 64, 0, stream>>>(
            (const block_tq4p_d256 *)src0->data, Sq, q_rot, dst_d,
            ne11, ne2, s01, s02, s03, d_s1, d_s2, d_s3,
            channel_ratio, sample_ratio, c_tqp_centroids_d256);
    }
}
#endif
