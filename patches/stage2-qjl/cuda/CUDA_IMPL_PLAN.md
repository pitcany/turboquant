# CUDA Follow-up for TQ4P — Implementation Plan

> **⚠ Superseded by the WHT variant.** The kernels in this directory no
> longer implement the paper-faithful Haar Π GEMV described below. They
> implement the Randomized Hadamard Transform variant, matching the CPU
> path in `../c/ggml-tq-paper.c`. See `PLAN.md` for the updated kernel
> shapes (Kernel 1 and Kernel 2). This document is kept verbatim because
> its kernel-numbering table, launch config matrix, build-integration
> steps, and risk mitigations are still useful for the fork integration
> work that hasn't landed yet; the algorithmic pieces about Π/S matmuls
> should be read as *pre-WHT* historical design.

## Context

Port the CPU reference TQ4P quantization (`ggml-tq-paper.c`) to CUDA for RTX 4090 (sm_89, Ada Lovelace) and RTX 5090 (sm_120, Blackwell). The CPU path stays as correctness oracle and fallback. TQ4P implements Zandieh et al.'s two-stage algorithm: Stage 1 = Haar rotation Π + 3-bit Lloyd-Max quantization; Stage 2 = Gaussian JL projection S + 1-bit sign packing. Total 4.25 bpw (d=128) / 4.13 bpw (d=256).

**Branch:** `claude/arxiv-llama-ollama-integration-yKp1r`
**Repo:** `/home/yannik/Work/turboquant`

**Correctness bar:**
- Quantize: byte-identical to CPU (achievable with fp32 matrices + `__fmul_rn`/`__fadd_rn` intrinsics + sequential norms)
- vec_dot: agreement within <1e-4 absolute vs CPU
- If byte-identical quantize fails (FMA/rounding), loosen to within-1-ULP on fp16 fields, bit-exact on qs/signs

---

## Reference files (read-only)

| File | Purpose |
|------|---------|
| `patches/stage2-qjl/c/ggml-tq-paper.c` | CPU reference (354 lines) — the oracle |
| `patches/stage2-qjl/c/ggml-tq-paper.h` | Block structs, public API |
| `patches/stage2-qjl/c/tqp_constants_d{128,256}.h` | fp32 Π, S matrices (generated) |
| `patches/stage2-qjl/c/tqp_centroids_d{128,256}.h` | Lloyd-Max codebook + boundaries |
| `patches/stage2-qjl/python/test_c_vs_python.py` | Test pattern to mirror |
| `patches/stage2-qjl/python/tq_paper_reference.py` | Python oracle |
| `patches/stage2-qjl/BYTE_LAYOUT.md` | Block format spec |
| `patches/stage2-qjl/cuda/PLAN.md` | Design spec (kernel shapes, perf targets) |
| `patches/stage2-qjl/apply_hooks.sh` | CPU hook script (pattern to extend) |
| `patches/stage2-qjl/hooks.md` | Manual hook documentation |
| `scripts/build_ollama_tq.sh` | Build orchestrator |

---

## Deliverables — 7 new files + 3 modified files

### New files under `patches/stage2-qjl/cuda/`

#### 1. `tqp-constants-cuda.cuh`

Declare and initialize per-layer Π, S, centroids, boundaries for CUDA.

**Per-layer constants:** 32 precomputed Π_i and S_i pairs per head_dim, indexed by `layer_idx` stored in each block header.

**Memory layout decision:**
- **`__constant__`**: centroids (32B) + boundaries (28B) per dim. Total ~120B. Broadcast-friendly, tiny. Shared across all layers.
- **Device global via `cudaMalloc`**: Per-layer Π and S as **fp32** in 3D arrays `[32][d*d]`. Accessed via `const float * __restrict__` + `__ldg()`.
  - fp32 (not fp16) to match CPU for byte-identical quantize
  - d=128: 32×64KB + 32×64KB = 4MB → fits L2 easily (72MB on 4090)
  - d=256: 32×256KB + 32×256KB = 16MB → fits L2
  - Kernel indexes as `d_tqp_pi_dN + (size_t)layer_idx * d * d`
  - Only the layer being quantized/queried is accessed per kernel launch, so effective bandwidth is same as before (one d×d matrix per launch)

**Why not `__constant__` for matrices:** The 64KB hardware limit for `__constant__` memory. Even one d=128 Π is 64KB in fp32; 32 layers × 2 matrices is impossible.

**Contents:**
```cuda
__constant__ float c_tqp_centroids_d128[8];
__constant__ float c_tqp_boundaries_d128[7];
__constant__ float c_tqp_centroids_d256[8];
__constant__ float c_tqp_boundaries_d256[7];

// Device pointers — initialized by tqp_cuda_init()
// Per-layer: [TQP_MAX_LAYERS * d * d] contiguous, indexed by layer_idx * d * d
extern float * d_tqp_pi_d128;   // [32 * 128*128] 2 MB
extern float * d_tqp_s_d128;    // [32 * 128*128] 2 MB
extern float * d_tqp_pi_d256;   // [32 * 256*256] 8 MB
extern float * d_tqp_s_d256;    // [32 * 256*256] 8 MB

void tqp_cuda_init(int head_dim);   // cudaMemcpyToSymbol + cudaMalloc + cudaMemcpy (all 32 layers)
void tqp_cuda_cleanup();
```

`tqp_cuda_init(d)` initializes only the requested head_dim. Idempotent (checks a static bool). Called lazily on first TQ4P dispatch.

Also include: `__device__ uint16_t tqp_fp32_to_fp16_device(float)` — exact port of the CPU function for byte-identical fp16 conversion. And `__device__ float tqp_fp16_to_fp32_device(uint16_t)`.

#### 2. `tqp-kernels.cuh`

Shared `__device__` helpers used by all kernel files:

- `tqp_fp32_to_fp16_device(float)` / `tqp_fp16_to_fp32_device(uint16_t)` — portable, matches CPU
- `tqp_bucketize_d3(float x, const float * bounds)` — 7-comparison branchless, same as CPU
- Bitplane unpack: `tqp_unpack_indices_warp(const uint8_t * qs, int lane_id, int warp_id, uint8_t * idx_out, int elems_per_thread)` — each thread gets its indices
- Warp-shuffle reduction: `tqp_warp_reduce_sum(float val)` — `__shfl_xor_sync` in 5 steps
- Block struct definitions: include `ggml-tq-paper.h` via relative path, or redefine for standalone CUDA compilation
- Constants: `static constexpr float SQRT_PI_OVER_2 = 1.2533141373155001f;`

#### 3. `tqp-quantize.cu` — compile with `--fmad=false` OR use intrinsics

One CTA per block to quantize. Target: byte-identical output to CPU.

**Kernel `tqp_quantize_kernel_d128`** — 128 threads, one per coordinate:

```
Shared memory (aliased, ~640B):
  float smem_vec[128]   // reused for x_unit, residual
  uint8_t smem_idx[128] // indices

Input: layer_idx passed as kernel argument, used to offset into per-layer arrays:
  const float * pi = d_tqp_pi_d128 + (size_t)layer_idx * 128 * 128;
  const float * s  = d_tqp_s_d128  + (size_t)layer_idx * 128 * 128;

Steps:
 1. Each thread: smem_vec[tid] = x[block_offset + tid]
 2. __syncthreads()
 3. Thread 0: sequential sq = Σ smem_vec[i]² → orig_norm = sqrtf(sq)
    (Sequential matches CPU accumulation order)
 4. __syncthreads(), all threads: inv_norm = 1.0f / orig_norm
 5. All threads: smem_vec[tid] *= inv_norm  (now x_unit)
 6. __syncthreads()
 7. Each thread: x_rot = Σⱼ __ldg(&pi[tid*128+j]) * smem_vec[j]
    Use __fadd_rn/__fmul_rn for bit-exact accumulation:
      float acc = 0.0f;
      #pragma unroll 1
      for (int j = 0; j < 128; ++j)
          acc = __fadd_rn(acc, __fmul_rn(__ldg(&pi[tid*128+j]), smem_vec[j]));
 8. Each thread: smem_idx[tid] = bucketize(x_rot, boundaries)
 9. __syncthreads()
10. Each thread: x_hat = Σᵢ __ldg(&pi[i*128+tid]) * centroids[smem_idx[i]]
    (Πᵀ · centroids[idx], same accumulation order as CPU)
    Use __fadd_rn/__fmul_rn.
11. Each thread: residual = smem_vec[tid] - x_hat → smem_vec[tid] = residual
12. __syncthreads()
13. Thread 0: sequential r_sq = Σ smem_vec[i]² → res_d = sqrtf(r_sq)
14. __syncthreads()
15. Each thread: proj = Σⱼ __ldg(&s[tid*128+j]) * smem_vec[j]
    Use __fadd_rn/__fmul_rn.
16. Bitplane index packing via __ballot_sync:
    Per warp (4 warps, 32 threads each):
      ballot_lo  = __ballot_sync(0xFFFFFFFF, idx & 1)
      ballot_mid = __ballot_sync(0xFFFFFFFF, (idx >> 1) & 1)
      ballot_hi  = __ballot_sync(0xFFFFFFFF, (idx >> 2) & 1)
    Lane 0 writes 4 groups × 3 bytes = 12 bytes per warp → 48B total
17. Sign packing via __ballot_sync:
    sign_mask = __ballot_sync(0xFFFFFFFF, proj < 0.0f)
    Lane 0 writes 4 bytes per warp → 16B total
18. Thread 0 writes block: tqp_fp32_to_fp16_device(orig_norm),
    tqp_fp32_to_fp16_device(res_d), layer_idx, qs[48], qjl_signs[16]
```

**d=256 variant:** 256 threads, adjust all sizes (96B qs, 32B signs, 8 warps).

**Host wrapper:**
```c
void ggml_cuda_tqp_quantize_row_d128(const float * x, void * y, int64_t k, cudaStream_t stream);
void ggml_cuda_tqp_quantize_row_d256(const float * x, void * y, int64_t k, cudaStream_t stream);
```

**FMA strategy:** Use `__fadd_rn()` + `__fmul_rn()` intrinsics on all accumulation loops in the quantize kernel. This prevents FMA fusion at instruction level without affecting the entire .cu file's compilation. Combined with `#pragma unroll 1` on inner loops, this ensures identical accumulation order to CPU.

#### 4. `tqp-prepare-query.cu`

One CTA per query. Computes **both** `Sq = S · q` and `q_rot = Π · q` (fused into one kernel to avoid a second launch).

**Kernel `tqp_prepare_query_kernel_d128`** — 128 threads:

```
Shared memory: float q_smem[128] (512B)

1. Each thread: q_smem[tid] = q[tid]
2. __syncthreads()
3. Each thread: Sq[tid] = Σⱼ __ldg(&s[tid*128+j]) * q_smem[j]
4. Each thread: q_rot[tid] = Σⱼ __ldg(&pi[tid*128+j]) * q_smem[j]
5. Write Sq[tid] and q_rot[tid] to output buffers
```

FMA enabled (default) — exact match not required, feeds into vec_dot with <1e-4 tolerance.

**Host wrapper:**
```c
void ggml_cuda_tqp_prepare_query_d128(const float * q, float * Sq, float * q_rot, cudaStream_t stream);
void ggml_cuda_tqp_prepare_query_d256(const float * q, float * Sq, float * q_rot, cudaStream_t stream);
```

#### 5. `tqp-vec-dot.cu` — HOT PATH

One K-block per CTA, 32 threads (1 warp) for d=128, 64 threads (2 warps) for d=256.

**Kernel `tqp_vec_dot_kernel_d128`** — 32 threads:

```
Thread-to-element mapping: thread tid handles elements [4*tid .. 4*tid+3]
Registers per thread: ~17 (4 q_rot, 4 Sq, accumulators, temporaries)
Shared memory: ~128B (block qs + signs, loaded cooperatively)

1. Load q_rot[4*tid..4*tid+3] and Sq[4*tid..4*tid+3] from global memory
   (precomputed by prepare_query, cached in L1 across K-blocks for same query)
2. Load block header: one thread reads orig_norm + res_d (4 bytes), broadcast via __shfl_sync
   orig_norm_f = tqp_fp16_to_fp32_device(orig_norm)
   res_d_f = tqp_fp16_to_fp32_device(res_d)
3. Load qs[48] cooperatively into shared memory (32 threads × ~2B each)
4. Unpack 4 × 3-bit indices per thread from bitplane format:
   group = 4*tid / 8; bit_offset = (4*tid) % 8
   Extract from smem_qs[group*3+0..2]
5. Stage 1: 
   float t1 = 0.0f;
   for (int e = 0; e < 4; ++e)
       t1 += q_rot_reg[e] * c_tqp_centroids_d128[idx[e]];
   t1 = tqp_warp_reduce_sum(t1);  // __shfl_xor_sync, 5 steps
   t1 *= orig_norm_f;
6. Load qjl_signs[16] cooperatively
7. Unpack 4 sign bits per thread:
   sign_byte = smem_signs[4*tid / 8]
   For each of 4 elements: sign_val = (bit set ? -1.0f : 1.0f)
8. Stage 2:
   float t2 = 0.0f;
   for (int e = 0; e < 4; ++e)
       t2 += Sq_reg[e] * sign_val[e];
   t2 = tqp_warp_reduce_sum(t2);
   t2 *= orig_norm_f * res_d_f * (SQRT_PI_OVER_2 / 128.0f);
9. Lane 0: output[block_idx] = t1 + t2;
```

**Amortization:** q_rot and Sq are computed once per (query, layer) by prepare_query. For N cached keys, the setup is O(d²) and per-key work is O(d). At context length N > 100, Stage-2 overhead is <1% of total.

**d=256 variant:** 64 threads (2 warps), 4 elements per thread. Cross-warp reduction: lane 0 of each warp writes partial sum to shared memory, then lane 0 of warp 0 sums them.

**Host wrapper — `ggml_cuda_op_tqp_vec_dot`:**
This is registered in ggml-cuda.cu's dispatch. It orchestrates the full attention computation:
1. Allocate scratch: `Sq` (d floats) + `q_rot` (d floats) per query
2. Launch `tqp_prepare_query_kernel` once per query
3. Launch `tqp_vec_dot_kernel` with grid = (num_k_blocks, num_heads, batch_size)
4. Read results

```c
void ggml_cuda_op_tqp_vec_dot(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

#### 6. `CMakeLists.txt` — standalone test build

For building `libggml_tq_paper_cuda.so` independently of ollama (used by test_cuda_vs_cpu.py):

```cmake
cmake_minimum_required(VERSION 3.18)
project(tqp_cuda CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES "89;120")

add_library(ggml_tq_paper_cuda SHARED
    tqp-quantize.cu
    tqp-prepare-query.cu
    tqp-vec-dot.cu
)
target_include_directories(ggml_tq_paper_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../c
)
```

Also exports C-callable wrappers for ctypes: `tqp_cuda_quantize_row_d128(float* x_host, void* y_host, int64_t k)` — handles host↔device copies internally for test convenience.

### New file under `patches/stage2-qjl/python/`

#### 7. `test_cuda_vs_cpu.py`

Mirrors `test_c_vs_python.py` structure. Loads both `libggml_tq_paper.so` (CPU) and `libggml_tq_paper_cuda.so` (CUDA).

**Tests:**
1. `test_byte_identical_quantize(d, vectors)` — CUDA quantize output byte-identical to CPU. If fails, diagnose: check fp16 fields (orig_norm, res_d) for within-1-ULP, check qs/signs for bit-exact.
2. `test_vec_dot_agreement(d, vectors)` — CUDA vec_dot vs CPU, <1e-4 absolute max diff across 25×25 query×key pairs.
3. `test_prepare_query_agreement(d, vectors)` — CUDA Sq vs CPU Sq, <1e-5 max diff.
4. `test_dispatch_wrapper_matches_block_api(d, vectors)` — end-to-end regression (same as CPU test).

Same fixtures: seed 54321, 50 normalized vectors, d ∈ {128, 256}.

---

### Modified files

#### 8. `scripts/build_ollama_tq.sh` — add CUDA kernel copying

After existing C file copy block (line ~113), add:

```bash
if [[ "$CUDA" = "1" ]]; then
    GGML_CUDA="$GGML/src/ggml-cuda"
    echo "[+] copying TQ4P CUDA kernels into $GGML_CUDA/"
    for f in tqp-quantize.cu tqp-prepare-query.cu tqp-vec-dot.cu \
             tqp-kernels.cuh tqp-constants-cuda.cuh; do
        cp "$STAGE2_DIR/cuda/$f" "$GGML_CUDA/"
    done
    # CUDA kernels need the constant headers too
    for f in tqp_constants_d128.h tqp_constants_d256.h \
             tqp_centroids_d128.h tqp_centroids_d256.h \
             ggml-tq-paper.h; do
        cp "$STAGE2_DIR/c/$f" "$GGML_CUDA/"
    done
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_CUDA_ARCHITECTURES=89;120"
fi
```

The `.cu` files are auto-discovered by the existing `file(GLOB GGML_SOURCES_CUDA "*.cu")` in ggml-cuda's CMakeLists.txt.

#### 9. `patches/stage2-qjl/apply_hooks.sh` — add CUDA dispatch hook

Add hook 5 (after existing hooks 1-4). Only runs when CUDA .cu files are present in ggml-cuda/:

```bash
# ---------- 5. ggml-cuda.cu dispatch (CUDA only) ----------
CUDA_CU="$GGML/src/ggml-cuda/ggml-cuda.cu"
if [[ -f "$CUDA_CU" && -f "$GGML/src/ggml-cuda/tqp-vec-dot.cu" ]]; then
    if grep -q "tqp" "$CUDA_CU" 2>/dev/null; then
        echo "[=] ggml-cuda.cu already patched"
    else
        echo "[+] ggml-cuda.cu: TQ4P CUDA dispatch"
        # Python script inserts:
        # 1. Forward declaration of ggml_cuda_op_tqp_vec_dot
        # 2. Early-return case in ggml_cuda_mul_mat for TQ4P types
    fi
fi
```

The dispatch hook inserts in `ggml_cuda_mul_mat()`:
```c
// TQ4P custom dispatch — before standard type checks
if (src0->type == GGML_TYPE_TQ4P_D128 || src0->type == GGML_TYPE_TQ4P_D256) {
    ggml_cuda_op_tqp_vec_dot(ctx, dst);
    return;
}
```

#### 10. `patches/stage2-qjl/hooks.md` — document CUDA hooks

Add section documenting hook 5 (the CUDA dispatch registration) for manual application if the script fails.

---

## Kernel launch configurations

| Kernel | Grid | Block | Shared Mem | Registers/thread |
|--------|------|-------|-----------|-----------------|
| `tqp_quantize_d128` | `(k/128, 1, 1)` | `(128, 1, 1)` | 640B | ~24 |
| `tqp_quantize_d256` | `(k/256, 1, 1)` | `(256, 1, 1)` | 1280B | ~24 |
| `tqp_prepare_query_d128` | `(1, 1, 1)` per query | `(128, 1, 1)` | 512B | ~20 |
| `tqp_prepare_query_d256` | `(1, 1, 1)` per query | `(256, 1, 1)` | 1024B | ~20 |
| `tqp_vec_dot_d128` | `(n_k_blocks, n_heads, batch)` | `(32, 1, 1)` | 128B | ~17 |
| `tqp_vec_dot_d256` | `(n_k_blocks, n_heads, batch)` | `(64, 1, 1)` | 256B | ~17 |

All kernels compile for sm_89 and sm_120 via `CMAKE_CUDA_ARCHITECTURES="89;120"`.

---

## Implementation sequence

1. **`tqp-kernels.cuh`** + **`tqp-constants-cuda.cuh`** — foundations, all other files depend on these
2. **`tqp-quantize.cu`** — port quantize; build standalone test lib; run byte-identical test
3. **`tqp-prepare-query.cu`** — port prepare_query + q_rot
4. **`tqp-vec-dot.cu`** — port vec_dot (hot path); run agreement test
5. **`CMakeLists.txt`** + **`test_cuda_vs_cpu.py`** — standalone build + test harness
6. **Build integration** — modify `build_ollama_tq.sh`, extend `apply_hooks.sh`, update `hooks.md`
7. **Build ollama** with `CUDA=1`, run validate.py on GPU, benchmark

---

## Verification

1. **Unit test:** `pytest test_cuda_vs_cpu.py -v` — byte-identical quantize + <1e-4 vec_dot
2. **Ollama build:** `CUDA=1 scripts/build_ollama_tq.sh` — must compile clean
3. **Integration:** `OLLAMA_KV_CACHE_TYPE=tq4p_d128 python validate.py` — cosine sim matches CPU path
4. **Benchmark:** `llama-bench` with TQ4P on GPU — target ≥5× vs CPU (d=128, 4090), ≥0.85× vs Q4_0 KV cache

---

## Risk mitigations

| Risk | Mitigation |
|------|-----------|
| FMA fusion breaks byte-identical quantize | Use `__fadd_rn`/`__fmul_rn` intrinsics + `#pragma unroll 1` |
| sqrtf differs CPU vs GPU | Both are IEEE 754 correctly-rounded; if diverges, use software sqrtf matching CPU |
| Norm computation order | Thread 0 computes sequentially, matching CPU loop order |
| d=256 __constant__ overflow | Matrices in global memory with `__ldg()`, only centroids/boundaries in `__constant__` |
| Cross-warp reduction for d=256 vec_dot | Use 2-warp design: partial sums via shared memory, single `__syncthreads()` |

---

## Explicit non-goals (per task spec)

- No tensor cores (wmma) — d=128/d=256 below efficient TC tile size
- No Hopper-only features (wgmma, TMA)
- No Q8_K query path — vec_dot_type stays GGML_TYPE_F32
- No changes to CPU fallback path

## Per-layer constants (implemented)

Per-layer Π_i and S_i are now the default. Each block's `layer_idx` header byte
(offset 4) selects the rotation/projection pair at quant and dot-product time.
The CUDA kernels read `layer_idx` from the block header and index into the 3D
device arrays. Block size is now 69B (d=128) / 133B (d=256).
