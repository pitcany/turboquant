# CUDA kernel plan — follow-up commit

Scope: port `ggml-tq-paper.c` to CUDA for sm_89 (RTX 4090) and sm_120
(RTX 5090). CPU path stays as the validation oracle and fallback.

## Files to add

```
patches/stage2-qjl/cuda/
├── tqp-quantize.cu          // quantize_row on-device (per-block parallel)
├── tqp-prepare-query.cu     // Sq = S·q once per query token
├── tqp-vec-dot.cu           // (query, K-block) → score
└── tqp-kernels.cuh          // shared helpers (bitplane unpack, fp16 conv)
```

Plus wiring:
- `cuda/ggml-cuda.cu` dispatch table entries for `GGML_TYPE_TQ4P_D128/D256`.
- `ggml/src/CMakeLists.txt` CUDA target picks up the new `.cu` files.

## Constants in `__constant__` memory

| Symbol | Size (d=128) | Size (d=256) | Fits const cache? |
|---|---|---|---|
| `tqp_pi_d{D}` | 32 KB | 128 KB | d=128 yes (Ada 64K/Blackwell 128K); d=256 spills to L2 |
| `tqp_s_d{D}` | 32 KB | 128 KB | same |
| `tqp_centroids_d{D}` | 32 B | 32 B | trivial |
| `tqp_boundaries_d{D}` | 28 B | 28 B | trivial |

d=256 spilling for Ada is tolerable — access pattern is broadcast-heavy and
L2 is well-cached. Predicted ~15% throughput loss vs. d=128; still a big
win over fp16 KV cache.

## Kernel 1: `tqp_quantize_kernel_d{D}`

**Grid:** one CTA per block to quantize. 128 or 256 threads per CTA (one
per coord).

**Steps per CTA:**
1. Load 128/256 input floats into shared memory.
2. Compute `orig_norm` via warp-level sum reduction.
3. Divide by `orig_norm` into shared memory (`x_unit`).
4. Compute `x_rot = Π · x_unit`: each thread does one output coord,
   reading `Π[i, :]` broadcasted from constant cache.
5. `idx[i] = bucketize(x_rot[i], boundaries)`: each thread computes own idx.
6. Reconstruct `x_hat_unit = Πᵀ · centroids[idx]`: each thread computes one
   output coord, `Σ_j Π[j, i] · centroids[idx[j]]` — reads `idx[]` and
   `Π[:, i]` strided, one fused multiply-add per j.
7. `residual[i] = x_unit[i] - x_hat_unit[i]`, tree-reduce `‖residual‖`.
8. `proj = S · residual`: same shape as step 4, constant cache broadcast.
9. Warp-ballot for sign packing; warp 0 writes 4 `uint32_t`s (or 8 for
   d=256) of `qjl_signs`.
10. Warp 0 writes `orig_norm`, `res_d` (fp16).
11. Bitplane-pack indices via warp shuffles: for each of 3 bit-planes,
    `__ballot_sync` collects 8 bits per 32-thread group.

**Register pressure:** mostly `float` accumulators; `int8` idx and
`float` centroid values. ≤ 48 regs per thread at d=128 — high occupancy
(64 warps/SM on Blackwell, 48 on Ada).

## Kernel 2: `tqp_prepare_query_kernel_d{D}`

**Grid:** one CTA per (layer × query token). 128/256 threads.

**Steps:**
1. Load `q` into shared memory.
2. Each thread computes `Sq[i] = Σ_j S[i, j] · q[j]`, `i = threadIdx.x`.
3. Write to a per-layer `Sq` buffer in global memory, indexed by layer ×
   token. Max d×sizeof(float) bytes per entry = 1 KB at d=256.

Called once per token per layer. Total bytes added to KV cache: 1 KB/token
at d=256, irrelevant next to the 132 B/head × N_heads of quant data.

## Kernel 3: `tqp_vec_dot_kernel_d{D}`

**Grid:** one CTA per (query, K-chunk) pair, where K-chunk covers a
cache-line-aligned span of K positions. 32 threads (one warp) per CTA
works well for d=128, 64 threads for d=256.

**Steps:**
1. Each thread loads one K block's `idx` and `signs` via bitplane unpack.
2. Warp loads `orig_norm`, `res_d` (one thread), broadcast.
3. Stage 1: `term1 = Σ_i q_rot[i] · centroids[idx[i]]`, where `q_rot`
   comes from a precomputed per-query buffer (upstream rotation kernel),
   or recomputed on-the-fly from `q` and constant-memory Π if the fork's
   attention path doesn't separate q-rotation. Warp-sum reduction.
4. Stage 2: `term2 = res_d · sqrt_pi_over_2 / d · Σ_i Sq[i] · (signs[i] ? -1 : 1)`.
   `Sq` comes from the prepare_query output buffer. Warp-sum reduction.
5. Return `orig_norm · (term1 + term2)`.

**Amortization:** `Sq` is reused across all K positions in a single
attention pass. For N cached keys, Stage 2 adds `O(d)` flops per key and
`O(d²)` setup, vs. Stage 1's `O(d)` per key — so Stage 2 is <1% overhead
at N > 100.

## Performance targets

Measured on `TQ3_0` (the fork's existing type) as baseline:

| Config | 4090 decode | 5090 decode |
|---|---|---|
| TQ3_0 (baseline) | 100% | 100% |
| TQ4P_D128 (this plan) | ~85% | ~90% |
| TQ4P_D256 (this plan) | ~70% | ~80% |

The 4090/5090 gap widens at d=256 because Blackwell's 128 KB constant
cache fits both `Π` and `S` at d=256 (128+128 KB); Ada's 64 KB cache
spills to L2.

## Build integration

`build_ollama_tq.sh --stage2-cuda` (new flag in follow-up): appends the
`.cu` sources to `ggml/src/CMakeLists.txt` and sets
`CMAKE_CUDA_ARCHITECTURES="89;120"` so both GPUs get kernels in one build.

## Explicit non-goals of the CUDA commit

- No tensor-core (wmma) usage. d=128 / d=256 matvecs are below efficient TC
  tile size; plain FMA via CUDA cores is equivalent and simpler.
- No sm_90 (Hopper) paths — wgmma/TMA are unavailable on Ada and irrelevant
  for this data size on Blackwell.
- No quantized query (int8) path — keeping `vec_dot_type = GGML_TYPE_F32`
  matches the CPU draft. Adding Q8_K query support is a second CUDA
  optimization commit.
- No multi-GPU-specific code — llama.cpp's `--tensor-split` + per-device
  constant memory replication handle this for free.

## Validation for the CUDA commit

The same byte-exact tests in `python/test_c_vs_python.py` will run against
the CUDA implementation by switching the ctypes dlopen target. Adds:
`test_cuda_vs_cpu.py` that quantizes on both and asserts byte-identical
output and inner-product agreement within fp32 accumulation noise
(<1e-4). If the CUDA path drifts from the CPU reference, the test fails
and the deviation is visible at the block level.
