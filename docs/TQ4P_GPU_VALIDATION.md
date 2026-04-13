# TQ4P GPU Validation Report

Base commit: f75910c (main, post-PR #10 merge)
Hardware: NVIDIA RTX 4090 (24 GB) + RTX 5090 (32 GB)
CUDA: 12.9, nvcc V12.9.86
Python: 3.11 (miniconda3/dsml), torch 2.10.0+cu128
ollama: 0.20.0

## 1. CUDA shared library build

**Initial result:** FAIL — nvlink constant memory overflow (0x24180 / 0x10000 max).

**Root cause:** `CUDA_SEPARABLE_COMPILATION ON` in CMakeLists.txt caused nvlink
to sum `__constant__` data across all 3 translation units (3 x ~49 KB = ~148 KB > 64 KB).
No cross-TU device function calls exist — separable compilation was unnecessary.

**Fix:** Remove `CUDA_SEPARABLE_COMPILATION ON` from CMakeLists.txt.

## 2. CUDA vs CPU test suite

**Initial result:** 56 of 104 tests FAIL. All failures at layer_idx > 0.

**Root cause:** Per-layer S matrix (QJL projection) pointer was passed to CUDA
kernels without the per-layer offset `layer * D * D`. The sigma and pi offsets
were correct; only S was missing its layer stride.

Affected files and sites:
- `tqp-quantize.cu`: `tqp_quantize_kernel_d128` (line 188), `tqp_quantize_kernel_d256` (line 217)
- `tqp-prepare-query.cu`: all 4 host dispatch functions (d128, d256, batch d128, batch d256)

**Fix:** Add `+ (size_t)layer * D * D` to every S pointer passed to kernels.

**Post-fix result:** 104/104 CUDA tests pass.

| Test | Count | Status |
|------|-------|--------|
| test_byte_identical_quantize (d128/d256, 5 layers, WHT/Haar) | 20 | PASS |
| test_layer_byte_stored_in_block | 20 | PASS |
| test_cuda_per_layer_produces_different_bytes | 4 | PASS |
| test_prepare_query_agreement | 20 | PASS |
| test_vec_dot_agreement | 20 | PASS |
| test_dispatch_wrapper_matches_block_api | 20 | PASS |

## 3. CPU test suite (floor check)

226/226 pass (excluding 11 pre-existing Haar rotation matrix precision tests
that fail due to torch 2.10 vs the version used to generate the `.pt` constants;
max diff ~1e-7, purely cosmetic).

## 4. ollama integration

**Build:** `scripts/build_ollama_tq.sh --rebuild` with CUDA=1. Required manual
fix of duplicate `GGML_TYPE_TQ4P_D128` enum entries in ggml.h (prior patching
left stale entries at values 40/41; a second pass added 42/43).
`apply_hooks.sh` now strips pre-existing TQ4P entries before re-inserting.

**Smoke test:** `scripts/smoke_test_tq4p.sh` passes.
- Model: gemma4:26b-a4b-it-q4_K_M
- KvCacheType: tq4p_d128
- Output: coherent, no NaN/inf

Also fixed: smoke test `grep` for garbled output was matching ANSI terminal
escape codes from ollama's spinner — now strips ANSI before checking.

## 5. Bug summary

| # | File | Bug | Severity |
|---|------|-----|----------|
| 1 | cuda/CMakeLists.txt | CUDA_SEPARABLE_COMPILATION caused nvlink overflow | Build-blocking |
| 2 | cuda/tqp-quantize.cu | S matrix missing per-layer offset | Data corruption (layers > 0) |
| 3 | cuda/tqp-prepare-query.cu | S matrix missing per-layer offset | Data corruption (layers > 0) |
| 4 | apply_hooks.sh | Enum patch not idempotent across ollama updates | Build-blocking on rebuild |
| 5 | scripts/smoke_test_tq4p.sh | ANSI escape codes trigger garbled-output check | False positive |
