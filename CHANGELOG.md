# TurboQuant — Changelog

## 2026-04-24: Restore vLLM plugin with configurable bit-widths

Re-introduced the vLLM attention backend plugin (`vllm_plugin/`), which was
removed on 2026-04-12 to focus on Ollama. The plugin is now restored from
git history with the following enhancements:

### Added

- Configurable MSE bit-width: `TQ_B_MSE` supports 2, 3, or 4 (was hardcoded to 2)
- Head dimension support: d64, d128, d256 (was hardcoded to d128)
- Generic bitplane packing for 3-bit and 5-bit value indices
- Config validation for `b_mse ∈ {2,3,4}` and `head_dim ∈ {64,128,256}`
- `tests/test_vllm_plugin.py`: 57 tests covering config, packing, layout, KV spec

### Preserved

- Backward-compatible wire format: d128/b_mse=2 still produces 118-byte blocks
- Triton decode kernel works unchanged for b_mse=2 (auto-fallback to torch for b_mse=3,4)
- All existing infrastructure (`gpu-models` presets, env files, profile switching) works as-is

### Restored

- `vllm_plugin/` — 10 Python modules (attention, config, platform, triton kernels, etc.)
- `setup.py` — vLLM entry point registration (`vllm.general_plugins`)
- `tests/test_triton_kernels.py` — Triton kernel tests

## 2026-04-14: Fix garbled output with TQ4P KV cache + flash attention

Flash attention with TQ4P KV cache produced completely garbled output
(random Unicode tokens from token 1). Three independent bugs found and
fixed, plus infrastructure hardening.

### Bug 1: `static __constant__` invisible across CUDA translation units

**Symptom**: Dequantize kernels read all-zero constants (sigma, centroids).
**Root cause**: CUDA `static __constant__` variables are per-TU in
whole-program compilation. `cudaMemcpyToSymbol` from the quantize TU
wrote to a different symbol than the dequant kernel read from. Confirmed
via probe kernel: `sigma[0][0]=0.0`, expected `-1.0`.
**Fix**: Moved sigma, centroids, and boundaries from `static __constant__`
to device global memory in the shared `TqpDeviceState` struct (same
pattern as Π and S matrices). All 6 CUDA files updated to pass constants
as kernel arguments.

### Bug 2: SET_ROWS only wrote first block in multi-head rows

**Symptom**: With GQA models (qwen2.5 has `n_heads_kv=2`), the KV cache
tensor has `ne[0]=256` (2 heads × 128 head_dim). Each row contains 2
TQ4P blocks (138 bytes = 2 × 69). SET_ROWS launched one CTA with D=128
threads, writing only the first block — the second head's block stayed
zero-initialized. FA staging dequantized all blocks, producing half-valid
half-zero K/V data that scrambled attention.
**Fix**: SET_ROWS now launches one CTA per block (`n_rows × n_blocks_per_row`),
deriving the row and block-within-row from `blockIdx.x`. Each CTA
processes D=128 elements at the correct source and destination offsets.

### Bug 3: Misaligned uint16 access at odd-offset blocks

**Symptom**: `CUDA error: misaligned address` on the multi-block SET_ROWS
fix. TQ4P blocks are 69 bytes; the second block in a row starts at byte
69 (odd). Direct `uint16_t` stores to `orig_norm` at odd addresses fault
on GPU hardware.
**Fix**: Use `memcpy` for `uint16_t` reads/writes in the quantize and
dequantize kernels, which the compiler lowers to byte-level loads/stores.

### Also fixed

- **`ggml_bf16_t` type conflict**: Our header typedef (`uint16_t`) clashed
  with newer ggml's struct definition. Guarded behind `#ifndef GGML_FILE_MAGIC`.
- **FA gate on quantized KV types**: Stock ollama refuses quantized KV cache
  when `OLLAMA_FLASH_ATTENTION=0`. Removed the gate in `llm/server.go` so
  TQ4P works regardless of FA setting (though V cache still requires FA for
  the dequant path).

### Infrastructure

- Pinned `OLLAMA_REF` to `9330bb91` to prevent upstream ggml drift.
- Added `scripts/bump_ollama_ref.sh` for controlled ref updates.
- GitHub Actions CI: C-vs-Python parity (131 tests), hook anchor check.
- Branch protection: PRs required, status checks gating, no force push.
- Closed PR #25 (staging bypass — dead code, caused build contamination).

### Benchmark (RTX 4090, FA=on)

| Model | Cache | KV Memory | Gen (t/s) |
|-------|-------|-----------|-----------|
| llama3.1:8b | f16 | 16,384 MiB | 251 |
| | tq4p | 4,416 MiB (−73%) | 194 |
| qwen2.5-coder:32b | f16 | 8,192 MiB | 70 |
| | tq4p | 2,208 MiB (−73%) | 61 |
| qwen3.5:35b (5090) | f16 | 6,500 MiB (2 GPUs) | 137 |
| | tq4p | 2,900 MiB (1 GPU) | 150 (+10%) |

## 2026-04-12: Remove vLLM plugin, focus on native Ollama

Removed the vLLM attention backend plugin (`vllm_plugin/`), associated
benchmarks, and serving scripts. TurboQuant now targets Ollama exclusively
via patched ggml (the `TQ4P` KV cache types).

### Removed

- `vllm_plugin/` — vLLM attention backend (Triton kernels, platform
  registration, hybrid TQ+SDPA backend)
- `setup.py` — vLLM entry-point registration
- `validate_vllm.py` — vLLM-specific validation
- `benchmark_openai.py`, `benchmark_decode.py`, `benchmark_tq_comparison.py`,
  `benchmark_tq_results.tsv` — vLLM benchmarks
- `serve_ollama_tq.sh` — Harbor/Ollama→vLLM launcher
- `SERVING.md` — vLLM serving guide
- `OPTIMIZATION.md` — vLLM decode performance roadmap
- `tests/test_triton_kernels.py` — Triton kernel tests

### Kept

- Core algorithm: `turboquant.py`, `lloyd_max.py`, `compressors.py`
- `ollama_resolver.py` — Ollama GGUF path resolution
- `patches/stage2-qjl/` — TQ4P C/CUDA implementation for ggml
- `scripts/` — Ollama build, patch, and smoke test scripts
- `validate.py` — C-library attention fidelity validation
- `tests/test_core.py`, `tests/test_ollama_resolver.py`,
  `tests/test_stability.py`

## 2026-04 (earlier): Native Ollama TQ4P integration

Added paper-faithful TQ4P as native ggml quantization types (`tq4p_d128`,
`tq4p_d256`). `scripts/build_ollama_tq.sh` patches ollama's vendored ggml
in place — no llama.cpp fork, no vLLM dependency.

## 2026-03-28 / 2026-03-29: Initial vLLM 0.17.1 integration (removed)

*Historical record.* The initial integration registered TurboQuant as a
vLLM general plugin with custom Triton decode kernels and bit-packed
compressed KV cache storage. This path was superseded by the native Ollama
approach.
