# TurboQuant — Changelog

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
