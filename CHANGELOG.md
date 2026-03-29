# TurboQuant vLLM Plugin — Changelog

## 2026-03-28 / 2026-03-29: Initial vLLM 0.17.1 Integration

### What was done

#### 1. Fixed plugin registration for vLLM 0.17.1

**Problem**: The original plugin registered as a `vllm.platform_plugins` entry point, returning a `TurboQuantPlatform` class. vLLM 0.17.1 expects platform plugins to be functions returning a class qualname string. The plugin hijacked the CUDA platform, causing crashes.

**Fix**: Changed to `vllm.general_plugins` entry point with a `register_turboquant()` function that calls `register_backend(AttentionBackendEnum.CUSTOM, ...)`.

**Files changed**: `setup.py`, `vllm_plugin/platform.py`

#### 2. Ported attention backend to vLLM v1 API

**Problem**: The original `TurboQuantAttentionBackend` and `TurboQuantAttentionImpl` implemented vLLM's v0 attention API (`vllm.attention.backends.abstract`). vLLM 0.17.1 uses v1 (`vllm.v1.attention.backend`) with different abstract classes, method signatures, and data flow.

**What changed**:
- `TurboQuantAttentionBackend` now implements v1 `AttentionBackend` ABC (`get_name`, `get_impl_cls`, `get_builder_cls`, `get_kv_cache_shape`)
- `TurboQuantAttentionImpl` implements v1 `AttentionImpl` with the correct `forward(layer, query, key, value, kv_cache, attn_metadata, output, ...)` signature
- New `TurboQuantMetadataBuilder` implements v1 `AttentionMetadataBuilder`
- New `TurboQuantMetadata` dataclass holds per-batch metadata
- Handles `None` attn_metadata during vLLM profile/warmup runs
- Handles 3D output tensor `[tokens, heads, dim]` (v1) vs 2D `[tokens, heads*dim]`

**File**: `vllm_plugin/attention.py` (complete rewrite)

#### 3. Implemented bit-packed compressed KV cache storage

**Problem**: TurboQuant's compression is only useful if the compressed data takes less memory. The original stored indices as int8/float tensors with no packing.

**What was built**:
- `_CompressedLayout` class defining the byte-level format (118 bytes per token per head)
- `_pack_2bit` / `_unpack_2bit`: Pack 2-bit MSE indices (4 values per byte)
- `_pack_4bit` / `_unpack_4bit`: Pack 3-bit value indices into 4-bit nibbles
- `_pack_1bit` / `_unpack_1bit`: Pack QJL signs (8 per byte)
- Compressed data stored directly in vLLM's `kv_cache` tensor

**Result**: 118 bytes per token per head vs 512 bytes FP16 = **4.3x compression**

#### 4. Patched vLLM's KV cache allocator for compressed page sizes

**Problem**: vLLM allocates KV cache pages based on `AttentionSpec.page_size_bytes`, which assumes full FP16 storage. Our compressed shape didn't match the flat buffer size, causing reshape errors.

**Fix**:
- Created `TurboQuantSpec` (subclass of `FullAttentionSpec`) in `vllm_plugin/kv_spec.py` with overridden `real_page_size_bytes`
- `register_turboquant()` monkey-patches `Attention.get_kv_cache_spec` to return `TurboQuantSpec`
- Registered `TurboQuantSpec` in vLLM's `spec_manager_map` so the KV cache manager recognizes it

**Result**: vLLM allocates 4.3x less memory per KV cache page → **32K context** on hardware that only supported 8K

**Files**: `vllm_plugin/kv_spec.py` (new), `vllm_plugin/platform.py`

#### 5. Fixed multi-GPU tensor parallelism (RTX 4090 + RTX 5090)

**Problem**: NCCL hung during model loading with TP=2 across mixed GPU architectures (Ada Lovelace 8.9 + Blackwell 12.0).

**Root causes found**:
- GPU P2P transfers not supported across different architectures
- The broken platform plugin was causing hangs (not NCCL itself)
- NCCL allreduce works fine with `NCCL_P2P_DISABLE=1`

**Fix**: `NCCL_P2P_DISABLE=1` + `--disable-custom-all-reduce` + fixing the plugin

#### 6. Performance optimization: eliminated Python loops

**Problem**: Original attention computation used Python for-loops over KV heads (4 iterations per layer × 80 layers = 320 loop iterations per decode step).

**Optimizations applied**:
- Batched attention score computation across all KV heads using `torch.einsum`
- Batched store (compress + write) across all KV heads in one pass
- Vectorized `_unpack_1bit` using tensor broadcasting instead of Python loop

**Result**: 2.2 tok/s → **6.9 tok/s** (3.1x speedup)

### Performance summary

| Config | Speed | Max context |
|--------|-------|-------------|
| FlashAttention (baseline) | ~20 tok/s | 8K |
| TurboQuant (before optimization) | ~2.2 tok/s | 32K |
| TurboQuant (optimized) | ~6.9 tok/s | 32K |

### Hardware tested

- GPU 0: NVIDIA GeForce RTX 4090 (24 GB, Ada Lovelace, compute 8.9)
- GPU 1: NVIDIA GeForce RTX 5090 (32 GB, Blackwell, compute 12.0)
- Model: casperhansen/llama-3.3-70b-instruct-awq (AWQ 4-bit, 70B params)
- vLLM: 0.17.1
- NCCL: 2.27.5
- Driver: 575.57.08, CUDA 12.9

### Files added/modified

```
vllm_plugin/
  attention.py    — Complete rewrite (v1 API, bit-packed storage, batched ops)
  platform.py     — Rewritten (general_plugins + KV spec patching)
  kv_spec.py      — NEW (TurboQuantSpec with compressed page sizes)
  __init__.py     — Simplified exports
  config.py       — Unchanged
  README.md       — Rewritten for v1 API

setup.py          — Entry point: vllm.platform_plugins → vllm.general_plugins
SERVING.md        — NEW (how to run guide)
CHANGELOG.md      — NEW (this file)
```
