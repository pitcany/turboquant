# TurboQuant Optimization Roadmap

Current: **~6.9 tok/s** (vs 20 tok/s FlashAttention) = 0.35x

## Where time is spent (per decode step, 1 layer, 512 cached tokens)

| Component | Time | Notes |
|-----------|------|-------|
| Gather from block table | ~0.1 ms | Tensor indexing, fast |
| Unpack bits | ~0.4 ms | Vectorized but still multiple ops |
| Dequantize (codebook + rotation) | ~0.3 ms | Two matmuls per head group |
| Attention scores (einsum) | ~0.3 ms | Batched across heads |
| QJL correction (einsum) | ~0.2 ms | Batched across heads |
| Softmax + weighted sum | ~0.1 ms | Standard |
| **Total** | **~1.2 ms** | × 80 layers = 98 ms |

## Optimization tiers

### Tier 1: Low-hanging fruit (pure PyTorch, ~2x speedup → ~12 tok/s)

**1a. FP16 computation throughout**
Currently casting to float32 for attention math. The codebook lookup, rotation inverse, and score computation can all run in float16, halving memory bandwidth.

```python
# Before:
k_mse = self._key_q.mse.dequantize(km_idx).float()
# After:
k_mse = self._key_q.mse.dequantize(km_idx).half()
```

**1b. Precompute `R @ codebook` lookup table**
Dequantize does: `centroids[indices] @ R` (codebook lookup + rotation inverse). Since the codebook has only 4 entries (2-bit) or 8 entries (3-bit), precompute `R @ centroids` into a small LUT: `(num_levels, head_dim)`. Then dequantize is just an index gather — no matmul.

```python
# Precompute once per layer:
self._key_lut = self._key_q.mse.R @ self._key_q.mse.centroids  # (4, 128)

# Per forward:
k_mse = self._key_lut[km_idx]  # Pure gather, no matmul
```

This eliminates the most expensive operation (rotation matmul) entirely.

**1c. Precompute `S.T @ R @ codebook` for QJL scores**
The QJL term computes `q @ S.T`, then multiplies by signs. If we precompute `S.T` applied to the dequantized MSE, we can merge term1 and term2 into fewer operations.

**1d. Cache the causal mask**
During decode (q_len=1), no causal mask is needed. During prefill, the mask can be computed once and reused.

### Tier 2: Triton kernels (~3-5x speedup → ~20-35 tok/s)

**2a. Fused unpack + dequantize kernel**
Single Triton kernel that reads packed bytes from kv_cache, unpacks 2-bit indices, looks up the precomputed LUT, and outputs the dequantized vectors. Eliminates intermediate tensor allocations and kernel launch overhead.

```
Input:  kv_cache[block, pos, head, :59] (packed bytes)
Output: k_mse[S, nkh, D], k_signs[S, nkh, D], norms[S, nkh, 2]
```

**2b. Fused asymmetric attention kernel**
Triton kernel that computes the full TurboQuant asymmetric score:
```
score = (q @ k_mse.T + corr * (q @ S.T) @ signs.T * r_norm) * k_norm / sqrt(d)
```
Fusing this avoids multiple matmul kernel launches and intermediate buffers.

**2c. Fused softmax + weighted value sum**
Online softmax with streaming value accumulation (similar to FlashAttention's approach), operating on the compressed representation.

### Tier 3: Architectural changes (~matching FlashAttention)

**3a. Replace dense rotation with Hadamard transform**
The rotation matrix R is dense (128×128), making dequantize O(d²). The Walsh-Hadamard transform is O(d log d) and can be implemented as a fast recursive butterfly. The paper mentions this as "FWHT" (Fast Walsh-Hadamard Transform).

**3b. Block-level compression instead of per-token**
Compress entire blocks (e.g., 16 tokens) at once, enabling vectorized SIMD-style operations and better memory access patterns.

**3c. Paged attention integration**
Write a custom paged attention kernel that reads compressed data directly from the block table, computing attention scores on-the-fly without materializing the full dequantized KV cache. This is the "holy grail" — it would make TurboQuant as fast as FlashAttention while using 4x less memory.

**3d. CUDA graph support**
Remove `--enforce-eager` requirement by making the attention graph static (fixed tensor shapes per batch configuration). This eliminates Python interpreter overhead between layers.

## Estimated impact

| Tier | Change | Est. tok/s | vs FA |
|------|--------|-----------|-------|
| Current | Batched PyTorch | 6.9 | 0.35x |
| 1b | Precomputed LUT | ~12 | 0.6x |
| 2a+2b | Triton kernels | ~25 | 1.2x |
| 3a+3c | Hadamard + paged attn | ~30+ | 1.5x |

Tier 2 is where TurboQuant becomes competitive with FlashAttention on speed while providing 4x better memory efficiency. Tier 3 could make it *faster* than FlashAttention (less memory bandwidth due to compression).

## Quick win to try first

**Precomputed LUT (1b)** is the single highest-impact change:
- Eliminates the 128×128 rotation matmul in dequantize
- Only needs a 4×128 or 8×128 lookup table per layer
- Pure PyTorch, no Triton needed
- Expected: ~2x speedup from this alone
