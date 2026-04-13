# Byte layout spec — `TQ4P_D128` and `TQ4P_D256`

All multi-byte fields are little-endian. fp16 is IEEE 754 half-precision
(round-to-nearest-even), matching `ggml_fp32_to_fp16` and
`struct.pack('<e', ...)` in Python.

## `block_tq4p_d128` (69 bytes per 128 elements = 4.3125 bpw)

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | `orig_norm` (fp16) | `‖x‖` before unit normalization |
| 2 | 2 | `res_d` (fp16) | `‖x_unit − Πᵀ · centroids[idx]‖` |
| 4 | 1 | `layer_idx` (uint8) | Layer index [0, 31]; selects per-layer σ and S |
| 5 | 48 | `qs[48]` | 128 × 3-bit Lloyd-Max indices, bitplane-packed |
| 53 | 16 | `qjl_signs[16]` | 128 × 1-bit sign bits (1 = negative) |

## `block_tq4p_d256` (133 bytes per 256 elements = 4.15625 bpw)

| Offset | Size | Field |
|---|---|---|
| 0 | 2 | `orig_norm` (fp16) |
| 2 | 2 | `res_d` (fp16) |
| 4 | 1 | `layer_idx` (uint8) |
| 5 | 96 | `qs[96]` — 256 × 3-bit indices |
| 101 | 32 | `qjl_signs[32]` — 256 × 1-bit signs |

## 3-bit bitplane index packing

For each group of 8 consecutive coordinates, 3 bytes are emitted:

```
byte[0]  bit i = (idx_i >> 0) & 1       // low bits
byte[1]  bit i = (idx_i >> 1) & 1       // mid bits
byte[2]  bit i = (idx_i >> 2) & 1       // high bits
```

d=128 → 16 groups × 3 B = 48 B.
d=256 → 32 groups × 3 B = 96 B.

This is the standard ggml bitplane pattern. Unpacking is branchless bitwise
ops; the 3 planes are independent so SIMD / warp can broadcast each plane.

## QJL sign packing

Sign for coord `i` lives at bit `i % 8` of byte `qjl_signs[i / 8]`.

- bit set (1) → sign is **negative**
- bit clear (0) → sign is **positive**

## Per-layer constants

Each block stores a `layer_idx` byte at offset 4. This selects the per-layer
sign vector σ_i and QJL projection matrix S_i. The rotation Π_i applied to
`x_unit` is the Randomized Hadamard Transform Π_i = (1/√d) · H · diag(σ_i):

- σ_i: Rademacher ±1 vector, seed = 42 + layer_idx.
- S_i: `generate_qjl_matrix(d, m=d, seed=43+layer_idx)`.

32 layers are precomputed (layer_idx in [0, 31]). The layer index is passed
explicitly during quantization (not derived from tensor naming).

Centroids and boundaries are shared across all layers (they depend only on
the post-rotation distribution — which for random unit vectors is the same
Gaussian approximation under either Haar or RHT).

## Reference data

- σ_i (RHT sign vector): `(2 * torch.randint(0, 2, (d,), generator=Generator().manual_seed(42+i)) - 1).float()`.
- S_i (Gaussian JL): `torch.randn(d, d, generator=Generator().manual_seed(43+i))`.
- Centroids and boundaries: `LloydMaxCodebook(d, bits=3)` with Gaussian
  approximation.

Per-layer data lives in the C headers as:
- `TQP_SIGMA_D{D}[32][D]`: ±1 fp32 sign vector per layer.
- `TQP_S_D{D}[32][D*D]`: fp32 QJL matrix per layer, row-major.

## Branch note

This branch replaces the paper's Haar random orthogonal Π with a
Randomized Hadamard Transform (RHT): Π := (1/√d) · H · diag(σ). The apply
cost drops from O(d²) to O(d log d) and per-layer storage drops from d²
floats to d. This is orthogonal just like Haar, so the paper's MSE and
inner-product-correlation bounds still apply for random unit-vector
inputs — but the stored indices / QJL signs no longer match
`turboquant.py::TurboQuantProd` byte-for-byte.
