# Byte layout spec — `TQ4P_D128` and `TQ4P_D256`

All multi-byte fields are little-endian. fp16 is IEEE 754 half-precision
(round-to-nearest-even), matching `ggml_fp32_to_fp16` and
`struct.pack('<e', ...)` in Python.

## `block_tq4p_d128` (69 bytes per 128 elements = 4.3125 bpw)

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | `orig_norm` (fp16) | `‖x‖` before unit normalization |
| 2 | 2 | `res_d` (fp16) | `‖x_unit − Πᵀ · centroids[idx]‖` |
| 4 | 1 | `layer_idx` (uint8) | Layer index [0, 31]; selects per-layer Π and S |
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
rotation matrix Π_i and QJL projection matrix S_i:

- Π_i: `generate_rotation_matrix(d, seed=42+layer_idx)`
- S_i: `generate_qjl_matrix(d, m=d, seed=43+layer_idx)`

32 layers are precomputed (layer_idx in [0, 31]). The layer index is passed
explicitly during quantization (not derived from tensor naming).

Centroids and boundaries are shared across all layers (they depend only on
the Gaussian post-rotation distribution, which is seed-independent).

## Reference data

- Π_i (Haar rotation): generated from `torch.linalg.qr(torch.randn(d, d,
  generator=Generator().manual_seed(42+i)))`, sign-normalized so `det(Π) = +1`.
- S_i (Gaussian JL): `torch.randn(d, d, generator=Generator().manual_seed(43+i))`.
- Centroids and boundaries: `LloydMaxCodebook(d, bits=3)` with Gaussian
  approximation (`use_exact=False`, which matches `TurboQuantProd`'s default).

All 32 per-layer Π and S matrices are stored fp32 in the C headers as 3D
arrays `TQP_PI_D{D}[32][D*D]` and `TQP_S_D{D}[32][D*D]`.

## Paper correspondence

Maps to `turboquant.py::TurboQuantProd(d, bits=4, seed=42+layer_idx)`:

| Block field | Python field |
|---|---|
| `orig_norm` | (caller-side norm in `TurboQuantCompressorV2`) |
| `res_d` | `compressed["residual_norm"]` |
| `layer_idx` | (new) passed explicitly, stored in block header |
| `qs` | `compressed["mse_indices"]` (bitplane-packed) |
| `qjl_signs` | `compressed["qjl_signs"]` (sign bit packed) |

`bits=4` in `TurboQuantProd` means `mse_bits = 3` (Stage 1) + 1 QJL bit
(Stage 2). The "4" is total per-coord budget, not Stage 1 bits.
