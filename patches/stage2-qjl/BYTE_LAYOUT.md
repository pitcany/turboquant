# Byte layout spec — `TQ4P_D128` and `TQ4P_D256`

All multi-byte fields are little-endian. fp16 is IEEE 754 half-precision
(round-to-nearest-even), matching `ggml_fp32_to_fp16` and
`struct.pack('<e', ...)` in Python.

## `block_tq4p_d128` (68 bytes per 128 elements = 4.25 bpw)

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | `orig_norm` (fp16) | `‖x‖` before unit normalization |
| 2 | 2 | `res_d` (fp16) | `‖x_unit − Πᵀ · centroids[idx]‖` |
| 4 | 48 | `qs[48]` | 128 × 3-bit Lloyd-Max indices, bitplane-packed |
| 52 | 16 | `qjl_signs[16]` | 128 × 1-bit sign bits (1 = negative) |

## `block_tq4p_d256` (132 bytes per 256 elements = 4.125 bpw)

| Offset | Size | Field |
|---|---|---|
| 0 | 2 | `orig_norm` (fp16) |
| 2 | 2 | `res_d` (fp16) |
| 4 | 96 | `qs[96]` — 256 × 3-bit indices |
| 100 | 32 | `qjl_signs[32]` — 256 × 1-bit signs |

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

## Reference data

- Π (Haar rotation): generated from `torch.linalg.qr(torch.randn(d, d,
  generator=Generator().manual_seed(42)))`, sign-normalized so `det(Π) = +1`.
- S (Gaussian JL): `torch.randn(d, d, generator=Generator().manual_seed(43))`.
- Centroids and boundaries: `LloydMaxCodebook(d, bits=3)` with Gaussian
  approximation (`use_exact=False`, which matches `TurboQuantProd`'s default).

Both matrices are fp32 in the C header, shared across all layers and all
blocks. Byte-exact agreement with the Python reference holds at fp32;
switching Π / S to fp16 would halve constant memory at the cost of the
byte-exact test (would loosen to ~1e-3 relative).

## Paper correspondence

Maps to `turboquant.py::TurboQuantProd(d, bits=4, seed=42)` as follows:

| Block field | Python field |
|---|---|
| `orig_norm` | (caller-side norm in `TurboQuantCompressorV2`) |
| `res_d` | `compressed["residual_norm"]` |
| `qs` | `compressed["mse_indices"]` (bitplane-packed) |
| `qjl_signs` | `compressed["qjl_signs"]` (sign bit packed) |

`bits=4` in `TurboQuantProd` means `mse_bits = 3` (Stage 1) + 1 QJL bit
(Stage 2). The "4" is total per-coord budget, not Stage 1 bits.
