# Byte layout spec — `TQ4P_D128` and `TQ4P_D256`

All multi-byte fields are little-endian. fp16 is IEEE 754 half-precision
(round-to-nearest-even), matching `ggml_fp32_to_fp16` and
`struct.pack('<e', ...)` in Python.

## `block_tq4p_d128` (69 bytes per 128 elements = 4.3125 bpw)

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 2 | `orig_norm` (fp16) | `‖x‖` before unit normalization |
| 2 | 2 | `res_d` (fp16) | `‖x_unit − Πᵀ · centroids[idx]‖` |
| 4 | 1 | `layer_idx` (uint8) | Packed (rotation << 7) \| (layer & 0x1f); selects σ_i/Π_i/S_i and rotation mode |
| 5 | 48 | `qs[48]` | 128 × 3-bit Lloyd-Max indices, bitplane-packed |
| 53 | 16 | `qjl_signs[16]` | 128 × 1-bit sign bits (1 = negative) |

## `block_tq4p_d256` (133 bytes per 256 elements = 4.15625 bpw)

| Offset | Size | Field |
|---|---|---|
| 0 | 2 | `orig_norm` (fp16) |
| 2 | 2 | `res_d` (fp16) |
| 4 | 1 | `layer_idx` (uint8) | Packed (rotation << 7) \| (layer & 0x1f) |
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

## Packed layer / rotation byte

Each block stores a packed byte at offset 4:

```
bit 7        = rotation (0 = TQP_ROT_WHT, 1 = TQP_ROT_HAAR)
bits 5..6    = reserved (must be 0)
bits 0..4    = layer_idx (0..31)
```

Use `TQP_LAYER_BYTE(layer, rotation)` / `TQP_EXTRACT_LAYER` /
`TQP_EXTRACT_ROT` from `ggml-tq-paper.h` to build and unpack it.

## Per-layer constants

Every layer has three pre-generated constants, all selected via the low
5 bits of the packed byte:

- **σ_i** (WHT sign vector): `(2 * torch.randint(0, 2, (d,), gen(seed=42+i)) - 1).float()`.
- **Π_i** (Haar rotation): `generate_rotation_matrix(d, seed=42+i)`
  (dense d×d, matches `turboquant.py::TurboQuantProd(seed=42+i)`).
- **S_i** (Gaussian QJL): `generate_qjl_matrix(d, m=d, seed=43+i)`.

The kernel uses σ_i when bit 7 is clear (WHT) and Π_i when it's set (Haar).
S_i is used in both modes (QJL operates on the residual in original space,
not rotated space). Centroids and boundaries are shared across all layers
and rotations — they depend only on the post-rotation marginal, which
for d ≥ 64 is the same Gaussian-approximated distribution under Haar and
RHT (identical on uniform unit vectors by sphere symmetry).

32 layers are precomputed (layer_idx in [0, 31]). The layer index and
rotation mode are passed explicitly at quantize time; they are not
derived from tensor naming.

## Reference data

Per-layer data lives in the C headers as:
- `TQP_SIGMA_D{D}[32][D]`: ±1 fp32 sign vector per layer (for WHT).
- `TQP_PI_D{D}[32][D*D]`: fp32 Haar rotation matrix per layer, row-major.
- `TQP_S_D{D}[32][D*D]`: fp32 QJL matrix per layer, row-major.
