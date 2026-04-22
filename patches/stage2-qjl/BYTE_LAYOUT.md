# Byte layout spec — `TQP_D{dim}_B{bits}` and legacy `TQ4P_*` aliases

All multi-byte fields are little-endian. `fp16` is IEEE 754 half-precision
(round-to-nearest-even), matching `ggml_fp32_to_fp16` and
`struct.pack('<e', ...)` in Python.

Legacy compatibility:

- `tq4p_d128` = `tqp_d128_b3`
- `tq4p_d256` = `tqp_d256_b3`

## Block sizes

| Type | Offset 0..1 | Offset 2..3 | Offset 4 | `qs` bytes | `qjl_signs` bytes | total |
|---|---|---|---|---:|---:|---:|
| `tqp_d128_b2` | `orig_norm` | `res_d` | `layer_idx` | 32 | 16 | 53 |
| `tqp_d128_b3` / `tq4p_d128` | `orig_norm` | `res_d` | `layer_idx` | 48 | 16 | 69 |
| `tqp_d128_b4` | `orig_norm` | `res_d` | `layer_idx` | 64 | 16 | 85 |
| `tqp_d256_b2` | `orig_norm` | `res_d` | `layer_idx` | 64 | 32 | 101 |
| `tqp_d256_b3` / `tq4p_d256` | `orig_norm` | `res_d` | `layer_idx` | 96 | 32 | 133 |
| `tqp_d256_b4` | `orig_norm` | `res_d` | `layer_idx` | 128 | 32 | 165 |

`orig_norm` stores `‖x‖` before unit normalization. `res_d` stores
`‖x_unit − Πᵀ · centroids[idx]‖`.

## Bitplane index packing

For each group of 8 consecutive coordinates, `bits` bytes are emitted:

```text
byte[plane] bit i = (idx_i >> plane) & 1
```

Examples:

- `d=128, bits=2` → 16 groups × 2 B = 32 B
- `d=128, bits=4` → 16 groups × 4 B = 64 B
- `d=256, bits=3` → 32 groups × 3 B = 96 B

This is the standard ggml bitplane pattern. Unpacking is branchless bitwise
ops; the planes are independent so SIMD / warps can broadcast each plane.

## QJL sign packing

Sign for coord `i` lives at bit `i % 8` of byte `qjl_signs[i / 8]`.

- bit set (1) → sign is negative
- bit clear (0) → sign is positive

## Packed layer / rotation byte

Each block stores a packed byte at offset 4:

```text
bit 7        = rotation (0 = TQP_ROT_WHT, 1 = TQP_ROT_HAAR)
bits 5..6    = reserved (must be 0 in stored blocks)
bits 0..4    = layer_idx (0..31)
```

Use `TQP_LAYER_BYTE(layer, rotation)` / `TQP_EXTRACT_LAYER` /
`TQP_EXTRACT_ROT` from `ggml-tq-paper.h` to build and unpack it.

## Per-layer constants

Every layer has three pre-generated constants, all selected via the low
5 bits of the packed byte:

- `σ_i` (WHT sign vector): `(2 * torch.randint(0, 2, (d,), gen(seed=42+i)) - 1).float()`
- `Π_i` (Haar rotation): `generate_rotation_matrix(d, seed=42+i)`
- `S_i` (Gaussian QJL): `generate_qjl_matrix(d, m=d, seed=43+i)`

The kernel uses `σ_i` when bit 7 is clear (WHT) and `Π_i` when it's set
(Haar). `S_i` is used in both modes because QJL operates on the residual in
original space, not rotated space.

Centroids and boundaries are shared across all layers and rotations for a
given `(d, bits)` pair and live in the generated headers as:

- `TQP_CENTROIDS_D{D}_B{B}[2^B]`
- `TQP_BOUNDARIES_D{D}_B{B}[2^B - 1]`

The legacy headers `tqp_centroids_d{D}.h` alias the B3 symbols to preserve
the original include names.
