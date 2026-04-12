# ggml-side hooks — exact edits `apply_hooks.sh` makes

`scripts/build_ollama_tq.sh` copies the TQ4P C files into ollama's ggml
tree (`ml/backend/ggml/ggml/src/`) and then runs
`patches/stage2-qjl/apply_hooks.sh` to make these four additive edits.
This file documents them, so if the auto-apply fails you can apply by hand.

Anchors below are based on ollama's vendored ggml at the time of writing.
`apply_hooks.sh` uses value-based anchors (it reads `GGML_TYPE_COUNT = N`
dynamically) so small upstream drift doesn't break it.

Line numbers below are from the fork's `main` branch at the time of
authoring. The script uses context-based anchors (not line numbers) so it
survives small drift.

## Summary of edits

| File | Kind | What |
|---|---|---|
| `ggml/include/ggml.h` | additive | 2 enum values, bump `GGML_TYPE_COUNT` |
| `ggml/src/ggml.c` | additive | 2 entries in backend-agnostic `type_traits` table |
| `ggml/src/ggml-cpu/ggml-cpu.c` | additive | 2 entries in `type_traits_cpu` dispatch table |
| `ggml/src/CMakeLists.txt` | additive | 1 source file added |

`ggml-common.h` is **not** touched — our block structs live in
`ggml-tq-paper.h`, which `ggml.c` `#include`s so `sizeof(block_tq4p_d128)`
resolves.

## 1. `ggml/include/ggml.h` — enum values

Find:

```c
        GGML_TYPE_TQ3_4S  = 46, // TurboQuant 3-bit with four u8 per-8 scales (4.0 bpw)
        GGML_TYPE_COUNT   = 47,
```

Replace with:

```c
        GGML_TYPE_TQ3_4S  = 46, // TurboQuant 3-bit with four u8 per-8 scales (4.0 bpw)

        // TurboQuant paper-faithful (Stage 1 Lloyd-Max + Stage 2 QJL).
        // See ggml/src/ggml-tq-paper.h.
        GGML_TYPE_TQ4P_D128 = 47,
        GGML_TYPE_TQ4P_D256 = 48,

        GGML_TYPE_COUNT   = 49,
```

If other types have been added upstream and 47/48 are now taken, bump to
the next free values and update steps 2 and 3 accordingly.

## 2. `ggml/src/ggml.c` — backend-agnostic `type_traits` table

At the top of the file, after the existing includes, add:

```c
#include "ggml-tq-paper.h"
```

Find the `TQ3_4S` entry in the `type_traits` array (around line 911-918):

```c
    [GGML_TYPE_TQ3_4S] = {
        .type_name                = "tq3_4s",
        .blck_size                = QK_TQ3_0,
        .type_size                = sizeof(block_tq3_4s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq3_4s,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq3_4s_ref,
    },
```

Immediately after its closing `},` add:

```c
    [GGML_TYPE_TQ4P_D128] = {
        .type_name                = "tq4p_d128",
        .blck_size                = QK_TQ4P_D128,
        .type_size                = sizeof(block_tq4p_d128),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) ggml_dequantize_row_tq4p_d128,
        .from_float_ref           = (ggml_from_float_t) ggml_quantize_row_tq4p_d128,
    },
    [GGML_TYPE_TQ4P_D256] = {
        .type_name                = "tq4p_d256",
        .blck_size                = QK_TQ4P_D256,
        .type_size                = sizeof(block_tq4p_d256),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) ggml_dequantize_row_tq4p_d256,
        .from_float_ref           = (ggml_from_float_t) ggml_quantize_row_tq4p_d256,
    },
```

## 3. `ggml/src/ggml-cpu/ggml-cpu.c` — `type_traits_cpu` dispatch

At the top of the file, add:

```c
#include "../ggml-tq-paper.h"
```

Find the `TQ3_4S` entry in the `type_traits_cpu` array (around line 407):

```c
    [GGML_TYPE_TQ3_4S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_tq3_4s_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
```

Immediately after its closing `},` add:

```c
    [GGML_TYPE_TQ4P_D128] = {
        .from_float               = (ggml_from_float_t) ggml_quantize_row_tq4p_d128,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_tq4p_d128_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ4P_D256] = {
        .from_float               = (ggml_from_float_t) ggml_quantize_row_tq4p_d256,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_tq4p_d256_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
```

## 4. `ggml/src/CMakeLists.txt` — add source

Find the `add_library(ggml-base ...)` call (the library that compiles
`ggml.c`, `ggml-alloc.c`, `ggml-quants.c`, etc.). Add `ggml-tq-paper.c` to
its source list. Depending on your version of the CMakeLists, it looks
like one of:

```cmake
add_library(ggml-base
    ...
    ggml.c
    ggml-quants.c
    ggml-tq-paper.c            # <-- add
    ...
)
```

Or a `target_sources(ggml-base PRIVATE ...)` block; same idea.

---

## After applying

```bash
scripts/build_ollama_tq.sh --rebuild
```

Check that the types are registered by running the fork's quantize test:

```bash
cd ~/.local/src/ollama-tq/llama.cpp-tq3/build
./bin/llama-quantize --help 2>&1 | grep -iE "tq4p|tq3_0"
# expected: both tq3_0 and tq4p_d128 / tq4p_d256 appear
```

## Troubleshooting

- **`undefined reference to ggml_quantize_row_tq4p_d128`** — step 4 missed;
  ensure `ggml-tq-paper.c` is in the `ggml-base` sources list.
- **`redefinition of 'block_tq4p_d128'`** — you added the struct to
  `ggml-common.h` by hand; remove it. The struct is defined once, in
  `ggml-tq-paper.h`.
- **`no member named 'vec_dot' in 'struct ggml_type_traits'`** — you put
  the 4-field dispatch entry in the wrong table. Steps 2 and 3 are
  different tables: `ggml.c` (6 fields) vs `ggml-cpu/ggml-cpu.c` (4 fields).
- **Enum value 47/48 collision** — upstream has added types since this
  patch was written. Pick the next free enum value, update all three
  places (ggml.h, ggml.c table indices, ggml-cpu.c table indices).

## Rollback

The changes are all additive, so rollback is `git checkout --`:

```bash
cd ~/.local/src/ollama-tq/llama.cpp-tq3
git checkout -- ggml/include/ggml.h ggml/src/ggml.c \
                ggml/src/ggml-cpu/ggml-cpu.c ggml/src/CMakeLists.txt
rm ggml/src/ggml-tq-paper.{c,h} \
   ggml/src/tqp_constants_d{128,256}.h \
   ggml/src/tqp_centroids_d{128,256}.h
```

## Ollama Go plumbing — `apply_go_plumbing.sh`

The C-level types registered above are invisible to ollama's Go layer
until four switch statements are extended. `apply_go_plumbing.sh` does
this automatically; the manual edits are documented below.

### Summary of Go edits

| File | Kind | What |
|---|---|---|
| `ml/backend.go` | additive | 2 `DType` iota constants |
| `ml/backend/ggml/ggml.go` | additive | 2 cases in `DType()` + 2 in `ggmlDType()` |
| `runner/ollamarunner/cache.go` | additive | 2 cases in `kvCacheTypeFromStr` |
| `llama/llama.go` | additive | 2 cases in `kvCacheTypeFromStr` |

### 5. `ml/backend.go` — DType constants

Find:

```go
const (
	DTypeOther DType = iota
	DTypeF32
	DTypeF16
	DTypeQ80
	DTypeQ40
	DTypeI32
	DTypeMXFP4
)
```

Add after `DTypeMXFP4`:

```go
	DTypeTQ4P_D128
	DTypeTQ4P_D256
```

### 6. `ml/backend/ggml/ggml.go` — `DType()` method (C→Go)

Find the `DType()` method's switch. After:

```go
	case C.GGML_TYPE_MXFP4:
		return ml.DTypeMXFP4
```

Add:

```go
	case C.GGML_TYPE_TQ4P_D128:
		return ml.DTypeTQ4P_D128
	case C.GGML_TYPE_TQ4P_D256:
		return ml.DTypeTQ4P_D256
```

### 7. `ml/backend/ggml/ggml.go` — `ggmlDType()` function (Go→C)

Find the `ggmlDType()` function's switch. After:

```go
	case ml.DTypeMXFP4:
		return C.GGML_TYPE_MXFP4
```

Add:

```go
	case ml.DTypeTQ4P_D128:
		return C.GGML_TYPE_TQ4P_D128
	case ml.DTypeTQ4P_D256:
		return C.GGML_TYPE_TQ4P_D256
```

### 8. `runner/ollamarunner/cache.go` — `kvCacheTypeFromStr`

Find:

```go
	case "q4_0":
		return ml.DTypeQ40
	default:
```

Insert before `default:`:

```go
	case "tq4p_d128":
		return ml.DTypeTQ4P_D128
	case "tq4p_d256":
		return ml.DTypeTQ4P_D256
```

### 9. `llama/llama.go` — `kvCacheTypeFromStr`

Find:

```go
	case "q4_0":
		return C.GGML_TYPE_Q4_0
	default:
```

Insert before `default:`:

```go
	case "tq4p_d128":
		return C.GGML_TYPE_TQ4P_D128
	case "tq4p_d256":
		return C.GGML_TYPE_TQ4P_D256
```

### Anchor strategy

`apply_go_plumbing.sh` uses value-based anchors — it matches the
exact text of existing case arms or constant names (e.g.
`DTypeMXFP4`, `case "q4_0":\n\t\treturn ml.DTypeQ40\n`) and inserts
after them. If ollama restructures these switch statements, the script
will refuse (non-zero exit) rather than silently miss.

Markers:
- `DTypeTQ4P_D128` in `backend.go` and `ggml.go`
- `tq4p_d128` string literal in `cache.go`
- `GGML_TYPE_TQ4P` in `llama.go`

## Ollama-side follow-up

Once the types work in llama.cpp, extend `scripts/patch_ollama_kv_types.sh`
to allowlist `tq4p_d128` and `tq4p_d256`. This commit already does that.
