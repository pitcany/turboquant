# Fork-side hooks — 4 hand-edits for `turbo-tan/llama.cpp-tq3`

After `scripts/build_ollama_tq.sh --stage2` has copied the C files into
your fork clone (default: `~/.local/src/ollama-tq/llama.cpp-tq3/ggml/src/`),
apply these edits once. All additions, no in-place modifications to
existing functions.

Paths below are relative to the fork root.

## 1. `ggml/include/ggml.h` — register the enum values

Find the `ggml_type` enum. Locate the last `GGML_TYPE_TQ3_*` entry and add
two lines after it, picking the next free slot numbers (the fork currently
uses 41, 44, 46 for TQ3_0, TQ3_1S, TQ3_4S — so 47 and 48 should be free;
adjust if not):

```c
    GGML_TYPE_TQ3_4S     = 46,
    // --- TurboQuant paper-faithful (Stage 1 + QJL) ---
    GGML_TYPE_TQ4P_D128  = 47,
    GGML_TYPE_TQ4P_D256  = 48,
    GGML_TYPE_COUNT,
```

Also find the `ggml_op` enum and add a forward-compat entry after the
existing `GGML_OP_TURBO_WHT`:

```c
    GGML_OP_TURBO_WHT,
    // Reserved for the TurboQuant paper-faithful rotation op (CUDA follow-up).
    GGML_OP_TQP_ROTATE,
```

(Not actually used in the CPU-only patch — vec_dot rotates q internally.
Reserving the op keeps the CUDA follow-up's enum stable.)

## 2. `ggml/src/ggml-common.h` — register the block structs

Find the block_* struct definitions (near the fork's `block_tq3_0`). Add:

```c
// TurboQuant paper-faithful blocks; see ggml/src/ggml-tq-paper.h
typedef struct {
    ggml_half orig_norm;
    ggml_half res_d;
    uint8_t   qs[48];
    uint8_t   qjl_signs[16];
} block_tq4p_d128;
static_assert(sizeof(block_tq4p_d128) == 68, "wrong block_tq4p_d128 size");

typedef struct {
    ggml_half orig_norm;
    ggml_half res_d;
    uint8_t   qs[96];
    uint8_t   qjl_signs[32];
} block_tq4p_d256;
static_assert(sizeof(block_tq4p_d256) == 132, "wrong block_tq4p_d256 size");
```

Note: `ggml-tq-paper.h` uses `uint16_t` for fp16 fields to avoid a
dependency on `ggml-common.h`. The layouts are identical because `ggml_half`
is a `uint16_t` typedef. If your compiler complains about type mismatch at
link time, replace the `uint16_t` fields in `ggml-tq-paper.h` with
`ggml_half` and `#include "ggml-common.h"`.

## 3. `ggml/src/ggml-quants.c` — include and register

Near the top, add:

```c
#include "ggml-tq-paper.h"
```

Find the big `type_traits` (or `ggml_type_traits` / `GGML_TYPE_TRAITS`)
table. Append two entries at the bottom, just before the terminating
`[GGML_TYPE_COUNT] = {...}` or end of the array:

```c
    [GGML_TYPE_TQ4P_D128] = {
        .type_name                = "tq4p_d128",
        .blck_size                = QK_TQ4P_D128,
        .type_size                = sizeof(block_tq4p_d128),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) ggml_dequantize_row_tq4p_d128,
        .from_float_ref           = (ggml_from_float_t) ggml_quantize_row_tq4p_d128,
        .vec_dot                  = ggml_vec_dot_tq4p_d128_f32,
        .vec_dot_type             = GGML_TYPE_F32,   // reference takes fp32 query
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ4P_D256] = {
        .type_name                = "tq4p_d256",
        .blck_size                = QK_TQ4P_D256,
        .type_size                = sizeof(block_tq4p_d256),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) ggml_dequantize_row_tq4p_d256,
        .from_float_ref           = (ggml_from_float_t) ggml_quantize_row_tq4p_d256,
        .vec_dot                  = ggml_vec_dot_tq4p_d256_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
```

Fields other than the ones listed here (e.g. `from_float` optimized, or
CUDA dispatch pointers) use whatever the type_traits struct defaults are in
your version of the fork. If there are new fields added by the fork that
aren't mentioned above, leave them zero-initialized — the CPU draft doesn't
need them.

## 4. `ggml/src/CMakeLists.txt` — add the source

Find the list of source files passed to `add_library(ggml ...)` (or
`target_sources(ggml ...)`). Add:

```cmake
    ggml-tq-paper.c
```

If the fork splits CPU/CUDA into separate targets, put the file in the CPU
target. The file has no CUDA dependency.

---

## After applying all four

```bash
scripts/build_ollama_tq.sh --rebuild
```

Verify the types are recognized by `llama.cpp`:

```bash
~/.local/src/ollama-tq/llama.cpp-tq3/build/bin/llama-cli --help 2>&1 | grep tq4p
# expected: tq4p_d128 and tq4p_d256 appear in the cache-type list
```

If they don't, re-check steps 1-4. Most common failure is the enum number
colliding with something added upstream since this patch was written —
change to the next free value and update the matching `[GGML_TYPE_...]`
entries in step 3.

## Ollama-side follow-up

Once the types work in llama.cpp, extend `scripts/patch_ollama_kv_types.sh`
to allowlist `tq4p_d128` (and `tq4p_d256`) the same way it does `tq3_0`.
That's a one-line change and a separate commit.
