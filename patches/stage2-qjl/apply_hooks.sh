#!/usr/bin/env bash
# Apply the TQ4P hook edits to a ggml source tree. Additive only.
#
# Usage: apply_hooks.sh <ggml-root>
#
#   <ggml-root> is the directory containing  include/ggml.h  and
#   src/ggml.c . In ollama it is  ml/backend/ggml/ggml/ ; in stock
#   llama.cpp it is  ggml/ .
#
# What gets edited:
#   1. include/ggml.h          — 2 new enum values, bump GGML_TYPE_COUNT
#   2. src/ggml.c              — 2 entries in type_traits + #include
#   3. src/ggml-cpu/ggml-cpu.c — 2 entries in type_traits_cpu + #include
#   4. src/CMakeLists.txt      — add ggml-tq-paper.c to ggml-base sources
#   5. src/ggml-cuda/ggml-cuda.cu — TQ4P CUDA vec_dot dispatch
#   6. src/ggml-cuda/ggml-cuda.cu — F32 -> TQ4P CUDA quantize in CPY
#   7. src/ggml-cuda/{ggml-cuda.cu,set-rows.cu} — TQ4P SET_ROWS dispatch
#   8. src/ggml-cuda/{convert.cu,fattn.cu} — TQ4P flash-attention staging
#
# Anchors are value-based (reads GGML_TYPE_COUNT = N dynamically) so this
# survives minor ggml drift. Idempotent via "tq4p" marker check.

set -euo pipefail

GGML="${1:?usage: apply_hooks.sh <ggml-root>}"
[[ -f "$GGML/include/ggml.h" && -f "$GGML/src/ggml.c" ]] || {
    echo "not a ggml tree: $GGML (missing include/ggml.h or src/ggml.c)" >&2
    exit 1
}

GGML_H="$GGML/include/ggml.h"
GGML_C="$GGML/src/ggml.c"
CPU_C="$GGML/src/ggml-cpu/ggml-cpu.c"
CMAKE="$GGML/src/CMakeLists.txt"
CUDA_CU="$GGML/src/ggml-cuda/ggml-cuda.cu"
CONVERT_CU="$GGML/src/ggml-cuda/convert.cu"
FATTN_CU="$GGML/src/ggml-cuda/fattn.cu"

MARKER="tq4p"

# ---------- 1. enum ----------
if grep -qi "$MARKER" "$GGML_H" 2>/dev/null; then
    echo "[=] ggml.h already patched"
else
    echo "[+] ggml.h: adding TQ4P_D128/TQ4P_D256 enum values"
    python3 - "$GGML_H" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()
# Strip any pre-existing TQ4P entries (idempotent re-patching).
t = re.sub(r'\s*//.*TurboQuant paper-faithful.*\n', '', t)
t = re.sub(r'\s*//.*See src/ggml-tq-paper\.h.*\n', '', t)
t = re.sub(r'\s*GGML_TYPE_TQ4P_D128\s*=\s*\d+,\s*\n', '', t)
t = re.sub(r'\s*GGML_TYPE_TQ4P_D256\s*=\s*\d+,\s*\n', '', t)
m = re.search(r"\s*GGML_TYPE_COUNT\s*=\s*(\d+),", t)
if not m:
    sys.exit("ggml.h: no GGML_TYPE_COUNT line found")
n = int(m.group(1))
insert = (
    "\n"
    "        // TurboQuant paper-faithful (Haar rotation + Lloyd-Max + QJL).\n"
    "        // See src/ggml-tq-paper.h. Added by patches/stage2-qjl.\n"
    f"        GGML_TYPE_TQ4P_D128 = {n},\n"
    f"        GGML_TYPE_TQ4P_D256 = {n+1},"
)
t = t[:m.start()] + insert + re.sub(
    r"GGML_TYPE_COUNT\s*=\s*\d+,",
    f"GGML_TYPE_COUNT   = {n+2},",
    t[m.start():],
    count=1,
)
p.write_text(t)
print(f"[+] TQ4P_D128 = {n}, TQ4P_D256 = {n+1}, COUNT -> {n+2}")
PY
fi

# ---------- 2. ggml.c type_traits ----------
if grep -q "$MARKER" "$GGML_C" 2>/dev/null; then
    echo "[=] ggml.c already patched"
else
    echo "[+] ggml.c: #include + type_traits entries"
    python3 - "$GGML_C" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()

# Insert #include after "ggml-quants.h" — always at file scope.
m = re.search(r'^#include\s+"ggml-quants\.h"\s*\n', t, re.MULTILINE)
if not m:
    sys.exit('ggml.c: could not find #include "ggml-quants.h" anchor')
t = t[:m.end()] + '#include "ggml-tq-paper.h"\n' + t[m.end():]

# Find the type_traits array and insert entries before its closing `};`.
arr_start = re.search(
    r"static\s+const\s+struct\s+ggml_type_traits\s+type_traits\s*\[[^\]]*\]\s*=\s*\{",
    t,
)
if not arr_start:
    sys.exit("ggml.c: type_traits array not found")
# Scan forward to matching closer.
depth = 1
i = arr_start.end()
while i < len(t) and depth > 0:
    if t[i] == '{':
        depth += 1
    elif t[i] == '}':
        depth -= 1
    i += 1
if depth != 0:
    sys.exit("ggml.c: unbalanced braces in type_traits array")
# i is now 1 past the closing '}'; next char should be ';'.
# Insert just before the closing '}'.
close = i - 1
entries = (
    "    [GGML_TYPE_TQ4P_D128] = {\n"
    "        .type_name                = \"tq4p_d128\",\n"
    "        .blck_size                = QK_TQ4P_D128,\n"
    "        .type_size                = sizeof(block_tq4p_d128),\n"
    "        .is_quantized             = true,\n"
    "        .to_float                 = (ggml_to_float_t) ggml_dequantize_row_tq4p_d128,\n"
    "        .from_float_ref           = ggml_quantize_row_tq4p_d128_default,\n"
    "    },\n"
    "    [GGML_TYPE_TQ4P_D256] = {\n"
    "        .type_name                = \"tq4p_d256\",\n"
    "        .blck_size                = QK_TQ4P_D256,\n"
    "        .type_size                = sizeof(block_tq4p_d256),\n"
    "        .is_quantized             = true,\n"
    "        .to_float                 = (ggml_to_float_t) ggml_dequantize_row_tq4p_d256,\n"
    "        .from_float_ref           = ggml_quantize_row_tq4p_d256_default,\n"
    "    },\n"
)
t = t[:close] + entries + t[close:]
p.write_text(t)
PY
fi

# ---------- 3. ggml-cpu.c type_traits_cpu ----------
if [[ -f "$CPU_C" ]]; then
    if grep -q "$MARKER" "$CPU_C" 2>/dev/null; then
        echo "[=] ggml-cpu.c already patched"
    else
        echo "[+] ggml-cpu.c: #include + type_traits_cpu entries"
        python3 - "$CPU_C" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()

# ollama renamed ggml-quants.h -> quants.h in its vendored tree; upstream
# llama.cpp / turbo-tan still uses ggml-quants.h. Try both.
m = (re.search(r'^#include\s+"ggml-quants\.h"\s*\n', t, re.MULTILINE)
     or re.search(r'^#include\s+"quants\.h"\s*\n', t, re.MULTILINE))
if not m:
    sys.exit('ggml-cpu.c: could not find quants.h or ggml-quants.h anchor')
t = t[:m.end()] + '#include "../ggml-tq-paper.h"\n' + t[m.end():]

arr_start = re.search(
    r"static\s+const\s+struct\s+ggml_type_traits_cpu\s+type_traits_cpu\s*\[[^\]]*\]\s*=\s*\{",
    t,
)
if not arr_start:
    sys.exit("ggml-cpu.c: type_traits_cpu array not found")
depth = 1
i = arr_start.end()
while i < len(t) and depth > 0:
    if t[i] == '{':
        depth += 1
    elif t[i] == '}':
        depth -= 1
    i += 1
if depth != 0:
    sys.exit("ggml-cpu.c: unbalanced braces in type_traits_cpu array")
close = i - 1
entries = (
    "    [GGML_TYPE_TQ4P_D128] = {\n"
    "        .from_float               = ggml_quantize_row_tq4p_d128_default,\n"
    "        // .from_float_bf16       = (ggml_from_float_t) ggml_quantize_row_tq4p_d128_bf16,\n"
    "        // .from_float_f16        = (ggml_from_float_t) ggml_quantize_row_tq4p_d128_f16,\n"
    "        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_tq4p_d128_f32,\n"
    "        .vec_dot_type             = GGML_TYPE_F32,\n"
    "        .nrows                    = 1,\n"
    "    },\n"
    "    [GGML_TYPE_TQ4P_D256] = {\n"
    "        .from_float               = ggml_quantize_row_tq4p_d256_default,\n"
    "        // .from_float_bf16       = (ggml_from_float_t) ggml_quantize_row_tq4p_d256_bf16,\n"
    "        // .from_float_f16        = (ggml_from_float_t) ggml_quantize_row_tq4p_d256_f16,\n"
    "        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_tq4p_d256_f32,\n"
    "        .vec_dot_type             = GGML_TYPE_F32,\n"
    "        .nrows                    = 1,\n"
    "    },\n"
)
t = t[:close] + entries + t[close:]
p.write_text(t)
PY
    fi
else
    echo "[!] $CPU_C not found; skipping (CPU backend may be built differently)"
fi

# ---------- 4. CMakeLists.txt source list ----------
if grep -q "ggml-tq-paper.c" "$CMAKE" 2>/dev/null; then
    echo "[=] CMakeLists.txt already patched"
else
    echo "[+] CMakeLists.txt: adding ggml-tq-paper.c to ggml-base sources"
    python3 - "$CMAKE" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()
m = re.search(r"(ggml-quants\.c\b)", t)
if not m:
    m = re.search(r"(ggml\.c\b)(?!pp)", t)
if not m:
    sys.exit("CMakeLists.txt: could not find ggml-quants.c or ggml.c anchor")
line_end = t.find("\n", m.end())
# Match the existing indentation of the anchor line.
line_start = t.rfind("\n", 0, m.start()) + 1
indent = t[line_start:m.start()]
t = t[:line_end] + "\n" + indent + "ggml-tq-paper.c" + t[line_end:]
p.write_text(t)
PY
fi

# ---------- 5. ggml-cuda.cu dispatch (CUDA only) ----------
if [[ -f "$CUDA_CU" && -f "$GGML/src/ggml-cuda/tqp-vec-dot.cu" ]]; then
    if grep -q "ggml_cuda_op_tqp_vec_dot" "$CUDA_CU" 2>/dev/null; then
        echo "[=] ggml-cuda.cu already patched"
    else
        echo "[+] ggml-cuda.cu: TQ4P CUDA dispatch"
        python3 - "$CUDA_CU" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()

include_anchor = '#include "ggml.h"\n'
if include_anchor not in t:
    sys.exit('ggml-cuda.cu: could not find #include "ggml.h" anchor')

decl = (
    '\nextern "C" void ggml_cuda_op_tqp_vec_dot(\n'
    '    ggml_backend_cuda_context & ctx,\n'
    '    const ggml_tensor * src0,\n'
    '    const ggml_tensor * src1,\n'
    '    ggml_tensor * dst);\n'
)
t = t.replace(include_anchor, include_anchor + decl, 1)

sig = re.search(
    r'static\s+void\s+ggml_cuda_mul_mat\s*\(\s*'
    r'ggml_backend_cuda_context\s*&\s*ctx\s*,\s*'
    r'const\s+ggml_tensor\s*\*\s*src0\s*,\s*'
    r'const\s+ggml_tensor\s*\*\s*src1\s*,\s*'
    r'ggml_tensor\s*\*\s*dst\s*\)\s*\{\s*',
    t,
)
if not sig:
    sys.exit('ggml-cuda.cu: ggml_cuda_mul_mat signature not found')

split_line = re.search(
    r'(\s*const\s+bool\s+split\s*=\s*ggml_backend_buft_is_cuda_split\s*\(\s*src0->buffer->buft\s*\)\s*;\s*)',
    t[sig.end():],
)
if not split_line:
    sys.exit('ggml-cuda.cu: split declaration in ggml_cuda_mul_mat not found')

insert_at = sig.end() + split_line.end()
dispatch = (
    '\n'
    '    if (!split && (src0->type == GGML_TYPE_TQ4P_D128 || src0->type == GGML_TYPE_TQ4P_D256)) {\n'
    '        ggml_cuda_op_tqp_vec_dot(ctx, src0, src1, dst);\n'
    '        return;\n'
    '    }\n'
)
t = t[:insert_at] + dispatch + t[insert_at:]
p.write_text(t)
PY
fi
fi

# ---------- 6. ggml-cuda.cu: F32 → TQ4P on-device quantize in CPY dispatch ----
if [[ -f "$CUDA_CU" && -f "$GGML/src/ggml-cuda/tqp-quantize.cu" ]]; then
    if grep -q "ggml_cuda_tqp_quantize_row" "$CUDA_CU" 2>/dev/null; then
        echo "[=] ggml-cuda.cu CPY→TQ4P dispatch already patched"
    else
        echo "[+] ggml-cuda.cu: F32 → TQ4P on-device quantize in CPY dispatch"
        python3 - "$CUDA_CU" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()

# Add extern declarations for TQ4P quantize kernels. Place near the
# existing ggml_cuda_op_tqp_vec_dot declaration (added by hook 5).
vec_dot_decl = 'extern "C" void ggml_cuda_op_tqp_vec_dot('
if vec_dot_decl not in t:
    sys.exit('ggml-cuda.cu: ggml_cuda_op_tqp_vec_dot declaration not found (hook 5 missing?)')

quant_decl = (
    '\nextern "C" void ggml_cuda_tqp_quantize_row_d128(\n'
    '    const float * x, void * y, int64_t k, uint8_t layer_byte,\n'
    '    cudaStream_t stream);\n'
    'extern "C" void ggml_cuda_tqp_quantize_row_d256(\n'
    '    const float * x, void * y, int64_t k, uint8_t layer_byte,\n'
    '    cudaStream_t stream);\n'
)
idx = t.index(vec_dot_decl)
# Insert before the vec_dot declaration
t = t[:idx] + quant_decl + t[idx:]

# Patch the GGML_OP_CPY case to intercept F32 → TQ4P_D128/D256 before
# falling through to ggml_cuda_cpy. We replace the simple dispatch line
# with a TQ4P-aware block that reads layer_byte from dst->op_params[0].
old_cpy = 'case GGML_OP_CPY:\n            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);\n            break;'
if old_cpy not in t:
    sys.exit('ggml-cuda.cu: GGML_OP_CPY dispatch not found')

new_cpy = (
    'case GGML_OP_CPY:\n'
    '            {\n'
    '                const ggml_tensor * cpy_src = dst->src[0];\n'
    '                ggml_tensor * cpy_dst = dst->src[1];\n'
    '                const bool dst_tq4p = (cpy_dst->type == GGML_TYPE_TQ4P_D128 || cpy_dst->type == GGML_TYPE_TQ4P_D256);\n'
    '                const bool src_f32  = (cpy_src->type == GGML_TYPE_F32);\n'
    '                // BF16/F16 → TQ4P: upcast to fp32 on host, then quantize on device.\n'
    '                // TODO: add device-side bf16/f16 load kernels for zero-copy path.\n'
    '                if ((src_f32) && dst_tq4p) {\n'
    '                    const uint8_t layer_byte = (uint8_t)(dst->op_params[0] & 0xff);\n'
    '                    cudaStream_t stream = ctx.stream();\n'
    '                    const float * src_d = (const float *)cpy_src->data;\n'
    '                    void * dst_d = cpy_dst->data;\n'
    '                    const int64_t ne = ggml_nelements(cpy_src);\n'
    '                    if (cpy_dst->type == GGML_TYPE_TQ4P_D128) {\n'
    '                        ggml_cuda_tqp_quantize_row_d128(src_d, dst_d, ne, layer_byte, stream);\n'
    '                    } else {\n'
    '                        ggml_cuda_tqp_quantize_row_d256(src_d, dst_d, ne, layer_byte, stream);\n'
    '                    }\n'
    '                } else {\n'
    '                    ggml_cuda_cpy(ctx, cpy_src, cpy_dst);\n'
    '                }\n'
    '            }\n'
    '            break;'
)
t = t.replace(old_cpy, new_cpy, 1)

# Also patch the supports_op query for GGML_OP_CPY to report F32→TQ4P
# as supported. Find the last F32→IQ4_NL check in the CPY supports block
# and append TQ4P entries.
support_anchor = 'if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {\n                    return true;\n                }'
if support_anchor not in t:
    print('WARNING: CPY supports_op anchor not found, skipping', file=sys.stderr)
else:
    tq4p_support = (
        'if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {\n'
        '                    return true;\n'
        '                }\n'
        '                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_TQ4P_D128) {\n'
        '                    return true;\n'
        '                }\n'
        '                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_TQ4P_D256) {\n'
        '                    return true;\n'
        '                }'
    )
    t = t.replace(support_anchor, tq4p_support, 1)

p.write_text(t)
PY
    fi
fi

# ---------- 7. ggml-cuda.cu: SET_ROWS + MUL_MAT supports_op + SET_ROWS dispatch
#
# Newer ollama uses SET_ROWS (not CPY) for KV cache writes. Also, without a
# MUL_MAT supports_op entry for TQ4P types the attention Q×K^T falls back
# to CPU (the "74 graph splits" issue).
#
# Three edits, each guarded on its own dependencies:
#   a) MUL_MAT supports_op: add TQ4P to the a->type switch. Only requires
#      the MUL_MAT dispatch (hook 5), which in turn needs tqp-vec-dot.cu.
#   b) SET_ROWS supports_op: add TQ4P to the SET_ROWS type list. Requires
#      set-rows.cu (ollama must support SET_ROWS) and tqp-set-rows.cu
#      (our kernel impl). Skip independently if either is missing.
#   c) SET_ROWS dispatch in set-rows.cu (same guard as (b)).
#
# Splitting (a) from (b) matters because older ollama trees lack set-rows.cu
# but still have MUL_MAT. Bundling all three under the SET_ROWS guard
# (as the first version of this hook did) would leave MUL_MAT unpatched
# on those trees, silently forcing CPU fallback for attention.

MARKER_MUL_MAT='TQ4P MUL_MAT on GPU'
MARKER_SET_ROWS='TQ4P_D128 SET_ROWS'
SET_ROWS_FILE="$GGML/src/ggml-cuda/set-rows.cu"
TQP_SET_ROWS_IMPL="$GGML/src/ggml-cuda/tqp-set-rows.cu"
TQP_VEC_DOT_IMPL="$GGML/src/ggml-cuda/tqp-vec-dot.cu"

# (a) MUL_MAT supports_op in ggml-cuda.cu — only depends on tqp-vec-dot.cu.
if [[ -f "$CUDA_CU" && -f "$TQP_VEC_DOT_IMPL" ]]; then
    if grep -qF "$MARKER_MUL_MAT" "$CUDA_CU"; then
        echo "[=] ggml-cuda.cu MUL_MAT supports_op already patched"
    else
        echo "[+] patching ggml-cuda.cu MUL_MAT supports_op for TQ4P"
        python3 - "$CUDA_CU" <<'PY'
import sys, pathlib

p = pathlib.Path(sys.argv[1])
t = p.read_text()

# Anchor on BF16 which is the last type before "return true; default:".
mulmat_anchor = '                    case GGML_TYPE_BF16:\n                        return true;\n                    default:\n                        return false;'
if mulmat_anchor not in t:
    print("ERROR: MUL_MAT supports_op anchor (BF16) not found", file=sys.stderr)
    sys.exit(1)
mulmat_new = (
    '                    case GGML_TYPE_BF16:\n'
    '                    case GGML_TYPE_TQ4P_D128: // Hook 7: TQ4P MUL_MAT on GPU\n'
    '                    case GGML_TYPE_TQ4P_D256:\n'
    '                        return true;\n'
    '                    default:\n'
    '                        return false;'
)
t = t.replace(mulmat_anchor, mulmat_new, 1)

p.write_text(t)
print(f"[+] patched MUL_MAT supports_op: {p}")
PY
    fi
fi

# (b) SET_ROWS supports_op in ggml-cuda.cu — needs set-rows.cu + tqp-set-rows.cu.
if [[ -f "$CUDA_CU" && -f "$SET_ROWS_FILE" && -f "$TQP_SET_ROWS_IMPL" ]]; then
    if grep -qF "$MARKER_SET_ROWS" "$CUDA_CU"; then
        echo "[=] ggml-cuda.cu SET_ROWS supports_op already patched"
    else
        echo "[+] patching ggml-cuda.cu SET_ROWS supports_op for TQ4P"
        python3 - "$CUDA_CU" <<'PY'
import sys, pathlib

p = pathlib.Path(sys.argv[1])
t = p.read_text()

anchor = 'op->type == GGML_TYPE_IQ4_NL) &&'
if anchor not in t:
    print("ERROR: SET_ROWS supports_op anchor (IQ4_NL) not found", file=sys.stderr)
    sys.exit(1)
replacement = (
    'op->type == GGML_TYPE_IQ4_NL ||\n'
    '                       // TQ4P_D128 SET_ROWS — Hook 7\n'
    '                       op->type == GGML_TYPE_TQ4P_D128 || op->type == GGML_TYPE_TQ4P_D256) &&'
)
t = t.replace(anchor, replacement, 1)

p.write_text(t)
print(f"[+] patched SET_ROWS supports_op: {p}")
PY
    fi
fi

# (c) dispatch in set-rows.cu — same guard as (b).
if [[ -f "$SET_ROWS_FILE" && -f "$TQP_SET_ROWS_IMPL" ]]; then
    if grep -qF "$MARKER_SET_ROWS" "$SET_ROWS_FILE"; then
        echo "[=] set-rows.cu TQ4P dispatch already patched"
    else
        echo "[+] patching set-rows.cu TQ4P dispatch"
        python3 - "$SET_ROWS_FILE" <<'PY'
import sys, pathlib

p = pathlib.Path(sys.argv[1])
t = p.read_text()

# Add extern "C" declarations near the top (after includes).
include_anchor = '#include "set-rows.cuh"\n'
if include_anchor not in t:
    print("ERROR: set-rows.cu include anchor not found", file=sys.stderr)
    sys.exit(1)
decl_block = (
    '#include "set-rows.cuh"\n'
    '\n'
    '// TQ4P SET_ROWS — Hook 7. Defined in tqp-set-rows.cu.\n'
    'extern "C" void ggml_cuda_set_rows_tq4p_d128(\n'
    '    const float *, const void *, void *, uint8_t, bool,\n'
    '    int64_t, int64_t, int64_t, cudaStream_t);\n'
    'extern "C" void ggml_cuda_set_rows_tq4p_d256(\n'
    '    const float *, const void *, void *, uint8_t, bool,\n'
    '    int64_t, int64_t, int64_t, cudaStream_t);\n'
)
t = t.replace(include_anchor, decl_block, 1)

# Intercept TQ4P before the existing I64/I32 dispatch.
dispatch_anchor = (
    '    GGML_ASSERT(src0->type == GGML_TYPE_F32);\n'
    '    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);\n'
    '\n'
    '    if (src1->type == GGML_TYPE_I64) {'
)
if dispatch_anchor not in t:
    print("ERROR: ggml_cuda_op_set_rows dispatch anchor not found in set-rows.cu", file=sys.stderr)
    sys.exit(1)
dispatch_new = (
    '    GGML_ASSERT(src0->type == GGML_TYPE_F32);\n'
    '    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);\n'
    '\n'
    '    // TQ4P_D128 SET_ROWS — Hook 7.\n'
    '    if (dst->type == GGML_TYPE_TQ4P_D128 || dst->type == GGML_TYPE_TQ4P_D256) {\n'
    '        const uint8_t layer_byte = (uint8_t)(dst->op_params[0] & 0xff);\n'
    '        const float * src0_d = (const float *)src0->data;\n'
    '        const bool idx_i64 = (src1->type == GGML_TYPE_I64);\n'
    '        cudaStream_t stream = ctx.stream();\n'
    '        const int64_t n_rows = src0->ne[1];\n'
    '        const int64_t src0_stride = src0->nb[1] / sizeof(float);\n'
    '        const int64_t dst_stride = dst->nb[1];\n'
    '        if (dst->type == GGML_TYPE_TQ4P_D128) {\n'
    '            ggml_cuda_set_rows_tq4p_d128(src0_d, src1->data, dst->data, layer_byte, idx_i64, n_rows, src0_stride, dst_stride, stream);\n'
    '        } else {\n'
    '            ggml_cuda_set_rows_tq4p_d256(src0_d, src1->data, dst->data, layer_byte, idx_i64, n_rows, src0_stride, dst_stride, stream);\n'
    '        }\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    if (src1->type == GGML_TYPE_I64) {'
)
t = t.replace(dispatch_anchor, dispatch_new, 1)
p.write_text(t)
print(f"[+] patched: {p}")
PY
    fi
fi

# ---------- 8. ggml-cuda flash attention: TQ4P staging dequantize --------
# Flash attention can consume f16 K/V data through launch_fattn's staging
# buffer. Register TQ4P -> f16/f32 converters, and let the fattn selector map
# TQ4P K/V tensors to the existing F16 kernel variants.

TQP_DEQUANT_IMPL="$GGML/src/ggml-cuda/tqp-dequantize.cu"

if [[ -f "$CONVERT_CU" && -f "$TQP_DEQUANT_IMPL" ]]; then
    if grep -q "dequantize_row_tq4p_d128_cuda" "$CONVERT_CU" 2>/dev/null; then
        echo "[=] convert.cu TQ4P dequantize dispatch already patched"
    else
        echo "[+] patching convert.cu TQ4P dequantize dispatch"
        python3 - "$CONVERT_CU" <<'PY'
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
t = p.read_text()

include_anchor = '#include "dequantize.cuh"\n'
if include_anchor not in t:
    print('ERROR: convert.cu include anchor not found', file=sys.stderr)
    sys.exit(1)

decl = (
    '#include "dequantize.cuh"\n'
    '\n'
    '// TQ4P flash-attention staging dequantizers - Hook 8.\n'
    'extern "C" void dequantize_row_tq4p_d128_cuda(const void *, half *, int64_t, cudaStream_t);\n'
    'extern "C" void dequantize_row_tq4p_d256_cuda(const void *, half *, int64_t, cudaStream_t);\n'
    'extern "C" void dequantize_row_tq4p_d128_f32_cuda(const void *, float *, int64_t, cudaStream_t);\n'
    'extern "C" void dequantize_row_tq4p_d256_f32_cuda(const void *, float *, int64_t, cudaStream_t);\n'
    'extern "C" void dequantize_row_tq4p_d128_nc_cuda(\n'
    '    const void *, half *, int64_t, int64_t, int64_t, int64_t,\n'
    '    int64_t, int64_t, int64_t, cudaStream_t);\n'
    'extern "C" void dequantize_row_tq4p_d256_nc_cuda(\n'
    '    const void *, half *, int64_t, int64_t, int64_t, int64_t,\n'
    '    int64_t, int64_t, int64_t, cudaStream_t);\n'
)
t = t.replace(include_anchor, decl, 1)

fp16_anchor = (
    '        case GGML_TYPE_MXFP4:\n'
    '            return dequantize_row_mxfp4_cuda;\n'
    '        case GGML_TYPE_F32:'
)
if fp16_anchor not in t:
    print('ERROR: convert.cu ggml_get_to_fp16_cuda MXFP4 anchor not found', file=sys.stderr)
    sys.exit(1)
fp16_replacement = (
    '        case GGML_TYPE_MXFP4:\n'
    '            return dequantize_row_mxfp4_cuda;\n'
    '        case GGML_TYPE_TQ4P_D128:\n'
    '            return dequantize_row_tq4p_d128_cuda;\n'
    '        case GGML_TYPE_TQ4P_D256:\n'
    '            return dequantize_row_tq4p_d256_cuda;\n'
    '        case GGML_TYPE_F32:'
)
t = t.replace(fp16_anchor, fp16_replacement, 1)

fp32_anchor = (
    '        case GGML_TYPE_MXFP4:\n'
    '            return dequantize_row_mxfp4_cuda;\n'
    '        case GGML_TYPE_F16:'
)
if fp32_anchor not in t:
    print('ERROR: convert.cu ggml_get_to_fp32_cuda MXFP4 anchor not found', file=sys.stderr)
    sys.exit(1)
fp32_replacement = (
    '        case GGML_TYPE_MXFP4:\n'
    '            return dequantize_row_mxfp4_cuda;\n'
    '        case GGML_TYPE_TQ4P_D128:\n'
    '            return dequantize_row_tq4p_d128_f32_cuda;\n'
    '        case GGML_TYPE_TQ4P_D256:\n'
    '            return dequantize_row_tq4p_d256_f32_cuda;\n'
    '        case GGML_TYPE_F16:'
)
t = t.replace(fp32_anchor, fp32_replacement, 1)

fp16_nc_anchor = (
    '        case GGML_TYPE_Q8_0:\n'
    '            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;\n'
    '        case GGML_TYPE_BF16:'
)
if fp16_nc_anchor not in t:
    print('ERROR: convert.cu ggml_get_to_fp16_nc_cuda Q8_0 anchor not found', file=sys.stderr)
    sys.exit(1)
fp16_nc_replacement = (
    '        case GGML_TYPE_Q8_0:\n'
    '            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;\n'
    '        case GGML_TYPE_TQ4P_D128:\n'
    '            return dequantize_row_tq4p_d128_nc_cuda;\n'
    '        case GGML_TYPE_TQ4P_D256:\n'
    '            return dequantize_row_tq4p_d256_nc_cuda;\n'
    '        case GGML_TYPE_BF16:'
)
t = t.replace(fp16_nc_anchor, fp16_nc_replacement, 1)

p.write_text(t)
print(f"[+] patched: {p}")
PY
    fi
fi

if [[ -f "$FATTN_CU" && -f "$TQP_DEQUANT_IMPL" ]]; then
    if grep -q "TQ4P flash-attention staging" "$FATTN_CU" 2>/dev/null; then
        echo "[=] fattn.cu TQ4P staging already patched"
    else
        echo "[+] patching fattn.cu TQ4P flash-attention staging"
        python3 - "$FATTN_CU" <<'PY'
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
t = p.read_text()

old_macro = (
    '#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \\\n'
    '    {                                                                                                            \\\n'
    '        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \\\n'
    '        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \\\n'
    '        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \\\n'
)
new_macro = (
    '#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \\\n'
    '    {                                                                                                            \\\n'
    '        const bool type_K_tq4p = K->type == GGML_TYPE_TQ4P_D128 || K->type == GGML_TYPE_TQ4P_D256;               \\\n'
    '        const bool type_V_tq4p = V->type == GGML_TYPE_TQ4P_D128 || V->type == GGML_TYPE_TQ4P_D256;               \\\n'
    '        const bool type_K_okay = K->type == (type_K)                                                            \\\n'
    '            || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16)                                           \\\n'
    '            || (type_K_tq4p && (type_K) == GGML_TYPE_F16); /* TQ4P flash-attention staging */                    \\\n'
    '        const bool type_V_okay = V->type == (type_V)                                                            \\\n'
    '            || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16)                                           \\\n'
    '            || (type_V_tq4p && (type_V) == GGML_TYPE_F16);                                                       \\\n'
    '        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \\\n'
)
if old_macro not in t:
    print('ERROR: fattn.cu FATTN_VEC_CASE anchor not found', file=sys.stderr)
    sys.exit(1)
t = t.replace(old_macro, new_macro, 1)

switch_anchor = (
    '    switch (K->type) {\n'
    '        case GGML_TYPE_F32:\n'
    '        case GGML_TYPE_F16:\n'
    '            break;'
)
if switch_anchor not in t:
    print('ERROR: fattn.cu K->type switch anchor not found', file=sys.stderr)
    sys.exit(1)
switch_replacement = (
    '    switch (K->type) {\n'
    '        case GGML_TYPE_F32:\n'
    '        case GGML_TYPE_F16:\n'
    '        case GGML_TYPE_TQ4P_D128:\n'
    '        case GGML_TYPE_TQ4P_D256:\n'
    '            break;'
)
t = t.replace(switch_anchor, switch_replacement, 1)

# V->type switch: TQ4P is a KV-cache type, so the V tensor is also TQ4P.
# Without this patch, the default case in the V->type switch fires and
# GGML_ABORTs during flash attention.
v_switch_anchor = (
    '    switch (V->type) {\n'
    '        case GGML_TYPE_F32:\n'
    '        case GGML_TYPE_F16:\n'
    '            break;'
)
if v_switch_anchor not in t:
    print('ERROR: fattn.cu V->type switch anchor not found', file=sys.stderr)
    sys.exit(1)
v_switch_replacement = (
    '    switch (V->type) {\n'
    '        case GGML_TYPE_F32:\n'
    '        case GGML_TYPE_F16:\n'
    '        case GGML_TYPE_TQ4P_D128:\n'
    '        case GGML_TYPE_TQ4P_D256:\n'
    '            break;'
)
t = t.replace(v_switch_anchor, v_switch_replacement, 1)

p.write_text(t)
print(f"[+] patched: {p}")
PY
    fi
fi

echo "All hooks applied."
