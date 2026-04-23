#!/usr/bin/env bash
# Patch ollama's Go layer so TurboQuant KV cache-type strings map
# end-to-end to the correct GGML enum values instead of falling back to f16.
#
# Eight patch sites (sections 1–8b):
#   1. ml/backend.go              — DType constants (iota enum)
#   2. ml/backend/ggml/ggml.go    — DType() C→Go and ggmlDType() Go→C
#   3. runner/ollamarunner/cache.go — kvCacheTypeFromStr  string→ml.DType
#   4. llama/llama.go              — kvCacheTypeFromStr  string→C.enum_ggml_type
#   5. ml/backend/ggml/ggml.go    — Copy() op_params for TQ4P
#   6. OLLAMA_TQP_ROTATION        — init hook
#   7. fs/ggml/ggml.go            — kvCacheBytesPerElement() bytes/element
#                                    estimator (searched dynamically)
#  8a. runner/ollamarunner/*.go   — KV cache type slog.Info hook
#  8b. fs/ggml/ggml.go            — KV memory estimate slog.Info hook
#
# Anchors are value-based (grep for known surrounding patterns), not
# line-number based, so small upstream drift doesn't break this.
#
# Idempotent: each site is guarded by a marker string.  Reruns are no-ops.
#
# Usage: apply_go_plumbing.sh <ollama-source-dir>

set -euo pipefail

OLLAMA_DIR="${1:?usage: apply_go_plumbing.sh <ollama-source-dir>}"
MARKER='DTypeTQP_D128_B2'        # Go-visible marker in backend.go
MARKER_LLAMA='GGML_TYPE_TQ4P'    # C-visible marker in llama.go
MARKER_STR='tq4p_d128'           # string literal marker in both cache switches
MARKER_BPE='69.0 / 128.0'        # kvCacheBytesPerElement() patch marker
MARKER_LOG_CONFIG='TQ4P: KV cache type configured'
MARKER_LOG_ESTIMATE='TQ4P: KV cache memory estimate'

# ---------- helpers ----------------------------------------------------------

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[+] $*"; }
skip() { echo "[=] $*"; }

patch_file() {
    # patch_file <file> <marker> <python-script>
    local file="$1" marker="$2"
    shift 2
    if ! [[ -f "$file" ]]; then
        die "expected file not found: $file"
    fi
    if grep -qF "$marker" "$file"; then
        skip "already patched: $file"
        return 0
    fi
    python3 "$@"
}

find_go_file_with_literal() {
    # find_go_file_with_literal <root> <literal>
    local root="$1" literal="$2"
    python3 - "$root" "$literal" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
literal = sys.argv[2]
matches = []

if not root.exists():
    print(f"ERROR: search root does not exist: {root}", file=sys.stderr)
    sys.exit(1)

for path in root.rglob("*.go"):
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        continue
    if literal in text:
        matches.append(path)

if not matches:
    print(f"ERROR: no Go file under {root} contains literal: {literal!r}", file=sys.stderr)
    sys.exit(1)

if len(matches) > 1:
    print(
        f"ERROR: multiple Go files under {root} contain literal {literal!r}:",
        file=sys.stderr,
    )
    for match in matches:
        print(f"  - {match}", file=sys.stderr)
    sys.exit(1)

print(matches[0])
PY
}

# ---------- 1. ml/backend.go — DType constants -------------------------------

BACKEND_GO="$OLLAMA_DIR/ml/backend.go"
info "patching DType constants in $BACKEND_GO"
python3 - "$BACKEND_GO" <<'PY'
import sys, pathlib, re

path = pathlib.Path(sys.argv[1])
text = path.read_text()

for name in [
    "DTypeTQ4P_D64",
    "DTypeTQ4P_D128",
    "DTypeTQ4P_D256",
    "DTypeTQP_D64_B2",
    "DTypeTQP_D64_B4",
    "DTypeTQP_D128_B2",
    "DTypeTQP_D128_B4",
    "DTypeTQP_D256_B2",
    "DTypeTQP_D256_B4",
]:
    text = re.sub(rf'^\t{name}\r?\n', '', text, flags=re.MULTILINE)

# Anchor: the DTypeMXFP4 line inside a const ( ... ) iota block.
# We insert our two types right after it.
anchor = 'DTypeMXFP4\n'
if anchor not in text:
    # Try with \r\n just in case
    anchor = 'DTypeMXFP4\r\n'
if anchor not in text:
    print(f"ERROR: cannot find 'DTypeMXFP4' constant in {path}", file=sys.stderr)
    sys.exit(1)

insertion = (
    "\tDTypeTQ4P_D64\n"
    "\tDTypeTQ4P_D128\n"
    "\tDTypeTQ4P_D256\n"
    "\tDTypeTQP_D64_B2\n"
    "\tDTypeTQP_D64_B4\n"
    "\tDTypeTQP_D128_B2\n"
    "\tDTypeTQP_D128_B4\n"
    "\tDTypeTQP_D256_B2\n"
    "\tDTypeTQP_D256_B4\n"
)
text = text.replace(anchor, anchor.rstrip('\r\n') + '\n' + insertion, 1)
path.write_text(text)
print(f"[+] patched: {path}")
PY

# ---------- 2. ml/backend/ggml/ggml.go — DType() and ggmlDType() ------------

GGML_GO="$OLLAMA_DIR/ml/backend/ggml/ggml.go"
FS_GGML_GO="$OLLAMA_DIR/fs/ggml/ggml.go"
info "patching DType() and ggmlDType() in $GGML_GO"
python3 - "$GGML_GO" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()

dtype_function = (
    "func (t *Tensor) DType() ml.DType {\n"
    "\tswitch t.t._type {\n"
    "\tcase C.GGML_TYPE_F32:\n"
    "\t\treturn ml.DTypeF32\n"
    "\tcase C.GGML_TYPE_F16:\n"
    "\t\treturn ml.DTypeF16\n"
    "\tcase C.GGML_TYPE_Q8_0:\n"
    "\t\treturn ml.DTypeQ80\n"
    "\tcase C.GGML_TYPE_Q4_0:\n"
    "\t\treturn ml.DTypeQ40\n"
    "\tcase C.GGML_TYPE_I32:\n"
    "\t\treturn ml.DTypeI32\n"
    "\tcase C.GGML_TYPE_MXFP4:\n"
    "\t\treturn ml.DTypeMXFP4\n"
    "\tcase C.GGML_TYPE_TQ4P_D64:\n"
    "\t\treturn ml.DTypeTQ4P_D64\n"
    "\tcase C.GGML_TYPE_TQ4P_D128:\n"
    "\t\treturn ml.DTypeTQ4P_D128\n"
    "\tcase C.GGML_TYPE_TQ4P_D256:\n"
    "\t\treturn ml.DTypeTQ4P_D256\n"
    "\tcase C.GGML_TYPE_TQP_D64_B2:\n"
    "\t\treturn ml.DTypeTQP_D64_B2\n"
    "\tcase C.GGML_TYPE_TQP_D64_B4:\n"
    "\t\treturn ml.DTypeTQP_D64_B4\n"
    "\tcase C.GGML_TYPE_TQP_D128_B2:\n"
    "\t\treturn ml.DTypeTQP_D128_B2\n"
    "\tcase C.GGML_TYPE_TQP_D128_B4:\n"
    "\t\treturn ml.DTypeTQP_D128_B4\n"
    "\tcase C.GGML_TYPE_TQP_D256_B2:\n"
    "\t\treturn ml.DTypeTQP_D256_B2\n"
    "\tcase C.GGML_TYPE_TQP_D256_B4:\n"
    "\t\treturn ml.DTypeTQP_D256_B4\n"
    "\tdefault:\n"
    "\t\treturn ml.DTypeOther\n"
    "\t}\n"
    "}\n"
)
dtype_pattern = re.compile(r'func \(t \*Tensor\) DType\(\) ml\.DType \{\n.*?\n\}\n', re.S)
text, count = dtype_pattern.subn(dtype_function, text, count=1)
if count == 0:
    print(f"ERROR: DType() function not found in {path}", file=sys.stderr)
    sys.exit(1)

ggml_dtype_function = (
    "func ggmlDType(dtype ml.DType) uint32 {\n"
    "\tswitch dtype {\n"
    "\tcase ml.DTypeF32:\n"
    "\t\treturn C.GGML_TYPE_F32\n"
    "\tcase ml.DTypeF16:\n"
    "\t\treturn C.GGML_TYPE_F16\n"
    "\tcase ml.DTypeQ80:\n"
    "\t\treturn C.GGML_TYPE_Q8_0\n"
    "\tcase ml.DTypeQ40:\n"
    "\t\treturn C.GGML_TYPE_Q4_0\n"
    "\tcase ml.DTypeI32:\n"
    "\t\treturn C.GGML_TYPE_I32\n"
    "\tcase ml.DTypeMXFP4:\n"
    "\t\treturn C.GGML_TYPE_MXFP4\n"
    "\tcase ml.DTypeTQ4P_D64:\n"
    "\t\treturn C.GGML_TYPE_TQ4P_D64\n"
    "\tcase ml.DTypeTQ4P_D128:\n"
    "\t\treturn C.GGML_TYPE_TQ4P_D128\n"
    "\tcase ml.DTypeTQ4P_D256:\n"
    "\t\treturn C.GGML_TYPE_TQ4P_D256\n"
    "\tcase ml.DTypeTQP_D64_B2:\n"
    "\t\treturn C.GGML_TYPE_TQP_D64_B2\n"
    "\tcase ml.DTypeTQP_D64_B4:\n"
    "\t\treturn C.GGML_TYPE_TQP_D64_B4\n"
    "\tcase ml.DTypeTQP_D128_B2:\n"
    "\t\treturn C.GGML_TYPE_TQP_D128_B2\n"
    "\tcase ml.DTypeTQP_D128_B4:\n"
    "\t\treturn C.GGML_TYPE_TQP_D128_B4\n"
    "\tcase ml.DTypeTQP_D256_B2:\n"
    "\t\treturn C.GGML_TYPE_TQP_D256_B2\n"
    "\tcase ml.DTypeTQP_D256_B4:\n"
    "\t\treturn C.GGML_TYPE_TQP_D256_B4\n"
    "\tdefault:\n"
    "\t\tpanic(\"unsupported dtype\")\n"
    "\t}\n"
    "}\n"
)
ggml_dtype_pattern = re.compile(r'func ggmlDType\(dtype ml\.DType\) uint32 \{\n.*?\n\}\n', re.S)
text, count = ggml_dtype_pattern.subn(ggml_dtype_function, text, count=1)
if count == 0:
    print(f"ERROR: ggmlDType() function not found in {path}", file=sys.stderr)
    sys.exit(1)

path.write_text(text)
print(f"[+] patched: {path}")
PY

# ---------- 3. runner/ollamarunner/cache.go — kvCacheTypeFromStr -------------

CACHE_GO="$OLLAMA_DIR/runner/ollamarunner/cache.go"
info "patching kvCacheTypeFromStr in $CACHE_GO"
patch_file "$CACHE_GO" "$MARKER_STR" - "$CACHE_GO" <<'PY'
import sys, pathlib

path = pathlib.Path(sys.argv[1])
text = path.read_text()

# Anchor: the q4_0 case block, immediately before "default:".
# We match the exact two-line pattern and append our cases.
anchor = 'case "q4_0":\n\t\treturn ml.DTypeQ40\n'
if anchor not in text:
    print(f"ERROR: cache.go kvCacheTypeFromStr anchor not found in {path}", file=sys.stderr)
    sys.exit(1)

insert = (
    '\tcase "tq4p_d64":\n'
    '\t\treturn ml.DTypeTQ4P_D64\n'
    '\tcase "tq4p_d128":\n'
    '\t\treturn ml.DTypeTQ4P_D128\n'
    '\tcase "tq4p_d256":\n'
    '\t\treturn ml.DTypeTQ4P_D256\n'
    '\tcase "tqp_d64_b2":\n'
    '\t\treturn ml.DTypeTQP_D64_B2\n'
    '\tcase "tqp_d64_b4":\n'
    '\t\treturn ml.DTypeTQP_D64_B4\n'
    '\tcase "tqp_d128_b2":\n'
    '\t\treturn ml.DTypeTQP_D128_B2\n'
    '\tcase "tqp_d128_b4":\n'
    '\t\treturn ml.DTypeTQP_D128_B4\n'
    '\tcase "tqp_d256_b2":\n'
    '\t\treturn ml.DTypeTQP_D256_B2\n'
    '\tcase "tqp_d256_b4":\n'
    '\t\treturn ml.DTypeTQP_D256_B4\n'
)
text = text.replace(anchor, anchor + insert, 1)
path.write_text(text)
print(f"[+] patched: {path}")
PY

# ---------- 4. llama/llama.go — kvCacheTypeFromStr ---------------------------

LLAMA_GO="$OLLAMA_DIR/llama/llama.go"
info "patching kvCacheTypeFromStr in $LLAMA_GO"
patch_file "$LLAMA_GO" "$MARKER_LLAMA" - "$LLAMA_GO" <<'PY'
import sys, pathlib

path = pathlib.Path(sys.argv[1])
text = path.read_text()

# Anchor: the q4_0 case block, immediately before "default:".
anchor = 'case "q4_0":\n\t\treturn C.GGML_TYPE_Q4_0\n'
if anchor not in text:
    print(f"ERROR: llama.go kvCacheTypeFromStr anchor not found in {path}", file=sys.stderr)
    sys.exit(1)

insert = (
    '\tcase "tq4p_d64":\n'
    '\t\treturn C.GGML_TYPE_TQ4P_D64\n'
    '\tcase "tq4p_d128":\n'
    '\t\treturn C.GGML_TYPE_TQ4P_D128\n'
    '\tcase "tq4p_d256":\n'
    '\t\treturn C.GGML_TYPE_TQ4P_D256\n'
    '\tcase "tqp_d64_b2":\n'
    '\t\treturn C.GGML_TYPE_TQP_D64_B2\n'
    '\tcase "tqp_d64_b4":\n'
    '\t\treturn C.GGML_TYPE_TQP_D64_B4\n'
    '\tcase "tqp_d128_b2":\n'
    '\t\treturn C.GGML_TYPE_TQP_D128_B2\n'
    '\tcase "tqp_d128_b4":\n'
    '\t\treturn C.GGML_TYPE_TQP_D128_B4\n'
    '\tcase "tqp_d256_b2":\n'
    '\t\treturn C.GGML_TYPE_TQP_D256_B2\n'
    '\tcase "tqp_d256_b4":\n'
    '\t\treturn C.GGML_TYPE_TQP_D256_B4\n'
)
text = text.replace(anchor, anchor + insert, 1)
path.write_text(text)
print(f"[+] patched: {path}")
PY

# ---------- 5. ml/backend/ggml/ggml.go — Copy() op_params for TQ4P ----------

MARKER_COPY='tqp_layer_byte_explicit_bits'
info "patching Copy() to set op_params[0] for TQ4P in $GGML_GO"
patch_file "$GGML_GO" "$MARKER_COPY" - "$GGML_GO" <<'PY'
import sys, pathlib, re

path = pathlib.Path(sys.argv[1])
text = path.read_text()

# Regex-based replacement: match the entire Copy() method body regardless of
# its current state (pristine, tq4p-only, or explicit-bits). This avoids
# accumulating whole-function fallback chains for each patch generation.
copy_re = re.compile(
    r'(func \(t \*Tensor\) Copy\(ctx ml\.Context, t2 ml\.Tensor\) ml\.Tensor \{)'
    r'.*?'
    r'(\n\})',
    re.DOTALL,
)

new_body = (
    r'\1' '\n'
    '\tresult := C.ggml_cpy(ctx.(*Context).ctx, t.t, t2.(*Tensor).t)\n'
    '\t// tqp_layer_byte_explicit_bits: pass layer index to CUDA quantize dispatch.\n'
    '\tc := ctx.(*Context)\n'
    '\tif c.layer >= 0 {\n'
    '\t\tdstType := t2.(*Tensor).t._type\n'
    '\t\tif dstType == C.GGML_TYPE_TQ4P_D64 || dstType == C.GGML_TYPE_TQ4P_D128 || dstType == C.GGML_TYPE_TQ4P_D256 ||\n'
    '\t\t\tdstType == C.GGML_TYPE_TQP_D64_B2 || dstType == C.GGML_TYPE_TQP_D64_B4 ||\n'
    '\t\t\tdstType == C.GGML_TYPE_TQP_D128_B2 || dstType == C.GGML_TYPE_TQP_D128_B4 ||\n'
    '\t\t\tdstType == C.GGML_TYPE_TQP_D256_B2 || dstType == C.GGML_TYPE_TQP_D256_B4 {\n'
    '\t\t\tresult.op_params[0] = C.int32_t(c.layer & 0x1f)\n'
    '\t\t}\n'
    '\t}\n'
    '\treturn &Tensor{b: t.b, t: result}\n'
    '}'
)

new_text, count = copy_re.subn(new_body, text, count=1)
if count == 0:
    print(f"ERROR: Copy() method not found in {path}", file=sys.stderr)
    sys.exit(1)
text = new_text
path.write_text(text)
print(f"[+] patched Copy() for TQ4P layer_byte: {path}")
PY

# ---------- 6. model-aware KV cache type guard --------------------------------
#
# Ollama's SupportsKVCacheType() is only a string allowlist upstream. For the
# TurboQuant d128/d256 families that is insufficient: a d256 cache type must
# only be used when the model's resolved key/value head lengths are 256.

MARKER_KV_SUPPORT='tqp_kv_cache_head_dim_guard'
info "patching SupportsKVCacheType() head-dim guard in $FS_GGML_GO"
patch_file "$FS_GGML_GO" "$MARKER_KV_SUPPORT" - "$FS_GGML_GO" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()

old_support = (
    '// SupportsKVCacheType checks if the requested cache type is supported\n'
    'func (f GGML) SupportsKVCacheType(cacheType string) bool {\n'
    '\tif cacheType == "" || cacheType == "f16" {\n'
    '\t\treturn true\n'
    '\t}\n'
    '\n'
    '\treturn slices.Contains([]string{"q8_0", "q4_0", "tq3_0", "tq4p_d128", "tq4p_d256", "tqp_d128_b2", "tqp_d128_b4", "tqp_d256_b2", "tqp_d256_b4"}, cacheType)\n'
    '}\n'
)
old_support_pristine = (
    '// SupportsKVCacheType checks if the requested cache type is supported\n'
    'func (f GGML) SupportsKVCacheType(cacheType string) bool {\n'
    '\tif cacheType == "" || cacheType == "f16" {\n'
    '\t\treturn true\n'
    '\t}\n'
    '\n'
    '\treturn slices.Contains([]string{"q8_0", "q4_0"}, cacheType)\n'
    '}\n'
)
old_support_patterns = [
    re.compile(
        r'// SupportsKVCacheType checks if the requested cache type is supported\n'
        r'func \(f GGML\) SupportsKVCacheType\(cacheType string\) bool \{\n'
        r'\tif cacheType == "" \|\| cacheType == "f16" \{\n'
        r'\t\treturn true\n'
        r'\t\}\n'
        r'\n'
        r'\treturn slices\.Contains\(\[\]string\{.*?\}, cacheType\)\n'
        r'\}\n',
        re.S,
    ),
]

new_support = (
    '// tqp_kv_cache_head_dim_guard: constrain TurboQuant KV types to models\n'
    '// whose resolved key/value head lengths match the requested block size.\n'
    'func kvCacheTypeHeadDim(cacheType string) (uint64, bool) {\n'
    '\tswitch cacheType {\n'
    '\tcase "tq4p_d64", "tqp_d64_b2", "tqp_d64_b4":\n'
    '\t\treturn 64, true\n'
    '\tcase "tq4p_d128", "tqp_d128_b2", "tqp_d128_b4":\n'
    '\t\treturn 128, true\n'
    '\tcase "tq4p_d256", "tqp_d256_b2", "tqp_d256_b4":\n'
    '\t\treturn 256, true\n'
    '\tdefault:\n'
    '\t\treturn 0, false\n'
    '\t}\n'
    '}\n'
    '\n'
    '// SupportsKVCacheType checks if the requested cache type is supported.\n'
    'func (f GGML) SupportsKVCacheType(cacheType string) bool {\n'
    '\tif cacheType == "" || cacheType == "f16" {\n'
    '\t\treturn true\n'
    '\t}\n'
    '\n'
    '\tif !slices.Contains([]string{"q8_0", "q4_0", "tq3_0", "tq4p_d64", "tq4p_d128", "tq4p_d256", "tqp_d64_b2", "tqp_d64_b4", "tqp_d128_b2", "tqp_d128_b4", "tqp_d256_b2", "tqp_d256_b4"}, cacheType) {\n'
    '\t\treturn false\n'
    '\t}\n'
    '\n'
    '\texpectedHeadDim, constrained := kvCacheTypeHeadDim(cacheType)\n'
    '\tif !constrained {\n'
    '\t\treturn true\n'
    '\t}\n'
    '\n'
    '\theadCountK := f.KV().EmbeddingHeadCountK()\n'
    '\theadCountV := f.KV().EmbeddingHeadCountV()\n'
    '\treturn headCountK == expectedHeadDim && headCountV == expectedHeadDim\n'
    '}\n'
)

if old_support in text:
    text = text.replace(old_support, new_support, 1)
elif old_support_pristine in text:
    text = text.replace(old_support_pristine, new_support, 1)
else:
    for pattern in old_support_patterns:
        text, count = pattern.subn(new_support, text, count=1)
        if count:
            break
    else:
        print(f"ERROR: SupportsKVCacheType() block not found in {path}", file=sys.stderr)
        sys.exit(1)
path.write_text(text)
print(f"[+] patched SupportsKVCacheType() head-dim guard: {path}")
PY

# ---------- 7. OLLAMA_TQP_ROTATION init hook ---------------------------------
#
# Read the env var once at ollama startup and call tqp_set_default_rotation
# via cgo. This ensures the process-level default is set before any quantize
# call. The C library also reads the env var via pthread_once on first use,
# but this explicit init allows the Go layer to log the choice.

MARKER_ROT_INIT='tqp_set_default_rotation(uint8_t rot);'
info "patching rotation init in $GGML_GO"
patch_file "$GGML_GO" "$MARKER_ROT_INIT" - "$GGML_GO" <<'PY'
import sys, pathlib

path = pathlib.Path(sys.argv[1])
text = path.read_text()

# Add the cgo declaration into the preamble immediately before import "C".
preamble_anchor = '// #include "ggml-backend.h"\nimport "C"\n'
if preamble_anchor in text:
    text = text.replace(
        preamble_anchor,
        '// #include "ggml-backend.h"\n// void tqp_set_default_rotation(uint8_t rot);\nimport "C"\n',
        1,
    )
elif 'tqp_set_default_rotation(uint8_t rot);' not in text:
    print(f"WARNING: cgo preamble anchor not found in {path}; skipping rotation init", file=sys.stderr)
    sys.exit(0)

import re
init_code = (
    '// tqp_rotation_init: read OLLAMA_TQP_ROTATION env var and forward\n'
    '// to the C rotation selector.  Called from init().\n'
    'func init() {\n'
    '\tval := os.Getenv("OLLAMA_TQP_ROTATION")\n'
    '\tif val == "" {\n'
    '\t\treturn\n'
    '\t}\n'
    '\tvar rot C.uint8_t = 0xff // unset\n'
    '\tswitch {\n'
    '\tcase len(val) > 0 && (val[0] == \'h\' || val[0] == \'H\'):\n'
    '\t\trot = 1 // TQP_ROT_HAAR\n'
    '\tcase len(val) > 0 && (val[0] == \'w\' || val[0] == \'W\'):\n'
    '\t\trot = 0 // TQP_ROT_WHT\n'
    '\t}\n'
    '\tif rot != 0xff {\n'
    '\t\tC.tqp_set_default_rotation(rot)\n'
    '\t}\n'
    '}\n\n'
)

if '"os"' not in text:
    text = text.replace('import "C"\n', 'import "C"\nimport "os"\n', 1)

if 'tqp_rotation_init: read OLLAMA_TQP_ROTATION env var and forward' not in text:
    func_match = re.search(r'^func ', text, re.MULTILINE)
    if not func_match:
        print(f"WARNING: no func found in {path}; skipping rotation init", file=sys.stderr)
        sys.exit(0)
    text = text[:func_match.start()] + init_code + text[func_match.start():]

path.write_text(text)
print(f"[+] patched rotation init: {path}")
PY

# ---------- 7. fs/ggml/ggml.go — kvCacheBytesPerElement ----------------------

BPE_GO="$(find_go_file_with_literal "$OLLAMA_DIR" 'func kvCacheBytesPerElement')"
info "patching kvCacheBytesPerElement in $BPE_GO"
patch_file "$BPE_GO" "$MARKER_BPE" - "$BPE_GO" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()

func_match = re.search(
    r'func kvCacheBytesPerElement\s*\([^)]*\)\s*float64\s*\{(?P<body>.*?)^\}',
    text,
    re.MULTILINE | re.DOTALL,
)
if not func_match:
    print(f"ERROR: kvCacheBytesPerElement() not found in {path}", file=sys.stderr)
    sys.exit(1)

body = func_match.group("body")
if 'case "tq4p_d128":' in body:
    print(f"[=] kvCacheBytesPerElement already handles TQ4P: {path}")
    sys.exit(0)

default_match = re.search(r'^(?P<indent>\s*)default:\n', body, re.MULTILINE)
if not default_match:
    print(f"ERROR: default: case not found in kvCacheBytesPerElement() in {path}", file=sys.stderr)
    sys.exit(1)

indent = default_match.group("indent")
case_indent = indent
return_indent = indent + "\t"
insert = (
    f'{case_indent}case "tq4p_d64":\n'
    f'{return_indent}return 37.0 / 64.0\n'
    f'{case_indent}case "tqp_d64_b2":\n'
    f'{return_indent}return 29.0 / 64.0\n'
    f'{case_indent}case "tqp_d64_b4":\n'
    f'{return_indent}return 45.0 / 64.0\n'
    f'{case_indent}case "tq4p_d128":\n'
    f'{return_indent}return 69.0 / 128.0\n'
    f'{case_indent}case "tq4p_d256":\n'
    f'{return_indent}return 133.0 / 256.0\n'
    f'{case_indent}case "tqp_d128_b2":\n'
    f'{return_indent}return 53.0 / 128.0\n'
    f'{case_indent}case "tqp_d128_b4":\n'
    f'{return_indent}return 85.0 / 128.0\n'
    f'{case_indent}case "tqp_d256_b2":\n'
    f'{return_indent}return 101.0 / 256.0\n'
    f'{case_indent}case "tqp_d256_b4":\n'
    f'{return_indent}return 165.0 / 256.0\n'
)

body = body[:default_match.start()] + insert + body[default_match.start():]
text = text[:func_match.start("body")] + body + text[func_match.end("body"):]
path.write_text(text)
print(f"[+] patched kvCacheBytesPerElement: {path}")
PY

# ---------- 8a. runner/ollamarunner/*.go — KV cache type logging ------------

CONFIG_LOG_GO="$(find_go_file_with_literal "$OLLAMA_DIR/runner/ollamarunner" 'kvCacheTypeFromStr(')"
info "patching KV cache type logging in $CONFIG_LOG_GO"
patch_file "$CONFIG_LOG_GO" "$MARKER_LOG_CONFIG" - "$CONFIG_LOG_GO" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()

anchor = re.search(
    r'^(?P<indent>\s*)cache := model\.Config\(\)\.Cache\s*$',
    text,
    re.MULTILINE,
)
if not anchor:
    print(f"ERROR: could not find 'cache := model.Config().Cache' in {path}", file=sys.stderr)
    sys.exit(1)

log_line = (
    f'{anchor.group("indent")}slog.Info('
    f'"TQ4P: KV cache type configured", "type", kvCacheType)\n'
)
desired_start = anchor.end() + 1

log_line_pattern = re.compile(
    r'^\s*slog\.Info\("TQ4P: KV cache type configured", "type", kvCacheType\)\n',
    re.MULTILINE,
)
matches = list(log_line_pattern.finditer(text))

if matches:
    first = matches[0]
    if first.start() == desired_start and len(matches) == 1:
        print(f"[=] KV cache type logging already present: {path}")
        sys.exit(0)

    text = log_line_pattern.sub("", text)
    anchor = re.search(
        r'^(?P<indent>\s*)cache := model\.Config\(\)\.Cache\s*$',
        text,
        re.MULTILINE,
    )
    desired_start = anchor.end() + 1

text = text[:desired_start] + log_line + text[desired_start:]

if '"log/slog"' not in text:
    group_import = re.search(r'import\s*\(\n(?P<body>.*?)\n\)', text, re.DOTALL)
    if group_import:
        body = group_import.group("body")
        if '\t"log/slog"' not in body:
            body += '\n\t"log/slog"'
            text = text[:group_import.start("body")] + body + text[group_import.end("body"):]
    else:
        single_imports = list(re.finditer(r'^import\s+"[^"]+"\n', text, re.MULTILINE))
        if not single_imports:
            print(f"ERROR: could not add slog import to {path}", file=sys.stderr)
            sys.exit(1)
        last = single_imports[-1]
        text = text[:last.end()] + 'import "log/slog"\n' + text[last.end():]

path.write_text(text)
print(f"[+] patched KV cache type logging: {path}")
PY

# ---------- 8b. fs/ggml/ggml.go — KV memory estimate logging ----------------

info "patching KV memory estimate logging in $BPE_GO"
patch_file "$BPE_GO" "$MARKER_LOG_ESTIMATE" - "$BPE_GO" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()

if 'TQ4P: KV cache memory estimate' in text:
    print(f"[=] KV memory estimate logging already present: {path}")
    sys.exit(0)

assignment = re.search(
    r'^(?P<indent>\s*)(?P<bpe>[A-Za-z_]\w*)\s*(?::=|=)\s*kvCacheBytesPerElement\((?P<kv>[A-Za-z_]\w*)\)$',
    text,
    re.MULTILINE,
)
if not assignment:
    print(f"ERROR: could not find kvCacheBytesPerElement() call site in {path}", file=sys.stderr)
    sys.exit(1)

log_line = (
    f'{assignment.group("indent")}slog.Info('
    f'"TQ4P: KV cache memory estimate", "bytes_per_element", {assignment.group("bpe")}, '
    f'"type", {assignment.group("kv")})\n'
)
insert_at = assignment.end() + 1
text = text[:insert_at] + log_line + text[insert_at:]

if '"log/slog"' not in text:
    group_import = re.search(r'import\s*\(\n(?P<body>.*?)\n\)', text, re.DOTALL)
    if group_import:
        body = group_import.group("body")
        if '\t"log/slog"' not in body:
            body += '\n\t"log/slog"'
            text = text[:group_import.start("body")] + body + text[group_import.end("body"):]
    else:
        single_imports = list(re.finditer(r'^import\s+"[^"]+"\n', text, re.MULTILINE))
        if not single_imports:
            print(f"ERROR: could not add slog import to {path}", file=sys.stderr)
            sys.exit(1)
        last = single_imports[-1]
        text = text[:last.end()] + 'import "log/slog"\n' + text[last.end():]

path.write_text(text)
print(f"[+] patched KV memory estimate logging: {path}")
PY

# ---------- summary ----------------------------------------------------------

echo
echo "Go plumbing complete. The following cache-type strings now resolve"
echo "to their correct GGML types instead of falling back to f16:"
echo "  tq4p_d64    → GGML_TYPE_TQ4P_D64"
echo "  tq4p_d128   → GGML_TYPE_TQ4P_D128"
echo "  tq4p_d256   → GGML_TYPE_TQ4P_D256"
echo "  tqp_d64_b2  → GGML_TYPE_TQP_D64_B2"
echo "  tqp_d64_b4  → GGML_TYPE_TQP_D64_B4"
echo "  tqp_d128_b2 → GGML_TYPE_TQP_D128_B2"
echo "  tqp_d128_b4 → GGML_TYPE_TQP_D128_B4"
echo "  tqp_d256_b2 → GGML_TYPE_TQP_D256_B2"
echo "  tqp_d256_b4 → GGML_TYPE_TQP_D256_B4"
echo
echo "kvCacheBytesPerElement() now reports the correct TQ4P byte rates:"
echo "  tq4p_d64    → 37/64 bytes per element"
echo "  tqp_d64_b2  → 29/64 bytes per element"
echo "  tqp_d64_b4  → 45/64 bytes per element"
echo "  tq4p_d128   → 69/128 bytes per element"
echo "  tqp_d128_b2 → 53/128 bytes per element"
echo "  tqp_d128_b4 → 85/128 bytes per element"
echo "  tq4p_d256   → 133/256 bytes per element"
echo "  tqp_d256_b2 → 101/256 bytes per element"
echo "  tqp_d256_b4 → 165/256 bytes per element"
echo
echo "Debug logging now emits:"
echo "  TQ4P: KV cache type configured"
echo "  TQ4P: KV cache memory estimate"
echo
echo "OLLAMA_TQP_ROTATION={wht,haar} is read at startup and forwarded"
echo "to tqp_set_default_rotation() for process-wide rotation default."
