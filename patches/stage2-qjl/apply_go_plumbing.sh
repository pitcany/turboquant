#!/usr/bin/env bash
# Patch ollama's Go layer so tq4p_d128 / tq4p_d256 cache-type strings map
# end-to-end to the correct GGML enum values instead of falling back to f16.
#
# Four sites need cases:
#   1. ml/backend.go              — DType constants (iota enum)
#   2. ml/backend/ggml/ggml.go    — DType() C→Go and ggmlDType() Go→C
#   3. runner/ollamarunner/cache.go — kvCacheTypeFromStr  string→ml.DType
#   4. llama/llama.go              — kvCacheTypeFromStr  string→C.enum_ggml_type
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

# ---------- 1. ml/backend.go — DType constants -------------------------------

BACKEND_GO="$OLLAMA_DIR/ml/backend.go"
info "patching DType constants in $BACKEND_GO"
patch_file "$BACKEND_GO" "$MARKER" - "$BACKEND_GO" <<'PY'
import sys, pathlib, re

path = pathlib.Path(sys.argv[1])
text = path.read_text()

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
    "\tDTypeTQ4P_D128\n"
    "\tDTypeTQ4P_D256\n"
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
patch_file "$GGML_GO" "$MARKER" - "$GGML_GO" <<'PY'
import sys, pathlib

path = pathlib.Path(sys.argv[1])
text = path.read_text()

# --- DType() method: C enum → ml.DType ---
# Anchor: "case C.GGML_TYPE_MXFP4:\n\t\treturn ml.DTypeMXFP4\n"
# Insert TQ4P cases right after it, before the default.
anchor_dtype = "case C.GGML_TYPE_MXFP4:\n\t\treturn ml.DTypeMXFP4\n"
if anchor_dtype not in text:
    print(f"ERROR: DType() anchor not found in {path}", file=sys.stderr)
    sys.exit(1)

insert_dtype = (
    "\tcase C.GGML_TYPE_TQ4P_D128:\n"
    "\t\treturn ml.DTypeTQ4P_D128\n"
    "\tcase C.GGML_TYPE_TQ4P_D256:\n"
    "\t\treturn ml.DTypeTQ4P_D256\n"
    "\tcase C.GGML_TYPE_TQP_D128_B2:\n"
    "\t\treturn ml.DTypeTQP_D128_B2\n"
    "\tcase C.GGML_TYPE_TQP_D128_B4:\n"
    "\t\treturn ml.DTypeTQP_D128_B4\n"
    "\tcase C.GGML_TYPE_TQP_D256_B2:\n"
    "\t\treturn ml.DTypeTQP_D256_B2\n"
    "\tcase C.GGML_TYPE_TQP_D256_B4:\n"
    "\t\treturn ml.DTypeTQP_D256_B4\n"
)
text = text.replace(anchor_dtype, anchor_dtype + insert_dtype, 1)

# --- ggmlDType() function: ml.DType → C enum ---
# Anchor: "case ml.DTypeMXFP4:\n\t\treturn C.GGML_TYPE_MXFP4\n"
anchor_ggml = "case ml.DTypeMXFP4:\n\t\treturn C.GGML_TYPE_MXFP4\n"
if anchor_ggml not in text:
    print(f"ERROR: ggmlDType() anchor not found in {path}", file=sys.stderr)
    sys.exit(1)

insert_ggml = (
    "\tcase ml.DTypeTQ4P_D128:\n"
    "\t\treturn C.GGML_TYPE_TQ4P_D128\n"
    "\tcase ml.DTypeTQ4P_D256:\n"
    "\t\treturn C.GGML_TYPE_TQ4P_D256\n"
    "\tcase ml.DTypeTQP_D128_B2:\n"
    "\t\treturn C.GGML_TYPE_TQP_D128_B2\n"
    "\tcase ml.DTypeTQP_D128_B4:\n"
    "\t\treturn C.GGML_TYPE_TQP_D128_B4\n"
    "\tcase ml.DTypeTQP_D256_B2:\n"
    "\t\treturn C.GGML_TYPE_TQP_D256_B2\n"
    "\tcase ml.DTypeTQP_D256_B4:\n"
    "\t\treturn C.GGML_TYPE_TQP_D256_B4\n"
)
text = text.replace(anchor_ggml, anchor_ggml + insert_ggml, 1)

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
    '\tcase "tq4p_d128":\n'
    '\t\treturn ml.DTypeTQ4P_D128\n'
    '\tcase "tq4p_d256":\n'
    '\t\treturn ml.DTypeTQ4P_D256\n'
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
    '\tcase "tq4p_d128":\n'
    '\t\treturn C.GGML_TYPE_TQ4P_D128\n'
    '\tcase "tq4p_d256":\n'
    '\t\treturn C.GGML_TYPE_TQ4P_D256\n'
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
    '\t\tif dstType == C.GGML_TYPE_TQ4P_D128 || dstType == C.GGML_TYPE_TQ4P_D256 ||\n'
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
import sys, pathlib

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

new_support = (
    '// tqp_kv_cache_head_dim_guard: constrain TurboQuant KV types to models\n'
    '// whose resolved key/value head lengths match the requested block size.\n'
    'func kvCacheTypeHeadDim(cacheType string) (uint64, bool) {\n'
    '\tswitch cacheType {\n'
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
    '\tif !slices.Contains([]string{"q8_0", "q4_0", "tq3_0", "tq4p_d128", "tq4p_d256", "tqp_d128_b2", "tqp_d128_b4", "tqp_d256_b2", "tqp_d256_b4"}, cacheType) {\n'
    '\t\treturn false\n'
    '\t}\n'
    '\n'
    '\texpectedHeadDim, constrained := kvCacheTypeHeadDim(cacheType)\n'
    '\tif !constrained {\n'
    '\t\treturn true\n'
    '\t}\n'
    '\n'
    '\tkv := f.KV()\n'
    '\theadCountK := uint64(kv.Uint("attention.key_length"))\n'
    '\theadCountV := uint64(kv.Uint("attention.value_length"))\n'
    '\tif headCountK == 0 || headCountV == 0 {\n'
    '\t\tif heads := kv.HeadCountMax(); heads > 0 {\n'
    '\t\t\tfallbackHeadDim := kv.EmbeddingLength() / heads\n'
    '\t\t\tif headCountK == 0 {\n'
    '\t\t\t\theadCountK = fallbackHeadDim\n'
    '\t\t\t}\n'
    '\t\t\tif headCountV == 0 {\n'
    '\t\t\t\theadCountV = fallbackHeadDim\n'
    '\t\t\t}\n'
    '\t\t}\n'
    '\t}\n'
    '\tif headCountK == 0 {\n'
    '\t\theadCountK = kv.EmbeddingHeadCountK()\n'
    '\t}\n'
    '\tif headCountV == 0 {\n'
    '\t\theadCountV = kv.EmbeddingHeadCountV()\n'
    '\t}\n'
    '\treturn headCountK == expectedHeadDim && headCountV == expectedHeadDim\n'
    '}\n'
)

if old_support in text:
    text = text.replace(old_support, new_support, 1)
elif old_support_pristine in text:
    text = text.replace(old_support_pristine, new_support, 1)
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

# ---------- summary ----------------------------------------------------------

echo
echo "Go plumbing complete. The following cache-type strings now resolve"
echo "to their correct GGML types instead of falling back to f16:"
echo "  tq4p_d128 → GGML_TYPE_TQ4P_D128 (enum 40)"
echo "  tq4p_d256 → GGML_TYPE_TQ4P_D256 (enum 41)"
echo
echo "OLLAMA_TQP_ROTATION={wht,haar} is read at startup and forwarded"
echo "to tqp_set_default_rotation() for process-wide rotation default."
