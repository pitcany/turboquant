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
MARKER='DTypeTQ4P_D128'          # Go-visible marker in backend.go
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
)
text = text.replace(anchor, anchor.rstrip('\r\n') + '\n' + insertion, 1)
path.write_text(text)
print(f"[+] patched: {path}")
PY

# ---------- 2. ml/backend/ggml/ggml.go — DType() and ggmlDType() ------------

GGML_GO="$OLLAMA_DIR/ml/backend/ggml/ggml.go"
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
)
text = text.replace(anchor, anchor + insert, 1)
path.write_text(text)
print(f"[+] patched: {path}")
PY

# ---------- 5. ml/backend/ggml/ggml.go — Copy() op_params for TQ4P ----------

MARKER_COPY='tq4p_layer_byte'
info "patching Copy() to set op_params[0] for TQ4P in $GGML_GO"
patch_file "$GGML_GO" "$MARKER_COPY" - "$GGML_GO" <<'PY'
import sys, pathlib

path = pathlib.Path(sys.argv[1])
text = path.read_text()

# Anchor: the Copy method body.
old_copy = (
    'func (t *Tensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {\n'
    '\treturn &Tensor{\n'
    '\t\tb: t.b,\n'
    '\t\tt: C.ggml_cpy(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),\n'
    '\t}\n'
    '}'
)
if old_copy not in text:
    print(f"ERROR: Copy() method not found in {path}", file=sys.stderr)
    sys.exit(1)

# Replace with version that sets op_params[0] = layer_byte for TQ4P.
# The tq4p_layer_byte marker allows idempotent re-patching.
new_copy = (
    'func (t *Tensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {\n'
    '\tresult := C.ggml_cpy(ctx.(*Context).ctx, t.t, t2.(*Tensor).t)\n'
    '\t// tq4p_layer_byte: pass layer index to CUDA quantize dispatch.\n'
    '\tc := ctx.(*Context)\n'
    '\tif c.layer >= 0 {\n'
    '\t\tdstType := t2.(*Tensor).t._type\n'
    '\t\tif dstType == C.GGML_TYPE_TQ4P_D128 || dstType == C.GGML_TYPE_TQ4P_D256 {\n'
    '\t\t\tresult.op_params[0] = C.int32_t(c.layer & 0x1f)\n'
    '\t\t}\n'
    '\t}\n'
    '\treturn &Tensor{b: t.b, t: result}\n'
    '}'
)
text = text.replace(old_copy, new_copy, 1)
path.write_text(text)
print(f"[+] patched Copy() for TQ4P layer_byte: {path}")
PY

# ---------- summary ----------------------------------------------------------

echo
echo "Go plumbing complete. The following cache-type strings now resolve"
echo "to their correct GGML types instead of falling back to f16:"
echo "  tq4p_d128 → GGML_TYPE_TQ4P_D128 (enum 40)"
echo "  tq4p_d256 → GGML_TYPE_TQ4P_D256 (enum 41)"
