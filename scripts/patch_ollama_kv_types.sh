#!/usr/bin/env bash
# Widen ollama's KV-cache-type allowlist to include TurboQuant types.
#
# ollama validates OLLAMA_KV_CACHE_TYPE against a slice literal of allowed
# strings. This script finds that slice literal and adds our custom types.
#
#   tq3_0       — turbo-tan/llama.cpp-tq3 Walsh-Hadamard TurboQuant (legacy;
#                 only relevant if you've manually added tq3_0 to ollama's
#                 ggml fork; harmless otherwise)
#   tq4p_d128   — paper-faithful TurboQuant (Llama / Qwen2/3, head_dim 128)
#   tq4p_d256   — paper-faithful TurboQuant (Qwen3.5 gated attention, head_dim 256)
#
# Targets the pattern []string{"q8_0", "q4_0"} specifically — NOT any file
# that happens to mention both strings. An earlier version of this script was
# too aggressive and broke return statements, case bodies, and test data in
# five files. We now match only Go slice literals.
#
# Idempotent: reruns are no-ops if the slice already contains "tq4p_d128".
#
# Usage: patch_ollama_kv_types.sh <ollama-source-dir>

set -euo pipefail

OLLAMA_DIR="${1:?usage: patch_ollama_kv_types.sh <ollama-source-dir>}"
NEW_TYPES='"tq3_0", "tq4p_d128", "tq4p_d256"'
MARKER='tq4p_d128'

# Find Go files containing a []string{ ... "q8_0" ... "q4_0" ... } literal.
# Pre-filter by grep for cheap elimination; full match is done in Python.
mapfile -t CANDIDATES < <(
    grep -rlE --include='*.go' '\[\]string\s*\{' "$OLLAMA_DIR" \
        | xargs grep -l '"q8_0"' 2>/dev/null \
        | xargs grep -l '"q4_0"' 2>/dev/null \
        || true
)

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "ERROR: no Go file with a []string{...} literal containing q8_0 and q4_0." >&2
    echo "The ollama KV cache allowlist has moved; inspect and patch manually." >&2
    exit 1
fi

patched_any=0
for f in "${CANDIDATES[@]}"; do
    python3 - "$f" "$NEW_TYPES" "$MARKER" <<'PY'
import re, sys, pathlib
path = pathlib.Path(sys.argv[1])
new_types = sys.argv[2]
marker = sys.argv[3]
text = path.read_text()

# Strictly match a Go []string{...} literal whose body is only string literals
# separated by commas (no function calls, no identifiers, no filepath.Join).
# This rules out manifest_test.go's []string{filepath.Join(..., "q4_0"), ...}
# where q8_0/q4_0 are nested inside function calls, not direct elements.
pat = re.compile(
    r'(\[\]string\s*\{\s*)'          # opener
    r'("(?:[^"\\]|\\.)*"'            # first string
    r'(?:\s*,\s*"(?:[^"\\]|\\.)*")*' # ,"string" repeats
    r')'
    r'(\s*,?\s*\})',                 # optional trailing comma + closer
    re.DOTALL,
)

def repl(m):
    inner = m.group(2)
    if '"q8_0"' not in inner or '"q4_0"' not in inner:
        return m.group(0)
    if marker in inner:
        return m.group(0)
    return m.group(1) + inner + ', ' + new_types + m.group(3)

new, n = pat.subn(repl, text)
if new != text:
    path.write_text(new)
    print(f"[+] patched: {path}")
elif marker in text:
    print(f"[=] already patched: {path}")
PY
    if [[ $? -eq 0 ]]; then
        patched_any=1
    fi
done

if [[ $patched_any -eq 0 ]]; then
    echo "ERROR: no file was patched. Inspect CANDIDATES and the pattern above." >&2
    printf '  %s\n' "${CANDIDATES[@]}" >&2
    exit 1
fi
