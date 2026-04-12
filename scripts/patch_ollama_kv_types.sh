#!/usr/bin/env bash
# Widen ollama's KV-cache-type allowlist to include tq3_0.
#
# ollama validates OLLAMA_KV_CACHE_TYPE against a small set (f16, q8_0, q4_0)
# and rejects anything else before it ever reaches llama.cpp. This script
# locates the check in the Go source and appends "tq3_0" to the allowlist.
#
# Designed to be idempotent and resilient: if the allowlist has moved between
# ollama versions, it prints the candidate files and exits non-zero so the
# build script surfaces a clear error rather than silently producing a broken
# binary.
#
# Usage: patch_ollama_kv_types.sh <ollama-source-dir>

set -euo pipefail

OLLAMA_DIR="${1:?usage: patch_ollama_kv_types.sh <ollama-source-dir>}"
MARKER="// turboquant: tq3_0 allowlisted"

# Find Go files that mention both "q8_0" and "q4_0" near each other -- this is
# the validator signature across ollama versions.
mapfile -t CANDIDATES < <(
    grep -rl --include='*.go' '"q8_0"' "$OLLAMA_DIR" | \
        xargs grep -l '"q4_0"' 2>/dev/null || true
)

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "ERROR: could not find ollama KV cache type allowlist." >&2
    echo "Looked for a .go file containing both \"q8_0\" and \"q4_0\"." >&2
    echo "Manually add \"tq3_0\" to the validator and rerun build with --rebuild." >&2
    exit 1
fi

patched=0
for f in "${CANDIDATES[@]}"; do
    if grep -q "$MARKER" "$f"; then
        echo "[=] already patched: $f"
        patched=1
        continue
    fi

    # Common patterns we handle:
    #   "f16", "q8_0", "q4_0"
    #   case "f16", "q8_0", "q4_0":
    # Conservative: append ", \"tq3_0\"" after the first occurrence of "q4_0"
    # in the file. Keeps a comment marker so reruns are idempotent.
    if grep -qE '"q4_0"' "$f"; then
        python3 - "$f" "$MARKER" <<'PY'
import re, sys, pathlib
path = pathlib.Path(sys.argv[1])
marker = sys.argv[2]
text = path.read_text()
# Insert tq3_0 after the first "q4_0" literal that isn't already followed by tq3_0.
def repl(m):
    return m.group(0) + ', "tq3_0"'
new, n = re.subn(r'"q4_0"(?!, "tq3_0")', repl, text, count=1)
if n == 0:
    sys.exit("no q4_0 literal replaced in " + str(path))
# Add marker comment at end of file so reruns are idempotent.
if marker not in new:
    new = new.rstrip() + "\n\n" + marker + "\n"
path.write_text(new)
print(f"[+] patched: {path}")
PY
        patched=1
    fi
done

if [[ $patched -eq 0 ]]; then
    echo "ERROR: found candidate files but no q4_0 literal to patch:" >&2
    printf '  %s\n' "${CANDIDATES[@]}" >&2
    exit 1
fi
