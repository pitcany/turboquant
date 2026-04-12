#!/usr/bin/env bash
# Apply the 4 TQ4P hook edits to a ggml source tree. Additive only.
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
    "        .from_float_ref           = (ggml_from_float_t) ggml_quantize_row_tq4p_d128,\n"
    "    },\n"
    "    [GGML_TYPE_TQ4P_D256] = {\n"
    "        .type_name                = \"tq4p_d256\",\n"
    "        .blck_size                = QK_TQ4P_D256,\n"
    "        .type_size                = sizeof(block_tq4p_d256),\n"
    "        .is_quantized             = true,\n"
    "        .to_float                 = (ggml_to_float_t) ggml_dequantize_row_tq4p_d256,\n"
    "        .from_float_ref           = (ggml_from_float_t) ggml_quantize_row_tq4p_d256,\n"
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
    "        .from_float               = (ggml_from_float_t) ggml_quantize_row_tq4p_d128,\n"
    "        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_tq4p_d128_f32,\n"
    "        .vec_dot_type             = GGML_TYPE_F32,\n"
    "        .nrows                    = 1,\n"
    "    },\n"
    "    [GGML_TYPE_TQ4P_D256] = {\n"
    "        .from_float               = (ggml_from_float_t) ggml_quantize_row_tq4p_d256,\n"
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

echo "All hooks applied."
