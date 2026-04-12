#!/usr/bin/env bash
# Apply the 4 hook edits to a turbo-tan/llama.cpp-tq3 clone so it compiles
# against the Stage-2 TQ4P types. Idempotent.
#
# Usage: apply_hooks.sh <fork-root>
#
# Looks for context anchors rather than fixed line numbers so small upstream
# drift doesn't break it. If an anchor can't be found or already contains a
# marker, the step is skipped with a note.

set -euo pipefail

FORK="${1:?usage: apply_hooks.sh <fork-root>}"
[[ -d "$FORK/ggml/include" ]] || { echo "not a llama.cpp tree: $FORK" >&2; exit 1; }

GGML_H="$FORK/ggml/include/ggml.h"
GGML_C="$FORK/ggml/src/ggml.c"
CPU_C="$FORK/ggml/src/ggml-cpu/ggml-cpu.c"
CMAKE="$FORK/ggml/src/CMakeLists.txt"

MARKER="tq4p"

if grep -q "$MARKER" "$GGML_H" 2>/dev/null; then
    echo "[=] ggml.h already patched"
else
    echo "[+] ggml.h: adding TQ4P_D128/TQ4P_D256 enum values"
    python3 - "$GGML_H" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()
pattern = re.compile(
    r"(GGML_TYPE_TQ3_4S\s*=\s*46,[^\n]*\n)\s*GGML_TYPE_COUNT\s*=\s*47,",
    re.MULTILINE,
)
m = pattern.search(t)
if not m:
    sys.exit("ggml.h: could not find TQ3_4S + COUNT=47 anchor")
insert = (
    m.group(1)
    + "\n"
    + "        // TurboQuant paper-faithful (Lloyd-Max + QJL). See ggml-tq-paper.h.\n"
    + "        GGML_TYPE_TQ4P_D128 = 47,\n"
    + "        GGML_TYPE_TQ4P_D256 = 48,\n\n"
    + "        GGML_TYPE_COUNT   = 49,"
)
p.write_text(t[:m.start()] + insert + t[m.end():])
PY
fi

if grep -q "$MARKER" "$GGML_C" 2>/dev/null; then
    echo "[=] ggml.c already patched"
else
    echo "[+] ggml.c: #include + type_traits entries"
    python3 - "$GGML_C" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()

# Insert the #include immediately after the #include "ggml-quants.h" line,
# which is always at file scope (not inside an #if). This avoids the
# "last-include-in-first-80-lines" heuristic that falsely matched inside
# an Android conditional.
m = re.search(r'^#include\s+"ggml-quants\.h"\s*\n', t, re.MULTILINE)
if not m:
    sys.exit('ggml.c: could not find #include "ggml-quants.h" anchor')
t = t[:m.end()] + '#include "ggml-tq-paper.h"\n' + t[m.end():]

# Add type_traits entries after TQ3_4S.
anchor = re.search(
    r"(\[GGML_TYPE_TQ3_4S\]\s*=\s*\{[^}]*\},)",
    t,
    re.DOTALL,
)
if not anchor:
    sys.exit("ggml.c: could not find TQ3_4S type_traits entry")
entries = """
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
    },"""
t = t[:anchor.end()] + entries + t[anchor.end():]
p.write_text(t)
PY
fi

if grep -q "$MARKER" "$CPU_C" 2>/dev/null; then
    echo "[=] ggml-cpu.c already patched"
else
    echo "[+] ggml-cpu.c: #include + type_traits_cpu entries"
    python3 - "$CPU_C" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()

# Anchor on #include "ggml-quants.h" at file scope (same reasoning as ggml.c).
m = re.search(r'^#include\s+"ggml-quants\.h"\s*\n', t, re.MULTILINE)
if not m:
    sys.exit('ggml-cpu.c: could not find #include "ggml-quants.h" anchor')
t = t[:m.end()] + '#include "../ggml-tq-paper.h"\n' + t[m.end():]

anchor = re.search(
    r"(\[GGML_TYPE_TQ3_4S\]\s*=\s*\{[^}]*\},)",
    t,
    re.DOTALL,
)
if not anchor:
    sys.exit("ggml-cpu.c: could not find TQ3_4S type_traits_cpu entry")
entries = """
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
    },"""
t = t[:anchor.end()] + entries + t[anchor.end():]
p.write_text(t)
PY
fi

if grep -q "ggml-tq-paper.c" "$CMAKE" 2>/dev/null; then
    echo "[=] CMakeLists.txt already patched"
else
    echo "[+] CMakeLists.txt: adding ggml-tq-paper.c to ggml-base sources"
    python3 - "$CMAKE" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()

# Best-effort: find a line listing ggml-quants.c and insert after it within
# the same source list. Covers both add_library(... FILES ...) and
# target_sources(... PRIVATE FILES ...).
m = re.search(r"(ggml-quants\.c\b)", t)
if not m:
    # Fall back to ggml.c
    m = re.search(r"(ggml\.c\b)(?!pp)", t)
if not m:
    sys.exit("CMakeLists.txt: could not find ggml-quants.c or ggml.c anchor")
line_end = t.find("\n", m.end())
t = t[:line_end] + "\n            ggml-tq-paper.c" + t[line_end:]
p.write_text(t)
PY
fi

echo "All hooks applied."
