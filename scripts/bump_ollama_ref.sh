#!/usr/bin/env bash
# Bump the pinned OLLAMA_REF to a new upstream commit.
#
# Usage:
#   scripts/bump_ollama_ref.sh                # bump to latest ollama main
#   scripts/bump_ollama_ref.sh abc1234        # bump to specific commit
#
# What it does:
#   1. Fetches the target ref from upstream ollama
#   2. Updates OLLAMA_REF in build_ollama_tq.sh
#   3. Applies hooks to verify anchors still match
#   4. Builds the full binary (CPU + CUDA + Go)
#   5. Runs the smoke test
#   6. Creates a branch + commit (you PR it)
#
# If any step fails, the script stops and tells you what broke.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BUILD_SCRIPT="$REPO_ROOT/scripts/build_ollama_tq.sh"
SMOKE_SCRIPT="$REPO_ROOT/scripts/smoke_test_tq4p.sh"
WORKDIR="${WORKDIR:-$HOME/.local/src/ollama-tq}"
OLLAMA_DIR="$WORKDIR/ollama"

# ── 0. Parse target ref ──────────────────────────────────────────────

TARGET_REF="${1:-}"

if [[ -z "$TARGET_REF" ]]; then
    echo "[+] fetching latest ollama main..."
    git -C "$OLLAMA_DIR" fetch origin main --quiet
    TARGET_REF=$(git -C "$OLLAMA_DIR" rev-parse origin/main)
    echo "    latest: $TARGET_REF"
else
    echo "[+] target ref: $TARGET_REF"
    git -C "$OLLAMA_DIR" fetch origin --quiet
fi

SHORT_REF="${TARGET_REF:0:8}"

# Read current ref
CURRENT_REF=$(grep 'OLLAMA_REF:-' "$BUILD_SCRIPT" | grep -oP ':-\K[^}"]+' | head -1)
echo "    current: ${CURRENT_REF:-<unpinned>}"

if [[ "$SHORT_REF" == "$CURRENT_REF" ]]; then
    echo "[=] already at $SHORT_REF, nothing to do"
    exit 0
fi

# ── 1. Update the ref in the build script ────────────────────────────

echo "[+] updating OLLAMA_REF: $CURRENT_REF → $SHORT_REF"
sed -i "s|OLLAMA_REF:-${CURRENT_REF}|OLLAMA_REF:-${SHORT_REF}|" "$BUILD_SCRIPT"

# ── 2. Checkout the target ref in the ollama tree ────────────────────

echo "[+] checking out $SHORT_REF in ollama tree..."
git -C "$OLLAMA_DIR" checkout "$TARGET_REF" --quiet --detach

# ── 3. Full build (copies files, applies hooks, builds CPU+CUDA+Go) ──

echo "[+] running full build..."
if ! bash "$BUILD_SCRIPT" --rebuild 2>&1 | tee /tmp/bump-build.log | tail -5; then
    echo
    echo "BUILD FAILED. Check /tmp/bump-build.log"
    echo "Common causes:"
    echo "  - Hook anchor text changed in new ggml (grep for 'anchor not found')"
    echo "  - New ggml API breaks our header (grep for 'error:')"
    echo
    echo "Reverting OLLAMA_REF..."
    sed -i "s|OLLAMA_REF:-${SHORT_REF}|OLLAMA_REF:-${CURRENT_REF}|" "$BUILD_SCRIPT"
    git -C "$OLLAMA_DIR" checkout "${CURRENT_REF}" --quiet --detach 2>/dev/null || true
    exit 1
fi

# ── 4. Smoke test ────────────────────────────────────────────────────

echo "[+] running smoke test..."
pkill -x ollama 2>/dev/null || true
sleep 3

if ! bash "$SMOKE_SCRIPT" 2>&1 | tee /tmp/bump-smoke.log | tail -10; then
    echo
    echo "SMOKE TEST FAILED. Check /tmp/bump-smoke.log"
    echo "Reverting OLLAMA_REF..."
    sed -i "s|OLLAMA_REF:-${SHORT_REF}|OLLAMA_REF:-${CURRENT_REF}|" "$BUILD_SCRIPT"
    exit 1
fi

systemctl start ollama 2>/dev/null || true

# ── 5. Describe the upstream changes ─────────────────────────────────

echo
echo "── Upstream changes ($CURRENT_REF..$SHORT_REF) ──"
git -C "$OLLAMA_DIR" log --oneline "${CURRENT_REF}..${TARGET_REF}" 2>/dev/null | head -20

# ── 6. Create branch + commit ────────────────────────────────────────

BRANCH="chore/bump-ollama-${SHORT_REF}"
cd "$REPO_ROOT"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
git add scripts/build_ollama_tq.sh
git commit -m "chore: bump OLLAMA_REF to $SHORT_REF

Build + smoke test passed. Upstream changes:
$(git -C "$OLLAMA_DIR" log --oneline "${CURRENT_REF}..${TARGET_REF}" 2>/dev/null | head -10)"

echo
echo "Done. Push and PR:"
echo "  git push -u origin $BRANCH"
echo "  gh pr create --title 'chore: bump ollama ref to $SHORT_REF'"
