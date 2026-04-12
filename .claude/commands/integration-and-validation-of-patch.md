---
name: integration-and-validation-of-patch
description: Workflow command scaffold for integration-and-validation-of-patch in turboquant.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /integration-and-validation-of-patch

Use this workflow when working on **integration-and-validation-of-patch** in `turboquant`.

## Goal

Integrates a patch into a forked codebase (e.g., llama.cpp), automates required codebase edits, and validates the integration via smoke/integration tests.

## Common Files

- `patches/stage2-qjl/hooks.md`
- `patches/stage2-qjl/apply_hooks.sh`
- `patches/stage2-qjl/c/ggml-tq-paper.c`
- `patches/stage2-qjl/c/ggml-tq-paper.h`
- `patches/stage2-qjl/python/test_c_vs_python.py`
- `scripts/patch_ollama_kv_types.sh`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Cross-read and verify the upstream/forked codebase to identify integration points and required edits.
- Fix any latent bugs or mismatches found during integration.
- Update or create hooks.md to document all required codebase edits.
- Automate codebase edits with scripts (e.g., apply_hooks.sh) using context anchors for idempotency.
- Update or extend patch scripts (e.g., patch_ollama_kv_types.sh) to allowlist new types.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.