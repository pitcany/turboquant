---
name: add-or-update-quantization-patch
description: Workflow command scaffold for add-or-update-quantization-patch in turboquant.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-or-update-quantization-patch

Use this workflow when working on **add-or-update-quantization-patch** in `turboquant`.

## Goal

Implements or updates a quantization algorithm patch (e.g., Stage-2 QJL) for integration with a llama.cpp fork, including C reference, Python oracle, constants generation, and integration scripts.

## Common Files

- `patches/stage2-qjl/PLAN.md`
- `patches/stage2-qjl/README.md`
- `patches/stage2-qjl/hooks.md`
- `patches/stage2-qjl/apply_hooks.sh`
- `patches/stage2-qjl/c/ggml-tq-paper.c`
- `patches/stage2-qjl/c/ggml-tq-paper.h`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Draft or update a plan/README documenting the quantization algorithm and validation strategy.
- Implement or modify C reference code and headers for the quantization algorithm.
- Implement or update Python reference/oracle and test scripts for byte-exact validation.
- Generate or update constants using Python scripts and export to C headers.
- Update or add integration scripts (e.g., build_ollama_tq.sh) to automate patch application and build.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.