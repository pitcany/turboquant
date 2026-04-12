```markdown
# turboquant Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you how to contribute to the `turboquant` repository, a Python-based project focused on developing, patching, and validating quantization algorithms (e.g., Stage-2 QJL) for efficient model inference, especially in the context of integrating with projects like `llama.cpp`. You'll learn the repository's coding conventions, how to add or update quantization patches, and how to integrate and validate those patches in forked codebases using automated scripts and rigorous testing.

## Coding Conventions

### File Naming

- Use **snake_case** for Python files and scripts.
  - Example: `generate_constants.py`, `build_ollama_tq.sh`
- C source and header files also follow snake_case.
  - Example: `ggml-tq-paper.c`, `ggml-tq-paper.h`
- Test files include `.test.` in their names.
  - Example: `test_c_vs_python.py`

### Import Style

- Use **relative imports** within Python modules.
  ```python
  from .utils import quantize_weights
  ```

### Export Style

- Use **named exports** (explicit function and class definitions).
  ```python
  def generate_constants(...):
      ...
  ```

### Commit Patterns

- Commit messages are freeform, typically around 55 characters.
  - No strict prefixing required.

## Workflows

### Add or Update Quantization Patch

**Trigger:** When you want to add or update a quantization method for model inference.  
**Command:** `/new-quantization-patch`

1. **Draft or update documentation**  
   - Create or edit `PLAN.md` and `README.md` in the patch directory (e.g., `patches/stage2-qjl/`) to describe the quantization algorithm and validation plan.
2. **Implement or modify C reference code**  
   - Write or update the C source (`ggml-tq-paper.c`) and header (`ggml-tq-paper.h`) files for the quantization algorithm.
3. **Develop Python oracle and tests**  
   - Implement or update Python scripts for reference/oracle logic and byte-exact validation (e.g., `test_c_vs_python.py`).
4. **Generate constants**  
   - Use Python scripts (e.g., `generate_constants.py`) to generate constants and export them as C headers (`tqp_constants_*.h`).
   ```python
   # Example: generate_constants.py
   import numpy as np

   def export_constants():
       constants = np.array([...])
       with open('tqp_constants_example.h', 'w') as f:
           f.write("// Auto-generated constants\n")
           for i, val in enumerate(constants):
               f.write(f"#define CONST_{i} {val}\n")
   ```
5. **Update integration scripts**  
   - Edit or add scripts like `build_ollama_tq.sh` to automate patch application and building.
6. **Document required codebase edits**  
   - Update `hooks.md` to list manual or automated changes needed for upstream/forked code.
7. **Add or update test scripts**  
   - Ensure test scripts validate correctness and integration.
8. **Extend patch scripts**  
   - Update scripts like `patch_ollama_kv_types.sh` to allowlist new quantization types.

### Integration and Validation of Patch

**Trigger:** When you need to ensure a new or updated patch is correctly integrated and functional in the target codebase.  
**Command:** `/integrate-patch`

1. **Review integration points**  
   - Cross-read the upstream/forked codebase to identify where and how to integrate the patch.
2. **Fix bugs or mismatches**  
   - Address any issues found during integration.
3. **Document codebase edits**  
   - Update or create `hooks.md` to describe all required changes.
4. **Automate codebase edits**  
   - Use scripts like `apply_hooks.sh` to automate edits using context anchors for idempotency.
   ```bash
   # Example: apply_hooks.sh
   patch -p1 < hooks.patch
   ```
5. **Update patch scripts**  
   - Extend scripts such as `patch_ollama_kv_types.sh` to allowlist new types.
6. **Add or update smoke/integration tests**  
   - Implement or update scripts like `smoke_test_tq4p.sh` for end-to-end validation.
7. **Run and document tests**  
   - Execute all relevant tests and record the results.

## Testing Patterns

- Test files follow the `*.test.*` naming pattern (e.g., `test_c_vs_python.py`).
- The testing framework is not explicitly defined; tests are typically run as standalone scripts.
- Tests compare C and Python implementations for byte-exact results.
  ```python
  # Example: test_c_vs_python.py
  from .c_reference import quantize_c
  from .python_reference import quantize_py

  def test_byte_exact():
      input_data = ...
      assert quantize_c(input_data) == quantize_py(input_data)
  ```

## Commands

| Command                  | Purpose                                                      |
|--------------------------|--------------------------------------------------------------|
| /new-quantization-patch  | Start the workflow to add or update a quantization patch     |
| /integrate-patch         | Begin integration and validation of a patch in target codebase|
```