You're continuing the turboquant TQ4P maturation work. Current state of main:
- Full TQ4P stack landed: CPU + CUDA, both rotations, per-layer σ/Π/S,
  69/133 B blocks, Q8_K query on CUDA, on-device CUDA quantize dispatch,
  Lloyd-Max sanity check (passes), on-GPU validation runbook.
- PRs #8, #10, #11, #12, #13, #14 all merged.
- CPU test suite floor (must not regress): 226+ pytest cases across
  patches/stage2-qjl/python/{test_tq_paper,test_c_vs_python,test_cuda_vs_cpu}.py.
- CUDA test suite on hardware: 104+ cases (byte-exact quantize,
  prepare_query, vec_dot, both rotations, all test layers, plus Q8_K).

Environment: GPU (4090/5090 class) with nvcc on PATH, ollama installed,
network available for HF downloads and github clones. If any of those is
missing, stop and report before doing work that depends on it.

Tackle the five items below in order. One PR per item. Branch naming
claude/tq4p-maturity-<item-number>-<slug>. After each commit run:
    cd patches/stage2-qjl/c && gcc -O2 -Wall -fPIC -shared -o libggml_tq_paper.so ggml-tq-paper.c -lm
    cd ../python && python3 -m pytest test_tq_paper.py test_c_vs_python.py -q
    # CUDA:
    cd ../cuda && cmake -S . -B build && cmake --build build -j
    cd ../python && python3 -m pytest test_cuda_vs_cpu.py -q

Don't touch main directly. Don't merge your own PRs. No emoji, no marketing
copy. When blocked, stop and report.

====================================================================
(1) Runtime rotation selector
====================================================================

Today the rotation is fixed per-block at quantize time via bit 7 of
layer_byte. There's no way for an ollama user to pick WHT vs Haar without
calling the C quantize function directly with a different byte. Add a
three-tier runtime knob.

Design (do not redesign — this is settled):

  Precedence: per-call explicit > per-thread > per-process > compile-time (WHT).

  1. Per-call: layer_byte bit 6 = 1 means "bit 7 is authoritative" (today's
     behavior, preserved). bit 6 = 0 means "resolve via defaults below".
     Stored block byte always has bit 6 = 0 and bit 7 = resolved rotation,
     so existing dequantize / vec_dot readers don't change.
  2. Per-thread: `tqp_set_thread_rotation(rot)` sets a thread-local; set
     to -1 to clear. Used by ollama's attention module to pre-declare a
     rotation for a scope of quantize calls.
  3. Per-process: env var OLLAMA_TQP_ROTATION=haar|wht read once at
     library init into a process-global. Fallback when thread-local is
     unset.
  4. Compile-time: TQP_ROT_WHT if nothing else is configured.

Deliverables:

  - patches/stage2-qjl/c/ggml-tq-paper.{h,c}: new API
      void    tqp_set_default_rotation(uint8_t rot);   // 0, 1, or 0xff to clear
      void    tqp_set_thread_rotation(uint8_t rot);    // same
      void    tqp_clear_thread_rotation(void);
      uint8_t tqp_resolve_rotation(uint8_t layer_byte); // apply precedence
    tqp_quantize_block calls tqp_resolve_rotation(layer_byte) before
    indexing σ/Π. Writes resolved rotation into block byte with bit 6
    cleared.

  - First call into the library resolves OLLAMA_TQP_ROTATION via
    pthread_once / call_once. Never re-read at runtime.

  - patches/stage2-qjl/python/tq_paper_reference.py: mirror the same
    resolution logic so tests can exercise it end-to-end.

  - New tests in test_tq_paper.py parametrized on a
    (rotation_source ∈ {explicit, thread, process, compile_time}) axis.
    Verify resolved rotation always matches block byte bit 7 and that
    explicit > thread > process > compile_time ordering holds.

  - Ollama wiring:
    - patches/stage2-qjl/apply_go_plumbing.sh: add a small init hook
      that reads OLLAMA_TQP_ROTATION once at ollama startup and calls
      tqp_set_default_rotation via cgo.
    - Fix the pre-existing bug where apply_hooks.sh hook 3
      (type_traits_cpu.from_float) passes garbage layer_byte via the
      3-arg cast: wrap with a ggml_quantize_row_tq4p_d*_resolved() that
      looks up layer via a thread-local + resolves rotation via our
      resolver. Document what "layer" means for this fallback call site
      (probably layer 0 with resolved rotation).

Acceptance:
  - All 226+ CPU tests pass with no behavior change when no env var /
    thread-local is set.
  - New tests pass across all four rotation_source values.
  - `OLLAMA_TQP_ROTATION=haar scripts/smoke_test_tq4p.sh` produces a
    Haar-quantized K-cache (verify by peeking the first block byte).
  - Per-call with bit 6 set still overrides everything.

PR title: "Runtime rotation selector for TQ4P (env var + thread-local)".

====================================================================
(2) GGUF metadata for rotation (file-level default)
====================================================================

Background: pre-quantized model files currently have no file-level
indicator of which rotation was used. Inspection tooling has to read
a block. Add a GGUF-level KV pair so tools can report rotation at a
glance, and optionally default-override future quantizes for that file.

Deliverables:

  - gguf key: "tq4p.default_rotation" (string: "wht" | "haar"). Write
    it at quantize time; read it at inspection time.

  - Update the writer side: whoever calls our CPU quantize at file-prep
    time (typically `llama-quantize` in llama.cpp / ollama vendored
    tree) needs to set the GGUF KV. Add a small patch to
    apply_hooks.sh hook 7 that instruments the quantize path to write
    the KV. If the writer doesn't have a rotation context (plain
    from_float_ref), skip setting it (leave tooling to read the
    per-block byte).

  - Reader side: `scripts/tq4p_inspect.py` (new) — takes a .gguf path,
    reports the file-level KV if present plus a histogram of per-block
    rotation bits from the first quantized tensor. Useful for
    sanity-checking that a file was built with the expected rotation.

  - Test: round-trip a small synthetic .gguf through the patched
    quantize path and verify the KV is present and matches per-block
    bits.

Not in scope: teaching the runtime to honor the file-level default
(that's already handled by the per-block bit). The KV is advisory.

PR title: "Record TQ4P rotation in GGUF metadata (advisory)".

====================================================================
(4) CI with a GPU runner
====================================================================

Set up .github/workflows/gpu-tests.yml that runs the CUDA test suite
on a self-hosted GPU runner on every PR touching patches/stage2-qjl/
or scripts/smoke_test_tq4p.sh.

Runner: assume the user has already registered a self-hosted runner
with a "cuda" label. If not, stop and report — this is the only
infrastructure step that requires human GH access.

Workflow responsibilities:
  - Build libggml_tq_paper.so (CPU) and libggml_tq_paper_cuda.so (CUDA).
  - pytest test_tq_paper.py test_c_vs_python.py test_cuda_vs_cpu.py.
  - If patches/stage2-qjl/apply_hooks.sh or scripts/build_ollama_tq.sh
    changed, also run scripts/smoke_test_tq4p.sh end-to-end against
    a small model cached on the runner.
  - Post a summary to the PR with test counts + any diffs vs. the
    baseline numbers captured in docs/TQ4P_GPU_VALIDATION.md.

CPU tests continue to run on GH-hosted ubuntu-latest in a separate
workflow (no GPU needed).

Acceptance:
  - Open a no-op PR against main; verify the workflow fires and passes.
  - Open a PR that introduces a known regression (e.g., set a wrong
    stride in a CUDA kernel); verify the workflow catches it.

PR title: "Add self-hosted GPU CI for TQ4P".

====================================================================
(5) Head dims beyond 128 / 256
====================================================================

Right now SUPPORTED_DIMS = (128, 256) is hardcoded in generate_constants.py
and block types are GGML_TYPE_TQ4P_D128 / D256. Make it trivial to add
more head dims (d=64 and d=96 are the common asks; d=512 occurs in a
few larger models).

Deliverables:

  - Parametrize generate_constants.py so it reads SUPPORTED_DIMS from a
    config file or CLI flag. Keep (128, 256) as the default.

  - Add a template macro in patches/stage2-qjl/c/ggml-tq-paper.{h,c}
    and cuda/tqp-*.cu so adding a new dim D is a single
    TQP_DEFINE_D(D) invocation — block struct, quantize row, dequant
    row, vec_dot row, ggml dispatch wrappers, Q8_K wrappers.

  - Done: d=64 support added as a demonstration/common small-head path.
    Regenerate constants, enum values, traits entries, CUDA dispatch,
    and Go plumbing via the hooks.

  - Add test coverage for d=64 in test_tq_paper.py and
    test_c_vs_python.py (parametrize existing tests; they should
    auto-expand).

Acceptance:
  - All existing 128/256 tests continue to pass untouched.
  - d=64 tests pass at the same byte-exact and numerical thresholds.
  - Adding a hypothetical d=96 costs exactly one line in
    SUPPORTED_DIMS + regenerate + rebuild (verify by trying it, then
    revert before the PR).

PR title: "Parametrize TQ4P by head_dim; add GGML_TYPE_TQ4P_D64".

====================================================================
(6) BF16 / FP16 input support
====================================================================

Current quantize entry points accept fp32 only. Most ollama inference
is bf16 at the activation side; the current path silently upcasts
before calling our quantize, costing a tensor copy per K-cache write.

Deliverables:

  - New entry points:
      void ggml_quantize_row_tq4p_d*_bf16(const ggml_bf16_t * x,
                                          block_tq4p_d* * y,
                                          int64_t k, uint8_t layer_byte);
      void ggml_quantize_row_tq4p_d*_f16(const ggml_fp16_t * x,
                                         block_tq4p_d* * y,
                                         int64_t k, uint8_t layer_byte);

    Cast to fp32 at the per-coord load inside tqp_quantize_block.
    Nothing downstream changes (Lloyd-Max / rotations / QJL all stay
    fp32 internally).

  - CUDA mirrors: tqp-quantize.cu grows a __device__ load template
    parametrized on the input dtype. Host wrappers
    ggml_cuda_tqp_quantize_row_d*_{bf16,f16} dispatch. Same pattern
    as the rotation templating that's already in place.

  - Update apply_hooks.sh hook 3 to register from_float_bf16 /
    from_float_f16 in type_traits_cpu alongside from_float.
    Update apply_hooks.sh hook 6 (CUDA cpy dispatch) to intercept
    BF16→TQ4P and F16→TQ4P copies, not just F32→TQ4P.

  - Tests: extend test_c_vs_python.py and test_cuda_vs_cpu.py with
    a dtype axis ∈ {fp32, bf16, f16}. The bf16/f16 paths should match
    fp32 within 3e-3 absolute on vec_dot (bf16 has ~2e-3 input rounding
    error that propagates).

Acceptance:
  - CPU + CUDA tests pass across all three dtypes.
  - smoke_test_tq4p.sh with a bf16 model shows no extra d2h upcast
    copies in `nsys profile`.

PR title: "Add BF16/F16 input support for TQ4P quantize".

====================================================================
Start by running the CPU + CUDA test suites to set a baseline, then
begin (1). Item (4) does not depend on any code changes in the other
items, so if the GPU runner isn't registered yet, pause (4) and carry
on with (1)(2)(5)(6) in order.
