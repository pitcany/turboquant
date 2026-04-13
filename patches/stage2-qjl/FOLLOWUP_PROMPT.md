You're picking up the turboquant TQ4P follow-up work. pitcany/turboquant#8
just merged to main. Five stacked commits now live:
  19fdef9 CPU Haar→WHT swap
  0b5fc65 CUDA WHT port
  4642569 README update
  8c928d8 Per-layer σ + 69/133 B block struct alignment
  9280770 Stale-banner cleanup
  e4a91a0 User-selectable rotation mode (WHT default, Haar opt-in)

Current state on main:
- Blocks are 69 B (d128) / 133 B (d256); layer_idx byte packs
  `(rotation << 7) | (layer & 0x1f)`. TQP_ROT_WHT=0, TQP_ROT_HAAR=1.
- Haar mode is byte-identical to turboquant.py::TurboQuantProd(seed=42+i).
- 246 pytest cases pass on CPU:
    cd patches/stage2-qjl/c && gcc -O2 -Wall -fPIC -shared -o libggml_tq_paper.so ggml-tq-paper.c -lm
    cd ../python && python3 -m pytest test_tq_paper.py test_c_vs_python.py -q
- Ollama wiring exists: scripts/build_ollama_tq.sh, apply_hooks.sh (5 hooks),
  apply_go_plumbing.sh, scripts/patch_ollama_kv_types.sh, scripts/smoke_test_tq4p.sh.

Environment you have: a working GPU (4090 or 5090 class, nvcc on PATH) and
ollama installed. test_cuda_vs_cpu.py should actually run on this machine.
If `nvcc --version` fails or there's no `ollama` on PATH, stop and report —
don't fake it.

Work through the follow-ups below in order. One PR per item unless two are
genuinely coupled. After every commit, run the CPU suite above as a floor
(246 pass). After CUDA changes, also run:
    cd patches/stage2-qjl/cuda && cmake -S . -B build && cmake --build build -j
    cd ../python && python3 -m pytest test_cuda_vs_cpu.py -q
If CUDA tests don't pass, fix before moving on.

Don't touch main directly. Don't merge your own PRs. No emoji, no marketing
copy, no files you didn't have to touch.

====================================================================
(1) Baseline validation on real hardware — DO THIS FIRST
====================================================================

Before writing anything new, confirm the already-merged stack actually
works on GPU:

1. Build the CUDA shared lib (cmake commands above). If it fails to
   compile, STOP and fix — the WHT/Haar CUDA code has never been
   compile-tested. Common suspects: bank-conflict-adjacent WHT butterfly
   access patterns, the offsetof() in ggml_cuda_op_tqp_vec_dot, the
   #pragma pack around the block structs.

2. Run test_cuda_vs_cpu.py with both d=128 and d=256 parametrizations,
   all 5 test layers, both rotations. All byte-exact and numerical
   agreement tests should pass. If any fail at the byte level, that's a
   real bug — investigate the kernel that produced the diverging output.

3. Build ollama via scripts/build_ollama_tq.sh with CUDA=1, run
   scripts/smoke_test_tq4p.sh end-to-end with
   OLLAMA_KV_CACHE_TYPE=tq4p_d128 (WHT, default) and confirm a short
   generation produces coherent output vs. fp16 KV baseline.

4. Write up results in docs/TQ4P_GPU_VALIDATION.md — one page, numbers
   and commit hash, no narrative. Include any bugs found + their fixes.

PR title: "TQ4P on-GPU validation runbook + fixes".

Only proceed past this step if (1) is green. If anything diverges, every
other item becomes suspect — fix first, then continue.

====================================================================
(4) Lloyd-Max codebook sanity on real activations
====================================================================

Question: does the Gaussian-approx 3-bit codebook still hold for
realistic (non-isotropic) LLM activations, under both rotations?

Deliverable: scripts/lloyd_max_sanity.py

Phase A (synthetic, no model weights):
  - Build a few non-isotropic distributions: iid N(0,1/d) baseline;
    "outlier channel" (4 channels with ~10× variance); heavy-tailed
    (Student-t, df=3, scaled).
  - Sample 10k vectors at d=128 and d=256, quantize via TurboQuantMSE
    at 3 bits for both rotations (Haar via turboquant.py, WHT via
    tq_paper_reference.rht_apply).
  - Report: post-rotation per-coord distribution stats, Stage-1 MSE
    per vector, ratio of measured MSE to the Gaussian-approx bound.
  - Pass if MSE stays within ~1.5× of the bound for both rotations.

Phase B (real activations, use the local GPU):
  - Load Qwen2.5-7B or Llama-3.1-8B via transformers.
  - Run 8–16 prompts; hook attention to dump K and V from 3 layers
    (first, middle, last).
  - Run the same MSE analysis. Same pass/fail threshold.
  - Commit the script and a short results table in a markdown alongside.

If Phase A or B shows MSE > 1.5× bound for a rotation on any
distribution, propose (do not immediately implement) a re-solved
Lloyd-Max using empirical marginals — open an issue instead of a PR for
that follow-up decision.

PR title: "Add Lloyd-Max codebook sanity check for TQ4P".

====================================================================
(2) CUDA quantize/dequantize dispatch hook
====================================================================

Without this, fp32 → TQ4P K-cache writes on GPU fall back to D→H→CPU
quantize→H→D. Slow but correct.

Goal: new hook 6 in apply_hooks.sh that registers
ggml_cuda_tqp_quantize_row_d* into ollama's ggml-cuda cpy/quantize
dispatch so K-cache writes happen on-device.

Before writing: locate the ollama source tree. scripts/build_ollama_tq.sh
should have left a clone somewhere (check the OLLAMA_DIR variable in
that script). If not, git clone github.com/ollama/ollama into /tmp,
pin the commit, and record the pinned hash in hooks.md so the anchor
can be refreshed later when ollama drifts.

Acceptance:
  - apply_hooks.sh hook 6 is idempotent (guard with "tqp" marker).
  - Full scripts/build_ollama_tq.sh with CUDA=1 produces a binary whose
    K-cache writes happen on-device. Verify by extending
    scripts/smoke_test_tq4p.sh to time a prefill and compare against
    pre-hook-6 timing (document the speedup in the PR).

PR title: "Add on-device TQ4P quantize dispatch to ggml-cuda".

====================================================================
(3) Q8_K query path on CUDA
====================================================================

Mirror the existing CPU `ggml_vec_dot_tq4p_d*_q8k` in CUDA.

Deliverable:
  - New Q8_K-aware entry point in tqp-vec-dot.cu that dequantizes one
    Q8_K block worth of query (256 int8 × scale) on device in a tiny
    kernel, then feeds the resulting fp32 slice into the existing
    prepare_query_batch + vec_dot_ggml_kernel pipeline.
  - Extend ggml_cuda_op_tqp_vec_dot to branch on src1->type: Q8_K goes
    through the new path, fp32 through the existing one.
  - Update apply_hooks.sh hook 5 only if the existing early-return
    doesn't cover the Q8_K src1 case — if it does, leave it and note why.

Tests:
  - Extend test_cuda_vs_cpu.py with test_q8k_cuda_vs_cpu that reuses
    the CPU Q8_K quantization helper from test_c_vs_python.py. Must
    actually run here (GPU available).

PR title: "Add Q8_K query path to TQ4P CUDA vec_dot".

====================================================================
(5) Perf polish in CUDA kernels — profile-gated
====================================================================

Only after (1) + (2) + (3) land. Don't speculate.

Steps:
  - Run nsight-compute on scripts/smoke_test_tq4p.sh's inner loop and
    capture a report. Check: is the ‖x‖ + ‖residual‖ thread-0-serial
    reduction in tqp-quantize.cu on the critical path? Does the FWHT
    butterfly hit shared-memory bank conflicts at h ≥ 32?
  - If yes to either, implement fixes: warp-shuffle reduction for the
    norms; skewed or hybrid butterfly for the FWHT.
  - If both profile as noise relative to S·residual, close (5) as
    not-applicable with a one-paragraph note in cuda/PLAN.md.

PR title: "Optimize TQ4P CUDA quantize hotspots" (only if there are any).

====================================================================
Start by running the CPU suite to establish a baseline, then jump
straight into (1). Anything that fails there outranks everything below.
