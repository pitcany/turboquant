#!/usr/bin/env python3
"""Profile the TP=2 vLLM pipeline to identify non-TQ overhead.

Attaches a torch.profiler to the running model and captures a trace of
decode iterations.  The trace includes NCCL all-reduce, vLLM scheduler,
sampling, and token handling — the ~43ms "base" overhead that micro-
benchmarks miss.

Usage:
    # Start vLLM with profiling hooks enabled:
    TQ_USE_TRITON=1 python scripts/profile_tp2_pipeline.py \
        --model <model_path> \
        --tp 2 \
        --trace-dir ./traces \
        --warmup-iters 5 \
        --profile-iters 10

    # Then view the trace in Chrome:  chrome://tracing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _check_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() >= 1
    except ImportError:
        return False


def run_profile(
    model: str,
    tp: int,
    trace_dir: str,
    warmup_iters: int,
    profile_iters: int,
    max_tokens: int,
    prompt: str,
) -> None:
    """Run profiled inference and save trace."""
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule

    # Lazy import — vLLM must be importable
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vLLM not installed. Install with: pip install vllm",
              file=sys.stderr)
        sys.exit(1)

    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {model} with TP={tp}...")
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        enforce_eager=False,  # Allow CUDAGraph if supported
        gpu_memory_utilization=0.85,
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    # Warmup
    print(f"Warming up ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        llm.generate([prompt], params)

    torch.cuda.synchronize()

    # Profile
    print(f"Profiling ({profile_iters} iterations)...")
    trace_file = trace_path / f"tp{tp}_decode_trace.json"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=1,
            warmup=1,
            active=profile_iters,
            repeat=1,
        ),
        on_trace_ready=lambda p: p.export_chrome_trace(str(trace_file)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(profile_iters + 2):  # +2 for wait+warmup
            llm.generate([prompt], params)
            prof.step()

    torch.cuda.synchronize()

    # Summary
    print(f"\nTrace saved to: {trace_file}")
    print("\n--- Key Averages (top 20 CUDA kernels) ---")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
    )

    # Breakdown by category
    print("\n--- Time Breakdown ---")
    key_avgs = prof.key_averages()
    categories = {
        "TQ kernels": ["_tq_", "turboquant"],
        "NCCL": ["nccl", "ncclKernel", "all_reduce"],
        "GEMM/Linear": ["gemm", "cutlass", "cublas", "sm90"],
        "Attention": ["flash", "sdpa", "attention"],
        "Sampling": ["sample", "argmax", "multinomial"],
        "Memory": ["memcpy", "memset"],
    }

    total_cuda_us = sum(
        e.cuda_time_total for e in key_avgs if e.cuda_time_total > 0
    )

    for cat_name, patterns in categories.items():
        cat_time = sum(
            e.cuda_time_total
            for e in key_avgs
            if any(p in e.key.lower() for p in patterns)
        )
        pct = 100.0 * cat_time / total_cuda_us if total_cuda_us > 0 else 0
        print(f"  {cat_name:20s}: {cat_time / 1000:8.1f} ms  ({pct:5.1f}%)")

    other_time = total_cuda_us - sum(
        sum(
            e.cuda_time_total
            for e in key_avgs
            if any(p in e.key.lower() for p in patterns)
        )
        for patterns in categories.values()
    )
    pct = 100.0 * other_time / total_cuda_us if total_cuda_us > 0 else 0
    print(f"  {'Other':20s}: {other_time / 1000:8.1f} ms  ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total_cuda_us / 1000:8.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile TP=2 vLLM pipeline to identify non-TQ overhead",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model path or HF model ID",
    )
    parser.add_argument(
        "--tp", type=int, default=2,
        help="Tensor parallel size (default: 2)",
    )
    parser.add_argument(
        "--trace-dir", type=str, default="./traces",
        help="Directory for Chrome trace output",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--profile-iters", type=int, default=10,
        help="Number of profiled iterations",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64,
        help="Max tokens to generate per request",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="Explain the theory of general relativity in simple terms.",
        help="Prompt for generation",
    )
    args = parser.parse_args()

    if not _check_cuda():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    run_profile(
        model=args.model,
        tp=args.tp,
        trace_dir=args.trace_dir,
        warmup_iters=args.warmup_iters,
        profile_iters=args.profile_iters,
        max_tokens=args.max_tokens,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
