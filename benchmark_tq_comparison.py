#!/usr/bin/env python3
"""
Benchmark TurboQuant vs standard vLLM inference.

Cycles through model configurations, restarts vLLM for each, and measures
token throughput. Requires sudo for systemctl — the script will pause and
prompt you to restart the service when needed.

Usage:
    python3 benchmark_tq_comparison.py [--requests N] [--max-tokens N] [--warmup N]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

VLLM_URL = "http://localhost:8003/v1"
ENV_FILE = Path(os.environ.get(
    "VLLM_ENV_FILE",
    "/home/yannik/ai/local-llm-stack/services/vllm-tp.env",
))
RESULTS_FILE = Path(__file__).parent / "benchmark_tq_results.tsv"

PROMPTS = [
    "Say hello.",
    "Explain the theory of general relativity in detail.",
    "Write a Python function that implements a binary search tree with insert, delete, and search operations. Include docstrings and type hints.",
]


@dataclass
class Config:
    name: str
    hf_model: str
    served_name: str
    quantization: str
    max_model_len: int
    extra_args: str
    turboquant: bool
    hybrid: bool = False


# Only difference between standard/TQ/hybrid: --attention-backend and TQ_HYBRID
# Everything else (quantization, enforce-eager, context) is held constant.
CONFIGS = [
    # ── Llama 3.3 70B: find max standard context ────────────────────
    Config(
        name="llama70b-standard-8k",
        hf_model="casperhansen/llama-3.3-70b-instruct-awq",
        served_name="llama-3.3-70b-bench",
        quantization="awq_marlin",
        max_model_len=8192,
        extra_args="--enforce-eager",
        turboquant=False,
    ),
    Config(
        name="llama70b-standard-12k",
        hf_model="casperhansen/llama-3.3-70b-instruct-awq",
        served_name="llama-3.3-70b-bench",
        quantization="awq_marlin",
        max_model_len=12288,
        extra_args="--enforce-eager",
        turboquant=False,
    ),
    Config(
        name="llama70b-standard-16k",
        hf_model="casperhansen/llama-3.3-70b-instruct-awq",
        served_name="llama-3.3-70b-bench",
        quantization="awq_marlin",
        max_model_len=16384,
        extra_args="--enforce-eager",
        turboquant=False,
    ),
    Config(
        name="llama70b-standard-32k",
        hf_model="casperhansen/llama-3.3-70b-instruct-awq",
        served_name="llama-3.3-70b-bench",
        quantization="awq_marlin",
        max_model_len=32768,
        extra_args="--enforce-eager",
        turboquant=False,
    ),
    # ── Llama 3.3 70B: TQ and hybrid at 32K ─────────────────────────
    Config(
        name="llama70b-turboquant-32k",
        hf_model="casperhansen/llama-3.3-70b-instruct-awq",
        served_name="llama-3.3-70b-bench",
        quantization="awq_marlin",
        max_model_len=32768,
        extra_args="--enforce-eager --attention-backend CUSTOM",
        turboquant=True,
    ),
    Config(
        name="llama70b-hybrid-32k",
        hf_model="casperhansen/llama-3.3-70b-instruct-awq",
        served_name="llama-3.3-70b-bench",
        quantization="awq_marlin",
        max_model_len=32768,
        extra_args="--enforce-eager --attention-backend CUSTOM",
        turboquant=True,
        hybrid=True,
    ),
    # ── Qwen 2.5 72B: find max standard context ─────────────────────
    Config(
        name="qwen2.5-72b-standard-8k",
        hf_model="Qwen/Qwen2.5-72B-Instruct-AWQ",
        served_name="qwen2.5-72b-bench",
        quantization="awq_marlin",
        max_model_len=8192,
        extra_args="--enforce-eager",
        turboquant=False,
    ),
    Config(
        name="qwen2.5-72b-standard-12k",
        hf_model="Qwen/Qwen2.5-72B-Instruct-AWQ",
        served_name="qwen2.5-72b-bench",
        quantization="awq_marlin",
        max_model_len=12288,
        extra_args="--enforce-eager",
        turboquant=False,
    ),
    Config(
        name="qwen2.5-72b-standard-16k",
        hf_model="Qwen/Qwen2.5-72B-Instruct-AWQ",
        served_name="qwen2.5-72b-bench",
        quantization="awq_marlin",
        max_model_len=16384,
        extra_args="--enforce-eager",
        turboquant=False,
    ),
    # ── Qwen 2.5 72B: TQ and hybrid at 16K ──────────────────────────
    Config(
        name="qwen2.5-72b-turboquant-16k",
        hf_model="Qwen/Qwen2.5-72B-Instruct-AWQ",
        served_name="qwen2.5-72b-bench",
        quantization="awq_marlin",
        max_model_len=16384,
        extra_args="--enforce-eager --attention-backend CUSTOM",
        turboquant=True,
    ),
    Config(
        name="qwen2.5-72b-hybrid-16k",
        hf_model="Qwen/Qwen2.5-72B-Instruct-AWQ",
        served_name="qwen2.5-72b-bench",
        quantization="awq_marlin",
        max_model_len=16384,
        extra_args="--enforce-eager --attention-backend CUSTOM",
        turboquant=True,
        hybrid=True,
    ),
]


@dataclass
class BenchResult:
    config_name: str
    prompt_label: str
    prompt_tokens: int
    completion_tokens: int
    wall_time_s: float
    tokens_per_sec: float


def update_env_file(config: Config) -> None:
    lines = ENV_FILE.read_text().splitlines()
    updates = {
        "VLLM_TP_MODEL": config.hf_model,
        "VLLM_TP_SERVED_NAME": config.served_name,
        "VLLM_TP_QUANTIZATION": config.quantization,
        "VLLM_TP_MAX_MODEL_LEN": str(config.max_model_len),
        "VLLM_TP_EXTRA_ARGS": config.extra_args,
        "TQ_HYBRID": "1" if config.hybrid else "0",
    }
    new_lines = []
    seen_keys = set()
    for line in lines:
        key = line.split("=", 1)[0] if "=" in line else None
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            seen_keys.add(key)
        else:
            new_lines.append(line)
    for key, val in updates.items():
        if key not in seen_keys:
            new_lines.append(f"{key}={val}")
    ENV_FILE.write_text("\n".join(new_lines) + "\n")


def wait_for_vllm(timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{VLLM_URL}/models", timeout=3)
            if r.ok:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    return False


def run_request(model: str, prompt: str, max_tokens: int) -> dict:
    r = requests.post(
        f"{VLLM_URL}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def benchmark_config(
    config: Config, num_requests: int, max_tokens: int, warmup: int
) -> list[BenchResult]:
    results = []
    prompt_labels = ["short", "medium", "long"]

    for prompt, label in zip(PROMPTS, prompt_labels):
        # Warmup
        for _ in range(warmup):
            run_request(config.served_name, prompt, max_tokens)

        times = []
        completions = []
        prompt_toks = []
        for _ in range(num_requests):
            start = time.perf_counter()
            data = run_request(config.served_name, prompt, max_tokens)
            elapsed = time.perf_counter() - start
            usage = data["usage"]
            times.append(elapsed)
            completions.append(usage["completion_tokens"])
            prompt_toks.append(usage["prompt_tokens"])

        avg_time = sum(times) / len(times)
        avg_comp = sum(completions) / len(completions)
        avg_prompt = sum(prompt_toks) / len(prompt_toks)
        tps = avg_comp / avg_time

        results.append(BenchResult(
            config_name=config.name,
            prompt_label=label,
            prompt_tokens=int(avg_prompt),
            completion_tokens=int(avg_comp),
            wall_time_s=round(avg_time, 2),
            tokens_per_sec=round(tps, 1),
        ))
        print(f"    {label:8s}: {avg_comp:.0f} tok in {avg_time:.2f}s = {tps:.1f} t/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant vs standard vLLM")
    parser.add_argument("--requests", type=int, default=3, help="Requests per prompt (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per request (default: 256)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests per prompt (default: 1)")
    parser.add_argument("--configs", nargs="*", help="Run only these configs (by name)")
    args = parser.parse_args()

    configs = CONFIGS
    if args.configs:
        configs = [c for c in CONFIGS if c.name in args.configs]
        if not configs:
            print(f"No matching configs. Available: {[c.name for c in CONFIGS]}")
            sys.exit(1)

    all_results: list[BenchResult] = []

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] {config.name}")
        print(f"  Model:   {config.hf_model}")
        print(f"  Quant:   {config.quantization}")
        print(f"  Context: {config.max_model_len}")
        print(f"  TQ:      {config.turboquant}")
        print(f"  Hybrid:  {config.hybrid}")
        print(f"{'='*60}")

        # Update env file
        update_env_file(config)
        print("  Env file updated.")

        # Prompt user to restart
        print("\n  >>> Please restart vLLM:")
        print("  >>> sudo systemctl restart vllm-tp")
        input("  >>> Press Enter after running the command...")

        # Wait for vLLM
        print("  Waiting for vLLM to be ready...")
        if not wait_for_vllm(timeout=300):
            print(f"  ERROR: vLLM did not start for {config.name}. Skipping.")
            continue

        # Verify correct model
        r = requests.get(f"{VLLM_URL}/models")
        loaded = r.json()["data"][0]["id"]
        print(f"  Model loaded: {loaded}")

        # Run benchmark
        print(f"  Benchmarking ({args.requests} requests x {args.max_tokens} max tokens)...")
        results = benchmark_config(config, args.requests, args.max_tokens, args.warmup)
        all_results.extend(results)

    # Write results
    if all_results:
        with open(RESULTS_FILE, "w") as f:
            f.write("config\tprompt\tprompt_tokens\tcompletion_tokens\twall_time_s\ttokens_per_sec\n")
            for r in all_results:
                f.write(f"{r.config_name}\t{r.prompt_label}\t{r.prompt_tokens}\t{r.completion_tokens}\t{r.wall_time_s}\t{r.tokens_per_sec}\n")
        print(f"\nResults saved to {RESULTS_FILE}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<30s} {'Prompt':<8s} {'Comp tok':>8s} {'Time':>7s} {'t/s':>7s}")
    print("-" * 62)
    for r in all_results:
        print(f"{r.config_name:<30s} {r.prompt_label:<8s} {r.completion_tokens:>8d} {r.wall_time_s:>6.2f}s {r.tokens_per_sec:>6.1f}")


if __name__ == "__main__":
    main()
