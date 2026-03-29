"""Benchmark an OpenAI-compatible chat completion endpoint.

Measures end-to-end latency and completion throughput using the server's
reported token usage. Intended for vLLM TurboQuant smoke and perf checks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def make_payload(model: str, prompt: str, max_tokens: int) -> bytes:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    return json.dumps(payload).encode("utf-8")


def issue_request(base_url: str, model: str, prompt: str, max_tokens: int,
                  timeout: float) -> tuple[float, dict]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=make_payload(model, prompt, max_tokens),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - start
    return elapsed, body


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible completions")
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain tail recursion in 3 bullet points.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    print(f"base_url={args.base_url}")
    print(f"model={args.model}")
    print(f"warmup={args.warmup}, requests={args.requests}, max_tokens={args.max_tokens}")

    try:
        for _ in range(args.warmup):
            issue_request(args.base_url, args.model, args.prompt, args.max_tokens, args.timeout)
    except urllib.error.URLError as exc:
        print(f"warmup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    latencies: list[float] = []
    tok_rates: list[float] = []
    completion_tokens_seen: list[int] = []

    for idx in range(args.requests):
        elapsed, body = issue_request(
            args.base_url,
            args.model,
            args.prompt,
            args.max_tokens,
            args.timeout,
        )
        usage = body.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", 0))
        tok_s = (completion_tokens / elapsed) if completion_tokens > 0 and elapsed > 0 else 0.0
        latencies.append(elapsed)
        tok_rates.append(tok_s)
        completion_tokens_seen.append(completion_tokens)
        print(
            f"req={idx + 1} latency={elapsed:.3f}s "
            f"completion_tokens={completion_tokens} tok/s={tok_s:.2f}"
        )

    print("")
    print(f"latency_mean={statistics.mean(latencies):.3f}s")
    print(f"latency_p50={statistics.median(latencies):.3f}s")
    print(f"latency_min={min(latencies):.3f}s")
    print(f"latency_max={max(latencies):.3f}s")
    print(f"completion_tokens_mean={statistics.mean(completion_tokens_seen):.1f}")
    print(f"tok_s_mean={statistics.mean(tok_rates):.2f}")
    print(f"tok_s_max={max(tok_rates):.2f}")


if __name__ == "__main__":
    main()
