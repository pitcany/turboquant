"""Microbenchmark TurboQuant decode paths.

Compares the rotated-domain PyTorch reference path against the Triton decode
wrapper on CUDA. This script is a focused kernel benchmark, not an end-to-end
vLLM throughput benchmark.
"""

from __future__ import annotations

import math
import time

import torch

from turboquant import TurboQuantMSE, TurboQuantProd
from vllm_plugin.attention import _CompressedLayout
from vllm_plugin.triton_wrapper import (
    turboquant_decode_attention,
    turboquant_decode_attention_pytorch,
)


def build_fixture(
    *,
    seq_len: int,
    num_kv_heads: int = 8,
    heads_per_kv: int = 8,
    head_dim: int = 128,
    device: str = "cuda",
):
    torch.manual_seed(11)
    num_heads = num_kv_heads * heads_per_kv
    layout = _CompressedLayout(head_dim, key_mse_bits=2, key_qjl_bits=1,
                               val_mse_bits=3)

    key_q = TurboQuantProd(head_dim, 3, seed=19, device=device)
    val_q = TurboQuantMSE(head_dim, 3, seed=37, device=device)

    keys = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    values = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    queries = torch.randn(1, num_heads, head_dim, device=device, dtype=torch.float16)

    k_norm = torch.norm(keys.float(), dim=-1)
    k_unit = keys.float() / (k_norm.unsqueeze(-1) + 1e-8)
    v_norm = torch.norm(values.float(), dim=-1)
    v_unit = values.float() / (v_norm.unsqueeze(-1) + 1e-8)

    key_comp = key_q.quantize(k_unit.reshape(-1, head_dim))
    val_idx = val_q.quantize(v_unit.reshape(-1, head_dim))
    packed = layout.pack(
        key_comp["mse_indices"],
        key_comp["qjl_signs"],
        key_comp["residual_norm"],
        k_norm.reshape(-1),
        val_idx,
        v_norm.reshape(-1),
    )
    comp_bytes = packed.reshape(seq_len, num_kv_heads, layout.total_bytes)
    return queries, comp_bytes, layout, key_q, val_q, heads_per_kv


def run_once(
    queries,
    comp_bytes,
    layout,
    key_q,
    val_q,
    heads_per_kv,
    *,
    use_triton: bool,
):
    if use_triton:
        return turboquant_decode_attention(
            queries,
            comp_bytes,
            layout,
            key_centroids=key_q.mse.centroids,
            val_centroids=val_q.centroids,
            key_pi=key_q.mse.Pi,
            key_pi_t=key_q.mse.Pi.T,
            val_pi=val_q.Pi,
            s_t=key_q.S.T,
            heads_per_kv=heads_per_kv,
            qjl_dim=key_q.qjl_dim,
            sm_scale=1.0 / math.sqrt(layout.head_dim),
            causal=True,
            pos_offset=comp_bytes.shape[0] - 1,
            num_kv_splits=8,
            use_triton=True,
        )
    return turboquant_decode_attention_pytorch(
        queries,
        comp_bytes,
        layout,
        key_centroids=key_q.mse.centroids,
        val_centroids=val_q.centroids,
        key_pi=key_q.mse.Pi,
        key_pi_t=key_q.mse.Pi.T,
        val_pi=val_q.Pi,
        s_t=key_q.S.T,
        heads_per_kv=heads_per_kv,
        qjl_dim=key_q.qjl_dim,
        sm_scale=1.0 / math.sqrt(layout.head_dim),
        causal=True,
        pos_offset=comp_bytes.shape[0] - 1,
    )


def benchmark(label: str, fn, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms = elapsed * 1000.0 / iters
    print(f"{label:>12}: {ms:7.3f} ms/call")
    return ms


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available in this environment; Triton benchmark skipped.")
        return

    device = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}")
    for seq_len in (512, 2048, 8192):
        print(f"\nseq_len={seq_len}")
        queries, comp_bytes, layout, key_q, val_q, heads_per_kv = build_fixture(
            seq_len=seq_len,
            device=device,
        )
        ref = run_once(
            queries, comp_bytes, layout, key_q, val_q, heads_per_kv,
            use_triton=False,
        )
        tri = run_once(
            queries, comp_bytes, layout, key_q, val_q, heads_per_kv,
            use_triton=True,
        )
        cosine = torch.nn.functional.cosine_similarity(
            ref.reshape(1, -1).float(),
            tri.reshape(1, -1).float(),
        ).item()
        print(f"{'cosine':>12}: {cosine:7.5f}")
        ref_ms = benchmark(
            "pytorch",
            lambda: run_once(
                queries, comp_bytes, layout, key_q, val_q, heads_per_kv,
                use_triton=False,
            ),
        )
        tri_ms = benchmark(
            "triton",
            lambda: run_once(
                queries, comp_bytes, layout, key_q, val_q, heads_per_kv,
                use_triton=True,
            ),
        )
        print(f"{'speedup':>12}: {ref_ms / tri_ms:7.2f}x")


if __name__ == "__main__":
    main()
