"""Tests for the optimized vectorized decode stage1 kernel.

Compares the Triton decode path (stage1 + stage2) against the torch
reference to ensure the vectorized unpacking produces numerically
equivalent results.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from turboquant import TurboQuantMSE, TurboQuantProd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

try:
    from vllm_plugin.triton_kernels import TRITON_AVAILABLE
except ImportError:
    TRITON_AVAILABLE = False

pytestmark = [
    pytestmark,
    pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton required"),
]


# ── Helpers ────────────────────────────────────────────────────────────


def _make_quantizers(
    head_dim: int, b_total: int, seed: int = 42, device: str = "cuda",
) -> tuple[TurboQuantProd, TurboQuantMSE]:
    key_q = TurboQuantProd(
        head_dim, b_total, seed=seed, device=device, rotation="wht",
    )
    val_q = TurboQuantMSE(
        head_dim, b_total, seed=seed + 500, device=device, rotation="wht",
    )
    return key_q, val_q


def _build_compressed_kv(
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    key_q: TurboQuantProd,
    val_q: TurboQuantMSE,
    layout: "_CompressedLayout",
    seed: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    """Build packed compressed KV bytes from random key/value vectors."""
    torch.manual_seed(seed)
    k = torch.randn(seq_len * num_kv_heads, head_dim, device=device)
    v = torch.randn(seq_len * num_kv_heads, head_dim, device=device)

    eps = 1e-8
    k_norms = torch.norm(k, dim=-1).clamp_min(eps)
    v_norms = torch.norm(v, dim=-1).clamp_min(eps)
    k_units = k / k_norms.unsqueeze(-1)
    v_units = v / v_norms.unsqueeze(-1)

    ck = key_q.quantize(k_units)
    vi = val_q.quantize(v_units)

    packed = layout.pack(
        ck["mse_indices"], ck["qjl_signs"],
        ck["residual_norm"], k_norms,
        vi, v_norms,
    )
    return packed.reshape(seq_len, num_kv_heads, layout.total_bytes)


def _run_decode_paths(
    seq_len: int = 512,
    num_kv_heads: int = 8,
    heads_per_kv: int = 8,
    head_dim: int = 128,
    b_mse: int = 2,
    b_total: int = 3,
    num_kv_splits: int = 8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run both torch and Triton decode paths, return partial_acc and partial_lse."""
    from vllm_plugin.attention import _CompressedLayout
    from vllm_plugin.triton_kernels import _stage1_torch, _stage1_triton

    key_q, val_q = _make_quantizers(head_dim, b_total)
    layout = _CompressedLayout(head_dim, b_mse, 1, b_total)

    comp_bytes = _build_compressed_kv(
        seq_len, num_kv_heads, head_dim, key_q, val_q, layout, seed=seed,
    )

    torch.manual_seed(seed + 999)
    q_rot = torch.randn(num_kv_heads, heads_per_kv, head_dim, device="cuda")
    q_sketch = torch.randn(num_kv_heads, heads_per_kv, head_dim, device="cuda")
    qjl_corr = math.sqrt(math.pi / 2.0) / key_q.qjl_dim
    sm_scale = 1.0 / math.sqrt(head_dim)

    ref_acc, ref_lse = _stage1_torch(
        q_rot, q_sketch, comp_bytes, layout,
        key_centroids=key_q.mse.centroids,
        val_centroids=val_q.centroids,
        qjl_corr=qjl_corr, sm_scale=sm_scale,
        num_kv_splits=num_kv_splits,
    )
    tri_acc, tri_lse = _stage1_triton(
        q_rot, q_sketch, comp_bytes, layout,
        key_centroids=key_q.mse.centroids,
        val_centroids=val_q.centroids,
        qjl_corr=qjl_corr, sm_scale=sm_scale,
        num_kv_splits=num_kv_splits,
    )
    return ref_acc, ref_lse, tri_acc, tri_lse


# ── Tests ──────────────────────────────────────────────────────────────


class TestFastDecodeStage1:
    """Correctness tests: optimized Triton stage1 vs torch reference."""

    @pytest.mark.parametrize("seq_len", [64, 256, 512, 1024])
    def test_partial_acc_matches(self, seq_len: int) -> None:
        """Partial accumulator values match within tolerance."""
        ref_acc, ref_lse, tri_acc, tri_lse = _run_decode_paths(seq_len=seq_len)

        # LSE should be very close (drives attention weight correctness)
        lse_diff = (tri_lse - ref_lse).abs().max().item()
        assert lse_diff < 0.05, f"LSE max diff {lse_diff:.4f} >= 0.05"

        # Partial acc: compare weighted output per split
        acc_diff = (tri_acc - ref_acc).abs().max().item()
        assert acc_diff < 0.02, f"Partial acc max diff {acc_diff:.6f} >= 0.02"

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_head_dim_variants(self, head_dim: int) -> None:
        """Works for different head dimensions."""
        ref_acc, ref_lse, tri_acc, tri_lse = _run_decode_paths(
            head_dim=head_dim, seq_len=256,
        )
        lse_diff = (tri_lse - ref_lse).abs().max().item()
        assert lse_diff < 0.05, f"LSE max diff {lse_diff:.4f} (head_dim={head_dim})"

        acc_diff = (tri_acc - ref_acc).abs().max().item()
        assert acc_diff < 0.02, f"Acc max diff {acc_diff:.6f} (head_dim={head_dim})"

    def test_single_token(self) -> None:
        """Edge case: seq_len=1 (first decode step)."""
        ref_acc, ref_lse, tri_acc, tri_lse = _run_decode_paths(
            seq_len=1, num_kv_splits=1,
        )
        lse_diff = (tri_lse - ref_lse).abs().max().item()
        assert lse_diff < 0.05, f"LSE max diff {lse_diff:.4f} (single token)"

    def test_short_sequence(self) -> None:
        """Edge case: seq_len < BLOCK_N."""
        ref_acc, ref_lse, tri_acc, tri_lse = _run_decode_paths(
            seq_len=17, num_kv_splits=2,
        )
        lse_diff = (tri_lse - ref_lse).abs().max().item()
        assert lse_diff < 0.05


class TestFastDecodeEndToEnd:
    """End-to-end: full decode attention (stage1 + stage2) Triton vs torch."""

    @pytest.mark.parametrize("seq_len", [128, 512, 2048])
    def test_full_decode_matches(self, seq_len: int) -> None:
        """Full decode output matches torch reference."""
        from vllm_plugin.attention import _CompressedLayout
        from vllm_plugin.triton_wrapper import (
            turboquant_decode_attention,
            turboquant_decode_attention_pytorch,
        )

        head_dim = 128
        num_kv_heads = 8
        heads_per_kv = 8
        b_mse, b_total = 2, 3

        key_q, val_q = _make_quantizers(head_dim, b_total)
        layout = _CompressedLayout(head_dim, b_mse, 1, b_total)

        comp_bytes = _build_compressed_kv(
            seq_len, num_kv_heads, head_dim, key_q, val_q, layout,
        )

        torch.manual_seed(77)
        queries = torch.randn(
            1, num_kv_heads * heads_per_kv, head_dim, device="cuda",
        )
        sm_scale = 1.0 / math.sqrt(head_dim)

        common = dict(
            comp_bytes=comp_bytes, layout=layout,
            key_centroids=key_q.mse.centroids,
            val_centroids=val_q.centroids,
            key_pi=key_q.mse.Pi if hasattr(key_q.mse, "Pi") else torch.eye(
                head_dim, device="cuda"),
            key_pi_t=key_q.mse.Pi.T if hasattr(key_q.mse, "Pi") else torch.eye(
                head_dim, device="cuda"),
            val_pi=val_q.Pi if hasattr(val_q, "Pi") else torch.eye(
                head_dim, device="cuda"),
            s_t=key_q.S.T,
            heads_per_kv=heads_per_kv,
            qjl_dim=key_q.qjl_dim,
            sm_scale=sm_scale,
            causal=False,
            pos_offset=0,
            rotation="wht",
            key_sigma=key_q.mse.sigma,
            val_sigma=val_q.sigma,
        )

        ref = turboquant_decode_attention_pytorch(queries, **common)
        tri = turboquant_decode_attention(
            queries, **common, use_triton=True, num_kv_splits=8,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            ref.float().flatten(), tri.float().flatten(), dim=0,
        ).item()
        assert cos_sim > 0.99, f"cos_sim {cos_sim:.4f} < 0.99 (seq_len={seq_len})"

        abs_diff = (ref.float() - tri.float()).abs().max().item()
        assert abs_diff < 0.1, f"max abs diff {abs_diff:.4f} (seq_len={seq_len})"


class TestFastDecodeBenchmark:
    """Micro-benchmark for kernel timing (not correctness)."""

    @pytest.mark.parametrize("seq_len", [512, 2048])
    def test_stage1_timing(self, seq_len: int) -> None:
        """Measure stage1 kernel time. Informational; no hard assert."""
        from vllm_plugin.attention import _CompressedLayout
        from vllm_plugin.triton_kernels import _stage1_triton

        head_dim = 128
        num_kv_heads = 8
        heads_per_kv = 8
        b_mse, b_total = 2, 3

        key_q, val_q = _make_quantizers(head_dim, b_total)
        layout = _CompressedLayout(head_dim, b_mse, 1, b_total)
        comp_bytes = _build_compressed_kv(
            seq_len, num_kv_heads, head_dim, key_q, val_q, layout,
        )

        torch.manual_seed(99)
        q_rot = torch.randn(num_kv_heads, heads_per_kv, head_dim, device="cuda")
        q_sketch = torch.randn(num_kv_heads, heads_per_kv, head_dim, device="cuda")
        qjl_corr = math.sqrt(math.pi / 2.0) / key_q.qjl_dim
        sm_scale = 1.0 / math.sqrt(head_dim)

        kwargs = dict(
            q_rot=q_rot, q_sketch=q_sketch, comp_bytes=comp_bytes,
            layout=layout, key_centroids=key_q.mse.centroids,
            val_centroids=val_q.centroids,
            qjl_corr=qjl_corr, sm_scale=sm_scale, num_kv_splits=8,
        )

        # Warmup (also triggers Triton JIT compilation)
        for _ in range(3):
            _stage1_triton(**kwargs)
        torch.cuda.synchronize()

        # Timed runs
        n_iter = 20
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _stage1_triton(**kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms_per_call = (t1 - t0) / n_iter * 1000
        print(f"\n  stage1 @ seq_len={seq_len}: {ms_per_call:.3f} ms/call")

        # Soft target: <1ms for seq_len=2048 (80-layer model needs <1ms/layer)
        if seq_len == 2048:
            assert ms_per_call < 3.0, (
                f"stage1 too slow: {ms_per_call:.2f}ms (target <1ms)"
            )
