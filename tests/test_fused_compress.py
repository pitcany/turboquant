"""Tests for the fused KV compression Triton kernel.

Compares the fused Triton path against the reference torch path
to ensure numerical equivalence.
"""

from __future__ import annotations

import pytest
import torch

from turboquant import TurboQuantMSE, TurboQuantProd

# Skip the entire module if CUDA is not available.
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


def _reference_compress(
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    key_q: TurboQuantProd,
    val_q: TurboQuantMSE,
) -> dict[str, torch.Tensor]:
    """Run the reference (pure-torch) compress path."""
    eps = 1e-8
    key_norms = torch.norm(k_flat, dim=-1).clamp_min(eps)
    val_norms = torch.norm(v_flat, dim=-1).clamp_min(eps)
    key_units = k_flat / key_norms.unsqueeze(-1)
    val_units = v_flat / val_norms.unsqueeze(-1)

    ck = key_q.quantize(key_units)
    vi = val_q.quantize(val_units)

    return {
        "key_mse_indices": ck["mse_indices"],
        "qjl_signs": ck["qjl_signs"],
        "key_residual_norm": ck["residual_norm"],
        "key_norm": key_norms,
        "val_mse_indices": vi,
        "val_norm": val_norms,
    }


# ── Tests ──────────────────────────────────────────────────────────────


class TestFusedCompress:

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("b_mse", [2, 3])
    @pytest.mark.parametrize("num_rows", [1, 8, 32])
    def test_indices_match(
        self, head_dim: int, b_mse: int, num_rows: int,
    ) -> None:
        """Fused kernel produces identical MSE indices as reference."""
        from vllm_plugin.triton_kernels import _fused_compress_triton

        b_total = b_mse + 1
        key_q, val_q = _make_quantizers(head_dim, b_total)

        torch.manual_seed(0)
        k = torch.randn(num_rows, head_dim, device="cuda")
        v = torch.randn(num_rows, head_dim, device="cuda")

        ref = _reference_compress(k, v, key_q, val_q)
        fused = _fused_compress_triton(
            k, v,
            key_sigma=key_q.mse.sigma,
            val_sigma=val_q.sigma,
            key_boundaries=key_q.mse.boundaries,
            key_centroids=key_q.mse.centroids,
            val_boundaries=val_q.boundaries,
            s_matrix=key_q.S,
            head_dim=head_dim,
            qjl_dim=key_q.qjl_dim,
        )

        # Key MSE indices: exact match expected
        assert (fused["key_mse_indices"] == ref["key_mse_indices"]).all(), (
            f"key MSE index mismatch: "
            f"{(fused['key_mse_indices'] != ref['key_mse_indices']).sum()} diffs"
        )

        # Value MSE indices: exact match expected
        assert (fused["val_mse_indices"] == ref["val_mse_indices"]).all(), (
            f"val MSE index mismatch: "
            f"{(fused['val_mse_indices'] != ref['val_mse_indices']).sum()} diffs"
        )

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_signs_match(self, head_dim: int) -> None:
        """QJL sign bits match between fused and reference."""
        from vllm_plugin.triton_kernels import _fused_compress_triton

        key_q, val_q = _make_quantizers(head_dim, 3)

        torch.manual_seed(1)
        k = torch.randn(16, head_dim, device="cuda")
        v = torch.randn(16, head_dim, device="cuda")

        ref = _reference_compress(k, v, key_q, val_q)
        fused = _fused_compress_triton(
            k, v,
            key_sigma=key_q.mse.sigma,
            val_sigma=val_q.sigma,
            key_boundaries=key_q.mse.boundaries,
            key_centroids=key_q.mse.centroids,
            val_boundaries=val_q.boundaries,
            s_matrix=key_q.S,
            head_dim=head_dim,
            qjl_dim=key_q.qjl_dim,
        )

        match_rate = (fused["qjl_signs"] == ref["qjl_signs"]).float().mean()
        assert match_rate > 0.99, f"QJL sign match rate {match_rate:.4f} < 0.99"

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_norms_close(self, head_dim: int) -> None:
        """Norms match within tolerance."""
        from vllm_plugin.triton_kernels import _fused_compress_triton

        key_q, val_q = _make_quantizers(head_dim, 3)

        torch.manual_seed(2)
        k = torch.randn(16, head_dim, device="cuda")
        v = torch.randn(16, head_dim, device="cuda")

        ref = _reference_compress(k, v, key_q, val_q)
        fused = _fused_compress_triton(
            k, v,
            key_sigma=key_q.mse.sigma,
            val_sigma=val_q.sigma,
            key_boundaries=key_q.mse.boundaries,
            key_centroids=key_q.mse.centroids,
            val_boundaries=val_q.boundaries,
            s_matrix=key_q.S,
            head_dim=head_dim,
            qjl_dim=key_q.qjl_dim,
        )

        for name in ("key_norm", "val_norm", "key_residual_norm"):
            fused_key = name if name != "key_residual_norm" else "key_residual_norm"
            ref_key = name if name != "key_residual_norm" else "key_residual_norm"
            diff = (fused[fused_key] - ref[ref_key]).abs().max().item()
            assert diff < 1e-3, f"{name} max diff {diff} >= 1e-3"

    def test_end_to_end_pack_roundtrip(self) -> None:
        """Full compress + pack via fused path matches reference packed bytes."""
        from vllm_plugin.attention import _CompressedLayout
        from vllm_plugin.triton_kernels import _fused_compress_triton

        head_dim = 128
        b_mse = 2
        b_total = 3
        num_rows = 8

        key_q, val_q = _make_quantizers(head_dim, b_total)
        layout = _CompressedLayout(head_dim, b_mse, 1, b_total)

        torch.manual_seed(3)
        k = torch.randn(num_rows, head_dim, device="cuda")
        v = torch.randn(num_rows, head_dim, device="cuda")

        # Reference path
        ref = _reference_compress(k, v, key_q, val_q)
        ref_packed = layout.pack(
            ref["key_mse_indices"],
            ref["qjl_signs"],
            ref["key_residual_norm"],
            ref["key_norm"],
            ref["val_mse_indices"],
            ref["val_norm"],
        )

        # Fused path
        fused = _fused_compress_triton(
            k, v,
            key_sigma=key_q.mse.sigma,
            val_sigma=val_q.sigma,
            key_boundaries=key_q.mse.boundaries,
            key_centroids=key_q.mse.centroids,
            val_boundaries=val_q.boundaries,
            s_matrix=key_q.S,
            head_dim=head_dim,
            qjl_dim=key_q.qjl_dim,
        )
        fused_packed = layout.pack(
            fused["key_mse_indices"],
            fused["qjl_signs"],
            fused["key_residual_norm"],
            fused["key_norm"],
            fused["val_mse_indices"],
            fused["val_norm"],
        )

        # Unpack both and compare reconstructed fields
        ref_fields = layout.unpack(ref_packed)
        fused_fields = layout.unpack(fused_packed)

        # Indices should be exact
        assert (fused_fields[0] == ref_fields[0]).all(), "key MSE packed mismatch"
        assert (fused_fields[4] == ref_fields[4]).all(), "val MSE packed mismatch"

        # Signs should mostly match
        sign_match = (fused_fields[1] == ref_fields[1]).float().mean()
        assert sign_match > 0.99, f"packed sign match {sign_match:.4f}"

        # Norms (fp16 roundtripped)
        for i, name in [(2, "key_rnorm"), (3, "key_norm"), (5, "val_norm")]:
            diff = (fused_fields[i] - ref_fields[i]).abs().max().item()
            assert diff < 1e-3, f"packed {name} max diff {diff}"

    def test_store_compressed_kv_fused_path(self) -> None:
        """store_compressed_kv uses the fused path and produces valid output."""
        from vllm_plugin.attention import _CompressedLayout
        from vllm_plugin.compress_utils import store_compressed_kv

        head_dim = 128
        b_mse = 2
        b_total = 3
        num_tokens = 4
        num_kv_heads = 2
        block_size = 16

        key_q, val_q = _make_quantizers(head_dim, b_total)
        layout = _CompressedLayout(head_dim, b_mse, 1, b_total)

        # Simulate kv_cache
        num_blocks = 2
        kv_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, layout.fp16_elems,
            dtype=torch.float16, device="cuda",
        )
        slot_mapping = torch.arange(num_tokens, device="cuda")

        torch.manual_seed(4)
        key = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda")
        value = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda")

        store_compressed_kv(
            key, value, kv_cache, slot_mapping, block_size,
            num_kv_heads, head_dim, layout, key_q, val_q,
        )

        # Check that cache slots have non-zero bytes (data was written).
        # Use a byte view because packed data may contain fp16 NaN patterns.
        for t in range(num_tokens):
            blk = t // block_size
            off = t % block_size
            raw = kv_cache[blk, off].contiguous().view(torch.uint8)
            assert raw.any(), f"slot {t} is all zeros"
