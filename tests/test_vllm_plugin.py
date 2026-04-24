"""Tests for the vLLM TurboQuant plugin.

Covers config validation, compressed layout pack/unpack roundtrips,
KV spec page size calculations, and fused decode kernel correctness.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from vllm_plugin.attention import (
    _CompressedLayout,
    _pack_nbits,
    _unpack_nbits,
    _pack_2bit,
    _unpack_2bit,
    _pack_4bit,
    _unpack_4bit,
    _pack_bitplane,
    _unpack_bitplane,
    _pack_1bit,
    _unpack_1bit,
    _compressed_fp16_elems,
)
from vllm_plugin.config import TurboQuantConfig


# ── Config validation ────────────────────────────────────────────────


class TestTurboQuantConfig:

    def test_defaults(self) -> None:
        cfg = TurboQuantConfig()
        assert cfg.b_mse == 2
        assert cfg.b_qjl == 1
        assert cfg.head_dim == 128
        assert cfg.b_total == 3

    @pytest.mark.parametrize("b_mse", [2, 3, 4])
    def test_valid_b_mse(self, b_mse: int) -> None:
        cfg = TurboQuantConfig(b_mse=b_mse)
        assert cfg.b_mse == b_mse
        assert cfg.b_total == b_mse + 1

    @pytest.mark.parametrize("b_mse", [0, 1, 5, 8])
    def test_invalid_b_mse(self, b_mse: int) -> None:
        with pytest.raises(ValueError, match="b_mse"):
            TurboQuantConfig(b_mse=b_mse)

    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    def test_valid_head_dim(self, head_dim: int) -> None:
        cfg = TurboQuantConfig(head_dim=head_dim)
        assert cfg.head_dim == head_dim

    @pytest.mark.parametrize("head_dim", [32, 96, 512])
    def test_invalid_head_dim(self, head_dim: int) -> None:
        with pytest.raises(ValueError, match="head_dim"):
            TurboQuantConfig(head_dim=head_dim)

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TQ_B_MSE", "3")
        monkeypatch.setenv("TQ_HEAD_DIM", "256")
        cfg = TurboQuantConfig()
        assert cfg.b_mse == 3
        assert cfg.head_dim == 256

    def test_compression_ratio(self) -> None:
        cfg = TurboQuantConfig(b_mse=2, head_dim=128)
        assert cfg.compression_ratio > 3.0

    def test_summary_string(self) -> None:
        cfg = TurboQuantConfig()
        s = cfg.summary()
        assert "TurboQuant" in s
        assert "3b/coord" in s


# ── Bit-packing primitives ──────────────────────────────────────────


class TestBitPacking:

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_2bit_roundtrip(self, d: int) -> None:
        n = 16
        indices = torch.randint(0, 4, (n, d))
        packed = _pack_2bit(indices)
        assert packed.shape == (n, d // 4)
        recovered = _unpack_2bit(packed, d)
        assert (recovered == indices).all()

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_4bit_roundtrip(self, d: int) -> None:
        n = 16
        indices = torch.randint(0, 16, (n, d))
        packed = _pack_4bit(indices)
        assert packed.shape == (n, d // 2)
        recovered = _unpack_4bit(packed, d)
        assert (recovered == indices).all()

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("bits", [3, 5])
    def test_bitplane_roundtrip(self, d: int, bits: int) -> None:
        n = 16
        indices = torch.randint(0, 2**bits, (n, d))
        packed = _pack_bitplane(indices, bits)
        expected_len = bits * ((d + 7) // 8)
        assert packed.shape == (n, expected_len)
        recovered = _unpack_bitplane(packed, d, bits)
        assert (recovered == indices).all()

    def test_1bit_sign_roundtrip(self) -> None:
        n, d = 8, 128
        signs = torch.where(
            torch.rand(n, d) > 0.5,
            torch.ones(n, d), -torch.ones(n, d))
        packed = _pack_1bit(signs)
        assert packed.shape == (n, d // 8)
        recovered = _unpack_1bit(packed, d)
        assert (recovered == signs).all()

    @pytest.mark.parametrize("bits", [2, 3, 4, 5])
    def test_nbits_dispatch(self, bits: int) -> None:
        n, d = 8, 128
        indices = torch.randint(0, 2**bits, (n, d))
        packed = _pack_nbits(indices, bits)
        recovered = _unpack_nbits(packed, d, bits)
        assert (recovered == indices).all()


# ── Compressed layout ───────────────────────────────────────────────


class TestCompressedLayout:

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("b_mse", [2, 3, 4])
    def test_pack_unpack_roundtrip(self, d: int, b_mse: int) -> None:
        val_bits = b_mse + 1
        layout = _CompressedLayout(d, b_mse, 1, val_bits)
        n = 8

        k_mse = torch.randint(0, 2**b_mse, (n, d))
        k_signs = torch.where(
            torch.rand(n, d) > 0.5,
            torch.ones(n, d), -torch.ones(n, d))
        k_rnorm = torch.rand(n)
        k_norm = torch.rand(n)
        v_mse = torch.randint(0, 2**val_bits, (n, d))
        v_norm = torch.rand(n)

        buf = layout.pack(k_mse, k_signs, k_rnorm, k_norm, v_mse, v_norm)
        assert buf.shape == (n, layout.total_bytes)
        assert buf.dtype == torch.uint8

        km2, ks2, kr2, kn2, vm2, vn2 = layout.unpack(buf)
        assert (km2 == k_mse).all(), "key MSE indices mismatch"
        assert (vm2 == v_mse).all(), "val MSE indices mismatch"
        # Signs should match (converted through fp16)
        assert (ks2 == k_signs).all(), "key signs mismatch"

    def test_backward_compat_b2_d128(self) -> None:
        """The original plugin used 118 bytes for d=128, b_mse=2."""
        layout = _CompressedLayout(128, 2, 1, 3)
        assert layout.total_bytes == 118
        assert layout.fp16_elems == 59

    def test_total_bytes_even(self) -> None:
        """Total bytes must be even (padded for fp16 view)."""
        for d in [64, 128, 256]:
            for b_mse in [2, 3, 4]:
                layout = _CompressedLayout(d, b_mse, 1, b_mse + 1)
                assert layout.total_bytes % 2 == 0

    def test_fp16_elems_consistent(self) -> None:
        for d in [64, 128, 256]:
            for b_mse in [2, 3, 4]:
                layout = _CompressedLayout(d, b_mse, 1, b_mse + 1)
                assert layout.fp16_elems == layout.total_bytes // 2

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("b_mse", [2, 3, 4])
    def test_compressed_fp16_elems(self, d: int, b_mse: int) -> None:
        val_bits = b_mse + 1
        fp16 = _compressed_fp16_elems(d, b_mse, 1, val_bits)
        layout = _CompressedLayout(d, b_mse, 1, val_bits)
        assert fp16 == layout.fp16_elems


# ── KV spec page size ──────────────────────────────────────────────


class TestKVSpec:

    def test_import(self) -> None:
        from vllm_plugin.kv_spec import TurboQuantSpec
        assert TurboQuantSpec is not None

    def test_page_size_smaller_than_fp16(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TQ_B_MSE", "2")
        monkeypatch.setenv("TQ_B_QJL", "1")
        from vllm_plugin.kv_spec import TurboQuantSpec

        spec = TurboQuantSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.float16,
        )
        tq_size = spec.real_page_size_bytes
        # FP16 page: block_size * num_kv_heads * head_size * 2 bytes * 2 (K+V)
        fp16_size = 16 * 8 * 128 * 2 * 2
        assert tq_size < fp16_size, (
            f"TQ page ({tq_size}) should be smaller than FP16 ({fp16_size})"
        )


# ── Fused decode kernel ───────────────────────────────────────────


_CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not _CUDA, reason="CUDA required for fused decode kernel")
class TestFusedDecode:
    """Compare fused Triton decode kernel against per-request reference."""

    HEAD_DIM = 128
    NUM_KV_HEADS = 8
    HEADS_PER_KV = 4
    B_MSE = 2
    B_QJL = 1
    BLOCK_SIZE = 16

    @pytest.fixture
    def setup(self):
        """Build quantizers, kv_cache with block table, and random queries."""

        from vllm_plugin.compress_utils import (
            initialize_quantizers,
            store_compressed_kv,
        )

        hd = self.HEAD_DIM
        nkh = self.NUM_KV_HEADS
        hpkv = self.HEADS_PER_KV
        b_total = self.B_MSE + self.B_QJL
        bs = self.BLOCK_SIZE
        device = torch.device("cuda")

        layout = _CompressedLayout(hd, self.B_MSE, self.B_QJL, b_total)
        quants = initialize_quantizers(hd, b_total, 0, device, rotation="wht")

        # Two requests with different seq_lens
        seq_lens_list = [48, 80]
        num_reqs = len(seq_lens_list)

        # Allocate kv_cache blocks
        total_blocks = sum((s + bs - 1) // bs for s in seq_lens_list)
        kv_cache = torch.zeros(
            total_blocks + 2, bs, nkh, layout.fp16_elems,
            dtype=torch.float16, device=device,
        )

        # Sequential block allocation
        max_blocks = max((s + bs - 1) // bs for s in seq_lens_list)
        block_table = torch.zeros(
            num_reqs, max_blocks, dtype=torch.int32, device=device)
        blk_cursor = 0
        for ri, sl in enumerate(seq_lens_list):
            n_blk = (sl + bs - 1) // bs
            for bi in range(n_blk):
                block_table[ri, bi] = blk_cursor
                blk_cursor += 1

        # Compress random KV data and store into the cache
        key_q = quants["key_q"]
        val_q = quants["val_q"]
        for ri, sl in enumerate(seq_lens_list):
            keys = torch.randn(sl, nkh, hd, device=device)
            values = torch.randn(sl, nkh, hd, device=device)
            slot_mapping = torch.empty(sl, dtype=torch.long, device=device)
            for t in range(sl):
                blk = t // bs
                off = t % bs
                slot_mapping[t] = int(block_table[ri, blk].item()) * bs + off
            store_compressed_kv(
                keys, values, kv_cache, slot_mapping,
                bs, nkh, hd, layout, key_q, val_q,
            )

        queries = torch.randn(
            num_reqs, nkh * hpkv, hd, device=device, dtype=torch.float16)

        return {
            "queries": queries,
            "kv_cache": kv_cache,
            "block_table": block_table,
            "seq_lens": torch.tensor(
                seq_lens_list, dtype=torch.int32, device=device),
            "layout": layout,
            "quants": quants,
            "hd": hd,
            "nkh": nkh,
            "hpkv": hpkv,
            "bs": bs,
        }

    def test_fused_vs_reference(self, setup) -> None:
        """Fused kernel output matches per-request torch reference < 1e-3."""
        from vllm_plugin.triton_wrapper import (
            fused_decode_attention,
            turboquant_decode_attention,
        )

        d = setup
        q = d["quants"]
        key_q = q["key_q"]

        fused_out = fused_decode_attention(
            d["queries"], d["kv_cache"],
            d["block_table"], d["seq_lens"], d["layout"],
            key_centroids=q["key_centroids"].float(),
            val_centroids=q["val_centroids"].float(),
            key_pi_t=key_q.mse.Pi.T,
            val_pi=q["val_pi"],
            s_t=key_q.S.T,
            heads_per_kv=d["hpkv"],
            qjl_dim=key_q.qjl_dim,
            sm_scale=1.0 / math.sqrt(d["hd"]),
            rotation="wht",
            key_sigma=q["key_sigma"],
            val_sigma=q["val_sigma"],
        )
        assert fused_out is not None, "Fused path should be available on CUDA"

        # Per-request reference (gather + non-Triton decode)
        ref_parts = []
        bs = d["bs"]
        for ri in range(d["seq_lens"].shape[0]):
            sl = d["seq_lens"][ri].item()
            n_blk = (sl + bs - 1) // bs
            blk_ids = d["block_table"][ri, :n_blk]
            comp = d["kv_cache"][blk_ids].reshape(
                -1, d["nkh"], d["layout"].fp16_elems)[:sl]
            comp_bytes = comp.contiguous().view(torch.uint8).reshape(
                sl, d["nkh"], d["layout"].total_bytes)

            ref = turboquant_decode_attention(
                d["queries"][ri:ri + 1], comp_bytes, d["layout"],
                key_centroids=q["key_centroids"].float(),
                val_centroids=q["val_centroids"].float(),
                key_pi=key_q.mse.Pi,
                key_pi_t=key_q.mse.Pi.T,
                val_pi=q["val_pi"],
                s_t=key_q.S.T,
                heads_per_kv=d["hpkv"],
                qjl_dim=key_q.qjl_dim,
                sm_scale=1.0 / math.sqrt(d["hd"]),
                causal=False,
                pos_offset=0,
                num_kv_splits=1,
                use_triton=False,
                rotation="wht",
                key_sigma=q["key_sigma"],
                val_sigma=q["val_sigma"],
            )
            ref_parts.append(ref)

        ref_out = torch.cat(ref_parts, dim=0)
        max_diff = (fused_out.float() - ref_out.float()).abs().max().item()
        assert max_diff < 1e-3, f"fused vs reference max diff = {max_diff}"

    def test_single_request(self, setup) -> None:
        """Fused kernel works with a single request."""
        from vllm_plugin.triton_wrapper import fused_decode_attention

        d = setup
        q = d["quants"]
        key_q = q["key_q"]

        # Take only the first request
        out = fused_decode_attention(
            d["queries"][:1], d["kv_cache"],
            d["block_table"][:1], d["seq_lens"][:1], d["layout"],
            key_centroids=q["key_centroids"].float(),
            val_centroids=q["val_centroids"].float(),
            key_pi_t=key_q.mse.Pi.T,
            val_pi=q["val_pi"],
            s_t=key_q.S.T,
            heads_per_kv=d["hpkv"],
            qjl_dim=key_q.qjl_dim,
            sm_scale=1.0 / math.sqrt(d["hd"]),
            rotation="wht",
            key_sigma=q["key_sigma"],
            val_sigma=q["val_sigma"],
        )
        assert out is not None
        assert out.shape == (1, d["nkh"] * d["hpkv"], d["hd"])
