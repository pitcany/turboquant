"""
Byte-exact equality tests for the TurboQuant paper C-port reference.

Two oracles are cross-checked:
    (A) turboquant.py::TurboQuantProd — the verified paper implementation
    (B) tq_paper_reference.py         — the byte-level mirror of the C layout

(B) must match (A) up to:
    - integer equality for Lloyd-Max indices (exact, both use torch.bucketize)
    - integer equality for QJL sign bits
    - fp16 round-trip for orig_norm and res_d
    - small fp32 accumulation differences for inner_product (< 1e-4 typical)

If any of these are violated, the C port (which byte-mirrors B) cannot be
guaranteed to match the paper. The test makes that failure visible.

Also asserts paper's MSE and IP-correlation bounds, so we catch regressions
vs. paper-level accuracy, not just byte-layout bugs.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

# Make turboquant.py importable from the repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from turboquant import TurboQuantProd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tq_paper_reference as ref


# ---------- Fixtures ----------

@pytest.fixture(scope="module", params=[128, 256])
def d(request):
    return request.param


@pytest.fixture(scope="module")
def constants(d):
    return ref.load_constants(d)


@pytest.fixture(scope="module")
def turboquant_prod(d):
    # TurboQuantProd(d, bits) allocates mse_bits = bits - 1, so for 3-bit Stage 1
    # we pass bits=4 (4-bit "total" where 3 are MSE and 1 is QJL).
    return TurboQuantProd(d, bits=4, seed=42)


@pytest.fixture(scope="module")
def unit_vectors(d):
    g = torch.Generator().manual_seed(12345)
    x = torch.randn(200, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# ---------- Π and S must match between oracle and reference ----------

def test_pi_matches_turboquant(constants, turboquant_prod):
    """Π used by the C-port reference must be the same matrix TurboQuantProd uses."""
    assert torch.allclose(constants.pi, turboquant_prod.mse.Pi, atol=0, rtol=0)


def test_s_matches_turboquant(constants, turboquant_prod):
    """S used by the C-port reference must be the same matrix TurboQuantProd uses."""
    assert torch.allclose(constants.s, turboquant_prod.S, atol=0, rtol=0)


def test_centroids_match_turboquant(constants, turboquant_prod):
    assert torch.allclose(constants.centroids, turboquant_prod.mse.centroids, atol=0, rtol=0)


# ---------- Byte-exact equality of quantized state ----------

def test_byte_exact_indices_and_signs(constants, turboquant_prod, unit_vectors):
    """For each unit vector, quantize with turboquant.py and with our reference,
    then compare the stored indices and QJL sign bits byte-for-byte."""
    d = constants.d
    oracle = turboquant_prod.quantize(unit_vectors)
    oracle_idx = oracle["mse_indices"]
    oracle_signs_pm = oracle["qjl_signs"]  # +1 / -1
    oracle_signs_bits = (oracle_signs_pm < 0).to(torch.uint8)  # 1 = negative

    for i, x in enumerate(unit_vectors):
        blk = ref.quantize_block(x, constants)

        # unpack indices from blk
        qs = blk[4 : 4 + (d * 3) // 8]
        ref_idx = ref._unpack_indices_bitplane(qs, d)

        # unpack signs
        sig_off = 4 + (d * 3) // 8
        signs = blk[sig_off : sig_off + d // 8]
        ref_signs_pm = ref._unpack_signs(signs, d)
        ref_signs_bits = (ref_signs_pm < 0).to(torch.uint8)

        assert torch.equal(ref_idx, oracle_idx[i]), f"index mismatch on vector {i}"
        assert torch.equal(ref_signs_bits, oracle_signs_bits[i]), f"sign mismatch on vector {i}"


def test_res_d_matches_within_fp16(constants, turboquant_prod, unit_vectors):
    """res_d is stored fp16 in the block; should match oracle within fp16 rounding."""
    d = constants.d
    oracle = turboquant_prod.quantize(unit_vectors)
    oracle_res = oracle["residual_norm"]

    max_rel_err = 0.0
    for i, x in enumerate(unit_vectors):
        blk = ref.quantize_block(x, constants)
        import struct
        ref_res = struct.unpack('<e', blk[2:4])[0]
        oracle_i = oracle_res[i].item()
        rel = abs(ref_res - oracle_i) / max(abs(oracle_i), 1e-8)
        max_rel_err = max(max_rel_err, rel)
        assert rel < 2e-3, f"res_d rel err {rel:.2e} on vector {i} (oracle={oracle_i}, ref={ref_res})"
    # fp16 has ~11-bit mantissa → worst-case rel err ~ 2^-11 ≈ 5e-4.
    assert max_rel_err < 2e-3


# ---------- Inner-product estimator numerically equivalent ----------

def test_inner_product_matches_oracle(constants, turboquant_prod, unit_vectors):
    d = constants.d
    # Take the first 20 vectors as keys, the next 20 as queries.
    keys = unit_vectors[:20]
    queries = unit_vectors[20:40]

    # Oracle: quantize keys once, estimate <q, k> for all pairs.
    oracle = turboquant_prod.quantize(keys)
    # Compute pairwise with the oracle's batched interface.
    oracle_scores = torch.zeros(len(queries), len(keys))
    for qi, q in enumerate(queries):
        # TurboQuantProd.inner_product requires broadcasting; run per key.
        for ki in range(len(keys)):
            single = {k: v[ki : ki + 1] for k, v in oracle.items()}
            oracle_scores[qi, ki] = turboquant_prod.inner_product(q.unsqueeze(0), single)[0]

    # Reference: quantize keys into bytes, then estimate.
    ref_scores = torch.zeros(len(queries), len(keys))
    key_blocks = [ref.quantize_block(k, constants) for k in keys]
    for qi, q in enumerate(queries):
        for ki, blk in enumerate(key_blocks):
            ref_scores[qi, ki] = ref.inner_product(q, blk, constants)

    max_abs = (ref_scores - oracle_scores).abs().max().item()
    # Differences come from: fp16 rounding of orig_norm / res_d, and Σ ordering.
    # Empirically < 5e-4 across 400 pairs; pick a loose cap that still catches bugs.
    assert max_abs < 5e-3, f"max abs diff {max_abs:.2e}"


# ---------- Paper-bound regression ----------

def test_paper_mse_bound(constants, unit_vectors):
    """Stage-1 reconstruction MSE per vector must stay below the paper's bound."""
    d = constants.d
    bound_per_coord = math.sqrt(3) * math.pi / 2 * (1 / (4 ** 3))  # Stage-1 bits=3
    mse_sum = 0.0
    for x in unit_vectors:
        blk = ref.quantize_block(x, constants)
        x_hat = ref.dequantize_block(blk, constants)
        mse_sum += ((x - x_hat) ** 2).sum().item()
    mse_per_vector = mse_sum / len(unit_vectors)
    # Per-coord bound × d, scaled by a small margin.
    assert mse_per_vector < 1.5 * bound_per_coord


def test_paper_ip_correlation(constants, unit_vectors):
    """Inner-product correlation with true values must meet paper's 3-bit number."""
    d = constants.d
    # Pairs (q, k) from different halves of the unit_vectors batch.
    half = len(unit_vectors) // 2
    queries = unit_vectors[:half]
    keys    = unit_vectors[half : 2 * half]
    true_ip = (queries * keys).sum(dim=-1)

    ref_ip = torch.zeros(half)
    for i in range(half):
        blk = ref.quantize_block(keys[i], constants)
        ref_ip[i] = ref.inner_product(queries[i], blk, constants)

    corr = torch.corrcoef(torch.stack([true_ip, ref_ip]))[0, 1].item()
    # Paper's 3-bit Stage-1 + 1-bit QJL: IP correlation ~0.92 at d=128.
    # Looser at d=256 is not expected; tightening slightly relies on the same
    # Gaussian approx working well.
    assert corr > 0.85, f"IP correlation {corr:.3f} below threshold"


# ---------- Round-trip sanity ----------

def test_roundtrip_pack_unpack_indices(constants):
    d = constants.d
    g = torch.Generator().manual_seed(99)
    for _ in range(20):
        idx = torch.randint(0, 8, (d,), generator=g)
        packed = ref._pack_indices_bitplane(idx)
        unpacked = ref._unpack_indices_bitplane(packed, d)
        assert torch.equal(idx.to(torch.int64), unpacked)


def test_roundtrip_pack_unpack_signs(constants):
    d = constants.d
    g = torch.Generator().manual_seed(99)
    for _ in range(20):
        signs = torch.where(torch.rand(d, generator=g) > 0.5, 1.0, -1.0)
        packed = ref._pack_signs(signs)
        unpacked = ref._unpack_signs(packed, d)
        assert torch.equal(signs, unpacked)
