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

Per-layer verification: tests iterate over multiple layer indices and verify
each layer's Π_i/S_i (seed=42+i / 43+i) matches TurboQuantProd(seed=42+i).

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


# Layer indices to test: layer 0 (backward compat), plus a spread of others.
TEST_LAYERS = [0, 1, 7, 15, 31]


# ---------- Fixtures ----------

@pytest.fixture(scope="module", params=[128, 256])
def d(request):
    return request.param


@pytest.fixture(scope="module")
def constants(d):
    return ref.load_constants(d)


@pytest.fixture(scope="module", params=TEST_LAYERS)
def layer_idx(request):
    return request.param


@pytest.fixture(scope="module")
def turboquant_prod(d, layer_idx):
    # TurboQuantProd(d, bits) allocates mse_bits = bits - 1, so for 3-bit Stage 1
    # we pass bits=4 (4-bit "total" where 3 are MSE and 1 is QJL).
    # Per-layer seed = 42 + layer_idx.
    return TurboQuantProd(d, bits=4, seed=42 + layer_idx)


@pytest.fixture(scope="module")
def unit_vectors(d):
    g = torch.Generator().manual_seed(12345)
    x = torch.randn(200, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# ---------- Pi and S must match between oracle and reference ----------

def test_pi_matches_turboquant(constants, turboquant_prod, layer_idx):
    """Pi used by the C-port reference must be the same matrix TurboQuantProd uses."""
    assert torch.allclose(constants.pi[layer_idx], turboquant_prod.mse.Pi, atol=0, rtol=0)


def test_s_matches_turboquant(constants, turboquant_prod, layer_idx):
    """S used by the C-port reference must be the same matrix TurboQuantProd uses."""
    assert torch.allclose(constants.s[layer_idx], turboquant_prod.S, atol=0, rtol=0)


def test_centroids_match_turboquant(constants, turboquant_prod):
    assert torch.allclose(constants.centroids, turboquant_prod.mse.centroids, atol=0, rtol=0)


# ---------- Byte-exact equality of quantized state ----------

def test_byte_exact_indices_and_signs(constants, turboquant_prod, unit_vectors, layer_idx):
    """For each unit vector, quantize with turboquant.py and with our reference,
    then compare the stored indices and QJL sign bits byte-for-byte."""
    d = constants.d
    oracle = turboquant_prod.quantize(unit_vectors)
    oracle_idx = oracle["mse_indices"]
    oracle_signs_pm = oracle["qjl_signs"]  # +1 / -1
    oracle_signs_bits = (oracle_signs_pm < 0).to(torch.uint8)  # 1 = negative

    qs_off = ref._qs_offset()
    signs_off = ref._signs_offset(d)

    for i, x in enumerate(unit_vectors):
        blk = ref.quantize_block(x, constants, layer_idx=layer_idx)

        # unpack indices from blk
        qs = blk[qs_off : qs_off + (d * 3) // 8]
        ref_idx = ref._unpack_indices_bitplane(qs, d)

        # unpack signs
        signs = blk[signs_off : signs_off + d // 8]
        ref_signs_pm = ref._unpack_signs(signs, d)
        ref_signs_bits = (ref_signs_pm < 0).to(torch.uint8)

        assert torch.equal(ref_idx, oracle_idx[i]), f"index mismatch on vector {i} layer {layer_idx}"
        assert torch.equal(ref_signs_bits, oracle_signs_bits[i]), f"sign mismatch on vector {i} layer {layer_idx}"


def test_layer_idx_stored_in_block(constants, unit_vectors, layer_idx):
    """The layer_idx byte must be stored at offset 4 in the block."""
    d = constants.d
    blk = ref.quantize_block(unit_vectors[0], constants, layer_idx=layer_idx)
    assert blk[4] == layer_idx, f"Expected layer_idx={layer_idx} at offset 4, got {blk[4]}"


def test_res_d_matches_within_fp16(constants, turboquant_prod, unit_vectors, layer_idx):
    """res_d is stored fp16 in the block; should match oracle within fp16 rounding."""
    d = constants.d
    oracle = turboquant_prod.quantize(unit_vectors)
    oracle_res = oracle["residual_norm"]

    max_rel_err = 0.0
    for i, x in enumerate(unit_vectors):
        blk = ref.quantize_block(x, constants, layer_idx=layer_idx)
        import struct
        ref_res = struct.unpack('<e', blk[2:4])[0]
        oracle_i = oracle_res[i].item()
        rel = abs(ref_res - oracle_i) / max(abs(oracle_i), 1e-8)
        max_rel_err = max(max_rel_err, rel)
        assert rel < 2e-3, f"res_d rel err {rel:.2e} on vector {i} (oracle={oracle_i}, ref={ref_res})"
    assert max_rel_err < 2e-3


# ---------- Inner-product estimator numerically equivalent ----------

def test_inner_product_matches_oracle(constants, turboquant_prod, unit_vectors, layer_idx):
    d = constants.d
    # Take the first 20 vectors as keys, the next 20 as queries.
    keys = unit_vectors[:20]
    queries = unit_vectors[20:40]

    # Oracle: quantize keys once, estimate <q, k> for all pairs.
    oracle = turboquant_prod.quantize(keys)
    oracle_scores = torch.zeros(len(queries), len(keys))
    for qi, q in enumerate(queries):
        for ki in range(len(keys)):
            single = {k: v[ki : ki + 1] for k, v in oracle.items()}
            oracle_scores[qi, ki] = turboquant_prod.inner_product(q.unsqueeze(0), single)[0]

    # Reference: quantize keys into bytes, then estimate.
    ref_scores = torch.zeros(len(queries), len(keys))
    key_blocks = [ref.quantize_block(k, constants, layer_idx=layer_idx) for k in keys]
    for qi, q in enumerate(queries):
        for ki, blk in enumerate(key_blocks):
            ref_scores[qi, ki] = ref.inner_product(q, blk, constants)

    max_abs = (ref_scores - oracle_scores).abs().max().item()
    assert max_abs < 5e-3, f"max abs diff {max_abs:.2e} at layer {layer_idx}"


# ---------- Different layers produce different results ----------

def test_different_layers_produce_different_output(constants, unit_vectors):
    """Quantizing the same vector with different layer indices must give different bytes."""
    d = constants.d
    x = unit_vectors[0]
    blk0 = ref.quantize_block(x, constants, layer_idx=0)
    blk1 = ref.quantize_block(x, constants, layer_idx=1)
    # At minimum, layer_idx byte differs. The qs/signs should also differ
    # since the rotation matrices are different.
    assert blk0 != blk1, "Layer 0 and layer 1 should produce different quantized blocks"


# ---------- Paper-bound regression (layer 0 only for stability) ----------

def test_paper_mse_bound(constants, unit_vectors):
    """Stage-1 reconstruction MSE per vector must stay below the paper's bound."""
    d = constants.d
    bound_per_coord = math.sqrt(3) * math.pi / 2 * (1 / (4 ** 3))
    mse_sum = 0.0
    for x in unit_vectors:
        blk = ref.quantize_block(x, constants, layer_idx=0)
        x_hat = ref.dequantize_block(blk, constants)
        mse_sum += ((x - x_hat) ** 2).sum().item()
    mse_per_vector = mse_sum / len(unit_vectors)
    assert mse_per_vector < 1.5 * bound_per_coord


def test_paper_ip_correlation(constants, unit_vectors):
    """Inner-product correlation with true values must meet paper's 3-bit number."""
    d = constants.d
    half = len(unit_vectors) // 2
    queries = unit_vectors[:half]
    keys    = unit_vectors[half : 2 * half]
    true_ip = (queries * keys).sum(dim=-1)

    ref_ip = torch.zeros(half)
    for i in range(half):
        blk = ref.quantize_block(keys[i], constants, layer_idx=0)
        ref_ip[i] = ref.inner_product(queries[i], blk, constants)

    corr = torch.corrcoef(torch.stack([true_ip, ref_ip]))[0, 1].item()
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
