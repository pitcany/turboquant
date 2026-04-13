"""
Self-consistency tests for the TurboQuant-ish C-port reference (WHT variant).

This branch replaces the paper's Haar rotation Π with the Randomized Hadamard
Transform Π = (1/√d) · H · diag(σ), so byte-exact equality with
turboquant.py::TurboQuantProd (which uses a true Haar Π) no longer holds —
see the commit that introduced this branch.

What we still test:
    - σ is ±1 and the stored matrix S matches what generate_qjl_matrix yields.
    - The RHT rotation in the reference is actually orthogonal (‖Π x‖ = ‖x‖).
    - Internal layouts round-trip (pack/unpack).
    - Different layers produce different quantized bytes.
    - The algorithm still meets paper-adjacent bounds on MSE and IP
      correlation for random unit-vector inputs — we expect RHT to match
      Haar very closely on that workload.
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

from turboquant import generate_qjl_matrix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tq_paper_reference as ref


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
def unit_vectors(d):
    g = torch.Generator().manual_seed(12345)
    x = torch.randn(200, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# ---------- Per-layer σ and S sanity ----------

def test_sigma_is_plus_minus_one(constants, layer_idx):
    sigma = constants.sigma[layer_idx]
    assert sigma.shape == (constants.d,)
    assert torch.all((sigma == 1.0) | (sigma == -1.0))


def test_s_matches_generate_qjl_matrix(constants, layer_idx):
    """S is the paper's QJL matrix (unchanged on this branch)."""
    expected = generate_qjl_matrix(constants.d, m=constants.d, seed=43 + layer_idx)
    assert torch.allclose(constants.s[layer_idx], expected, atol=0, rtol=0)


# ---------- RHT orthogonality ----------

def test_rht_is_orthogonal(constants, layer_idx, unit_vectors):
    """Π = (1/√d) · H · diag(σ) must preserve norms (to fp32 precision)."""
    sigma = constants.sigma[layer_idx]
    for x in unit_vectors[:20]:
        y = ref.rht_apply(sigma, x)
        assert abs(y.norm().item() - x.norm().item()) < 1e-5


def test_rht_inverse(constants, layer_idx, unit_vectors):
    """Πᵀ · Π · x ≈ x."""
    sigma = constants.sigma[layer_idx]
    for x in unit_vectors[:20]:
        recovered = ref.rht_apply_t(sigma, ref.rht_apply(sigma, x))
        assert torch.allclose(recovered, x, atol=1e-5, rtol=0)


# ---------- Layout round-trips ----------

def test_layer_idx_stored_in_block(constants, unit_vectors, layer_idx):
    blk = ref.quantize_block(unit_vectors[0], constants, layer_idx=layer_idx)
    assert blk[4] == layer_idx


def test_different_layers_produce_different_output(constants, unit_vectors):
    x = unit_vectors[0]
    blk0 = ref.quantize_block(x, constants, layer_idx=0)
    blk1 = ref.quantize_block(x, constants, layer_idx=1)
    assert blk0 != blk1


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


# ---------- Paper-adjacent bound regression (layer 0 only) ----------

def test_paper_mse_bound(constants, unit_vectors):
    """Stage-1 reconstruction MSE per vector must stay below the paper's
    3-bit bound. RHT is orthogonal like Haar, so the same bound applies."""
    bound_per_coord = math.sqrt(3) * math.pi / 2 * (1 / (4 ** 3))
    mse_sum = 0.0
    for x in unit_vectors:
        blk = ref.quantize_block(x, constants, layer_idx=0)
        x_hat = ref.dequantize_block(blk, constants)
        mse_sum += ((x - x_hat) ** 2).sum().item()
    mse_per_vector = mse_sum / len(unit_vectors)
    assert mse_per_vector < 1.5 * bound_per_coord


def test_paper_ip_correlation(constants, unit_vectors):
    """Inner-product correlation with true values must meet paper's 3-bit
    number. RHT is expected to match Haar's correlation for random unit
    vectors (identical marginals by sphere symmetry)."""
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
