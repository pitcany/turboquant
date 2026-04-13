"""
Self-consistency tests for the TurboQuant(-ish) Python/C reference, over
both rotation modes (TQP_ROT_WHT and TQP_ROT_HAAR).

Haar mode additionally asserts byte-exact equality with
turboquant.py::TurboQuantProd(seed=42+i), the paper-faithful oracle.
WHT mode only claims self-consistency + paper-adjacent bounds — by
construction it diverges from turboquant.py byte-for-byte.
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

from turboquant import TurboQuantProd, generate_qjl_matrix, generate_rotation_matrix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tq_paper_reference as ref


TEST_LAYERS = [0, 1, 7, 15, 31]
ROTATIONS = [ref.TQP_ROT_WHT, ref.TQP_ROT_HAAR]
ROTATION_IDS = {ref.TQP_ROT_WHT: "wht", ref.TQP_ROT_HAAR: "haar"}


# ---------- Fixtures ----------

@pytest.fixture(scope="module", params=[64, 128, 256])
def d(request):
    return request.param


@pytest.fixture(scope="module")
def constants(d):
    return ref.load_constants(d)


@pytest.fixture(scope="module", params=TEST_LAYERS)
def layer_idx(request):
    return request.param


@pytest.fixture(scope="module", params=ROTATIONS, ids=lambda r: ROTATION_IDS[r])
def rotation(request):
    return request.param


@pytest.fixture(scope="module")
def unit_vectors(d):
    g = torch.Generator().manual_seed(12345)
    x = torch.randn(200, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# ---------- Per-layer constants sanity ----------

def test_sigma_is_plus_minus_one(constants, layer_idx):
    sigma = constants.sigma[layer_idx]
    assert sigma.shape == (constants.d,)
    assert torch.all((sigma == 1.0) | (sigma == -1.0))


def test_pi_matches_generate_rotation_matrix(constants, layer_idx):
    """Π_i must match turboquant.py's generate_rotation_matrix at seed 42+i."""
    expected = generate_rotation_matrix(constants.d, seed=42 + layer_idx)
    assert torch.allclose(constants.pi[layer_idx], expected, atol=0, rtol=0)


def test_s_matches_generate_qjl_matrix(constants, layer_idx):
    expected = generate_qjl_matrix(constants.d, m=constants.d, seed=43 + layer_idx)
    assert torch.allclose(constants.s[layer_idx], expected, atol=0, rtol=0)


# ---------- Rotation-agnostic orthogonality ----------

def test_rotation_is_orthogonal(constants, layer_idx, rotation, unit_vectors):
    sigma = constants.sigma[layer_idx]
    pi = constants.pi[layer_idx]
    for x in unit_vectors[:20]:
        y = ref.rot_apply(rotation, sigma, pi, x)
        assert abs(y.norm().item() - x.norm().item()) < 1e-4


def test_rotation_inverse(constants, layer_idx, rotation, unit_vectors):
    sigma = constants.sigma[layer_idx]
    pi = constants.pi[layer_idx]
    for x in unit_vectors[:20]:
        recovered = ref.rot_apply_t(rotation, sigma, pi, ref.rot_apply(rotation, sigma, pi, x))
        assert torch.allclose(recovered, x, atol=1e-4, rtol=0)


# ---------- Block header packing ----------

def test_layer_byte_packs_layer_and_rotation(constants, unit_vectors, layer_idx, rotation):
    blk = ref.quantize_block(unit_vectors[0], constants, layer_idx=layer_idx, rotation=rotation)
    assert ref.extract_layer(blk[4]) == layer_idx
    assert ref.extract_rotation(blk[4]) == rotation


def test_different_rotations_produce_different_output(constants, unit_vectors, layer_idx):
    """With equal rotation-mode distributions, WHT and Haar blocks disagree."""
    x = unit_vectors[0]
    wht  = ref.quantize_block(x, constants, layer_idx=layer_idx, rotation=ref.TQP_ROT_WHT)
    haar = ref.quantize_block(x, constants, layer_idx=layer_idx, rotation=ref.TQP_ROT_HAAR)
    assert wht != haar


def test_different_layers_produce_different_output(constants, unit_vectors, rotation):
    x = unit_vectors[0]
    blk0 = ref.quantize_block(x, constants, layer_idx=0, rotation=rotation)
    blk1 = ref.quantize_block(x, constants, layer_idx=1, rotation=rotation)
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


# ---------- Paper-adjacent bounds (layer 0 only) ----------

def test_paper_mse_bound(constants, unit_vectors, rotation):
    """Stage-1 reconstruction MSE per vector stays below the paper's
    3-bit bound for both rotations."""
    bound_per_coord = math.sqrt(3) * math.pi / 2 * (1 / (4 ** 3))
    mse_sum = 0.0
    for x in unit_vectors:
        blk = ref.quantize_block(x, constants, layer_idx=0, rotation=rotation)
        x_hat = ref.dequantize_block(blk, constants)
        mse_sum += ((x - x_hat) ** 2).sum().item()
    mse_per_vector = mse_sum / len(unit_vectors)
    assert mse_per_vector < 1.5 * bound_per_coord


def test_paper_ip_correlation(constants, unit_vectors, rotation):
    """Inner-product correlation with true values must meet the paper's
    3-bit number (≥ 0.85) for both rotations."""
    half = len(unit_vectors) // 2
    queries = unit_vectors[:half]
    keys    = unit_vectors[half : 2 * half]
    true_ip = (queries * keys).sum(dim=-1)

    ref_ip = torch.zeros(half)
    for i in range(half):
        blk = ref.quantize_block(keys[i], constants, layer_idx=0, rotation=rotation)
        ref_ip[i] = ref.inner_product(queries[i], blk, constants)

    corr = torch.corrcoef(torch.stack([true_ip, ref_ip]))[0, 1].item()
    assert corr > 0.85, f"IP correlation {corr:.3f} below threshold (rotation={ROTATION_IDS[rotation]})"


# ---------- Paper-faithful byte-exact check (Haar only) ----------

def test_haar_byte_exact_vs_turboquant(constants, layer_idx, unit_vectors):
    """Under TQP_ROT_HAAR, the stored indices and QJL signs must be
    byte-identical to turboquant.py::TurboQuantProd(seed=42+layer_idx).
    This is the paper-faithfulness guarantee and was never claimed for WHT."""
    d = constants.d
    qp = TurboQuantProd(d, bits=4, seed=42 + layer_idx)
    oracle = qp.quantize(unit_vectors)
    oracle_idx = oracle["mse_indices"]
    oracle_signs_bits = (oracle["qjl_signs"] < 0).to(torch.uint8)

    qs_off = ref._qs_offset()
    signs_off = ref._signs_offset(d)

    for i, x in enumerate(unit_vectors):
        blk = ref.quantize_block(x, constants, layer_idx=layer_idx,
                                 rotation=ref.TQP_ROT_HAAR)
        qs = blk[qs_off : qs_off + (d * 3) // 8]
        ref_idx = ref._unpack_indices_bitplane(qs, d)
        signs = blk[signs_off : signs_off + d // 8]
        ref_signs_bits = (ref._unpack_signs(signs, d) < 0).to(torch.uint8)

        assert torch.equal(ref_idx, oracle_idx[i]), f"idx mismatch vec {i} layer {layer_idx}"
        assert torch.equal(ref_signs_bits, oracle_signs_bits[i]), \
            f"sign mismatch vec {i} layer {layer_idx}"


# ---------- Runtime rotation resolver ----------

import os

ROTATION_SOURCES = ["explicit", "thread", "process", "compile_time"]


@pytest.fixture(params=ROTATION_SOURCES)
def rotation_source(request):
    return request.param


class TestRotationResolver:
    """Verify the four-tier rotation resolution: explicit > thread > process > compile_time."""

    def _resolve_with_source(self, source, target_rot, layer_idx):
        """Set up the resolution state for a given source, return the resolved byte."""
        # Clear all overrides first
        ref.clear_thread_rotation()
        ref.set_default_rotation(ref._ROT_UNSET)

        if source == "explicit":
            # bit 6 = 1, bit 7 = target_rot
            lb = ref.layer_byte(layer_idx, target_rot)
            return ref.resolve_rotation(lb)
        elif source == "thread":
            ref.set_thread_rotation(target_rot)
            lb = ref.stored_byte(layer_idx, 0)  # bit 6 = 0, bit 7 irrelevant
            return ref.resolve_rotation(lb)
        elif source == "process":
            ref.set_default_rotation(target_rot)
            lb = ref.stored_byte(layer_idx, 0)
            return ref.resolve_rotation(lb)
        elif source == "compile_time":
            # No overrides: should always be WHT regardless of bit 7
            lb = ref.stored_byte(layer_idx, 0)
            return ref.resolve_rotation(lb)
        else:
            raise ValueError(f"unknown source: {source}")

    def test_resolved_rotation_matches_expectation(self, rotation_source):
        """Each source should produce the expected rotation."""
        layer = 5
        for target_rot in [ref.TQP_ROT_WHT, ref.TQP_ROT_HAAR]:
            resolved = self._resolve_with_source(rotation_source, target_rot, layer)

            expected_rot = target_rot
            if rotation_source == "compile_time":
                expected_rot = ref.TQP_ROT_WHT  # always WHT

            assert ref.extract_rotation(resolved) == expected_rot, \
                f"source={rotation_source} target={target_rot} resolved_rot={ref.extract_rotation(resolved)}"
            assert ref.extract_layer(resolved) == layer
            assert ref.extract_explicit(resolved) == 0  # bit 6 always cleared

        # Clean up
        ref.clear_thread_rotation()
        ref.set_default_rotation(ref._ROT_UNSET)

    def test_explicit_overrides_thread(self):
        """Per-call explicit (bit 6) overrides thread-local."""
        ref.set_thread_rotation(ref.TQP_ROT_HAAR)
        lb = ref.layer_byte(3, ref.TQP_ROT_WHT)  # explicit WHT
        resolved = ref.resolve_rotation(lb)
        assert ref.extract_rotation(resolved) == ref.TQP_ROT_WHT
        ref.clear_thread_rotation()

    def test_thread_overrides_process(self):
        """Thread-local overrides process default."""
        ref.set_default_rotation(ref.TQP_ROT_WHT)
        ref.set_thread_rotation(ref.TQP_ROT_HAAR)
        lb = ref.stored_byte(3, 0)  # bit 6 = 0
        resolved = ref.resolve_rotation(lb)
        assert ref.extract_rotation(resolved) == ref.TQP_ROT_HAAR
        ref.clear_thread_rotation()
        ref.set_default_rotation(ref._ROT_UNSET)

    def test_process_overrides_compile_time(self):
        """Process default overrides compile-time WHT."""
        ref.clear_thread_rotation()
        ref.set_default_rotation(ref.TQP_ROT_HAAR)
        lb = ref.stored_byte(3, 0)
        resolved = ref.resolve_rotation(lb)
        assert ref.extract_rotation(resolved) == ref.TQP_ROT_HAAR
        ref.set_default_rotation(ref._ROT_UNSET)

    def test_compile_time_fallback_is_wht(self):
        """With no overrides, compile-time default is WHT."""
        ref.clear_thread_rotation()
        ref.set_default_rotation(ref._ROT_UNSET)
        lb = ref.stored_byte(3, ref.TQP_ROT_HAAR)  # bit 7 = HAAR, but bit 6 = 0
        resolved = ref.resolve_rotation(lb)
        assert ref.extract_rotation(resolved) == ref.TQP_ROT_WHT

    def test_resolved_block_byte_matches_quantize(self, constants):
        """Block byte stored by quantize_block matches resolved rotation."""
        x = torch.randn(constants.d)
        for rot in [ref.TQP_ROT_WHT, ref.TQP_ROT_HAAR]:
            blk = ref.quantize_block(x, constants, layer_idx=5, rotation=rot)
            assert ref.extract_rotation(blk[4]) == rot
            assert ref.extract_layer(blk[4]) == 5
            assert ref.extract_explicit(blk[4]) == 0  # stored byte has bit 6 = 0
