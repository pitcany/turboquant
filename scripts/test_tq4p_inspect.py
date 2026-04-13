"""Round-trip test for tq4p_inspect.py GGUF metadata reader.

Creates a synthetic .gguf file with the tq4p.default_rotation KV pair
and one fake TQ4P_D128 tensor, then verifies the inspector reads both
the KV and the per-block rotation histogram correctly.
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tq4p_inspect import inspect_gguf, GGUF_MAGIC, GGUF_TYPE_STRING

# Also need the block size constant and layer byte packing.
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "patches" / "stage2-qjl" / "python"))


BLOCK_SIZE_D128 = 69
QK_D128 = 128
ALIGNMENT = 32


def _write_gguf_string(buf: bytearray, s: str) -> None:
    """Append a GGUF string (uint64 length + UTF-8) to buf."""
    encoded = s.encode("utf-8")
    buf += struct.pack("<Q", len(encoded))
    buf += encoded


def _make_synthetic_gguf(rotation_kv: str, n_blocks: int, block_rotation: int) -> bytes:
    """Build a minimal GGUF v3 file with one KV and one tensor.

    rotation_kv: "wht" or "haar" — the file-level KV value.
    n_blocks: number of TQ4P_D128 blocks in the tensor.
    block_rotation: 0 (WHT) or 1 (HAAR) — written into each block's layer_byte bit 7.
    """
    buf = bytearray()

    # Header
    n_tensors = 1
    n_kv = 1
    buf += struct.pack("<I", GGUF_MAGIC)     # magic
    buf += struct.pack("<I", 3)               # version
    buf += struct.pack("<Q", n_tensors)       # n_tensors
    buf += struct.pack("<Q", n_kv)            # n_kv

    # KV pairs
    _write_gguf_string(buf, "tq4p.default_rotation")
    buf += struct.pack("<I", GGUF_TYPE_STRING)
    _write_gguf_string(buf, rotation_kv)

    # Tensor info: name, n_dims, dims[], type, offset
    tensor_name = "test_tensor.weight"
    n_elements = n_blocks * QK_D128
    _write_gguf_string(buf, tensor_name)
    buf += struct.pack("<I", 1)               # n_dims = 1
    buf += struct.pack("<Q", n_elements)      # dims[0]
    buf += struct.pack("<I", 255)             # type (placeholder; inspector uses block size heuristic)
    buf += struct.pack("<Q", 0)               # offset (relative to data start)

    # Align to ALIGNMENT boundary
    header_end = len(buf)
    pad = (ALIGNMENT - (header_end % ALIGNMENT)) % ALIGNMENT
    buf += b"\x00" * pad

    # Tensor data: n_blocks x BLOCK_SIZE_D128 bytes
    for _ in range(n_blocks):
        block = bytearray(BLOCK_SIZE_D128)
        # layer_byte at offset 4: layer=0, rotation in bit 7
        block[4] = (block_rotation & 1) << 7
        buf += block

    return bytes(buf)


class TestGgufInspect:
    def test_kv_present_wht(self, tmp_path: Path):
        """File-level KV reads correctly for WHT."""
        gguf = _make_synthetic_gguf("wht", n_blocks=4, block_rotation=0)
        path = tmp_path / "test.gguf"
        path.write_bytes(gguf)

        info = inspect_gguf(path)
        assert info["kv_rotation"] == "wht"

    def test_kv_present_haar(self, tmp_path: Path):
        """File-level KV reads correctly for HAAR."""
        gguf = _make_synthetic_gguf("haar", n_blocks=4, block_rotation=1)
        path = tmp_path / "test.gguf"
        path.write_bytes(gguf)

        info = inspect_gguf(path)
        assert info["kv_rotation"] == "haar"

    def test_block_histogram_all_wht(self, tmp_path: Path):
        """Block histogram shows all WHT when blocks have rotation=0."""
        gguf = _make_synthetic_gguf("wht", n_blocks=10, block_rotation=0)
        path = tmp_path / "test.gguf"
        path.write_bytes(gguf)

        info = inspect_gguf(path)
        assert info["histogram"][0] == 10
        assert info["histogram"].get(1, 0) == 0

    def test_block_histogram_all_haar(self, tmp_path: Path):
        """Block histogram shows all HAAR when blocks have rotation=1."""
        gguf = _make_synthetic_gguf("haar", n_blocks=8, block_rotation=1)
        path = tmp_path / "test.gguf"
        path.write_bytes(gguf)

        info = inspect_gguf(path)
        assert info["histogram"].get(0, 0) == 0
        assert info["histogram"][1] == 8

    def test_kv_matches_block_bits(self, tmp_path: Path):
        """KV and per-block bits agree when the file is consistent."""
        for rot_name, rot_bit in [("wht", 0), ("haar", 1)]:
            gguf = _make_synthetic_gguf(rot_name, n_blocks=5, block_rotation=rot_bit)
            path = tmp_path / f"test_{rot_name}.gguf"
            path.write_bytes(gguf)

            info = inspect_gguf(path)
            assert info["kv_rotation"] == rot_name
            # All blocks should match the declared rotation
            assert info["histogram"][rot_bit] == 5, \
                f"Expected all blocks to be {rot_name}, got {info['histogram']}"

    def test_no_tq4p_tensor(self, tmp_path: Path):
        """File without TQ4P tensors reports no tensor found."""
        # Build a GGUF with just the KV and no tensor data that matches
        # TQ4P block sizes (use 0 blocks).
        buf = bytearray()
        buf += struct.pack("<I", GGUF_MAGIC)
        buf += struct.pack("<I", 3)
        buf += struct.pack("<Q", 0)  # n_tensors = 0
        buf += struct.pack("<Q", 1)  # n_kv = 1

        _write_gguf_string(buf, "tq4p.default_rotation")
        buf += struct.pack("<I", GGUF_TYPE_STRING)
        _write_gguf_string(buf, "wht")

        path = tmp_path / "empty.gguf"
        path.write_bytes(bytes(buf))

        info = inspect_gguf(path)
        assert info["kv_rotation"] == "wht"
        assert info["tensor_name"] is None
        assert info["n_blocks"] == 0
