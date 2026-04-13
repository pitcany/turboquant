#!/usr/bin/env python3
"""Inspect a .gguf file for TQ4P rotation metadata.

Reports:
  1. The file-level GGUF KV "tq4p.default_rotation" if present.
  2. A histogram of per-block rotation bits (bit 7 of layer_byte) from
     the first TQ4P-quantized tensor found in the file.

Usage:
    python3 scripts/tq4p_inspect.py path/to/model.gguf

Requires no external dependencies beyond Python 3.8+ stdlib.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Dict


# GGUF magic and header layout.
# Supports GGUF v2 and v3 (same layout for the fields we read).
GGUF_MAGIC = 0x46554747  # "GGUF" in LE
GGUF_MIN_VERSION = 2

# GGUF value types.
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# Block sizes.
BLOCK_SIZE_D128 = 69   # bytes per 128 elements
BLOCK_SIZE_D256 = 133  # bytes per 256 elements


def _read_string(f) -> str:
    """Read a GGUF string: uint64 length + UTF-8 bytes."""
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8")


def _read_value(f, vtype: int):
    """Read a single GGUF value of the given type."""
    if vtype == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif vtype == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif vtype == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif vtype == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif vtype == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == GGUF_TYPE_BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    elif vtype == GGUF_TYPE_STRING:
        return _read_string(f)
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    elif vtype == GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (n_elem,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, elem_type) for _ in range(n_elem)]
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def inspect_gguf(path: Path) -> dict:
    """Parse GGUF header and return TQ4P rotation info.

    Returns a dict with keys:
        "kv_rotation": str or None — file-level "tq4p.default_rotation" KV.
        "tensor_name": str or None — name of the first TQ4P tensor found.
        "histogram": dict — {0: count_wht, 1: count_haar} from block bytes.
        "n_blocks": int — total blocks inspected.
    """
    result = {
        "kv_rotation": None,
        "tensor_name": None,
        "histogram": {0: 0, 1: 0},
        "n_blocks": 0,
    }

    with open(path, "rb") as f:
        # Header: magic(4) + version(4) + n_tensors(8) + n_kv(8)
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic: {magic:#x})")
        version = struct.unpack("<I", f.read(4))[0]
        if version < GGUF_MIN_VERSION:
            raise ValueError(f"Unsupported GGUF version: {version}")
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))

        # Read KV pairs.
        kv = {}
        for _ in range(n_kv):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            val = _read_value(f, vtype)
            kv[key] = val

        result["kv_rotation"] = kv.get("tq4p.default_rotation")

        # Read tensor metadata to find the first TQ4P tensor.
        # Tensor info: name(string) + n_dims(uint32) + dims(uint64 * n_dims)
        #            + type(uint32) + offset(uint64)
        # We don't use the type ID directly — GGML assigns it dynamically
        # and TQ4P is a fork-specific addition, so we infer TQ4P tensors
        # by block-size heuristics over the data instead.

        tensor_infos = []
        for _ in range(n_tensors):
            name = _read_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            (ttype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            tensor_infos.append({
                "name": name,
                "dims": dims,
                "type": ttype,
                "offset": offset,
            })

        # Data starts at aligned offset after header.
        # GGUF v2/v3: data is aligned to `general.alignment` (default 32)
        # after header. Files may override via the KV pair of the same name.
        alignment_kv = kv.get("general.alignment")
        ALIGNMENT = int(alignment_kv) if alignment_kv is not None else 32
        if ALIGNMENT <= 0:
            ALIGNMENT = 32
        header_end = f.tell()
        data_start = (header_end + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

        # Try to find a TQ4P tensor by checking block sizes.
        #
        # Heuristic rejection: for a real TQ4P-quantized tensor, every
        # block's layer_byte at offset 4 must have bits 5 & 6 clear
        # (reserved bits in the stored layout). For a tensor of a
        # different dtype reinterpreted through our block size, ~3/4 of
        # bytes fail this check, so requiring ALL blocks pass is a
        # strong signature.
        #
        # Order matters: D128 is tried first because its 69 B block size
        # consumes the tensor data end-to-end at the right granularity
        # for actual TQ4P_D128 tensors. Trying D256 first on a D128
        # tensor would read half-blocks at the wrong stride and get
        # lucky-looking layer bytes from qjl_signs / qs regions. The
        # histogram validation still catches real D256 tensors on
        # fallback because a real D256 tensor reinterpreted at 69 B
        # stride puts byte 4 into random positions of each D256 block
        # almost always hitting a byte with bits 5 or 6 set.
        def _collect_histogram(abs_offset: int, n_blocks: int, block_size: int):
            """Read up to `n_blocks` blocks from `abs_offset`, validate
            reserved bits, and build a rotation histogram. Returns
            (histogram, usable) where `usable` is the actually inspected
            block count (may be < n_blocks on a truncated file), or None
            if any block fails validation or no bytes were available."""
            f.seek(abs_offset)
            data = f.read(n_blocks * block_size)
            usable = len(data) // block_size
            if usable == 0:
                return None
            # Always populate both rotation keys so consumers can index
            # histogram[0] / histogram[1] directly without KeyError when
            # all blocks happen to share a rotation.
            histogram: Dict[int, int] = {0: 0, 1: 0}
            for b in range(usable):
                lb = data[b * block_size + 4]
                if lb & 0x60:  # reserved bits must be 0 in stored blocks
                    return None
                r = (lb >> 7) & 1
                histogram[r] += 1
            return histogram, usable

        for ti in tensor_infos:
            n_elements = 1
            for d in ti["dims"]:
                n_elements *= d
            abs_offset = data_start + ti["offset"]

            matched = False
            # Try D128 first.
            if n_elements % 128 == 0:
                n_blocks = n_elements // 128
                res = _collect_histogram(abs_offset, n_blocks, BLOCK_SIZE_D128)
                if res is not None:
                    histogram, usable = res
                    result["tensor_name"] = ti["name"]
                    result["n_blocks"] = usable
                    result["histogram"] = histogram
                    matched = True

            # If D128 rejected, try D256.
            if not matched and n_elements % 256 == 0:
                n_blocks = n_elements // 256
                res = _collect_histogram(abs_offset, n_blocks, BLOCK_SIZE_D256)
                if res is not None:
                    histogram, usable = res
                    result["tensor_name"] = ti["name"]
                    result["n_blocks"] = usable
                    result["histogram"] = histogram
                    matched = True

            if matched:
                break

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: tq4p_inspect.py <path.gguf>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    info = inspect_gguf(path)

    print(f"File: {path}")
    print(f"GGUF KV tq4p.default_rotation: {info['kv_rotation'] or '(not set)'}")
    if info["tensor_name"]:
        print(f"First TQ4P tensor: {info['tensor_name']}")
        print(f"Blocks inspected: {info['n_blocks']}")
        print(f"Rotation histogram: WHT={info['histogram'].get(0, 0)}, "
              f"HAAR={info['histogram'].get(1, 0)}")
    else:
        print("No TQ4P tensors found in file.")


if __name__ == "__main__":
    main()
