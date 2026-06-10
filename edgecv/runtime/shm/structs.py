"""Single source of truth for shared-memory layouts (ARCHITECTURE.md §7.5).

Every shared header begins with MAGIC + ABI_VERSION, validated on attach. Any
change to a shared layout MUST bump ABI_VERSION and update both producer and
consumer.
"""

from __future__ import annotations

import ctypes

import numpy as np

MAGIC = 0xED6EC711          # "edgecv" tag; arbitrary but fixed
ABI_VERSION = 2

# numpy dtype <-> stable integer code. Append-only; never renumber.
_CODE_TO_NAME: dict[int, str] = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "int32",
    5: "int64",
    6: "float16",
    7: "float32",
    8: "float64",
    9: "complex64",
    10: "complex128",
    11: "bool",
}
_NAME_TO_CODE: dict[str, int] = {v: k for k, v in _CODE_TO_NAME.items()}


def dtype_to_code(dtype: np.dtype) -> int:
    name = np.dtype(dtype).name
    try:
        return _NAME_TO_CODE[name]
    except KeyError as e:
        raise ValueError(f"unsupported dtype for IPC: {name}") from e


def code_to_dtype(code: int) -> np.dtype:
    try:
        return np.dtype(_CODE_TO_NAME[code])
    except KeyError as e:
        raise ValueError(f"unknown dtype code: {code}") from e


class FrameControl(ctypes.Structure):
    """Control word published by the frame-ring producer (one writer)."""

    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("seq", ctypes.c_uint64),
        ("seqlock", ctypes.c_uint64),     # seqlock word (odd while writing)
        ("timestamp", ctypes.c_double),
        ("slot", ctypes.c_uint32),
        ("h", ctypes.c_uint32),
        ("w", ctypes.c_uint32),
        ("c", ctypes.c_uint32),
        ("dtype_code", ctypes.c_uint32),
    ]


class SearchROIControl(ctypes.Structure):
    """Control word for the search-ROI channel (caller → detector worker).

    Published by the caller/sender at full frame rate. Carries a normalised
    bounding box (the crop region for local detection).
    """

    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("seq", ctypes.c_uint64),
        ("seqlock", ctypes.c_uint64),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("w", ctypes.c_double),
        ("h", ctypes.c_double),
        ("timestamp", ctypes.c_double),
    ]


def validate_header(magic: int, abi_version: int) -> None:
    if magic != MAGIC:
        raise ValueError(f"bad shared-memory magic: {magic:#x} != {MAGIC:#x}")
    if abi_version != ABI_VERSION:
        raise ValueError(
            f"ABI mismatch: segment v{abi_version}, library v{ABI_VERSION}"
        )
