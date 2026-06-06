"""Host-only model conversion framework (ARCHITECTURE.md §11). See tools/CONVERSION.md.

convert_lib is imported by tools/convert.py (which puts tools/ on sys.path). The registry
is torch-free; the harness, adapters, and rknn helpers import torch / rknn-toolkit2 lazily,
so importing the registry never pulls heavy deps.
"""

from __future__ import annotations

from .registry import Adapter, get, register, registered_names

__all__ = ["Adapter", "get", "register", "registered_names"]
