"""Importing this package registers every adapter. Add new adapters here."""

from __future__ import annotations

from . import siamfc  # noqa: F401  (import side effect: registers the adapter)
from . import yolo  # noqa: F401  (import side effect: registers the adapter)
