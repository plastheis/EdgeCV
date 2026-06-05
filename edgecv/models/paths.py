"""Artifact path resolution (ARCHITECTURE.md §10.1, §11).

Manifests carry relative artifact paths (e.g. ``siamfc_generic.onnx``). Model
blobs are host-only and gitignored, living under a models directory rather than
in the package. Backends resolve a relative path against ``$EDGECV_MODEL_DIR``
(default ``models``); absolute paths pass through unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_artifact_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    base = Path(os.environ.get("EDGECV_MODEL_DIR", "models"))
    return str(base / p)
