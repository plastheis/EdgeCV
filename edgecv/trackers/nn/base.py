"""NN tracker base: backend/model resolution, lifecycle, and the Template
appearance type (ARCHITECTURE.md §6.2). Trackers depend on the HAL only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgecv.backends.base import Model
from edgecv.backends.registry import available_backends, get_backend
from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.core.tracker import Tracker
from edgecv.models.manifest import ModelManifest, load_manifest


@dataclass
class Template:
    """Transferable target appearance — the NN analogue of CF FilterState."""
    arrays: dict[str, np.ndarray]
    bbox: BoundingBox
    meta: dict


def _select_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    avail = available_backends()
    for pref in ("rknn", "onnx"):
        if pref in avail:
            return pref
    raise RuntimeError(
        "no inference backend available; install edgecv[onnx], run on-device with "
        "[rknn], or pass backend='mock' for canned outputs"
    )


def resolve_model(manifest, backend: str, model: Model | None) -> Model:
    """DI seam: explicit model wins; else resolve a backend and load the manifest."""
    if model is not None:
        return model
    if manifest is None:
        raise ValueError("NNTracker needs a manifest (or an injected model=)")
    mf = manifest if isinstance(manifest, ModelManifest) else load_manifest(manifest)
    return get_backend(_select_backend(backend)).load(mf)


class NNTracker(Tracker):
    def __init__(self, manifest: ModelManifest | str | Path | None = None, *,
                 backend: str = "auto", model: Model | None = None) -> None:
        self._model: Model | None = resolve_model(manifest, backend, model)
        self._status: TrackStatus = TrackStatus.INITIALIZING
        self._seq: int = 0
        self._closed: bool = False

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:  # pragma: no cover
        raise NotImplementedError

    def update(self, frame: np.ndarray) -> TrackResult:  # pragma: no cover
        raise NotImplementedError

    def name(self) -> str:  # pragma: no cover
        raise NotImplementedError

    @property
    def status(self) -> TrackStatus:
        return self._status

    def close(self) -> None:
        if not self._closed and self._model is not None:
            self._model.close()
            self._closed = True
