"""RKNN backend adapter (ARCHITECTURE.md §10). Lazy: rknn-toolkit-lite2 is NOT on
PyPI and is installed manually on-device. This adapter reports unavailability
cleanly off-device and raises an actionable error if used without the runtime.
It must be initialised inside the worker process, never the parent."""

from __future__ import annotations

from edgecv.backends.base import InferenceBackend, Model
from edgecv.models.manifest import ModelManifest

_INSTALL_HINT = (
    "rknn-toolkit-lite2 is not available. It is not on PyPI; install it manually "
    "on the Rockchip device (see README RKNN note). The [rknn] extra only registers "
    "this adapter."
)


def _import_rknnlite():
    from rknnlite.api import RKNNLite  # type: ignore

    return RKNNLite


class RknnBackend(InferenceBackend):
    name = "rknn"

    def is_available(self) -> bool:
        try:
            _import_rknnlite()
        except Exception:
            return False
        return True

    def load(self, manifest: ModelManifest) -> Model:
        try:
            _import_rknnlite()
        except Exception as e:
            raise RuntimeError(_INSTALL_HINT) from e
        # Concrete RKNNLite model wiring lands with the first NN tracker; the
        # adapter is intentionally minimal in the foundation build.
        raise NotImplementedError(
            "RKNN model loading is implemented alongside the first NN tracker."
        )
