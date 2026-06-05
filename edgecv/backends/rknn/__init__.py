"""RKNN backend adapter (ARCHITECTURE.md §10). Lazy: rknn-toolkit-lite2 is NOT on
PyPI and is installed manually on-device. This adapter reports unavailability
cleanly off-device and raises an actionable error if used without the runtime.
It must be initialised inside the worker process, never the parent."""

from __future__ import annotations

from typing import Any

from edgecv.backends.base import (
    InferenceBackend,
    IOSpec,
    Model,
    TensorSpec,
)
from edgecv.models.manifest import ModelManifest

_INSTALL_HINT = (
    "rknn-toolkit-lite2 is not available. It is not on PyPI; install it manually "
    "on the Rockchip device (see README RKNN note). The [rknn] extra only registers "
    "this adapter."
)


def _import_rknnlite():
    from rknnlite.api import RKNNLite  # type: ignore

    return RKNNLite


def _specs(entries: list[dict]) -> tuple[TensorSpec, ...]:
    return tuple(
        TensorSpec(
            name=e["name"],
            shape=tuple(e["shape"]),
            dtype=e.get("dtype", "float32"),
            layout=e.get("layout", "NCHW"),
            quant=e.get("quant"),
        )
        for e in entries
    )


class RknnModel(Model):
    """Wraps RKNNLite. Built INSIDE the using process only (ARCHITECTURE §14.7)."""

    def __init__(self, rknn: Any, io_spec: IOSpec, output_order: list[str]) -> None:
        self._rknn: Any = rknn
        self._io_spec = io_spec
        self._output_order = output_order

    @property
    def io_spec(self) -> IOSpec:
        return self._io_spec

    def infer(self, inputs: dict) -> dict:
        ordered = [inputs[s.name] for s in self._io_spec.inputs]
        results = self._rknn.inference(inputs=ordered)
        return dict(zip(self._output_order, results, strict=False))

    def close(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None


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
            rknn_lite = _import_rknnlite()
        except Exception as e:
            raise RuntimeError(_INSTALL_HINT) from e
        artifact = manifest.artifacts.get("rknn")
        if not artifact or "path" not in artifact:
            raise ValueError(f"manifest {manifest.name!r} has no rknn artifact path")
        rknn = rknn_lite()
        if rknn.load_rknn(artifact["path"]) != 0:
            raise RuntimeError(f"failed to load rknn model {artifact['path']!r}")
        core_mask = artifact.get("npu_core") or 0
        if rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError("rknn init_runtime failed")
        io_spec = IOSpec(
            inputs=_specs(manifest.inputs),
            outputs=_specs(manifest.outputs),
        )
        return RknnModel(rknn, io_spec, [o["name"] for o in manifest.outputs])
