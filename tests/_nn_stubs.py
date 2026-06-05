"""Deterministic stub Models for NN-tracker tests (no weights, no backend)."""
from __future__ import annotations

import numpy as np

from edgecv.backends.base import IOSpec, Model, TensorSpec


class ScriptedModel(Model):
    """Returns pre-set output arrays per infer() call, cycling through `outputs`.

    `outputs` is a list of dicts {output_name: ndarray}. io_spec is supplied so the
    tracker can read names/shapes. infer() ignores its inputs (geometry is driven
    entirely by the scripted outputs)."""

    def __init__(self, io_spec: IOSpec, outputs: list[dict[str, np.ndarray]]):
        self._io_spec = io_spec
        self._outputs = outputs
        self.calls = 0

    @property
    def io_spec(self) -> IOSpec:
        return self._io_spec

    def infer(self, inputs):
        out = self._outputs[self.calls % len(self._outputs)]
        self.calls += 1
        return out

    def close(self) -> None:
        self.closed = True


def siam_io(score_size: int = 17) -> IOSpec:
    return IOSpec(
        inputs=(TensorSpec("exemplar", (1, 1, 127, 127), "float32"),
                TensorSpec("search", (1, 1, 255, 255), "float32")),
        outputs=(TensorSpec("score_map", (1, 1, score_size, score_size), "float32"),))


def score_map_peaked(score_size: int, cy: int, cx: int, peak: float = 1.0) -> np.ndarray:
    m = np.zeros((1, 1, score_size, score_size), np.float32)
    m[0, 0, cy, cx] = peak
    return m
