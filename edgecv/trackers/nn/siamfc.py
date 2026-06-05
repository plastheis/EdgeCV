"""SiamFC tracker (ARCHITECTURE.md §6.2). Single two-input graph
(exemplar, search) -> score_map; multi-scale search adapts position and size.
Reference defaults: HonglinChu/SiamTrackers."""

from __future__ import annotations

import math
import time

import numpy as np

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.trackers.cf.ops import psr
from edgecv.trackers.nn.base import NNTracker, Template
from edgecv.trackers.nn.preprocess import crop_with_context, resize_bilinear, to_input


def _hann2d(n: int) -> np.ndarray:
    h = np.hanning(n).astype(np.float32)
    win = np.outer(h, h)
    s = win.sum()
    return win / s if s > 0 else win


class SiamFC(NNTracker):
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 exemplar_size=127, search_size=255, context=0.5,
                 total_stride=8, response_up=16, scale_num=3, scale_step=1.0375,
                 scale_penalty=0.9745, scale_lr=0.59, window_influence=0.176,
                 color="gray", score_lock=8.0, score_lost=4.0) -> None:
        super().__init__(manifest, backend=backend, model=model)
        self._exemplar_size = exemplar_size
        self._search_size = search_size
        self._context = context
        self._total_stride = total_stride
        self._response_up = response_up
        self._scale_num = scale_num
        self._scale_step = scale_step
        self._scale_penalty = scale_penalty
        self._scale_lr = scale_lr
        self._window_influence = window_influence
        self._color = color
        self._score_lock = score_lock
        self._score_lost = score_lost
        out = self._model.io_spec.outputs[0]
        self._out_name = out.name
        self._score_size = out.shape[-1]
        self._up_size = self._score_size * self._response_up
        self._hann = _hann2d(self._up_size)
        self._template: Template | None = None
        self._box: BoundingBox | None = None

    def name(self) -> str:
        return "SiamFC"

    def get_template(self) -> Template:
        assert self._template is not None, "init() must run first"
        return self._template

    def set_template(self, template: Template,
                     search_box: BoundingBox | None = None) -> None:
        self._template = template
        self._box = search_box if search_box is not None else template.bbox

    def _exemplar_side(self, pix: PixelBox) -> float:
        p = self._context * (pix.w + pix.h)
        return math.sqrt((pix.w + p) * (pix.h + p))

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        h_img, w_img = frame.shape[0], frame.shape[1]
        pix = bbox.to_pixels(w_img, h_img)
        s_z = self._exemplar_side(pix)
        patch, _ = crop_with_context(frame, pix.center, (s_z, s_z),
                                     (self._exemplar_size, self._exemplar_size))
        spec_z = self._model.io_spec.inputs[0]
        z = to_input(patch, spec_z, color=self._color)
        self._template = Template(arrays={"exemplar": z}, bbox=bbox, meta={"s_z": s_z})
        self._box = bbox
        self._status = TrackStatus.LOCKED
        self._seq = 0

    def _status_from(self, value: float) -> TrackStatus:
        if value >= self._score_lock:
            return TrackStatus.LOCKED
        if value >= self._score_lost:
            return TrackStatus.COASTING
        return TrackStatus.LOST

    def update(self, frame: np.ndarray) -> TrackResult:  # pragma: no cover
        raise NotImplementedError("update() is implemented in Task 7")
