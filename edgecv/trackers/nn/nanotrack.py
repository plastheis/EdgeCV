"""NanoTrack V3 tracker (ARCHITECTURE.md §6.2). Single two-input graph
(exemplar, search) -> (cls, loc); MobileNetV3-small-v3 + AdjustLayer + DepthwiseBAN
anchor-free head. Reference defaults: HonglinChu/SiamTrackers NanoTrack configv3."""

from __future__ import annotations

import math
import time

import numpy as np

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.trackers.nn.base import UNSET, NNTracker, Template, resolve_pp
from edgecv.trackers.nn.preprocess import crop_with_context, points_grid, to_input


def _hann2d(n: int) -> np.ndarray:
    h = np.hanning(n).astype(np.float32)
    return np.outer(h, h).reshape(-1)


def _softmax_fg(cls: np.ndarray) -> np.ndarray:
    """cls (1,2,S,S) logits -> foreground prob per location, flattened (S*S,)."""
    c = np.asarray(cls, np.float32).reshape(2, -1)
    c = c - c.max(axis=0, keepdims=True)
    e = np.exp(c)
    return (e / e.sum(axis=0, keepdims=True))[1]


class NanoTrack(NNTracker):
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 exemplar_size=UNSET, search_size=UNSET, context=UNSET,
                 stride=UNSET, base_size=UNSET, penalty_k=UNSET,
                 window_influence=UNSET, size_lr=UNSET, color=UNSET, scale=UNSET,
                 score_lock=0.6, score_lost=0.35) -> None:
        super().__init__(manifest, backend=backend, model=model)
        pp = self._preprocessing
        self._exemplar_size = resolve_pp(exemplar_size, pp, "exemplar", 127)
        self._search_size = resolve_pp(search_size, pp, "search", 255)
        self._context = resolve_pp(context, pp, "context", 0.5)
        self._stride = resolve_pp(stride, pp, "stride", 16)
        self._base_size = resolve_pp(base_size, pp, "base_size", 7)
        self._penalty_k = resolve_pp(penalty_k, pp, "penalty_k", 0.138)
        self._window_influence = resolve_pp(window_influence, pp, "window_influence", 0.455)
        self._size_lr = resolve_pp(size_lr, pp, "size_lr", 0.348)
        self._color = resolve_pp(color, pp, "color", "rgb")
        self._scale = resolve_pp(scale, pp, "scale", 1.0)
        self._score_lock = score_lock
        self._score_lost = score_lost
        names = [o.name for o in self._model.io_spec.outputs]
        self._cls_name = "cls" if "cls" in names else names[0]
        self._loc_name = "loc" if "loc" in names else names[1]
        self._score_size = self._model.io_spec.outputs[0].shape[-1]
        self._points = points_grid(self._stride, self._score_size)   # (2, S*S)
        self._hann = _hann2d(self._score_size)
        self._template: Template | None = None
        self._box: BoundingBox | None = None

    def name(self) -> str:
        return "NanoTrack"

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
        z = to_input(patch, spec_z, color=self._color, scale=self._scale)
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

    def update(self, frame: np.ndarray) -> TrackResult:
        assert self._template is not None and self._box is not None, "init() first"
        h_img, w_img = frame.shape[0], frame.shape[1]
        pix = self._box.to_pixels(w_img, h_img)
        cx, cy = pix.center
        s_z = self._exemplar_side(pix)
        s_x = s_z * self._search_size / self._exemplar_size
        scale_z = self._exemplar_size / s_z                 # frame px -> search-crop px
        spec_x = self._model.io_spec.inputs[1]
        z = self._template.arrays["exemplar"]

        patch, _ = crop_with_context(frame, (cx, cy), (s_x, s_x),
                                     (self._search_size, self._search_size))
        x = to_input(patch, spec_x, color=self._color, scale=self._scale)
        out = self._model.infer({"exemplar": z, "search": x})

        score = _softmax_fg(out[self._cls_name])            # (S*S,)
        loc = np.asarray(out[self._loc_name], np.float32).reshape(4, -1)  # l,t,r,b
        px, py = self._points[0], self._points[1]           # search-crop px, centred at 0
        x1, y1 = px - loc[0], py - loc[1]
        x2, y2 = px + loc[2], py + loc[3]
        pred_cx, pred_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pred_w, pred_h = (x2 - x1), (y2 - y1)

        def _change(r):
            return np.maximum(r, 1.0 / r)

        def _sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        tw, th = pix.w * scale_z, pix.h * scale_z           # target size in search-crop px
        s_c = _change(_sz(pred_w, pred_h) / _sz(tw, th))
        r_c = _change((tw / th) / (pred_w / pred_h))
        penalty = np.exp(-(r_c * s_c - 1.0) * self._penalty_k)
        pscore = penalty * score
        pscore = (pscore * (1.0 - self._window_influence)
                  + self._hann * self._window_influence)
        best = int(pscore.argmax())

        lr = float(penalty[best] * score[best] * self._size_lr)
        new_cx = cx + float(pred_cx[best]) / scale_z
        new_cy = cy + float(pred_cy[best]) / scale_z
        new_w = pix.w * (1.0 - lr) + (float(pred_w[best]) / scale_z) * lr
        new_h = pix.h * (1.0 - lr) + (float(pred_h[best]) / scale_z) * lr

        new_pix = PixelBox(x=new_cx - new_w / 2.0, y=new_cy - new_h / 2.0,
                           w=new_w, h=new_h)
        self._box = BoundingBox.from_pixels(new_pix, w_img, h_img)

        conf = float(score[best])
        self._status = self._status_from(conf)
        self._seq += 1
        return TrackResult(bbox=self._box, confidence=conf, status=self._status,
                           timestamp=time.monotonic(), seq=self._seq)
