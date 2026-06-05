"""Class-agnostic YOLO detector + standalone single-object tracker
(ARCHITECTURE.md §6.2; MAFiD local-detection mode, sensors-23-07082 §3.3).

YoloDetector.detect -> DetectorOutput is the reusable primitive a future hybrid
worker calls. YoloTracker wraps it for standalone use."""

from __future__ import annotations

import time

import numpy as np

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.fusion.policy import DetectorOutput
from edgecv.trackers.nn.base import NNTracker, resolve_model
from edgecv.trackers.nn.preprocess import (
    class_agnostic_nms,
    crop_with_context,
    letterbox,
    to_input,
)


class YoloDetector:
    """Boxes in the returned DetectorOutput are (N,4) normalised xywh top-left."""

    def __init__(self, manifest=None, *, backend="auto", model=None,
                 input_size=640, color="rgb", scale=1.0 / 255.0,
                 output_format="yolov5", conf_thresh=0.25, iou_thresh=0.45,
                 class_agnostic=True) -> None:
        self._model = resolve_model(manifest, backend, model)
        self._input_size = input_size
        self._color = color
        self._scale = scale
        self._output_format = output_format
        self._conf = conf_thresh
        self._iou = iou_thresh
        self._class_agnostic = class_agnostic
        self._spec = self._model.io_spec.inputs[0]
        self._out_name = self._model.io_spec.outputs[0].name

    def detect(self, image: np.ndarray) -> DetectorOutput:
        h_img, w_img = image.shape[0], image.shape[1]
        n = self._input_size
        lb, xf = letterbox(image, (n, n))
        inp = to_input(lb, self._spec, color=self._color, scale=self._scale)
        raw = np.asarray(self._model.infer({self._spec.name: inp})[self._out_name], np.float32)
        preds = raw[0]  # (N, 5+nc)
        if preds.shape[0] == 0:
            return DetectorOutput(boxes=np.empty((0, 4), np.float32),
                                  scores=np.empty((0,), np.float32))
        if self._output_format == "yolov5":
            xywh, obj, cls = preds[:, :4], preds[:, 4], preds[:, 5:]
            score = obj * (cls.max(axis=1) if cls.shape[1] > 0 else 1.0)
        else:  # "decoded": model already emits xywh + score
            xywh, score = preds[:, :4], preds[:, 4]
        keep = score >= self._conf
        xywh, score = xywh[keep], score[keep]
        if len(score) == 0:
            return DetectorOutput(boxes=np.empty((0, 4), np.float32),
                                  scores=np.empty((0,), np.float32))
        # centre xywh (letterbox px) -> xyxy (letterbox px)
        cxs, cys, ws, hs = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        xyxy = np.stack([cxs - ws / 2, cys - hs / 2, cxs + ws / 2, cys + hs / 2], axis=1)
        kept = class_agnostic_nms(xyxy, score, self._iou)
        xyxy, score = xyxy[kept], score[kept]
        # invert letterbox -> original px -> normalised xywh top-left
        boxes = np.empty((len(kept), 4), np.float32)
        for i, b in enumerate(xyxy):
            ox1, oy1, ox2, oy2 = xf.to_orig_xyxy((b[0], b[1], b[2], b[3]))
            boxes[i] = [ox1 / w_img, oy1 / h_img, (ox2 - ox1) / w_img, (oy2 - oy1) / h_img]
        return DetectorOutput(boxes=boxes, scores=score.astype(np.float32))

    def close(self) -> None:
        self._model.close()
