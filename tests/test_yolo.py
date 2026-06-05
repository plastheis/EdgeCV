import numpy as np
import pytest

from edgecv.backends.base import IOSpec, TensorSpec
from edgecv.trackers.nn.yolo import YoloDetector
from tests._nn_stubs import ScriptedModel

IN = 64  # small model input for tests


def _yolo_io(nc=80):
    return IOSpec(inputs=(TensorSpec("images", (1, 3, IN, IN), "float32"),),
                  outputs=(TensorSpec("output0", (1, -1, 5 + nc), "float32"),))


def _raw(dets, nc=80):
    """dets: list of (cx, cy, w, h, obj, cls_idx) in INPUT (letterbox) px."""
    out = np.zeros((1, len(dets), 5 + nc), np.float32)
    for i, (cx, cy, w, h, obj, ci) in enumerate(dets):
        out[0, i, :4] = [cx, cy, w, h]
        out[0, i, 4] = obj
        out[0, i, 5 + ci] = 1.0
    return {"output0": out}


def _detector(raw, **kw):
    return YoloDetector(model=ScriptedModel(_yolo_io(), [raw]), input_size=IN, **kw)


def test_detect_returns_normalised_xywh_and_score():
    # one detection centred in a square input -> centred normalised box
    det = _detector(_raw([(32, 32, 16, 16, 0.9, 3)]))
    out = det.detect(np.zeros((IN, IN, 3), np.uint8))
    assert out.boxes.shape == (1, 4)
    assert out.scores[0] == pytest.approx(0.9, abs=1e-3)  # obj * max(cls)=0.9*1.0
    bx, by, bw, bh = out.boxes[0]
    assert (bx + bw / 2) == pytest.approx(0.5, abs=0.05)


def test_detect_is_class_agnostic_and_pure():
    det = _detector(_raw([(32, 32, 16, 16, 0.8, 17), (10, 10, 8, 8, 0.7, 2)]))
    img = np.zeros((IN, IN, 3), np.uint8)
    out1 = det.detect(img)
    out2 = det.detect(img)            # purity: same result, no internal mutation
    assert len(out1.scores) == 2
    np.testing.assert_array_equal(out1.boxes, out2.boxes)


def test_detect_thresholds_low_confidence():
    det = _detector(_raw([(32, 32, 16, 16, 0.1, 0)]), conf_thresh=0.25)
    out = det.detect(np.zeros((IN, IN, 3), np.uint8))
    assert len(out.scores) == 0
