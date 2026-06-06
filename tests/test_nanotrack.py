import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.models.manifest import ModelManifest
from edgecv.trackers.nn.nanotrack import NanoTrack
from tests._nn_stubs import ScriptedModel, cls_peaked, loc_const, nano_io

S = 15  # score size


def _frame(h=240, w=320):
    return np.zeros((h, w, 3), np.uint8)


def _box():
    return BoundingBox(x=(160 - 20) / 320, y=(120 - 20) / 240, w=40 / 320, h=40 / 240)


def _nano(outputs, **kw):
    return NanoTrack(model=ScriptedModel(nano_io(S), outputs), **kw)


def _out(cy, cx, l=8.0, t=8.0, r=8.0, b=8.0, fg=8.0):
    return {"cls": cls_peaked(S, cy, cx, fg), "loc": loc_const(S, l, t, r, b)}


def test_name_and_instantiation():
    t = _nano([_out(S // 2, S // 2)])
    assert t.name() == "NanoTrack"


def test_init_builds_127_exemplar_template():
    t = _nano([_out(S // 2, S // 2)])
    t.init(_frame(), _box())
    z = t.get_template().arrays["exemplar"]
    assert z.shape == (1, 3, 127, 127)
    assert t.status == TrackStatus.LOCKED


def test_set_template_round_trips():
    t = _nano([_out(S // 2, S // 2)])
    t.init(_frame(), _box())
    tmpl = t.get_template()
    sb = BoundingBox(0.1, 0.1, 0.2, 0.2)
    t.set_template(tmpl, search_box=sb)
    assert t.get_template() is tmpl


def test_manifest_preprocessing_reaches_nanotrack():
    mf = ModelManifest(name="t", task="sot_template_matching",
                       preprocessing={"window_influence": 0.99, "context": 0.7})
    t = NanoTrack(mf, model=ScriptedModel(nano_io(S), [_out(S // 2, S // 2)]))
    assert t._window_influence == 0.99
    assert t._context == 0.7


def test_explicit_kwarg_overrides_manifest():
    mf = ModelManifest(name="t", task="sot_template_matching",
                       preprocessing={"window_influence": 0.99})
    t = NanoTrack(mf, model=ScriptedModel(nano_io(S), [_out(S // 2, S // 2)]),
                  window_influence=0.1)
    assert t._window_influence == 0.1


def test_nn_package_exports_nanotrack():
    import edgecv.trackers.nn as nn
    assert hasattr(nn, "NanoTrack")


def test_centred_peak_keeps_centre():
    # symmetric loc (l==r, t==b) + centred fg peak -> no displacement.
    t = _nano([_out(S // 2, S // 2)], window_influence=0.0)
    t.init(_frame(), _box())
    res = t.update(_frame())
    cx, cy = res.bbox.to_pixels(320, 240).center
    assert cx == pytest.approx(160.0, abs=2.0)
    assert cy == pytest.approx(120.0, abs=2.0)
    assert res.seq == 1


def test_offcentre_peak_moves_box_right():
    # fg peak one column to the +x of centre -> centre moves +x.
    t = _nano([_out(S // 2, S // 2 + 1)], window_influence=0.0)
    t.init(_frame(), _box())
    res = t.update(_frame())
    cx, _ = res.bbox.to_pixels(320, 240).center
    assert cx > 161.0


def test_larger_predicted_box_grows_box():
    # big symmetric distances at the centred peak -> predicted box wider than target.
    t = _nano([_out(S // 2, S // 2, l=40.0, t=40.0, r=40.0, b=40.0)],
              window_influence=0.0)
    t.init(_frame(), _box())
    w0 = t.get_template().bbox.w
    res = t.update(_frame())
    assert res.bbox.w > w0


def test_window_suppresses_far_peak():
    # one frame: a near peak and a far peak of equal fg logit; high window
    # influence makes the near peak win (centre stays near image centre).
    out = _out(S // 2, S // 2)
    out["cls"][0, 1, 0, 0] = 8.0                      # add an equal far peak at corner
    t = _nano([out], window_influence=0.9)
    t.init(_frame(), _box())
    res = t.update(_frame())
    cx, cy = res.bbox.to_pixels(320, 240).center
    assert cx == pytest.approx(160.0, abs=8.0)
    assert cy == pytest.approx(120.0, abs=8.0)


def test_high_score_locks_low_score_lost():
    locked = _nano([_out(S // 2, S // 2, fg=8.0)])
    locked.init(_frame(), _box())
    assert locked.update(_frame()).status == TrackStatus.LOCKED

    # all-zero cls logits -> fg prob 0.5 everywhere -> below score_lock(0.6) and
    # above score_lost(0.35) -> COASTING.
    coasting = _nano([{"cls": np.zeros((1, 2, S, S), np.float32),
                       "loc": loc_const(S, 8, 8, 8, 8)}])
    coasting.init(_frame(), _box())
    assert coasting.update(_frame()).status == TrackStatus.COASTING


def test_output_box_is_normalised_and_unclamped():
    # target near the right edge with a strong +x peak -> centre may exceed 1.0
    # and must be reported truthfully (no clamp).
    t = _nano([_out(S // 2, S - 1)], window_influence=0.0)
    edge = BoundingBox(x=0.95, y=0.5, w=0.1, h=0.1)
    t.init(_frame(), edge)
    res = t.update(_frame())
    assert isinstance(res.bbox, BoundingBox)
    assert res.confidence is not None
