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
