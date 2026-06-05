import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.trackers.nn.siamfc import SiamFC
from tests._nn_stubs import ScriptedModel, score_map_peaked, siam_io

SS = 17  # score_size


def _frame(h=240, w=320):
    return np.zeros((h, w, 3), np.uint8)


def _box():
    return BoundingBox(x=(160 - 20) / 320, y=(120 - 20) / 240, w=40 / 320, h=40 / 240)


def _siam(maps, **kw):
    return SiamFC(model=ScriptedModel(siam_io(SS), maps), **kw)


def test_name_and_instantiation():
    t = _siam([{"score_map": score_map_peaked(SS, 8, 8)}])
    assert t.name() == "SiamFC"


def test_init_builds_127_exemplar_template():
    # 3 maps because update() runs scale_num=3 infers; init runs 0 infers.
    t = _siam([{"score_map": score_map_peaked(SS, 8, 8)}])
    t.init(_frame(), _box())
    z = t.get_template().arrays["exemplar"]
    assert z.shape == (1, 1, 127, 127)
    assert t.status == TrackStatus.LOCKED
