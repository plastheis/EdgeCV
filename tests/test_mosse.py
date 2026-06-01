import numpy as np
import pytest

from edgecv.trackers.cf.mosse import _crop_patch


def test_crop_patch_fully_inside_keeps_shape():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = _crop_patch(frame, center=(5.0, 5.0), size=(6, 6))
    assert patch.shape == (6, 6)


def test_crop_patch_edge_pads_when_window_crosses_border():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = _crop_patch(frame, center=(1.0, 1.0), size=(6, 6))
    assert patch.shape == (6, 6)
    # top-left is outside the frame; edge mode replicates frame[0, 0] == 0
    assert patch[0, 0] == frame[0, 0]


def test_crop_patch_preserves_channels():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    patch = _crop_patch(frame, center=(5.0, 5.0), size=(6, 6))
    assert patch.shape == (6, 6, 3)
