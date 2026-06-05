import numpy as np
import pytest

from edgecv.trackers.nn.preprocess import crop_with_context, letterbox, resize_bilinear


def test_resize_bilinear_identity():
    img = np.random.default_rng(0).standard_normal((8, 8, 1)).astype(np.float32)
    out = resize_bilinear(img, (8, 8))
    np.testing.assert_allclose(out, img, atol=1e-5)


def test_resize_bilinear_changes_shape_keeps_channels():
    img = np.zeros((4, 4, 3), np.float32)
    out = resize_bilinear(img, (8, 16))
    assert out.shape == (8, 16, 3)


def test_crop_with_context_centre_and_shape():
    frame = np.zeros((100, 120, 3), np.uint8)
    frame[48:52, 58:62] = 200  # bright square at centre (~ (60, 50))
    patch, xf = crop_with_context(frame, (60.0, 50.0), (20.0, 20.0), (40, 40))
    assert patch.shape == (40, 40, 3)
    py, px = np.unravel_index(int(patch[..., 0].argmax()), patch.shape[:2])
    fx, fy = xf.to_frame((px, py))
    assert fx == pytest.approx(60.0, abs=2.0)
    assert fy == pytest.approx(50.0, abs=2.0)


def test_crop_with_context_edge_replicates_off_frame():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch, _ = crop_with_context(frame, (0.0, 0.0), (6.0, 6.0), (12, 12))
    assert patch.shape == (12, 12)
    assert np.isfinite(patch).all()


def test_crop_xform_to_frame_roundtrip():
    _, xf = crop_with_context(np.zeros((50, 50)), (25.0, 30.0), (10.0, 20.0), (32, 64))
    fx, fy = xf.to_frame((32.0 - 0.5, 16.0 - 0.5))  # (ow/2-0.5, oh/2-0.5)
    assert fx == pytest.approx(25.0, abs=1e-6)
    assert fy == pytest.approx(30.0, abs=1e-6)


def test_letterbox_preserves_aspect_and_pads():
    img = np.zeros((50, 100, 3), np.uint8)  # 2:1 wide
    out, xf = letterbox(img, (64, 64), pad_value=114)
    assert out.shape == (64, 64, 3)
    # 100->64 sets scale 0.64; height 50*0.64=32, padded symmetrically in 64
    assert xf.scale == pytest.approx(0.64)
    assert xf.pad[1] == pytest.approx((64 - 32) / 2.0)  # vertical pad


def test_letterbox_inverts_box():
    img = np.zeros((50, 100, 3), np.uint8)
    _, xf = letterbox(img, (64, 64))
    # a box covering the whole original maps from the unpadded letterbox region
    px, py = xf.pad
    s = xf.scale
    x1, y1, x2, y2 = xf.to_orig_xyxy((px, py, px + 100 * s, py + 50 * s))
    assert (x1, y1, x2, y2) == pytest.approx((0.0, 0.0, 100.0, 50.0), abs=1e-4)
