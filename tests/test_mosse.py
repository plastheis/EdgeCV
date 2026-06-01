import numpy as np
import pytest

from edgecv.trackers.cf.mosse import (
    _bilinear_sample,
    _crop_patch,
    _preprocess,
    _rand_warp,
    _subpixel_peak,
)


def test_crop_patch_fully_outside_frame_returns_filled_patch():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = _crop_patch(frame, center=(-100.0, -100.0), size=(6, 6))
    assert patch.shape == (6, 6)
    # window is entirely off the top-left; edge policy fills from the nearest pixel
    assert np.all(patch == frame[0, 0])


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
    assert patch[3, 3] == frame[1, 1]   # interior pixel maps to the correct frame location


def test_crop_patch_preserves_channels():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    patch = _crop_patch(frame, center=(5.0, 5.0), size=(6, 6))
    assert patch.shape == (6, 6, 3)


def test_bilinear_sample_identity_grid_returns_same_image():
    img = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
    ys, xs = np.indices((8, 8)).astype(np.float32)
    out = _bilinear_sample(img, xs, ys)
    np.testing.assert_allclose(out, img, atol=1e-5)


def test_rand_warp_preserves_shape_and_constant_image():
    rng = np.random.default_rng(1)
    const = np.full((16, 16), 5.0, np.float32)
    out = _rand_warp(const, rng)
    assert out.shape == (16, 16)
    np.testing.assert_allclose(out, 5.0, atol=1e-4)


def test_rand_warp_is_seed_deterministic():
    patch = np.random.default_rng(2).standard_normal((16, 16)).astype(np.float32)
    a = _rand_warp(patch, np.random.default_rng(7))
    b = _rand_warp(patch, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_preprocess_constant_patch_is_all_zero():
    # z-score of a constant has zero std -> zero; windowing keeps it zero.
    patch = np.full((16, 16, 3), 100, np.uint8)
    window = np.ones((16, 16), np.float32)
    out = _preprocess(patch, window)
    assert out.shape == (16, 16)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 0.0, atol=1e-5)


def test_subpixel_peak_interpolates_fractional_offset():
    r = np.zeros((5, 5), np.float32)
    r[2, 1], r[2, 2], r[2, 3] = 2.0, 4.0, 3.0  # peak at (2,2), skewed toward +x
    py, px = _subpixel_peak(r)
    assert py == pytest.approx(2.0, abs=1e-6)
    assert px == pytest.approx(2.0 + 0.5 * (2.0 - 3.0) / (2.0 - 8.0 + 3.0), abs=1e-6)
