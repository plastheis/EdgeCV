"""MOSSE correlation-filter tracker (Bolme et al. 2010).

Grayscale, no scale adaptation. Implements the full transferable-filter contract
(ARCHITECTURE.md §6.1) on top of edgecv.trackers.cf.ops. Desired-output Gaussian
peaks at the window centre, so target displacement is peak - centre (no fftshift
wrap)."""

from __future__ import annotations

import numpy as np

from edgecv.trackers.cf import ops


def _crop_patch(
    frame: np.ndarray, center: tuple[float, float], size: tuple[int, int]
) -> np.ndarray:
    """Fixed-size patch centred at ``center`` (cx, cy) pixels, edge-padded at borders."""
    cx, cy = center
    th, tw = size
    h, w = frame.shape[0], frame.shape[1]
    x0 = int(round(cx - tw / 2.0))
    y0 = int(round(cy - th / 2.0))
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x0 + tw - w), max(0, y0 + th - h)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + tw), min(h, y0 + th)
    patch = frame[sy0:sy1, sx0:sx1]
    if px0 or px1 or py0 or py1:
        pad = [(py0, py1), (px0, px1)] + [(0, 0)] * (frame.ndim - 2)
        patch = np.pad(patch, pad, mode="edge")
    return patch


def _bilinear_sample(img: np.ndarray, src_x: np.ndarray, src_y: np.ndarray) -> np.ndarray:
    """Sample ``img`` at floating (src_x, src_y) coords with clamped bilinear interpolation."""
    h, w = img.shape[0], img.shape[1]
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    wx = (src_x - x0).astype(np.float32)
    wy = (src_y - y0).astype(np.float32)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    if img.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    ia, ib = img[y0c, x0c], img[y0c, x1c]
    ic, idd = img[y1c, x0c], img[y1c, x1c]
    top = ia * (1.0 - wx) + ib * wx
    bot = ic * (1.0 - wx) + idd * wx
    return (top * (1.0 - wy) + bot * wy).astype(img.dtype)


def _rand_warp(
    patch: np.ndarray,
    rng: np.random.Generator,
    max_rot_deg: float = 2.0,
    max_scale: float = 0.02,
) -> np.ndarray:
    """Small random rotation+scale about the patch centre (Bolme init augmentation).

    Rotation/scale keep the target centred, so the centred desired-output Gaussian
    stays valid across augmented samples.
    """
    h, w = patch.shape[0], patch.shape[1]
    ang = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
    scale = 1.0 + rng.uniform(-max_scale, max_scale)
    cos_a = np.cos(ang) / scale
    sin_a = np.sin(ang) / scale
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.indices((h, w)).astype(np.float32)
    xr, yr = xs - cx, ys - cy
    src_x = cos_a * xr + sin_a * yr + cx
    src_y = -sin_a * xr + cos_a * yr + cy
    return _bilinear_sample(patch, src_x, src_y)


def _preprocess(patch: np.ndarray, window: np.ndarray) -> np.ndarray:
    """MOSSE preprocessing: grayscale -> log -> z-score -> cosine window."""
    gray = ops.extract_raw(patch)[..., 0]
    x = np.log(gray + 1.0)
    x = (x - x.mean()) / (x.std() + 1e-5)
    return (x * window).astype(np.float32)


def _subpixel_peak(response: np.ndarray) -> tuple[float, float]:
    """Refined (py, px) peak location via per-axis parabolic interpolation."""
    h, w = response.shape
    iy, ix = np.unravel_index(int(np.argmax(response)), response.shape)
    py, px = float(iy), float(ix)
    if 0 < ix < w - 1:
        left, ctr, right = response[iy, ix - 1], response[iy, ix], response[iy, ix + 1]
        denom = left - 2.0 * ctr + right
        if denom != 0:
            px += 0.5 * (left - right) / denom
    if 0 < iy < h - 1:
        up, ctr, down = response[iy - 1, ix], response[iy, ix], response[iy + 1, ix]
        denom = up - 2.0 * ctr + down
        if denom != 0:
            py += 0.5 * (up - down) / denom
    return py, px
