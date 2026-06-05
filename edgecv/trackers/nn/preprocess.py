"""Pure numpy preprocessing for NN trackers (ARCHITECTURE.md §6.2, §7.4).

Module-level functions only, so a future spawned worker can import them. Numpy
reference today; RK RGA / DMA crop-resize can swap in behind this boundary later
with no tracker change (ARCHITECTURE §16). All crop<->frame and letterbox<->image
coordinate inversion lives here, in one place (ARCHITECTURE §5.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sample_clamped(img: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Bilinear sample at float (gx, gy) frame coords; clamp = edge-replicate."""
    h, w = img.shape[0], img.shape[1]
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    wx = (gx - x0).astype(np.float32)
    wy = (gy - y0).astype(np.float32)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    if img.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    ia, ib = img[y0c, x0c], img[y0c, x1c]
    ic, idd = img[y1c, x0c], img[y1c, x1c]
    top = ia * (1.0 - wx) + ib * wx
    bot = ic * (1.0 - wx) + idd * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def resize_bilinear(img: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize (H,W[,C]) to out_hw with centre-aligned bilinear sampling."""
    oh, ow = out_hw
    h, w = img.shape[0], img.shape[1]
    xs = (np.arange(ow) + 0.5) * (w / ow) - 0.5
    ys = (np.arange(oh) + 0.5) * (h / oh) - 0.5
    gx, gy = np.meshgrid(xs, ys)
    return _sample_clamped(np.asarray(img, dtype=np.float32), gx, gy)


@dataclass(frozen=True)
class CropXform:
    center: tuple[float, float]    # crop centre in frame px (cx, cy)
    size_px: tuple[float, float]   # crop side in frame px (sh, sw)
    out_size: tuple[int, int]      # resized output (oh, ow)

    def to_frame(self, out_xy: tuple[float, float]) -> tuple[float, float]:
        """Map an output-patch pixel index ``(ox, oy)`` back to frame pixel
        coordinates; the ``+0.5`` term is the half-pixel sampling offset."""
        ox, oy = out_xy
        oh, ow = self.out_size
        sh, sw = self.size_px
        cx, cy = self.center
        fx = (cx - sw / 2.0) + (ox + 0.5) / ow * sw
        fy = (cy - sh / 2.0) + (oy + 0.5) / oh * sh
        return fx, fy


def crop_with_context(
    frame: np.ndarray,
    center: tuple[float, float],
    size_px: tuple[float, float],
    out_size: tuple[int, int],
) -> tuple[np.ndarray, CropXform]:
    """Crop a (sh, sw)-px window centred at `center`, edge-replicate at borders,
    resize to out_size in one gather. Returns the patch and the inversion transform."""
    cx, cy = center
    sh, sw = size_px
    oh, ow = out_size
    fx = (cx - sw / 2.0) + (np.arange(ow) + 0.5) / ow * sw
    fy = (cy - sh / 2.0) + (np.arange(oh) + 0.5) / oh * sh
    gx, gy = np.meshgrid(fx, fy)
    patch = _sample_clamped(frame, gx, gy)
    if frame.ndim == 2:
        patch = patch.reshape(oh, ow)
    return patch, CropXform(center, (sh, sw), (oh, ow))
