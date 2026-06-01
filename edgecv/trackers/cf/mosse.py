"""MOSSE correlation-filter tracker (Bolme et al. 2010).

Grayscale, no scale adaptation. Implements the full transferable-filter contract
(ARCHITECTURE.md §6.1) on top of edgecv.trackers.cf.ops. Desired-output Gaussian
peaks at the window centre, so target displacement is peak - centre (no fftshift
wrap)."""

from __future__ import annotations

import time

import numpy as np

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.trackers.cf import ops
from edgecv.trackers.cf.base import CorrelationFilterTracker, EvalResult, FilterState


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
