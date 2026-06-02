#!/usr/bin/env python3
"""Host-only interactive webcam harness for qualitatively testing edgecv trackers.

Owns ONLY camera I/O and rendering; drives trackers through the public Tracker API
(init/update/status/name), exactly as a downstream user would. Not a runtime dependency
of the library (ARCHITECTURE.md §11, §13). Lives in repo-root tools/, excluded from the wheel.

Controls:
  [space] lock tracker on the white square    [r] release back to setup
  [+/-]   grow/shrink the selection square     [q]/[ESC] quit
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackStatus
from edgecv.core.tracker import Tracker
from edgecv.trackers.cf import Mosse

if TYPE_CHECKING:
    import cv2
else:
    try:
        import cv2
    except ImportError:  # host tool only; keep the module importable for unit tests
        cv2 = None

# --- constants ---
DEFAULT_BOX_PX = 96
MIN_BOX_PX = 24
BOX_STEP_PX = 16

# BGR colors (OpenCV order)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

TRACKERS: dict[str, Callable[[], Tracker]] = {
    "mosse": Mosse,
}


# --- pure helpers (unit-tested; no cv2) ---
def clamp_box_size(size_px: int, frame_h: int, frame_w: int) -> int:
    """Clamp a square side to [MIN_BOX_PX, min(frame_h, frame_w)]."""
    upper = min(frame_h, frame_w)
    if upper < MIN_BOX_PX:
        return upper
    return max(MIN_BOX_PX, min(size_px, upper))


def centered_square(frame_h: int, frame_w: int, size_px: int) -> PixelBox:
    """A centered square PixelBox of side ``size_px`` (clamped to the frame)."""
    side = clamp_box_size(size_px, frame_h, frame_w)
    x = (frame_w - side) / 2.0
    y = (frame_h - side) / 2.0
    return PixelBox(x=x, y=y, w=float(side), h=float(side))


def status_color(status: TrackStatus) -> tuple[int, int, int]:
    """Map track status to a BGR result-box color."""
    if status == TrackStatus.COASTING:
        return YELLOW
    if status == TrackStatus.LOST:
        return RED
    return ORANGE  # LOCKED / INITIALIZING -> nominal
