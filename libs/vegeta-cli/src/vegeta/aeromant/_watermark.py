"""The Vegeta logo as a small watermark in the bottom-right corner of every movie frame.

The logo is scaled to at most ``MAX_FRACTION`` of the frame's width and of its height (aspect ratio kept),
set ``MARGIN`` in from the corner and blended at ``OPACITY``. Each tool package keeps its own copy of this
module and of ``assets/logo.png`` (the packages do not import each other).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

LOGO = Path(__file__).parent / "assets" / "logo.png"
MAX_FRACTION = 0.10
MARGIN = 0.015
OPACITY = 0.85


@lru_cache(maxsize=8)
def _logo(width: int, height: int):
    """The logo (BGR uint8) sized for a ``width`` x ``height`` frame, or None if it cannot be read."""
    import cv2

    img = cv2.imread(str(LOGO), cv2.IMREAD_COLOR)
    if img is None:
        return None
    lh, lw = img.shape[:2]
    s = min(MAX_FRACTION * width / lw, MAX_FRACTION * height / lh)
    size = (max(1, int(lw * s)), max(1, int(lh * s)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def watermark(frame):
    """Blend the logo into the bottom-right corner of a BGR uint8 ``frame`` (in place) and return it."""
    h, w = frame.shape[:2]
    logo = _logo(w, h)
    if logo is None:
        return frame
    lh, lw = logo.shape[:2]
    m = int(round(MARGIN * min(w, h)))
    y0, x0 = h - lh - m, w - lw - m
    if y0 < 0 or x0 < 0:
        return frame
    roi = frame[y0:y0 + lh, x0:x0 + lw, :3].astype(np.float32)
    frame[y0:y0 + lh, x0:x0 + lw, :3] = np.clip(OPACITY * logo + (1 - OPACITY) * roi, 0, 255).astype(np.uint8)
    return frame
