"""The shutter: fire a full-resolution still of the selected bird when the picture would be good.

From the target's last detection (fresh within ``fresh_s``): the bird's width in pixels (measured, so it does not
depend on the uncertain range), its distance from the image centre, and the motion blur predicted from the
tracker's relative velocity and the aircraft's rotation during the exposure. Fires at most every ``min_interval_s``
when the bird spans ``min_px`` or more; a photo counts as good with ``good_px``, blur under ``max_blur_px`` and the
bird within ``max_centring`` of the centre. The simulator judges the same photo from the truth afterwards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bus import Node
from .camera import PinholeCamera
from .messages import PhotoEvent

__all__ = ["PhotoConfig", "PhotoTrigger", "photo_score"]


@dataclass(frozen=True)
class PhotoConfig:
    min_px: float = 60.0
    good_px: float = 120.0
    max_blur_px: float = 2.0
    max_centring: float = 0.75
    min_interval_s: float = 0.25
    fresh_s: float = 0.25


def photo_score(px, blur, centring, cfg: PhotoConfig = PhotoConfig()):
    return float(min(px / (1.5 * cfg.good_px), 1.0) * math.exp(-blur / 3.0) * (1 - 0.5 * min(centring, 1.0)))


class PhotoTrigger(Node):
    name = "photo"

    def __init__(self, cam: PinholeCamera, cfg: PhotoConfig = PhotoConfig()):
        self.cam, self.cfg = cam, cfg
        self.t_last = -1e9

    def step(self, t, bus):
        cfg, cam = self.cfg, self.cam
        sel, tracks, nav = bus.latest("target"), bus.latest("tracks"), bus.latest("nav")
        if sel is None or sel.track_id is None or tracks is None or nav is None:
            return
        k = tracks.get(sel.track_id)
        if k is None or k.box is None or k.since_update > cfg.fresh_s or t - self.t_last < cfg.min_interval_s:
            return
        x0, y0, x1, y1 = k.box
        px = x1 - x0
        if px < cfg.min_px:
            return
        c = np.array([0.5 * (x0 + x1) - cam.cx, 0.5 * (y0 + y1) - cam.cy])
        centring = float(np.linalg.norm(c / np.array([cam.cx, cam.cy])) / math.sqrt(2))
        blur = cam.blur_px(k.vel - nav.vel, k.pos - nav.pos, nav.omega_b, None)
        good = px >= cfg.good_px and blur <= cfg.max_blur_px and centring <= cfg.max_centring
        self.t_last = t
        rng = float(np.linalg.norm(k.pos - nav.pos))
        bus.publish("photos", PhotoEvent(t, k.id, rng, px, blur, centring, photo_score(px, blur, centring, cfg), good))
