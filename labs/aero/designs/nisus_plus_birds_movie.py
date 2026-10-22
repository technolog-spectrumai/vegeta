"""Nisus+ Zero's bird hunt as a video: NISUS's bird movie (``nisus_birds_movie``: the synthetic onboard camera with the
mission software's boxes and tracks, the chase view, the map, the photo log) over the mountains.

Two things differ from NISUS's flat field, and only those are changed here: the onboard camera's synthetic ground is
the local terrain (a plane at the ground's height under the camera, not at z = 0 — the aircraft flies 3000 m above sea
level but 200-400 m above the slope), and the chase view's terrain is cropped around the flight (``nisus_plus_scenario``'s
crop: the whole 10 km height field would stretch the renderer's clipping range)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import nisus_plus_robot as fr
import nisus_plus_scenario as fsc
import nisus_birds_movie as nm

__all__ = ["render_movie"]


def render_movie(ep, path, *, massif: fr.Massif | None = None, **kw):
    """The MP4 (see the module and ``nisus_birds_movie.render_movie``); returns the path."""
    massif = massif or fr.Massif()
    original = nm._sky_ground

    def sky_ground(R_wc, p_wc, cam, size):
        p = np.asarray(p_wc, float).copy()
        p[2] = max(p[2] - float(massif.height(p[0], p[1])), 1.0)          # the ground as a plane at the height under the camera
        return original(R_wc, p, cam, size)

    view = SimpleNamespace(**{k: getattr(ep, k) for k in ("outcome", "meta", "birds", "bird_scenario", "controller") if hasattr(ep, k)})
    view.log = fsc._cropped(ep) if "geom_pose" in ep.log else ep.log
    nm._sky_ground = sky_ground
    try:
        return nm.render_movie(view, path, **kw)
    finally:
        nm._sky_ground = original
