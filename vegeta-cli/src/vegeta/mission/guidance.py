"""Photo passes with a fixed camera: come in from behind, close slowly, shoot, break off, reposition, again.

A fixed-wing aircraft with the camera fixed in the nose cannot orbit a bird and keep it in view; it has to point at
it. A head-on pass closes at the sum of the speeds (25-30 m/s): the bird is in range for a second and the line of
sight turns fast (blur). From behind the closure is the difference (a few m/s): seconds of sharp frames with the bird
near the centre. So each target gets stern passes:

- ``search``: while no target is selected, a star of straight legs through the search centre (the points of a
  pentagram of ``search_radius``: each leg crosses the centre in a new direction, so the forward camera sweeps the
  area from all sides; an orbit would keep the centre off to the side, out of view).
- ``intercept``: towards the point ``behind_m`` behind the bird's predicted position (along its velocity; on its arc
  when the tracker's IMM says the bird is circling: a thermalling gull is met on its circle), at the
  dash speed when far and the cruise speed within ``slow_range``; the height is the bird's plus ``range x tan(tilt)``
  (the camera looks ``tilt`` below the nose: the bird then sits near the image centre), within ``h_min``/``h_max``.
- ``pass``: once within ``pass_range``, inside ``pass_cone_deg`` of the nose and not head-on (the courses within
  ``max_aspect_deg``): pursuit of the bird (aimed ``lead_s`` ahead) at the bird's speed plus a closure that shrinks to
  ``closure_min`` at ``photo_range`` (the aircraft holds behind the bird while the shutter works), never below
  ``v_min``.
- ``breakoff``: when any recent track (the target or another bird) is predicted inside the minimum separation within
  ``react_s``, or is inside it now: turn 90° away from its line of sight (to the side it is not flying to) and climb
  ``climb_m``, for ``breakoff_s``.
- ``reposition``: straight on until ``reposition_range`` from the bird (room for the next turn-in), then intercept.

The range is the tracker's estimate from the bird's size: uncertain (and biased when the species' wingspan differs
from the prior). Separation checks use ``range_margin`` x the estimate (the conservative side).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bus import Node, wrap
from .geometry import closest_approach, intercept_on_track, predict
from .messages import GuidanceCmd

__all__ = ["GuidanceConfig", "PhotoPassGuidance"]


@dataclass(frozen=True)
class GuidanceConfig:
    search_centre: tuple = (0.0, 300.0)
    search_radius: float = 150.0
    search_height: float = 70.0
    cruise: float = 15.0
    dash: float = 18.0
    v_min: float = 12.0             # 1.3 x the stall speed with margin (Nisus-Zero stalls near 8.2 m/s; bank needs more)
    slow_range: float = 140.0
    behind_m: float = 25.0
    pass_range: float = 60.0
    pass_cone_deg: float = 30.0
    max_aspect_deg: float = 75.0
    photo_range: float = 18.0
    closure_max: float = 8.0
    closure_min: float = 2.0
    lead_s: float = 1.0
    lead_cap_s: float = 15.0
    min_sep: float = 10.0
    range_margin: float = 0.65
    react_s: float = 2.5
    breakoff_s: float = 4.0
    climb_m: float = 10.0
    reposition_range: float = 100.0
    reposition_max_s: float = 12.0
    tilt_deg: float = 15.0
    h_min: float = 30.0
    h_max: float = 115.0
    recent_s: float = 2.0
    pass_stale_s: float = 1.0       # a pass needs a fresh track: a coasting prediction of a fleeing bird is not chased           # tracks updated this recently count for the separation check


class PhotoPassGuidance(Node):
    name = "guidance"

    def __init__(self, cfg: GuidanceConfig = GuidanceConfig()):
        self.cfg = cfg
        self.mode = "search"
        self.t_mode = 0.0
        self.target = None
        self.break_dir = 0.0
        self.break_h = cfg.search_height
        self.events: list = []
        self.star_k = 0

    def _set(self, mode, t, note=""):
        if mode != self.mode:
            self.events.append((t, self.target, f"{self.mode} → {mode}" + (f" ({note})" if note else "")))
            self.mode, self.t_mode = mode, t

    def _search(self, t, nav):
        cfg = self.cfg
        c = np.asarray(cfg.search_centre, float)
        a = math.radians(90.0 + 144.0 * self.star_k)
        p = c + cfg.search_radius * np.array([math.cos(a), math.sin(a)])
        if float(np.linalg.norm(p - nav.pos[:2])) < 30.0:
            self.star_k += 1
        course = math.atan2(p[1] - nav.pos[1], p[0] - nav.pos[0])
        return GuidanceCmd(t, "search", course, cfg.search_height, cfg.cruise, note=f"search leg {self.star_k % 5 + 1}/5")

    def _threat(self, nav, tracks):
        """The most urgent separation threat among the recent tracks: (track, estimated range, closest approach, t).
        During a stern pass the target's predicted closest approach is small by design (the aircraft aims at it and
        controls the closure): for the target it counts only when the bird closes faster than the pass would (it
        turned towards the aircraft); the range limit always counts."""
        cfg = self.cfg
        worst = None
        for k in (tracks.items if tracks is not None else ()):
            if k.since_update > cfg.recent_s or k.status == "tentative":
                continue
            r = k.pos - nav.pos
            rng = float(np.linalg.norm(r))
            v_rel = k.vel - nav.vel
            t_cpa, d_cpa = closest_approach(r, v_rel)
            firm = k.hits >= 6 and math.sqrt(max(np.trace(k.cov[3:6, 3:6]), 0.0)) < 8.0
            closing = -float(r @ v_rel) / max(rng, 1e-3)
            cpa_counts = not (self.mode == "pass" and k.id == self.target) or closing > cfg.closure_max + 3.0
            danger = cfg.range_margin * rng < cfg.min_sep or (firm and cpa_counts and t_cpa < cfg.react_s and cfg.range_margin * d_cpa < cfg.min_sep)
            if danger and (worst is None or rng < worst[1]):
                worst = (k, rng, d_cpa, t_cpa)
        return worst

    def step(self, t, bus):
        cfg = self.cfg
        nav, sel, tracks = bus.latest("nav"), bus.latest("target"), bus.latest("tracks")
        if nav is None:
            return
        tid = sel.track_id if sel is not None else None
        k = tracks.get(tid) if (tracks is not None and tid is not None) else None
        if tid != self.target:
            self.target = tid
            if self.mode not in ("breakoff",):
                self._set("intercept" if k is not None else "search", t, "new target")
        # separation first, whatever the mode
        th = self._threat(nav, tracks)
        if th is not None and self.mode != "breakoff":
            kk, rng, d_cpa, t_cpa = th
            r = kk.pos - nav.pos
            side = 1.0 if (kk.vel[0] * r[1] - kk.vel[1] * r[0]) > 0 else -1.0
            self.break_dir = math.atan2(r[1], r[0]) + side * math.pi / 2
            self.break_h = min(max(nav.pos[2], kk.pos[2]) + cfg.climb_m, cfg.h_max)
            self._set("breakoff", t, f"track {kk.id}: estimated {rng:.0f} m, closest approach {d_cpa:.0f} m in {t_cpa:.1f} s")
        if self.mode == "breakoff":
            if t - self.t_mode < cfg.breakoff_s:
                bus.publish("guidance", GuidanceCmd(t, "breakoff", self.break_dir, self.break_h, cfg.cruise, note=f"target {tid}"))
                return
            self._set("reposition" if k is not None else "search", t)
        if k is None:
            self._set("search", t)
            bus.publish("guidance", self._search(t, nav))
            return
        r = k.pos - nav.pos
        rng = float(np.linalg.norm(r))
        los = math.atan2(r[1], r[0])
        off_nose = abs(wrap(los - nav.course))
        vb = k.vel[:2]
        sb = float(np.linalg.norm(vb))
        u_b = vb / sb if sb > 3.0 else np.array([math.cos(los), math.sin(los)])
        aspect = abs(wrap(math.atan2(u_b[1], u_b[0]) - nav.course))          # 0: same direction (from behind)
        h = k.pos[2] + min(rng * math.tan(math.radians(cfg.tilt_deg)), 25.0)
        h = min(max(h, cfg.h_min), cfg.h_max)
        if self.mode == "reposition":
            if rng > cfg.reposition_range or t - self.t_mode > cfg.reposition_max_s:
                self._set("intercept", t)
            else:
                bus.publish("guidance", GuidanceCmd(t, "reposition", nav.course, h, cfg.cruise, note=f"target {tid} at {rng:.0f} m"))
                return
        if self.mode == "search":
            self._set("intercept", t)
        if self.mode == "intercept" and k.since_update < 0.5 and rng < cfg.pass_range and off_nose < math.radians(cfg.pass_cone_deg) and aspect < math.radians(cfg.max_aspect_deg):
            self._set("pass", t)
        elif self.mode == "pass" and (rng > 1.4 * cfg.pass_range or off_nose > math.radians(70)):
            self._set("intercept", t, "the bird left the nose")
        elif self.mode == "pass" and k.since_update > cfg.pass_stale_s:
            self._set("intercept", t, f"no detection for {k.since_update:.1f} s (the bird escaped the frame)")
        w = k.turn_rate if k.p_turn > 0.5 else 0.0                           # a circling bird: lead along its arc
        if self.mode == "pass":
            aim = predict(k.pos, k.vel, w, cfg.lead_s)
            course = math.atan2(aim[1] - nav.pos[1], aim[0] - nav.pos[0])
            closure = cfg.closure_min + (cfg.closure_max - cfg.closure_min) * float(np.clip((rng - cfg.photo_range) / 40.0, 0.0, 1.0))
            v_along = float(k.vel[:2] @ np.array([math.cos(nav.course), math.sin(nav.course)]))
            speed = float(np.clip(v_along + closure, cfg.v_min, cfg.dash))
        else:
            t_lead, w = intercept_on_track(nav.pos, k, cfg.dash if rng > cfg.slow_range else cfg.cruise, t_max=cfg.lead_cap_s)
            aim = predict(k.pos, k.vel, w, t_lead)
            v_then = predict(k.pos, k.vel, w, t_lead + 0.5) - aim                # the bird's direction at the intercept
            n = float(np.linalg.norm(v_then[:2]))
            u_then = v_then[:2] / n if n > 1e-3 else u_b
            aim[:2] -= cfg.behind_m * u_then * min(1.0, rng / cfg.pass_range)
            course = math.atan2(aim[1] - nav.pos[1], aim[0] - nav.pos[0])
            speed = cfg.dash if rng > cfg.slow_range else cfg.cruise
        bus.publish("guidance", GuidanceCmd(t, self.mode, course, h, speed, note=f"target {tid} at {rng:.0f} m"))
