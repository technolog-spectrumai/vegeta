"""One bird at a time: choose the next track to photograph, keep it with hysteresis, release it when done.

Cost of a candidate (confirmed, not yet done, not blacklisted, inside the height band and the area): the straight-line
intercept time at the dash speed, plus the turn needed to point at it (``w_turn`` s per radian), plus the track's
position uncertainty (``w_sigma`` s per metre). Birds the aircraft cannot catch (faster and going away) are skipped.
The current target is kept unless another one is cheaper by ``hysteresis`` (never during a photo pass, in the first
``commit_s`` on it or within ``commit_range`` of it: switching wastes the approach already flown); it is
released when ``n_good`` good photos are taken (done), after ``timeout_s`` on it, or when its track is lost for
``lost_release_s`` (both: blacklisted for ``blacklist_s``). Below ``min_margin_wh`` of energy margin no new target is
chosen (the autopilot's energy manager orders the return anyway).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bus import Node, wrap
from .geometry import intercept_time
from .messages import TargetSelection

__all__ = ["SchedulerConfig", "TargetScheduler"]


@dataclass(frozen=True)
class SchedulerConfig:
    n_good: int = 2
    timeout_s: float = 90.0
    lost_release_s: float = 3.0
    lost_release_manoeuvre_s: float = 22.0   # while breaking off or repositioning the bird is expected out of view
    blacklist_s: float = 30.0
    hysteresis: float = 0.5
    commit_s: float = 20.0          # no switch in the first seconds on a target ...
    commit_range: float = 150.0     # ... nor once the aircraft is this close to it
    dash_speed: float = 18.0
    w_turn: float = 4.0
    w_sigma: float = 0.3
    h_band: tuple = (25.0, 115.0)
    area_centre: tuple = (0.0, 0.0)
    area_radius: float = 500.0
    min_margin_wh: float = 1.0
    max_targets: int = 99


class TargetScheduler(Node):
    name = "scheduler"
    subscriptions = ("photos",)

    def __init__(self, cfg: SchedulerConfig = SchedulerConfig()):
        self.cfg = cfg
        self.current = None
        self.t_selected = 0.0
        self.good = {}                  # track id -> good photos
        self.shots = {}
        self.done: list = []
        self.blacklist: dict = {}       # id -> until
        self.history: list = []         # (t, id, reason)

    def cost(self, k, nav):
        cfg = self.cfg
        r = k.pos - nav.pos
        t_int = intercept_time(r, k.vel, cfg.dash_speed, t_max=120.0)
        if t_int is None:
            return math.inf
        bearing = math.atan2(r[1], r[0])
        turn = abs(wrap(bearing - nav.course))
        sigma = math.sqrt(max(np.trace(k.cov[:3, :3]), 0.0))
        return t_int + cfg.w_turn * turn + cfg.w_sigma * sigma

    def eligible(self, k, t):
        cfg = self.cfg
        if k.id in self.done or self.blacklist.get(k.id, -1) > t:
            return False
        if k.status != "confirmed":
            return False
        if not cfg.h_band[0] <= k.pos[2] <= cfg.h_band[1]:
            return False
        return float(np.hypot(*(k.pos[:2] - np.asarray(cfg.area_centre)))) <= cfg.area_radius

    def _release(self, t, reason, blacklist):
        if self.current is not None:
            if blacklist:
                self.blacklist[self.current] = t + self.cfg.blacklist_s
            self.history.append((t, self.current, reason))
        self.current = None

    def step(self, t, bus):
        cfg = self.cfg
        for ph in bus.take("photos", self.name):
            self.shots[ph.track_id] = self.shots.get(ph.track_id, 0) + 1
            if ph.good:
                self.good[ph.track_id] = self.good.get(ph.track_id, 0) + 1
        tracks, nav = bus.latest("tracks"), bus.latest("nav")
        guid = bus.latest("guidance")
        self._guid = guid
        if tracks is None or nav is None:
            return
        reason = "keep"
        cur = tracks.get(self.current) if self.current is not None else None
        if self.current is not None:
            if self.good.get(self.current, 0) >= cfg.n_good:
                self.done.append(self.current)
                self._release(t, "done", False); reason = "done"
            elif cur is None:
                self._release(t, "track deleted", True); reason = "track deleted"
            elif cur.status == "lost" and cur.since_update > (cfg.lost_release_manoeuvre_s if (guid is not None and guid.mode in ("breakoff", "reposition")) else cfg.lost_release_s):
                self._release(t, "track lost", True); reason = "track lost"
            elif t - self.t_selected > cfg.timeout_s:
                self._release(t, "timeout", True); reason = "timeout"
            elif not cfg.h_band[0] - 10 <= cur.pos[2] <= cfg.h_band[1] + 10:
                self._release(t, "left the height band", True); reason = "height band"
        in_pass = guid is not None and guid.mode in ("pass", "breakoff")
        low_energy = nav.energy_margin_wh < cfg.min_margin_wh
        enough = len(self.done) >= cfg.max_targets
        if not low_energy and not enough:
            cands = [(self.cost(k, nav), k.id) for k in tracks.items if self.eligible(k, t)]
            cands = [c for c in cands if math.isfinite(c[0])]
            if cands:
                c_best, best = min(cands)
                if self.current is None:
                    self.current, self.t_selected, reason = best, t, "selected"
                    self.history.append((t, best, "selected"))
                elif best != self.current and not in_pass and t - self.t_selected > cfg.commit_s and \
                        (cur is None or float(np.linalg.norm(cur.pos - nav.pos)) > cfg.commit_range):
                    c_cur = next((c for c, i in cands if i == self.current), math.inf)
                    if c_best < cfg.hysteresis * c_cur:
                        self._release(t, f"switched to {best}", False)
                        self.current, self.t_selected, reason = best, t, "switched"
                        self.history.append((t, best, "selected"))
        elif self.current is not None and low_energy and not in_pass:
            self._release(t, "energy", False); reason = "energy low"
        cost = math.nan
        if self.current is not None and tracks.get(self.current) is not None:
            cost = self.cost(tracks.get(self.current), nav)
        status = "low energy" if low_energy else ("enough targets" if enough else reason)
        bus.publish("target", TargetSelection(t, self.current, status, cost, tuple(self.done),
                                              tuple(k for k, until in self.blacklist.items() if until > t)))
