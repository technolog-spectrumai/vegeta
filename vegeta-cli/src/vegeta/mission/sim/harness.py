"""The closed loop for a simulation: the birds, the simulated detector, the mission stack and an aircraft.

``SimLoop`` holds the simulator side and feeds the stack each step; the aircraft is anything that gives a
``NavState`` (the kinematic one here, the MuJoCo Nisus in notebooks/designs/nisus_birds.py). It records the truth at
``sample_hz`` for the scores, judges every photo from the truth, and keeps what the video needs.
``run_kinematic`` flies a whole encounter with the kinematic aircraft in a few seconds of wall time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..camera import camera_pose
from ..messages import CameraFrame
from ..metrics import Sample, evaluate
from ..pipeline import MissionConfig, MissionStack
from .birds import BirdField
from .detector import SimDetector, SimDetectorConfig
from .kinematic import KinematicAircraft

__all__ = ["SimLoop", "judge_photo", "run_kinematic"]


def judge_photo(ev, truth, nav, frame, cam, pcfg):
    """The photo seen by the truth: which bird is in it (the one nearest the target's box centre), how many pixels it
    really spans, its true blur and centring, and whether it is good by the trigger's own thresholds."""
    uv, z = cam.project(truth.pos, frame.R_wc, frame.p_wc)
    vis = cam.in_frame(uv, z)
    if not vis.any():
        return {"bird": None, "good_true": False, "px_true": 0.0, "range_true": float("nan"), "blur_true": float("nan"), "centring_true": float("nan")}
    target = np.array([cam.cx, cam.cy])
    rng = np.linalg.norm(truth.pos - frame.p_wc, axis=1)
    px = cam.pixels_across(truth.size, rng) * 0.8
    d = np.where(vis, np.linalg.norm(uv - target, axis=1) - 5 * px, np.inf)     # the biggest bird near the centre
    i = int(np.argmin(d))
    c = (uv[i] - target) / np.array([cam.cx, cam.cy])
    centring = float(np.linalg.norm(c) / math.sqrt(2))
    blur = float(cam.blur_px(truth.vel[i] - nav.vel, truth.pos[i] - frame.p_wc, nav.omega_b, frame.R_wc))
    good = px[i] >= pcfg.good_px and blur <= pcfg.max_blur_px and centring <= pcfg.max_centring
    return {"bird": int(truth.ids[i]), "species": truth.species[i], "good_true": bool(good), "px_true": float(px[i]),
            "range_true": float(rng[i]), "blur_true": blur, "centring_true": centring}


class SimLoop:
    def __init__(self, birds: BirdField, cfg: MissionConfig, *, seed: int = 0, sim_det: SimDetectorConfig = SimDetectorConfig(),
                 sample_hz: float = 2.0, record_hz: float = 10.0, t0: float = 0.0):
        self.birds, self.cfg = birds, cfg
        self.detector = SimDetector(cfg.camera, cfg.detector, sim_det, seed=seed)
        self.stack = MissionStack(cfg, self.detector, t0=t0)
        self.stack.bus.subscribe("photos", "sim/judge")
        self.samples: list = []
        self.photos: list = []
        self.rec: list = []                 # per record tick: dict for the video
        self.frame_id = 0
        self.t_sample = t0
        self.t_rec = t0
        self.sample_dt, self.rec_dt = 1.0 / sample_hz, 1.0 / record_hz
        self.min_true_range = math.inf
        self.min_true_range_bird = None
        self.near_m = 10.0
        self.near: list = []
        self.n_photos_seen = 0
        self.active = True

    def step(self, t, nav, dt_birds):
        truth = self.birds.step(dt_birds, nav.pos, nav.vel)
        cam = self.cfg.camera
        R_wc, p_wc = camera_pose(cam, nav.R_wb, nav.pos)
        self.frame_id += 1
        frame = CameraFrame(t, self.frame_id, R_wc, p_wc, nav.omega_b)
        d = np.linalg.norm(truth.pos - nav.pos, axis=1)
        if len(d) and d.min() < self.min_true_range:
            self.min_true_range = float(d.min())
            self.min_true_range_bird = int(truth.ids[int(np.argmin(d))])
        if len(d) and d.min() < self.near_m:                         # a near miss: who, and what the mission was doing
            i = int(np.argmin(d))
            g = self.stack.bus.latest("guidance")
            sel = self.stack.bus.latest("target")
            last = self.near[-1] if self.near else None
            if last is None or t - last["t_end"] > 1.0 or last["bird"] != int(truth.ids[i]):
                uv, z = cam.project(truth.pos[i][None], R_wc, p_wc)
                self.near.append({"t": t, "t_end": t, "bird": int(truth.ids[i]), "species": truth.species[i], "min_m": float(d[i]),
                                  "mode": g.mode if g else "", "target": sel.track_id if sel else None,
                                  "in_view": bool(cam.in_frame(uv, z)[0]), "bird_fleeing": bool(truth.afraid[i])})
            else:
                last["t_end"] = t
                if d[i] < last["min_m"]:
                    last["min_m"] = float(d[i])
        bus = self.stack.bus
        if self.active:
            self.stack.tick(t, nav, frame, **{"sim/truth": truth})
        for ev in bus.take("photos", "sim/judge"):
            j = judge_photo(ev, truth, nav, frame, cam, self.cfg.photo)
            self.photos.append({"t": ev.t, "track": ev.track_id, "px_est": ev.pixels_across, "blur_est": ev.blur_px, "good_est": ev.good,
                                "range_est": ev.range_m, "score": ev.score, **j})
        tracks = bus.latest("tracks") if self.active else None
        if self.active and t >= self.t_sample - 1e-9:
            self.t_sample = t + self.sample_dt
            conf = [k for k in (tracks.items if tracks else ()) if k.status == "confirmed"]
            self.samples.append(Sample(t, truth.ids, truth.pos, truth.size, R_wc, p_wc, np.array([k.id for k in conf], int),
                                       np.array([k.pos for k in conf]).reshape(-1, 3)))
        if t >= self.t_rec - 1e-9:
            self.t_rec = t + self.rec_dt
            on = self.active
            dets = bus.latest("detections") if on else None
            sel = bus.latest("target") if on else None
            g = bus.latest("guidance") if on else None
            h = bus.latest("health") if on else None
            self.rec.append({"t": t, "nav_pos": nav.pos.copy(), "nav_vel": nav.vel.copy(), "R_wb": nav.R_wb.copy(), "R_wc": R_wc, "p_wc": p_wc,
                             "birds": truth.pos.copy(), "bird_ids": truth.ids.copy(), "bird_vel": truth.vel.copy(), "bird_afraid": truth.afraid.copy(),
                             "tracks": [(k.id, k.status, k.pos.copy(), k.cov[:3, :3].copy(), k.box) for k in (tracks.items if tracks else ())],
                             "dets": (dets.source, dets.roi, [(x.box, x.score) for x in dets.items], dets.t_capture) if dets else None,
                             "target": sel.track_id if sel else None, "done": tuple(sel.done) if sel else (),
                             "mode": g.mode if g else "", "note": g.note if g else "",
                             "health": (h.gpu_load, h.measured_hz, h.device, h.model) if h else None,
                             "n_photos": len(self.photos), "n_good": sum(p["good_true"] for p in self.photos), "active": on})
        return self.stack.command() if self.active else None

    def activate(self, t):
        """Start the mission software at ``t`` (its node clocks start there: no catch-up of the time before)."""
        self.active = True
        self.stack.ex.reset_clock(t)
        self.t_sample = self.t_rec = t

    def scores(self) -> dict:
        m = evaluate(self.samples, self.cfg.camera)
        good_birds = sorted({p["bird"] for p in self.photos if p["good_true"] and p["bird"] is not None})
        sch = self.stack.scheduler
        m.update({"photos": len(self.photos), "good photos (truth)": sum(p["good_true"] for p in self.photos),
                  "good photos (trigger's estimate)": sum(p["good_est"] for p in self.photos),
                  "birds photographed well": len(good_birds), "birds in the field": self.birds.n,
                  "targets done": len(sch.done), "targets given up": sum(1 for _, _, r in sch.history if r in ("timeout", "track lost", "track deleted", "left the height band")),
                  "closest approach to any bird [m]": self.min_true_range,
                  "approaches under 10 m": len(self.near), "approaches under 5 m": sum(n["min_m"] < 5 for n in self.near),
                  "breakoffs": sum(1 for e in self.stack.guidance.events if "→ breakoff" in e[2]),
                  "detector frames": self.detector.frames})
        return m


def run_kinematic(birds: BirdField, cfg: MissionConfig, *, duration=240.0, dt=0.02, start=(0.0, 0.0, 70.0), course=math.pi / 2,
                  wind=(0.0, 0.0, 0.0), seed=0, sim_det: SimDetectorConfig = SimDetectorConfig()):
    ac = KinematicAircraft(np.asarray(start, float), course=course, wind=wind)
    loop = SimLoop(birds, cfg, seed=seed, sim_det=sim_det)
    t = 0.0
    while t < duration:
        nav = ac.nav(t)
        cmd = loop.step(t, nav, dt)
        ac.step(cmd, dt)
        t += dt
    return loop
