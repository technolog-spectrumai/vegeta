"""The mock flyer: the reference quad as a point mass with a saturated position–velocity loop following the racing
line's schedule, the same referee, the same statistics and the same trajectory format as the MuJoCo run — for
developing courses, rules and the notebook without the physics. Energy from the real drive table (the thrust the
acceleration demands, the inflow along it). Collisions: the quad as a sphere of ``radius`` against the obstacles'
geometry (no physics: it is detected, then the rules apply). ``wobble`` adds a lateral disturbance to exercise the
rules; ``crash_at`` forces a collision at a zone index to exercise the reset."""
from __future__ import annotations

import math

import numpy as np

from . import refquad
from .path import Path, carrot, resync
from .rules import Referee, Rules

G = 9.81


def schedule(course, *, safety: float = 0.8, v_cap: float | None = None) -> Path:
    p = refquad.propulsion()
    W = refquad.MASS_KG * G
    tw = 0.7 * 4 * p.max_thrust_N / W
    a_lat = G * math.sqrt(max(tw * tw - 1.0, 0.0))
    a_lat = min(a_lat, 3.0 * G)                                   # a pilot's 70° of tilt, not the motors' limit
    v_top = v_cap or 28.0                                         # a sustainable racing pace for a 5-inch quad (the model's 34 m/s sustained level, minus margin)
    return Path(course.points, course.radii, a_lateral=a_lat, a_brake=min(0.4 * a_lat, 1.2 * G), a_accel=min(0.5 * a_lat, 1.5 * G), v_top=v_top, safety=safety)


def run_mock(course, *, rules: Rules | None = None, dt: float = 0.02, lookahead_s: float = 0.35, omega: float = 3.0, radius: float = 0.16, wobble: float = 0.0,
             crash_at: int | None = None, seed: int = 0, safety: float = 0.8, v_cap: float | None = None, max_time: float = 400.0) -> dict:
    rules = rules or Rules()
    rng = np.random.default_rng(seed)
    path = schedule(course, safety=safety, v_cap=v_cap)
    path.zone_half = np.array([min(z.half) for z in course.zones])
    ref = Referee(course, rules)
    prop = refquad.propulsion()
    m = refquad.MASS_KG
    W = m * G
    a_lim = 0.7 * 4 * prop.max_thrust_N / m - G
    p = np.array(course.zones[0].center, dtype=float)
    v = np.zeros(3)
    s = 0.0
    t = 0.0
    hist, events, energy_J = [], [], 0.0
    hover_until = -1.0
    crashed_once = False
    while t < max_time and ref.status == "racing":
        if ref.reset_request is not None:
            p = ref.reset_request[0].copy(); v = np.zeros(3); hover_until = ref.reset_request[1]
            s = path.project(p, 0.0) if ref.last_checkpoint == 0 else path.s[ref.last_checkpoint]
            ref.reset_request = None
        if t < hover_until:
            p_ref, v_ref, a_ref = p.copy(), np.zeros(3), np.zeros(3)
        else:
            s = path.project(p, s)
            nz = min(ref.next, len(path.s) - 1)
            s, parked = resync(path, s, path.s[nz], nz >= len(path.s) - 1)
            p_ref, v_ref, a_ref = carrot(path, s, nz, lookahead_s=lookahead_s, parked=parked)
        a = a_ref - omega ** 2 * (p - p_ref) - 2 * omega * (v - v_ref)
        if wobble:
            a += rng.normal(0.0, wobble, 3) * np.array([1.0, 1.0, 0.3])
        na = np.linalg.norm(a)
        if na > a_lim:
            a *= a_lim / na
        T = m * np.linalg.norm(a + np.array([0.0, 0.0, G]))         # the thrust the manoeuvre needs
        thrust_dir = (a + np.array([0.0, 0.0, G])) / max(T / m, 1e-6)
        v_axial = max(float(np.dot(v, thrust_dir)), 0.0)
        T = min(T, 4 * prop.drive.t_max(v_axial))
        drag = -0.5 * refquad.RHO * refquad.CDA_FORWARD_M2 * np.linalg.norm(v) * v
        a_true = thrust_dir * T / m - np.array([0.0, 0.0, G]) + drag / m
        v = v + a_true * dt
        p = p + v * dt
        P = 4 * prop.drive.power(T / 4, v_axial) + refquad.FPV_W
        energy_J += P * dt
        t += dt
        # collisions: the sphere against the scene, or the forced one
        d, which = course.nearest_obstacle(p)
        contact = d < radius or p[2] < radius
        if crash_at is not None and not crashed_once and ref.next == crash_at + 1:
            contact, crashed_once = True, True
        airborne = t > 1.0 and hover_until < t
        ev = ref.update(t, p, contact, airborne)
        events += ev
        hist.append([t, *p, *v, T, P, float(np.linalg.norm(v))])
    H = np.array(hist)
    st = ref.stats()
    st.update({"course": course.name, "aircraft": "reference 5-inch quad", "mode": "mock", "energy [Wh]": energy_J / 3600, "Wh per km": energy_J / 3600 / (course.length() / 1e3),
               "max speed [m/s]": float(H[:, 9].max()), "mean speed [m/s]": course.length() / st["lap time [s]"] if st["status"] == "finished" else float("nan"),
               "peak power [W]": float(H[:, 8].max()), "mean power [W]": float(H[:, 8].mean()), "course length [m]": course.length(), "schedule time [s]": path.time_estimate(), "racing time [s]": t})
    return {"stats": st, "history": H, "events": events, "gate times": ref.gate_times, "path": path, "referee": ref, "course": course}
