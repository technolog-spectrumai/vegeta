"""ARIGO-2 in MuJoCo: ARIGO-1's simulator (``arigo1_mujoco``) with the two motors apart (``arigo2_model``).

``Sim2`` keeps ARIGO-1's control interface (de, da, dr, throttle) — so ARIGO-1's stability tests and manoeuvres run
unchanged on it — and adds: ``diff`` (the left motor's throttle above the right's), ``motor_ok`` (an engine out),
a yaw damper by differential thrust, and the engine-out protection (the good motor throttled back to what the
rudder can hold). The tests added here: engine out at cruise and at full power, open and closed loop, with and
without the protection; differential-thrust yaw authority.
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

import mujoco
import numpy as np

import arigo1_mujoco as amj
import arigo1_model as am
import arigo2 as a2
import arigo2_model as m2

DT = amj.DT


class Sim2(amj.Sim):
    def __init__(self, variant="hybrid", propulsion="twin", mesh=None, bandit=False):
        super().__init__(variant, "twin", mesh, bandit)
        self.C = m2.coefficients(variant)
        self.diff = 0.0
        self.motor_ok = [True, True]                         # left, right
        self.yaw_damper = True
        self.protect = True

    def throttles(self):
        thr = self.controls[3]
        tl, tr = thr + self.diff, thr - self.diff
        if self.yaw_damper:                                  # differential thrust damps yaw rate (gain by hand)
            r = self.state()["omega_frd"][2]
            tl, tr = tl - 0.15 * r, tr + 0.15 * r
        if not all(self.motor_ok) and self.protect:          # one out: the good one limited to what the rudder holds
            lim = m2.max_throttle_one_motor(self.C, max(self.state()["V"], 3.0))
            tl, tr = min(tl, lim), min(tr, lim)
        return (tl if self.motor_ok[0] else 0.0), (tr if self.motor_ok[1] else 0.0)

    def step(self, n=1):
        for _ in range(n):
            s = self.state()
            tl, tr = self.throttles()
            F, M, info = m2.aero2(self.C, s["v_air_frd"], s["omega_frd"], *self.controls[:3], tl, tr)
            R = self.R()
            self.d.xfrc_applied[self.bid, :3] = R @ (F * amj.FLIP)
            self.d.xfrc_applied[self.bid, 3:] = R @ (M * amj.FLIP)
            self.info = {**info, "load_factor": float(-F[2] / (self.C["mass"] * am.G))}
            mujoco.mj_step(self.m, self.d)


@contextlib.contextmanager
def as_arigo2():
    """ARIGO-1's tests and manoeuvres build ``amj.Sim``: point that at ``Sim2`` for the duration."""
    old = amj.Sim
    amj.Sim = Sim2
    try:
        yield
    finally:
        amj.Sim = old


def stability_tests(wing="hybrid", V=18.0, post_stall_cm_factor=1.0) -> dict:
    with as_arigo2():
        out = amj.stability_tests(wing, "twin", V, post_stall_cm_factor)
    out["aircraft"] = "ARIGO-2"
    out["engine_out"] = engine_out_tests(wing)
    out["differential_thrust_yaw"] = diff_thrust_test(wing, V)
    out["verdict"]["V_mc_full_power_m_s"] = round(m2.v_mc(m2.coefficients(wing)), 1)
    return out


def _engine_out(wing, V, closed_loop, protect, throttle=1.0, seconds=8.0):
    sim = Sim2(wing)
    sim.start(V, 80.0)
    sim.protect = protect
    ap = amj.Autopilot(sim)
    psi0 = sim.state()["psi"]

    def fn(t, s):
        if closed_loop:
            ap(phi_c=0.0, theta_c=math.radians(10.0 if V < 20 else 0.0), throttle=throttle)   # slow: a full-power climb keeps it slow
        else:
            s.controls[3] = throttle
        if t > 1.0:
            s.motor_ok[1] = False                            # the right motor stops
    h = amj.run(sim, seconds, fn)
    k = h["t"] > 1.0
    return {"speed_m_s": V, "closed_loop": closed_loop, "protection": protect,
            "max_sideslip_deg": round(float(np.abs(h["beta"][k]).max()), 1), "max_bank_deg": round(float(np.abs(h["phi"][k]).max()), 1),
            "heading_change_deg": round(float(np.degrees(np.unwrap(np.radians(h["psi"]))[-1]) - math.degrees(psi0)), 1),
            "altitude_change_m": round(float(h["z"][-1] - 80.0), 1), "speed_end_m_s": round(float(h["V"][-1]), 1),
            "stalled": bool(h["stall"].max() > 0.5), "departed": bool(np.abs(h["phi"][k]).max() > 60)}


def engine_out_tests(wing="hybrid") -> dict:
    """The right motor stops 1 s in, the left at full throttle: open loop (controls frozen), closed loop without the
    protection (the rudder alone), closed loop with it; at 16 m/s (a full-power climb, below V_mc) and 24 m/s level."""
    C = m2.coefficients(wing)
    out = {"V_mc_full_power_m_s": round(m2.v_mc(C), 1)}
    for V in (16.0, 24.0):
        out[f"V{V:.0f}"] = {"open_loop": _engine_out(wing, V, False, False), "rudder_only": _engine_out(wing, V, True, False),
                            "protected": _engine_out(wing, V, True, True)}
    return out


def diff_thrust_test(wing="hybrid", V=18.0) -> dict:
    """Rudder fixed, full differential throttle (left 1, right 0) for 2 s: the yaw rate and sideslip it gives."""
    sim = Sim2(wing)
    sim.start(V, 80.0)
    sim.yaw_damper = False

    def fn(t, s):
        s.controls[3] = 0.5
        s.diff = 0.5 if t < 2.0 else 0.0
    h = amj.run(sim, 3.0, fn)
    k = h["t"] < 2.0
    return {"peak_yaw_rate_deg_s": round(float(np.abs(h["r"][k]).max()), 1), "peak_sideslip_deg": round(float(np.abs(h["beta"][k]).max()), 1),
            "bank_after_2s_deg": round(float(np.interp(2.0, h["t"], h["phi"])), 1)}


MANOEUVRES = amj.MANOEUVRES + ("engine_out",)


def manoeuvre(name, sim, record=None):
    if name == "engine_out":                                 # climbing at full power, the right motor dies; the autopilot holds
        sim.start(16.0, 60.0)
        ap = amj.Autopilot(sim)

        def fn(t, s):
            ap(phi_c=0.0, theta_c=math.radians(8.0), throttle=1.0)
            if t > 2.0:
                s.motor_ok[1] = False
        h = amj.run(sim, 10.0, fn, record)
        return h, {"duration_s": round(float(h["t"][-1]), 2), "max_sideslip_deg": round(float(np.abs(h["beta"]).max()), 1),
                   "max_bank_deg": round(float(np.abs(h["phi"]).max()), 1), "altitude_change_m": round(float(h["z"][-1] - h["z"][0]), 1),
                   "min_speed_m_s": round(float(h["V"].min()), 1), "stalled": bool(h["stall"].max() > 0.5)}
    return amj.manoeuvre(name, sim, record)


def export_mesh(folder, wing="hybrid") -> str:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"arigo2_{wing}.stl"
    if not path.exists():
        a2.Arigo2().generate(**a2.variant(wing), show_props=True).export_stl(str(path), tolerance=0.5)
    return str(path)
