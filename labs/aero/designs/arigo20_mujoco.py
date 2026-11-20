"""ARIGO-20 in MuJoCo: ARIGO-1's simulator (``arigo1_mujoco``) with ARIGO-20's coefficients (``arigo20_model``),
its two fans (the thrust line above the CG pitches the nose down with power; one fan out yaws it, little: they sit
62 mm off the centre plane), tested at its cruise of 30 m/s and flown through ARIGO-1's manoeuvres 8 m/s faster.
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

import mujoco
import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import arigo20 as a20
import arigo20_model as m20

V_TEST = m20.CRUISE
V_EXTRA = 8.0


class Sim20(amj.Sim):
    def __init__(self, variant=None, propulsion=None, mesh=None, bandit=False):
        self.C = m20.get()
        self.m = mujoco.MjModel.from_xml_string(amj.mjcf(self.C, mesh, bandit))
        self.d = mujoco.MjData(self.m)
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arigo")
        self.controls = np.array([0.0, 0.0, 0.0, 0.5])
        self.wind = np.zeros(3)
        self.info = {}
        self.motor_ok = [True, True]

    def step(self, n=1):
        for _ in range(n):
            s = self.state()
            ok = sum(self.motor_ok)
            thr = self.controls[3] * ok / 2                       # the thrust of the fans still running
            F, M, info = am.aero(self.C, s["v_air_frd"], s["omega_frd"], *self.controls[:3], thr)
            if ok == 1:                                           # one fan: its thrust off the centre line yaws the nose
                T = am.max_thrust(self.C, s["v_air_frd"][0]) * thr
                M[2] += (1 if self.motor_ok[0] else -1) * self.C["y_motor_m"] * T
            R = self.R()
            self.d.xfrc_applied[self.bid, :3] = R @ (F * amj.FLIP)
            self.d.xfrc_applied[self.bid, 3:] = R @ (M * amj.FLIP)
            self.info = {**info, "load_factor": float(-F[2] / (self.C["mass"] * am.G))}
            mujoco.mj_step(self.m, self.d)


@contextlib.contextmanager
def as_arigo20():
    old = amj.Sim
    amj.Sim = Sim20
    try:
        yield
    finally:
        amj.Sim = old


def stability_tests(V=V_TEST, post_stall_cm_factor=1.0) -> dict:
    with as_arigo20():
        out = amj.stability_tests("arigo20", "edf", V, post_stall_cm_factor)
    out["aircraft"] = "ARIGO-20"
    out["top_speed_m_s"] = round(m20.top_speed(), 1)
    out["engine_out"] = engine_out(V)
    out["power_pitch"] = power_pitch(V)
    return out


def engine_out(V=V_TEST) -> dict:
    """The right fan stops 1 s into level flight at full power, controls frozen: sideslip, bank, heading after 6 s."""
    sim = Sim20()
    sim.start(V, 80.0)
    def fn(t, s):
        s.controls[3] = 1.0
        if t > 1.0:
            s.motor_ok[1] = False
    h = amj.run(sim, 6.0, fn)
    k = h["t"] > 1.0
    return {"max_sideslip_deg": round(float(np.abs(h["beta"][k]).max()), 1), "max_bank_deg": round(float(np.abs(h["phi"][k]).max()), 1),
            "heading_change_deg": round(float(h["psi"][-1] - h["psi"][0]), 1), "altitude_change_m": round(float(h["z"][-1] - 80.0), 1)}


def power_pitch(V=V_TEST) -> dict:
    """Throttle from trim to full, elevator held: the nose-down pitch the high thrust line gives (the pitch rate, the
    pitch angle change after 2 s)."""
    sim = Sim20()
    tr = sim.start(V, 80.0)
    def fn(t, s):
        s.controls[3] = 1.0
    h = amj.run(sim, 2.0, fn)
    return {"trim_throttle": round(float(tr["throttle"]), 3), "pitch_rate_min_deg_s": round(float(h["q"].min()), 1),
            "pitch_change_2s_deg": round(float(h["theta"][-1] - h["theta"][0]), 1), "speed_after_2s_m_s": round(float(h["V"][-1]), 1)}


MANOEUVRES = amj.MANOEUVRES


def manoeuvre(name, sim, record=None):
    return amj.manoeuvre(name, sim, record, V=amj.ENTRY[name][0] + V_EXTRA)


def export_mesh(folder) -> str:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "arigo20.stl"
    if not path.exists():
        a20.Arigo20().generate(**a20.variant()).export_stl(str(path), tolerance=0.5)
    return str(path)
