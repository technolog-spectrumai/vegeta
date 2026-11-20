"""ASTOR in MuJoCo: ARIGO-1's simulator (``arigo1_mujoco``) with ASTOR's coefficients (``astor_model``), tested at
its 30 m/s transit and its 60 m/s dash, flown through the family's manoeuvres 10 m/s faster, and against the targets
it exists for: fixed-wing drones at 30 and 50 m/s (``FAST_CHASES``).
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

import mujoco
import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import astor as A
import astor_model as M

V_TESTS = (M.CRUISE, 60.0)
V_EXTRA = 10.0
# the targets ASTOR exists for (the bandit weaves +-20 deg with a 20 s period, as the open-field scenario's)
FAST_CHASES = {
    "fast_30": dict(Vb=30.0, h=120.0, gap=300.0, T=45.0, V=45.0, what="a 30 m/s fixed wing, gentle weave, 300 m ahead"),
    "fast_50": dict(Vb=50.0, h=120.0, gap=300.0, T=60.0, V=60.0, what="a 50 m/s fixed wing, gentle weave, 300 m ahead (the requirement)"),
}


class SimAstor(amj.Sim):
    def __init__(self, variant=None, propulsion=None, mesh=None, bandit=False, C=None):
        self.C = C or M.get()
        self.m = mujoco.MjModel.from_xml_string(amj.mjcf(self.C, mesh, bandit))
        self.d = mujoco.MjData(self.m)
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arigo")
        self.controls = np.array([0.0, 0.0, 0.0, 0.5])
        self.wind = np.zeros(3)
        self.info = {}


@contextlib.contextmanager
def as_astor():
    old = amj.Sim
    amj.Sim = SimAstor
    try:
        yield
    finally:
        amj.Sim = old


def stability_tests(V=M.CRUISE, post_stall_cm_factor=1.0) -> dict:
    with as_astor():
        out = amj.stability_tests("astor", "pusher", V, post_stall_cm_factor)
    out["aircraft"] = "ASTOR"
    out["top_speed_m_s"] = round(M.top_speed(), 1)
    return out


MANOEUVRES = amj.MANOEUVRES


def manoeuvre(name, sim, record=None):
    return amj.manoeuvre(name, sim, record, V=amj.ENTRY[name][0] + V_EXTRA)


def fast_chase(sim_factory, name, starts=None) -> dict:
    """A FAST_CHASES scenario under proportional navigation from the perturbed starts (``arigo_perf.CHASE_STARTS``):
    the median time to 5 m, the catch rate, the closest approach."""
    import arigo_perf as P
    amj.CHASES[name] = FAST_CHASES[name]
    try:
        c = FAST_CHASES[name]
        runs = [amj.chase(sim_factory(bandit=True), name, V=c["V"] + dv, heading_deg=hd, y0=y)[1]
                for hd, y, dv in (starts or P.CHASE_STARTS)]
        return {**P._chase_stats(runs, name), "what": c["what"]}
    finally:
        amj.CHASES.pop(name, None)


def export_mesh(folder) -> str:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "astor.stl"
    if not path.exists():
        A.Astor().generate(**{**A.variant(), "show_props": True}).export_stl(str(path), tolerance=0.5)
    return str(path)
