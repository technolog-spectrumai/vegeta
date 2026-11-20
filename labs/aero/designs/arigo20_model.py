"""ARIGO-20's flight model: ARIGO-1's machinery (``arigo1_model.build``: lattice + hand estimates) on ARIGO-20's
wing and H tail, with two ducted fans.

Hand inputs (marked, with margins): the mass 2.7 kg (two 70 mm fan units and a 6S pack instead of one motor, prop
and 4S pack: +0.2 kg on ARIGO-1's 2.5, +-0.3); the fans' net thrust T = T0 (1 - V / V_jet) with T0 = 2 x 20 N
(70 mm, 12 blades, 6S: the class's catalogue static thrust 2.0-2.3 kgf, +-25 %) and V_jet = 70 m/s (efflux,
+-15 %); CD0 0.034 (nacelles and pylons outside; the ducts' internal loss is in the net thrust, +-0.008). The
thrust line 66 mm above the fuselage axis: power pitches the nose down (the A-10's trait), in the model.
"""
from __future__ import annotations

import math
from functools import lru_cache

import arigo1_model as am
import arigo20 as a20

MASS = 2.7                         # fallback only: the mass comes from arigo_mass
THRUST = {"T0_N": 40.0, "V0_m_s": 70.0}
CD0 = 0.034
CRUISE = 30.0


@lru_cache(maxsize=None)
def coefficients() -> dict:
    q = a20.variant()
    p = a20.resolve(q)
    L = a20.Arigo20.layout(p)
    import arigo_mass
    mb = arigo_mass.budget("edf", p, L)
    C = am.build(q, p, L, "arigo20", "edf", mass=mb["total_g"] / 1000.0, CD0=CD0, thrust=dict(THRUST))
    C["mass_budget"] = mb
    C.update({"aircraft": "ARIGO-20", "aero_source": "model (lattice + hand)", "thrust_z_m": (p["edf_z"] / 1e3 - C["z_cg_m"]),
              "y_motor_m": p["edf_y"] / 1e3})
    t = cfd_table(C)
    if t is not None:
        # the CFD's drag includes the open ducts' internal walls, which the fans' net thrust already pays for: CD0 is
        # left at the model's (HAND), the measured CD(alpha) shifted down by the difference at its minimum
        t = dict(t)
        t["CD"] = t["CD"] - (t["CD"].min() - CD0)
        am.calibrate(C, t, CRUISE)
        C["aero_source"] = "cfd (screening RANS, ducts' internal drag removed)"
    return C


def cfd_table(C):
    """ARIGO-20's CFD rows (data/arigo20_cfd_screening.yaml), if computed for this geometry (S and CG match)."""
    import yaml
    f = am.DATA / "arigo20_cfd_screening.yaml"
    if not f.exists():
        return None
    d = yaml.safe_load(f.read_text()) or {}
    ref = d.get("reference", {})
    if abs(ref.get("S", 0) - C["S"]) > 1e-4 or abs(ref.get("cg_mm", [0])[0] - C["x_cg_m"] * 1e3) > 1.0:
        return None
    rows = am.clean_rows(d.get("rows", []))
    return am.table(rows) if len(rows) >= 6 else None


def get() -> dict:
    return dict(coefficients())


def top_speed() -> float:
    """Level flight's top speed [m/s] at full thrust (bisection on thrust - drag, the lift at that speed's alpha)."""
    C = get()
    W = C["mass"] * am.G
    def excess(V):
        qS = 0.5 * am.RHO * V * V * C["S"]
        CL = W / qS
        return C["thrust"]["T0_N"] * (1 - V / C["thrust"]["V0_m_s"]) - qS * (C["CD0"] + C["k"] * CL * CL)
    lo, hi = 15.0, C["thrust"]["V0_m_s"] - 0.1
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    return lo
