"""The hull, in water: what a surface boat's and a submarine's hulls share, lifted from notebooks 12, 13 and 14 and
scenarios/run_scenario.py as they were. The geometry is the hull designs' (``survey_boat.SurveyBoat``,
``submarine.Submarine``, copied unchanged).

- ``ittc57_cf``: the ITTC-57 friction line (12 cell 12, 13 cell 8, 14 cell 2).
- ``boat_resistance``: friction times a form factor plus an assumed wave-resistance hump (12 cell 12).
- ``body_of_revolution_resistance``: bare body with Hoerner's form factor plus appendages (13 cell 8, 14 cell 2).
- ``double_body``: the hull below the waterline mirrored about it, for a CFD without a free surface (12 cell 13).
- ``nose_upstream``: the vehicle turned so the nose meets the flow (13 cell 10, 14 cell 9, run_scenario.sub).
"""
from __future__ import annotations

import math

SEA_WATER = {"density": 1025.0, "kinematic_viscosity": 1.05e-6}
G = 9.81


def ittc57_cf(V: float, L_m: float, nu: float = SEA_WATER["kinematic_viscosity"]) -> float:
    Re = V * L_m / nu
    return 0.075 / (math.log10(Re) - 2) ** 2


def boat_resistance(V: float, L_wl_m: float, S_wet_m2: float, *, form_factor: float = 1.25,
                    rho: float = SEA_WATER["density"], nu: float = SEA_WATER["kinematic_viscosity"]):
    """``(R, Fn, Cf, Cw)``: ITTC-57 friction times the form factor plus an assumed wave hump for a small chine hull."""
    Cf = ittc57_cf(V, L_wl_m, nu)
    Fn = V / math.sqrt(G * L_wl_m)
    Cw = 0.004 * (Fn / 0.45) ** 4 / (1 + (Fn / 0.45) ** 4) * 1.6
    q = 0.5 * rho * V ** 2
    return q * S_wet_m2 * (form_factor * Cf + Cw), Fn, Cf, Cw


def body_of_revolution_resistance(V: float, L_m: float, D_m: float, S_wet_m2: float, S_body_m2: float, *,
                                  rho: float = SEA_WATER["density"], nu: float = SEA_WATER["kinematic_viscosity"]):
    """``(R, R_body, R_appendages, Cf, k)``: the bare body with Hoerner's form factor ``k`` and the appendages (sail,
    fins) at 1.5 x 1.3 times flat-plate friction."""
    Cf = ittc57_cf(V, L_m, nu)
    k = 1.5 * (D_m / L_m) ** 1.5 + 7 * (D_m / L_m) ** 3
    R_body = 0.5 * rho * V ** 2 * S_body_m2 * Cf * (1 + k)
    R_app = 0.5 * rho * V ** 2 * (S_wet_m2 - S_body_m2) * Cf * 1.5 * 1.3
    return R_body + R_app, R_body, R_app, Cf, k


def double_body(hull_shape, p: dict, draft_mm: float, name: str = "double_body"):
    """The hull below the waterline, mirrored about it: a Dedalus geometry (12 cell 13)."""
    import cadquery as cq
    from vegeta import dedalus

    under = cq.Workplane("XY").add(hull_shape).intersect(
        cq.Workplane("XY").box(2 * p["length"], 2 * p["beam"], draft_mm, centered=(True, True, False)))
    return dedalus.Geometry.from_cadquery(under.union(under.mirror("XY", basePointVector=(0, 0, draft_mm))), name=name)


def nose_upstream(geometry, name: str = "vehicle_nose_upstream"):
    """The vehicle turned 180 deg about z, its nose to the oncoming flow along +x."""
    from vegeta import dedalus

    return dedalus.Geometry.from_cadquery(geometry.shape.rotate((0, 0, 0), (0, 0, 1), 180), name=name)
