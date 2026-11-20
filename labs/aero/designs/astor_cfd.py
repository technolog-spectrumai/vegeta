"""ASTOR's CFD cases (Aeromant / OpenFOAM), prepared here, RUN BY YOU (``sims/astor_cfd.sh``): the whole airframe,
propeller off (folded: not in the geometry), an angle-of-attack sweep — CL, CD, Cm about the CG; the stall, CD0 and
the static margin with the fuselage in it. Steady RANS k-omega SST with wall functions: trends, not absolutes.

Two speeds: the 30 m/s transit (the sweep: the stall, the polar, the margin) and the 65 m/s dash (three small
angles: the zero-lift drag at the Reynolds number that sets the top speed).
"""
from __future__ import annotations

import math
from pathlib import Path

import arigo1_cfd as ac
import arigo1_model as am
import astor as A
import astor_model as M
from nisus_cfd import NU, QUALITY, RHO

ALPHAS = (-4.0, 0.0, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
SPEED = M.CRUISE
DASH_ALPHAS = (-1.0, 0.0, 1.0)
DASH_SPEED = M.DASH


def reference() -> dict:
    C = M.get()
    return {"S": C["S"], "c": C["c"], "b": C["b"], "mass": C["mass"], "cg_mm": (C["x_cg_m"] * 1e3, C["z_cg_m"] * 1e3),
            "model": {k: C[k] for k in ("CL0", "CLa", "CL_MAX", "alpha_stall", "CD0", "k", "Cm0", "Cma", "x_np_m")}}


def l_ref() -> float:
    p = A.resolve()
    return max(reference()["c"], max(p["span"], p["fuselage_length"] + 40.0) / 1000.0 / 3.9)


def export_stl(folder, alpha_deg):
    import arigo_stl

    def build(path):
        from vegeta import dedalus
        shape = A.Astor().generate(**A.variant()).shape.rotate((0, 0, 0), (0, 1, 0), alpha_deg)
        dedalus.Geometry.from_cadquery(shape, name="astor").export_stl(str(path), tolerance=0.2)
    return arigo_stl.stl(folder, "astor", alpha_deg, build)


def case(workdir, *, alpha_deg=0.0, V=SPEED, quality="screening", env=None):
    from vegeta import aeromant
    ref = reference()
    stl, tag = export_stl(Path(workdir) / "stl", alpha_deg)
    params = dict(velocity=V, kinematic_viscosity=NU, density=RHO, reference_area=ref["S"], reference_length=l_ref(),
                  center_of_rotation=tuple(ac.to_flow(*ref["cg_mm"], alpha_deg).tolist()), **QUALITY[quality])
    env = env or aeromant.OpenFOAMEnvironment.detect()
    return aeromant.CFDCase("rans_ksst_external", str(stl), params, workdir=str(Path(workdir) / f"astor{tag}_a{alpha_deg:+.0f}_V{V:.0f}_{quality}"),
                            geometry_units="mm", environment=env)


def model_row(alpha_deg, V=SPEED) -> dict:
    C = {**M.get(), "thrust": {"T0_N": 0.0, "V0_m_s": 1.0}}
    al = math.radians(alpha_deg)
    F, Mo, _ = am.aero(C, (V * math.cos(al), 0.0, V * math.sin(al)), (0, 0, 0))
    qS = 0.5 * RHO * V * V * C["S"]
    return {"Cl": round((-F[2] * math.cos(al) + F[0] * math.sin(al)) / qS, 4), "Cd": round((-F[0] * math.cos(al) - F[2] * math.sin(al)) / qS, 4),
            "Cm": round(Mo[1] / (qS * C["c"]), 4)}


def row(alpha_deg, V, result) -> dict:
    out = {"alpha_deg": alpha_deg, "V": V, "model": model_row(alpha_deg, V)}
    if result is None or not getattr(result, "ok", False):
        out["status"] = "NOT RUN" if result is None or any("NOT RUN" in s for s in getattr(result, "messages", [])) else "FAILED"
        out["messages"] = [str(m)[:200] for m in list(getattr(result, "messages", []))[:3]]
        return out
    k = result.metrics
    f = lambda v: None if v is None else float(v)
    out.update({"status": "solved", "Cl": f(k.get("Cl")), "Cd": f(k.get("Cd")),
                "Cm_cg": None if k.get("Cm") is None else float(k.get("Cm")) * l_ref() / reference()["c"],
                "converged": k.get("converged"), "iterations": k.get("iterations"), "cells": k.get("mesh_cells"), "mesh_ok": k.get("mesh_ok")})
    return out


summary = ac.summary
