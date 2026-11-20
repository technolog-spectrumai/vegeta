"""ARIGO-20's CFD cases (Aeromant / OpenFOAM), prepared here, RUN BY YOU (``sims/arigo_all.sh``): the airframe
without the fans running (the ducts open, flow through them), an angle-of-attack sweep at its 30 m/s cruise — CL,
CD, Cm about the CG; the stall, CD0, the static margin with the fuselage, nacelles and pylons in it. Steady RANS
k-omega SST, wall functions: trends, not absolutes.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import arigo1_cfd as ac
import arigo1_model as am
import arigo20 as a20
import arigo20_model as m20
from nisus_cfd import NU, QUALITY, RHO

ALPHAS = (-4.0, 0.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0)
SPEED = m20.CRUISE


def reference() -> dict:
    C = m20.get()
    return {"S": C["S"], "c": C["c"], "b": C["b"], "mass": C["mass"], "cg_mm": (C["x_cg_m"] * 1e3, C["z_cg_m"] * 1e3),
            "model": {k: C[k] for k in ("CL0", "CLa", "CL_MAX", "alpha_stall", "CD0", "k", "Cm0", "Cma", "x_np_m")}}


def l_ref() -> float:
    p = a20.resolve()
    return max(reference()["c"], max(p["span"], p["fuselage_length"] + 40.0) / 1000.0 / 3.9)


def export_stl(folder, alpha_deg):
    import arigo_stl

    def build(path):
        from vegeta import dedalus
        shape = a20.Arigo20().generate(**a20.variant()).shape.rotate((0, 0, 0), (0, 1, 0), alpha_deg)
        dedalus.Geometry.from_cadquery(shape, name="arigo20").export_stl(str(path), tolerance=0.2)
    return arigo_stl.stl(folder, "arigo20", alpha_deg, build)


def case(workdir, *, alpha_deg=0.0, V=SPEED, quality="screening", env=None):
    from vegeta import aeromant
    ref = reference()
    stl, tag = export_stl(Path(workdir) / "stl", alpha_deg)
    params = dict(velocity=V, kinematic_viscosity=NU, density=RHO, reference_area=ref["S"], reference_length=l_ref(),
                  center_of_rotation=tuple(ac.to_flow(*ref["cg_mm"], alpha_deg).tolist()), **QUALITY[quality])
    env = env or aeromant.OpenFOAMEnvironment.detect()
    return aeromant.CFDCase("rans_ksst_external", str(stl), params, workdir=str(Path(workdir) / f"arigo20{tag}_a{alpha_deg:+.0f}_V{V:.0f}_{quality}"),
                            geometry_units="mm", environment=env)


def model_row(alpha_deg, V=SPEED) -> dict:
    C = {**m20.get(), "thrust": {"T0_N": 0.0, "V0_m_s": 1.0}}
    al = math.radians(alpha_deg)
    F, M, _ = am.aero(C, (V * math.cos(al), 0.0, V * math.sin(al)), (0, 0, 0))
    qS = 0.5 * RHO * V * V * C["S"]
    return {"Cl": round((-F[2] * math.cos(al) + F[0] * math.sin(al)) / qS, 4), "Cd": round((-F[0] * math.cos(al) - F[2] * math.sin(al)) / qS, 4),
            "Cm": round(M[1] / (qS * C["c"]), 4)}


def row(alpha_deg, V, result) -> dict:
    out = {"alpha_deg": alpha_deg, "model": model_row(alpha_deg, V)}
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
