"""ASTOR-V / ASTOR-BOOST CFD cases (Aeromant / OpenFOAM), prepared here, RUN BY YOU (``sims/astor_cfd.sh``): the
whole airframe, propeller folded away (not in the geometry), at three wing sweeps, an angle-of-attack sweep each —
CL, CD, Cm about the CG.

What they are for: the lattice says the sliding carriage keeps the static margin at 8-13 % from 12 to 60 deg of
sweep — the CFD checks it with the body in the flow; the drag per sweep (what tucking the wing buys, what the pod
costs); the maximum lift spread (the launch and the stall). Steady RANS k-omega SST with wall functions: trends.
Reference area and chord are fixed for every sweep (``astor_v_model.S_REF``, ``C_REF``).
"""
from __future__ import annotations

import math
from pathlib import Path

import arigo1_cfd as ac
import astor_v as V
import astor_v_model as VM
from nisus_cfd import NU, QUALITY, RHO

# sweep [deg] -> (speed m/s, tail fan, angles of attack)
CASES = {12.0: (25.0, 1.0, (-4.0, 0.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0)),
         35.0: (40.0, 1.0, (-2.0, 0.0, 2.0, 4.0, 8.0, 12.0)),
         60.0: (60.0, 0.6, (-2.0, 0.0, 1.0, 2.0, 4.0, 6.0))}
DESIGNS = ("v", "boost")


def key(design, sweep) -> str:
    return f"{design}_sweep{sweep:.0f}"


def _cls(design):
    if design == "boost":
        import astor_boost as B
        return B.AstorBoost, B.variant
    return V.AstorV, V.variant


def reference() -> dict:
    return {"S": VM.S_REF, "c": VM.C_REF, "cg_mm": [V.CG_MM, 0.0]}


def l_ref(design="v", sweep=12.0) -> float:
    """The CFD's reference length [m]: the reference chord, unless the body is over 4 of them (the template's limit:
    spread, the wing is 1.08 m across on a 0.20 m chord); then the largest dimension / 3.7, and ``row`` scales Cm back
    to the reference chord."""
    cls, var = _cls(design)
    p = cls().resolve(**var())
    length = (p.get("edf_x", 0.0) + p.get("edf_length", 0.0)) or p["fuselage_length"]
    body = max(V.wing_numbers(p, sweep)["span_mm"], max(length, p["fuselage_length"]) + 80.0) / 1000.0
    return max(VM.C_REF, body / 3.7)


def export_stl(folder, design, sweep, alpha_deg):
    import arigo_stl
    cls, var = _cls(design)
    tf = CASES[sweep][1]

    def build(path):
        from vegeta import dedalus
        shape = cls().generate(**var(wing_sweep_deg=sweep, tail_fan=tf)).shape.rotate((0, 0, 0), (0, 1, 0), alpha_deg)
        dedalus.Geometry.from_cadquery(shape, name=f"astor_{design}").export_stl(str(path), tolerance=0.2)
    return arigo_stl.stl(folder, f"astor_{key(design, sweep)}", alpha_deg, build)


def case(workdir, design, sweep, *, alpha_deg=0.0, quality="screening", env=None):
    from vegeta import aeromant
    Vv = CASES[sweep][0]
    ref = reference()
    stl, tag = export_stl(Path(workdir) / "stl", design, sweep, alpha_deg)
    params = dict(velocity=Vv, kinematic_viscosity=NU, density=RHO, reference_area=ref["S"], reference_length=l_ref(design, sweep),
                  center_of_rotation=tuple(ac.to_flow(*ref["cg_mm"], alpha_deg).tolist()), **QUALITY[quality])
    env = env or aeromant.OpenFOAMEnvironment.detect()
    return aeromant.CFDCase("rans_ksst_external", str(stl), params,
                            workdir=str(Path(workdir) / f"astor_{key(design, sweep)}{tag}_a{alpha_deg:+.0f}_V{Vv:.0f}_{quality}"),
                            geometry_units="mm", environment=env)


def lattice_row(sweep, alpha_deg) -> dict:
    """The lattice's CL and Cm about the CG at this sweep and angle (no drag: inviscid), for the comparison."""
    r = VM.lattice(sweep, tail_fan=CASES[sweep][1], alpha=alpha_deg)
    return {"Cl": round(r["CL"], 4), "Cm": round(r["Cm"] + r["CL"] * V.CG_MM / 1000.0 / VM.C_REF, 4)}


def row(design, sweep, alpha_deg, result) -> dict:
    out = {"alpha_deg": alpha_deg, "V": CASES[sweep][0], "lattice": lattice_row(sweep, alpha_deg)}
    if result is None or not getattr(result, "ok", False):
        out["status"] = "NOT RUN" if result is None or any("NOT RUN" in s for s in getattr(result, "messages", [])) else "FAILED"
        out["messages"] = [str(m)[:200] for m in list(getattr(result, "messages", []))[:3]]
        return out
    k = result.metrics
    f = lambda v: None if v is None else float(v)
    out.update({"status": "solved", "Cl": f(k.get("Cl")), "Cd": f(k.get("Cd")),
                "Cm_cg": None if k.get("Cm") is None else float(k.get("Cm")) * l_ref(design, sweep) / VM.C_REF,
                "converged": k.get("converged"), "iterations": k.get("iterations"), "cells": k.get("mesh_cells"), "mesh_ok": k.get("mesh_ok")})
    return out


def summary(rows) -> dict:
    """Per configuration: CD0, the lift slope, the static margin (dCm/dCL), the maximum lift."""
    import numpy as np
    ok = [r for r in rows if r.get("status") == "solved" and r.get("Cl") is not None]
    if len(ok) < 3:
        return {"solved": len(ok)}
    al, cl, cd, cm = (np.array([r[k] for r in ok]) for k in ("alpha_deg", "Cl", "Cd", "Cm_cg"))
    lin = np.abs(al) <= 6.0
    out = {"solved": len(ok), "CD_min": float(cd.min()), "CL_max": float(cl.max()), "alpha_CL_max_deg": float(al[int(np.argmax(cl))])}
    if lin.sum() >= 2:
        a1 = float(np.polyfit(np.radians(al[lin]), cl[lin], 1)[0])
        m1 = float(np.polyfit(cl[lin], cm[lin], 1)[0])
        out.update({"CL_alpha_per_rad": a1, "static_margin": -m1, "drag_area_cm2": float(cd.min() * VM.S_REF * 1e4)})
    return out
