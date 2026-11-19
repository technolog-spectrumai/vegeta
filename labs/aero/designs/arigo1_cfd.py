"""ARIGO-1's CFD cases (Aeromant / OpenFOAM) — prepared here, RUN BY YOU (``sims/arigo1.sh``), not in a session.

The question: what the lattice and the hand estimates in ``arigo1_model`` cannot give — the **stall** (CL_max, the
angle, whether the break is gentle and the nose drops: the brief's "predictable stall, rapid recovery"), the
**zero-lift drag** of the real fuselage, camera and nacelles, and the **pitching moment about the CG** through the
stall (the neutral point with the fuselage in it, which the lattice leaves out).

The cases: the whole airframe without propellers, an angle-of-attack sweep -4 to 24 degrees at 18 m/s (cruise),
for each wing variant x propulsion layout asked. Steady RANS k-omega SST, wall functions, no transition model:
trends, not absolutes; beyond the stall the flow is unsteady and a steady solver gives its mean at best — the
coefficients' scatter over the last iterations is written out. Aeromant reports Cl, Cd, Cm only: sideslip cases
(Cn_beta, Cl_beta) need its force-axis template extended and are not here.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import arigo1 as a1
import arigo1_model as am
from nisus_cfd import NU, QUALITY, RHO

ALPHAS = (-4.0, 0.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0)
SPEED = 18.0


def reference(variant: str, propulsion: str) -> dict:
    C = am.coefficients(variant, propulsion)
    return {"S": C["S"], "c": C["c"], "b": C["b"], "mass": C["mass"], "cg_mm": (C["x_cg_m"] * 1e3, C["z_cg_m"] * 1e3),
            "model": {k: C[k] for k in ("CL0", "CLa", "CL_MAX", "alpha_stall", "CD0", "k", "Cm0", "Cma", "x_np_m")}}


def l_ref(variant, propulsion) -> float:
    """The CFD's reference length [m]: the mean chord, unless the body is over 4 of them (the template's limit: the
    Zero's 1 m span on a 0.23 m chord); then body/3.9, and ``row`` scales its Cm back to the mean chord."""
    ref = reference(variant, propulsion)
    p = a1.resolve(a1.variant(variant, propulsion))
    body = max(p["span"], p["fuselage_length"] + 40.0) / 1000.0
    return max(ref["c"], body / 3.9)


def to_flow(x_mm, z_mm, alpha_deg):
    a = math.radians(alpha_deg)
    x, z = x_mm / 1000, z_mm / 1000
    return np.array([x * math.cos(a) + z * math.sin(a), 0.0, -x * math.sin(a) + z * math.cos(a)])


def export_stl(folder, variant, propulsion, alpha_deg) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"arigo1_{variant}_{propulsion}_a{alpha_deg:+.0f}.stl"
    if not path.exists():
        from vegeta import dedalus
        shape = a1.Arigo1().generate(**a1.variant(variant, propulsion)).shape.rotate((0, 0, 0), (0, 1, 0), alpha_deg)
        dedalus.Geometry.from_cadquery(shape, name="arigo1").export_stl(str(path), tolerance=0.2)
    return path


def case(workdir, variant="hybrid", propulsion="pusher", *, alpha_deg=0.0, V=SPEED, quality="screening", env=None):
    from vegeta import aeromant
    ref = reference(variant, propulsion)
    stl = export_stl(Path(workdir) / "stl", variant, propulsion, alpha_deg)
    params = dict(velocity=V, kinematic_viscosity=NU, density=RHO, reference_area=ref["S"], reference_length=l_ref(variant, propulsion),
                  center_of_rotation=tuple(to_flow(*ref["cg_mm"], alpha_deg).tolist()), **QUALITY[quality])
    env = env or aeromant.OpenFOAMEnvironment.detect()
    name = f"{variant}_{propulsion}_a{alpha_deg:+.0f}_V{V:.0f}_{quality}"
    return aeromant.CFDCase("rans_ksst_external", str(stl), params, workdir=str(Path(workdir) / name), geometry_units="mm", environment=env)


def model_row(variant, propulsion, alpha_deg, V=SPEED) -> dict:
    """What arigo1_model predicts for the same case (to compare side by side in the YAML)."""
    C = am.coefficients(variant, propulsion)
    al = math.radians(alpha_deg)
    F, M, s = am.aero({**C, "thrust": {"T0_N": 0.0, "V0_m_s": 1.0}}, (V * math.cos(al), 0.0, V * math.sin(al)), (0, 0, 0))
    qS = 0.5 * RHO * V * V * C["S"]
    lift = -F[2] * math.cos(al) + F[0] * math.sin(al)
    drag = -F[0] * math.cos(al) - F[2] * math.sin(al)
    return {"Cl": round(lift / qS, 4), "Cd": round(drag / qS, 4), "Cm": round(M[1] / (qS * C["c"]), 4)}


def row(variant, propulsion, alpha_deg, V, result) -> dict:
    out = {"alpha_deg": alpha_deg, "model": model_row(variant, propulsion, alpha_deg, V)}
    if result is None or not getattr(result, "ok", False):
        out["status"] = "NOT RUN" if result is None or any("NOT RUN" in s for s in getattr(result, "messages", [])) else "FAILED"
        out["messages"] = [str(m)[:200] for m in list(getattr(result, "messages", []))[:3]]
        return out
    k = result.metrics
    f = lambda v: None if v is None else float(v)
    cm_scale = l_ref(variant, propulsion) / reference(variant, propulsion)["c"]          # Cm back to the mean chord
    out.update({"status": "solved", "Cl": f(k.get("Cl")), "Cd": f(k.get("Cd")), "Cm_cg": None if k.get("Cm") is None else float(k.get("Cm")) * cm_scale,
                "lift_N": f(k.get("lift_force_N")), "drag_N": f(k.get("drag_force_N")),
                "converged": k.get("converged"), "iterations": k.get("iterations"), "Cl_scatter": f(k.get("Cl_std_lastN")),
                "Cd_scatter": f(k.get("Cd_std_lastN")), "cells": k.get("mesh_cells"), "mesh_ok": k.get("mesh_ok"), "reynolds": f(k.get("reynolds_number"))})
    return out


def summary(rows: list) -> dict:
    """The stall read off the solved rows: CL_max and its angle, CD0 (at the smallest |Cl|), dCm/dCl before the stall
    (minus the static margin, the fuselage included) and whether Cm turns nose-down past CL_max."""
    s = [r for r in rows if r.get("status") == "solved" and r.get("Cl") is not None]
    if len(s) < 3:
        return {"note": "too few solved cases"}
    s.sort(key=lambda r: r["alpha_deg"])
    i = int(np.argmax([r["Cl"] for r in s]))
    lin = [r for r in s if -4.0 <= r["alpha_deg"] <= 8.0]
    out = {"CL_max": s[i]["Cl"], "alpha_CL_max_deg": s[i]["alpha_deg"], "CD_min": min(r["Cd"] for r in s)}
    if len(lin) >= 2:
        cl, cm = np.array([r["Cl"] for r in lin]), np.array([r["Cm_cg"] for r in lin])
        a = np.radians([r["alpha_deg"] for r in lin])
        out["CL_alpha_per_rad"] = float(np.polyfit(a, cl, 1)[0])
        out["dCm_dCL"] = float(np.polyfit(cl, cm, 1)[0])
        out["static_margin_mac"] = -out["dCm_dCL"]
    post = s[i + 1:]
    if post:
        out["nose_drops_after_stall"] = bool(post[0]["Cm_cg"] < s[i]["Cm_cg"])
        out["lift_kept_after_stall"] = float(post[-1]["Cl"] / s[i]["Cl"])
    return out
