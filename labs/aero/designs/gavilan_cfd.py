"""GAVILAN's CFD cases (Aeromant / OpenFOAM) — prepared here, RUN BY YOU (``sims/gavilan_cfd.py``), not in a session.

The question is the stall. A tailsitter crosses the whole range of angle of attack every time it stops, starts or
turns hard, and the reduced model (``gavilan_flight``) takes the wing beyond its stall from a flat-plate blend
(``ogma.aero.Polar``) and has no pitching moment at all. So the cases are an **angle-of-attack sweep of the whole
aircraft from 0 to 90 degrees**, without propellers: lift, drag and the pitching moment about the centre of gravity.
What they answer:

- where the wing stalls and what lift it keeps beyond (the model's ``CL_MAX_3D`` and its flat plate);
- the drag broadside (the stop from 25 m/s leans on it);
- **the pitching moment the motors must hold** at every angle, against what they can give by differential thrust
  (``pitch_authority``): if the moment anywhere exceeds the authority, the aircraft does not survive that angle.

The fixed wing (``F``) gets the conventional sweep, -4 to 24 degrees: its stall angle, its maximum lift (the pass
speed and the turn radius hang on it) and whether the nose drops at the stall (the moment's slope there).

Atmosphere ISA sea level. Steady RANS k-omega SST with wall functions, no transition model (``nisus_cfd``'s notes
apply: trends, not absolutes; a separated flow at 30-90 degrees is unsteady and a steady solver gives its mean at
best — the forces' scatter over the last iterations is written out, read it). The propellers are left out: the blown
wing stalls later than these cases say, so they are the pessimistic side for the stall.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import gavilan as gv
import gavilan_flight as gf
from nisus_cfd import NU, QUALITY, RHO

ALPHAS = {"A": (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0),
          "B": (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0),
          "F": (-4.0, 0.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0),
          "H": (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0)}      # the hawk flares: the whole range
SPEED = 20.0


def reference(design: str) -> dict:
    """Reference area and length [m2, m], the centre of gravity in the CAD frame [mm] and the mass, per design."""
    if design == "H":
        g = gf.f_geometry(gf.H_PARAMS)
        pr = gf.f_mass_budget(gf.H_PARAMS, gf.CONFIG_H, 0).properties()
        q = gv.h_resolve()
        return {"S": g["S"], "c": g["mac"], "mass": pr.mass, "cg_mm": (pr.com[0] * 1e3 - q["nose_length"], -q["fuselage_diameter"] / 2)}
    if design == "F":
        g = gf.f_geometry()
        pr = gf.f_mass_budget().properties()
        q = gv.f_resolve()
        # f_mass_budget measures x from the nose tip along the body axis; the CAD's origin is the wing root's leading edge
        return {"S": g["S"], "c": g["mac"], "mass": pr.mass, "cg_mm": (pr.com[0] * 1e3 - q["nose_length"], -q["fuselage_diameter"] / 2)}
    p = gv.variant(design)
    g = gf.geometry(p)
    pr = gf.mass_budget(p).properties()
    return {"S": g["S"], "c": g["mac"], "mass": pr.mass, "cg_mm": (pr.com[0] * 1e3, pr.com[2] * 1e3)}


def to_flow(x_mm, z_mm, alpha_deg):
    """A point of the CAD frame [mm] in the flow frame [m] after the CAD's nose-up turn by alpha about +y."""
    a = math.radians(alpha_deg)
    x, z = x_mm / 1000, z_mm / 1000
    return np.array([x * math.cos(a) + z * math.sin(a), 0.0, -x * math.sin(a) + z * math.cos(a)])


def export_stl(folder, design: str, alpha_deg: float) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"gavilan_{design.lower()}_a{alpha_deg:+.0f}.stl"
    if not path.exists():
        if design in ("F", "H"):
            cls = gv.GavilanF if design == "F" else gv.GavilanH
            g = cls().generate(angle_of_attack_deg=alpha_deg) if alpha_deg <= 20.0 else None
            if g is None:                                              # FixedWing's own parameter stops at 20 degrees
                from vegeta import dedalus
                g = dedalus.Geometry.from_cadquery(cls().generate().shape.rotate((0, 0, 0), (0, 1, 0), alpha_deg), name=f"gavilan_{design.lower()}")
        else:
            g = gv.Gavilan().generate(**gv.variant(design), angle_of_attack_deg=alpha_deg)
        g.export_stl(str(path), tolerance=0.2)
    return path


def case(workdir, design: str = "A", *, alpha_deg: float = 0.0, V: float = SPEED, quality: str = "screening", env=None):
    """One Aeromant case (prepared, not run): the whole aircraft turned nose-up by alpha in the CAD, the flow along +x."""
    from vegeta import aeromant
    ref = reference(design)
    stl = export_stl(Path(workdir) / "stl", design, alpha_deg)
    params = dict(velocity=V, kinematic_viscosity=NU, density=RHO, reference_area=ref["S"], reference_length=ref["c"],
                  center_of_rotation=tuple(to_flow(*ref["cg_mm"], alpha_deg).tolist()), **QUALITY[quality])
    env = env or aeromant.OpenFOAMEnvironment.detect()
    name = f"{design.lower()}_a{alpha_deg:+.0f}_V{V:.0f}_{quality}"
    return aeromant.CFDCase("rans_ksst_external", str(stl), params, workdir=str(Path(workdir) / name), geometry_units="mm", environment=env)


def pitch_authority(design: str, V: float = SPEED) -> float:
    """The pitching moment the motors can give [N m]: the upper pair at full thrust, the lower pair at idle (or the
    reverse) — an upper bound, the thrust the flight itself needs not taken out. None for the fixed wing (its
    elevator is not modelled)."""
    if design == "F":
        return math.nan
    if design == "H":                                                  # the blown tail at full travel and full thrust (hand estimate)
        return gf.h_tail(V, gf.gavilan_h().t_max(V))["pitch_Nm"]
    m = gf.gavilan(design)
    return 0.5 * m.t_max(0.0) * gf.geometry(gv.variant(design))["motor_z"]


def row(design: str, alpha_deg: float, V: float, result) -> dict:
    """One case's line for the output file: the coefficients, the forces, the moment against the authority."""
    ref = reference(design)
    out = {"design": design, "alpha_deg": alpha_deg, "V": V, "S_ref": ref["S"], "c_ref": ref["c"], "mass_kg": ref["mass"]}
    if result is None or not getattr(result, "ok", False):
        out["status"] = "NOT RUN" if result is None or any("NOT RUN" in s for s in getattr(result, "messages", [])) else "FAILED"
        out["messages"] = list(getattr(result, "messages", []))[:3]
        return out
    k = result.metrics
    q = 0.5 * RHO * V * V
    moment = (k.get("Cm") or 0.0) * q * ref["S"] * ref["c"]
    auth = pitch_authority(design, V)
    out.update({"status": "solved", "Cl": k.get("Cl"), "Cd": k.get("Cd"), "Cm_cg": k.get("Cm"), "lift_N": k.get("lift_force_N"),
                "drag_N": k.get("drag_force_N"), "pitch_moment_Nm": moment, "pitch_authority_Nm": auth,
                "moment_over_authority": abs(moment) / auth if auth == auth and auth > 0 else None,
                "load_factor": (k.get("lift_force_N") or 0.0) / (ref["mass"] * 9.80665), "reynolds": k.get("reynolds_number"),
                "iterations": k.get("iterations"), "converged": k.get("converged"), "Cd_scatter_last50": k.get("Cd_std_lastN"),
                "Cl_scatter_last50": k.get("Cl_std_lastN"), "cells": k.get("mesh_cells"), "mesh_ok": k.get("mesh_ok")})
    return out
