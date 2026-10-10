"""FALCO whole-aircraft CFD with Aeromant / OpenFOAM: the cases, their presets and the reading of the results
(notebook 33).

NISUS's study (``nisus_cfd``, whose quality presets and frame conversion this module reuses) at altitude, with the
questions FALCO adds:

- **the clean aircraft at the work altitude** — the α sweep at the cruise EAS (18 m/s: 20.9 m/s TAS at 3000 m) in the
  ISA air of 3000 m (ρ and ν from ``falco_systems.atmosphere``: the Reynolds number on the MAC is ~2.6e5 instead of
  3.1e5 at sea level at the same EAS);
- **crow** — the aircraft with the flaps at 55° and the ailerons at −25° in the CAD (``falco.Falco(flap_deg,
  aileron_deg)``) at V_FE (22 m/s EAS) and α 0 and +4°: the drag increment the derivative table assumes
  (``falco_flight.crow_increments``: Raymer's plain-flap formula) and the pitching moment it computes with the lattice
  — the check of both. The deflected surfaces are separate solids touching the wing at their hinge line: the STL
  carries the gap's small overlaps; snappyHexMesh resolves them at the surface level (a small error source);
- **the installed pusher and its brake** — the cruise point with the 15x8 as a rotor disk (``hull_rotor_disk``) at the
  cruise rpm, and at a braking rpm (the propeller windmilling below its free speed: the disk takes energy out of the
  flow; the template's blade polar spans ±180°, so the disk's negative thrust is represented — that the sign comes out
  right is checked in the notebook from the case's force on the disk).

Everything is prepared here and runs when the notebook runs (``VEGETA_SKIP_OPENFOAM=1``: NOT RUN). Turbulence,
boundaries and the domain as NISUS's: fully turbulent k-ω SST, no transition model — trends, not absolutes.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import falco
import falco_flight as ff
import falco_systems as fs
from nisus_cfd import QUALITY, to_flow

H_CFD = 3000.0
ALPHAS = (-2.0, 2.0, 6.0, 10.0)


def air(h_m: float = H_CFD, dT: float = 0.0) -> dict:
    a = fs.atmosphere(h_m, dT)
    return {"rho": a["rho"], "nu": a["nu"], "sigma": a["sigma"]}


def l_ref(p=None) -> float:
    return falco.resolve(p)["span"] / 4000


def export_stl(workdir, part="aircraft", alpha_deg=0.0, p=None, *, flap_deg=0.0, aileron_deg=0.0) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tag = f"_f{flap_deg:+.0f}_a{aileron_deg:+.0f}" if (flap_deg or aileron_deg) else ""
    path = workdir / f"falco_{part}_a{alpha_deg:+.0f}{tag}.stl"
    if not path.exists():
        q = falco.overrides(p)
        g = falco.Falco().generate(**q, part=part, angle_of_attack_deg=alpha_deg, flap_deg=flap_deg, aileron_deg=aileron_deg)
        g.export_stl(str(path), tolerance=0.3)
    return path


def case(workdir, *, alpha_deg=2.0, V_eas=ff.V_CRUISE_EAS, h_m=H_CFD, part="aircraft", quality="screening", x_cg_m=0.085, z_cg_m=-0.05,
         prop=None, crow=None, env=None, p=None):
    """One Aeromant case (prepared, not run) at the altitude ``h_m``'s air and the TAS of ``V_eas``. ``prop``: None or
    ``{'rpm': ...}`` (the 15x8 as a rotor disk); ``crow``: None or ``(flap_deg, aileron_deg)``."""
    from vegeta import aeromant
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    a = air(h_m)
    V = V_eas / math.sqrt(a["sigma"])
    fl, al = crow if crow else (0.0, 0.0)
    stl = export_stl(Path(workdir).parent / "stl", part, alpha_deg, p, flap_deg=fl, aileron_deg=al)
    params = dict(velocity=V, kinematic_viscosity=a["nu"], density=a["rho"], reference_area=L["S_ref"], reference_length=l_ref(p),
                  center_of_rotation=tuple(to_flow(p, x_cg_m * 1000, z_cg_m * 1000, alpha_deg).tolist()), **QUALITY[quality])
    template = "rans_ksst_external"
    if prop is not None:
        pr, af = fs.propeller_15x8(), fs.blade_section()
        th = math.radians(alpha_deg + p["motor_downthrust_deg"])
        centre = to_flow(p, L["prop_x"], p["motor_z"], alpha_deg)
        alphas = np.unique(np.r_[np.arange(-180, -30, 10), np.arange(-30, 31, 1), np.arange(40, 181, 10)]).astype(float)
        cl, cd = af.coefficients(np.radians(alphas))
        params.update(disk1_center=centre.tolist(), disk_axis=[-math.cos(th), 0.0, -math.sin(th)], diameter=pr.diameter, rpm=float(prop["rpm"]),
                      blades=pr.blades, blade=[[r, b, c] for r, b, c in zip(pr.r, pr.beta_deg, pr.chord)],
                      polar=[[float(x), float(d), float(l)] for x, l, d in zip(alphas, cl, cd)], rotation1=1, disk_level=5)
        template = "hull_rotor_disk"
    env = env or aeromant.OpenFOAMEnvironment.detect()
    name = f"{part}_a{alpha_deg:+.0f}_V{V_eas:.0f}eas_h{h_m:.0f}_{quality}" + (f"_crow{fl:.0f}_{al:.0f}" if crow else "") + (f"_prop{prop['rpm']:.0f}" if prop else "")
    return aeromant.CFDCase(template, str(stl), params, workdir=str(Path(workdir) / name), geometry_units="mm", environment=env)


def specs(*, cruise_rpm, brake_rpm, quality="screening", fine=True) -> dict:
    """The cases by name and their ``case`` arguments: the α sweep at cruise (3000 m), crow at V_FE (α 0 and +4°), the
    pusher at its cruise rpm and braked, the fine cruise point. Needs no OpenFOAM (the NOT RUN table lists them)."""
    out = {}
    for a in ALPHAS:
        out[f"α {a:+.0f}°, {ff.V_CRUISE_EAS:g} m/s EAS, 3000 m"] = dict(alpha_deg=a, quality=quality)
    for a in (0.0, 4.0):
        out[f"crow 55/-25, α {a:+.0f}°, {ff.V_FE_EAS:g} m/s EAS"] = dict(alpha_deg=a, V_eas=ff.V_FE_EAS, quality=quality, crow=(55.0, -25.0))
    out[f"clean, α +0°, {ff.V_FE_EAS:g} m/s EAS (crow's reference)"] = dict(alpha_deg=0.0, V_eas=ff.V_FE_EAS, quality=quality)
    out[f"α +2°, cruise, propeller {cruise_rpm:.0f} rpm"] = dict(alpha_deg=2.0, quality=quality, prop={"rpm": cruise_rpm})
    out[f"crow at V_FE, propeller braked {brake_rpm:.0f} rpm"] = dict(alpha_deg=0.0, V_eas=ff.V_FE_EAS, quality=quality, crow=(55.0, -25.0),
                                                                   prop={"rpm": brake_rpm})
    if fine:
        out["α +2°, cruise, fine mesh"] = dict(alpha_deg=2.0, quality="fine")
    return out


def plan(workdir, *, x_cg_m, z_cg_m, cruise_rpm, brake_rpm, quality="screening", fine=True, env=None, p=None) -> dict:
    """The cases of ``specs`` built (STL and OpenFOAM case; needs an OpenFOAM environment)."""
    return {name: case(workdir, x_cg_m=x_cg_m, z_cg_m=z_cg_m, env=env, p=p, **kw)
            for name, kw in specs(cruise_rpm=cruise_rpm, brake_rpm=brake_rpm, quality=quality, fine=fine).items()}


def table(results: dict, p=None) -> pd.DataFrame:
    """One row per case: Cl, Cd, Cm (about the CG, referred to the MAC), forces, Re, convergence, mesh size — or NOT RUN."""
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    k_cm = l_ref(p) / (L["mac"] / 1000)
    rows = {}
    for name, r in results.items():
        if r is None or not getattr(r, "ok", False):
            msg = "NOT RUN" if r is None or any("NOT RUN" in m for m in getattr(r, "messages", [])) else "FAILED"
            rows[name] = {"status": msg}
            continue
        m = r.metrics
        rows[name] = {"status": "solved", "Cl": m.get("Cl"), "Cd": m.get("Cd"), "Cm (CG, MAC)": (m.get("Cm") or np.nan) * k_cm,
                      "L/D": (m.get("Cl") or np.nan) / m["Cd"] if m.get("Cd") else np.nan, "lift [N]": m.get("lift_force_N"),
                      "drag [N]": m.get("drag_force_N"), "Re (L_ref)": m.get("reynolds_number"), "iterations": m.get("iterations"),
                      "converged": m.get("converged"), "Cd scatter (last 50)": m.get("Cd_std_lastN"), "cells": m.get("mesh_cells")}
    return pd.DataFrame(rows).T


def crow_check(tab: pd.DataFrame, deriv: pd.DataFrame) -> dict:
    """The CFD's crow increment (crow − clean at α 0, V_FE) against the derivative table's (assumed ΔCD, lattice ΔCm)."""
    c = ff.coefficients(deriv)
    try:
        crow = tab.loc[next(k for k in tab.index if k.startswith("crow 55/-25, α +0°"))]
        clean = tab.loc[next(k for k in tab.index if k.startswith("clean, α +0°"))]
    except StopIteration:
        return {"status": "no cases"}
    if crow.get("status") != "solved" or clean.get("status") != "solved":
        return {"status": "NOT RUN", "dCD_table": c["dCD_crow"], "dCm_table": c["dCm_crow"]}
    return {"status": "solved", "dCD_cfd": crow["Cd"] - clean["Cd"], "dCD_table": c["dCD_crow"], "dCL_cfd": crow["Cl"] - clean["Cl"], "dCL_table": c["dCL_crow"],
            "dCm_cfd": crow["Cm (CG, MAC)"] - clean["Cm (CG, MAC)"], "dCm_table": c["dCm_crow"]}


__all__ = ["specs", "H_CFD", "QUALITY", "ALPHAS", "air", "l_ref", "export_stl", "case", "plan", "table", "crow_check"]
