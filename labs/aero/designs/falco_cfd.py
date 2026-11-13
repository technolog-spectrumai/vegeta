"""FALCO's CFD cases (Aeromant / OpenFOAM) — prepared, NOT RUN in the notebook (no OpenFOAM in the session).

NISUS+'s plan (``nisus_plus_cfd``: the α sweep at cruise, crow at V_FE, the fine cruise point) on FALCO's body, plus
the tractor's own case: the 20x13 as a rotor disk **ahead of the nose** at the cruise rpm (``hull_rotor_disk``: the
slipstream over the fuselage and the wing root that the build-up and the installation model do not hold), and crow with
the braked disk at V_FE. ``specs`` needs no OpenFOAM: the NOT RUN table lists the cases with their air.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import falco
import falco_flight as fl
import falco_systems as fsy
import nisus_plus_cfd as npc
from nisus_cfd import QUALITY, to_flow

H_CFD = npc.H_CFD
ALPHAS = npc.ALPHAS
air = npc.air


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


def case(workdir, *, alpha_deg=2.0, V_eas=fl.V_CRUISE_EAS, h_m=H_CFD, part="aircraft", quality="screening", x_cg_m=0.093, z_cg_m=-0.06,
         prop=None, crow=None, env=None, p=None):
    """One Aeromant case (prepared, not run) at the altitude ``h_m``'s air and the TAS of ``V_eas``. ``prop``: None or
    ``{'rpm': ...}`` (the 20x13 as a rotor disk at the nose, its axis along the thrust line with the down-thrust);
    ``crow``: None or ``(flap_deg, aileron_deg)``."""
    from vegeta import aeromant
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    a = air(h_m)
    V = V_eas / math.sqrt(a["sigma"])
    fla, al = crow if crow else (0.0, 0.0)
    stl = export_stl(Path(workdir).parent / "stl", part, alpha_deg, p, flap_deg=fla, aileron_deg=al)
    params = dict(velocity=V, kinematic_viscosity=a["nu"], density=a["rho"], reference_area=L["S_ref"], reference_length=l_ref(p),
                  center_of_rotation=tuple(to_flow(p, x_cg_m * 1000, z_cg_m * 1000, alpha_deg).tolist()), **QUALITY[quality])
    template = "rans_ksst_external"
    if prop is not None:
        pr, af = fsy.propeller(), fsy.blade_section()
        th = math.radians(alpha_deg + p["motor_downthrust_deg"])
        centre = to_flow(p, L["prop_x"], L["motor_z"], alpha_deg)
        alphas = np.unique(np.r_[np.arange(-180, -30, 10), np.arange(-30, 31, 1), np.arange(40, 181, 10)]).astype(float)
        cl, cd = af.coefficients(np.radians(alphas))
        params.update(disk1_center=centre.tolist(), disk_axis=[-math.cos(th), 0.0, -math.sin(th)], diameter=pr.diameter, rpm=float(prop["rpm"]),
                      blades=pr.blades, blade=[[r, b, c] for r, b, c in zip(pr.r, pr.beta_deg, pr.chord)],
                      polar=[[float(x), float(d), float(l)] for x, l, d in zip(alphas, cl, cd)], rotation1=1, disk_level=5)
        template = "hull_rotor_disk"
    env = env or aeromant.OpenFOAMEnvironment.detect()
    name = f"{part}_a{alpha_deg:+.0f}_V{V_eas:.0f}eas_h{h_m:.0f}_{quality}" + (f"_crow{fla:.0f}_{al:.0f}" if crow else "") + (f"_prop{prop['rpm']:.0f}" if prop else "")
    return aeromant.CFDCase(template, str(stl), params, workdir=str(Path(workdir) / name), geometry_units="mm", environment=env)


def specs(*, cruise_rpm, brake_rpm, quality="screening", fine=True) -> dict:
    """The cases by name and their ``case`` arguments: the α sweep at cruise (3000 m), crow at V_FE (α 0 and +4°), the
    tractor's disk at its cruise rpm (the slipstream over the body and the wing root: the installation check), crow
    with the braked disk, the fine cruise point."""
    out = {}
    for a in ALPHAS:
        out[f"α {a:+.0f}°, {fl.V_CRUISE_EAS:g} m/s EAS, 3000 m"] = dict(alpha_deg=a, quality=quality)
    for a in (0.0, 4.0):
        out[f"crow 55/-25, α {a:+.0f}°, {fl.V_FE_EAS:g} m/s EAS"] = dict(alpha_deg=a, V_eas=fl.V_FE_EAS, quality=quality, crow=(55.0, -25.0))
    out[f"clean, α +0°, {fl.V_FE_EAS:g} m/s EAS (crow's reference)"] = dict(alpha_deg=0.0, V_eas=fl.V_FE_EAS, quality=quality)
    out[f"α +2°, cruise, nose propeller {cruise_rpm:.0f} rpm (the slipstream over the body)"] = dict(alpha_deg=2.0, quality=quality, prop={"rpm": cruise_rpm})
    out[f"crow at V_FE, propeller braked {brake_rpm:.0f} rpm"] = dict(alpha_deg=0.0, V_eas=fl.V_FE_EAS, quality=quality, crow=(55.0, -25.0),
                                                                   prop={"rpm": brake_rpm})
    if fine:
        out["α +2°, cruise, fine mesh"] = dict(alpha_deg=2.0, quality="fine")
    return out


def plan(workdir, *, x_cg_m, z_cg_m, cruise_rpm, brake_rpm, quality="screening", fine=True, env=None, p=None) -> dict:
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


def not_run_table(*, cruise_rpm, brake_rpm, h_m=H_CFD) -> pd.DataFrame:
    """The planned cases with their air and the status NOT RUN (no OpenFOAM here)."""
    rows = {}
    a = air(h_m)
    for name, kw in specs(cruise_rpm=cruise_rpm, brake_rpm=brake_rpm).items():
        V_eas = kw.get("V_eas", fl.V_CRUISE_EAS)
        rows[name] = {"status": "NOT RUN", "α [deg]": kw["alpha_deg"], "V [m/s EAS]": V_eas, "TAS [m/s]": V_eas / math.sqrt(a["sigma"]), "altitude [m]": h_m,
                      "ρ [kg/m³]": a["rho"], "template": "hull_rotor_disk" if kw.get("prop") else "rans_ksst_external", "quality": kw.get("quality", "screening"),
                      "crow": str(kw.get("crow", "")), "propeller rpm": (kw.get("prop") or {}).get("rpm", "")}
    return pd.DataFrame(rows).T


crow_check = npc.crow_check

__all__ = ["H_CFD", "ALPHAS", "air", "l_ref", "export_stl", "case", "specs", "plan", "table", "not_run_table", "crow_check"]
