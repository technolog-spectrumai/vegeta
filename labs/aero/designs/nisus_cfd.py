"""NISUS whole-aircraft CFD with Aeromant / OpenFOAM: the cases, their presets and the reading of the results
(notebook 31).

The questions, and the case that answers each:

- **lift, drag and pitching moment against the angle of attack and the speed** — the whole aircraft (``part='aircraft'``,
  no propeller) in steady RANS k-ω SST (``rans_ksst_external``), the aircraft turned nose-up by α in the CAD (the
  flow stays along +x, so Cl is lift and Cd is drag), at the cruise speed over α and at α = 2° over the speed;
- **how converged the numbers are** — the cruise point again on a finer mesh (``QUALITY['fine']``): the screening's
  mesh error is the difference;
- **pod, boom and tail interference** — the wing alone at the same point: (aircraft − wing) is what the pod, the
  booms, the tail and their junctions add, against the build-up's sum of those parts and its 10 % interference;
- **the installed pusher** — the same cruise point with the 9x6 propeller as a rotor disk behind the pod
  (``hull_rotor_disk``: blade-element sources from the Boreas blade and section): the airframe's force change in the
  propeller's inflow (the pod's after-body sucked by the disk: the thrust deduction).

Atmosphere: ISA sea level (ρ = 1.225 kg/m³, ν = 1.46e-5 m²/s). Reynolds numbers on the MAC: 1.6e5 at 12 m/s, 2.2e5
at 16 m/s. Turbulence: fully turbulent k-ω SST with wall functions and no prism layers — no transition model: at
these Reynolds numbers the real boundary layers are partly laminar with laminar separation bubbles, so the drag is
an over-estimate of the friction and an unreliable estimate of the separation; trends, not absolutes (the template
says so too). Boundaries: velocity inlet, pressure outlet, slip sides, no-slip body; the domain scaled with
``L_REF`` (a quarter of the span) — 5 L_ref ahead and to the sides, 12 behind. Convergence: the residual target and
the forces' scatter over the last 50 iterations (``Cd_std_lastN``) are reported per case; an unconverged case says so.

Presets (``QUALITY``): ``screening`` (2 cells per L_ref, levels 4/3/2, 300 iterations: minutes per case on 4 cores),
``fine`` (3 cells, levels 5/4/2, 600 iterations).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import nisus
import nisus_systems as ns

RHO, NU = 1.225, 1.46e-5
QUALITY = {
    "screening": dict(cells_per_length=2.0, surface_level=4, near_level=3, wake_level=2, iterations=300, residual_target=1e-4),
    "fine": dict(cells_per_length=3.0, surface_level=5, near_level=4, wake_level=2, iterations=600, residual_target=1e-5),
}
ALPHAS = (-2.0, 2.0, 6.0, 10.0)
SPEEDS = (12.0, 16.0, 20.0)


def l_ref(p=None) -> float:
    return nisus.resolve(p)["span"] / 4000


def to_flow(p, x_mm, z_mm, alpha_deg):
    """A point of the nisus frame (mm) in the flow frame (m) after the CAD's nose-up turn by α about +y."""
    a = math.radians(alpha_deg)
    x, z = x_mm / 1000, z_mm / 1000
    return np.array([x * math.cos(a) + z * math.sin(a), 0.0, -x * math.sin(a) + z * math.cos(a)])


def export_stl(workdir, part="aircraft", alpha_deg=0.0, p=None) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"nisus_{part}_a{alpha_deg:+.0f}.stl"
    if not path.exists():
        q = nisus.overrides(p)
        if part == "aircraft":
            g = nisus.Nisus().generate(**q, part=part, angle_of_attack_deg=alpha_deg)
        else:
            from vegeta import dedalus
            shape = nisus.Nisus().generate(**q, part=part).shape.rotate((0, 0, 0), (0, 1, 0), alpha_deg)
            g = dedalus.Geometry.from_cadquery(shape, name=f"nisus_{part}")
        g.export_stl(str(path), tolerance=0.2)
    return path


def case(workdir, *, alpha_deg=2.0, V=16.0, part="aircraft", quality="screening", x_cg_m=0.07, z_cg_m=-0.037, prop=None, env=None, p=None):
    """One Aeromant case (prepared, not run). ``prop``: None, or a dict with ``rpm`` — the 9x6 as a rotor disk at
    the propeller plane (``hull_rotor_disk``)."""
    from vegeta import aeromant
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    stl = export_stl(Path(workdir).parent / "stl", part, alpha_deg, p)
    params = dict(velocity=V, kinematic_viscosity=NU, density=RHO, reference_area=L["S_ref"], reference_length=l_ref(p),
                  center_of_rotation=tuple(to_flow(p, x_cg_m * 1000, z_cg_m * 1000, alpha_deg).tolist()), **QUALITY[quality])
    template = "rans_ksst_external"
    if prop is not None:
        pr = ns.propeller_9x6()
        af = ns.blade_section()
        th = math.radians(alpha_deg + p["motor_downthrust_deg"])
        centre = to_flow(p, L["prop_x"], p["motor_z"], alpha_deg)
        alphas = np.unique(np.r_[np.arange(-180, -30, 10), np.arange(-30, 31, 1), np.arange(40, 181, 10)]).astype(float)
        cl, cd = af.coefficients(np.radians(alphas))
        params.update(disk1_center=centre.tolist(), disk_axis=[-math.cos(th), 0.0, -math.sin(th)], diameter=pr.diameter, rpm=float(prop["rpm"]),
                      blades=pr.blades, blade=[[r, b, c] for r, b, c in zip(pr.r, pr.beta_deg, pr.chord)],
                      polar=[[float(a), float(d), float(l)] for a, l, d in zip(alphas, cl, cd)], rotation1=1, disk_level=5)
        template = "hull_rotor_disk"
    env = env or aeromant.OpenFOAMEnvironment.detect()
    name = f"{part}_a{alpha_deg:+.0f}_V{V:.0f}_{quality}" + ("_prop" if prop else "")
    return aeromant.CFDCase(template, str(stl), params, workdir=str(Path(workdir) / name), geometry_units="mm", environment=env)


def plan(workdir, *, x_cg_m, z_cg_m, cruise_rpm, quality="screening", fine=True, env=None, p=None) -> dict:
    """The cases of the study, by name: the α sweep at 16 m/s, the speed sweep at α = 2°, the fine cruise point, the
    wing alone, the cruise point with the propeller."""
    kw = dict(x_cg_m=x_cg_m, z_cg_m=z_cg_m, env=env, p=p)
    out = {}
    for a in ALPHAS:
        out[f"α {a:+.0f}°, 16 m/s"] = case(workdir, alpha_deg=a, V=16.0, quality=quality, **kw)
    for V in SPEEDS:
        if V != 16.0:
            out[f"α +2°, {V:.0f} m/s"] = case(workdir, alpha_deg=2.0, V=V, quality=quality, **kw)
    out["wing alone, α +2°, 16 m/s"] = case(workdir, alpha_deg=2.0, V=16.0, part="wing", quality=quality, **kw)
    out["α +2°, 16 m/s, propeller running"] = case(workdir, alpha_deg=2.0, V=16.0, quality=quality, prop={"rpm": cruise_rpm}, **kw)
    if fine:
        out["α +2°, 16 m/s, fine mesh"] = case(workdir, alpha_deg=2.0, V=16.0, quality="fine", **kw)
    return out


def table(results: dict, p=None) -> pd.DataFrame:
    """One row per case: Cl, Cd, Cm (about the CG, referred to the MAC), forces, Re, convergence, mesh size — or NOT RUN."""
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
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
                      "converged": m.get("converged"), "Cd scatter (last 50)": m.get("Cd_std_lastN"), "cells": m.get("mesh_cells"),
                      "mesh ok": m.get("mesh_ok")}
    return pd.DataFrame(rows).T


__all__ = ["RHO", "NU", "QUALITY", "ALPHAS", "SPEEDS", "l_ref", "to_flow", "export_stl", "case", "plan", "table"]
