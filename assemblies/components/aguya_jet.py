"""AGUYA's hot jet at the dash: the free-jet estimate along the V-tail and the whole aircraft in CFD with the engine
running (Aeromant ``jet_external``: intake drawing, nozzle blowing hot gas with a tracer).

Promoted from notebook 29 section 7.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from vegeta import aeromant
from vegeta.boreas import microjet as mj

from ..vida import Assembly
from . import aguya as ag

FIDELITY = {
    "smoke": dict(iterations=100, surface_level=2, near_level=2, wake_level=1, engine_level=2, plume_level=2),
    "quick": dict(iterations=300, surface_level=3, near_level=2, wake_level=2, engine_level=3, plume_level=2),
    "full": dict(iterations=1500, surface_level=4, near_level=3, wake_level=2, engine_level=5, plume_level=4),
}


def dash_point(P: dict, af, unit, engine: mj.Microjet, flight) -> dict:
    """The dash of the design mission: speed, mass at 60 % fuel, shaft speed for level flight, the jet it makes, and
    the free-jet estimate of its excess temperature at the V-tail (notebook 29 section 7)."""
    L = ag.Aguya.layout(P)
    dash_mass = af.dry_mass_kg + 0.6 * flight.fuel_kg
    v_dash = float(flight.dash_speed)
    rpm = float(unit.for_thrust(v_dash, af.drag(v_dash, dash_mass))[1])
    jp = mj.solve(engine, rpm, v_dash)
    d_j = P["nozzle_diameter"] / 1000
    x_le, x_te = (L["x_tail_le"] - L["x_exit"]) / 1000, (L["x_end"] - L["x_exit"]) / 1000
    gap = (L["R"] + L["nacelle_z"]) / 1000 * math.cos(math.radians(P["vtail_dihedral_deg"]))
    return {"v_dash": v_dash, "dash_mass_kg": float(dash_mass), "rpm": rpm, "buildup_drag_N": float(af.drag(v_dash, dash_mass)),
            "mass_flow": float(jp.mass_flow), "fuel_flow": float(jp.fuel_flow), "jet_temperature_K": float(jp.jet_temperature),
            "jet_velocity": float(jp.jet_velocity), "nozzle_diameter_m": d_j, "tail_from_nozzle_m": [x_le, x_te],
            "jet_axis_to_tail_m": gap, "jet_half_width_at_tail_m": [d_j / 2 + 0.1 * x_le, d_j / 2 + 0.1 * x_te],
            "estimate_excess_K_at_tail": [float(jet_excess(x, d_j, jp.jet_temperature)) for x in (x_le, x_te)]}


def jet_excess(x, d_j: float, jet_temperature: float):
    """Free-jet estimate of the excess temperature on the jet's axis ``x`` [m] behind the nozzle: the potential core
    to 5 diameters, then falling as 1/x."""
    x = np.asarray(x, float)
    return (jet_temperature - 288.15) * np.where(x < 5 * d_j, 1.0, 5 * d_j / np.maximum(x, 1e-9))


def assembly(P: dict, wing_area_m2: float, point: dict, fidelity: str) -> Assembly:
    if fidelity not in FIDELITY:
        raise ValueError(f"fidelity must be one of {tuple(FIDELITY)}")
    return Assembly("jet_cfd", "jet_external", params=dict(
        template="jet_external", geometry=P, velocity=point["v_dash"], pressure=101325.0, temperature=288.15,
        reference_area=wing_area_m2, reference_length=P["span"] / 4000, center_of_rotation=[0.08, 0.0, 0.0],
        intake_mass_flow=point["mass_flow"], exhaust_mass_flow=point["mass_flow"] + point["fuel_flow"],
        exhaust_temperature=point["jet_temperature_K"], residual_target=1e-5, cells_per_length=2.0, plume_length=4.0,
        fidelity=fidelity, mesh_and_solver=FIDELITY[fidelity]))


def case(node: Assembly, out: Path, env: aeromant.OpenFOAMEnvironment) -> aeromant.CFDCase:
    """The case in ``out/dash``, the body, intake-face and nozzle-face STLs in ``out/stl`` (written here)."""
    p = node.params
    surf = ag.cfd_surfaces(p["geometry"], Path(out) / "stl")
    values = {k: p[k] for k in ("velocity", "pressure", "temperature", "reference_area", "reference_length",
                                "center_of_rotation", "intake_mass_flow", "exhaust_mass_flow", "exhaust_temperature",
                                "residual_target", "cells_per_length", "plume_length")}
    values.update(p["mesh_and_solver"])
    return aeromant.CFDCase(p["template"], surf["paths"]["body"], values, workdir=Path(out) / "dash", geometry_units="mm",
                            environment=env, surfaces={"intake": surf["paths"]["intake"], "exhaust": surf["paths"]["exhaust"]})


def solve(node: Assembly, out: Path, *, run: bool = True, processors: int = 1, progress=True,
          env: aeromant.OpenFOAMEnvironment | None = None) -> Assembly:
    """Solve the case (read back when solved with the same inputs) and record drag, Cd, Mach and the plume along the
    jet axis; NOT RUN with the reason otherwise."""
    from .impeller import environment

    c = case(node, out, env or environment(run))
    r = aeromant.run_cases([c], processors=processors, run=run, progress=progress)[0]
    for name in ("body", "intake", "exhaust"):
        node.attach(f"{name}.stl", Path(out) / "stl" / f"{name}.stl", "geometry")
    if c.workdir.exists():
        node.attach("dash", c.workdir, "cases")
    if r.ok:
        m = r.metrics
        keep = {k: m.get(k) for k in ("drag_force_N", "lift_force_N", "Cd", "Cl", "mach_number", "converged", "mesh_cells")}
        node.record(complete=True, forces=keep, plume={k: np.asarray(v, float) for k, v in (m.get("plume") or {}).items()},
                    jet_excess={k: v for k, v in m.items() if k.startswith("jet_excess_T_at_")})
    else:
        node.not_run("run_cfd=False" if not run else f"failed: {r.messages[-1] if r.messages else r.status}")
    return node
