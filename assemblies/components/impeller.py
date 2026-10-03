"""The microjet's radial compressor in CFD (Aeromant ``compressor_mrf``): a speed line at full shaft speed.

Promoted from notebook 28 section 5. The node's parameters hold everything the cases are built from (the sized
geometry, the shaft speed, the inlet state, the back pressures, the mesh and solver settings of the fidelity), so a
change of any of them makes a new node key, and each case directory a new case key.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from vegeta import aeromant
from vegeta.boreas import microjet as mj

from ..vida import Assembly
from . import turbojet as tj

BACK_FRACTIONS = (0.55, 0.75, 0.9, 1.0, 1.08)            # of the design pressure ratio times p2 / 1.25, static at the exit
FIDELITY = {
    "smoke": dict(points=(1, 3), cells_per_diameter=5.0, surface_level=2, shroud_level=3, rotor_level=2, iterations=100),
    "quick": dict(points=(1, 3), cells_per_diameter=6.0, surface_level=3, shroud_level=3, rotor_level=2, iterations=300),
    "full": dict(points=(0, 1, 2, 3, 4), cells_per_diameter=10.0, surface_level=4, shroud_level=5, rotor_level=3,
                 iterations=3000),
}
COLUMNS = ("back_pressure_kPa", "ok", "mass_flow_kg_s", "corrected_mass_flow_kg_s", "pressure_ratio_tt", "efficiency_tt",
           "efficiency_from_torque", "work_coefficient", "shaft_power_kW")


def assembly(engine: mj.Microjet, P: dict, fidelity: str) -> Assembly:
    """The speed-line node: what defines the cases, nothing computed."""
    if fidelity not in FIDELITY:
        raise ValueError(f"fidelity must be one of {tuple(FIDELITY)}")
    f = dict(FIDELITY[fidelity])
    p2 = mj.Atmosphere().pressure * engine.intake_recovery
    design = mj.solve(engine, engine.rpm_max)
    back = [float(BACK_FRACTIONS[i] * design.pressure_ratio * p2 / 1.25) for i in f.pop("points")]
    return Assembly("compressor", "compressor_speedline", params=dict(
        template="compressor_mrf", geometry=P, rpm=float(engine.rpm_max), diameter_m=P["impeller_diameter"] / 1000,
        inlet_total_pressure=float(p2), inlet_total_temperature=288.15, back_pressures=back, first_order=1.0,
        fidelity=fidelity, mesh_and_solver=f))


def environment(run: bool = True) -> aeromant.OpenFOAMEnvironment:
    """The OpenFOAM installation found on this machine. Without one: an error when CFD is to run, else a default
    environment (enough to build the cases and read solved ones back)."""
    try:
        return aeromant.OpenFOAMEnvironment.detect()
    except RuntimeError as exc:
        if run:
            raise RuntimeError(f"{exc}. Without OpenFOAM, run with --no-cfd (run_cfd=False).") from None
        return aeromant.OpenFOAMEnvironment()


def cases(node: Assembly, out: Path, env: aeromant.OpenFOAMEnvironment | None = None) -> list[aeromant.CFDCase]:
    """The cases of the speed line in ``out/p<i>_<kPa>kPa``, the passage's four STLs in ``out/stl`` (written here)."""
    p = node.params
    surf = tj.compressor_surfaces(p["geometry"], Path(out) / "stl")
    env = env or environment(run=False)
    ms = p["mesh_and_solver"]
    out_cases = []
    for i, pb in enumerate(p["back_pressures"]):
        values = dict(rpm=p["rpm"], diameter=p["diameter_m"], inlet_total_pressure=p["inlet_total_pressure"],
                      inlet_total_temperature=p["inlet_total_temperature"], outlet_pressure=pb,
                      location_in_mesh=surf["location_in_mesh"], cells_per_diameter=ms["cells_per_diameter"],
                      surface_level=ms["surface_level"], shroud_level=ms["shroud_level"], rotor_level=ms["rotor_level"],
                      iterations=ms["iterations"], first_order=p["first_order"])
        out_cases.append(aeromant.CFDCase(p["template"], surf["paths"]["impeller"], values,
                                          workdir=Path(out) / f"p{i}_{pb / 1e3:.0f}kPa", geometry_units="mm",
                                          environment=env,
                                          surfaces={k: surf["paths"][k] for k in ("shroud", "inlet", "outlet")}))
    return out_cases


def speedline(node: Assembly, out: Path, *, run: bool = True, processors: int = 1, jobs: int = 1, progress=True,
              env: aeromant.OpenFOAMEnvironment | None = None) -> Assembly:
    """Solve the speed line (cases already solved with the same inputs are read back) and record it on ``node``:
    ``speedline`` (one list per column of ``COLUMNS``) and ``complete`` (every point solved). With nothing solved the
    node says NOT RUN and why."""
    cs = cases(node, out, env or environment(run))
    rs = aeromant.run_cases(cs, jobs=jobs, processors=processors, run=run, progress=progress)
    table = {c: [] for c in COLUMNS}
    for pb, r in zip(node.params["back_pressures"], rs):
        m = r.metrics if r.ok else {}
        table["back_pressure_kPa"].append(pb / 1e3)
        table["ok"].append(bool(r.ok))
        for c in COLUMNS[2:-1]:
            v = m.get(c)
            table[c].append(float(v) if v is not None else float("nan"))
        sp = m.get("shaft_power_W")
        table["shaft_power_kW"].append(float(sp) / 1e3 if sp is not None else float("nan"))
    for name in ("impeller", "shroud", "inlet", "outlet"):
        node.attach(f"{name}.stl", Path(out) / "stl" / f"{name}.stl", "geometry")
    for c in cs:
        if c.workdir.exists():
            node.attach(c.workdir.name, c.workdir, "cases")
    if any(table["ok"]):
        node.record(speedline=table, complete=all(table["ok"]),
                    messages={c.workdir.name: r.messages[-3:] for c, r in zip(cs, rs) if not r.ok})
        for c, r in zip(cs, rs):
            if not r.ok:
                node.not_run(f"{c.workdir.name}: {r.messages[-1] if r.messages else r.status}")
    else:
        node.not_run("run_cfd=False" if not run else "every point failed: " + (rs[0].messages[-1] if rs and rs[0].messages else "?"))
    return node
