"""Halka's jet in Aeromant (OpenFOAM, steady RANS k-omega SST): the frame and the ground rings as the body in a
box of nearly still air (a residual 0.5 m/s inflow along +x for the steady solver's sake, as the static rotor
template does), and the corona's force on the air as ``fvOptions`` momentum sources in annular cell zones, binned
from Fulgur's solve (``vegeta.fulgur.jet``). Prepared here, run where OpenFOAM is (``run.sh`` in the case: the
pipeline with ``topoSet`` before the solver). Under ``vegeta.mock`` the run returns fixed numbers.

The thrust axis is +x in the case (Aeromant's flow axis): the thruster is rotated z -> x on export. The emitter
strips (0.2 mm) are left out of the STL: they would set the mesh size and they carry no flow; their force is in
the sources.

    python halka_cfd.py [workdir]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from vegeta.fulgur import DRY
from vegeta.fulgur.jet import force_zones, jet_estimate, convergence_angle, write_openfoam_sources
from vegeta.fulgur.thruster import PRESETS

from halka import Halka
import halka_fulgur as hf

RHO, NU = 1.204, 1.5e-5
STILL_AIR = 0.5                 # m/s: the residual inflow


def body_stl(preset: str, path, params=None):
    """The frame and the grounds as one STL in mm, the thrust axis turned to +x."""
    import cadquery as cq
    p = {q.name: q.default for q in Halka.parameters}
    p.update(params or {})
    p["preset"] = preset
    d = Halka()
    body = d.build({**p, "part": "frame"})
    for g in d.build({**p, "part": "grounds"}).solids().vals():
        body = body.union(cq.Workplane("XY").add(g))
    body = body.rotate((0, 0, 0), (0, 1, 0), 90)                     # z -> x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(body, str(path), tolerance=0.3, angularTolerance=0.2)
    return path


def case(preset: str, workdir, *, V: float = 22e3, stl=None, iterations: int = 600, environment=None, zones_rz=(10, 6)):
    """The Aeromant case for the preset at ``V`` and its EHD sources: returns (CFDCase, sources, estimate)."""
    from vegeta import aeromant
    t = PRESETS[preset]
    wd = Path(workdir)
    stl = Path(stl) if stl else body_stl(preset, wd.parent / "stl" / f"{preset}.stl")
    if environment is None:
        try:
            environment = aeromant.OpenFOAMEnvironment.detect()
        except RuntimeError:                                          # no OpenFOAM here: prepare only
            environment = aeromant.OpenFOAMEnvironment()
    env = environment
    A = math.pi * t.outer_radius ** 2
    c = aeromant.CFDCase(
        "rans_ksst_external", stl,
        dict(velocity=STILL_AIR, kinematic_viscosity=NU, density=RHO, reference_area=A, reference_length=2 * t.outer_radius,
             center_of_rotation=(0.0, 0.0, 0.0), iterations=iterations, cells_per_length=8.0, surface_level=4,
             upstream=4.0, downstream=10.0, lateral=4.0),
        workdir=wd, geometry_units="mm", environment=env)
    return c


def sources(preset: str, V: float, case_dir, zones_rz=(10, 6), lc: float = 1.0) -> dict:
    """Solve the corona (Fulgur) and write its force as OpenFOAM sources into the prepared case; the jet estimate
    comes along. Under the mocks the force is the 1-D model's, spread over one zone per emitter gap."""
    t = PRESETS[preset]
    r = hf.solve_section(preset, V, lc=lc)
    dz = max(z for _, role, z in hf.rings_of(preset) if role == "ground")
    if r["problem"] is None:                                        # mocked: the 1-D thrust in the gaps
        zones = []
        emitters = [rr for rr, role, _ in hf.rings_of(preset) if role == "emitter"]
        for rr in emitters:
            zones.append({"r1": rr - 0.012, "r2": rr + 0.012, "z1": 0.0, "z2": dz, "force_z": r["thrust"] / len(emitters),
                          "volume": math.pi * ((rr + 0.012) ** 2 - (rr - 0.012) ** 2) * dz})
        angle = 0.25
    else:
        r_edges = np.linspace(0.0, 1.3 * t.outer_radius, zones_rz[0] + 1)
        z_edges = np.linspace(-0.02, dz + 0.02, zones_rz[1] + 1)
        zones = force_zones(r["problem"], r_edges, z_edges)
        angle = convergence_angle(r["problem"], np.mean([rr for rr, role, _ in hf.rings_of(preset) if role == "emitter"]))
    written = write_openfoam_sources(case_dir, zones, axis=(1.0, 0.0, 0.0))
    est = jet_estimate(r["thrust"], t.exit_area, blockage=0.15, half_angle=angle,
                       r_mean=float(np.mean([rr for rr, role, _ in hf.rings_of(preset) if role == "emitter"])))
    return {"solve": {k: v for k, v in r.items() if k != "problem"}, "zones": zones, "written": written, "jet": est}


def prepare(preset: str, workdir, V: float = 22e3, *, run: bool = False, processors: int = 4, lc: float = 1.0) -> dict:
    """Prepare the case (always), write the sources, run it (``run``); the result or the exception in the table."""
    c = case(preset, workdir, V=V)
    c.prepare(overwrite=True)
    src = sources(preset, V, c.workdir, lc=lc)
    out = {"case": c, "sources": src, "result": None}
    if run:
        try:
            c.run(processors=processors)
            out["result"] = c.results()
        except Exception as exc:
            out["result"] = exc
    return out


def table(prepared: dict) -> dict:
    s, j = prepared["sources"]["solve"], prepared["sources"]["jet"]
    row = {"V [kV]": s["V"] / 1e3, "I [mA]": s["I"] * 1e3, "thrust [gf]": s["thrust"] / 9.81 * 1e3, "power [W]": s["power"],
           "zones": prepared["sources"]["written"]["zones"], "v_mean [m/s]": j.v_mean, "v_focal est. [m/s]": j.v_focal,
           "half-angle [deg]": math.degrees(j.half_angle), "focal distance [mm]": j.focal_distance * 1e3}
    r = prepared["result"]
    row["cfd"] = "NOT RUN" if r is None else ("FAILED" if isinstance(r, Exception) else ("solved" if r.ok else "FAILED"))
    return row


if __name__ == "__main__":
    wd = sys.argv[1] if len(sys.argv) > 1 else "cfd/halka_mark4c"
    pr = prepare("mark4c", wd, lc=2.0)
    print(table(pr))
    print("case:", pr["case"].workdir, "run with:", pr["sources"]["written"]["run"])
