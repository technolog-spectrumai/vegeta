"""Prepared Gmsh + CalculiX run (through Talos) for the Auseklis bus: quasi-static launch loads and first modes.

    python bus_fea.py --variant A --workdir runs/bus_A            # writes the STEP and the plan, runs nothing
    python bus_fea.py --variant A --workdir runs/bus_A --run      # meshes and solves (real tools, or mocks if VEGETA_DEBUG=1)

Model (PRELIMINARY, needs external verification): the 25 mm honeycomb shell is modelled as solid aluminium of equivalent
density so that the body load carries the *whole* wet mass (``rho_eq = M_wet / V_shell``); stiffness of a real sandwich
(thin facesheets, light core) is lower, so the modal result is optimistic and the stress result is not a panel check.
Interface: the aft face (x = -bus_x/2) fixed, a stand-in for the launcher separation ring.
Loads: ASSUMED quasi-static limit loads (axial 7 g, lateral 3 g) until a launcher user guide is chosen."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))                                  # labs/space/exp: import auseklis

G_MM = 9810.0                                                             # mm/s^2
CASES = {"axial_7g": {"ax": -7 * G_MM}, "lateral_3g": {"ay": 3 * G_MM}}   # ASSUMED limit load factors
YIELD_FACTOR = 1.25                                                       # ASSUMED required factor of safety on yield
MIN_FIRST_MODE_HZ = 45.0                                                  # ASSUMED, typical small-sat rideshare requirement


def build(variant: str, workdir: Path):
    """Write the bus STEP and return (talos model per case, meta). Imports Talos and CadQuery lazily."""
    import cadquery as cq
    from vegeta import talos
    from auseklis import variants as V
    from auseklis.cad import Auseklis

    workdir.mkdir(parents=True, exist_ok=True)
    d = Auseklis()
    p = d.resolve(variant=variant)
    shell = d.parts(p)["bus_shell"]
    step = workdir / "bus_shell.step"
    cq.exporters.export(cq.Workplane(obj=shell), str(step))
    wet = V.budget(V.VARIANTS[variant], wet_target=p["wet_kg"])["wet"]
    rho = (wet / 1000.0) / shell.Volume()                                # t/mm^3
    material = talos.Material("Al-6061-T6 (equivalent density)", youngs_modulus=68900.0, poissons_ratio=0.33, density=rho,
                              yield_strength=240.0, source="typical handbook values, confirm against MMPDS; density = M_wet / V_shell")
    regions = [talos.SurfacesOnPlane("interface", "x", -p["bus_x"] / 2)]
    models = {name: talos.StructuralModel(geometry=str(step), units="mm-N-MPa", material=material, regions=regions,
                                          supports=[talos.FixedSupport("interface")], loads=[talos.Acceleration(**acc)],
                                          mesh_settings=talos.MeshSettings(element_size=30.0, order=2))
              for name, acc in CASES.items()}
    meta = {"variant": variant, "wet_kg": wet, "rho_eq_t_mm3": rho, "shell_volume_mm3": shell.Volume(), "cases": CASES,
            "required_yield_factor": YIELD_FACTOR, "min_first_mode_hz": MIN_FIRST_MODE_HZ, "status": "PRELIMINARY"}
    (workdir / "plan.json").write_text(json.dumps(meta, indent=1))
    return models, meta


def run(models, workdir: Path, n_modes: int = 6) -> dict:
    out = {}
    for name, model in models.items():
        wd = workdir / name
        model.mesh(wd)
        res = model.solve(wd)
        out[name] = {"status": res.status, "max_von_mises_MPa": res.metrics.get("max_von_mises"),
                     "safety_factor_yield": res.metrics.get("safety_factor_yield")}
    first = next(iter(models.values()))
    modes = first.solve_modes(workdir / "modes", n_modes=n_modes)
    out["modes_hz"] = modes.metrics.get("frequencies_hz")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", default="A", choices="ABC")
    ap.add_argument("--workdir", default="runs/bus")
    ap.add_argument("--run", action="store_true", help="really mesh and solve (otherwise only the inputs are written)")
    a = ap.parse_args(argv)
    from vegeta import mock
    mock.setup(False)                                                   # VEGETA_DEBUG=1 in the environment installs the mocks
    wd = Path(a.workdir)
    models, meta = build(a.variant, wd)
    print(f"prepared {len(models)} load cases in {wd}: {meta}")
    if not a.run:
        print("not run (add --run). Needs: Gmsh + CalculiX (ccx).")
        return 0
    results = run(models, wd)
    (wd / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, indent=1, default=str))
    mocked = os.environ.get("VEGETA_DEBUG", "").lower() in ("1", "true", "on", "yes")
    print("MOCKED numbers: meaningless" if mocked else "real solver results: still PRELIMINARY (see module docstring)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
