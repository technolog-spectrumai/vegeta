"""The microjet: a model turbojet from its cycle to a CFD speed line of the compressor and the wheels in FEA.

Promoted from notebook 28. The tree::

    microjet (turbojet)            the engine class; results: the engine (calibrated to the CFD compressor when there is
                                   a solved speed line), its design point, its map over airspeed x shaft speed
      parts (turbojet_parts)       impeller, turbine wheel and engine outside as STEP; masses, sizes
      compressor (compressor_speedline)   Aeromant compressor_mrf at full speed over the back pressures of the fidelity
      wheels (wheel_fea)           Talos: turbine and impeller at full speed, hot, and at 115 % overspeed

The compressor and the wheels are sized from the catalogue engine (the cycle before the CFD refit), as in notebook 28;
the refit changes the export, not the geometry.

    python -m assemblies.workflows.microjet --fidelity quick -j 4        # solves what is missing; again: reads back
    python -m assemblies.workflows.microjet --no-cfd --no-fea            # the cycle only; CFD/FEA shown as NOT RUN
    python -m assemblies.workflows.microjet --show                       # the saved tree

Writes ``assemblies/data/microjet.vida`` and ``assemblies/data/microjet.json`` (the export notebook 29 reads).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from vegeta.boreas import microjet as mj

from .. import DATA, RUNS, vida, results
from .._cli import main, parser
from ..components import cycle, impeller, turbojet as tj, turbojet_parts, wheels
from ..vida import Assembly

NAME = "microjet"
DEFAULT_ENGINE = "140 N class"
ATMOSPHERES = {"ISA sea level": (0, 0), "ISA 1500 m": (1500, 0), "1500 m, ISA+20 (≈ 25 °C)": (1500, 20),
               "sea level, 35 °C": (0, 20)}       # notebook 28 cell 10: name -> mj.isa(altitude [m], ISA offset [K])


def build(engine_class: str = DEFAULT_ENGINE, fidelity: str = "full") -> Assembly:
    """The tree with its parameters; nothing is computed (seconds)."""
    engine = cycle.from_catalogue(engine_class)
    P = tj.sized(engine)
    root = Assembly(NAME, "turbojet", params={"engine_class": engine_class, "fidelity": fidelity,
                                              "catalogue_engine": engine.to_dict()})
    root.add(turbojet_parts.assembly(P))
    root.add(impeller.assembly(engine, P, fidelity))
    root.add(wheels.assembly(engine, P, fidelity))
    return root


def run(*, engine_class: str = DEFAULT_ENGINE, fidelity: str = "full", run_cfd: bool = True, run_fea: bool = True,
        processors: int = 1, jobs: int = 1, threads: int = 1, out: Path | None = None, vida_path: Path | None = None,
        include: str = "results", export: bool = True, export_path: Path | None = None, force: bool = False,
        redo=(), progress: bool = True) -> Assembly:
    """Compute what the saved tree does not have, save the tree, write the export. Returns the tree."""
    out = Path(out or RUNS / NAME).resolve()
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    if force and out.exists():
        shutil.rmtree(out)
    root = build(engine_class, fidelity)
    root.reuse(vida.load(vida_path) if vida_path.is_file() and not force else None)
    for path in redo:
        root.child(path).forget()

    steps = turbojet_parts.make(root.child("parts"), out / "parts")

    comp = root.child("compressor")
    if not comp.results.get("complete"):
        impeller.speedline(comp, out / "compressor", run=run_cfd, processors=processors, jobs=jobs, progress=progress)

    wh = root.child("wheels")
    if not wh.results.get("complete"):
        wheels.solve(wh, out / "wheels", steps, run=run_fea, threads=threads, progress=progress)

    catalogue = cycle.from_catalogue(engine_class)
    engine, calibration = cycle.calibrate_from_speedline(catalogue, engine_class, comp.results.get("speedline"))
    root.record(engine=engine.to_dict(), calibration=calibration, design_point=cycle.design_point(engine),
                catalogue_design_point=cycle.design_point(catalogue), map=cycle.performance_map(engine),
                metal_temperatures_K=tj_metal_temperatures(engine))
    engines = {k: catalogue if k == engine_class else cycle.from_catalogue(k) for k in mj.CATALOGUE}
    root.record(atmosphere_cases=atmosphere_cases(engine), catalogue_classes=catalogue_classes(engines),
                catalogue_full_throttle=catalogue_full_throttle(engines))
    if calibration is None:
        root.not_run("calibration to the CFD compressor (no solved speed line point): the catalogue cycle is exported")
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:                                                    # plain data for notebooks (assemblies.results)
        root.meta["results_to"] = str(results.write(root, vida_path))

    if export:
        path = Path(export_path or DATA / f"{NAME}.json")
        stress = wh.results.get("stress", {})
        fea = {k: {"ok": v["ok"], "max_von_mises_MPa": v["max_von_mises_MPa"], "safety_factor": v["safety_factor_yield"]}
               for k, v in stress.items()}
        root.meta["exported_to"] = str(cycle.export(engine, path, extra={
            "engine_class": engine_class, "turbojet_params": root.child("parts").params["geometry"],
            "wheel_masses_kg": root.child("parts").results.get("masses_kg"),
            "metal_temperatures_K": root.results["metal_temperatures_K"], "compressor_cfd": calibration, "fea": fea,
            "source": {"workflow": root.meta["workflow"], "fidelity": fidelity, "vida": vida_path.name,
                       "written_at": vida.utc_now(), "git": vida.manifest(vida_path)["git"],
                       "compressor_cfd_complete": bool(comp.results.get("complete")),
                       "wheels_fea_complete": bool(wh.results.get("complete"))}}))
    return root


def tj_metal_temperatures(engine) -> dict:
    from vegeta.boreas import microjet as mj

    return mj.metal_temperatures(mj.solve(engine, engine.rpm_max))


def atmosphere_cases(engine: mj.Microjet) -> dict:
    """Full throttle in the four atmospheres of notebook 28 section 3 (cell 10): density, static thrust, thrust and fuel
    flow at 150 m/s, the static EGT. One row per atmosphere of ``ATMOSPHERES`` (``results.load("microjet").table("",
    "atmosphere_cases")``). Run on the exported engine (the catalogue one when no speed line calibrated it)."""
    cases = {name: mj.isa(*h_dt) for name, h_dt in ATMOSPHERES.items()}
    rows = {}
    for name, atm in cases.items():
        p0, p150 = mj.solve(engine, engine.rpm_max, 0.0, atm), mj.solve(engine, engine.rpm_max, 150.0, atm)
        rows[name] = {"density_kg_m3": atm.density, "static_thrust_N": p0.thrust, "thrust_150_m_s_N": p150.thrust,
                      "fuel_150_m_s_g_min": p150.fuel_flow_g_min, "egt_K": p0.t5,
                      "altitude_m": ATMOSPHERES[name][0], "delta_isa_K": ATMOSPHERES[name][1]}
    return rows


def catalogue_classes(engines: dict) -> dict:
    """The design-point row of every catalogue class (notebook 28 cell 4: ``cycle_row`` is ``cycle.design_point``),
    ``{class: row}``; ``results.load("microjet").table("", "catalogue_classes").T`` is the cell's table."""
    return {k: cycle.design_point(e) for k, e in engines.items()}


def catalogue_full_throttle(engines: dict, V=cycle.V_MAP) -> dict:
    """Full throttle against airspeed for every catalogue class (notebook 28 cell 8): net thrust, fuel flow, TSFC and
    the overall and propulsive efficiencies over ``V`` (the export's airspeed axis, the cell's ``linspace(0, 220, 23)``).
    ``{class: {column: array}}``: ``pd.DataFrame(r[""]["catalogue_full_throttle"][class])`` is one class's curve."""
    out = {}
    for k, e in engines.items():
        full = [mj.solve(e, e.rpm_max, v) for v in V]
        out[k] = {"V_m_s": np.asarray(V, float), "thrust_N": np.array([p.thrust for p in full], float),
                  "fuel_g_min": np.array([p.fuel_flow_g_min for p in full], float),
                  "tsfc_kg_N_h": np.array([p.tsfc for p in full], float),
                  "overall_efficiency": np.array([p.overall_efficiency(e.lhv) for p in full], float),
                  "propulsive_efficiency": np.array([p.propulsive_efficiency for p in full], float)}
    return out


def _parser():
    ap = parser("The microjet: cycle, compressor speed line (CFD), wheels (FEA); writes microjet.vida and microjet.json")
    ap.add_argument("--engine-class", default=DEFAULT_ENGINE, help=f"a microjet.CATALOGUE class; default {DEFAULT_ENGINE!r}")
    return ap


if __name__ == "__main__":
    raise SystemExit(main(run, _parser(), DATA / f"{NAME}.vida"))
