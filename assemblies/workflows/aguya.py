"""AGUYA: the turbojet fast sampler, built on the microjet assembly.

Promoted from notebook 29 (without its race against MERLIN: MERLIN is not an assembly yet). The tree::

    aguya (aircraft)               the mission, the design range, the distances
      engine (turbojet)            the saved microjet tree (assemblies/data/microjet.vida), grafted as it is
      airframe (aguya_airframe)    tank and fuselage sized for the design range; speeds; fuel per mission; the design
                                   flight; the dash point; the free-jet estimate at the V-tail; the own-exhaust check
      jet_cfd (jet_external)       the aircraft at the dash with the engine running (Aeromant)
      wing (wing_fea)              the gust and 6 g cases and the vibration modes (Talos)

The engine is not computed here: run ``python -m assemblies.workflows.microjet`` first (or give ``--engine``). A new
engine (a new microjet.vida) changes the engine's key and so every node built on it.

    python -m assemblies.workflows.aguya --fidelity quick -j 4
    python -m assemblies.workflows.aguya --no-cfd --no-fea
    python -m assemblies.workflows.aguya --show

Writes ``assemblies/data/aguya.vida`` and ``assemblies/data/aguya.json``.
"""
from __future__ import annotations

import math
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
from vegeta.boreas import microjet as mj

from .. import DATA, RUNS, vida
from .._cli import main, parser
from ..components import aguya as ag, aguya_flight as F, aguya_jet, aguya_wing
from ..vida import Assembly

NAME = "aguya"
MISSION = F.JetMission(altitude_m=150.0, sampling_s=180.0, sampling_bank_deg=30.0, start_s=45.0, recovery_s=40.0,
                       launch_margin=1.25, reserve=0.15, dash_speed=None)     # dash_speed None: 98 % of the top speed
DISTANCES_KM = (5, 8, 10, 20, 30)
DESIGN_RANGE_KM = 30


def load_engine(path: Path) -> tuple[Assembly, mj.Microjet, F.JetUnit]:
    """The saved microjet tree, its engine and its map as AGUYA's propulsor. Raises when it is not there."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run `python -m assemblies.workflows.microjet` first")
    tree = vida.load(path)
    if "engine" not in tree.results or "map" not in tree.results:
        raise ValueError(f"{path} has no engine results; run the microjet workflow again")
    engine = mj.Microjet(**tree.results["engine"])
    unit = F.jet_unit_from_export({"engine": engine, "map": tree.results["map"]})
    node = tree.copy("engine")
    node.meta["reused"] = {"computed_at": tree.meta.get("computed_at"), "from": str(path)}
    for k in ("loaded_from", "manifest"):
        node.meta.pop(k, None)
    return node, engine, unit


def airframe(node: Assembly, engine: mj.Microjet, unit: F.JetUnit) -> Assembly:
    """Size the tank and fuselage for every distance, fly the missions on the design aircraft, the dash point and the
    exhaust checks (notebook 29 sections 1-8, seconds). Recorded on ``node``."""
    p = node.params
    mission, budget = F.JetMission(**p["mission"]), dict(p["budget"])
    P0 = p["start_geometry"]
    sizing, sized = {}, {}
    for d in p["distances_km"]:
        q, a, f = F.size_for(P0, unit, d * 1000.0, mission, budget=budget)
        sized[d] = (q, a, f)
        sizing[str(d)] = {"fuselage_diameter_mm": q["fuselage_diameter"], "tank_kg": a.tank_kg, **_plain(f.summary())}
    P, af, design_flight = sized[p["design_range_km"]]
    flights = {d: F.fuel_for(af, unit, d * 1000.0, mission) for d in p["distances_km"]}
    m_full, m_dry = af.dry_mass_kg + af.tank_kg, af.dry_mass_kg
    speeds = {"top_speed_full_m_s": F.top_speed(af, unit, m_full), "top_speed_dry_m_s": F.top_speed(af, unit, m_dry),
              "stall_full_m_s": af.stall_speed(m_full), "catapult_release_m_s": mission.launch_margin * af.stall_speed(m_full),
              "cd0": af.cd0}
    speeds["catapult_energy_J"] = 0.5 * m_full * speeds["catapult_release_m_s"] ** 2
    fl = flights[p["design_range_km"]]
    point = aguya_jet.dash_point(P, af, unit, engine, fl)
    node.record(feasible=bool(design_flight.feasible), geometry=P, airframe=asdict(af), sizing=sizing, speeds=speeds,
                missions={str(d): _plain(f.summary()) for d, f in flights.items()},
                design_flight={"track": {k: np.asarray(v, float) for k, v in fl.track.items()},
                               "phases": {k: {"time_s": v[0], "fuel_kg": v[1], "distance_m": v[2]} for k, v in fl.phases.items()}},
                dash=point, own_exhaust=own_exhaust(af, unit, mission, fl))
    if not design_flight.feasible:
        node.not_run(f"no fuselage up to 180 mm carries the fuel for {p['design_range_km']} km")
    return node


def own_exhaust(af, unit, mission, flight) -> dict:
    """Does AGUYA sample its own exhaust? The CO2 and CO it adds to its trail after one lap (notebook 29 section 8)."""
    g = 9.80665
    v_s = 1.3 * af.stall_speed(af.dry_mass_kg + 0.5 * flight.fuel_kg, n=1 / math.cos(math.radians(30)))
    ff_idle = unit.idle(v_s)[1]
    radius = v_s ** 2 / (g * math.tan(math.radians(mission.sampling_bank_deg)))
    lap = 2 * math.pi * radius / v_s + 2 * 400 / v_s
    sigma = math.sqrt(2 * 1.0 * lap)
    air = math.pi * (2 * sigma) ** 2

    def ppm(g_per_m3, molar):
        return g_per_m3 / (molar / 0.0224) * 1e6 * (273.15 / 288.15)

    return {"pass_speed_m_s": v_s, "turn_radius_m": radius, "lap_s": lap, "trail_diameter_m": 4 * sigma,
            "idle_fuel_g_min": ff_idle * 60e3, "co2_increment_ppm": ppm(3.15 * ff_idle * 1000 / v_s / air, 44.0),
            "co_increment_ppm": ppm(0.100 * ff_idle * 1000 / v_s / air, 28.0)}


def _plain(d: dict) -> dict:
    return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in d.items()}


def build(engine_node: Assembly, engine: mj.Microjet, fidelity: str = "full", design_range_km: int = DESIGN_RANGE_KM,
          distances_km=DISTANCES_KM) -> Assembly:
    """The tree down to the airframe (the CFD and FEA nodes need the sized airframe: ``run`` adds them)."""
    if design_range_km not in distances_km:
        raise ValueError("design_range_km must be one of distances_km")
    root = Assembly(NAME, "aircraft", params={"fidelity": fidelity, "design_range_km": design_range_km,
                                              "distances_km": list(distances_km)})
    root.add(engine_node)
    root.add(Assembly("airframe", "aguya_airframe", params={
        "mission": asdict(MISSION), "budget": dict(F.DRY_MASS_KG), "start_geometry": ag.for_engine(engine),
        "design_range_km": design_range_km, "distances_km": list(distances_km)}))
    return root


def run(*, fidelity: str = "full", design_range_km: int = DESIGN_RANGE_KM, distances_km=DISTANCES_KM,
        engine_vida: Path | None = None, run_cfd: bool = True, run_fea: bool = True, processors: int = 1, jobs: int = 1,
        threads: int = 1, out: Path | None = None, vida_path: Path | None = None, include: str = "results",
        export: bool = True, export_path: Path | None = None, force: bool = False, redo=(), progress: bool = True) -> Assembly:
    """Compute what the saved tree does not have, save the tree, write the export. Returns the tree."""
    out = Path(out or RUNS / NAME).resolve()
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    engine_node, engine, unit = load_engine(Path(engine_vida or DATA / "microjet.vida"))
    if any(r == "engine" or r.startswith("engine/") for r in redo):
        raise ValueError("the engine is the microjet assembly's: redo it with python -m assemblies.workflows.microjet")
    if force and out.exists():
        shutil.rmtree(out)
    prior = vida.load(vida_path) if vida_path.is_file() and not force else None

    root = build(engine_node, engine, fidelity, design_range_km, distances_km)
    root.reuse(prior)
    _redo(root, redo)
    air = root.child("airframe")
    if not air.results:
        airframe(air, engine, unit)

    P = air.results["geometry"]
    af = F.JetAirframe(**air.results["airframe"])
    point = air.results["dash"]
    root.add(aguya_jet.assembly(P, af.wing_area_m2, point, fidelity))
    root.add(aguya_wing.assembly(P, af, point["v_dash"], fidelity))
    root.reuse(prior)
    _redo(root, [r for r in redo if r.split("/")[0] in ("jet_cfd", "wing")])

    jet = root.child("jet_cfd")
    if not jet.results.get("complete"):
        aguya_jet.solve(jet, out / "jet_cfd", run=run_cfd, processors=processors, progress=progress)
    wing = root.child("wing")
    if not wing.results.get("complete"):
        aguya_wing.solve(wing, out / "wing", run=run_fea, threads=threads, progress=progress)

    root.record(summary={"fuselage_diameter_mm": P["fuselage_diameter"], "tank_kg": af.tank_kg, "dash_m_s": point["v_dash"],
                         "drag_at_dash_buildup_N": point["buildup_drag_N"],
                         "drag_at_dash_cfd_N": (jet.results.get("forces") or {}).get("drag_force_N"),
                         "gust_load_factor": wing.params["gust"]["load_factor"],
                         "gust_penetration_speed_m_s": wing.params["gust"]["penetration_speed"]})
    root.not_run("race against MERLIN: MERLIN is not an assembly yet (notebook 29 section 5 has it)")
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:
        path = Path(export_path or DATA / f"{NAME}.json")
        doc = {"params": P, "engine": engine.to_dict(), "engine_source": str(engine_vida or DATA / "microjet.vida"),
               "airframe": air.results["airframe"], "mass_budget_kg": air.params["budget"], "speeds": air.results["speeds"],
               "missions": air.results["missions"], "sizing": air.results["sizing"], "design_range_km": design_range_km,
               "jet": {**point, "cfd": jet.results.get("forces")}, "gust": wing.params["gust"],
               "fea": (wing.results.get("stress") or {}), "modes_hz": (wing.results.get("modes") or {}).get("frequency_hz"),
               "source": {"workflow": root.meta["workflow"], "fidelity": fidelity, "vida": vida_path.name,
                          "written_at": vida.utc_now(), "git": vida.manifest(vida_path)["git"],
                          "jet_cfd_complete": bool(jet.results.get("complete")), "wing_fea_complete": bool(wing.results.get("complete"))}}
        path.write_text(_json(doc))
        root.meta["exported_to"] = str(path)
    return root


def _redo(root: Assembly, redo) -> None:
    for path in redo:
        try:
            root.child(path).forget()
        except KeyError:
            if path.split("/")[0] not in ("jet_cfd", "wing"):      # added after the airframe: forgotten then
                raise


def _json(doc) -> str:
    import json

    def plain(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v

    return json.dumps(plain(doc), indent=1, default=str)


def _parser():
    ap = parser("AGUYA on the microjet: sizing, missions, jet CFD at the dash, wing FEA; writes aguya.vida and aguya.json")
    ap.add_argument("--engine", dest="engine_vida", type=Path, default=None,
                    help="the microjet .vida to fly (default assemblies/data/microjet.vida)")
    ap.add_argument("--design-range-km", type=int, default=DESIGN_RANGE_KM, help="the range the tank is sized for")
    return ap


if __name__ == "__main__":
    raise SystemExit(main(run, _parser(), DATA / f"{NAME}.vida"))
