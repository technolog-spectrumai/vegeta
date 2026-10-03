"""The microjet's spinning wheels in FEA (Talos): the cast Inconel turbine wheel and the forged aluminium impeller under
centrifugal load and their metal temperatures, at the maximum speed and at 115 % overspeed.

Promoted from notebook 28 section 6. The bore is the clamped shaft fit (a simplification: a real wheel sits on a
nut-clamped hub). Material values are typical handbook numbers; replace them with the supplier's. Creep is not in the
safety factor.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from vegeta import talos
from vegeta.boreas import microjet as mj

from ..vida import Assembly
from . import turbojet as tj

IN713C = talos.Material("Inconel 713C (cast)", 200000, 0.30, density=7.91e-9, yield_strength=740, units="mm-N-MPa",
                        thermal_expansion=1.45e-5, reference_temperature=293.15,
                        temperature_table=((293.15, 200000, 0.30, 740), (811.0, 176000, 0.31, 705),
                                           (1033.0, 165000, 0.32, 650), (1144.0, 157000, 0.32, 545)),
                        source="typical handbook values (Special Metals / ASM); replace with the supplier's")
AL2618 = talos.Material("Al 2618-T61 (forged)", 74000, 0.33, density=2.76e-9, yield_strength=370, units="mm-N-MPa",
                        thermal_expansion=2.23e-5, reference_temperature=293.15,
                        temperature_table=((293.15, 74000, 0.33, 370), (373.15, 71000, 0.33, 350),
                                           (423.15, 68000, 0.33, 315), (473.15, 63000, 0.33, 255)),
                        source="typical handbook values (ASM); replace with the supplier's")
MATERIALS = {"turbine": IN713C, "impeller": AL2618}
ELEMENT_MM = {"smoke": 4.0, "quick": 2.5, "full": 1.0}
OVERSPEED = 1.15
CASES = ("turbine_spin", "turbine_hot", "turbine_overspeed", "impeller_hot", "impeller_overspeed")
AXIS = (1.0, 0.0, 0.0)


def assembly(engine: mj.Microjet, P: dict, fidelity: str) -> Assembly:
    """The wheels node: geometry, speeds, metal temperatures from the cycle at full speed, materials, element size."""
    if fidelity not in ELEMENT_MM:
        raise ValueError(f"fidelity must be one of {tuple(ELEMENT_MM)}")
    hot = mj.solve(engine, engine.rpm_max)
    mt = mj.metal_temperatures(hot)
    ts, st = tj.Turbojet.turbine_stations(P), tj.Turbojet.stations(P)
    return Assembly("wheels", "wheel_fea", params=dict(
        geometry=P, rpm=float(engine.rpm_max), overspeed=OVERSPEED, element_mm=ELEMENT_MM[fidelity], fidelity=fidelity,
        turbine_temperature={"radii": [ts["r_bore"], ts["r_rim"], ts["r_tip"]],
                             "temperatures": [mt["bore_K"], mt["rim_K"], mt["blade_K"]]},
        impeller_temperature={"radii": [st["r1h"], st["r2"]], "temperatures": [hot.t2, hot.t3]},
        materials={k: asdict(m) for k, m in MATERIALS.items()}))


def models(node: Assembly, steps: dict[str, Path]) -> dict[str, talos.StructuralModel]:
    """The five static models of ``CASES`` on the wheels' STEP files."""
    p = node.params
    rb = p["geometry"]["bore_diameter"] / 2 + 0.05
    regions = [talos.SurfacesInBox("bore", (-200.0, -rb, -rb, 200.0, rb, rb))]
    t_turb = talos.RadialTemperature(radii=tuple(p["turbine_temperature"]["radii"]),
                                     temperatures=tuple(p["turbine_temperature"]["temperatures"]), axis=AXIS)
    t_imp = talos.RadialTemperature(radii=tuple(p["impeller_temperature"]["radii"]),
                                    temperatures=tuple(p["impeller_temperature"]["temperatures"]), axis=AXIS)
    spin, over = talos.Centrifugal(p["rpm"], axis=AXIS), talos.Centrifugal(p["overspeed"] * p["rpm"], axis=AXIS)
    loads = {"turbine_spin": [spin], "turbine_hot": [spin, t_turb], "turbine_overspeed": [over, t_turb],
             "impeller_hot": [spin, t_imp], "impeller_overspeed": [over, t_imp]}
    out = {}
    for name in CASES:
        part = name.split("_")[0]
        out[name] = talos.StructuralModel(steps[part], "mm-N-MPa", MATERIALS[part], regions, [talos.FixedSupport("bore")],
                                          loads[name], talos.MeshSettings(element_size=p["element_mm"]), name=name)
    return out


def solve(node: Assembly, out: Path, steps: dict[str, Path], *, run: bool = True, threads: int = 1,
          progress=True) -> Assembly:
    """Mesh and solve the five models in ``out/<case>`` on the wheels' STEP files ``steps`` (``{"turbine": ...,
    "impeller": ...}``, from ``turbojet_parts.make``; solved models are read back) and record on ``node``: one row per
    case (``stress``: ok, max von Mises [MPa], safety factor on yield at temperature and where, max displacement [mm],
    elements). With nothing solved the node says NOT RUN and why."""
    out = Path(out)
    ms = models(node, steps)
    rs = talos.solve_models(list(ms.values()), [out / name for name in ms], threads=threads, run=run, progress=progress)
    rows = {}
    for (name, model), r in zip(ms.items(), rs):
        m = r.metrics if r.ok else {}
        rows[name] = {"ok": bool(r.ok), "max_von_mises_MPa": m.get("max_von_mises"),
                      "safety_factor_yield": m.get("safety_factor_yield"),
                      "safety_factor_temperature_K": m.get("safety_factor_temperature"),
                      "max_displacement_mm": m.get("max_displacement"), "elements": m.get("n_elements")}
        if (out / name).exists():
            node.attach(name, out / name, "mesh")
        if not r.ok:
            node.not_run(f"{name}: {r.messages[-1] if r.messages else r.status}")
    if any(row["ok"] for row in rows.values()):
        node.record(stress=rows, complete=all(row["ok"] for row in rows.values()))
    elif not run:
        node.meta["not_run"] = ["run_fea=False"]
    return node
