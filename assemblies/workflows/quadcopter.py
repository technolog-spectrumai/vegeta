"""The quadcopter: the printed X frame, its propellers and drive, from notebook 08 (Parts 1 and 2).

The tree::

    quadcopter (multirotor)        the mass budget, the drive; results: hover / cruise / full-throttle points, endurance,
                                   the map over rpm x airspeed (exported like notebook 08's quad_5x43.json)
      frame (quad_frame)           the frame CAD; volume and mass
      frame_fea (frame_fea)        full thrust on all motors, a hard landing on one pad (Talos)
      canopy_cfd (frame_canopy)    the frame with a canopy at 15 m/s (Aeromant rans_ksst_external)
      propeller (propeller)        the catalogue propeller's CAD
      rotor_cfd (rotor)            the propeller in hover at the hover rpm (Aeromant rotor_mrf_static)
      blade_fea (blade)            one blade at full throttle (Talos)

The frame is the design's defaults unless ``frame`` overrides some (notebook 08 compared three and preferred one: give
its parameters, e.g. ``--frame arm_width=16 --frame taper=0.55``). The vibration and life part (notebook 08 Part 3)
stays in the notebook.

    python -m assemblies.workflows.quadcopter --fidelity quick -j 4
    python -m assemblies.workflows.quadcopter --no-cfd --no-fea --show
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from vegeta import boreas

from .. import DATA, RUNS, vida
from .._cli import main, parser
from .._cli_util import parse_assignments
from ._common import add_after, run_cfd as _cfd_node, solve_fea
from ..components import propeller as pr, quad_frame_analysis as qa
from ..components._cad import export_kept
from ..components.impeller import environment
from ..components.quad_frame import QuadFrame
from ..vida import Assembly
from vegeta import aeromant

NAME = "quadcopter"
PARTS_G = {"motors 2306 (4x)": 4 * 30.0, "propellers 5x4.3 (4x)": 4 * 5.0, "4-in-1 ESC": 15.0, "flight controller": 10.0,
           "battery 4S 1500 mAh": 180.0, "camera + VTX + antenna": 35.0, "receiver, wiring, bolts": 30.0}   # cell 5
DRIVE = {"propeller": "5x4.3 tri-blade", "motor": "2306-2400KV", "battery": "4S 1500 mAh", "motors": 4, "rho": 1.2}
MAX_THRUST_PER_MOTOR_N, LANDING_N, CANOPY_SPEED = 8.0, 40.0, 15.0
BLADE_ELEMENT_MM = {"smoke": 2.0, "quick": 1.2, "full": 0.8}                               # full: the notebook's
MAP_RPM, MAP_V = np.linspace(5000, 30000, 11), np.linspace(0, 15, 6)                       # cell 62
G = 9.81


def build(fidelity: str = "full", frame: dict | None = None) -> Assembly:
    """The tree with its parameters; nothing is computed."""
    p = QuadFrame().resolve(**dict(frame or {}))
    root = Assembly(NAME, "multirotor", params={"fidelity": fidelity, "parts_g": PARTS_G, "drive": DRIVE})
    root.add(Assembly("frame", "quad_frame", params={"frame": p, "density_g_mm3": qa.DENSITY_G_MM3}))
    root.add(Assembly("frame_fea", "frame_fea", params={"frame": p, "material": qa.PETG_CF, "element_mm": qa.ELEMENT_MM[fidelity],
                                                        "max_thrust_per_motor_N": MAX_THRUST_PER_MOTOR_N, "landing_N": LANDING_N}))
    root.add(Assembly("canopy_cfd", "frame_canopy", params={"frame": dict(p, canopy_height=25.0),
                                                            "cfd": qa.canopy_params(p, CANOPY_SPEED, fidelity)}))
    spec = pr.get(DRIVE["propeller"])
    root.add(Assembly("propeller", "propeller", params={"spec": spec.to_dict(), "section": pr.SECTIONS[spec.section]}))
    return root


def drive_points(auw_kg: float):
    spec = pr.get(DRIVE["propeller"])
    prop, sec = spec.model(), spec.airfoil()
    system = boreas.Propulsion(prop, sec, pr.motor(DRIVE["motor"]), pr.battery(DRIVE["battery"]), rho=DRIVE["rho"])
    hover_thrust = auw_kg * G / DRIVE["motors"]
    pts = {"hover": system.for_thrust(hover_thrust), "cruise": system.for_thrust(hover_thrust * 1.3),
           "full": system.at_throttle(1.0)}                                                 # cell 56
    return prop, sec, system, pts


def _point(p) -> dict:
    return {"throttle": float(p.throttle), "rpm": float(p.rpm), "thrust_N": float(p.thrust), "current_A": float(p.current),
            "electrical_W": float(p.electrical_power), "current_limited": bool(p.current_limited)}


def run(*, fidelity: str = "full", frame: dict | None = None, run_cfd: bool = True, run_fea: bool = True, processors: int = 1,
        jobs: int = 1, threads: int = 1, out: Path | None = None, vida_path: Path | None = None, include: str = "results",
        export: bool = True, export_path: Path | None = None, force: bool = False, redo=(), progress: bool = True) -> Assembly:
    out = Path(out or RUNS / NAME).resolve()
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    if force and out.exists():
        shutil.rmtree(out)
    prior = vida.load(vida_path) if vida_path.is_file() and not force else None
    root = build(fidelity, frame)
    root.reuse(prior)
    for path in redo:
        if path not in ("rotor_cfd", "blade_fea"):                    # added later, after the drive is known
            root.child(path).forget()

    # the frame: CAD, mass (seconds; files kept while the parameters are unchanged)
    fr = root.child("frame")
    fp = fr.params["frame"]
    files, info = export_kept(lambda: QuadFrame().generate(**fp), fp, out / "frame", "frame")
    for f, path in files.items():
        fr.attach(f"frame.{f}", path, "geometry")
    if not fr.results:
        fr.record(volume_mm3=info["volume_mm3"], mass_g=info["volume_mm3"] * qa.DENSITY_G_MM3, valid=info["valid"],
                  dimensions_mm=info["dimensions_mm"])
    auw_kg = (sum(PARTS_G.values()) + fr.results["mass_g"]) / 1000

    # the frame in FEA
    fea = root.child("frame_fea")
    if not fea.results.get("complete"):
        fp2 = fea.params
        models = qa.frame_models(files["step"], fp2["frame"], max_thrust_per_motor_N=fp2["max_thrust_per_motor_N"],
                                 landing_N=fp2["landing_N"], element_mm=fp2["element_mm"])
        solve_fea(fea, models, out / "frame_fea", run=run_fea, threads=threads, progress=progress)

    # the frame with a canopy in CFD
    can = root.child("canopy_cfd")
    if not can.results.get("complete"):
        cp = can.params["frame"]
        cfiles, _ = export_kept(lambda: QuadFrame().generate(**cp), cp, out / "canopy", "frame_canopy", formats=("stl",),
                                stl_tolerance=0.1)
        can.attach("frame_canopy.stl", cfiles["stl"], "geometry")
        case = qa.canopy_case(cfiles["stl"], can.params["cfd"], out / "canopy_cfd" / "forward_15ms", environment(run_cfd))
        _cfd_node(can, case, ("drag_force_N", "lift_force_N", "Cd", "Cl", "converged", "mesh_cells"), run=run_cfd,
                  processors=processors, progress=progress)

    # the drive: hover, cruise, full throttle; the map (seconds)
    spec = pr.get(DRIVE["propeller"])
    prop, sec, system, pts = drive_points(auw_kg)
    battery = pr.battery(DRIVE["battery"])
    hover_min = battery.usable_wh / (DRIVE["motors"] * pts["hover"].electrical_power) * 60
    grid = boreas.performance_map(prop, sec, MAP_RPM, MAP_V, DRIVE["rho"])

    # the propeller: CAD, rotor CFD at hover, one blade in FEA at full throttle
    pn = root.child("propeller")
    pfiles = pr.cad_files(spec, out / "propeller")
    bfiles = pr.cad_files(spec, out / "blade", blades=1)
    pn.attach("propeller.step", pfiles["step"], "geometry")
    pn.attach("propeller_axis_x.stl", pfiles["stl_axis_x"], "geometry")
    if not pn.results:
        pn.record(describe=prop.describe())
    rc = add_after(root, Assembly("rotor_cfd", "rotor", params=pr.rotor_params(spec, pts["hover"].rpm, medium="air", fidelity=fidelity)
                               | {"density": DRIVE["rho"]}), prior, redo)
    if not rc.results.get("complete"):
        case = pr.rotor_case(pfiles["stl_axis_x"], rc.params, out / "rotor_cfd" / "hover", environment(run_cfd))
        _cfd_node(rc, case, ("thrust_N", "torque_Nm", "power_W", "figure_of_merit", "converged", "mesh_cells"), run=run_cfd,
                  processors=processors, progress=progress)
    loads = pr.blade_loads(prop, pts["full"])
    bl = add_after(root, Assembly("blade_fea", "blade", params={"spec": spec.to_dict(), "loads": loads,
                                                             "element_mm": BLADE_ELEMENT_MM[fidelity],
                                                             "material": pr.BLADE_MATERIAL}), prior, redo)
    if not bl.results.get("complete"):
        model = pr.blade_model(spec, bfiles["step"], loads, element_mm=bl.params["element_mm"], name="blade_full_throttle")
        solve_fea(bl, {"blade_full_throttle": model}, out / "blade_fea", run=run_fea, threads=threads, progress=progress)

    bemt_hover = pts["hover"].thrust
    cfd_hover = (rc.results.get("forces") or {}).get("thrust_N")
    root.record(auw_kg=auw_kg, points={k: _point(v) for k, v in pts.items()}, hover_endurance_min=hover_min,
                thrust_to_weight=DRIVE["motors"] * pts["full"].thrust / (auw_kg * G),
                map={"rpm": np.asarray(grid["rpm"], float), "airspeed_m_s": np.asarray(grid["airspeed_m_s"], float),
                     "thrust_N": np.asarray(grid["thrust_n"], float)},
                hover_thrust_bemt_vs_cfd={"bemt_N": float(bemt_hover), "cfd_N": cfd_hover})
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:
        path = Path(export_path or DATA / f"{NAME}.json")
        res = boreas.export(path, prop, sec, map=grid, motor=pr.motor(DRIVE["motor"]), battery=battery,
                            points={"hover": pts["hover"], "cruise": pts["cruise"], "full": pts["full"]},
                            notes=f"quadcopter from assemblies.workflows.{NAME} ({fidelity}); AUW {auw_kg:.3f} kg, "
                                  f"{DRIVE['motors']} motors; hover endurance {hover_min:.1f} min")
        res.raise_for_status()
        root.meta["exported_to"] = str(path)
    return root


def _parser():
    ap = parser("The quadcopter: frame FEA and canopy CFD, propeller drive, rotor CFD, blade FEA; writes quadcopter.vida/.json")
    ap.add_argument("--frame", action="append", default=[], type=str, metavar="NAME=VALUE",
                    help="a frame parameter other than the default (repeatable), e.g. --frame arm_width=16")
    return ap


def _run_cli(*, frame=(), **kw):
    return run(frame=parse_assignments(frame), **kw)


if __name__ == "__main__":
    raise SystemExit(main(_run_cli, _parser(), DATA / f"{NAME}.vida"))
