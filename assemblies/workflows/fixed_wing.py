"""The fixed wing: the twin-motor drone of notebook 09a (Parts 1-3), whose hand-off notebook 09b reads.

The tree::

    fixed_wing (aircraft)          the mass budget; the polar through the CFD point; cruise / climb / static /
                                   engine-out points; endurance, range (the hand-off of 09a -> 09b)
      airframe (fixed_wing)        the aircraft CAD; planform, shell and nacelle masses, all-up mass
      wing_fea (wing_fea)          the wing in a 2.5 g pull-up and with one engine out (Talos)
      aero_cfd (aircraft_4deg)     the whole aircraft at 4 deg in cruise (Aeromant rans_ksst_external)
      propeller (propeller)        the 9x6 propeller's CAD
      rotor_cfd (rotor)            the propeller at cruise (Aeromant rotor_mrf)
      blade_fea (blade)            one blade at full throttle in the climb (Talos)
      installed_cfd (rotor_disks)  the aircraft with both propellers as rotor disks (Aeromant aircraft_rotor_disks)

The aircraft is the design's defaults unless ``aircraft`` overrides some: notebook 09a preferred the thicker NACA 2415
wing, ``--aircraft thickness=0.15``. Without the whole-aircraft CFD the polar goes through notebook 09a's recorded
point (Cl 0.40, Cd 0.070, coarse mesh) and the tree says so. Notebook 09b (durability) stays a notebook.

    python -m assemblies.workflows.fixed_wing --fidelity quick -j 4 --aircraft thickness=0.15
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from vegeta import aeromant, boreas, talos

from .. import DATA, RUNS, vida
from .._cli import main, parser
from .._cli_util import parse_assignments
from ..components import propeller as pr, wing
from ..components._cad import export_kept
from ..components.fixed_wing import FixedWing
from ..components.impeller import environment
from ..vida import Assembly
from ._common import add_after, run_cfd as cfd_node, solve_fea

NAME = "fixed_wing"
PARTS_G = {"motors 2212 (2x)": 2 * 55.0, "propellers 9x6 (2x)": 2 * 12.0, "ESC 30 A (2x)": 2 * 25.0,
           "battery 3S 5000 mAh": 380.0, "flight controller + GPS": 35.0, "servos (4x)": 4 * 12.0,
           "receiver, wiring, bolts": 45.0, "payload (camera)": 150.0}                       # 09a cell 4
SHELL_AREAL_DENSITY_G_MM2, NACELLE_DENSITY_G_MM3 = 1.1e-4, 0.5e-3                          # 09a cell 4
DRIVE = {"propeller": "9x6 electric", "motor": "2212-920KV", "battery": "3S 5000 mAh", "motors": 2, "rho": 1.2}
V_CRUISE, AOA_DEG, G = 14.0, 4.0, 9.81
RECORDED_POLAR = {"Cl": 0.40, "Cd": 0.070}                                                 # 09a cell 40
OSWALD = 0.8
WING_ELEMENT_MM = {"smoke": 20.0, "quick": 14.0, "full": 10.0}                             # full: the notebook's
BLADE_ELEMENT_MM = {"smoke": 2.5, "quick": 1.8, "full": 1.2}
AERO = {"smoke": dict(iterations=60, surface_level=3, near_level=2, wake_level=1),
        "quick": dict(iterations=400, surface_level=3, near_level=2, wake_level=1),
        "full": dict(iterations=400, surface_level=4, near_level=3, wake_level=2)}        # full: the notebook's
DISK_ITERATIONS = {"smoke": 60, "quick": 300, "full": 600}                                 # full: the notebook's
MAP_RPM, MAP_V = np.linspace(3000, 10000, 8), np.linspace(0, 24, 7)                        # 09a cell 58


def build(fidelity: str = "full", aircraft: dict | None = None) -> Assembly:
    p = FixedWing().resolve(**dict(aircraft or {}, part="aircraft", angle_of_attack_deg=0.0))
    spec = wing.WingSpec.from_params(p)
    root = Assembly(NAME, "aircraft", params={"fidelity": fidelity, "parts_g": PARTS_G, "drive": DRIVE, "v_cruise": V_CRUISE})
    root.add(Assembly("airframe", "fixed_wing", params={"aircraft": p, "shell_g_mm2": SHELL_AREAL_DENSITY_G_MM2,
                                                        "nacelle_g_mm3": NACELLE_DENSITY_G_MM3}))
    root.add(Assembly("propeller", "propeller", params={"spec": pr.get(DRIVE["propeller"]).to_dict()}))
    return root


def _point(v) -> dict:
    return {"throttle": float(v.throttle), "rpm": float(v.rpm), "thrust_N": float(v.thrust), "current_A": float(v.current),
            "electrical_W": float(v.electrical_power), "prop_eff": float(v.aero.efficiency),
            "motor_eff": float(v.motor_efficiency), "J": float(v.aero.advance_ratio), "current_limited": bool(v.current_limited)}


def drive(auw_kg: float, wing_area: float, aspect_ratio: float, cl: float, cd: float):
    """The polar through the CFD point and the four points of 09a cell 51."""
    spec = pr.get(DRIVE["propeller"])
    prop, sec = spec.model(), spec.airfoil()
    battery = pr.battery(DRIVE["battery"])
    system = boreas.Propulsion(prop, sec, pr.motor(DRIVE["motor"]), battery, rho=DRIVE["rho"])
    k = 1 / (math.pi * aspect_ratio * OSWALD)
    cd0 = cd - k * cl ** 2
    W, rho = auw_kg * G, DRIVE["rho"]
    q = 0.5 * rho * V_CRUISE ** 2
    cl_v = W / (q * wing_area)
    d_cruise = q * wing_area * (cd0 + k * cl_v ** 2)
    pts = {"cruise": system.for_thrust(d_cruise / DRIVE["motors"], V_CRUISE), "climb": system.at_throttle(1.0, 12.0),
           "static": system.at_throttle(1.0, 0.0), "engine_out": system.at_throttle(1.0, V_CRUISE)}
    p_cruise = DRIVE["motors"] * pts["cruise"].electrical_power
    endurance = battery.usable_wh / p_cruise * 60
    return prop, sec, battery, pts, {"cd0": cd0, "k_induced": k, "drag_cruise_N": d_cruise, "endurance_min": endurance,
                                     "range_km": V_CRUISE * endurance * 60 / 1000}


def run(*, fidelity: str = "full", aircraft: dict | None = None, run_cfd: bool = True, run_fea: bool = True,
        processors: int = 1, jobs: int = 1, threads: int = 1, out: Path | None = None, vida_path: Path | None = None,
        include: str = "results", export: bool = True, export_path: Path | None = None, force: bool = False, redo=(),
        progress: bool = True) -> Assembly:
    out = Path(out or RUNS / NAME).resolve()
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    if force and out.exists():
        shutil.rmtree(out)
    prior = vida.load(vida_path) if vida_path.is_file() and not force else None
    root = build(fidelity, aircraft)
    root.reuse(prior)
    for path in redo:
        if path in ("airframe", "propeller"):
            root.child(path).forget()

    # the airframe: CAD (aircraft, wing, nacelle), mass budget (09a cells 7-10)
    af = root.child("airframe")
    p = af.params["aircraft"]
    fw = FixedWing()
    ac_files, ac = export_kept(lambda: fw.generate(**p), p, out / "aircraft", "aircraft", formats=("step",))
    wp = dict(p, part="wing")
    wing_files, _ = export_kept(lambda: fw.generate(**wp), wp, out / "wing", "wing", formats=("step",))
    npar = dict(p, part="nacelle")
    _, nac = export_kept(lambda: fw.generate(**npar), npar, out / "nacelle", "nacelle", formats=("step",))
    af.attach("aircraft.step", ac_files["step"], "geometry")
    af.attach("wing.step", wing_files["step"], "geometry")
    spec = wing.WingSpec.from_params(p)
    pl = spec.planform()
    if not af.results:
        airframe_g = ((ac["surface_area_mm2"] - 2 * nac["surface_area_mm2"]) * SHELL_AREAL_DENSITY_G_MM2
                      + 2 * nac["volume_mm3"] * NACELLE_DENSITY_G_MM3)
        af.record(planform=pl, airframe_g=airframe_g, auw_kg=(sum(PARTS_G.values()) + airframe_g) / 1000)
    auw = af.results["auw_kg"]
    W = auw * G

    # the wing in FEA (09a cell 12)
    wf = add_after(root, Assembly("wing_fea", "wing_fea", params={
        "aircraft": p, "weight_N": W, "wing_area_m2": pl["area_m2"], "element_mm": WING_ELEMENT_MM[fidelity],
        "material": wing.LW_PLA, "cases": {"pull_up": {"n": 2.5, "motor_fx_N": [6.0, 6.0]},
                                           "engine_out": {"n": 1.0, "motor_fx_N": [8.0, 0.0]}}}), prior, redo)
    if not wf.results.get("complete"):
        step = wing_files["step"]
        regions = [wing.root_region(p["fuselage_diameter"]), talos.Surfaces("lift", wing.lower_skins(step)),
                   *wing.motor_regions(p["nacelle_y"], p["nacelle_diameter"], p["nacelle_forward"])]
        models = {}
        for name, c in wf.params["cases"].items():
            loads = [talos.Pressure("lift", wing.lift_pressure_MPa(c["n"], W, pl["area_m2"]))]
            loads += [talos.Force(side, fx=f) for side, f in zip(("motor_left", "motor_right"), c["motor_fx_N"]) if f]
            models[name] = wing.wing_model(step, regions, loads, element_mm=wf.params["element_mm"], name=name)
        solve_fea(wf, models, out / "wing_fea", run=run_fea, threads=threads, progress=progress)

    # the whole aircraft at 4 deg (09a cell 24)
    p4 = dict(p, angle_of_attack_deg=AOA_DEG)
    cfd_values = dict(velocity=V_CRUISE, kinematic_viscosity=1.5e-5, density=DRIVE["rho"], reference_area=pl["area_m2"],
                      reference_length=0.25, center_of_rotation=(0.05, 0.0, 0.0), residual_target=1e-4, cells_per_length=2.0,
                      **AERO[fidelity])
    ae = add_after(root, Assembly("aero_cfd", "aircraft_4deg", params={"aircraft": p4, "cfd": cfd_values}), prior, redo)
    stl4, _ = export_kept(lambda: fw.generate(**p4), p4, out / "aircraft_4deg", "aircraft_4deg", formats=("stl",),
                          stl_tolerance=0.2)
    ae.attach("aircraft_4deg.stl", stl4["stl"], "geometry")
    if not ae.results.get("complete"):
        case = aeromant.CFDCase("rans_ksst_external", stl4["stl"], cfd_values, workdir=out / "aero_cfd" / "cruise_4deg",
                                geometry_units="mm", environment=environment(run_cfd))
        cfd_node(ae, case, ("Cl", "Cd", "lift_force_N", "drag_force_N", "converged", "mesh_cells"), run=run_cfd,
                processors=processors, progress=progress)
    forces = ae.results.get("forces") or {}
    if forces.get("Cl") is not None:
        polar_point, polar_source = {"Cl": forces["Cl"], "Cd": forces["Cd"]}, "aero_cfd"
    else:
        polar_point, polar_source = dict(RECORDED_POLAR), "recorded in notebook 09a (CFD NOT RUN here)"

    # the drive (09a Part 2), the propeller CAD, rotor CFD at cruise, one blade at the climb
    prop, sec, battery, pts, perf = drive(auw, pl["area_m2"], pl["aspect_ratio"], polar_point["Cl"], polar_point["Cd"])
    ps = pr.get(DRIVE["propeller"])
    pn = root.child("propeller")
    pfiles, bfiles = pr.cad_files(ps, out / "propeller"), pr.cad_files(ps, out / "blade", blades=1)
    pn.attach("propeller.step", pfiles["step"], "geometry")
    if not pn.results:
        pn.record(describe=prop.describe())
    rc = add_after(root, Assembly("rotor_cfd", "rotor", params=pr.rotor_params(ps, pts["cruise"].rpm, airspeed=V_CRUISE,
                                                                               fidelity=fidelity) | {"density": DRIVE["rho"]}),
                   prior, redo)
    if not rc.results.get("complete"):
        case = pr.rotor_case(pfiles["stl_axis_x"], rc.params, out / "rotor_cfd" / "cruise", environment(run_cfd))
        cfd_node(rc, case, ("thrust_N", "torque_Nm", "power_W", "efficiency", "converged", "mesh_cells"), run=run_cfd,
                processors=processors, progress=progress)
    loads = pr.blade_loads(prop, pts["climb"])
    bl = add_after(root, Assembly("blade_fea", "blade", params={"spec": ps.to_dict(), "loads": loads,
                                                                "element_mm": BLADE_ELEMENT_MM[fidelity],
                                                                "material": pr.BLADE_MATERIAL}), prior, redo)
    if not bl.results.get("complete"):
        model = pr.blade_model(ps, bfiles["step"], loads, element_mm=bl.params["element_mm"], name="blade_full_throttle")
        solve_fea(bl, {"blade_full_throttle": model}, out / "blade_fea", run=run_fea, threads=threads, progress=progress)

    # the aircraft with both propellers as rotor disks (09a Part 3, cells 72-73)
    th = math.radians(AOA_DEG)

    def to_flow(x, y, z):
        return np.array([x * math.cos(th) + z * math.sin(th), y, -x * math.sin(th) + z * math.cos(th)])

    x_prop = -p["nacelle_forward"] - 12.0
    disk = dict(cfd_values, iterations=DISK_ITERATIONS[fidelity], surface_level=4, near_level=3, wake_level=2,
                disk1_center=(to_flow(x_prop, -p["nacelle_y"], 0.0) / 1000).tolist(),
                disk2_center=(to_flow(x_prop, p["nacelle_y"], 0.0) / 1000).tolist(), disk_axis=to_flow(-1.0, 0.0, 0.0).tolist(),
                diameter=prop.diameter, rpm=float(pts["cruise"].rpm), blades=prop.blades, blade=pr.blade_table(prop),
                polar=pr.polar_table(sec), rotation1=1, rotation2=-1, disk_level=5)
    ic = add_after(root, Assembly("installed_cfd", "rotor_disks", params={"aircraft": p4, "cfd": disk}), prior, redo)
    if not ic.results.get("complete"):
        case = aeromant.CFDCase("aircraft_rotor_disks", stl4["stl"], disk, workdir=out / "installed_cfd" / "cruise",
                                geometry_units="mm", environment=environment(run_cfd))
        cfd_node(ic, case, ("Cl", "Cd", "lift_force_N", "drag_force_N", "converged", "mesh_cells"), run=run_cfd,
                processors=processors, progress=progress)

    root.record(auw_kg=auw, polar_point=polar_point, polar_source=polar_source, polar=perf,
                points={k: _point(v) for k, v in pts.items()})
    if polar_source != "aero_cfd":
        root.not_run("polar through the whole-aircraft CFD: notebook 09a's recorded point used")
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:
        path = Path(export_path or DATA / f"{NAME}.json")
        handoff = {"source": f"assemblies.workflows.{NAME} ({fidelity})", "auw_kg": auw, "wing_area_m2": pl["area_m2"],
                   "v_cruise_m_s": V_CRUISE, "rho": DRIVE["rho"], "endurance_min": perf["endurance_min"],
                   "preferred_wing": {"revision": None, "thickness": p["thickness"]}, "parts_g": dict(PARTS_G),
                   "points": {k: {"rpm": float(pts[k].rpm), "thrust_N": float(pts[k].thrust)} for k in ("cruise", "climb", "engine_out")},
                   "polar_point": polar_point, "polar_source": polar_source,
                   "git": vida.manifest(vida_path)["git"], "written_at": vida.utc_now()}
        path.write_text(__import__("json").dumps(handoff, indent=2, default=str))
        prop_path = path.with_name(f"{path.stem}_propulsion.json")
        grid = boreas.performance_map(prop, sec, MAP_RPM, MAP_V, DRIVE["rho"])
        boreas.export(prop_path, prop, sec, map=grid, motor=pr.motor(DRIVE["motor"]), battery=battery,
                      points={k: pts[k] for k in ("cruise", "climb", "static", "engine_out")},
                      notes=f"fixed wing from assemblies.workflows.{NAME}; AUW {auw:.3f} kg").raise_for_status()
        root.meta["exported_to"] = f"{path}, {prop_path}"
    return root


def _parser():
    ap = parser("The fixed wing: airframe, wing FEA, aircraft CFD, 9x6 drive, rotor CFD, blade FEA, rotor-disk CFD")
    ap.add_argument("--aircraft", action="append", default=[], metavar="NAME=VALUE",
                    help="an aircraft parameter other than the default (repeatable), e.g. --aircraft thickness=0.15")
    return ap


def _run_cli(*, aircraft=(), **kw):
    return run(aircraft=parse_assignments(aircraft), **kw)


if __name__ == "__main__":
    raise SystemExit(main(_run_cli, _parser(), DATA / f"{NAME}.vida"))
