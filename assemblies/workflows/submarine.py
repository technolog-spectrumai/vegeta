"""The submarine: a 1.2 m AUV with a pressure hull, sail, cruciform fins and a 120 mm propeller (notebook 13).

The tree::

    submarine (submersible)        the drive: top speed, cruise, endurance and range vs speed with the hotel load
      hull (submarine)             vehicle, body and pressure-hull CAD; buoyancy, mass budget, the trim lead, BG
      hull_cfd (vehicle_cfd)       the vehicle nose-upstream at cruise (Aeromant rans_ksst_external)
      pressure_hull_fea (pressure_hull)  the pressure hull at the rated depth (Talos)
      propeller (propeller)        the 120 mm propeller's CAD
      rotor_cfd (rotor)            the propeller at cruise in its wake (Aeromant rotor_mrf)
      blade_fea (blade)            one aluminium blade at top speed
      scene_cfd (rotor_disk)       the vehicle with its propeller as a rotor disk (scenarios/run_scenario.py sub(), with the
                                   wetted surface, buoyancy centre and rpm of this tree instead of its recorded ones)

Diving, depth cycling and the propeller study of notebook 14 stay in the notebooks.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from vegeta import aeromant, boreas, talos

from .. import DATA, RUNS, vida
from .._cli import main, parser
from ..components import hull as H, propeller as pr
from ..components._cad import export_kept
from ..components.impeller import environment
from ..components.submarine import Submarine
from ..vida import Assembly
from ._common import add_after, run_cfd as cfd_node, solve_fea

NAME = "submarine"
RHO_W, NU_W, G = 1025.0, 1.05e-6, 9.81
DESIGN_DEPTH_M, V_CRUISE, X_PH, RESERVE = 200.0, 1.5, 420.0, 0.005                      # 13 cells 2, 4, 6
RHO_AL, RHO_GFRP, FAIRING_T = 2.70e-6, 1.8e-6, 2.0
ITEMS = [("battery 7S Li-ion 20 Ah", 6.0, 560.0, -30.0), ("electronics, INS, acoustic modem", 1.5, 780.0, 0.0),
         ("sonar payload (nose, free flooded)", 3.5, 1060.0, -15.0), ("thruster, propeller, stern gear", 1.5, 80.0, 0.0)]
DRIVE = {"propeller": "120 mm 3-blade", "motor": "thruster 100KV", "battery": "7S Li-ion 20 Ah", "wake": 0.85,
         "thrust_deduction": 0.10, "hotel_W": 25.0}                                              # 13 cell 13
AL6061 = dict(name="6061-T6", youngs_modulus=69000.0, poissons_ratio=0.33, density=2.7e-9, yield_strength=240.0,
              source="handbook, machined")
ELEMENT = {"pressure_hull": {"smoke": 12.0, "quick": 9.0, "full": 6.0}, "blade": {"smoke": 4.0, "quick": 3.0, "full": 2.0}}
HULL_CFD = {"smoke": dict(iterations=60, surface_level=3, near_level=2, wake_level=1, cells_per_length=3.0),
            "quick": dict(iterations=400, surface_level=3, near_level=2, wake_level=1, cells_per_length=3.0),
            "full": dict(iterations=400, surface_level=4, near_level=3, wake_level=2, cells_per_length=3.0)}
SCENE_ITERATIONS = {"smoke": 60, "quick": 300, "full": 600}


def build(fidelity: str = "full", sub: dict | None = None) -> Assembly:
    p = Submarine().resolve(**dict(sub or {}, part="vehicle"))
    root = Assembly(NAME, "submersible", params={"fidelity": fidelity, "drive": DRIVE, "v_cruise": V_CRUISE,
                                                 "design_depth_m": DESIGN_DEPTH_M})
    root.add(Assembly("hull", "submarine", params={"sub": p, "items": [list(i) for i in ITEMS], "reserve": RESERVE, "x_ph": X_PH}))
    root.add(Assembly("propeller", "propeller", params={"spec": pr.get(DRIVE["propeller"]).to_dict()}))
    return root


def mass_and_trim(mv: dict, mh: dict) -> dict:
    """Buoyancy, the mass budget with the computed trim lead, CG, BG (13 cell 6)."""
    buoyancy_kg = mv["volume_mm3"] * 1e-9 * RHO_W
    CB = np.array(mv["center_of_mass_mm"])
    items = [("pressure hull (6061-T6, from CAD)", mh["volume_mm3"] * RHO_AL, X_PH + mh["center_of_mass_mm"][0], 0.0),
             ("fairing, sail and fins (GFRP 2 mm)", mv["surface_area_mm2"] * FAIRING_T * RHO_GFRP, CB[0], CB[2]), *ITEMS]
    m = np.array([i[1] for i in items])
    M_target = buoyancy_kg * (1 - RESERVE)
    m_lead = M_target - m.sum()
    if m_lead < 0:
        raise ValueError(f"the vehicle is {-m_lead:.2f} kg too heavy for its volume: enlarge the hull or cut the budget")
    x_lead = (CB[0] * M_target - (m * [i[2] for i in items]).sum()) / m_lead
    items.append(("trim lead (computed)", m_lead, x_lead, -60.0))
    m = np.array([i[1] for i in items])
    M = m.sum()
    CG = np.array([(m * [i[2] for i in items]).sum() / M, 0.0, (m * [i[3] for i in items]).sum() / M])
    BG = CB[2] - CG[2]
    return {"buoyancy_kg": buoyancy_kg, "mass_kg": M, "net_buoyancy_N": (buoyancy_kg - M) * G, "CB_mm": CB, "CG_mm": CG,
            "BG_mm": BG, "righting_moment_at_90deg_Nm": M * G * BG / 1000, "trim_lead_kg": m_lead, "trim_lead_x_mm": x_lead,
            "items_kg": {i[0]: i[1] for i in items}}


def drive(L_m: float, D_m: float, S_wet: float, S_body: float):
    """Top speed, cruise, endurance and range (13 cell 13)."""
    spec = pr.get(DRIVE["propeller"])
    prop, sec = spec.model(), spec.airfoil()
    battery = pr.battery(DRIVE["battery"])
    d = boreas.Propulsion(prop, sec, pr.motor(DRIVE["motor"]), battery, rho=RHO_W)
    WAKE, T_DED, HOTEL_W = DRIVE["wake"], DRIVE["thrust_deduction"], DRIVE["hotel_W"]

    def resistance(V):
        return H.body_of_revolution_resistance(V, L_m, D_m, S_wet, S_body)

    def point_at(V):
        return d.for_thrust(resistance(V)[0] / (1 - T_DED), WAKE * V)

    lo, hi = 0.3, 5.0
    for _ in range(24):
        Vm = 0.5 * (lo + hi)
        lo, hi = (Vm, hi) if d.at_throttle(1.0, WAKE * Vm).thrust * (1 - T_DED) > resistance(Vm)[0] else (lo, Vm)
    V_MAX = lo
    cruise, full = point_at(V_CRUISE), d.at_throttle(1.0, WAKE * V_MAX)
    speeds = np.linspace(0.5, V_MAX, 14)
    pts = [point_at(v) for v in speeds]
    endurance_h = np.array([battery.usable_wh / (pt.electrical_power + HOTEL_W) for pt in pts])
    return prop, sec, battery, {"cruise": cruise, "full": full}, {
        "top_speed_m_s": V_MAX, "resistance_at_cruise_N": resistance(V_CRUISE)[0], "speeds_m_s": speeds,
        "endurance_h": endurance_h, "range_km": endurance_h * speeds * 3.6}


def scene_params(S_wet: float, cb_x_m: float, L_m: float, rpm: float, prop, section, iterations: int = 600) -> dict:
    """run_scenario.sub(): the propeller as a rotor disk 40 mm behind the tail tip, the vehicle nose-upstream."""
    return dict(velocity=V_CRUISE, kinematic_viscosity=NU_W, density=RHO_W, reference_area=S_wet, reference_length=L_m / 4,
                center_of_rotation=(-cb_x_m, 0.0, 0.0), iterations=iterations, residual_target=1e-4,
                cells_per_length=3.0, surface_level=4, near_level=3, wake_level=2,
                disk1_center=[0.04, 0.0, 0.0], disk_axis=[-1.0, 0.0, 0.0],
                diameter=prop.diameter, rpm=float(rpm), blades=prop.blades,
                blade=pr.blade_table(prop), polar=pr.polar_table(section), rotation1=1, disk_level=5)


def _point(v) -> dict:
    return {"rpm": float(v.rpm), "thrust_N": float(v.thrust), "current_A": float(v.current), "electrical_W": float(v.electrical_power),
            "prop_eff": float(v.aero.efficiency), "motor_eff": float(v.motor_efficiency), "current_limited": bool(v.current_limited)}


def run(*, fidelity: str = "full", sub: dict | None = None, run_cfd: bool = True, run_fea: bool = True, processors: int = 1,
        jobs: int = 1, threads: int = 1, out: Path | None = None, vida_path: Path | None = None, include: str = "results",
        export: bool = True, export_path: Path | None = None, force: bool = False, redo=(), progress: bool = True) -> Assembly:
    out = Path(out or RUNS / NAME).resolve()
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    if force and out.exists():
        shutil.rmtree(out)
    prior = vida.load(vida_path) if vida_path.is_file() and not force else None
    root = build(fidelity, sub)
    root.reuse(prior)
    for path in redo:
        if path in ("hull", "propeller"):
            root.child(path).forget()

    hn = root.child("hull")
    p = hn.params["sub"]
    sd = Submarine()
    files, info = {}, {}
    for part, fmts, tol in (("vehicle", ("step", "stl"), 0.2), ("body", ("step",), 0.2), ("pressure_hull", ("step",), 0.2)):
        q = dict(p, part=part)
        files[part], info[part] = export_kept(lambda q=q: sd.generate(**q), q, out / "cad" / part, part, formats=fmts, stl_tolerance=tol)
        hn.attach(f"{part}.step", files[part]["step"], "geometry")
    if not hn.results:
        hn.record(vehicle_volume_L=info["vehicle"]["volume_mm3"] * 1e-6, wetted_surface_m2=info["vehicle"]["surface_area_mm2"] * 1e-6,
                  body_surface_m2=info["body"]["surface_area_mm2"] * 1e-6, **mass_and_trim(info["vehicle"], info["pressure_hull"]))
    L_M, D_M = p["length"] / 1000, p["diameter"] / 1000
    S_WET, S_BODY, CB = hn.results["wetted_surface_m2"], hn.results["body_surface_m2"], np.asarray(hn.results["CB_mm"])

    # the vehicle nose-upstream in CFD (13 cell 10)
    hull_cfd_values = dict(velocity=V_CRUISE, kinematic_viscosity=NU_W, density=RHO_W, reference_area=S_WET, reference_length=L_M / 4,
                           center_of_rotation=(-CB[0] / 1000, 0.0, 0.0), residual_target=1e-4, **HULL_CFD[fidelity])
    hc = add_after(root, Assembly("hull_cfd", "vehicle_cfd", params={"sub": p, "cfd": hull_cfd_values}), prior, redo)
    nose = None
    if not hc.results.get("complete"):
        nose, _ = export_kept(lambda: H.nose_upstream(sd.generate(**dict(p, part="vehicle"))), {"sub": p},
                              out / "cad" / "nose_upstream", "vehicle_nose_upstream", formats=("stl",), stl_tolerance=0.2)
        hc.attach("vehicle_nose_upstream.stl", nose["stl"], "geometry")
        case = aeromant.CFDCase("rans_ksst_external", nose["stl"], hull_cfd_values, workdir=out / "hull_cfd" / "cruise",
                                geometry_units="mm", environment=environment(run_cfd))
        cfd_node(hc, case, ("drag_force_N", "Cd", "converged", "mesh_cells"), run=run_cfd, processors=processors, progress=progress)

    # the pressure hull at the rated depth (13 cell 15)
    P_DESIGN = RHO_W * G * DESIGN_DEPTH_M / 1e6
    pf = add_after(root, Assembly("pressure_hull_fea", "pressure_hull", params={"sub": p, "material": AL6061, "pressure_MPa": P_DESIGN,
                                                                                "element_mm": ELEMENT["pressure_hull"][fidelity]}),
                   prior, redo)
    if not pf.results.get("complete"):
        step = files["pressure_hull"]["step"]
        tinfo = talos.inspect_step(step, "mm-N-MPa")
        R_I, T_SH, D_CAP, L_PH = p["diameter"] / 2 - 12.0, p["shell"], p["end_cap_depth"], p["pressure_hull_length"]
        x_flat = D_CAP * math.cos(math.asin(0.12))
        wet = [s.tag for s in tinfo.surfaces if s.kind != "Plane" and s.bbox_max[1] > R_I - T_SH / 2]
        regions = [talos.Surfaces("wet", wet), talos.SurfacesOnPlane("stern_plate", "x", -x_flat, tol=0.05),
                   talos.SurfacesOnPlane("bow_plate", "x", L_PH + x_flat, tol=0.05)]
        model = talos.StructuralModel(step, "mm-N-MPa", talos.Material(**AL6061), regions, [talos.FixedSupport("stern_plate")],
                                      [talos.Pressure("wet", P_DESIGN)], talos.MeshSettings(element_size=pf.params["element_mm"]),
                                      name="design_depth")
        solve_fea(pf, {"design_depth": model}, out / "pressure_hull_fea", run=run_fea, threads=threads, progress=progress)

    # the drive, the propeller, its CFD at cruise and one blade at top speed (13 cells 13, 18, 25)
    prop, sec, battery, pts, perf = drive(L_M, D_M, S_WET, S_BODY)
    ps = pr.get(DRIVE["propeller"])
    pn = root.child("propeller")
    pfiles, bfiles = pr.cad_files(ps, out / "propeller"), pr.cad_files(ps, out / "blade", blades=1)
    pn.attach("propeller.step", pfiles["step"], "geometry")
    if not pn.results:
        pn.record(describe=prop.describe())
    rc = add_after(root, Assembly("rotor_cfd", "rotor", params=pr.rotor_params(ps, pts["cruise"].rpm, airspeed=DRIVE["wake"] * V_CRUISE,
                                                                               medium="sea water", fidelity=fidelity)), prior, redo)
    if not rc.results.get("complete"):
        case = pr.rotor_case(pfiles["stl_axis_x"], rc.params, out / "rotor_cfd" / "cruise", environment(run_cfd))
        cfd_node(rc, case, ("thrust_N", "torque_Nm", "power_W", "efficiency", "converged", "mesh_cells"), run=run_cfd,
                 processors=processors, progress=progress)
    loads = pr.blade_loads(prop, pts["full"])
    bl = add_after(root, Assembly("blade_fea", "blade", params={"spec": ps.to_dict(), "loads": loads, "material": AL6061,
                                                                "element_mm": ELEMENT["blade"][fidelity]}), prior, redo)
    if not bl.results.get("complete"):
        model = pr.blade_model(ps, bfiles["step"], loads, element_mm=bl.params["element_mm"], material=AL6061,
                               name="blade_top_speed", root_gap_mm=2.0)
        solve_fea(bl, {"blade_top_speed": model}, out / "blade_fea", run=run_fea, threads=threads, progress=progress)

    # the vehicle with its propeller as a rotor disk (run_scenario sub(), this tree's numbers)
    sp = scene_params(S_WET, CB[0] / 1000, L_M, pts["cruise"].rpm, prop, sec, SCENE_ITERATIONS[fidelity])
    sc = add_after(root, Assembly("scene_cfd", "rotor_disk", params={"sub": p, "cfd": sp}), prior, redo)
    if not sc.results.get("complete"):
        if nose is None:
            nose, _ = export_kept(lambda: H.nose_upstream(sd.generate(**dict(p, part="vehicle"))), {"sub": p},
                                  out / "cad" / "nose_upstream", "vehicle_nose_upstream", formats=("stl",), stl_tolerance=0.2)
        case = aeromant.CFDCase("hull_rotor_disk", nose["stl"], sp, workdir=out / "scene_cfd" / "cruise", geometry_units="mm",
                                environment=environment(run_cfd))
        cfd_node(sc, case, ("drag_force_N", "converged", "mesh_cells"), run=run_cfd, processors=processors, progress=progress)

    root.record(points={k: _point(v) for k, v in pts.items()}, performance=perf)
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:
        path = Path(export_path or DATA / f"{NAME}.json")
        boreas.export(path, prop, sec, motor=pr.motor(DRIVE["motor"]), battery=battery, points=pts,
                      notes=f"submarine from assemblies.workflows.{NAME} ({fidelity}); wake {DRIVE['wake']}").raise_for_status()
        root.meta["exported_to"] = str(path)
    return root


def _parser():
    return parser("The submarine: hull, trim, CFD, pressure-hull FEA, drive, rotor CFD, blade FEA, scene")


if __name__ == "__main__":
    raise SystemExit(main(run, _parser(), DATA / f"{NAME}.vida"))
