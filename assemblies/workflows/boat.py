"""The survey boat: a 1 m printed hull with a motor pod on a transom bracket (notebook 12, Parts 1 and 2).

The tree::

    boat (surface_vessel)          the drive in water: top speed, cruise, endurance, bollard pull
      hull (survey_boat)           hull, deck and bracket CAD; mass budget; draft, wetted surface, GM, the GZ curve
      hull_cfd (double_body)       the hull below the waterline mirrored about it at 2 m/s (no free surface)
      hull_fea (hull_bottom)       the bottom panels under 1 kPa, the deck edge held (Talos)
      bracket_fea (bracket)        the transom bracket under 10 N of thrust and 10 N sideways on the pod
      propeller (propeller)        the 60 mm marine propeller's CAD
      rotor_cfd (rotor)            the propeller at cruise in sea water (Aeromant rotor_mrf)
      blade_fea (blade)            one blade at bollard pull
      scene_cfd (rotor_disks)      the boat with its propeller as a rotor disk (scenarios/run_scenario.py boat(), with the
                                   draft and rpm of this tree instead of its recorded ones)

The sea states, fatigue and noise sections of notebook 12 stay in the notebook.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from vegeta import aeromant, boreas, talos

from .. import DATA, RUNS, vida, results
from .._cli import main, parser
from ..components import hull as H, propeller as pr
from ..components._cad import export_kept
from ..components.impeller import environment
from ..components.survey_boat import SurveyBoat
from ..vida import Assembly
from ._common import add_after, run_cfd as cfd_node, solve_fea

NAME = "boat"
RHO_W, NU_W, G = 1025.0, 1.05e-6, 9.81
RHO_PETG = 1.27e-3
ITEMS = [("motor pod + propeller", 300.0, -60.0, -25.0), ("battery 4S 10 Ah", 900.0, 380.0, 25.0),
         ("electronics, GPS, radio", 300.0, 500.0, 120.0), ("survey payload (sonar)", 500.0, 350.0, 15.0),
         ("hatches, fasteners, cabling", 250.0, 450.0, 100.0)]                                  # 12 cell 7
DRIVE = {"propeller": "60 mm 3-blade marine", "motor": "2836-500KV (water-cooled pod)", "battery": "4S 10 Ah", "wake": 0.9}
V_CRUISE, V_CFD, FORM_FACTOR = 1.5, 2.0, 1.25
PETG = dict(name="PETG (printed hull)", youngs_modulus=2000.0, poissons_ratio=0.38, density=1.27e-9, yield_strength=45.0,
            source="nominal, XY orientation")
PETG_CF = dict(name="PETG-CF (bracket)", youngs_modulus=4800.0, poissons_ratio=0.38, density=1.25e-9, yield_strength=45.0,
               source="nominal")
PA12CF = dict(name="PA12-CF (printed)", youngs_modulus=3500.0, poissons_ratio=0.40, density=1.1e-9, yield_strength=60.0,
              source="nominal, flat orientation")
ELEMENT = {"hull": {"smoke": 16.0, "quick": 12.0, "full": 8.0}, "bracket": {"smoke": 8.0, "quick": 5.0, "full": 3.0},
           "blade": {"smoke": 2.5, "quick": 1.5, "full": 1.0}}                                  # full: the notebook's
HULL_CFD = {"smoke": dict(iterations=60, surface_level=3, near_level=2, wake_level=1),
            "quick": dict(iterations=250, surface_level=3, near_level=2, wake_level=1),
            "full": dict(iterations=250, surface_level=4, near_level=3, wake_level=2)}
SCENE_ITERATIONS = {"smoke": 60, "quick": 300, "full": 600}


# ----------------------------------------------------------------------------------------------- hydrostatics (12 cells 9-10)
def hydrostatics(hull_solid, p: dict, mass_kg: float, cg: np.ndarray) -> dict:
    """Draft, displacement, wetted surface, KB, BM, GM and the GZ curve from the hull's mesh (notebook 12, as it was)."""
    from vegeta.dedalus import viz as dviz

    hull_mesh = dviz.to_pyvista(hull_solid, 0.3).triangulate().clean()

    def immersed(draft_mm, heel_deg=0.0):
        m = hull_mesh.rotate_x(heel_deg, point=(0, 0, 0), inplace=False) if heel_deg else hull_mesh
        cl = m.clip_closed_surface(normal=(0, 0, -1), origin=(0, 0, draft_mm))
        return (cl.volume * 1e-9 * RHO_W, np.array(cl.center_of_mass()), cl) if cl.n_points else (0.0, np.zeros(3), cl)

    def draft_for(m_kg, heel_deg=0.0):
        lo, hi = 0.0, p["depth"]
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if immersed(mid, heel_deg)[0] < m_kg else (lo, mid)
        return hi

    T = draft_for(mass_kg)
    disp, CB, clipped = immersed(T)
    wl = hull_mesh.slice(normal=(0, 0, 1), origin=(0, 0, T))
    wp_pts = wl.points[:, :2]

    def polygon_area(pts):
        c = pts.mean(axis=0); ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]); q = pts[np.argsort(ang)]
        return 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))

    A_wp = polygon_area(wp_pts)
    S_wet = (clipped.area - A_wp) * 1e-6
    I_wp = np.sum((wp_pts[:, 1] - wp_pts[:, 1].mean()) ** 2) / len(wp_pts) * A_wp
    BM = I_wp / (disp / RHO_W * 1e9); KB = CB[2]; GM_est = KB + BM - cg[2]

    heels = np.arange(0, 61, 5)

    def righting_arm(h):
        Th = draft_for(mass_kg, h)
        dh, cbh, _ = immersed(Th, h)
        r = math.radians(h)
        cg_r = np.array([cg[0], cg[1] * math.cos(r) - cg[2] * math.sin(r), cg[1] * math.sin(r) + cg[2] * math.cos(r)])
        deck_edge_z = min(-p["beam"] / 2 * math.sin(r), p["beam"] / 2 * math.sin(r)) + p["depth"] * math.cos(r)
        return cg_r[1] - cbh[1], deck_edge_z < Th

    gz, dfl = [], None
    for h in heels:
        a, wet = righting_arm(h)
        b, _ = righting_arm(-h)
        gz.append(0.5 * (a - b))
        if dfl is None and wet:
            dfl = int(h)
    gz = np.array(gz)
    GM = (gz[1] - gz[0]) / math.radians(heels[1] - heels[0])
    return {"draft_mm": float(T), "displacement_kg": float(disp), "freeboard_mm": float(p["depth"] - T), "LCB_mm": float(CB[0]),
            "LCG_mm": float(cg[0]), "waterplane_area_m2": float(A_wp * 1e-6), "wetted_surface_m2": float(S_wet), "KB_mm": float(KB),
            "BM_mm_rough": float(BM), "GM_mm_estimate": float(GM_est), "GM_mm": float(GM), "heel_deg": heels.astype(float),
            "GZ_mm": gz, "deck_edge_immersion_deg": dfl}


# ----------------------------------------------------------------------------------------------- the drive (12 cells 15, 35)
def drive(S_wet: float, length_mm: float):
    spec = pr.get(DRIVE["propeller"])
    prop, sec = spec.model(), spec.airfoil()
    battery = pr.battery(DRIVE["battery"])
    d = boreas.Propulsion(prop, sec, pr.motor(DRIVE["motor"]), battery, rho=RHO_W)
    L, w = length_mm / 1000, DRIVE["wake"]

    def resistance(V):
        return H.boat_resistance(V, L, S_wet, form_factor=FORM_FACTOR)

    Vs = np.linspace(0.3, 5.0, 48)
    R = np.array([resistance(v)[0] for v in Vs])
    T_full = np.array([d.at_throttle(1.0, w * v).thrust for v in Vs])
    ok = np.where(T_full > R)[0]
    V_MAX = float(Vs[ok.max()]) if len(ok) else float("nan")
    cruise = d.for_thrust(resistance(V_CRUISE)[0], w * V_CRUISE)
    full = d.at_throttle(1.0, w * V_MAX)
    bollard = d.at_throttle(1.0, 0.0)
    return prop, sec, battery, {"cruise": cruise, "full": full, "bollard": bollard}, {
        "top_speed_m_s": V_MAX, "resistance_limited": bool(len(ok) and ok.max() == len(Vs) - 1),
        "endurance_h": battery.usable_wh / cruise.electrical_power,
        "resistance_at_cruise_N": resistance(V_CRUISE)[0], "speeds_m_s": Vs, "resistance_N": R, "thrust_full_N": T_full}


def _point(v) -> dict:
    return {"rpm": float(v.rpm), "thrust_N": float(v.thrust), "torque_Nm": float(v.aero.torque), "current_A": float(v.current),
            "electrical_W": float(v.electrical_power), "prop_eff": float(v.aero.efficiency), "current_limited": bool(v.current_limited)}


# ----------------------------------------------------------------------------------------------- the scene (run_scenario boat())
def scene_geometry(p: dict, draft_mm: float):
    """The hull and the motor pod below the waterline, mirrored about it, turned bow-upstream."""
    import cadquery as cq
    from vegeta import dedalus

    b = SurveyBoat()
    hull = cq.Workplane("XY").add(b.generate(**dict(p, part="hull_solid")).shape)
    pod = cq.Workplane("XY").add(b.generate(**dict(p, part="bracket")).shape.translate((0, 0, p["depth"] * 0.55)))
    big = 4 * max(p["length"], p["beam"])
    below = cq.Workplane("XY").box(big, big, big, centered=(True, True, False)).translate((0, 0, draft_mm - big))
    under = hull.union(pod).intersect(below)
    body = under.union(under.mirror("XY", basePointVector=(0, 0, draft_mm)))
    return dedalus.Geometry.from_cadquery(body.val().rotate((0, 0, 0), (0, 0, 1), 180), name="boat_double_body")


def scene_params(p: dict, draft_mm: float, rpm: float, prop, section, iterations: int = 600) -> dict:
    z_prop = 0.55 * p["depth"] - p["bracket_height"]
    x_prop = 0.108
    return dict(velocity=1.5, kinematic_viscosity=NU_W, density=RHO_W, reference_area=2 * 0.231, reference_length=0.25,
                center_of_rotation=(-0.5, 0.0, draft_mm / 1000), iterations=iterations, residual_target=1e-4,
                surface_level=4, near_level=3, wake_level=2, cells_per_length=2.0,
                disk1_center=[x_prop, 0.0, z_prop / 1000], disk2_center=[x_prop, 0.0, (2 * draft_mm - z_prop) / 1000],
                disk_axis=[-1.0, 0.0, 0.0], diameter=prop.diameter, rpm=float(rpm), blades=prop.blades,
                blade=pr.blade_table(prop), polar=pr.polar_table(section), rotation1=1, rotation2=-1, disk_level=6)


# ----------------------------------------------------------------------------------------------- the workflow
def build(fidelity: str = "full", boat: dict | None = None) -> Assembly:
    p = SurveyBoat().resolve(**dict(boat or {}, part="boat"))
    root = Assembly(NAME, "surface_vessel", params={"fidelity": fidelity, "drive": DRIVE, "v_cruise": V_CRUISE})
    root.add(Assembly("hull", "survey_boat", params={"boat": p, "items": [list(i) for i in ITEMS], "rho_petg": RHO_PETG}))
    root.add(Assembly("propeller", "propeller", params={"spec": pr.get(DRIVE["propeller"]).to_dict()}))
    return root


def run(*, fidelity: str = "full", boat: dict | None = None, run_cfd: bool = True, run_fea: bool = True, processors: int = 1,
        jobs: int = 1, threads: int = 1, out: Path | None = None, vida_path: Path | None = None, include: str = "results",
        export: bool = True, export_path: Path | None = None, force: bool = False, redo=(), progress: bool = True) -> Assembly:
    out = Path(out or RUNS / NAME).resolve()
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    if force and out.exists():
        shutil.rmtree(out)
    prior = vida.load(vida_path) if vida_path.is_file() and not force else None
    root = build(fidelity, boat)
    root.reuse(prior)
    for path in redo:
        if path in ("hull", "propeller"):
            root.child(path).forget()

    hn = root.child("hull")
    p = hn.params["boat"]
    sb = SurveyBoat()
    files, info = {}, {}
    for part in ("boat", "hull_solid", "hull_shell", "bracket"):
        q = dict(p, part=part)
        files[part], info[part] = export_kept(lambda q=q: sb.generate(**q), q, out / "cad" / part, part, stl_tolerance=0.1)
        hn.attach(f"{part}.step", files[part]["step"], "geometry")
    if not hn.results:
        items = [("hull shell + deck (PETG, from CAD)", (info["boat"]["volume_mm3"] - info["bracket"]["volume_mm3"]) * RHO_PETG,
                  0.4 * p["length"], 70.0), ("transom bracket (PETG-CF, from CAD)", info["bracket"]["volume_mm3"] * 1.25e-3, -10.0, 60.0),
                 *[tuple(i) for i in hn.params["items"]]]
        m = np.array([i[1] for i in items])
        mass_kg = m.sum() / 1000
        cg = np.array([(m * [i[2] for i in items]).sum() / m.sum(), 0.0, (m * [i[3] for i in items]).sum() / m.sum()])
        hs = hydrostatics(sb.generate(**dict(p, part="hull_solid")), p, mass_kg, cg)
        hn.record(mass_kg=mass_kg, cg_mm=cg, items={i[0]: i[1] for i in items}, hydrostatics=hs)
    hs = hn.results["hydrostatics"]
    T, S_wet = hs["draft_mm"], hs["wetted_surface_m2"]

    # the double body in CFD (12 cell 13)
    hull_cfd_values = dict(velocity=V_CFD, kinematic_viscosity=NU_W, density=RHO_W, reference_area=2 * S_wet, reference_length=0.25,
                           center_of_rotation=(0.5, 0, T / 1000), residual_target=1e-4, **HULL_CFD[fidelity])
    hc = add_after(root, Assembly("hull_cfd", "double_body", params={"boat": p, "draft_mm": T, "cfd": hull_cfd_values}), prior, redo)
    if not hc.results.get("complete"):
        db, _ = export_kept(lambda: H.double_body(sb.generate(**dict(p, part="hull_solid")).shape, p, T), {"boat": p, "draft": T},
                            out / "cad" / "double_body", "double_body", formats=("stl",), stl_tolerance=0.3)
        hc.attach("double_body.stl", db["stl"], "geometry")
        case = aeromant.CFDCase("rans_ksst_external", db["stl"], hull_cfd_values, workdir=out / "hull_cfd" / "double_body",
                                geometry_units="mm", environment=environment(run_cfd))
        cfd_node(hc, case, ("drag_force_N", "Cd", "converged", "mesh_cells"), run=run_cfd, processors=processors, progress=progress)

    # the hull bottom and the bracket in FEA (12 cells 17-18)
    hf = add_after(root, Assembly("hull_fea", "hull_bottom", params={"boat": p, "material": PETG, "pressure_MPa": 0.001,
                                                                     "element_mm": ELEMENT["hull"][fidelity]}), prior, redo)
    if not hf.results.get("complete"):
        step = files["hull_shell"]["step"]
        tinfo = talos.inspect_step(step, units="mm-N-MPa")
        bottom = [s.tag for s in tinfo.surfaces if s.bbox_max[2] < p["depth"] * 0.45 and s.area > 500]
        regions = [talos.SurfacesInBox("deck_edge", (-1, -p["beam"], p["depth"] - 8, p["length"] + 1, p["beam"], p["depth"] + 1)),
                   talos.Surfaces("bottom", bottom)]
        model = talos.StructuralModel(step, "mm-N-MPa", talos.Material(**PETG), regions, [talos.FixedSupport("deck_edge")],
                                      [talos.Pressure("bottom", 0.001)], talos.MeshSettings(element_size=hf.params["element_mm"]),
                                      name="bottom_kpa")
        solve_fea(hf, {"bottom_kpa": model}, out / "hull_fea", run=run_fea, threads=threads, progress=progress)
    bf = add_after(root, Assembly("bracket_fea", "bracket", params={"boat": p, "material": PETG_CF, "force_N": 10.0,
                                                                    "element_mm": ELEMENT["bracket"][fidelity]}), prior, redo)
    if not bf.results.get("complete"):
        t_b, w_b, h_b, d_pod = p["bracket_thickness"], p["bracket_width"], p["bracket_height"], p["pod_diameter"]
        regions = [talos.SurfacesInBox("bolts", (-t_b - 1, -w_b / 2 + 3, -h_b + 15, 1, w_b / 2 - 3, 35)),
                   talos.SurfacesInBox("pod", (-t_b - 92, -d_pod / 2 - 1, -h_b - d_pod / 2 - 1, -t_b + 1, d_pod / 2 + 1, -h_b + d_pod / 2 + 1))]

        def bracket(loads, name):
            return talos.StructuralModel(files["bracket"]["step"], "mm-N-MPa", talos.Material(**PETG_CF), regions,
                                         [talos.FixedSupport("bolts")], loads, talos.MeshSettings(element_size=bf.params["element_mm"]),
                                         name=name)
        solve_fea(bf, {"thrust": bracket([talos.Force("pod", fx=10.0)], "thrust"),
                       "pod_side": bracket([talos.Force("pod", fy=10.0)], "pod_side")},
                  out / "bracket_fea", run=run_fea, threads=threads, progress=progress)

    # the drive, the propeller, its CFD at cruise and one blade at bollard pull (12 Part 2)
    prop, sec, battery, pts, perf = drive(S_wet, p["length"])
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
    loads = pr.blade_loads(prop, pts["bollard"])
    bl = add_after(root, Assembly("blade_fea", "blade", params={"spec": ps.to_dict(), "loads": loads, "material": PA12CF,
                                                                "element_mm": ELEMENT["blade"][fidelity]}), prior, redo)
    if not bl.results.get("complete"):
        model = pr.blade_model(ps, bfiles["step"], loads, element_mm=bl.params["element_mm"], material=PA12CF, name="bollard",
                               half_width_mm=25.0)
        solve_fea(bl, {"bollard": model}, out / "blade_fea", run=run_fea, threads=threads, progress=progress)

    # the whole boat with its propeller as a rotor disk (run_scenario boat(), this tree's draft and rpm)
    sp = scene_params(p, T, pts["cruise"].rpm, prop, sec, SCENE_ITERATIONS[fidelity])
    sc = add_after(root, Assembly("scene_cfd", "rotor_disks", params={"boat": p, "cfd": sp}), prior, redo)
    if not sc.results.get("complete"):
        sg, _ = export_kept(lambda: scene_geometry(p, T), {"boat": p, "draft": T}, out / "cad" / "scene", "boat_double_body",
                            formats=("stl",), stl_tolerance=0.2)
        sc.attach("boat_double_body.stl", sg["stl"], "geometry")
        case = aeromant.CFDCase("aircraft_rotor_disks", sg["stl"], sp, workdir=out / "scene_cfd" / "cruise", geometry_units="mm",
                                environment=environment(run_cfd))
        cfd_node(sc, case, ("drag_force_N", "lift_force_N", "converged", "mesh_cells"), run=run_cfd, processors=processors,
                 progress=progress)

    root.record(points={k: _point(v) for k, v in pts.items()}, performance=perf)
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:                                                    # plain data for notebooks (assemblies.results)
        root.meta["results_to"] = str(results.write(root, vida_path))
    if export:
        path = Path(export_path or DATA / f"{NAME}.json")
        res = boreas.export(path, prop, sec, motor=pr.motor(DRIVE["motor"]), battery=battery, points=pts,
                            notes=f"survey boat from assemblies.workflows.{NAME} ({fidelity}); draft {T:.1f} mm, wake {DRIVE['wake']}")
        res.raise_for_status()
        root.meta["exported_to"] = str(path)
    return root


def _parser():
    return parser("The survey boat: hull, hydrostatics, double-body CFD, hull and bracket FEA, drive, rotor CFD, blade FEA, scene")


if __name__ == "__main__":
    raise SystemExit(main(run, _parser(), DATA / f"{NAME}.vida"))
