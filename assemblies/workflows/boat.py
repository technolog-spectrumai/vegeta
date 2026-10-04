"""The survey boat: a 1 m printed hull with a motor pod on a transom bracket (notebook 12, Parts 1 and 2).

The tree::

    boat (surface_vessel)          the drive in water: top speed, cruise, endurance, bollard pull; the open-water curves
                                   (12 cell 36) and map; cavitation (cell 38) and noise (cell 55) at the points; the blade-pass
                                   margins (cell 51) and the frequency table (cell 53) once the modes are solved
      hull (survey_boat)           hull, deck and bracket CAD; mass budget (with x/z, cell 7); draft, wetted surface, GM,
                                   the GZ curve
      hull_cfd (double_body)       the hull below the waterline mirrored about it at 2 m/s (no free surface); half its drag
                                   is the hull's (cells 13, 31)
      hull_fea (hull_bottom)       the bottom panels under 1 kPa, the deck edge held (Talos)
      bracket_fea (bracket)        the transom bracket under 10 N of thrust and 10 N sideways on the pod; its 6 modes with the
                                   300 g pod as a point mass (cell 18)
      propeller (propeller)        the 60 mm marine propeller's CAD
      rotor_cfd (rotor)            the propeller at cruise in sea water (Aeromant rotor_mrf)
      blade_fea (blade)            one blade at bollard pull; its 4 modes in air and x0.65 in water (cell 51)
      scene_cfd (rotor_disks)      the boat with its propeller as a rotor disk (scenarios/run_scenario.py boat(), with the
                                   draft and rpm of this tree instead of its recorded ones)

The sea states and fatigue sections of notebook 12 stay in the notebook (the bracket's modes they need are recorded on
bracket_fea).
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
from ._common import add_after, run_cfd as cfd_node, solve_fea, solve_modes

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
POD_MASS_T, BRACKET_MODES = 300e-6, 6                       # 12 cell 18: the pod on the bracket as a point mass [t], 6 modes
BLADE_MODES, ADDED_MASS_FACTOR = 4, 0.65                    # 12 cell 51: in-water / in-air blade frequency, assumed
DEPTH_M, CP_MIN = 0.15, -1.0                                # 12 cell 35: propeller axis below the surface, section Cp_min
NOISE_DIST, NOISE_ANGLE, NOISE_HARMONICS = 1.0, 90.0, 5     # 12 cell 55: 1 m broadside, 5 harmonics, dB re 1 uPa
MAP_RPM, MAP_V = np.arange(500, 7001, 500), np.arange(0.0, 6.01, 0.25)    # the open-water map (boat.json's "map")


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


# ----------------------------------------------------------------------------------------------- notebook data (12 cells 7-55)
def mass_budget(info: dict, p: dict, items) -> dict:
    """12 cell 7's items table as data: ``{item: {mass_g, mass_kg, x_mm, z_mm}}``, the hull shell and the bracket from
    their CAD volumes (``info`` from ``export_kept``), then the fixed ``items`` (name, g, x, z)."""
    rows = [("hull shell + deck (PETG, from CAD)", (info["boat"]["volume_mm3"] - info["bracket"]["volume_mm3"]) * RHO_PETG,
             0.4 * p["length"], 70.0),
            ("transom bracket (PETG-CF, from CAD)", info["bracket"]["volume_mm3"] * 1.25e-3, -10.0, 60.0),
            *[tuple(i) for i in items]]
    return {n: {"mass_g": float(g), "mass_kg": float(g) / 1000, "x_mm": float(x), "z_mm": float(z)} for n, g, x, z in rows}


def bracket_modal_model(step, p: dict, element_mm: float) -> talos.StructuralModel:
    """12 cell 18 as it is: the bracket's regions and ``bracket_model([], "modal", masses=[PointMass("pod", 300e-6)])``,
    the bolts held, no loads, the 300 g pod as a point mass (its element size the node's; 3 mm at full, the cell's)."""
    material = talos.Material(**PETG_CF)
    t_b, w_b, h_b = p["bracket_thickness"], p["bracket_width"], p["bracket_height"]
    BR_REGIONS = [talos.SurfacesInBox("bolts", (-t_b - 1, -w_b / 2 + 3, -h_b + 15, 1, w_b / 2 - 3, 35)),
                  talos.SurfacesInBox("pod", (-t_b - 92, -p["pod_diameter"] / 2 - 1, -h_b - p["pod_diameter"] / 2 - 1, -t_b + 1, p["pod_diameter"] / 2 + 1, -h_b + p["pod_diameter"] / 2 + 1))]

    def bracket_model(loads, name, masses=()):
        return talos.StructuralModel(step, "mm-N-MPa", material, BR_REGIONS, [talos.FixedSupport("bolts")], loads,
                                     talos.MeshSettings(element_size=element_mm), name=name, masses=list(masses))
    return bracket_model([], "modal", masses=[talos.PointMass("pod", POD_MASS_T)])


def blade_modes(f_air) -> dict:
    """12 cell 51: the blade's modes in air and, times the assumed added-mass factor 0.65, in water."""
    f_air = [float(f) for f in f_air]
    f_water = [f * ADDED_MASS_FACTOR for f in f_air]
    return {"modes_hz_air": f_air, "added_mass_factor": ADDED_MASS_FACTOR, "modes_hz_water_assumed": f_water}


def blade_pass_margin(prop, pts: dict, f_water) -> dict:
    """12 cell 51: the blade-pass frequency at each point and its margin to the first wet blade mode (the cell prints it
    as a percentage; here the fraction)."""
    bpf = {k: float(boreas.excitations(prop, v.rpm)["blade_pass_hz"]) for k, v in pts.items()}
    return {"blade_pass_hz": bpf, "first_wet_mode_hz": float(f_water[0]),
            "margin_to_first_wet_mode": {k: abs(f_water[0] - v) / f_water[0] for k, v in bpf.items()}}


def frequency_table(bracket_hz, f_water, prop, cruise_rpm: float, bpf_cruise: float) -> dict:
    """12 cell 53's ``freq_table`` (the plot left out): the bracket modes (with the pod) and the wet blade modes, the rotor
    line nearest each at cruise and the margin to blade-pass at cruise (the cell prints it as a percentage; here the
    fraction). One row per mode, in the cell's order."""
    rows = {**{f"bracket mode {i + 1}": {"hz": float(f), "nearest_line": min(("1P", "BPF"), key=lambda l: abs(math.log(
                   (cruise_rpm / 60 * (1 if l == "1P" else prop.blades)) / f)))}
               for i, f in enumerate(bracket_hz)},
            **{f"blade mode {i + 1} (wet)": {"hz": float(f), "nearest_line": "BPF"} for i, f in enumerate(f_water)}}
    for r in rows.values():
        r["margin_to_BPF_at_cruise"] = abs(float(r["hz"]) - bpf_cruise) / float(r["hz"])
    return rows


def open_water(prop, section) -> dict:
    """12 cell 36 (the plots left out): efficiency and Ct against the advance ratio J at 1500, 2500 and 3500 rpm over
    0.2-6 m/s inflow, and thrust against rpm (500-4500) at 0, 1.35 and 4 m/s inflow, in sea water (Boreas BEMT)."""
    WATER = boreas.SEA_WATER
    curves = {}
    for rpm in (1500, 2500, 3500):
        Vs = np.linspace(0.2, 6, 25)
        ops = [boreas.solve(prop, section, rpm, v, WATER.density) for v in Vs]
        J = [o.advance_ratio for o in ops]
        curves[str(rpm)] = {"inflow_m_s": [float(v) for v in Vs], "J": [float(j) for j in J],
                            "efficiency": [float(o.efficiency) for o in ops], "ct": [float(o.ct) for o in ops]}
    rpms = np.linspace(500, 4500, 21)
    thrust = {}
    for v in (0.0, 1.35, 4.0):
        thrust[str(v)] = [float(boreas.solve(prop, section, r, v, WATER.density).thrust) for r in rpms]
    return {"rho": WATER.density, "vs_J": curves, "thrust_vs_rpm": {"rpm": [float(r) for r in rpms], "thrust_N": thrust}}


def _plain(d: dict) -> dict:
    return {k: v if v is None or isinstance(v, str) else bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
            for k, v in d.items()}


def cavitation_sweep(prop, wake: float = DRIVE["wake"], v_cruise: float = V_CRUISE) -> dict:
    """12 cell 38's ``cav`` table (the plot left out): the cavitation check at 0.7 R over 500-4500 rpm at the cruise inflow,
    0.15 m deep, Cp_min -1; one list per column."""
    rpms = np.linspace(500, 4500, 21)
    cav = [{**_plain(boreas.cavitation(prop, r, wake * v_cruise, DEPTH_M, boreas.SEA_WATER, cp_min=CP_MIN)), "rpm": float(r)}
           for r in rpms]
    return {k: [c[k] for c in cav] for k in ["rpm", *[k for k in cav[0] if k != "rpm"]]}


def noise_rows(prop, pts: dict, v_max: float, wake: float = DRIVE["wake"], v_cruise: float = V_CRUISE) -> tuple[dict, dict, dict]:
    """12 cell 55 (the plot left out): per operating point Gutin's tones and the broadband allowance at 1 m broadside in
    dB re 1 uPa (``propeller.noise``, the cell's loop body), their total and the cavitation number at 0.7 R at the
    point's inflow (cruise and full speed at the wake x speed, bollard pull at rest), in the cell's columns. Also each
    point's tone spectrum (the cell plots cruise's) and its whole cavitation check (at full speed: cell 38's ``incept``)."""
    WATER = boreas.SEA_WATER
    rotor = pr.noise(prop, pts, medium=WATER, distance=NOISE_DIST, angle_deg=NOISE_ANGLE, harmonics=NOISE_HARMONICS)
    rows, tones, cav = {}, {}, {}
    for name, pt in pts.items():
        n = rotor[name]
        c = boreas.cavitation(prop, pt.rpm, wake * v_cruise if name == "cruise" else (wake * v_max if name == "full" else 0.0),
                              DEPTH_M, WATER, cp_min=CP_MIN)
        rows[name] = {"rpm": n["rpm"], "BPF_hz": n["BPF_hz"], "tonal_dB_re_1uPa_1m": n["tonal_dB"],
                      "broadband_dB": n["broadband_dB"], "total_dB": n["one_rotor_dB"],
                      "cavitation_number": float(c["cavitation_number"]), "cavitates": bool(c["cavitates"])}
        tones[name] = {"frequency_hz": n["frequency_hz"], "spl_db": n["spl_db"]}
        cav[name] = _plain(c)
    return rows, tones, cav


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
    if "mass_budget" not in hn.results:                           # 12 cell 7: masses with units and positions
        hn.record(mass_budget=mass_budget(info, p, hn.params["items"]))
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
    if hc.results.get("complete") and "half_drag_N" not in hc.results:   # 12 cells 13, 31: the hull's, half the double body's
        d = (hc.results.get("forces") or {}).get("drag_force_N")
        hc.record(half_drag_N=d / 2 if d is not None else None, speed_m_s=hc.params["cfd"]["velocity"])

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
    if bf.results.get("complete") and "modes_hz" not in bf.results:     # 12 cell 18: the modes with the 300 g pod on it
        if run_fea:
            mdir = out / "bracket_fea" / "modal"
            mr = solve_modes(bracket_modal_model(files["bracket"]["step"], p, bf.params["element_mm"]), mdir,
                             n_modes=BRACKET_MODES, mesh_from=out / "bracket_fea" / "thrust", threads=threads, progress=progress)
            if mr.ok:
                bf.attach("modal", mdir, "mesh")
                bf.record(modes_hz=[float(f) for f in mr.metrics["frequencies_hz"]], pod_mass_kg=round(POD_MASS_T * 1000, 9))
            else:
                bf.not_run(f"modes: {mr.messages[-1] if mr.messages else mr.status}")
        else:
            bf.not_run("modes: run_fea=False")

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
    model = pr.blade_model(ps, bfiles["step"], loads, element_mm=bl.params["element_mm"], material=PA12CF, name="bollard",
                           half_width_mm=25.0)
    if not bl.results.get("complete"):
        solve_fea(bl, {"bollard": model}, out / "blade_fea", run=run_fea, threads=threads, progress=progress)
    if bl.results.get("complete") and "modes_hz_air" not in bl.results:   # 12 cell 51: same mesh and supports, 4 modes
        if run_fea:
            mdir = out / "blade_fea" / "modal"
            mr = solve_modes(model, mdir, n_modes=BLADE_MODES, mesh_from=out / "blade_fea" / "bollard", threads=threads,
                             progress=progress)
            if mr.ok:
                bl.attach("modal", mdir, "mesh")
                bl.record(**blade_modes(mr.metrics["frequencies_hz"]))
            else:
                bl.not_run(f"modes: {mr.messages[-1] if mr.messages else mr.status}")
        else:
            bl.not_run("modes: run_fea=False")

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
    grid = boreas.performance_map(prop, sec, MAP_RPM, MAP_V, RHO_W)                    # about 2 s
    noise, tones, cav = noise_rows(prop, pts, perf["top_speed_m_s"])                    # 12 cells 38, 55
    f_water, f_bracket = bl.results.get("modes_hz_water_assumed"), bf.results.get("modes_hz")
    margin = blade_pass_margin(prop, pts, f_water) if f_water else None               # 12 cell 51
    root.record(open_water=open_water(prop, sec), open_water_map=grid,                 # 12 cell 36
                cavitation={"depth_m": DEPTH_M, "cp_min": CP_MIN, "radial_station": 0.7,
                            "inflow_m_s": {"cruise": DRIVE["wake"] * V_CRUISE, "full": DRIVE["wake"] * perf["top_speed_m_s"],
                                           "bollard": 0.0},
                            "points": cav, "sweep_at_cruise_inflow": cavitation_sweep(prop)},
                noise={"distance_m": NOISE_DIST, "angle_deg": NOISE_ANGLE, "harmonics": NOISE_HARMONICS,
                       "reference": "dB re 1 uPa", "points": noise, "tones": tones},
                blade_pass_margin=margin,
                frequency_table=(frequency_table(f_bracket, f_water, prop, pts["cruise"].rpm, margin["blade_pass_hz"]["cruise"])
                                 if f_water and f_bracket else None),                  # 12 cell 53
                drive_wake={"inflow_ratio": DRIVE["wake"], "wake_fraction": round(1 - DRIVE["wake"], 12),
                            "meaning": "the propeller's inflow is inflow_ratio x boat speed (drive.wake); "
                                       "wake_fraction = 1 - inflow_ratio (the wake fraction convention)"})
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:                                                    # plain data for notebooks (assemblies.results)
        root.meta["results_to"] = str(results.write(root, vida_path))
    if export:
        path = Path(export_path or DATA / f"{NAME}.json")
        res = boreas.export(path, prop, sec, motor=pr.motor(DRIVE["motor"]), battery=battery, points=pts, map=grid, rho=RHO_W,
                            notes=f"survey boat from assemblies.workflows.{NAME} ({fidelity}); draft {T:.1f} mm, wake {DRIVE['wake']}")
        res.raise_for_status()
        root.meta["exported_to"] = str(path)
    return root


def _parser():
    return parser("The survey boat: hull, hydrostatics, double-body CFD, hull and bracket FEA, drive, rotor CFD, blade FEA, scene")


if __name__ == "__main__":
    raise SystemExit(main(run, _parser(), DATA / f"{NAME}.vida"))
