"""The rover: a printed four-wheel rover with trailing suspension arms (notebook 11, sections 1-4).

The tree::

    rover (rover)                  the mass budget; the wheel loads on a 30 deg hill and in mud, empty and loaded
      parts (rover_parts)          rover, arm and chassis CAD; their volumes
      terrains (quarter_car)       the quarter car over paved, gravel and rocky ISO 8608 roads (and the rocky field loaded)
      arm_fea (arm)                the suspension arm under the six cases of notebook 11 (Talos)
      chassis_fea (chassis)        the chassis in torsion, under the payload, and climbing loaded
      arm_modes (arm_unit)         the arm with the wheel at the axle: 4 modes; the 100 N unit cases wheel_z, wheel_x (11 §4, §6)
    the pins (axle, pivot) are checked by hand on the root.

The terrains also record the rainflow counts of their wheel forces (11 cell 21, per simulated duration, unscaled: a
notebook scales the cycles by its mission minutes); the root records body bounce, wheel hop and the arm's frequency
check (11 cells 18, 19). The fatigue (damage, life) and print sections of notebook 11 stay in the notebook.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from vegeta import chronos, talos

from .. import DATA, RUNS, vida, results
from .._cli import main, parser
from ..components import leg, road_wheel as rw
from ..components._cad import export_kept
from ..components.rover import Rover
from ..vida import Assembly
from ._common import add_after, solve_fea, unit_fea

NAME = "rover"
G = 9.81
RHO_PA12CF = 1.1e-3
PARTS_G = {"wheels + tyres 4x": 4 * 120.0, "motors + gearboxes 4x": 4 * 95.0, "battery 3S 5000 mAh": 380.0,
           "controller, radio, camera": 180.0, "wiring, bolts, bearings": 120.0}                       # 11 cell 6
M_WHEEL = (120.0 + 95.0 * 0.5) / 1000
SUSPENSION = {"k_susp": 520.0, "zeta": 0.3, "k_tyre": 6000.0}                                           # 11 cell 8
TERRAINS = {
    "paved patrol": {"class": "A", "gd": 16e-6, "speed": 1.5, "length": 300.0, "rocks": None, "drop": None, "seed": 1},
    "gravel trail": {"class": "C", "gd": 256e-6, "speed": 1.2, "length": 300.0, "rocks": (0.02, 0.10, 6.0), "drop": None, "seed": 2},
    "rocky field": {"class": "E", "gd": 4096e-6, "speed": 0.8, "length": 200.0, "rocks": (0.04, 0.12, 2.5), "drop": (150.0, 0.12),
                    "seed": 3}}
CLIMB = {"grade_deg": 30.0, "mu_dry": 0.8, "mu_mud": 0.35, "roll_mud": 0.30, "payload_kg": 6.0, "h_cg": 0.09, "t_stall": 1.2}  # 11 cell 13
PA12CF = dict(name="PA12-CF (printed)", youngs_modulus=3500.0, poissons_ratio=0.40, density=1.1e-9, yield_strength=60.0,
              source="nominal datasheet, flat orientation; assume 30 % less across layers")
STEEL_PIN = dict(name="steel pin (C45)", youngs_modulus=210000.0, poissons_ratio=0.30, density=7.85e-9, yield_strength=400.0,
                 source="handbook")
ELEMENT = {"arm": {"smoke": 4.0, "quick": 3.0, "full": 2.0}, "chassis": {"smoke": 12.0, "quick": 9.0, "full": 6.0}}
ARM_MODES = 4                    # 11 cell 18: the arm's modes with the wheel at the axle
ARM_DAMPING = 0.04               # 11 cell 18: chronos.Structure(..., damping_ratio=0.04)
UNIT_N = 100.0                   # 11 cell 23: the unit cases, 100 N vertical and 100 N longitudinal at the axle
RAINFLOW_DECIMATE = 4            # 11 cell 21: series[::4], 2 kHz -> 500 Hz


def rainflow_blocks(series, scale: float = 1.0) -> list:
    """One force history counted as in 11 cell 21: ASTM rainflow on ``series[::4]``, ranges under 0.5 N dropped,
    (mean, amplitude) rounded to 0.1 N and merged; ``[[mean_N, amplitude_N, cycles], ...]``. The cell multiplies each
    count by its mission scale (``MISSION_MIN * 60 / duration_s``); ``scale=1`` gives the counts per simulated duration."""
    merged = {}
    for rng_, mean, count in chronos.rainflow(series[::4]):                  # 2 kHz -> 500 Hz, ample for < 30 Hz content
        if rng_ < 0.5:                                                        # ignore sub-0.5 N noise
            continue
        key = (round(mean, 1), round(rng_ / 2, 1))
        merged[key] = merged.get(key, 0.0) + count * scale
    return [[float(m), float(a), float(c)] for (m, a), c in merged.items()]


def suspension_frequencies(m_total: float, k_susp: float, zeta: float, k_tyre: float, m_wheel: float = M_WHEEL) -> dict:
    """The sprung mass per corner (11 cell 6), the suspension damping (cell 8), body bounce and wheel hop (cell 18)."""
    M_SPRUNG_CORNER = (m_total - 4 * m_wheel) / 4
    C_SUSP = 2 * zeta * math.sqrt(k_susp * M_SPRUNG_CORNER)
    f_hop = math.sqrt((k_tyre + k_susp) / m_wheel) / (2 * math.pi)
    f_body = math.sqrt(k_susp / M_SPRUNG_CORNER) / (2 * math.pi)
    return {"m_sprung_corner_kg": M_SPRUNG_CORNER, "c_susp_N_s_m": C_SUSP, "f_body_hz": f_body, "f_hop_hz": f_hop}


def frequency_table(modes_hz, wheel_diameter_mm: float, terrains: dict = TERRAINS, damping_ratio: float = ARM_DAMPING) -> dict:
    """The wheel's once-per-revolution and the rock pulse per terrain against the arm modes (11 cells 18, 19): one row per
    terrain (``rock_pulse_hz`` NaN where the terrain has no rocks)."""
    structure = chronos.Structure(tuple(modes_hz), damping_ratio=damping_ratio)
    D_WHEEL = wheel_diameter_mm / 1000
    return {name: {"speed_m_s": s["speed"], "wheel_1_rev_hz": s["speed"] / (math.pi * D_WHEEL),
                   "rock_pulse_hz": (s["speed"] / terrains[name]["rocks"][1]) if terrains[name]["rocks"] else float("nan"),
                   "nearest_arm_mode_hz": structure.nearest_mode(s["speed"] / (math.pi * D_WHEEL)),
                   "margin_wheel_to_arm_mode": structure.margin(s["speed"] / (math.pi * D_WHEEL))} for name, s in terrains.items()}


def arm_unit_models(step, p: dict, element_mm: float, wheel_mass_kg: float = M_WHEEL, unit_N: float = UNIT_N):
    """The arm's modal model with the wheel as a point mass at the axle (11 cells 14, 18) and its unit cases ``wheel_z``,
    ``wheel_x`` at ``unit_N`` (cell 23): ``(arm_modal, {name: (model, load)})``, the input of ``_common.unit_fea``."""
    L, h = p["arm_length"], p["arm_height"]
    rp, ra = p["pivot_diameter"] / 2 + 0.2, p["axle_diameter"] / 2 + 0.2
    ARM_REGIONS = [talos.SurfacesInBox("pivot", (-rp, -h, -rp, rp, h, rp)), talos.SurfacesInBox("axle", (L - ra, -h, -ra, L + ra, h, ra))]
    WHEEL_MASS_T = wheel_mass_kg * 1e-3                                            # tonnes at the axle

    def arm_model(loads, name, masses=()):
        return talos.StructuralModel(step, "mm-N-MPa", talos.Material(**PA12CF), ARM_REGIONS, [talos.FixedSupport("pivot")], loads,
                                     talos.MeshSettings(element_size=element_mm), name=name, masses=list(masses))

    arm_modal = arm_model([], "arm_modal", masses=[talos.PointMass("axle", WHEEL_MASS_T)])
    unit_models = {"wheel_z": (arm_model([talos.Force("axle", fz=unit_N)], "unit_z"), unit_N),
                   "wheel_x": (arm_model([talos.Force("axle", fx=unit_N)], "unit_x"), unit_N)}
    return arm_modal, unit_models


def terrain_runs(m_total: float, k_susp: float, zeta: float, k_tyre: float, names=tuple(TERRAINS)) -> dict:
    """The quarter car over the terrains (11 cells 8 and 11); the static corner load is a quarter of the weight."""
    m_sprung = (m_total - 4 * M_WHEEL) / 4
    c_susp = 2 * zeta * math.sqrt(k_susp * m_sprung)
    out = {}
    for name in names:
        tr = TERRAINS[name]
        x, z = rw.iso8608_profile(tr["length"], 0.005, tr["gd"], tr["seed"])
        if tr["rocks"]:
            z = rw.add_rocks(x, z, *tr["rocks"], seed=tr["seed"] + 10)
        if tr["drop"]:
            z = rw.add_drop(x, z, *tr["drop"])
        r = rw.quarter_car(x, z, tr["speed"], k_susp=k_susp, c_susp=c_susp, k_tyre=k_tyre, m_sprung=m_sprung, m_unsprung=M_WHEEL,
                           static_corner_N=m_total * G / 4)
        r["Fx"] = rw.rock_strike_force(r, tr["speed"])
        out[name] = r
    return out


def wheel_loads(mass, grade_deg, roll, mu, *, wheelbase, r_wheel):
    """Axle loads and traction on a grade (11 cell 13)."""
    c = CLIMB
    W, th = mass * G, math.radians(grade_deg)
    shift = c["h_cg"] / wheelbase * math.tan(th)
    rear, front = W * math.cos(th) * (0.5 + shift) / 2, W * math.cos(th) * (0.5 - shift) / 2
    need = (W * math.sin(th) + roll * W * math.cos(th)) / 4
    avail_rear, avail_front = min(mu * rear, c["t_stall"] / r_wheel), min(mu * front, c["t_stall"] / r_wheel)
    demand = W * math.sin(th) + roll * W * math.cos(th)
    trac_rear = min(avail_rear, max(need, (demand - 2 * avail_front) / 2))
    return {"rear_N": rear, "front_N": front, "traction_needed_N": need, "traction_rear_N": trac_rear,
            "traction_available_N": 2 * avail_rear + 2 * avail_front, "can_climb": bool(2 * avail_rear + 2 * avail_front >= demand)}


def build(fidelity: str = "full", rover: dict | None = None) -> Assembly:
    p = Rover().resolve(**dict(rover or {}, part="rover"))
    root = Assembly(NAME, "rover", params={"fidelity": fidelity, "parts_g": PARTS_G, "m_wheel": M_WHEEL, "climb": CLIMB})
    root.add(Assembly("parts", "rover_parts", params={"rover": p}))
    return root


def run(*, fidelity: str = "full", rover: dict | None = None, run_fea: bool = True, threads: int = 1, out: Path | None = None,
        vida_path: Path | None = None, include: str = "results", export: bool = True, export_path: Path | None = None,
        force: bool = False, redo=(), progress: bool = True, **_ignored) -> Assembly:
    out = Path(out or RUNS / NAME).resolve()
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    if force and out.exists():
        shutil.rmtree(out)
    prior = vida.load(vida_path) if vida_path.is_file() and not force else None
    root = build(fidelity, rover)
    root.reuse(prior)
    if "parts" in redo:
        root.child("parts").forget()

    pn = root.child("parts")
    p = pn.params["rover"]
    rv = Rover()
    files, info = {}, {}
    for part in ("rover", "arm", "chassis"):
        q = dict(p, part=part)
        files[part], info[part] = export_kept(lambda q=q: rv.generate(**q), q, out / "cad" / part, part, stl_tolerance=0.05)
        pn.attach(f"{part}.step", files[part]["step"], "geometry")
    if not pn.results:
        pn.record(volume_mm3={k: v["volume_mm3"] for k, v in info.items()})
    vol = pn.results["volume_mm3"]
    parts = {"chassis (PA12-CF, from CAD)": vol["chassis"] * RHO_PA12CF, "suspension arms 4x (from CAD)": 4 * vol["arm"] * RHO_PA12CF,
             **PARTS_G}
    m_total = sum(parts.values()) / 1000
    c = CLIMB
    r_wheel, wheelbase = p["wheel_diameter"] / 2000, 2 * (p["length"] / 2 - p["pivot_x"]) / 1000
    m_loaded = m_total + c["payload_kg"]

    # the terrains (seconds; the rocky field again with the payload on board)
    tn = add_after(root, Assembly("terrains", "quarter_car", params={"m_total": m_total, "m_wheel": M_WHEEL, "suspension": SUSPENSION,
                                                                     "terrains": TERRAINS, "payload_kg": c["payload_kg"]}), prior, redo)
    runs = None
    if not tn.results:
        s = SUSPENSION
        runs = terrain_runs(m_total, s["k_susp"], s["zeta"], s["k_tyre"])
        loaded = terrain_runs(m_loaded, s["k_susp"], s["zeta"], s["k_tyre"], ("rocky field",))["rocky field"]
        summary = {k: {"speed_m_s": TERRAINS[k]["speed"], "duration_s": float(r["t"][-1]), "tyre_force_max_N": float(r["Ft"].max()),
                       "tyre_force_min_N": float(r["Ft"].min()), "arm_force_max_N": float(r["Fs"].max()),
                       "body_accel_rms_g": float(np.sqrt(np.mean(r["As"] ** 2)) / G),
                       "travel_max_mm": float(np.abs(r["travel"]).max() * 1000), "airborne_%": float(100 * np.mean(r["Ft"] <= 1e-6)),
                       "longitudinal_max_N": float(r["Fx"].max())} for k, r in runs.items()}
        tn.record(summary=summary, peak_z=max(v["tyre_force_max_N"] for v in summary.values()),
                  peak_x=max(v["longitudinal_max_N"] for v in summary.values()), peak_arm=max(v["arm_force_max_N"] for v in summary.values()),
                  peak_z_loaded=float(loaded["Ft"].max()), peak_x_loaded=float(loaded["Fx"].max()),
                  travel_loaded_mm=float(np.abs(loaded["travel"]).max() * 1000))
    if "rainflow" not in tn.results:            # the wheel-force cycles (11 cell 21), also on a saved node made before them
        if runs is None:
            s = SUSPENSION
            runs = terrain_runs(m_total, s["k_susp"], s["zeta"], s["k_tyre"])
        tn.record(rainflow={k: {"duration_s": float(r["t"][-1]), "dt_s": float(r["t"][1] - r["t"][0]) * RAINFLOW_DECIMATE,
                                "wheel_z": rainflow_blocks(r["Ft"]), "wheel_x": rainflow_blocks(r["Fx"])} for k, r in runs.items()})
    T = tn.results

    # the wheel loads (11 cell 13)
    kw = dict(wheelbase=wheelbase, r_wheel=r_wheel)
    hill, hill_loaded = wheel_loads(m_total, c["grade_deg"], 0.02, c["mu_dry"], **kw), wheel_loads(m_loaded, c["grade_deg"], 0.02, c["mu_dry"], **kw)
    mud, mud_loaded = wheel_loads(m_total, 0, c["roll_mud"], c["mu_mud"], **kw), wheel_loads(m_loaded, 0, c["roll_mud"], c["mu_mud"], **kw)
    level = wheel_loads(m_total, 0, 0.02, c["mu_dry"], **kw)

    # the arm (11 cell 14)
    arm_cases = {"level: rocky-field peak": dict(fz=T["peak_z"], fx=T["peak_x"]),
                 "hill climbing, rear wheel": dict(fz=hill["rear_N"], fx=hill["traction_rear_N"]),
                 "mud: dragged sideways": dict(fz=mud["rear_N"], fx=mud["traction_rear_N"], fy=0.5 * mud["rear_N"]),
                 "mud: stuck at stall torque": dict(fz=mud["rear_N"], fx=c["t_stall"] / r_wheel),
                 "heavy payload: rocky-field peak": dict(fz=T["peak_z_loaded"], fx=T["peak_x_loaded"]),
                 "heavy payload: hill climbing": dict(fz=hill_loaded["rear_N"], fx=hill_loaded["traction_rear_N"])}
    an = add_after(root, Assembly("arm_fea", "arm", params={"rover": p, "material": PA12CF, "cases": arm_cases,
                                                            "element_mm": ELEMENT["arm"][fidelity]}), prior, redo)
    if not an.results.get("complete"):
        L, h = p["arm_length"], p["arm_height"]
        rp, ra = p["pivot_diameter"] / 2 + 0.2, p["axle_diameter"] / 2 + 0.2
        regions = [talos.SurfacesInBox("pivot", (-rp, -h, -rp, rp, h, rp)), talos.SurfacesInBox("axle", (L - ra, -h, -ra, L + ra, h, ra))]
        models = {_key(n): talos.StructuralModel(files["arm"]["step"], "mm-N-MPa", talos.Material(**PA12CF), regions,
                                                 [talos.FixedSupport("pivot")], [talos.Force("axle", **f)],
                                                 talos.MeshSettings(element_size=an.params["element_mm"]), name=n)
                  for n, f in arm_cases.items()}
        solve_fea(an, models, out / "arm_fea", run=run_fea, threads=threads, progress=progress)

    # the chassis (11 cell 15)
    px, pz = p["length"] / 2 - p["pivot_x"], -p["plate_thickness"] - p["rail_height"] / 2
    rr = p["pivot_diameter"] / 2 + 0.3
    plate_load = c["payload_kg"] * G / (p["length"] * p["width"])
    cn = add_after(root, Assembly("chassis_fea", "chassis", params={"rover": p, "material": PA12CF, "arm_peak_N": T["peak_arm"],
                                                                    "hill_loaded": hill_loaded, "plate_load_MPa": plate_load,
                                                                    "element_mm": ELEMENT["chassis"][fidelity]}), prior, redo)
    if not cn.results.get("complete"):
        regions = [talos.SurfacesInBox(f"pivot_{'f' if sx > 0 else 'r'}{'l' if sy > 0 else 'r'}",
                                       (sx * px - rr, sy * p["width"] / 2 - 10, pz - rr, sx * px + rr, sy * p["width"] / 2 + 10, pz + rr))
                   for sx in (-1, 1) for sy in (-1, 1)] + [talos.SurfacesOnPlane("plate_top", "z", 0.0)]
        all_pivots = [talos.FixedSupport(f"pivot_{k}") for k in ("fl", "fr", "rl", "rr")]
        a = T["peak_arm"]
        cases = {
            "torsion: one wheel on a rock, the opposite in a hole": (
                [talos.FixedSupport("pivot_rl"), talos.FixedSupport("pivot_fr")],
                [talos.Force("pivot_fl", fz=-a), talos.Force("pivot_rr", fz=-a)]),
            "heavy payload on the plate": (all_pivots, [talos.Pressure("plate_top", plate_load * 2.0)]),
            "hill climbing, loaded: traction at the pivots": (
                [talos.FixedSupport("pivot_fl"), talos.FixedSupport("pivot_fr")],
                [talos.Force("pivot_rl", fx=-hill_loaded["traction_rear_N"], fz=-hill_loaded["rear_N"]),
                 talos.Force("pivot_rr", fx=-hill_loaded["traction_rear_N"], fz=-hill_loaded["rear_N"]),
                 talos.Pressure("plate_top", plate_load)])}
        models = {_key(n): talos.StructuralModel(files["chassis"]["step"], "mm-N-MPa", talos.Material(**PA12CF), regions, sup, loads,
                                                 talos.MeshSettings(element_size=cn.params["element_mm"]), name=n)
                  for n, (sup, loads) in cases.items()}
        solve_fea(cn, models, out / "chassis_fea", run=run_fea, threads=threads, progress=progress)

    # the arm's modes with the wheel on it and its unit cases, on one mesh (11 cells 18, 23)
    mn = add_after(root, Assembly("arm_modes", "arm_unit", params={"rover": p, "material": PA12CF, "wheel_mass_kg": M_WHEEL,
                                                                   "unit_N": UNIT_N, "n_modes": ARM_MODES,
                                                                   "element_mm": ELEMENT["arm"][fidelity]}), prior, redo)
    if not mn.results.get("complete"):
        base, unit = arm_unit_models(files["arm"]["step"], p, mn.params["element_mm"], mn.params["wheel_mass_kg"], mn.params["unit_N"])
        unit_fea(mn, base, unit, out / "arm_modes", n_modes=mn.params["n_modes"], run=run_fea, threads=threads, progress=progress)

    # the pins by hand (11 cell 16)
    lever_axle, lever_pivot = p["wheel_offset"] + p["wheel_width"] / 2, (p["arm_width"] / 2 + 3.0) / 2
    pins = {}
    for n, f in arm_cases.items():
        F = math.sqrt(f.get("fz", 0) ** 2 + f.get("fx", 0) ** 2 + f.get("fy", 0) ** 2)
        ab, pb = leg.pin_bending(F, lever_axle, p["axle_diameter"]), leg.pin_bending(F, lever_pivot, p["pivot_diameter"])
        pins[n] = {"resultant_N": F, "axle_bending_MPa": ab, "pivot_pin_bending_MPa": pb, "pin_SF": STEEL_PIN["yield_strength"] / max(ab, pb)}
    root.record(mass_kg=m_total, parts_g=parts, wheel_loads={"level": level, "hill": hill, "hill_loaded": hill_loaded, "mud": mud,
                                                             "mud_loaded": mud_loaded},
                sag_mm=c["payload_kg"] * G / 4 / SUSPENSION["k_susp"] * 1000, pins=pins)
    s = SUSPENSION                                                # body bounce, wheel hop, the arm against the wheel (11 cells 18, 19)
    modes_hz = mn.results.get("modes_hz") if mn.results.get("complete") else None
    root.record(**suspension_frequencies(m_total, s["k_susp"], s["zeta"], s["k_tyre"]),
                arm_frequency_table=frequency_table(modes_hz, p["wheel_diameter"], TERRAINS) if modes_hz else None)
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:                                                    # plain data for notebooks (assemblies.results)
        root.meta["results_to"] = str(results.write(root, vida_path))
    return root


def _key(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def _parser():
    return parser("The rover: parts, quarter car over three terrains, wheel loads, arm and chassis FEA, pins", cfd=False)


if __name__ == "__main__":
    raise SystemExit(main(run, _parser(), DATA / f"{NAME}.vida"))
