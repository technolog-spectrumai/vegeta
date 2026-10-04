"""MERLIN, the wildfire sampler: its design (notebook 26), its mission (26b) and the propulsor race (27), on the
propulsor libraries (``assemblies.workflows.propulsors``).

The tree::

    merlin (aircraft)              the design export other workflows read (assemblies/data/merlin_design.json)
      airframe (merlin_airframe)   the three aircraft (ducted fan, tractor, pusher): CAD, masses, drag build-up
      wing_fea (wing_fea)          the wing in a 3 g pull-up and in Pratt's gust at the 60 m/s dash (Talos)
      polar_cfd (merlin_polar)     the three aircraft at 30 m/s, 2 deg (Aeromant rans_ksst_external): the Cd0 corrections
      propulsors (propulsor_libraries)  the saved libraries (assemblies/data/propulsors.vida), grafted as they are
      race (propulsor_race)        every library entry on MERLIN at 1900, 2700 and 3500 W over the three races
      mission (merlin_mission)     the three MERLINs on one battery: performance, reach and return to 5-30 km; 26b's
                                   race table and winners, the propulsors and airframes flown, the pitch studies, the
                                   race to a fire 600 m up a mountain, where the sensor breathes and the winner's
                                   mission through the smoke with the fire's size estimated (``mission_extras``)

The wing is the design's defaults unless ``--merlin thickness=0.15`` (notebook 26 compared NACA 2412 and 2415). The
rotor-disk CFD and the movies of notebook 26b stay in the notebook.

    python -m assemblies.workflows.propulsors -j 4       # first (once): the libraries
    python -m assemblies.workflows.merlin --fidelity quick -j 4
"""
from __future__ import annotations

import dataclasses
import math
import shutil
from pathlib import Path

import numpy as np
from vegeta import aeromant, talos

from .. import DATA, RUNS, vida, results
from .._cli import main, parser
from .._cli_util import parse_assignments
from ..components import merlin as mdesign, merlin_flight as mf, merlin_race as mr, wing
from ..components._cad import export_kept
from ..components.impeller import environment
from ..vida import Assembly
from ._common import add_after, solve_fea
from .propulsors import load_library

NAME = "merlin"
KINDS = ("edf", "tractor", "pusher")
MISSION = mf.Mission(sampling_agl_m=150.0, sampling_s=180.0, launch_speed=12.0, battery_wh=6 * 3.7 * 8.0,
                     usable_fraction=0.85, reserve_fraction=0.15)                                # 26 cell 4
MAX_ELECTRICAL_W, DRIVE_EFF, SHELL_AREAL_DENSITY_G_MM2 = 1900.0, 0.85, 0.7e-3
RHO, NU, G = 1.225, 1.5e-5, 9.81
V_DESIGN, V_CFD, AOA = 25.0, 30.0, 2.0
V_GUST, U_GUST, PULL_UP_N = 60.0, 7.5, 3.0                                                      # 26 cell 12
LW_PLA = dict(wing.LW_PLA, source="assumed (09a); replace with coupon tests")
WING_ELEMENT_MM = {"smoke": 20.0, "quick": 14.0, "full": 10.0}                                  # full: the notebook's
POLAR = {"smoke": dict(iterations=60, surface_level=3, near_level=2, wake_level=1),
         "quick": dict(iterations=400, surface_level=3, near_level=2, wake_level=1),
         "full": dict(iterations=400, surface_level=4, near_level=3, wake_level=2)}             # full: the notebook's
FIRE_ELEVATION_M = 600.0                                                                         # 26b cell 12
SMOKE_PLUME = dict(source_xy=(20000.0, 0.0), heat_mw=40.0, wind_speed=5.0, wind_from_deg=225.0)  # 26b cell 18
SMOKE_TRACK_EVERY = 10                       # the smoke mission's track as recorded: every 10th 0.5 s step
RACE_COLUMNS = ("reach [min]", "return [min]", "landing [min]", "dash [m/s]", "dash climb [deg]", "return [m/s]", "limited by",
                "energy used [Wh]", "of [Wh]")                  # merlin_flight.race_table's (an unreachable row has two)
MOUNTAIN_KEYS = ("race_mountain", "race_mountain_vs_flat", "winner_per_distance_mountain")
MISSION_EXTRAS = ("race_table", "winner_per_distance", "first_home_per_distance", "units", "airframes", "pitch_study",
                  *MOUNTAIN_KEYS, "sensor_air", "smoke")                       # the mission node's results from 26b


def build(fidelity: str = "full", merlin: dict | None = None) -> Assembly:
    p = mdesign.Merlin().resolve(**dict(merlin or {}))
    root = Assembly(NAME, "aircraft", params={"fidelity": fidelity, "mission": dataclasses.asdict(MISSION),
                                              "distances_km": list(mf.DISTANCES_KM), "max_electrical_w": MAX_ELECTRICAL_W,
                                              "drive_efficiency": DRIVE_EFF})
    root.add(Assembly("airframe", "merlin_airframe", params={"merlin": p, "shell_g_mm2": SHELL_AREAL_DENSITY_G_MM2,
                                                             "common_mass_kg": mf.COMMON_MASS_KG, "unit_mass_kg": mf.UNIT_MASS_KG,
                                                             "v_design": V_DESIGN}))
    return root


def airframe(node: Assembly, out: Path) -> Assembly:
    """The three aircraft in CAD and the size table of notebook 26 cell 10."""
    p = node.params["merlin"]
    m = mdesign.Merlin()
    common = dict(node.params["common_mass_kg"])
    info = {}
    for k in KINDS:
        q = dict(p, propulsion=k, part="aircraft")
        files, info[k] = export_kept(lambda q=q: m.generate(**q), q, out / k, f"merlin_{k}", formats=("step",))
        node.attach(f"merlin_{k}.step", files["step"], "geometry")
    common["airframe (printed shells)"] = round(info["pusher"]["surface_area_mm2"] * SHELL_AREAL_DENSITY_G_MM2 / 1000, 3)
    size = {}
    for k in KINDS:
        b = mdesign.drag_buildup({"propulsion": k}, V_DESIGN, NU)
        mass = sum(common.values()) + node.params["unit_mass_kg"][k]
        S = b["planform"]
        size[k] = {"all_up_mass_kg": mass, "wing_area_m2": S, "aspect_ratio": b["aspect_ratio"], "wing_loading_N_m2": mass * G / S,
                   "stall_speed_m_s": math.sqrt(2 * mass * G / (RHO * S * 1.2)), "cd0_buildup": b["cd0"],
                   "cd0_with_allowance": (b["cd_area_m2"] + mf.EXTRA_CD_AREA_M2) / S,
                   "cd_area_cm2": {c: v * 1e4 for c, v in b["parts_m2"].items()}}
    node.record(common_mass_kg=common, size=size, valid={k: info[k]["valid"] for k in KINDS})
    return node


def design_export(root: Assembly, p: dict) -> dict:
    """The design notebook 26 cell 19 writes (the format ``merlin_flight.load_design`` reads)."""
    af, wf, pc = root.child("airframe"), root.child("wing_fea"), root.child("polar_cfd")
    params = {k: v for k, v in p.items() if k not in ("part", "propulsion", "angle_of_attack_deg") and not k.startswith("edf_")}
    cfd = {k: v["cd0"] for k, v in (pc.results.get("polar") or {}).items() if v.get("cd0") is not None}
    corr = {k: (cfd[k] - mdesign.drag_buildup(dict(params, propulsion=k), V_DESIGN, NU)["cd0"]) if k in cfd else 0.0 for k in KINDS}
    names = {"edf": "ducted fan", "tractor": "tractor propeller", "pusher": "pusher propeller"}
    size = af.results["size"]
    variants = {names[k]: {"all-up mass [kg]": size[k]["all_up_mass_kg"], "wing area [m²]": size[k]["wing_area_m2"],
                           "aspect ratio": size[k]["aspect_ratio"], "wing loading [N/m²]": size[k]["wing_loading_N_m2"],
                           "stall speed (Cl_max 1.2) [m/s]": size[k]["stall_speed_m_s"], "Cd0 build-up": size[k]["cd0_buildup"],
                           "Cd0 with allowance": size[k]["cd0_with_allowance"],
                           **{f"Cd·A {c} [cm²]": v for c, v in size[k]["cd_area_cm2"].items()}} for k in KINDS}
    stress = wf.results.get("stress") or {}
    return {"source": f"assemblies.workflows.{NAME}", "merlin_params": params, "common_mass_kg": af.results["common_mass_kg"],
            "unit_mass_kg_1900W": dict(mf.UNIT_MASS_KG), "extra_cd_area_m2": mf.EXTRA_CD_AREA_M2, "cd0_correction": corr,
            "cd0_cfd": cfd, "cl_max": 1.2, "oswald": 0.8, "variants": variants,
            "wing_fea": {"wing": {"note": f"NACA wing, thickness {p['thickness']}", "thickness": p["thickness"],
                                  **{f"{c} {q}": (stress.get(c) or {}).get(key) for c in ("pull_up", "dash_gust")
                                     for q, key in (("tip [mm]", "max_displacement_mm"), ("SF", "safety_factor"))}}},
            "preferred_wing": {"revision": None, "thickness": p["thickness"]},
            "gust": {"speed_m_s": V_GUST, "gust_m_s": U_GUST, "load_factor": wf.params["gust"]["load_factor"]},
            "mission": dataclasses.asdict(MISSION), "distances_km": list(mf.DISTANCES_KM),
            "max_electrical_w": MAX_ELECTRICAL_W, "drive_efficiency": DRIVE_EFF}


def run(*, fidelity: str = "full", merlin: dict | None = None, propulsors_vida: Path | None = None, run_cfd: bool = True,
        run_fea: bool = True, processors: int = 1, jobs: int = 1, threads: int = 1, out: Path | None = None,
        vida_path: Path | None = None, include: str = "results", export: bool = True, export_path: Path | None = None,
        force: bool = False, redo=(), progress: bool = True) -> Assembly:
    out = Path(out or RUNS / NAME).resolve()
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    lib_path = Path(propulsors_vida or DATA / "propulsors.vida")
    if not lib_path.is_file():
        raise FileNotFoundError(f"{lib_path} is missing: run `python -m assemblies.workflows.propulsors` first")
    if force and out.exists():
        shutil.rmtree(out)
    prior = vida.load(vida_path) if vida_path.is_file() and not force else None
    root = build(fidelity, merlin)
    root.reuse(prior)
    if "airframe" in redo:
        root.child("airframe").forget()

    af = root.child("airframe")
    p = af.params["merlin"]
    if not af.results:
        airframe(af, out / "airframe")
    size = af.results["size"]

    # the wing (26 cell 12): W at the heaviest variant, S of the first, Pratt's gust at 60 m/s
    W_MAX = max(v["all_up_mass_kg"] for v in size.values()) * G
    S_W = size["edf"]["wing_area_m2"]
    gust = wing.pratt_gust(W_MAX, S_W, p["root_chord"] * (1 + p["taper"]) / 2000, wing.lift_slope(6.5, 0.85), V_GUST,
                           gust_m_s=U_GUST, rho=RHO, g=G)
    wf = add_after(root, Assembly("wing_fea", "wing_fea", params={"merlin": p, "weight_N": W_MAX, "wing_area_m2": S_W,
                                                                  "gust": gust, "material": LW_PLA,
                                                                  "element_mm": WING_ELEMENT_MM[fidelity]}), prior, redo)
    if not wf.results.get("complete"):
        wp = dict(p, part="wing", propulsion="edf")
        files, _ = export_kept(lambda: mdesign.Merlin().generate(**wp), wp, out / "wing", "wing", formats=("step",))
        wf.attach("wing.step", files["step"], "geometry")
        step = files["step"]
        regions = [wing.root_region(p["fuselage_diameter"]), talos.Surfaces("lift", wing.lower_skins(step))]
        models = {name: wing.wing_model(step, regions, [talos.Pressure("lift", wing.lift_pressure_MPa(n, W_MAX, S_W))],
                                        element_mm=wf.params["element_mm"], material=LW_PLA, name=name)
                  for name, n in (("pull_up", PULL_UP_N), ("dash_gust", gust["load_factor"]))}
        solve_fea(wf, models, out / "wing_fea", run=run_fea, threads=threads, progress=progress)

    # the three aircraft at 30 m/s, 2 deg (26 cell 15)
    cfd_values = dict(velocity=V_CFD, kinematic_viscosity=NU, density=RHO, reference_area=S_W, reference_length=p["span"] / 4000,
                      center_of_rotation=(0.07, 0.0, 0.0), residual_target=1e-4, cells_per_length=2.0, **POLAR[fidelity])
    pc = add_after(root, Assembly("polar_cfd", "merlin_polar", params={"merlin": p, "aoa_deg": AOA, "cfd": cfd_values}), prior, redo)
    if not pc.results.get("complete"):
        cases = []
        for k in KINDS:
            q = dict(p, part="aircraft", propulsion=k, angle_of_attack_deg=AOA)
            files, _ = export_kept(lambda q=q: mdesign.Merlin().generate(**q), q, out / "polar" / k, f"merlin_{k}_2deg",
                                   formats=("stl",), stl_tolerance=0.2)
            pc.attach(f"merlin_{k}_2deg.stl", files["stl"], "geometry")
            cases.append(aeromant.CFDCase("rans_ksst_external", files["stl"], cfd_values, workdir=out / "polar_cfd" / k,
                                          geometry_units="mm", environment=environment(run_cfd)))
        rs = aeromant.run_cases(cases, jobs=jobs, processors=processors, run=run_cfd, progress=progress)
        polar = {}
        for k, c, r in zip(KINDS, cases, rs):
            if c.workdir.exists():
                pc.attach(f"case_{k}", c.workdir, "cases")
            if r.ok:
                cl, cd = r.metrics["Cl"], r.metrics["Cd"]
                polar[k] = {"Cl": cl, "Cd": cd, "cd0": cd - cl ** 2 / (math.pi * size[k]["aspect_ratio"] * 0.8)}
            else:
                pc.not_run(f"{k}: " + ("run_cfd=False" if not run_cfd else (r.messages[-1] if r.messages else r.status)))
        if polar:
            pc.record(polar=polar, complete=len(polar) == len(KINDS))

    design = design_export(root, p)

    # the propulsor libraries, grafted; the race of notebook 27 and the mission of 26b on them
    libs = vida.load(lib_path)
    libs_node = libs.copy("propulsors")
    libs_node.meta["reused"] = {"computed_at": libs.meta.get("computed_at"), "from": str(lib_path)}
    for k in ("loaded_from", "manifest"):
        libs_node.meta.pop(k, None)
    root.add(libs_node)
    lib = load_library(libs)
    race = add_after(root, Assembly("race", "propulsor_race", params={"design": design, "libraries": libs.key,
                                                                      "powers_w": list(mr.POWERS_W), "races": list(mr.RACES)}),
                     prior, redo)
    if not race.results:
        table = race_table(lib, design, progress=progress)
        winners = {}
        for name in mr.RACES:
            w = mr.winners(table, name).reset_index()
            winners[name] = {c: w[c].tolist() for c in ("kind", "power [W]", "config", name, "top speed [m/s]", "mass [kg]")}
        race.record(table={c: table[c].tolist() for c in table.columns}, winners=winners, n=len(table))
    mission = add_after(root, Assembly("mission", "merlin_mission", params={"design": design, "libraries": libs.key}), prior, redo)
    merlins = None
    if not mission.results:
        merlins = _merlins(design, lib)
        perf = {k: mf.performance(v["airframe"], v["unit"]) for k, v in merlins.items()}
        reach = {k: {str(d): {kk: mf.race(v["airframe"], v["unit"], d * 1000.0, MISSION)[kk]
                              for kk in ("time_to_fire_s", "return_time_s")} for d in mf.DISTANCES_KM} for k, v in merlins.items()}
        mission.record(performance={k: {kk: (float(vv) if np.ndim(vv) == 0 else np.asarray(vv, float)) for kk, vv in pf.items()}
                                    for k, pf in perf.items()},
                       mass_kg={k: v["airframe"].mass_kg for k, v in merlins.items()}, reach=reach)
    # what 26b shows besides the reach, on the same MERLINs (interpolation and a kinematic flight, seconds); a reused
    # mission gets the keys it lacks once, then keeps them
    missing = extras_missing(mission.results)
    if missing:
        extras = mission_extras(merlins or _merlins(design, lib))
        mission.record(**{k: extras[k] for k in missing})

    root.record(design=design)
    if not (pc.results.get("polar")):
        root.not_run("Cd0 corrections from CFD: none (the drag build-up is the polar)")
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:                                                    # plain data for notebooks (assemblies.results)
        root.meta["results_to"] = str(results.write(root, vida_path))
    path = Path(export_path or mf.DESIGN_JSON)
    solved = bool(wf.results.get("complete") or pc.results.get("polar"))
    if export and path.is_file() and not solved:
        # the shared design file is not replaced by a run that solved neither the wing FEA nor the polar CFD
        root.meta["export_kept"] = f"{path} kept: this run solved neither the wing FEA nor the polar CFD"
        print(root.meta["export_kept"])
    elif export:
        path.write_text(__import__("json").dumps(dict(design, written_at=vida.utc_now(), git=vida.manifest(vida_path)["git"],
                                                      fidelity=fidelity), indent=1, default=float))
        root.meta["exported_to"] = str(path)
    return root


def race_table(lib: dict, design: dict, *, powers=mr.POWERS_W, progress=True):
    """``merlin_race.study`` one power at a time under a progress bar (minutes), its rows put back in ``study``'s order
    (every map and layout, then every power): the same table as one ``study`` call."""
    import pandas as pd
    from tqdm.auto import tqdm

    parts = [mr.study(lib, powers=(P,), design=design) for P in tqdm(powers, desc="MERLIN race (per power)", disable=not progress)]
    rows = [t.iloc[[i]] for i in range(len(parts[0])) for t in parts]
    return pd.concat(rows, ignore_index=True) if rows else parts[0]


def _merlins(design: dict, lib: dict) -> dict:
    return mf.build_merlins(max_electrical_w=MAX_ELECTRICAL_W, drive_efficiency=DRIVE_EFF, mission=MISSION,
                            distance_m=20000.0, design=design, library=lib)


# ------------------------------------------------------------------------------------------------ 26b on the mission node
def extras_missing(res: dict) -> list[str]:
    """The ``MISSION_EXTRAS`` a mission node's results lack, plus those made with another ``FIRE_ELEVATION_M`` or
    ``SMOKE_PLUME`` than now (so a changed constant is not silently ignored)."""
    missing = [k for k in MISSION_EXTRAS if k not in res]
    mountain = res.get("race_mountain")
    if mountain and mountain[0].get("fire elevation [m]") != FIRE_ELEVATION_M:
        missing += [k for k in MOUNTAIN_KEYS if k not in missing]
    smoke = res.get("smoke")
    if smoke and "smoke" not in missing and (vida.canonical(smoke.get("plume"))
                                              != vida.canonical(dataclasses.asdict(mf.Plume(**SMOKE_PLUME)))):
        missing.append("smoke")
    return missing


def mission_extras(MERLINS: dict, MISSION: mf.Mission = MISSION, DISTANCES_KM=mf.DISTANCES_KM) -> dict:
    """What notebook 26b shows besides the reach, on the MERLINs of ``_merlins`` (the mission node's ``MISSION_EXTRAS``;
    the tables as lists of records, which ``results.table`` reads back as DataFrames):

    - ``race_table`` (cell 8's ``RACE_TAB``): per distance and propulsor the reach, return and landing times, the dash
      and return speeds, the dash's climb, what limited the dash, the energy used and the budget;
    - ``winner_per_distance`` (cell 9, the cell 24 hand-off: first over the fire) and ``first_home_per_distance``;
    - ``units``, ``airframes``, ``pitch_study`` (cells 4 and 6, the hand-off's polar and masses): what each MERLIN flies;
    - ``race_mountain``, ``race_mountain_vs_flat``, ``winner_per_distance_mountain`` (cell 12): the fire
      ``FIRE_ELEVATION_M`` up a mountain;
    - ``sensor_air`` (cell 14; None without both the ducted fan and the tractor);
    - ``smoke`` (cells 18 and 20): the 20 km winner through the smoke and the fire's size from its passes (None when
      nothing reaches 20 km)."""
    RACE_TAB = _all_columns(mf.race_table(MERLINS, DISTANCES_KM, MISSION))                                 # cell 8
    WINNER, FIRST_HOME = race_winners(RACE_TAB, DISTANCES_KM)                                               # cell 9
    MOUNTAIN_TAB = _all_columns(mf.race_table(MERLINS, DISTANCES_KM, MISSION, fire_elevation_m=FIRE_ELEVATION_M))   # cell 12
    cmp_ = mountain_vs_flat(RACE_TAB, MOUNTAIN_TAB)
    UPHILL = uphill_winners(MOUNTAIN_TAB, DISTANCES_KM)
    FASTEST = WINNER.get(20.0)                                     # cell 9: the design point, the 20 km fire

    def km(d):                                                     # cell 24's keys
        return f"{d:.0f} km"

    return {"race_table": _records(RACE_TAB),
            "winner_per_distance": {km(d): WINNER[d] for d in DISTANCES_KM},
            "first_home_per_distance": {km(d): FIRST_HOME[d] for d in DISTANCES_KM},
            "units": units_flown(MERLINS),
            "airframes": {k: dataclasses.asdict(v["airframe"]) for k, v in MERLINS.items()},
            "pitch_study": [dict(propulsor=k, **row) for k, v in MERLINS.items() for row in v["info"].get("pitch_study", [])],
            "race_mountain": _records(MOUNTAIN_TAB.assign(**{"fire elevation [m]": FIRE_ELEVATION_M})),
            "race_mountain_vs_flat": _records(cmp_.set_axis([" / ".join(c) for c in cmp_.columns], axis=1)),
            "winner_per_distance_mountain": {km(d): UPHILL[d] for d in DISTANCES_KM},
            "sensor_air": sensor_air(MERLINS) if {"edf", "tractor"} <= set(MERLINS) else None,
            "smoke": smoke_mission(MERLINS, FASTEST, MISSION) if FASTEST else None}


def _all_columns(tab):
    """A race table with every column, also when no row reached (``merlin_flight.race_table`` then has only two)."""
    return tab.reindex(columns=list(dict.fromkeys([*RACE_COLUMNS, *tab.columns])))


def _records(tab) -> list[dict]:
    """A table indexed by (distance, propulsor) as a list of records."""
    return tab.reset_index(names=["distance", "propulsor"]).to_dict("records")


def race_winners(RACE_TAB, DISTANCES_KM, KINDS=KINDS, NAME=mf.NAMES) -> tuple[dict, dict]:
    """Notebook 26b cell 9 (its print as data): per distance the propulsor first over the fire (``WINNER``) and the one
    first home (the earliest landing), as kinds; None where none reaches."""
    WINNER, FIRST_HOME = {}, {}
    for d in DISTANCES_KM:
        sub = RACE_TAB.xs(f"{d:g} km", level=0)["reach [min]"].astype(float)
        if sub.isna().all():                                       # (not in the cell: nothing reaches this fire)
            WINNER[d] = FIRST_HOME[d] = None
            continue
        best = sub.idxmin()
        WINNER[d] = next(k for k in KINDS if NAME[k] == best)
        land = RACE_TAB.xs(f"{d:g} km", level=0)["landing [min]"].astype(float)
        FIRST_HOME[d] = next(k for k in KINDS if NAME[k] == land.idxmin())
    return WINNER, FIRST_HOME


def mountain_vs_flat(RACE_TAB, MOUNTAIN_TAB):
    """Notebook 26b cell 12's ``cmp_``: reach, landing and dash speed on flat ground and to the fire up the mountain, the
    dash's climb angle and what the hill costs in reach [s] (two-level columns)."""
    import pandas as pd
    cmp_ = pd.DataFrame({("reach [min]", "flat"): RACE_TAB["reach [min]"], ("reach [min]", "mountain"): MOUNTAIN_TAB["reach [min]"],
                         ("landing [min]", "flat"): RACE_TAB["landing [min]"], ("landing [min]", "mountain"): MOUNTAIN_TAB["landing [min]"],
                         ("dash [m/s]", "flat"): RACE_TAB["dash [m/s]"], ("dash [m/s]", "mountain"): MOUNTAIN_TAB["dash [m/s]"],
                         ("dash", "climb [deg]"): MOUNTAIN_TAB["dash climb [deg]"]}).astype(float)
    cmp_[("reach [min]", "uphill costs [s]")] = (cmp_[("reach [min]", "mountain")] - cmp_[("reach [min]", "flat")]) * 60
    return cmp_


def uphill_winners(MOUNTAIN_TAB, DISTANCES_KM, KINDS=KINDS, NAME=mf.NAMES) -> dict:
    """Notebook 26b cell 12's loop (its print as data): per distance the propulsor first over the fire up the mountain,
    as a kind; None where none reaches."""
    out = {}
    for d in DISTANCES_KM:
        sub = MOUNTAIN_TAB.xs(f"{d:g} km", level=0)["reach [min]"].astype(float)
        out[d] = None if sub.isna().all() else next(k for k in KINDS if NAME[k] == sub.idxmin())
    return out


def units_flown(MERLINS: dict) -> dict:
    """Notebook 26b cell 4's prints and cell 6's static thrust, per MERLIN: the library entry it flies (``id``,
    ``label``, to find it in the propulsor libraries and the race table), the unit's mass and full-throttle static
    thrust; the fan's duct loss, exit area ratio and the fuselage area its jet scrubs; a propeller's size, rpm limit,
    effective wake ``w`` and thrust deduction ``t``."""
    out = {}
    for k, v in MERLINS.items():
        i = v["info"]
        row = {"id": i["entry"]["id"], "label": i["entry"]["label"], "unit_mass_kg": v["unit"].mass_kg,
               "static_thrust_N": v["unit"].full(0.0)[0]}                                                # cell 6
        if k == "edf":                                                                                 # cell 4
            row.update(duct_loss=i["duct_loss"], exit_area_ratio=i["housing"]["exit_area_ratio"], scrub_area_m2=i["scrub_area_m2"])
        else:
            row.update(prop=i["prop"].name, rpm_max=i["rpm_max"], w=i["w"], t=i["t"])
        out[k] = row
    return out


def sensor_air(MERLINS: dict, RHO=RHO) -> dict:
    """Notebook 26b cell 14, where the sensor breathes: at 15, 30 and 50 m/s the ducted fan's capture stream tube against
    its inlet's highlight area and its jet speed, and the tractor's slipstream excess in the far wake (the cell's
    ``rows``, by speed)."""
    e_fan = MERLINS["edf"]["info"]["entry"]                    # the library map: jet speed over (V, rpm)
    A_exit = e_fan["exit_area_ratio"] * e_fan["fan_area_m2"]
    A_high = math.pi * (MERLINS["edf"]["info"]["housing"]["highlight_radius"] / 1000) ** 2
    A_prop = math.pi * (MERLINS["tractor"]["info"]["prop"].diameter / 2) ** 2
    rows = {}
    for V in (15.0, 30.0, 50.0):
        rpm = MERLINS["edf"]["unit"].full(V)[2]
        vj = float(np.interp(rpm, e_fan["rpm"], [np.interp(V, e_fan["V"], e_fan["exit_velocity"][:, j]) for j in range(len(e_fan["rpm"]))]))
        T = MERLINS["tractor"]["unit"].full(V)[0]                 # far-wake excess from momentum: T = rho A (V + u) u, u = jet - V
        u = math.sqrt(V ** 2 + 2 * T / (RHO * A_prop)) - V
        rows[f"{V:.0f} m/s"] = {"EDF capture area / highlight area": A_exit * vj / V / A_high, "EDF jet speed [m/s]": vj,
                                "tractor slipstream excess, far wake [m/s]": u}
    return rows


def smoke_mission(MERLINS: dict, FASTEST: str, MISSION: mf.Mission = MISSION, plume: dict = SMOKE_PLUME,
                  every: int = SMOKE_TRACK_EVERY) -> dict:
    """Notebook 26b cells 18 and 20 without the figures: the winner ``FASTEST`` flies the mission through a 40 MW fire's
    smoke 20 km out (``mf.fly``, a kinematic flight at 0.5 s steps), and the fire's CO flux, plume height and heat
    release come back from its passes (``mf.source_estimate``) against the model's truth (cell 20's "ignition check").
    Returns the plume, the episode's summary and phase table (cell 24 hands off the summary and the estimate), the
    estimate, the check (both None without a pass through the smoke) and the track at every ``every``-th step."""
    import warnings
    from scipy.optimize import OptimizeWarning
    PLUME = mf.Plume(**plume)
    FOREST = mf.Forest()
    af, unit = MERLINS[FASTEST]["airframe"], MERLINS[FASTEST]["unit"]
    R20 = mf.race(af, unit, 20000.0, MISSION)
    EP = mf.fly(af, unit, FOREST, PLUME, race_result=R20, mission=MISSION)
    EST = check = None
    if EP.passes:                                                  # (not in the cell: no pass, nothing to fit)
        with warnings.catch_warnings():                            # three passes: the fit's covariance is undetermined
            warnings.simplefilter("ignore", OptimizeWarning)
            EST = mf.source_estimate(PLUME, EP.passes)             # cell 20
        check = {"peak CO [ppm]": EST["peak_ppm"], "CO flux [g/s]": EST["co_kg_s"] * 1000, "true CO flux [g/s]": PLUME.co_kg_s * 1000,
                 "plume height [m]": EST["plume_height_m"], "fire heat release [MW]": EST["heat_mw"], "true [MW]": PLUME.heat_mw,
                 "confirmed": EST["peak_ppm"] > 10 * 0.05}
    s = slice(None, None, every)
    return {"propulsor": FASTEST, "plume": dataclasses.asdict(PLUME), "summary": EP.summary(),
            "phases": EP.phase_table().reset_index().to_dict("records"), "source_estimate": EST, "ignition_check": check,
            "track": {"t_s": EP.t[s], "x_m": EP.pos[s, 0], "y_m": EP.pos[s, 1], "altitude_m": EP.pos[s, 2], "agl_m": EP.agl[s],
                      "airspeed_m_s": EP.airspeed[s], "power_w": EP.power[s], "energy_wh": EP.energy_wh[s], "ppm": EP.ppm[s]}}


def _parser():
    ap = parser("MERLIN: airframe, wing FEA, polar CFD, the propulsor race and the mission; writes merlin.vida and merlin_design.json")
    ap.add_argument("--merlin", action="append", default=[], metavar="NAME=VALUE",
                    help="a design parameter other than the default (repeatable), e.g. --merlin thickness=0.15")
    ap.add_argument("--propulsors", dest="propulsors_vida", type=Path, default=None,
                    help="the propulsor libraries' .vida (default assemblies/data/propulsors.vida)")
    return ap


def _run_cli(*, merlin=(), **kw):
    return run(merlin=parse_assignments(merlin), **kw)


if __name__ == "__main__":
    raise SystemExit(main(_run_cli, _parser(), DATA / f"{NAME}.vida"))
