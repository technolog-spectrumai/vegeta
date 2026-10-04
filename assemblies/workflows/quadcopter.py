"""The quadcopter: the printed X frame, its propellers and drive, from notebook 08 (Parts 1 and 2).

The tree::

    quadcopter (multirotor)        the mass budget, the drive; results: hover / cruise / full-throttle points, endurance,
                                   the map over rpm x airspeed (exported like notebook 08's quad_5x43.json); the
                                   throttle sweep (cell 55) and the noise table and tone spectra (cell 60)
      frame (quad_frame)           the frame CAD; volume and mass
      frame_fea (frame_fea)        full thrust on all motors, a hard landing on one pad (Talos)
      canopy_cfd (frame_canopy)    the frame with a canopy at 15 m/s (Aeromant rans_ksst_external)
      propeller (propeller)        the catalogue propeller's CAD
      rotor_cfd (rotor)            the propeller in hover at the hover rpm (Aeromant rotor_mrf_static)
      blade_fea (blade)            one blade at full throttle (Talos)
      life (life)                  the frame's vibration and life (notebook 08 Part 3)
        unit_fea (unit_fea)        motors and stack as point masses: 8 modes; unit cases thrust, unbalance, landing (Talos)
        fatigue (fatigue)          margins to 1P, three missions -> spectra -> damage, static re-check, life under the
                                   usage mix, balanced propellers and other mixes (chronos, talos.assess_fatigue)

The frame is the design's defaults unless ``frame`` overrides some (notebook 08 compared three and preferred one: give
its parameters, e.g. ``--frame arm_width=16 --frame taper=0.55``). The life is exported to quadcopter_life.json (the keys
of 08's life.json).

    python -m assemblies.workflows.quadcopter --fidelity quick -j 4
    python -m assemblies.workflows.quadcopter --no-cfd --no-fea --show
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from vegeta import boreas, talos

from .. import DATA, RUNS, vida, results
from .._cli import main, parser
from .._cli_util import parse_assignments
from ._common import add_after, run_cfd as _cfd_node, solve_fea, sub, unit_fea
from ..components import life as lf, propeller as pr, quad_frame_analysis as qa, quad_life as ql
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
THROTTLES = np.linspace(0.2, 1.0, 17)                                                       # cell 55
NOISE_DIST, NOISE_ANGLE = 1.0, 90.0                                                         # cell 60: 1 m broadside
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


def throttle_sweep(system, throttles=THROTTLES) -> dict:
    """Notebook 08 cell 55 (the plots left out): the motor curve meets the propeller's torque curve at each throttle,
    static; current above the motor's limit flagged. The cell's ``sw`` table as columns."""
    sweep = system.sweep(throttles, airspeed=0.0)
    sw = [{"throttle": s.throttle, "rpm": s.rpm, "thrust_N": s.thrust, "current_A": s.current,
           "electrical_W": s.electrical_power, "motor_eff": s.motor_efficiency, "current_limited": s.current_limited} for s in sweep]
    return {k: [bool(r[k]) if k == "current_limited" else float(r[k]) for r in sw] for k in sw[0]}


def noise_table(prop, pts, motors: int = DRIVE["motors"]) -> tuple[dict, dict]:
    """Notebook 08 cell 60 (the plot left out): per operating point one rotor's Gutin tones and broadband allowance at
    1 m broadside (``propeller.noise``, the cell's loop body) and the machine's level at 1, 10 and 50 m, in the cell's
    columns; and each point's tone spectrum (``frequency_hz``, ``spl_db``; the cell plots hover's)."""
    DIST, ANGLE, MOTORS = NOISE_DIST, NOISE_ANGLE, motors
    rotor = pr.noise(prop, pts, medium=boreas.AIR, distance=DIST, angle_deg=ANGLE, harmonics=6)
    noise_rows, tones = {}, {}
    for name, n in rotor.items():
        one = n["one_rotor_dB"]
        noise_rows[name] = {"rpm": n["rpm"], "BPF_hz": n["BPF_hz"], "tonal_dB": n["tonal_dB"], "broadband_dB": n["broadband_dB"],
                            "one_rotor_dB_at_1m": one, f"{MOTORS}_rotors_dB_at_1m": one + 10 * math.log10(MOTORS),
                            f"{MOTORS}_rotors_dB_at_10m": one + 10 * math.log10(MOTORS) - 20,
                            f"{MOTORS}_rotors_dB_at_50m": one + 10 * math.log10(MOTORS) - 20 * math.log10(50)}
        tones[name] = {"frequency_hz": n["frequency_hz"], "spl_db": n["spl_db"]}
    return noise_rows, tones


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
        if path not in ("rotor_cfd", "blade_fea", "life", "unit_fea", "fatigue"):   # added later, after the drive is known
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
    sweep = throttle_sweep(system)                                # cell 55 (about 4 s)
    noise, noise_tones = noise_table(prop, pts)                   # cell 60

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

    # the frame's vibration and life (notebook 08 Part 3)
    life_sum = frame_life(root, prior, redo, files["step"], fp, prop, pts, fidelity=fidelity, out=out / "life", run_fea=run_fea,
                          threads=threads, progress=progress)

    bemt_hover = pts["hover"].thrust
    cfd_hover = (rc.results.get("forces") or {}).get("thrust_N")
    root.record(auw_kg=auw_kg, points={k: _point(v) for k, v in pts.items()}, hover_endurance_min=hover_min,
                thrust_to_weight=DRIVE["motors"] * pts["full"].thrust / (auw_kg * G),
                map={"rpm": np.asarray(grid["rpm"], float), "airspeed_m_s": np.asarray(grid["airspeed_m_s"], float),
                     "thrust_N": np.asarray(grid["thrust_n"], float)},
                hover_thrust_bemt_vs_cfd={"bemt_N": float(bemt_hover), "cfd_N": cfd_hover},
                throttle_sweep=sweep, noise=noise, noise_tones=noise_tones)
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:                                                    # plain data for notebooks (assemblies.results)
        root.meta["results_to"] = str(results.write(root, vida_path))
    if export:
        path = Path(export_path or DATA / f"{NAME}.json")
        res = boreas.export(path, prop, sec, map=grid, motor=pr.motor(DRIVE["motor"]), battery=battery,
                            points={"hover": pts["hover"], "cruise": pts["cruise"], "full": pts["full"]},
                            notes=f"quadcopter from assemblies.workflows.{NAME} ({fidelity}); AUW {auw_kg:.3f} kg, "
                                  f"{DRIVE['motors']} motors; hover endurance {hover_min:.1f} min")
        res.raise_for_status()
        root.meta["exported_to"] = str(path)
        if life_sum is not None:
            lpath = path.with_name(f"{path.stem}_life.json")
            lpath.write_text(__import__("json").dumps(dict(life_sum, source=f"assemblies.workflows.{NAME} ({fidelity})",
                                                           written_at=vida.utc_now()), indent=2, default=float))
            root.meta["exported_to"] += f", {lpath}"
    return root


def frame_life(root, prior, redo, step, fp, prop, pts, *, fidelity, out, run_fea, threads, progress):
    """Notebook 08 Part 3 as two nodes under ``life``; returns the life.json summary (cell 106) when computed."""
    motor = pr.motor(DRIVE["motor"])
    PROP = ql.prop_points(prop, pts)
    mp_t, stack_t = ql.masses_t(sum(PARTS_G.values()), motor.mass_kg, prop.mass_kg, DRIVE["motors"])
    life = add_after(root, Assembly("life", "life"), prior, redo)
    prior_life = sub(prior, "life")
    uf = add_after(life, Assembly("unit_fea", "unit_fea", params={
        "frame": fp, "material": qa.PETG_CF, "masses_t": {"motor_prop": mp_t, "stack": stack_t},
        "max_thrust_N": MAX_THRUST_PER_MOTOR_N, "element_mm": ql.ELEMENT_MM[fidelity], "n_modes": ql.N_MODES}), prior_life, redo)
    base, unit = ql.models(step, fp, mp_t, stack_t, MAX_THRUST_PER_MOTOR_N, element_mm=uf.params["element_mm"])
    unit_cases = unit_fea(uf, base, unit, out, n_modes=ql.N_MODES, run=run_fea, threads=threads, progress=progress)
    fat = add_after(life, Assembly("fatigue", "fatigue", params={
        "prop": PROP, "max_thrust_N": MAX_THRUST_PER_MOTOR_N, "curve": dict(ql.CURVE.__dict__), "usage": ql.USAGE,
        "n_flights": ql.N_FLIGHTS, "mixes": ql.MIXES, "balance_factor": ql.BALANCE_FACTOR, "unit_fea": uf.key}), prior_life, redo)
    if not fat.results:
        if not unit_cases:
            fat.not_run("needs the modes and unit cases (life/unit_fea)")
        else:
            st = lf.structure(uf.results["modes_hz"], "Talos modal, PETG-CF nominal E, lumped masses")
            lines = {k: v["rpm"] / 60 for k, v in PROP.items()}
            missions = ql.missions(PROP, MAX_THRUST_PER_MOTOR_N)
            spectra = lf.spectra(missions, st)
            fatigue, table = lf.fatigue(unit_cases, spectra, missions, ql.CURVE, workdir=out, progress=progress)
            worst = lf.worst(table)
            damage = {k: table[k]["damage_per_mission"] for k in table}
            hours = {k: m.duration_h for k, m in missions.items()}
            sim = lf.usage_life(damage, hours, ql.USAGE, ql.N_FLIGHTS, seed=0)
            balanced = {k: talos.assess_fatigue(unit_cases, lf.balanced(sp, ql.BALANCE_FACTOR).to_dict(), ql.CURVE)
                        .result.metrics["damage_per_pass"] for k, sp in spectra.items()}
            mixes = {name: {"hours, G 6.3 props": lf.life_hours(damage, hours, mix),
                            "hours, balanced G 2.5": lf.life_hours(balanced, hours, mix)} for name, mix in ql.MIXES.items()}
            fat.record(margins=lf.margins(st, lines), excitations_hz=lines, missions={k: m.describe() for k, m in missions.items()},
                       spectra={k: sp.to_dict() for k, sp in spectra.items()}, life=table, worst=worst,
                       contributions=dict(fatigue[worst].contributions),
                       static_recheck=lf.static_recheck(fatigue[worst], unit_cases, spectra, qa.PETG_CF["yield_strength"]),
                       damage_per_mission=damage, hours_per_mission=hours, usage=ql.USAGE,
                       hours_to_failure=sim.hours_to_failure, flights_to_failure=sim.flights_to_failure,
                       damage_balanced=balanced, mixes=mixes)
    if not fat.results:
        return None
    r = fat.results
    return {"design": {"file": "assemblies/components/quad_frame.py", "parameters": fp}, "modes_hz": uf.results["modes_hz"],
            "damping_ratio": lf.DAMPING, "excitations_hz": r["excitations_hz"],
            "unit_cases": {k: {"load": v["load"], "max_von_mises": v["max_von_mises_MPa"]} for k, v in uf.results["unit"].items()},
            "curve": fat.params["curve"], "missions": r["missions"], "damage_per_mission": r["damage_per_mission"],
            "hours_per_mission": r["hours_per_mission"], "usage": r["usage"], "hours_to_failure": r["hours_to_failure"],
            "flights_to_failure": r["flights_to_failure"]}


def _parser():
    ap = parser("The quadcopter: frame FEA and canopy CFD, propeller drive, rotor CFD, blade FEA; writes quadcopter.vida/.json")
    ap.add_argument("--frame", action="append", default=[], type=str, metavar="NAME=VALUE",
                    help="a frame parameter other than the default (repeatable), e.g. --frame arm_width=16")
    return ap


def _run_cli(*, frame=(), **kw):
    return run(frame=parse_assignments(frame), **kw)


if __name__ == "__main__":
    raise SystemExit(main(_run_cli, _parser(), DATA / f"{NAME}.vida"))
