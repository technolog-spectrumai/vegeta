"""FEA, vibration and CFD of the transonic carbon propeller and fan (notebooks/designs/transonic.py), with notebook 25's
subsonic propeller and EDF alongside for comparison. Each run writes ONE structured file:

    output/transonic/runs/<fea|vib|cfd>_<YYYYmmdd-HHMMSS>_<host>.yaml   (+ PNG charts next to it)

USAGE (from the repository root; the shell wrappers fea.sh, vib.sh, cfd.sh, all.sh do the same)
  python scripts/transonic/transonic_runs.py fea -j 8            # spin + air load, 100 % and 120 % overspeed
  python scripts/transonic/transonic_runs.py vib -j 8            # modes, Campbell diagram, resonance margins
  python scripts/transonic/transonic_runs.py cfd -j 8            # MRF CFD: prop at 3 speeds, fan at 3 rpm
  python scripts/transonic/transonic_runs.py cfd -j 8 --compressible    # rhoSimpleFoam (experimental, see below)
  ... --parts transonic_prop transonic_fan        only these (default: all four, old ones for comparison)
  ... --quality smoke|screening|fine              CFD mesh (smoke: proves the pipeline in minutes)
  ... --fresh                                     ignore solved cases (otherwise reused: Ctrl-C and rerun)

PARTS: transonic_prop, subsonic_prop_nb25, transonic_fan, subsonic_edf_nb25.

FEA (Talos / CalculiX): one blade with its hub (propellers) or the whole rotor (fans), fixed at the bore. Loads: the
centrifugal field at the design rpm and at 120 % overspeed, plus the air load (thrust and the torque's tangential force
at 0.7 R, per blade). Carbon fibre as an equivalent isotropic laminate (``transonic.CFRP``: E 75 GPa, allowable 600 MPa
along the span, 50 MPa interlaminar); the old parts in PA6-GF30 as notebook 25. Out: max von Mises and where (r/R),
safety factor, tip deflection, mass.

VIBRATION: the first modes at rest (CalculiX), the centrifugal stiffening by Southwell's rule f(n)^2 = f0^2 + K n^2
(K = 1.2 flap, 0.2 lag/torsion — estimates, set with --southwell), the Campbell diagram against the excitation orders
(propeller: 1..2B per rev — the airframe, the nose, gusts; fan: 1..4 inlet distortion and the stator count V and 2V,
the stators' potential field seen by the rotor), and the margin of every crossing near the operating range.

CFD (Aeromant, OpenFOAM v2412 openfoam.com): the rotor in an MRF zone, k-omega SST, steady.
  propellers: ``rotor_mrf`` (isolated) at 100, 140 and 166.7 m/s with the design's rpm;
  fans: ``rotor_mrf_installed`` (rotor + housing) at 0.85, 1.0 and 1.1 x design rpm, static and at 166.7 m/s.
  Default incompressible (simpleFoam): fine for the old parts, a known under-estimate of the losses at M 0.95-1.2.
  ``--compressible``: the same meshed case converted to rhoSimpleFoam (perfect gas, energy equation, transonic SIMPLE),
  forces read from the force function objects in newtons. EXPERIMENTAL: written for v2412, not run here.
Out: thrust, torque, power, efficiency, against the design model, residuals and convergence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))
import transonic as tr  # noqa: E402

OUT = ROOT / "output" / "transonic"
PARTS = ("transonic_prop", "subsonic_prop_nb25", "transonic_fan", "subsonic_edf_nb25")
PA6 = dict(name="PA6-GF30 (moulded)", E=8000.0, nu=0.35, rho=1.35e-9, strength=100.0, ilss=40.0, fatigue_1e7=30.0, damping=0.02)
QUALITY = {
    "smoke": dict(iterations=60, cells_per_diameter=5.0, surface_level=3, near_level=2, rotor_level=2, wake_level=1,
                  upstream=2.0, downstream=3.0, lateral=2.0, static_level=2, static_box_level=1, static_margin=1.0),
    "screening": dict(iterations=400, cells_per_diameter=8.0, surface_level=4, near_level=3, rotor_level=3, wake_level=2,
                      upstream=3.0, downstream=5.0, lateral=3.0, static_level=3, static_box_level=1, static_margin=1.5),
    "fine": dict(iterations=1000, cells_per_diameter=10.0, surface_level=5, near_level=4, rotor_level=4, wake_level=3,
                 upstream=3.0, downstream=6.0, lateral=3.0, static_level=4, static_box_level=2, static_margin=1.5),
}


# ------------------------------------------------------------------------------------------------- operating points
def operating_points(part: str) -> list[dict]:
    """(name, airspeed, rpm, thrust, torque) the runs use, from the design models."""
    if part == "transonic_prop":
        rows = tr.prop_sizing(tr.TPROP, speeds=(100.0, 140.0, tr.V_MAX))
        return [dict(name=f"{r['speed_kmh']}kmh", airspeed=r["speed_m_s"], rpm=r["rpm"], thrust_N=r["thrust_N"],
                     torque_Nm=r["torque_Nm"], power_W=r["shaft_kW"] * 1000, efficiency=r["efficiency"]) for r in rows]
    if part == "subsonic_prop_nb25":             # its own job: notebook 25's 3 kg aircraft, cruise 20 m/s, climb, top
        from vegeta import boreas
        pr = tr.boreas_prop(tr.OLD_PROP)
        sec = boreas.Airfoil(cl_alpha=2 * math.pi * 0.9, alpha0_deg=-3.0, cl_max=1.1, cd0=0.018, k=0.04)
        out = []
        for name, V, T in (("cruise_20ms", 20.0, 2.6), ("fast_30ms", 30.0, 5.0), ("top_40ms", 40.0, 8.0)):
            op = boreas.rpm_for_thrust(pr, sec, T, V, tr.RHO)
            out.append(dict(name=name, airspeed=V, rpm=round(op.rpm), thrust_N=T, torque_Nm=op.torque, power_W=op.power,
                            efficiency=op.efficiency))
        return out
    if part == "transonic_fan":
        d = tr.TFAN
        out = []
        for f in (0.85, 1.0, 1.1):
            for V in (0.0, tr.V_MAX):
                m = tr.fan_meanline(dict(d, rpm=d["rpm"] * f, pressure_ratio=1 + (d["pressure_ratio"] - 1) * f ** 2), V)
                om = d["rpm"] * f * 2 * math.pi / 60
                out.append(dict(name=f"{f:.2f}rpm_{V * 3.6:.0f}kmh", airspeed=V, rpm=round(d["rpm"] * f), thrust_N=m["net_thrust_N"],
                                torque_Nm=m["shaft_kW"] * 1000 / om, power_W=m["shaft_kW"] * 1000, efficiency=m["propulsive_efficiency"]))
        return out
    if part == "subsonic_edf_nb25":
        rpm0 = 33000.0                            # ~ notebook 25's cruise rpm for the 90 mm unit (its CFD sweep: 0.85/1.0/1.25)
        return [dict(name=f"{f:.2f}rpm_20ms", airspeed=20.0, rpm=round(rpm0 * f), thrust_N=None, torque_Nm=None, power_W=None,
                     efficiency=None) for f in (0.85, 1.0, 1.25)]
    raise ValueError(part)


def design_of(part):
    return {"transonic_prop": tr.TPROP, "subsonic_prop_nb25": tr.OLD_PROP, "transonic_fan": tr.TFAN, "subsonic_edf_nb25": tr.OLD_EDF}[part]


def material_of(part):
    return tr.CFRP if part.startswith("transonic") else PA6


def is_prop(part):
    return "prop" in part


# ------------------------------------------------------------------------------------------------------------- output
def stamp():
    return time.strftime("%Y%m%d-%H%M%S"), socket.gethostname().split(".")[0]


def write_yaml(kind: str, doc: dict) -> Path:
    t, host = stamp()
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    path = OUT / "runs" / f"{kind}_{t}_{host}.yaml"
    path.write_text(yaml.safe_dump(_plain(doc), sort_keys=False, width=140))
    print("wrote", path)
    return path


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def progress(it, total, desc):
    try:
        from tqdm import tqdm
        return tqdm(it, total=total, desc=desc, unit="case")
    except ImportError:
        return it


# ---------------------------------------------------------------------------------------------------------------- FEA
def _cad_step(part: str, work: Path) -> tuple[Path, dict]:
    """STEP of one blade + hub (propellers) or the whole rotor (fans), about z; geometry numbers for the regions."""
    d = design_of(part)
    if is_prop(part):
        g = tr.prop_cad(d, blades=1)
        hub_d, hub_h, R, bore = d["hub_diameter"], d["hub_height"], d["diameter_mm"] / 2, d["bore"]
    else:
        g, _ = tr.edf_cad(d)
        hub_d, hub_h, R, bore = d["hub_diameter_mm"], d["hub_height_mm"], d["diameter_mm"] / 2, 4.0
    step = g.export(work / "cad", formats=("step",)).artifacts["step"]
    return Path(step), dict(hub_d=hub_d, hub_h=hub_h, R=R, bore=bore, volume_mm3=g.volume)


def _model(part: str, work: Path, loads, name: str, element: float):
    from vegeta import talos
    step, geo = _cad_step(part, work)
    m_ = material_of(part)
    mat = talos.Material(m_["name"], youngs_modulus=m_["E"], poissons_ratio=m_["nu"], density=m_["rho"], yield_strength=m_["strength"],
                         source="transonic.CFRP" if m_ is tr.CFRP else "notebook 25 PA6-GF30")
    rb, R = geo["bore"] / 2 + 0.6, geo["R"]
    zr = max(geo["hub_h"], 30.0)
    regions = [talos.SurfacesInBox("bore", (-rb, -rb, -zr, rb, rb, zr)),
               talos.SurfacesInBox("blade", (-R - 1, -R - 1, -zr, R + 1, R + 1, zr))]
    model = talos.StructuralModel(str(step), "mm-N-MPa", mat, regions, [talos.FixedSupport("bore")], loads,
                                  talos.MeshSettings(element_size=element), name=name)
    return model, geo


def run_fea(args) -> Path:
    from vegeta import talos
    rows = {}
    for part in progress(args.parts, len(args.parts), "FEA"):
        ops = operating_points(part)
        top = max(ops, key=lambda o: o["rpm"])
        d = design_of(part)
        B = d["blades"]
        R = d["diameter_mm"] / 2
        T_blade = (top["thrust_N"] or 0.0) / B if is_prop(part) else (top["thrust_N"] or 0.0)     # fan: the whole rotor
        Q = top["torque_Nm"] or 0.0
        F_tan = (Q / B if is_prop(part) else Q) / (0.7 * R / 1000)
        out = {}
        for case, rpm_f in (("at_design_rpm", 1.0), ("overspeed_120", 1.2)):
            work = OUT / "fea" / part / case
            if args.fresh and work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True, exist_ok=True)
            loads = [talos.Centrifugal(top["rpm"] * rpm_f, point=(0, 0, 0), axis=(0, 0, 1))]
            if case == "at_design_rpm" and T_blade:
                loads.append(talos.Force("blade", fz=T_blade, fy=-F_tan))
            el = args.element or (1.0 if is_prop(part) else 0.6)
            model, geo = _model(part, work, loads, f"{part}_{case}", el)
            model.mesh(work)
            res = model.solve(work, threads=args.cpu)
            if not res.ok:
                raise RuntimeError(f"FEA {part} {case} failed: {res.messages[-3:]}")
            fr = talos.read_frd(res.artifacts["frd"])
            vm = talos.frd.von_mises(fr.stress)
            i = int(np.argmax(vm))
            disp = np.linalg.norm(fr.displacement, axis=1)
            mat = material_of(part)
            out[case] = dict(rpm=top["rpm"] * rpm_f, thrust_per_blade_N=T_blade if case == "at_design_rpm" else 0.0,
                             max_von_mises_MPa=float(vm[i]), at_r_over_R=float(math.hypot(*fr.coords[i][:2]) / R),
                             at_z_mm=float(fr.coords[i][2]), safety_factor=float(mat["strength"] / vm[i]),
                             max_deflection_mm=float(disp.max()), nodes=int(len(fr.node_ids)), frd=str(res.artifacts["frd"])[len(str(ROOT)) + 1:])
            print(f"  {part} {case}: {vm[i]:.1f} MPa at r/R {out[case]['at_r_over_R']:.2f}, SF {out[case]['safety_factor']:.2f}, "
                  f"deflection {disp.max():.3f} mm")
        rows[part] = dict(material=material_of(part), design=d, operating_point=top, mass_g=geo["volume_mm3"] * material_of(part)["rho"] * 1e6,
                          tip_speed_m_s=top["rpm"] * 2 * math.pi / 60 * R / 1000, **out)
    path = write_yaml("fea", dict(kind="fea", run=_run_info(args), parts=rows,
                                  notes="equivalent isotropic laminate; ply-level (Tsai-Wu) and root-joint checks are not in this model"))
    _chart_fea(path, rows)
    return path


def _chart_fea(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(rows)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, (key, lab) in enumerate((("max_von_mises_MPa", "max von Mises [MPa]"), ("safety_factor", "safety factor"),
                                    ("max_deflection_mm", "max deflection [mm]"))):
        for j, case in enumerate(("at_design_rpm", "overspeed_120")):
            ax[k].bar(np.arange(len(names)) + 0.38 * j, [rows[n][case][key] for n in names], 0.38, label=case)
        ax[k].set_xticks(np.arange(len(names)) + 0.19, names, rotation=20, ha="right", fontsize=8)
        ax[k].set_title(lab, loc="left"); ax[k].grid(alpha=0.3)
    ax[1].axhline(1.5, color="#d03b3b", ls="--", lw=1, label="SF 1.5")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path.with_suffix("").as_posix() + "_fea.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------------------------------------- vibration
def run_vib(args) -> Path:
    from vegeta import talos
    rows = {}
    for part in progress(args.parts, len(args.parts), "modes"):
        work = OUT / "vib" / part
        if args.fresh and work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        d = design_of(part)
        model, geo = _model(part, work, [talos.Centrifugal(1.0, point=(0, 0, 0), axis=(0, 0, 1))], f"{part}_modes",
                            args.element or (1.0 if is_prop(part) else 0.6))
        model.mesh(work)
        modes = model.solve_modes(work, n_modes=args.modes, threads=args.cpu)
        if not modes.ok:
            raise RuntimeError(f"modes {part} failed: {modes.messages[-3:]}")
        f0 = [float(f) for f in _mode_freqs(modes)]
        ops = operating_points(part)
        n_lo, n_hi = min(o["rpm"] for o in ops) / 60, max(o["rpm"] for o in ops) / 60
        B = d["blades"]
        orders = list(range(1, 2 * B + 1)) if is_prop(part) else sorted({1, 2, 3, 4, d["stator_vanes"], 2 * d["stator_vanes"]})
        K = [args.southwell[0] if k == 0 else args.southwell[1] for k in range(len(f0))]      # 1st mode flap, the rest lag/torsion
        n_axis = np.linspace(0, 1.25 * n_hi, 400)
        campbell = {f"mode{k + 1}": np.sqrt(np.asarray(f0[k]) ** 2 + K[k] * n_axis ** 2).tolist() for k in range(len(f0))}
        crossings = []
        for k in range(len(f0)):
            for E in orders:                                         # f0^2 + K n^2 = (E n)^2
                if E * E > K[k]:
                    n_x = f0[k] / math.sqrt(E * E - K[k])
                    margin = min(abs(n_x - n_lo), abs(n_x - n_hi)) / n_hi if not (n_lo <= n_x <= n_hi) else 0.0
                    crossings.append(dict(mode=k + 1, order=E, rpm=n_x * 60, in_operating_range=bool(n_lo <= n_x <= n_hi),
                                          margin_of_top_rpm=margin))
        risky = [c for c in crossings if c["in_operating_range"] or c["margin_of_top_rpm"] < 0.10]
        rows[part] = dict(material=material_of(part), southwell_K=K, modes_at_rest_hz=f0, operating_rpm=[n_lo * 60, n_hi * 60],
                          excitation_orders=orders, crossings=crossings, risky_crossings=risky,
                          verdict="clear (no crossing within 10 % of the operating range)" if not risky else f"{len(risky)} crossing(s) to look at",
                          campbell={"rpm": (n_axis * 60).tolist(), **campbell})
        print(f"  {part}: modes {[round(f) for f in f0]} Hz, {rows[part]['verdict']}")
    path = write_yaml("vib", dict(kind="vibration", run=_run_info(args), parts=rows,
                                  notes="Southwell coefficients are estimates; a prestressed modal (spin-softened) analysis is the check"))
    _chart_campbell(path, rows)
    return path


def _mode_freqs(modes):
    for attr in ("frequencies_hz", "frequencies", "freq_hz"):
        if hasattr(modes, attr):
            return list(getattr(modes, attr))
    if hasattr(modes, "metrics"):
        for k in ("frequencies_hz", "frequencies"):
            if k in modes.metrics:
                return list(modes.metrics[k])
    if isinstance(modes, dict):
        return list(modes.get("frequencies_hz") or modes["frequencies"])
    raise AttributeError(f"cannot read the mode frequencies from {type(modes).__name__}: {modes!r}"[:300])


def _chart_campbell(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 4.4), squeeze=False)
    for ax, (part, r) in zip(axs[0], rows.items()):
        rpm = np.array(r["campbell"]["rpm"])
        for k in range(len(r["modes_at_rest_hz"])):
            ax.plot(rpm, r["campbell"][f"mode{k + 1}"], lw=1.6, label=f"mode {k + 1}")
        top = max(max(r["campbell"][f"mode{k + 1}"]) for k in range(len(r["modes_at_rest_hz"])))
        for E in r["excitation_orders"]:
            ax.plot(rpm, E * rpm / 60, color="#9a9a9a", lw=0.7, ls="--")
            ax.text(rpm[-1], min(E * rpm[-1] / 60, top), f"{E}E", fontsize=7, color="#6a6a6a")
        ax.axvspan(*r["operating_rpm"], color="#f2c14e", alpha=0.25, label="operating")
        for c in r["risky_crossings"]:
            ax.plot(c["rpm"], c["order"] * c["rpm"] / 60, "o", color="#d03b3b")
        ax.set_ylim(0, top * 1.05); ax.set_title(part, loc="left"); ax.set_xlabel("rpm"); ax.set_ylabel("Hz"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path.with_suffix("").as_posix() + "_campbell.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- CFD
def _stl_axis_x(part: str, work: Path) -> tuple[Path, Path | None]:
    """Rotor STL about +x (Aeromant's rotor templates), and the fan housing (already about x)."""
    from vegeta import dedalus
    d = design_of(part)
    work.mkdir(parents=True, exist_ok=True)
    if is_prop(part):
        g = tr.prop_cad(d)
        house = None
    else:
        g, h = tr.edf_cad(d)
        house = Path(h.export(work / "housing", formats=("stl",), stl_tolerance=0.05).artifacts["stl"])
    gx = dedalus.Geometry.from_cadquery(g.shape.rotate((0, 0, 0), (0, 1, 0), 90), name=f"{part}_axis_x")
    return Path(gx.export_stl(str(work / f"{part}_axis_x.stl"), tolerance=0.02)), house


def run_cfd(args) -> Path:
    from vegeta import aeromant
    try:
        env = aeromant.OpenFOAMEnvironment.detect()
    except RuntimeError:
        if not args.prepare_only:
            sys.exit("OpenFOAM not found: source its bashrc (v2412, openfoam.com), or use --prepare-only")
        env = aeromant.OpenFOAMEnvironment()
    pre = QUALITY[args.quality]
    rows, jobs = {}, []
    for part in args.parts:
        d = design_of(part)
        D = d["diameter_mm"] / 1000
        cad = OUT / "cfd" / "cad"
        stl, house = _stl_axis_x(part, cad)
        for op in operating_points(part):
            wd = OUT / "cfd" / f"{part}_{op['name']}_{args.quality}{'_rho' if args.compressible else ''}"
            par = dict(rpm=op["rpm"], airspeed=max(op["airspeed"], 0.5), diameter=D, kinematic_viscosity=1.5e-5, density=tr.RHO, rotation=1,
                       **{k: v for k, v in pre.items() if not k.startswith("static")})
            if house is None:
                case = aeromant.CFDCase("rotor_mrf", str(stl), par, workdir=str(wd), geometry_units="mm", environment=env)
            else:
                par.update({k: v for k, v in pre.items() if k.startswith("static")},
                           zone_radius=(d["diameter_mm"] / 2 + 0.5 * d["tip_clearance_mm"]) / d["diameter_mm"], zone_length=0.36)
                case = aeromant.CFDCase("rotor_mrf_installed", str(stl), par, workdir=str(wd), geometry_units="mm", environment=env,
                                        static_geometry=str(house))
            jobs.append((part, op, case, wd))
    for part, op, case, wd in progress(jobs, len(jobs), "CFD"):
        key = f"{part}/{op['name']}"
        done = (wd / "postProcessing").exists() and (wd / "log.solver").exists() and not args.fresh
        if not done:
            r = case.prepare(overwrite=True)
            if not r.ok:
                rows[key] = dict(error="prepare failed", messages=r.messages[-3:]); continue
            if args.compressible:
                to_compressible(wd, op["airspeed"])
        if args.prepare_only:
            rows[key] = dict(operating_point=op, prepared=str(wd)); continue
        t0 = time.time()
        if not done:
            if args.compressible:
                steps = [s.name for s in case.pipeline(args.cpu) if s.name != "solver"]
                res = case.run(steps=[s for s in steps if s != "reconstruct"], processors=args.cpu)
                if res.ok:
                    _run_rho(case, wd, args.cpu, env)
            else:
                res = case.run(processors=args.cpu)
        m = read_forces(wd, op["rpm"], op["airspeed"], compressible=args.compressible)
        if not args.compressible:
            try:
                r = case.results()
                m.update({k: v for k, v in r.metrics.items() if isinstance(v, (int, float, bool, str))})
            except Exception as e:                                       # noqa: BLE001
                m["results_error"] = str(e)
        rows[key] = dict(part=part, operating_point=op, minutes=round((time.time() - t0) / 60, 2), cfd=m,
                         thrust_ratio_cfd_to_model=(m.get("thrust_N") / op["thrust_N"]) if m.get("thrust_N") and op.get("thrust_N") else None)
        print(f"  {key}: thrust {m.get('thrust_N')}, power {m.get('power_W')}")
    path = write_yaml("cfd", dict(kind="cfd", run=_run_info(args), solver="rhoSimpleFoam (compressible)" if args.compressible else "simpleFoam MRF",
                                  quality=args.quality, cases=rows))
    return path


def read_forces(wd: Path, rpm: float, V: float, compressible: bool) -> dict:
    """Rotor thrust (+x force on the body patches, reversed: the force on the fluid's reaction), torque about x and power
    from postProcessing/forces (last 50 iterations averaged). Incompressible files are kinematic x rhoInf already."""
    base = wd / "postProcessing" / "forces"
    fs, ms = sorted(base.glob("*/force.dat")), sorted(base.glob("*/moment.dat"))
    if not fs:
        return {"error": "no postProcessing/forces"}
    F, M = _dat(fs[-1]), _dat(ms[-1]) if ms else None
    fx = float(np.mean(F[-50:, 1]))
    mx = float(np.mean(M[-50:, 1])) if M is not None else float("nan")
    om = rpm * 2 * math.pi / 60
    T, Q = -fx, abs(mx)                                    # the rotor pushes the air back (-x force on the blades = +thrust)
    out = dict(thrust_N=T, torque_Nm=Q, power_W=Q * om, efficiency=(T * V / (Q * om)) if V > 1 and Q > 0 else None,
               iterations=int(F[-1, 0]), force_scatter_last50=float(np.std(F[-50:, 1])))
    sf = wd / "postProcessing" / "staticForces"
    if sf.exists() and sorted(sf.glob("*/force.dat")):
        S = _dat(sorted(sf.glob("*/force.dat"))[-1])
        out["housing_force_x_N"] = float(np.mean(S[-50:, 1]))
        out["net_thrust_N"] = T - out["housing_force_x_N"]
    return out


def _dat(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        nums = re.findall(r"[-+0-9.eE]+", line.replace("(", " ").replace(")", " "))
        rows.append([float(x) for x in nums[:4]])
    return np.array(rows)


# --- incompressible case -> rhoSimpleFoam --------------------------------------------------------------------------
_HDR = "FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}\n"


def to_compressible(wd: Path, V: float, T_inf: float = tr.T0, p_inf: float = tr.P0) -> None:
    """Turn an Aeromant rotor case (openfoam.com dialect) into rhoSimpleFoam: absolute pressure, temperature, a perfect
    gas with Sutherland viscosity, compressible schemes and a transonic SIMPLE. The mesh steps are unchanged."""
    s, c, z = wd / "system", wd / "constant", wd / "0.orig"
    cd = (s / "controlDict").read_text()
    cd = re.sub(r"application\s+\w+;", "application     rhoSimpleFoam;", cd)
    cd = re.sub(r"rho\s+rhoInf;\s*\n\s*rhoInf\s+[^;]+;", "rho             rho;", cd)
    (s / "controlDict").write_text(cd)
    (c / "thermophysicalProperties").write_text(_HDR.format(cls="dictionary", obj="thermophysicalProperties") + """
thermoType { type hePsiThermo; mixture pureMixture; transport sutherland; thermo hConst; equationOfState perfectGas;
             specie specie; energy sensibleInternalEnergy; }
mixture { specie { molWeight 28.96; } thermodynamics { Cp 1005; Hf 0; } transport { As 1.458e-06; Ts 110.4; Pr 0.71; } }
""")
    p = (z / "p").read_text()
    p = p.replace("[0 2 -2 0 0 0 0]", "[1 -1 -2 0 0 0 0]")
    p = re.sub(r"uniform\s+0\b", f"uniform {p_inf}", p)
    (z / "p").write_text(p)
    walls = '"(body|static).*"'
    (z / "T").write_text(_HDR.format(cls="volScalarField", obj="T") + f"""
dimensions [0 0 0 1 0 0 0];
internalField uniform {T_inf};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {T_inf}; }}
    outlet {{ type inletOutlet; inletValue uniform {T_inf}; value uniform {T_inf}; }}
    sides {{ type zeroGradient; }}
    {walls} {{ type zeroGradient; }}
    #includeEtc "caseDicts/setConstraintTypes"
}}
""")
    (z / "alphat").write_text(_HDR.format(cls="volScalarField", obj="alphat") + f"""
dimensions [1 -1 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet {{ type calculated; value uniform 0; }}
    outlet {{ type calculated; value uniform 0; }}
    sides {{ type calculated; value uniform 0; }}
    {walls} {{ type compressible::alphatWallFunction; Prt 0.85; value uniform 0; }}
    #includeEtc "caseDicts/setConstraintTypes"
}}
""")
    (s / "fvSchemes").write_text(_HDR.format(cls="dictionary", obj="fvSchemes") + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; limited cellLimited Gauss linear 1; grad(U) $limited; grad(k) $limited; grad(omega) $limited; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind limited;
    div(phi,e) bounded Gauss linearUpwind limited;
    div(phi,K) bounded Gauss linear;
    div(phi,Ekp) bounded Gauss linear;
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
    div(phid,p) bounded Gauss upwind;
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default limited corrected 0.5; }
wallDist { method meshWave; }
""")
    fs = (s / "fvSolution").read_text()
    m = re.search(r"residualControl\s*\{[^}]*?p\s+([0-9.eE+-]+);", fs)
    tol = m.group(1) if m else "1e-4"
    (s / "fvSolution").write_text(_HDR.format(cls="dictionary", obj="fvSolution") + f"""
solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; }}
    "(U|e|k|omega)" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 1;
    transonic yes;
    consistent yes;
    pMinFactor 0.3;
    pMaxFactor 3.0;
    residualControl {{ p {tol}; "(U|e|k|omega)" {tol}; }}
}}
relaxationFactors
{{
    fields {{ p 0.3; rho 0.05; }}
    equations {{ U 0.6; e 0.6; ".*" 0.6; }}
}}
""")


def _run_rho(case, wd: Path, procs: int, env):
    from vegeta.aeromant._process import run_command
    shutil.copytree(wd / "0.orig", wd / "0", dirs_exist_ok=True)
    if procs > 1:
        argv_list = [["decomposePar", "-force"], ["mpirun", "-np", str(procs), "rhoSimpleFoam", "-parallel"], ["reconstructPar", "-latestTime"]]
    else:
        argv_list = [["rhoSimpleFoam"]]
    for argv in argv_list:
        log = wd / f"log.{'solver' if 'rhoSimpleFoam' in argv else argv[0]}"
        rec = run_command(env.command(argv), wd, env=env.env, log_path=log)
        if rec.returncode != 0:
            raise RuntimeError(f"{' '.join(argv)} failed in {wd}: see {log}")


# --------------------------------------------------------------------------------------------------------------- main
def _run_info(args):
    t, host = stamp()
    return dict(started=t, host=host, cpu=args.cpu, parts=list(args.parts), argv=sys.argv[1:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("fea", "vib", "cfd", "all"))
    ap.add_argument("-j", "--cpu", type=int, default=os.cpu_count() or 4, help="cores (CalculiX threads / MPI processes)")
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--element", type=float, default=None, help="FEA element size [mm] (default 1.0 props, 0.6 fans)")
    ap.add_argument("--modes", type=int, default=6)
    ap.add_argument("--southwell", type=float, nargs=2, default=(1.2, 0.2), metavar=("K_FLAP", "K_OTHER"))
    ap.add_argument("--quality", default="screening", choices=tuple(QUALITY))
    ap.add_argument("--compressible", action="store_true", help="CFD with rhoSimpleFoam (experimental)")
    ap.add_argument("--prepare-only", action="store_true", help="CFD: write and mesh-ready the cases, do not run")
    args = ap.parse_args(argv)
    if args.what in ("fea", "all"):
        run_fea(args)
    if args.what in ("vib", "all"):
        run_vib(args)
    if args.what in ("cfd", "all"):
        run_cfd(args)


if __name__ == "__main__":
    main()
