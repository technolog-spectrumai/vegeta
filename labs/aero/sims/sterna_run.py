"""STERNA and Talassa — every calculation from one script, one progress bar. RUN BY YOU.

    labs/aero/sims/sterna_run.sh --list                      # what would run; nothing is meshed or solved
    labs/aero/sims/sterna_run.sh                             # everything, both aircraft
    labs/aero/sims/sterna_run.sh --cfd                       # only the CFD
    labs/aero/sims/sterna_run.sh --stability --flight        # only those two (no solver: seconds)
    labs/aero/sims/sterna_run.sh --fea --variant talassa    # one aircraft
    labs/aero/sims/sterna_run.sh --cfd --quality fine --processors 8

The steps (a flag each; with no step flag, all of them):

- ``--cad``        the aircraft and the joiner as STEP (the STL goes to _runs/sterna/cad/)        (CadQuery; seconds)
- ``--export``     the Yax exchange file ``<aircraft>.yx2`` with its description: mass, inertia, the CAD mesh, coefficients
- ``--cfd``        the airframe without propeller at -2, 1, 4 and 8 degrees, 14 m/s           (OpenFOAM; ~1 min a case, screening)
- ``--fea``        the wing joiner tube at 6 g and the design mass                             (Gmsh + CalculiX; seconds)
- ``--stability``  neutral point and margin (lattice, and the CFD's when it has run), trim, modes, the CG of the loadings
- ``--flight``     the mission in MuJoCo, without the video: launch, climb, orbit, glide, landing

Results: ``labs/aero/sims/output/<aircraft>_<step>.yaml``, written after every case: stop and restart freely, a solved
CFD case is kept (``--fresh`` starts over). The CAD and the ``.yx2`` files go to ``labs/aero/designs/data/sterna/``. ``FOAM_PREFIX`` names an
OpenFOAM conda prefix that is not found by itself.

STERNA flies its standard loading (the daylight pallet, one battery module); Talassa 500 g and three modules.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
import sterna as st  # noqa: E402
import sterna_cfd as sc  # noqa: E402
import sterna_flight as am  # noqa: E402
import talassa as sl  # noqa: E402
import sterna_mass as sm  # noqa: E402
import sterna_model as mo  # noqa: E402

OUT = ROOT / "labs" / "aero" / "sims" / "output"
RUNS = ROOT / "labs" / "aero" / "_runs" / "sterna"
DATA = ROOT / "labs" / "aero" / "designs" / "data" / "sterna"
STEPS = ("cad", "export", "cfd", "fea", "stability", "flight")
CL_MAX_2D = sl.CL_MAX_2D
N_LIMIT = 6.0
TUBE_ALLOWABLE = 600.0             # MPa: a hand allowable for a wrapped carbon tube in bending (not tested)


def variants() -> dict:
    """name -> parameters, standard loading, drive, the other loadings for the CG table, how it is launched."""
    L = sl.parameters()
    return {"sterna": {"title": "STERNA", "over": {}, "loading": "daylight_1", "drive": {}, "launch": "hand",
                       "loadings": list(sm.CASES)},
            "talassa": {"title": "Talassa", "over": L, "loading": sl.case(500.0, 3, L), "drive": sl.DRIVE, "launch": "line",
                         "loadings": [sl.case(g, n, L) for g, n in ((120.0, 2), (500.0, 2), (250.0, 3), (500.0, 3), (120.0, 4))]}}


def r(x, n=4):
    return None if x is None else round(float(x), n)


def load(path):
    return yaml.safe_load(path.read_text()) if path.exists() else None


def save(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    tmp.replace(path)


def coefficients(v: dict, loading=None, cfd_rows=None):
    C = mo.coefficients(v["loading"] if loading is None else loading, cl_max_2d=CL_MAX_2D, **v["over"])
    if cfd_rows:
        C = mo.calibrate(C, cfd_rows)
    return C


def cfd_fit_rows(name: str):
    """The solved CFD cases of the attached range: this script's own, else (STERNA) the notebook's kept rows."""
    doc = load(OUT / f"{name}_cfd.yaml")
    rows = [c for c in (doc or {}).get("cases", {}).values() if c.get("status") == "solved" and c["alpha_deg"] <= sc.FIT_MAX_ALPHA]
    if len(rows) < 3 and name == "sterna":
        kept = load(ROOT / "labs" / "aero" / "designs" / "data" / "sterna_cfd_screening.yaml")
        rows = [c for c in (kept or {}).get("rows", []) if c.get("status") in ("solved", "success") and c["alpha_deg"] <= sc.FIT_MAX_ALPHA]
    return rows if len(rows) >= 3 else None


# ------------------------------------------------------------------------------------------------------------ steps
def do_cad(name, v, a):
    DATA.mkdir(parents=True, exist_ok=True)
    g = st.Sterna().generate(**v["over"])
    stem = name.replace("-", "_")
    (RUNS / "cad").mkdir(parents=True, exist_ok=True)
    g.export_step(str(DATA / f"{stem}.step")); g.export_stl(str(RUNS / "cad" / f"{stem}.stl"), tolerance=0.2)      # the STL (large) stays in _runs/
    st.Sterna().generate(**v["over"], part="joiner").export_step(str(DATA / f"{stem}_joiner.step"))
    m = g.measure()
    return f"{m['dimensions'][0]:.0f} x {m['dimensions'][1]:.0f} x {m['dimensions'][2]:.0f} mm, {m['n_solids']} solid"


def do_export(name, v, a):
    import sterna_export
    res = sterna_export.export_yx2(name, v, DATA / f"{name.replace('-', '_')}.yx2", cfd_rows=cfd_fit_rows(name), cl_max_2d=CL_MAX_2D)
    return f"{res['file']}: {res['mass_kg']:.2f} kg, {res['triangles']} triangles, {res['bytes'] / 1e3:.0f} kB"


def do_cfd_case(name, v, a, alpha):
    path = OUT / f"{name}_cfd.yaml"
    doc = (None if a.fresh_now.pop(path, False) else load(path)) or {}
    if doc.get("quality") != a.quality:
        doc = {"aircraft": v["title"], "speed_m_s": sc.SPEED, "quality": a.quality, "template": "rans_ksst_external",
               "note": "airframe without propeller; Cm about the design CG on the mean chord", "cases": {}}
    key = f"a{alpha:+.0f}"
    if doc["cases"].get(key, {}).get("status") == "solved":
        return "kept"
    t0 = time.time()
    try:
        case = sc.case(RUNS / "cfd", alpha_deg=alpha, quality=a.quality, over=v["over"], loading=v["loading"], tag=name.replace("-", "_"))
        res = case.ensure(run=True, processors=a.processors, timeout=a.timeout)
        row = sc.row(alpha, res, over=v["over"], loading=v["loading"])
        doc["cases"][key] = {k: (r(x, 6) if isinstance(x, float) else x) for k, x in row.items()}
        msg = f"CL {row['Cl']:.3f} CD {row['Cd']:.4f}"
    except Exception as exc:
        doc["cases"][key] = {"alpha_deg": alpha, "status": "failed", "message": f"{type(exc).__name__}: {exc}"[:300]}
        msg = "FAILED"
    doc["cases"][key]["seconds"] = round(time.time() - t0, 1)
    rows = [c for c in doc["cases"].values() if c.get("status") == "solved" and c["alpha_deg"] <= sc.FIT_MAX_ALPHA]
    if len(rows) >= 3:
        C0 = coefficients(v)
        C = mo.calibrate(C0, rows)
        doc["fit"] = {"cases": len(rows), "CLa_per_rad": r(C["CLa"], 3), "CL0": r(C["CL0"], 3), "CD0": r(C["CD0"], 4), "k": r(C["k"], 4),
                      "LD_max": r(mo.speeds(C)["LD_max"], 1), "x_np_mm": r(C["x_np_m"] * 1e3, 1), "static_margin": r(C["static_margin"], 3),
                      "lattice": {"CLa_per_rad": r(C0["CLa"], 3), "x_np_mm": r(C0["x_np_m"] * 1e3, 1), "CD0_hand": C0["CD0"], "k_hand": r(C0["k"], 4)}}
    save(path, doc)
    return msg


def joiner_load(v) -> dict:
    p = st.resolve(v["over"])
    k, J = p["scale"], st.JOINER
    M = N_LIMIT * p["mtow_g"] / 1000 * am.G / 2 * (0.43 * p["span"] / 2 - 50.0 * k) / 1000          # N m at the fuselage wall
    Z = math.pi * (J["od"] ** 4 - (J["od"] - 2 * J["wall"]) ** 4) / (32 * J["od"])
    od = J["od"]
    while math.pi * (od ** 4 - (od - 2 * J["wall"]) ** 4) / (32 * od) * TUBE_ALLOWABLE / 2.0 < M * 1000:   # a reserve of 2 on the allowable
        od += 0.5
    return {"moment_Nm": M, "end_load_N": M * 1000 / J["in_panel"], "hand_MPa": M * 1000 / Z, "y_wall": 50.0 * k, "od_mm": J["od"], "wall_mm": J["wall"],
            "od_for_reserve_2_mm": od}


def do_fea(name, v, a):
    from vegeta import talos
    jl = joiner_load(v)
    step = RUNS / "fea" / f"{name}_joiner.step"
    step.parent.mkdir(parents=True, exist_ok=True)
    st.Sterna().generate(**v["over"], part="joiner").export_step(str(step))
    mat = talos.Material("carbon tube, isotropic stand-in", youngs_modulus=110000, poissons_ratio=0.3, density=1.55e-9, yield_strength=TUBE_ALLOWABLE, source="HAND")
    model = talos.StructuralModel(geometry=str(step), units="mm-N-MPa", material=mat,
                                  regions=[talos.SurfacesOnPlane("wall", "y", jl["y_wall"]), talos.SurfacesOnPlane("end", "y", jl["y_wall"] + st.JOINER["in_panel"])],
                                  supports=[talos.FixedSupport("wall")], loads=[talos.Force("end", fz=jl["end_load_N"])],
                                  mesh_settings=talos.MeshSettings(element_size=a.element_size, order=2))
    work = str(RUNS / "fea" / f"{name}_joiner")
    mesh = model.mesh(work)
    res = model.solve(work, threads=a.processors)
    m = res.metrics
    save(OUT / f"{name}_fea.yaml", {"aircraft": v["title"], "part": "wing joiner tube, from the fuselage wall to its end in the panel",
                                    "load": f"{N_LIMIT:g} g at the design mass, half the lift at 43 % of the half-span",
                                    "tube": {"od_mm": jl["od_mm"], "wall_mm": jl["wall_mm"]}, "moment_Nm": r(jl["moment_Nm"], 2), "end_load_N": r(jl["end_load_N"], 1),
                                    "max_von_mises_MPa": r(m["max_von_mises"], 1), "hand_M_over_Z_MPa": r(jl["hand_MPa"], 1), "end_deflection_mm": r(m["max_displacement"], 3),
                                    "allowable_MPa": TUBE_ALLOWABLE, "reserve": r(TUBE_ALLOWABLE / m["max_von_mises"], 2),
                                    "od_for_a_reserve_of_2_mm": jl["od_for_reserve_2_mm"], "nodes": mesh.metrics["n_nodes"], "elements": mesh.metrics["n_elements"],
                                    "note": "isotropic stand-in material; 600 MPa is a hand allowable; the panel's socket spreads the load in reality"})
    return f"{m['max_von_mises']:.0f} MPa (hand {jl['hand_MPa']:.0f}), reserve {TUBE_ALLOWABLE / m['max_von_mises']:.1f}"


def do_stability(name, v, a):
    p = st.resolve(v["over"])
    L = st.Sterna.layout(p)
    pct = lambda x_m: 100 * (x_m * 1e3 - L["x_mac_le"]) / L["mac"]
    rows = cfd_fit_rows(name)
    C0 = coefficients(v)
    C = coefficients(v, cfd_rows=rows)
    Vs = mo.stall_speed(C)
    trims = []
    for V in (1.2 * Vs, 14.0, 18.0):
        try:
            t = am.trim(C, float(V))
            trims.append({"V_m_s": r(V, 1), "alpha_deg": r(t["alpha_deg"], 2), "elevator_deg": r(t["elevator_deg"], 2), "throttle": r(t["throttle"], 3), "ok": bool(t["ok"])})
        except Exception as exc:
            trims.append({"V_m_s": r(V, 1), "ok": False, "message": f"{type(exc).__name__}: {exc}"[:120]})
    modes = {}
    try:
        for mode, d in am.name_modes(am.modes(C, 14.0)).items():
            modes[mode] = {"stable": bool(d["stable"]), "period_s": r(d["period_s"], 2), "damping_ratio": r(d["zeta"], 3), "time_to_half_or_double_s": r(d["t_half_or_double_s"], 2)}
    except Exception as exc:
        modes = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    cg = []
    for ld in v["loadings"]:
        b = sm.budget(p, L, ld)
        cg.append({"loading": b["case"], "takeoff_g": b["total_g"], "tray_mm": r(b["battery"]["x_mm"], 0), "x_cg_mm": b["x_cg_mm"], "cg_pct_mac": r(100 * b["cg_mac"], 1),
                   "margin_to_neutral_point_pct": r(pct(C["x_np_m"]) - 100 * b["cg_mac"], 1), "within_design_mass": bool(b["within_mtow"])})
    save(OUT / f"{name}_stability.yaml", {
        "aircraft": v["title"], "loading": C["mass_budget"]["case"], "mass_kg": r(C["mass"], 3), "cg_pct_mac": r(pct(C["x_cg_m"]), 1),
        "tail_volumes": {"horizontal": r(L["V_h"], 3), "vertical": r(L["V_v"], 4)},
        "neutral_point_pct_mac": {"lattice": r(pct(C0["x_np_lattice_m"]), 1), "lattice_with_fuselage_allowance": r(pct(C0["x_np_m"]), 1),
                                  "cfd": r(pct(C["x_np_m"]), 1) if rows else "NOT RUN (--cfd)"},
        "static_margin_pct_mac": r(100 * C["static_margin"], 1), "derivatives_from": "CFD fit" if rows else "lattice and hand",
        "stall_m_s": r(Vs, 2), "trim": trims, "modes_at_14_m_s": modes, "loadings": cg,
        "note": "lateral derivatives, damping and inertia are hand estimates (+-30 %); no propeller effects"})
    return f"margin {100 * C['static_margin']:.0f} % MAC ({'CFD' if rows else 'lattice'})"


def do_flight(name, v, a):
    import sterna_mujoco as smj
    C = coefficients(v, cfd_rows=cfd_fit_rows(name))
    C["thrust"] = mo.thrust_table(mo.propulsion(modules=C["mass_budget"]["battery"]["modules"], **v["drive"]))
    release = smj.RELEASE_M_S if v["launch"] == "hand" else 1.25 * mo.stall_speed(C)      # the line releases above the stall
    _, summ = smj.mission(smj.Sim(C), release=release)
    summ["launch"] = v["launch"]
    save(OUT / f"{name}_flight.yaml", {"aircraft": v["title"], "loading": C["mass_budget"]["case"], **{k: (list(map(list, x)) if k == "phases" else x) for k, x in summ.items()}})
    return f"safe after {summ['time_to_safe_s']} s, touchdown {summ['touchdown']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in STEPS:
        ap.add_argument(f"--{s}", action="store_true", help=f"run the {s} step (no step flag: all)")
    ap.add_argument("--variant", nargs="*", default=["sterna", "talassa"], choices=("sterna", "talassa"))
    ap.add_argument("--quality", default="screening", choices=tuple(sc.QUALITY))
    ap.add_argument("--processors", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=3 * 3600.0, help="seconds per CFD case")
    ap.add_argument("--element-size", type=float, default=2.0, help="FEA mesh [mm]")
    ap.add_argument("--list", action="store_true", help="print the tasks and stop")
    ap.add_argument("--fresh", action="store_true", help="CFD: ignore the saved cases")
    a = ap.parse_args(argv)
    steps = [s for s in STEPS if getattr(a, s)] or list(STEPS)
    V = variants()
    a.fresh_now = {OUT / f"{n}_cfd.yaml": True for n in a.variant} if a.fresh else {}
    tasks = []
    for n in a.variant:
        v = V[n]
        for s in steps:
            if s == "cfd":
                tasks += [(f"{n} cfd {al:+.0f} deg", do_cfd_case, (n, v, a, al)) for al in sc.ALPHAS]
            else:
                tasks.append((f"{n} {s}", {"cad": do_cad, "export": do_export, "fea": do_fea, "stability": do_stability, "flight": do_flight}[s], (n, v, a)))
    if a.list:
        for n in a.variant:
            v = V[n]
            C = coefficients(v)
            jl = joiner_load(v)
            print(f"{v['title']}: {C['mass']:.2f} kg ({C['mass_budget']['case']}), S {C['S']:.3f} m2, stall {mo.stall_speed(C):.1f} m/s, lattice margin {100 * C['static_margin']:.0f} % MAC, "
                  f"joiner {jl['moment_Nm']:.1f} N m -> {jl['hand_MPa']:.0f} MPa by hand")
        for label, _, _ in tasks:
            print(f"  {label}")
        print(f"{len(tasks)} tasks: NOT RUN")
        return 0
    bad = 0
    bar = tqdm(tasks, unit="task", dynamic_ncols=True)
    for label, fn, args in bar:
        bar.set_description(label)
        try:
            msg = fn(*args)
        except Exception as exc:
            bad += 1
            msg = f"FAILED: {type(exc).__name__}: {exc}"[:160]
        tqdm.write(f"  {label}: {msg}")
    print(f"results in {OUT}" + (f" ({bad} task(s) failed)" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
