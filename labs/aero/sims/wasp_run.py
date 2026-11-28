"""Cotesia and Vespula — every calculation and export from one script, one progress bar. RUN BY YOU.

    labs/aero/sims/wasp_run.sh --list                        # what would run; nothing is meshed or solved
    labs/aero/sims/wasp_run.sh                               # everything, both builds
    labs/aero/sims/wasp_run.sh --cfd                         # only the CFD
    labs/aero/sims/wasp_run.sh --stability                   # only the stability calculation (no solver: seconds)
    labs/aero/sims/wasp_run.sh --fea --build Cotesia         # one build
    labs/aero/sims/wasp_run.sh --cad --bom --export          # the files to build from and to hand on
    labs/aero/sims/wasp_run.sh --cfd --quality fine --processors 8

The steps (a flag each; with no step flag, all of them):

- ``--cad``        the aircraft, one arm and the nose as STEP                                   (CadQuery; seconds)
- ``--bom``        the bill of materials as Markdown and CSV
- ``--export``     the Yax exchange file ``<build>.yx2`` with its description: mass, inertia, the CAD mesh, the rotors
- ``--cfd``        the airframe without propellers: 3 axial cases (drag) and 8 at incidence     (OpenFOAM; minutes a case)
- ``--fea``        one printed arm: thrust, manoeuvre, landing knock at x1.5, and its modes     (Gmsh + CalculiX)
- ``--stability``  neutral point and margin by hand, and from the CFD when it has run; pitch dynamics; motor authority

Results: ``labs/aero/sims/output/<build>_{cfd,fea,stability}.yaml``, written after every case (stop and restart freely;
a solved case is kept, ``--fresh`` starts over). CAD, BOM and ``.yx2`` go to ``labs/aero/designs/data/wasp/``.
``FOAM_PREFIX`` names an OpenFOAM conda prefix that is not found by itself. The steps are the modules beside this
file (``wasp_cfd``, ``wasp_fea``, ``wasp_stability``, ``wasp_bom_cad``), which also run on their own.
"""
import argparse
import time

from tqdm import tqdm

import wasp_common as wc
from wasp_common import fam

import emerillon as em  # noqa: E402
import emerillon_campaigns as ec  # noqa: E402
import wasp_bom_cad  # noqa: E402
import wasp_cfd  # noqa: E402
import wasp_fea  # noqa: E402
import wasp_stability  # noqa: E402

STEPS = ("cad", "bom", "export", "cfd", "fea", "stability")
DATA = wc.ROOT / "labs" / "aero" / "designs" / "data" / "wasp"
_VEH = {}


def veh(build):
    if build not in _VEH:
        _VEH[build] = fam.vehicle(build)
    return _VEH[build]


def do_cad(build, a):
    wasp_bom_cad.main(["--build", build])
    return f"{DATA / fam.slug(build)}.step, _arm.step, _nose.step"


def do_bom(build, a):
    wasp_bom_cad.main(["--build", build, "--no-cad"])
    return f"{DATA / fam.slug(build)}_bom.md"


def do_export(build, a):
    r = fam.export_yx2(build, DATA / f"{fam.slug(build)}.yx2")
    return f"{r['file']}: {r['mass_kg'] * 1e3:.1f} g, {r['triangles']} triangles, {r['bytes'] / 1e3:.0f} kB"


def do_cfd_case(build, a, key, spec):
    v = veh(build)
    path = wc.OUT / f"{fam.slug(build)}_cfd.yaml"
    doc = None if a.fresh_now.pop(path, False) else wc.load(path)
    if not doc or doc.get("quality") != a.quality:
        doc = {"aircraft": build, "config": v.label, "quality": a.quality, "template": "rans_ksst_external",
               "reference": {"frontal_area_m2": v.geo["frontal"], "length_m": v.geo["length"], "x_cg_m": v.x_cg, "mass_kg": v.mass, "V_top_m_s": v.V_top},
               "note": "airframe without propellers; coefficients on the frontal area, Cm about the CG on the length", "cases": {}}
    if doc["cases"].get(key, {}).get("status") == "solved":
        return "kept"
    t0 = time.time()
    try:
        res = wasp_cfd.case(wc.RUNS / "cfd", build, v, spec, a.quality).ensure(run=True, processors=a.processors, timeout=a.timeout)
        doc["cases"][key] = wasp_cfd.row(v, spec, res)
        msg = f"CdA {doc['cases'][key]['CdA_cm2']} cm2 (hand {doc['cases'][key]['hand']['CdA_clean_cm2']} clean)"
    except Exception as exc:
        doc["cases"][key] = {**spec, "status": "failed", "message": f"{type(exc).__name__}: {exc}"[:300]}
        msg = "FAILED"
    doc["cases"][key]["seconds"] = round(time.time() - t0, 1)
    drag = [c for c in doc["cases"].values() if c.get("campaign") == "drag" and c.get("status") == "solved"]
    if drag:
        top = max(drag, key=lambda c: c["V"])
        doc["summary"] = {"CdA_at_top_speed_cm2": top["CdA_cm2"], "hand_clean_cm2": top["hand"]["CdA_clean_cm2"],
                          "margin_the_cfd_supports": wc.num(top["CdA_cm2"] / top["hand"]["CdA_clean_cm2"], 2), "margin_assumed": wasp_cfd.ef.DRAG_MARGIN}
    wc.save(path, doc)
    return msg


def do_fea_case(build, a, n):
    v = veh(build)
    p = em.resolve(fam.cad_parameters(build))
    hb = wasp_fea.beam(v, p)
    path = wc.OUT / f"{fam.slug(build)}_fea.yaml"
    doc = wc.load(path) or {"aircraft": build, "config": v.label, "part": "one arm with its pod, clamped at the root pad",
                            "material": dict(ec.PA6CF), "ultimate_factor": ec.ULTIMATE, "hand": hb, "cases": {}}
    t0, work = time.time(), str(wc.RUNS / "fea" / f"{fam.slug(build)}_{n}")
    what = wasp_fea.loads(v)[n]["what"]
    try:
        mdl = wasp_fea.model(build, v, n, a.element_size)
        mesh = mdl.mesh(work)
        if n == "modes":
            res = mdl.solve_modes(work, n_modes=4, threads=a.processors)
            f = [float(x) for x in res.metrics["frequencies_hz"]]
            doc["cases"][n] = {"what": what, "status": "solved", "frequencies_Hz": [round(x, 1) for x in f], "rotor_1P_Hz": hb["rotor_1P_Hz"],
                               "first_mode_over_1P": wc.num(f[0] / (v.rpm_top / 60.0), 2), "hand_f1_Hz": hb["f1_thin_Hz"]}
            msg = f"first mode {f[0]:.0f} Hz, rotor 1P {v.rpm_top / 60:.0f} Hz"
        else:
            res = mdl.solve(work, threads=a.processors)
            m = res.metrics
            doc["cases"][n] = {"what": what, "status": "solved", "max_von_mises_MPa": wc.num(m["max_von_mises"], 2),
                               "at_mm": [round(float(x), 1) for x in m["max_von_mises_location"]], "max_displacement_mm": wc.num(m["max_displacement"], 3),
                               "reserve_in_plane": wc.num(ec.PA6CF["strength"] / m["max_von_mises"], 2),
                               "reserve_across_layers": wc.num(ec.PA6CF["strength_z"] / m["max_von_mises"], 2), "messages": list(res.messages)[:3]}
            msg = f"{m['max_von_mises']:.1f} MPa"
        doc["cases"][n].update({"nodes": mesh.metrics["n_nodes"], "elements": mesh.metrics["n_elements"]})
    except Exception as exc:
        doc["cases"][n] = {"what": what, "status": "failed", "message": f"{type(exc).__name__}: {exc}"[:300]}
        msg = "FAILED"
    doc["cases"][n]["seconds"] = round(time.time() - t0, 1)
    wc.save(path, doc)
    return msg


def do_stability(build, a):
    wasp_stability.main(["--build", build])
    d = wc.load(wc.OUT / f"{fam.slug(build)}_stability.yaml")
    return f"margin {d['by_hand']['static_margin_lengths']} lengths by hand" + ("" if isinstance(d["from_cfd"], str) else f", {d['from_cfd']['static_margin_lengths']} from the CFD")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in STEPS:
        ap.add_argument(f"--{s}", action="store_true", help=f"run the {s} step (no step flag: all)")
    ap.add_argument("--build", nargs="*", default=wc.BUILDS, choices=wc.BUILDS)
    ap.add_argument("--quality", default="screening", choices=tuple(ec.QUALITY))
    ap.add_argument("--processors", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=3 * 3600.0, help="seconds per CFD case")
    ap.add_argument("--element-size", type=float, default=1.2, help="FEA mesh [mm]")
    ap.add_argument("--list", action="store_true", help="print the tasks and stop")
    ap.add_argument("--fresh", action="store_true", help="CFD: ignore the saved cases")
    a = ap.parse_args(argv)
    steps = [s for s in STEPS if getattr(a, s)] or list(STEPS)
    if "cad" in steps and "bom" in steps:
        steps.remove("bom")                                       # the CAD step writes the bill of materials too
    a.fresh_now = {wc.OUT / f"{fam.slug(b)}_cfd.yaml": True for b in a.build} if a.fresh else {}
    tasks = []
    for b in a.build:
        for s in steps:
            if s == "cfd":
                tasks += [(f"{b} cfd {k}", do_cfd_case, (b, a, k, sp)) for k, sp in wasp_cfd.specs(veh(b), ("drag", "alpha")).items()]
            elif s == "fea":
                tasks += [(f"{b} fea {n}", do_fea_case, (b, a, n)) for n in wasp_fea.CASES]
            else:
                tasks.append((f"{b} {s}", {"cad": do_cad, "bom": do_bom, "export": do_export, "stability": do_stability}[s], (b, a)))
    if a.list:
        for b in a.build:
            v = veh(b)
            print(f"{b}: {v.label}\n  {v.mass * 1e3:.0f} g" + (f" with {v.payload_g:.0f} g aboard" if v.payload_g else "") + f", top speed {v.V_top * 3.6:.0f} km/h, CG {v.x_cg * 1e3:.0f} mm from the nose")
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
        if msg == "FAILED":
            bad += 1
        tqdm.write(f"  {label}: {msg}")
    print(f"results in {wc.OUT} and {DATA}" + (f" ({bad} task(s) failed)" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
