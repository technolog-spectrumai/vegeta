#!/usr/bin/env python3
"""Kestrel-4Q's transitions through the stall — hover → cruise and cruise → hover — on the model's polar and, with
``--cfd``, on the CFD's (notebook 38, §§ 6–7).

    python3 scenarios/kestrel_transitions.py                    # the matrix on the model's polar (~2 min)
    python3 scenarios/kestrel_transitions.py --quick            # 1.5 and 3 s schedules only
    python3 scenarios/kestrel_transitions.py --cfd -j 1 -n 8    # + Aeromant at the transition points (OpenFOAM), then the matrix again
    python3 scenarios/kestrel_transitions.py --part2             # + Kestrel-4Q-2 (elevons, faired posts, no X) and the X-wings

The matrix: the baseline with elevons, the rotors as the only effectors, the 2.0 kg vehicle with its CG aft; both
directions; 1.5 / 3 / 5 s pitch schedules; still air and a 6 m/s headwind (``kestrel_transition.matrix``). With
``--part2`` the Part-II vehicles join it: Kestrel-4Q-2 (the CAD's elevons, faired posts only, no diagonals) and the
two X-wings (``kestrel_xwing``: the same area, and twice it). With
``--cfd`` the airframe is solved at the points the nominal transitions fly (four rotor disks at each pair's rpm,
``kestrel_cfd``), the solved coefficients become the model's correction (``CFDAirframe``) and the matrix is flown again.
Writes ``scenarios/output/kestrel/``: the matrix tables (CSV), the CFD table, the figures of the nominal transitions
and ``kestrel_transitions.json`` (the verdicts). Exit 0 when every baseline transition passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import kestrel_cfd as kcfd  # noqa: E402
import kestrel_transition as kt  # noqa: E402
import kestrel_xwing as kx  # noqa: E402

OUT = ROOT / "scenarios" / "output" / "kestrel"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="the 1.5 and 3 s schedules only")
    ap.add_argument("--cfd", action="store_true", help="solve the transition points in OpenFOAM and fly the matrix again on them")
    ap.add_argument("--quality", default="screening", choices=tuple(kcfd.QUALITY))
    ap.add_argument("-j", "--jobs", type=int, default=1, help="CFD cases at a time")
    ap.add_argument("-n", "--processors", type=int, default=4, help="MPI ranks per CFD case")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--part2", action="store_true", help="add Kestrel-4Q-2 and the X-wings to the matrix")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    durs = (1.5, 3.0) if args.quick else (1.5, 3.0, 5.0)
    veh = kt.make_vehicle(name="Kestrel-4Q, elevons")
    vehicles = {"elevons": veh, "rotors only": kt.make_vehicle(elevons=False, name="Kestrel-4Q, rotors only"),
                "2.0 kg, CG aft": kt.make_vehicle(extra_kg=0.3, name="Kestrel-4Q, 2.0 kg")}
    if args.part2:
        p2 = dict(elevons=True, bracing="posts", strut_fairing=True)
        vehicles["Kestrel-4Q-2"] = kt.make_vehicle(p2, faired=True, name="Kestrel-4Q-2 (elevons, faired posts, no X)")
        for v in kx.VARIANTS:
            vehicles[v] = kx.make_vehicle(v)
    mat, _ = kt.matrix(vehicles, durations=durs, headwinds=(0.0, 6.0))
    mat.to_csv(OUT / "transition_matrix.csv")
    doc = {"model": kt.verdict(mat)}
    nominal = [kt.simulate(veh, k, t_tr=3.0) for k in ("forward", "back")]
    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import kestrel_drawings as kd
        for ep in nominal:
            kd.episode_figure(ep, veh).savefig(OUT / f"nominal_{ep.kind}.png", dpi=110)
            plt.close("all")
        kd.path_figure(nominal, veh, x_max=65).savefig(OUT / "nominal_paths.png", dpi=110)
        plt.close("all")
    for name, v in doc["model"].items():
        print(f"model polar, {name:16s}: {v['passes']} of {v['of']} pass")
        for w in v["failures"]:
            print(f"    - {w}")
    if args.cfd:
        from vegeta import aeromant
        pts = kcfd.points(nominal, mass_kg=veh.mass)
        spec = kcfd.specs(pts)
        cases = kcfd.plan(OUT / "cfd", pts, x_cg_m=veh.x_cg, z_cg_m=veh.z_cg, quality=args.quality)
        names = list(cases)
        res = aeromant.run_cases([cases[n] for n in names], jobs=args.jobs, processors=args.processors, progress=True, cache=True)
        tab = kcfd.table(dict(zip(names, res)), spec)
        tab.to_csv(OUT / "cfd_cases.csv")
        sm = kcfd.summary(tab)                                     # a failed or diverged case is a result, listed and counted
        print("CFD:", sm["line"])
        doc["cfd_summary"] = {k: v for k, v in sm.items() if k != "line"}
        kcfd.save_records(tab, kcfd.RESULTS_JSON, key="4Q-1", meta={"source": "scenarios/kestrel_transitions.py", "quality": args.quality})
        corr = kcfd.CFDAirframe.from_results(tab, veh) or kcfd.CFDAirframe.from_results(tab, veh, source="sweep")
        if corr is None:
            print("CFD: nothing usable solved — the verdict stands on the model's polar")
        else:
            print(f"CFD: correction from the {corr.source} ({len(corr.samples)} points)")
            vc = {"elevons, CFD": kt.make_vehicle(airframe=corr, name="Kestrel-4Q, elevons, CFD aero"),
                  "rotors only, CFD": kt.make_vehicle(airframe=corr, elevons=False, name="Kestrel-4Q, rotors only, CFD aero")}
            mat_c, _ = kt.matrix(vc, durations=durs, headwinds=(0.0, 6.0))
            mat_c.to_csv(OUT / "transition_matrix_cfd.csv")
            doc["cfd"] = kt.verdict(mat_c)
            for name, v in doc["cfd"].items():
                print(f"CFD polar, {name:16s}: {v['passes']} of {v['of']} pass")
                for w in v["failures"]:
                    print(f"    - {w}")
    (OUT / "kestrel_transitions.json").write_text(json.dumps(doc, indent=1, default=str))
    base = doc.get("cfd", {}).get("elevons, CFD") or doc["model"]["elevons"]
    return 0 if base["passes"] == base["of"] else 1


if __name__ == "__main__":
    sys.exit(main())
