#!/usr/bin/env python3
"""HAWK 2 (ASTOR-TWIN): the wing spar at 8 g, in Talos (CalculiX FEA). ``designs/astor_twin_fea`` has the cases.

    labs/aero/sims/astor_fea.sh -j 8             # mesh and solve the three cases (2 static, 1 modal)
    labs/aero/sims/astor_fea.sh --dry-run        # the beam reference and the cases, nothing solved
    labs/aero/sims/astor_fea.sh -j 8 --fresh     # forget the solved cases

``-j N``: CalculiX's threads. Solved cases are kept in ``--workdir`` and read back on a restart. One timestamped YAML
per run in ``sims/output/runs`` (the beam reference, each case against it and against the allowable). Needs CalculiX
(``ccx``) and Gmsh on this machine; without them the cases come back NOT RUN and the beam reference still prints.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parent / "designs"
sys.path[:0] = [str(DESIGNS), str(HERE)]

import astor_run as R                                                # noqa: E402

R.GLOBS_ACTIVE = ["arigo*.py", "astor*.py", "raptors*.py"]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", "--cpu", type=int, default=4, help="CalculiX threads (default 4)")
    ap.add_argument("--workdir", type=Path, default=HERE.parent / "_runs" / "astor_fea", help="the cases' folder")
    ap.add_argument("--element", type=float, default=3.0, help="the mesh's element size [mm] (default 3; the tube walls are 1.5 mm: 2 is finer and slower)")
    ap.add_argument("--fresh", action="store_true", help="delete the solved cases first")
    ap.add_argument("--dry-run", action="store_true", help="print the beam reference and the cases; solve nothing")
    a = ap.parse_args(argv)
    import yaml
    import astor_twin_fea as FE
    ref = FE.reference()
    print(yaml.safe_dump({"beam reference": ref}, sort_keys=False, allow_unicode=True, width=140))
    if a.fresh and a.workdir.exists():
        shutil.rmtree(a.workdir)
    try:
        cases = FE.cases(a.workdir, element_size=a.element)
    except Exception as exc:                                         # noqa: BLE001  (no Talos here: the reference alone)
        print(f"the FEA cases could not be built ({type(exc).__name__}: {exc}); the beam reference above stands alone")
        return 1
    if a.dry_run:
        for c in cases:
            print(f"fea  {c.name:20s} {c.kind:7s} {c.description}")
        print(len(cases), "cases")
        return 0
    bar = R._bar(len(cases), 0, "hawk2 fea", unit="case")
    rows, t0 = {}, time.time()
    for c in cases:
        t = time.time()
        r = FE.solve_one(c, a.workdir, run=True, threads=a.cpu)
        rows[c.name] = {**FE.row(c, r), "seconds": round(time.time() - t, 1)}
        msg = f"{c.name}: {rows[c.name]['status']}" + (f"  margin {rows[c.name].get('margin')}" if rows[c.name].get("margin") is not None else "")
        if bar:
            bar.set_postfix_str(msg[:60]); bar.update(1)
        print("  " + msg)
    if bar:
        bar.close()
    run = R._write("fea", {"run": {"aircraft": "ASTOR-TWIN (HAWK 2)", "phase": "fea", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename,
                                   "commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(), "cpu": a.cpu, "element_mm": a.element,
                                   "seconds": round(time.time() - t0, 1)},
                           "beam_reference": ref, "cases": rows}, "astor_twin")
    print("->", run)
    return 0 if all(v["status"] == "solved" for v in rows.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
