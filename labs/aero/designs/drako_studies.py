"""Drako's FEA (``drako_fea``) and CFD (``drako_cfd``) in one go, for a machine with CadQuery, Gmsh, CalculiX and
OpenFOAM. Writes ``<out>/fea_table.csv``, ``<out>/cfd_table.csv`` and ``<out>/summary.json``.

    python drako_studies.py --out runs/drako                 # build, mesh and solve everything
    python drako_studies.py --out runs/drako --only fea      # or cfd
    python drako_studies.py --out runs/drako --prepare-only  # CAD, cases and OpenFOAM case folders, nothing solved
    VEGETA_DEBUG=1 python drako_studies.py --out /tmp/drako  # the whole pipeline under vegeta.mock (numbers meaningless)

A case that fails is a row with its message, not an exception. The FEA is read against the beam reference and the
rotor harmonics; the CFD against the drag areas the sizing assumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import drako_cfd as dcfd
import drako_fea as dfea


def fea(out: Path, run: bool, threads: int) -> dict:
    cs = dfea.cases(out / "fea")
    res = dfea.solve(cs, out / "fea", run=run, threads=threads)
    tab = dfea.table(cs, res)
    tab.to_csv(out / "fea_table.csv")
    return {"cases": len(cs), "solved": int((tab["status"] == "solved").sum()), "table": str(out / "fea_table.csv"),
            "arm_reference": dfea.arm_reference(), "arm_frequency_check": dfea.arm_frequency_check()}


def cfd(out: Path, run: bool, processors: int, iterations: int) -> dict:
    cs = dcfd.cases(out / "cfd", iterations=iterations)
    res = dcfd.run(cs, run=run, processors=processors)
    tab = dcfd.table(cs, res)
    tab.to_csv(out / "cfd_table.csv")
    return {"cases": len(cs), "solved": int((tab["status"] == "solved").sum()), "table": str(out / "cfd_table.csv")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=("fea", "cfd"))
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--iterations", type=int, default=600)
    a = ap.parse_args(argv)
    from vegeta import mock
    mocked = mock.setup(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = not a.prepare_only
    summary = {"mock": mocked, "solved": run}
    if a.only in (None, "fea"):
        summary["fea"] = fea(out, run, a.threads)
    if a.only in (None, "cfd"):
        summary["cfd"] = cfd(out, run, a.threads, a.iterations)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    for k in ("fea", "cfd"):
        if k in summary:
            print(f"{k}: {summary[k]['solved']} of {summary[k]['cases']} solved -> {summary[k]['table']}")
    if mocked:
        print("vegeta.mock was on: the numbers are placeholders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
