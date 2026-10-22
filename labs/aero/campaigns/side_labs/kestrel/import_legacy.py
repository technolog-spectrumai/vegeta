#!/usr/bin/env python3
"""Import the first Kestrel CFD campaign (``designs/data/kestrel_cfd_results.json``) as run directories
of the workflow contract, one per campaign key (``4Q-1``, ``4Q-2``), like any other run.

    .venv/bin/python labs/aero/campaigns/side_labs/kestrel/import_legacy.py
    .venv/bin/python labs/aero/campaigns/side_labs/kestrel/import_legacy.py --results path/to/kestrel_cfd_results.json --out labs/aero/campaigns/side_labs/runs

The import keeps what the file holds — statuses, coefficients, convergence flags — and says what it is: screening
quality, 400 iterations, no native case directories, and a design fingerprint **assumed** from the current source
(the file recorded none); the run says so.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kestrel as wk  # noqa: E402
import kestrel_campaigns as kc  # noqa: E402
import kestrel_cfd as kcfd  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(kcfd.RESULTS_JSON))
    ap.add_argument("--out", default=str(kc.RUNS_BASE), help="the runs base directory (default labs/aero/campaigns/side_labs/runs)")
    ap.add_argument("--keys", default="4Q-1,4Q-2")
    args = ap.parse_args(argv)
    roots = kc.import_legacy(args.results, args.out, keys=tuple(k.strip() for k in args.keys.split(",")))
    for r in roots:
        s = kc.kcfd.load_records  # noqa: F841  (the module is on the path; nothing else to check)
        print(f"imported: {r.relative_to(wk.ROOT) if wk.ROOT in r.parents else r}")
    if not roots:
        print("nothing imported: no campaign of those keys in the file")
        return 1
    print("(the import holds the bare sweep and the blown points of its campaign)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
