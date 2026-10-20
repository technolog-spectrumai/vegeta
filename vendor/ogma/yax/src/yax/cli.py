"""``yax check DIR...`` (exit 1 on problems) and ``yax info DIR``."""
from __future__ import annotations

import argparse
import json
import sys

from .package import check, load


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="yax", description="Check or describe Yax design packages.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="validate packages").add_argument("dirs", nargs="+")
    sub.add_parser("info", help="print a package's manifest summary").add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        status = 0
        for d in a.dirs:
            problems = check(d)
            print(f"{d}: {'ok' if not problems else 'INVALID'}")
            for p in problems:
                print(f"  - {p}")
            status |= bool(problems)
        return status
    m = load(a.dir).manifest
    summary = {k: m.get(k) for k in ("name", "identity", "units", "frame", "geometry", "mass")}
    summary["files"] = sorted(m["files"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
