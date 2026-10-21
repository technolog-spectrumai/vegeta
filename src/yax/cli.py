"""``yax convert SRC DST`` (any two formats, through USD), ``yax show FILE`` (links, joints, mass), and for Yax 1
packages ``yax check DIR...`` (exit 1 on problems) and ``yax info DIR``."""
from __future__ import annotations

import argparse
import json
import sys

from .package import check, load


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="yax", description="Convert machine files between formats (USD hub); check Yax 1 packages.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="convert SRC to DST (formats from the suffixes; .xml: MJCF or JSBSim by content)")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--from", dest="src_fmt")
    c.add_argument("--to", dest="dst_fmt")
    sub.add_parser("show", help="links, joints and mass of a machine file").add_argument("file")
    sub.add_parser("check", help="validate packages").add_argument("dirs", nargs="+")
    sub.add_parser("info", help="print a package's manifest summary").add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        from .convert import convert
        print(convert(a.src, a.dst, src_fmt=a.src_fmt, dst_fmt=a.dst_fmt))
        return 0
    if a.cmd == "show":
        from .convert import info
        print(info(a.file))
        return 0
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
