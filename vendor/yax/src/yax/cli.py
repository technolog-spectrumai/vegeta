"""``yax convert SRC DST`` (any two formats, through USD), ``yax show FILE`` (links, joints, mass), ``yax check FILE.yx2 | DIR...``
(a YX2 file or a Yax 1 package; exit 1 on problems), ``yax info DIR``, ``yax tags``, ``yax select``, ``yax sync``."""
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
    y = sub.add_parser("sync", help="take a design's physical numbers into an SDF model (only what the map names)")
    y.add_argument("design")
    y.add_argument("target")
    y.add_argument("--map", required=True, dest="mapping")
    y.add_argument("--check", action="store_true", help="write nothing; exit 1 if the model has drifted")
    y.add_argument("--out", help="write here instead of in place")
    f = sub.add_parser("tags", help="what a machine file holds per tag; with requirements, whether it serves a consumer")
    f.add_argument("file")
    f.add_argument("--to", dest="fmt", help="judge for this target format (unsupported tags)")
    f.add_argument("--require", default="", help="comma-separated tags the consumer needs")
    f.add_argument("--optional", default="", help="comma-separated tags the consumer uses when present")
    f.add_argument("--requirements", help='a JSON file {"consumer", "required", "optional"}')
    s_ = sub.add_parser("select", help="write DST holding the properties of SRC that carry the kept tags (the rest withheld)")
    s_.add_argument("src")
    s_.add_argument("dst")
    s_.add_argument("--keep", default="", help="comma-separated tags to keep (default: all)")
    s_.add_argument("--drop", default="", help="comma-separated tags to withhold")
    s_.add_argument("--tag", action="append", default=[], metavar="PATTERN=CODE", help="tag the properties matching PATTERN with CODE (repeatable)")
    sub.add_parser("check", help="validate .yx2 files and Yax 1 packages (exit 1 on problems)").add_argument("dirs", nargs="+")
    sub.add_parser("validate", help="a machine file's engineering blocks: schemas, units, references, frames; its tags (exit 1 on problems)").add_argument("file")
    sub.add_parser("info", help="print a package's manifest summary").add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        from .convert import convert
        print(convert(a.src, a.dst, src_fmt=a.src_fmt, dst_fmt=a.dst_fmt))
        return 0
    if a.cmd == "sync":
        from .sync import sync
        rep = sync(a.design, a.target, mapping=a.mapping, check=a.check, out=a.out)
        print(rep)
        return 1 if a.check and (rep.changes or rep.dropped) else 0
    if a.cmd in ("tags", "select"):
        from . import tags
        from .convert import read, save
        m, _ = read(a.src if a.cmd == "select" else a.file)
        if a.cmd == "select":
            for spec in a.tag:
                pattern, _, code = spec.rpartition("=")
                tags.tag(m, pattern, code)
            keep = [x for x in a.keep.split(",") if x] or None
            drop = [x for x in a.drop.split(",") if x]
            print(save(tags.select(m, keep, drop) if (keep or drop) else m, a.dst))
            return 0
        print(tags.describe(m))
        req = tags.Requirements.load(a.requirements) if a.requirements else None
        if a.require or a.optional:
            req = tags.Requirements([x for x in a.require.split(",") if x], [x for x in a.optional.split(",") if x],
                                    req.consumer if req else "")
        if req is None:
            return 0
        verdict = tags.check(m, req, a.fmt)
        print(verdict)
        return 0 if verdict.ok else 1
    if a.cmd == "validate":
        from . import schema, tags
        from .convert import read
        m, _ = read(a.file)
        problems = schema.problems(m)
        for path, codes in (m.tags or {}).get("table", {}).items():
            for c in codes:
                if not tags.CODE.match(str(c)):
                    problems.append(f"tag {c!r} on {path}: not 1–8 alphanumerics")
        print(f"{a.file}: {'ok' if not problems else 'INVALID'}")
        for pr in problems:
            print(f"  - {pr}")
        return 1 if problems else 0
    if a.cmd == "show":
        from .convert import info
        print(info(a.file))
        return 0
    if a.cmd == "check":
        status = 0
        for d in a.dirs:
            if d.endswith(".yx2"):
                from .formats.yx2 import validate as check_yx2
                problems = check_yx2(d)
            else:
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
