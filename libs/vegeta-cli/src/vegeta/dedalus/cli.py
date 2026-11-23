"""Command line interface: ``dedalus params|generate|measure``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .loading import load_design


def _parse_overrides(design, items):
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--param expects name=value, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in design.params:
            raise ValueError(f"unknown parameter {name!r}; known: {[p.name for p in design.params]}")
        overrides[name] = design.params[name].parse(value)
    return overrides


def cmd_params(args) -> int:
    d = load_design(args.design)
    rows = d.params.table()
    if args.json:
        print(json.dumps({"design": d.name, "units": d.units, "parameters": rows}, indent=2))
        return 0
    print(f"{d.name} ({d.units})")
    for r in rows:
        rng = f"[{r['min'] if r['min'] is not None else '-inf'}, {r['max'] if r['max'] is not None else 'inf'}]"
        print(f"  {r['name']:<22} {r['default']!s:<10} {r['units'] or '':<5} {rng:<16} {r['description']}")
    return 0


def cmd_generate(args) -> int:
    d = load_design(args.design)
    overrides = _parse_overrides(d, args.param)
    outdir = Path(args.out)
    res = d.run(
        outdir, formats=[f.strip() for f in args.formats.split(",") if f.strip()],
        stl_tolerance=args.stl_tolerance, stl_angular_tolerance=args.stl_angular_tolerance, **overrides,
    )
    if res.ok and args.png:
        from .geometry import Geometry
        from .plotting import plot_views

        geom = Geometry.from_step(res.artifacts["step"]) if "step" in res.artifacts else d.generate(**overrides)
        geom.name = d.name
        fig = plot_views(geom)
        png = outdir / f"{d.name}_views.png"
        fig.savefig(png, dpi=120)
        res.artifacts["views_png"] = png
        res.save_json(outdir / "summary.json")
    print(res.to_json() if args.json else res)
    return 0 if res.ok else 1


def cmd_measure(args) -> int:
    from .geometry import load_step

    g = load_step(args.step, units=args.units)
    m = g.measure()
    if args.json:
        print(json.dumps({"file": args.step, "measurements": m, "messages": g.messages}, indent=2))
    else:
        print(f"{args.step}")
        for k, v in m.items():
            print(f"  {k:<16} {v}")
        for msg in g.messages:
            print(f"  ! {msg}")
    return 0


def build_parser(prog: str = "dedalus") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Parametric CAD on CadQuery")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("params", help="list a design's parameters")
    p.add_argument("design", help="file.py[:Name] or module:Name, e.g. vegeta.dedalus.examples:Bracket")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_params)

    g = sub.add_parser("generate", help="generate geometry and export STEP/STL")
    g.add_argument("design", help="file.py[:Name] or module:Name, e.g. vegeta.dedalus.examples:Bracket")
    g.add_argument("--param", "-p", action="append", metavar="NAME=VALUE", help="parameter override (repeatable)")
    g.add_argument("--out", "-o", default="dedalus_out", help="output directory")
    g.add_argument("--formats", default="step,stl", help="comma separated: step,stl")
    g.add_argument("--stl-tolerance", type=float, default=0.01, help="STL linear deflection (model units)")
    g.add_argument("--stl-angular-tolerance", type=float, default=0.1, help="STL angular deflection (rad)")
    g.add_argument("--png", action="store_true", help="also save a static multi-view PNG")
    g.add_argument("--json", action="store_true", help="print the JSON summary")
    g.set_defaults(func=cmd_generate)

    m = sub.add_parser("measure", help="measure an existing STEP file")
    m.add_argument("step")
    m.add_argument("--units", default="mm", help="length unit of the STEP data (for reporting)")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_measure)
    return ap


def main(argv=None, prog: str = "dedalus") -> int:
    args = build_parser(prog).parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
