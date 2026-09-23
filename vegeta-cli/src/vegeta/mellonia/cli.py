"""Command line interface: ``mellonia slice|parse``.

Settings are Python: ``settings.py`` defines a ``mellonia.PrintSettings`` (or use
``vegeta.mellonia.examples:GENERIC_PLA_0_2MM``).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

from .settings import Orientation, PrintSettings


def load_settings(spec: str) -> PrintSettings:
    target, _, attr = spec.partition(":")
    if target.endswith(".py") or Path(target).is_file():
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(f"no such settings file: {target}")
        name = f"_mellonia_user_{abs(hash(str(path.resolve())))}"
        spec_ = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec_)
        sys.modules[name] = module
        spec_.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    if attr:
        obj = getattr(module, attr, None)
        if not isinstance(obj, PrintSettings):
            raise ValueError(f"{spec} is not a mellonia.PrintSettings")
        return obj
    found = {k: v for k, v in vars(module).items() if isinstance(v, PrintSettings)}
    if len(found) != 1:
        raise ValueError(f"{target} defines {len(found)} PrintSettings {sorted(found)}; use {target}:<name>")
    return next(iter(found.values()))


def cmd_slice(args) -> int:
    from .slicer import slice_stl

    settings = load_settings(args.settings)
    orient = Orientation(args.rotate_x, args.rotate_y, args.rotate_z)
    exe = args.prusa_slicer.split() if args.prusa_slicer else "prusa-slicer"
    res = slice_stl(args.stl, settings, orient, args.out, executable=exe, timeout=args.timeout)
    if res.ok and args.png:
        import matplotlib

        matplotlib.use("Agg")
        from .plotting import plot_layers

        png = Path(args.out) / "layers.png"
        plot_layers(res).savefig(png, dpi=120)
        res.artifacts["layers_png"] = png
        res.save_json(Path(args.out) / "summary.json")
    print(res.to_json() if args.json else res)
    return 0 if res.ok else 1


def cmd_parse(args) -> int:
    from .gcode import read_gcode

    info = read_gcode(args.gcode)
    m = info.metrics()
    if args.json:
        print(json.dumps({"gcode": args.gcode, "generator": info.generator, "metrics": m}, indent=2))
    else:
        print(f"{args.gcode}  ({info.generator or 'unknown generator'})")
        for k, v in m.items():
            print(f"  {k:<26} {v}")
    return 0 if info.layer_count else 1


def build_parser(prog: str = "mellonia") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="3D-print manufacturability with PrusaSlicer")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("slice", help="slice an STL with explicit settings and orientation")
    s.add_argument("stl")
    s.add_argument("--settings", "-s", required=True,
                   help="settings.py[:NAME] or module:NAME defining a PrintSettings, e.g. vegeta.mellonia.examples:GENERIC_PLA_0_2MM")
    s.add_argument("--rotate-x", type=float, default=0.0, help="degrees")
    s.add_argument("--rotate-y", type=float, default=0.0, help="degrees")
    s.add_argument("--rotate-z", type=float, default=0.0, help="degrees")
    s.add_argument("--out", "-o", required=True, help="output directory")
    s.add_argument("--prusa-slicer", default=None, help='executable, e.g. "xvfb-run -a prusa-slicer"')
    s.add_argument("--timeout", type=float, default=None, help="seconds")
    s.add_argument("--png", action="store_true", help="save a per-layer extrusion plot")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_slice)

    p = sub.add_parser("parse", help="read metrics from an existing PrusaSlicer G-code file")
    p.add_argument("gcode")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_parse)
    return ap


def main(argv=None, prog: str = "mellonia") -> int:
    args = build_parser(prog).parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
