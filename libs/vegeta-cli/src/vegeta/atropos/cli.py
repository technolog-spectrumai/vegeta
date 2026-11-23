"""Command line interface: ``atropos rib|nest|info|settings``.

Settings are Python: ``settings.py`` defines an ``atropos.CutSettings`` (or use
``vegeta.atropos.examples:D1_PRO_BALSA_2MM``).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

from .settings import CutSettings


def load_settings(spec: str) -> CutSettings:
    target, _, attr = spec.partition(":")
    if target.endswith(".py") or Path(target).is_file():
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(f"no such settings file: {target}")
        name = f"_atropos_user_{abs(hash(str(path.resolve())))}"
        spec_ = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec_)
        sys.modules[name] = module
        spec_.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    if attr:
        obj = getattr(module, attr, None)
        if not isinstance(obj, CutSettings):
            raise ValueError(f"{spec} is not an atropos.CutSettings")
        return obj
    found = {k: v for k, v in vars(module).items() if isinstance(v, CutSettings)}
    if len(found) != 1:
        raise ValueError(f"{target} defines {len(found)} CutSettings {sorted(found)}; use {target}:<name>")
    return next(iter(found.values()))


def _floats(text: str, n_min: int, n_max: int, what: str) -> list[str]:
    parts = text.split(":")
    if not n_min <= len(parts) <= n_max:
        raise ValueError(f"{what}: expected {n_min}-{n_max} fields separated by ':', got {text!r}")
    return parts


def cmd_rib(args) -> int:
    from .airfoil import naca4
    from .geometry import parts_to_json
    from .rib import Lightening, Notch, RibSpec, Sheeting, TubeHole, make_rib
    from .svg import write_parts_svg

    digits = args.naca.strip()
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError(f"--naca needs four digits, e.g. 3412, got {args.naca!r}")
    up, lo = naca4(int(digits[0]) / 100, int(digits[1]) / 10, int(digits[2:]) / 100, n=args.points, te_cut=args.te_cut)
    tubes = []
    for t in args.tube or []:
        f = _floats(t, 2, 3, "--tube FRAC:DIAMETER[:CLEARANCE]")
        tubes.append(TubeHole(float(f[0]), float(f[1]), float(f[2]) if len(f) > 2 else 0.0))
    sheeting = []
    for s in args.sheeting or []:
        f = _floats(s, 3, 4, "--sheeting X0:X1:THICKNESS[:SIDE]")
        sheeting.append(Sheeting(float(f[0]), float(f[1]), float(f[2]), f[3] if len(f) > 3 else "both"))
    notches = []
    for s in args.notch or []:
        f = _floats(s, 4, 4, "--notch FRAC:SIDE:WIDTH:DEPTH")
        notches.append(Notch(float(f[0]), f[1], float(f[2]), float(f[3])))
    lightening = []
    for s in args.lightening or []:
        f = _floats(s, 3, 3, "--lightening FRAC:DIAMETER:WALL")
        lightening.append(Lightening(float(f[0]), float(f[1]), float(f[2])))
    spec = RibSpec(args.name, args.chord, up, lo, sheeting, tubes, lightening, notches, args.le_cut, args.te_cut_mm,
                   args.split, args.gap, args.bevel, meta={"naca": digits})
    parts = make_rib(spec)
    out = Path(args.out) if args.out else None
    if out is not None:
        if out.suffix.lower() == ".json":
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(parts_to_json(parts), indent=1))
        else:
            write_parts_svg(out, parts)
    info = [{"part": p.name, "width_mm": round(p.size()[0], 3), "height_mm": round(p.size()[1], 3),
             "area_mm2": round(p.area(), 3), "cut_length_mm": round(p.cut_length(), 3), "holes": len(p.holes)} for p in parts]
    if args.json:
        print(json.dumps({"out": str(out) if out else None, "parts": info}, indent=2))
    else:
        for row in info:
            print(f"{row['part']:<16} {row['width_mm']:8.2f} x {row['height_mm']:6.2f} mm  area {row['area_mm2']:9.1f} mm²  "
                  f"cut {row['cut_length_mm']:8.1f} mm  holes {row['holes']}")
        if out:
            print(f"written: {out}")
    return 0


def cmd_nest(args) -> int:
    from .geometry import parts_from_json
    from .job import cut_job

    settings = load_settings(args.settings)
    parts = parts_from_json(json.loads(Path(args.parts).read_text()))
    res = cut_job(parts, settings, args.out, name=Path(args.parts).stem, gcode=not args.no_gcode, png=not args.no_png)
    print(res.to_json() if args.json else res)
    return 0 if res.ok else 1


def cmd_info(args) -> int:
    path = Path(args.file)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    if path.suffix.lower() == ".svg":
        from .svg import read_svg_paths

        d = read_svg_paths(path)
        out = {"file": str(path), "page_mm": [d["width_mm"], d["height_mm"]], "paths": d["groups"],
               "length_mm": {k: round(v, 2) for k, v in d["length_mm"].items()}}
    else:
        from .gcode import read_gcode

        d = read_gcode(path)
        out = {"file": str(path), "bbox_mm": d["bbox"], "g1_length_mm": round(d["g1_length_mm"], 2),
               "g0_length_mm": round(d["g0_length_mm"], 2), "feeds": d["feeds"], "powers": d["powers"],
               "cut_time_s": round(d["cut_time_s"], 1)}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            print(f"  {k:<16} {v}")
    return 0


def cmd_settings(args) -> int:
    s = load_settings(args.settings)
    print(json.dumps(s.to_dict(), indent=2))
    return 0


def build_parser(prog: str = "atropos") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="flat parts for a GRBL diode laser: ribs, nesting, SVG and G-code")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("rib", help="one rib from a NACA 4-digit section (SVG or parts JSON)")
    r.add_argument("--chord", type=float, required=True, help="mm")
    r.add_argument("--naca", default="0012", help="four digits, e.g. 3412")
    r.add_argument("--name", default="rib")
    r.add_argument("--tube", action="append", metavar="FRAC:D[:CLEAR]", help="spar tube hole: chord fraction, diameter, clearance (mm)")
    r.add_argument("--sheeting", action="append", metavar="X0:X1:T[:SIDE]", help="sheeted stretch (fractions), sheet thickness, side upper/lower/both")
    r.add_argument("--notch", action="append", metavar="FRAC:SIDE:W:D", help="cap-strip slot")
    r.add_argument("--lightening", action="append", metavar="FRAC:D:WALL")
    r.add_argument("--le-cut", type=float, default=0.0, help="nose removed for the leading-edge stock [mm]")
    r.add_argument("--te-cut-mm", type=float, default=0.0, help="tail removed for the trailing-edge stock [mm]")
    r.add_argument("--split", type=float, default=None, help="hinge line as a chord fraction")
    r.add_argument("--gap", type=float, default=0.0, help="gap at the hinge [mm]")
    r.add_argument("--bevel", type=float, default=0.0, help="control surface nose bevel [deg]")
    r.add_argument("--points", type=int, default=60, help="points per surface")
    r.add_argument("--te-cut", type=float, default=0.97, help="the section's trailing-edge truncation (fraction)")
    r.add_argument("--out", "-o", default=None, help="rib.svg or parts.json")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_rib)

    n = sub.add_parser("nest", help="nest parts from a JSON file on sheets and write SVG + G-code per sheet")
    n.add_argument("parts", help="parts.json (as written by 'rib --out x.json' or atropos.parts_to_json)")
    n.add_argument("--settings", "-s", required=True,
                   help="settings.py[:NAME] or module:NAME defining a CutSettings, e.g. vegeta.atropos.examples:D1_PRO_BALSA_2MM")
    n.add_argument("--out", "-o", required=True, help="output directory")
    n.add_argument("--no-gcode", action="store_true")
    n.add_argument("--no-png", action="store_true")
    n.add_argument("--json", action="store_true")
    n.set_defaults(func=cmd_nest)

    i = sub.add_parser("info", help="read an Atropos SVG or a GRBL G-code file: size, lengths, time")
    i.add_argument("file")
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_info)

    s = sub.add_parser("settings", help="print a CutSettings as JSON")
    s.add_argument("settings")
    s.set_defaults(func=cmd_settings)
    return ap


def main(argv=None, prog: str = "atropos") -> int:
    args = build_parser(prog).parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
