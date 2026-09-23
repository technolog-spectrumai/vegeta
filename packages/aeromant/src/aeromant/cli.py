"""Command line interface: ``aeromant templates|prepare|run|results``.

Cases are Python: ``case.py`` defines ``case = aeromant.CFDCase(...)``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .case import CFDCase, read_case_results
from .templates import get_template, list_templates


def load_case(spec: str) -> CFDCase:
    target, _, attr = spec.partition(":")
    path = Path(target)
    if not path.is_file():
        raise FileNotFoundError(f"no such case file: {target}")
    name = f"_aeromant_user_{abs(hash(str(path.resolve())))}"
    spec_ = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec_)
    sys.modules[name] = module
    spec_.loader.exec_module(module)
    if attr:
        obj = getattr(module, attr, None)
        if not isinstance(obj, CFDCase):
            raise ValueError(f"{target}:{attr} is not an aeromant.CFDCase")
        return obj
    found = {k: v for k, v in vars(module).items() if isinstance(v, CFDCase)}
    if len(found) != 1:
        raise ValueError(f"{target} defines {len(found)} CFDCase objects {sorted(found)}; use {target}:<name>")
    return next(iter(found.values()))


def _emit(res, as_json: bool) -> int:
    print(res.to_json() if as_json else res)
    return 0 if res.ok else 1


def cmd_templates(args) -> int:
    names = [args.name] if args.name else list_templates()
    if args.json:
        out = []
        for n in names:
            t = get_template(n)
            out.append({"name": t.name, "description": t.description, "openfoam_flavor": t.openfoam_flavor,
                        "pipeline": t.step_names, "notes": list(t.notes),
                        "parameters": [{"name": p.name, "units": p.units, "required": p.required,
                                        "default": None if p.required else p.default, "description": p.description}
                                       for p in t.parameters]})
        print(json.dumps(out, indent=2))
    else:
        print("\n\n".join(get_template(n).describe() for n in names))
    return 0


def cmd_prepare(args) -> int:
    return _emit(load_case(args.case).prepare(overwrite=args.overwrite), args.json)


def cmd_run(args) -> int:
    case = load_case(args.case)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()] if args.steps else None
    res = case.run(steps=steps, timeout=args.timeout, progress=not (args.json or args.quiet))
    return _emit(res, args.json)


def cmd_results(args) -> int:
    res = read_case_results(args.casedir, average_window=args.window)
    if res.ok and args.png:
        import matplotlib

        matplotlib.use("Agg")
        from .plotting import plot_coefficients, plot_residuals

        for name, fn in (("coefficients", plot_coefficients), ("residuals", plot_residuals)):
            try:
                out = Path(args.casedir) / f"{name}.png"
                fn(args.casedir).savefig(out, dpi=120)
                res.artifacts[f"{name}_png"] = out
            except (FileNotFoundError, ValueError) as exc:
                res.messages.append(f"{name} plot not written: {exc}")
    return _emit(res, args.json)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aeromant", description="External aerodynamics with OpenFOAM template cases")
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("templates", help="list templates and their required parameters")
    t.add_argument("name", nargs="?")
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=cmd_templates)

    p = sub.add_parser("prepare", help="write the case directory from template, STL and explicit values")
    p.add_argument("case", help="case.py[:name] defining an aeromant.CFDCase")
    p.add_argument("--overwrite", action="store_true", help="replace an existing case directory")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_prepare)

    r = sub.add_parser("run", help="run the template pipeline (or selected steps) on a prepared case")
    r.add_argument("case", help="case.py[:name] defining an aeromant.CFDCase")
    r.add_argument("--steps", help="comma separated subset, e.g. blockMesh,snappyHexMesh,checkMesh")
    r.add_argument("--timeout", type=float, default=None, help="seconds per step")
    r.add_argument("--json", action="store_true")
    r.add_argument("--quiet", "-q", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("results", help="summarise an existing case directory (never runs anything)")
    s.add_argument("casedir")
    s.add_argument("--window", type=int, default=50, help="iterations averaged for *_mean values")
    s.add_argument("--png", action="store_true", help="save coefficient and residual plots")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_results)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"aeromant: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
