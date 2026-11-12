"""Command line interface: ``talos inspect|mesh|solve|results``.

Models are Python: ``model.py`` defines ``model = talos.StructuralModel(...)``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .model import StructuralModel


def load_model(spec: str) -> StructuralModel:
    """Load ``model.py[:name]``; without ``name`` the file must define exactly one StructuralModel."""
    target, _, attr = spec.partition(":")
    path = Path(target)
    if not path.is_file():
        raise FileNotFoundError(f"no such model file: {target}")
    mod_name = f"_talos_user_{abs(hash(str(path.resolve())))}"
    spec_ = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec_)
    sys.modules[mod_name] = module
    spec_.loader.exec_module(module)
    if attr:
        obj = getattr(module, attr, None)
        if not isinstance(obj, StructuralModel):
            raise ValueError(f"{target}:{attr} is not a talos.StructuralModel")
        return obj
    found = {k: v for k, v in vars(module).items() if isinstance(v, StructuralModel)}
    if len(found) != 1:
        raise ValueError(f"{target} defines {len(found)} StructuralModel objects {sorted(found)}; use {target}:<name>")
    return next(iter(found.values()))


def _emit(res, as_json: bool) -> int:
    print(res.to_json() if as_json else res)
    return 0 if res.ok else 1


def cmd_inspect(args) -> int:
    from .stepinfo import inspect_step

    info = inspect_step(args.step, args.units)
    if args.json:
        print(json.dumps(info.to_dict(), indent=2))
    else:
        print(f"{info.path}  [{info.units}]  bbox {', '.join(f'{v:.4g}' for v in info.bbox)}")
        for v in info.volumes:
            print(f"  volume {v['tag']}: {v['volume']:.6g}")
        print(info.table())
    return 0


def cmd_mesh(args) -> int:
    model = load_model(args.model)
    return _emit(model.mesh(args.workdir, progress=not args.json and not args.quiet), args.json)


def cmd_solve(args) -> int:
    model = load_model(args.model)
    res = model.solve(args.workdir, executable=args.ccx, threads=args.threads, timeout=args.timeout,
                      progress=not args.json and not args.quiet)
    if res.ok and args.png:
        import matplotlib

        matplotlib.use("Agg")
        from .plotting import plot_deformed, plot_von_mises_histogram

        for name, fn in (("deformed", plot_deformed), ("von_mises_hist", plot_von_mises_histogram)):
            fig = fn(res)
            out = Path(args.workdir) / f"{name}.png"
            fig.savefig(out, dpi=120)
            res.artifacts[f"{name}_png"] = out
        res.save_json(Path(args.workdir) / "summary.json")
    return _emit(res, args.json)


def cmd_results(args) -> int:
    summary = Path(args.workdir) / "summary.json"
    if not summary.is_file():
        print(f"talos: no summary.json in {args.workdir} (NOT RUN)", file=sys.stderr)
        return 1
    data = json.loads(summary.read_text())
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"{data['kind']}: {data['status'].upper()}")
        for k, v in data["metrics"].items():
            print(f"  {k:<28} {v}")
        for msg in data["messages"]:
            print(f"  - {msg}")
    return 0 if data["status"] == "success" else 1


def build_parser(prog: str = "talos") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Linear static FEA with Gmsh + CalculiX")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="list surfaces/volumes of a STEP file for region selection")
    p.add_argument("step")
    p.add_argument("--units", required=True, help="unit system: mm-N-MPa or m-N-Pa")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_inspect)

    for name, func, helptext in (("mesh", cmd_mesh, "mesh the model geometry with Gmsh"),
                                 ("solve", cmd_solve, "write the CalculiX deck from the existing mesh and solve")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("model", help="model.py[:name] defining a talos.StructuralModel")
        s.add_argument("--workdir", "-w", required=True, help="analysis directory (artifacts are kept here)")
        s.add_argument("--json", action="store_true")
        s.add_argument("--quiet", "-q", action="store_true", help="no progress bar")
        if name == "solve":
            s.add_argument("--ccx", default="ccx", help="CalculiX executable")
            s.add_argument("--threads", type=int, default=1)
            s.add_argument("--timeout", type=float, default=None, help="seconds")
            s.add_argument("--png", action="store_true", help="save deformed-shape and stress histogram PNGs")
        s.set_defaults(func=func)

    r = sub.add_parser("results", help="show the summary of an existing analysis directory")
    r.add_argument("workdir")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_results)
    return ap


def main(argv=None, prog: str = "talos") -> int:
    args = build_parser(prog).parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
