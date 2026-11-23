"""Command line interface: ``boreas point|map|for-thrust``.

Propellers are given as ``file.py:NAME`` (a ``boreas.Propeller``) or inline with ``--dp 10x4.7``
(inches, constant pitch, generic planform). Airfoil, motor and battery likewise come from a Python file.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

from .airfoil import Airfoil
from .bemt import rpm_for_thrust, solve
from .export import export, performance_map
from .propeller import Propeller, inches


def load_object(spec: str, cls):
    target, _, attr = spec.partition(":")
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {target}")
        name = f"_boreas_user_{abs(hash(str(path.resolve())))}"
        spec_ = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec_)
        sys.modules[name] = module
        spec_.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    if attr:
        obj = getattr(module, attr, None)
        if not isinstance(obj, cls):
            raise ValueError(f"{spec} is not a boreas.{cls.__name__}")
        return obj
    found = [v for v in vars(module).values() if isinstance(v, cls)]
    if len(found) != 1:
        raise ValueError(f"{target} defines {len(found)} {cls.__name__} objects; use {target}:<name>")
    return found[0]


def _prop(args) -> Propeller:
    if args.propeller:
        return load_object(args.propeller, Propeller)
    d, p = (float(x) for x in args.dp.lower().split("x"))
    dm, pm = inches(d, p)
    return Propeller.from_pitch(f"{args.dp} (generic planform)", dm, pm, args.blades, chord_root_m=0.07 * dm,
                                chord_max_m=0.10 * dm, chord_tip_m=0.03 * dm)


def _add_common(sp):
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("-p", "--propeller", help="file.py:NAME (boreas.Propeller)")
    g.add_argument("--dp", help="diameter x pitch in inches, e.g. 10x4.7 (generic planform)")
    sp.add_argument("--blades", type=int, default=2)
    sp.add_argument("--airfoil", help="file.py:NAME (boreas.Airfoil); default generic section")
    sp.add_argument("--rho", type=float, default=1.225)
    sp.add_argument("--json", action="store_true", help="machine-readable output")


def main(argv=None, prog: str = "boreas") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="Propeller performance by blade element momentum theory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("point", help="one operating point")
    _add_common(s)
    s.add_argument("--rpm", type=float, required=True)
    s.add_argument("--speed", type=float, default=0.0, help="axial airspeed [m/s]")
    s = sub.add_parser("for-thrust", help="rpm for a required thrust")
    _add_common(s)
    s.add_argument("--thrust", type=float, required=True, help="[N]")
    s.add_argument("--speed", type=float, default=0.0)
    s = sub.add_parser("map", help="rpm x airspeed map, exported to JSON")
    _add_common(s)
    s.add_argument("--rpm", nargs=3, type=float, metavar=("START", "STOP", "N"), required=True)
    s.add_argument("--speed", nargs=3, type=float, metavar=("START", "STOP", "N"), default=(0, 0, 1))
    s.add_argument("-o", "--out", required=True, help="output JSON")
    args = ap.parse_args(argv)
    try:
        prop = _prop(args)
        airfoil = load_object(args.airfoil, Airfoil) if args.airfoil else Airfoil()
        if args.cmd == "point":
            op = solve(prop, airfoil, args.rpm, args.speed, args.rho)
        elif args.cmd == "for-thrust":
            op = rpm_for_thrust(prop, airfoil, args.thrust, args.speed, args.rho)
        else:
            import numpy as np

            grid = performance_map(prop, airfoil, np.linspace(args.rpm[0], args.rpm[1], int(args.rpm[2])),
                                   np.linspace(args.speed[0], args.speed[1], int(args.speed[2])), args.rho)
            res = export(args.out, prop, airfoil, map=grid, rho=args.rho)
            print(res.to_json() if args.json else res)
            return 0 if res.ok else 1
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    d = {k: v for k, v in op.to_dict().items() if k != "radial"}
    if args.json:
        print(json.dumps(d, indent=2))
    else:
        print(f"{prop.name}: {op.rpm:.0f} rpm, {op.airspeed:g} m/s")
        for k in ("thrust", "torque", "power", "efficiency", "figure_of_merit", "ct", "cp", "advance_ratio", "tip_mach"):
            print(f"  {k:<16} {d[k]:.4g}")
        if not op.converged:
            print("  WARNING: not converged")
    return 0
