"""Command line interface: ``chronos spectrum|life``. Missions are Python (``file.py:NAME``)."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

from .dynamics import Structure
from .life import simulate_life
from .mission import Mission
from .spectrum import build_spectrum


def load_mission(spec: str) -> Mission:
    target, _, attr = spec.partition(":")
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {target}")
        name = f"_chronos_user_{abs(hash(str(path.resolve())))}"
        spec_ = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec_)
        sys.modules[name] = module
        spec_.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    if attr:
        obj = getattr(module, attr, None)
        if not isinstance(obj, Mission):
            raise ValueError(f"{spec} is not a chronos.Mission")
        return obj
    found = [v for v in vars(module).values() if isinstance(v, Mission)]
    if len(found) != 1:
        raise ValueError(f"{target} defines {len(found)} missions; use {target}:<name>")
    return found[0]


def _kv(items, cast=float) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not v:
            raise ValueError(f"expected name=value, got {it!r}")
        out[k] = cast(v)
    return out


def main(argv=None, prog: str = "chronos") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="Missions, load spectra and long-term life")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spectrum", help="rainflow + excitation blocks of a mission -> JSON")
    s.add_argument("mission", help="file.py:NAME (chronos.Mission)")
    s.add_argument("--modes", help="natural frequencies in Hz, comma separated (enables dynamic amplification)")
    s.add_argument("--damping", type=float, default=0.03)
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("life", help="fleet-usage simulation from damage per mission")
    s.add_argument("--damage", nargs="+", required=True, metavar="MISSION=DAMAGE")
    s.add_argument("--hours", nargs="+", required=True, metavar="MISSION=HOURS")
    s.add_argument("--usage", nargs="+", required=True, metavar="MISSION=FRACTION")
    s.add_argument("--flights", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "spectrum":
            mission = load_mission(args.mission)
            structure = Structure(tuple(float(x) for x in args.modes.split(",")), args.damping) if args.modes else None
            spec = build_spectrum(mission, structure)
            spec.save(args.out)
            if args.json:
                print(json.dumps({"mission": spec.mission, "blocks": len(spec.blocks), "cycles": spec.total_cycles, "out": args.out}))
            else:
                print(f"{spec.mission}: {len(spec.blocks)} blocks, {spec.total_cycles:.4g} cycles over {spec.duration_s / 60:.1f} min -> {args.out}")
            return 0
        sim = simulate_life(_kv(args.damage), _kv(args.hours), _kv(args.usage), args.flights, args.seed)
        out = {"flights_to_failure": sim.flights_to_failure, "hours_to_failure": sim.hours_to_failure,
               "final_damage": float(sim.damage[-1]), "flights": len(sim.flights)}
        print(json.dumps(out) if args.json else
              f"{len(sim.flights)} flights: damage {out['final_damage']:.3g}; failure after "
              f"{sim.flights_to_failure:.0f} flights / {sim.hours_to_failure:.0f} h")
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
