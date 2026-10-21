"""Export a machine as standard files through Yax: the one path from a Vegeta design to anyone who simulates it.

    python tools/export.py drongo --to out/drongo.usda --also sdf urdf xml [--lump]
    python tools/export.py drongo --to out/drongo.usda --exclude strength,connections --consumer tools/consumers/piccolo.json
    python tools/export.py drongo --to out/drongo.usda --features export.yaml     # include/exclude from a file
    python tools/export.py drongo --to out/drongo.usda --full                     # everything, said explicitly
    python tools/export.py --list

A machine is what a lab registers in its ``labs/aero/designs/machines.py`` (``vegeta.labkit.machines``): its Chiron robot plus
the part data a simulator needs and MJCF cannot hold — rotor positions, spins and motor constants for Gazebo's motor models
(``data["ogma"]``), the propeller maps (``data["propulsion"]``, ``vegeta.labkit.propulsion``), a materials catalogue, and the
provenance (the robot's ``sources`` and ``notes``: where every number comes from). The robot goes ``to_mjcf()`` → Yax (USD
hub); the part data rides in the USD (Yax keeps it; JSBSim gets the propellers as propeller files, SDF gets the reduction
reported, the other formats report the loss). ``--lump`` writes the body as one rigid link at q = 0 minus the rotors
(Gazebo's multicopters spin separate rotor links), with each rotor's own mass and inertia in the part data.

**What leaves is chosen per feature** (Yax's contract, ``vendor/yax/docs/interchange.md``: geometry, mass, inertia, joints,
motors, materials, strength, connections, aerodynamics, sensors, provenance). ``--include`` and ``--exclude`` (or a YAML/JSON
``--features`` file with ``include`` and ``exclude`` lists) pick them independently; ``--full`` says everything, which is also
the default so existing calls are unchanged. A withheld feature reaches no file, no metadata and no side file (Yax's
``select``: withheld, never invented; the USD records what was withheld). What Vegeta has no data for is *missing*, not
invented: the inventory printed after the export says which is which. ``--consumer FILE`` (repeatable; a Yax requirements
file, e.g. ``tools/consumers/piccolo.json``) checks whether the exported files serve that consumer, per feature and per
format — a report, never a change to what was exported; ``--strict`` exits 1 when a consumer is not served.
Who receives the files maps the names with ``yax sync`` (Piccolo: ``vehicles/<app>.map.json``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def designs_dirs() -> list[Path]:
    """Every lab's ``designs/`` (``labs/*/designs``)."""
    return sorted(ROOT.glob("labs/*/designs"))


def registry():
    from vegeta.labkit import machines
    machines.discover(*designs_dirs())
    return machines


# --------------------------------------------------------------------------------------------------------- features
def features_from_file(path) -> dict:
    """``{"include": [...], "exclude": [...]}`` from a YAML (needs PyYAML) or JSON file; either key may be absent."""
    text = Path(path).read_text()
    try:
        spec = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        spec = yaml.safe_load(text)
    spec = spec or {}
    unknown = set(spec) - {"include", "exclude"}
    if unknown or not isinstance(spec, dict):
        raise ValueError(f"{path}: a features file holds 'include' and/or 'exclude' lists, not {sorted(unknown) or spec!r}")
    return {"include": list(spec.get("include") or []), "exclude": list(spec.get("exclude") or [])}


def selection(include=None, exclude=(), full: bool = False, spec: dict | None = None) -> tuple:
    """The (include, exclude) a user asked for: ``full`` is everything; the file's lists are joined with the flags'."""
    from yax import features
    inc = list(include or []) + list((spec or {}).get("include") or [])
    exc = list(exclude or []) + list((spec or {}).get("exclude") or [])
    if full and (inc or exc):
        raise ValueError("--full says everything: it cannot go with --include or --exclude")
    unknown = (set(inc) | set(exc)) - set(features.FEATURES)
    if unknown:
        raise ValueError(f"unknown features {sorted(unknown)}; the features are {features.FEATURES}")
    return (tuple(inc) if inc else None), tuple(exc)


# ----------------------------------------------------------------------------------------------------------- export
def machine(name: str, *, lump: bool = False, include=None, exclude=()):
    """The named machine as a ``yax.Machine`` (from its Chiron robot) with its part data, holding only the chosen
    features (``include``: default all; ``exclude`` wins) — the rest withheld by Yax's ``select``, never invented."""
    import yax
    from yax import features, hub
    entry = registry().get(name)
    robot, parts = entry.robot(), entry.parts
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"{name}.xml"
        p.write_text(robot.to_mjcf())
        m, rep = yax.read(p)
    m.name = name
    m.data.pop("mjcf", None)
    if parts:
        m.data.update(parts(m))                                              # {"ogma": …, "propulsion": …, "materials": …}
    prov = {k: v for k, v in (("sources", getattr(robot, "sources", None)), ("notes", getattr(robot, "notes", None))) if v}
    if prov:
        m.data["provenance"] = {"robot": robot.name, **prov}                  # where every number comes from, as the design says
    m.producer = {"tool": "vegeta.tools.export", "machine": name, "lab": entry.lab, "revision": _revision(),
                  "lumped": lump}
    if lump:
        rot = m.data.get("ogma", {}).get("rotors", [])
        m = hub.lump(m)
        if rot:                                                                # the rotors spin as their own links
            _subtract(m.links[0], rot)
    if include is not None or exclude:
        m = features.select(m, include, exclude)
    return m, rep


def _subtract(lk, rotors):
    """Take the rotors' own mass out of a lumped link (they become separate links in the simulator)."""
    from yax import hub
    Rl = hub.R(lk.quat)
    com_w = np.asarray(lk.pos) + Rl @ np.asarray(lk.com)
    J = Rl @ np.asarray(lk.inertia) @ Rl.T
    M = lk.mass
    for r in rotors:
        d = np.asarray(r["pos"]) - com_w
        J = J - (np.diag(r["inertia"]) + r["kg"] * (d @ d * np.eye(3) - np.outer(d, d)))
    mr = sum(r["kg"] for r in rotors)
    new_com = (M * com_w - sum(r["kg"] * np.asarray(r["pos"]) for r in rotors)) / (M - mr)
    d = new_com - com_w                                                        # J above is about com_w: move it
    J = J - (M - mr) * (d @ d * np.eye(3) - np.outer(d, d))
    lk.mass, lk.com, lk.inertia = float(M - mr), tuple(Rl.T @ (new_com - np.asarray(lk.pos))), Rl.T @ J @ Rl


def _revision() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def export(name: str, to, *, also=(), lump: bool = False, include=None, exclude=(), consumers=()) -> list:
    """Write the machine at ``to`` (USD recommended) and in the formats ``also`` (suffixes) beside it, holding only the
    chosen features; the reports, then — when ``consumers`` (requirements files) are given — one ``Compatibility``
    verdict per consumer and format, about the files as written."""
    m, rin = machine(name, lump=lump, include=include, exclude=exclude)
    return [rin, *write(m, to, also=also, consumers=consumers)]


def write(m, to, *, also=(), consumers=()) -> list:
    """Save a built machine (``machine()``) as ``export`` does; the writing reports and the consumers' verdicts."""
    import yax
    from yax import features
    name = m.name
    to = Path(to)
    reports = [yax.save(m, to)]
    fmts = ["usd"]
    for ext in also:
        ext = ext.lstrip(".")
        if ext == "jsbsim":                                                   # aircraft/<name>/<name>.xml, as JSBSim loads it
            reports.append(yax.save(m, to.parent / "jsbsim" / "aircraft" / name / f"{name}.xml", fmt="jsbsim"))
            fmts.append("jsbsim")
        else:
            reports.append(yax.save(m, to.with_suffix("." + ext)))
            fmts.append(yax.detect(to.with_suffix("." + ext), writing=True))
    for c in consumers:
        req = features.Requirements.load(c)
        reports += [features.check(m, req, fmt) for fmt in fmts]
    return reports


def main(argv=None) -> int:
    from yax import features
    ap = argparse.ArgumentParser(description="Export a Vegeta machine as standard files through Yax, feature by feature.")
    ap.add_argument("machine", nargs="?")
    ap.add_argument("--to", help="the main file (.usda/.usdc recommended)")
    ap.add_argument("--also", nargs="*", default=[], help="more formats beside it: sdf urdf xml step stl jsbsim (jsbsim/aircraft/<name>/)")
    ap.add_argument("--lump", action="store_true", help="one rigid body minus the rotors (multicopter simulators)")
    ap.add_argument("--include", default="", help=f"comma-separated features to export (default: all of {', '.join(features.FEATURES)})")
    ap.add_argument("--exclude", default="", help="comma-separated features to withhold (wins over --include)")
    ap.add_argument("--full", action="store_true", help="export every feature (the default, said explicitly)")
    ap.add_argument("--features", help="a YAML or JSON file with include and/or exclude lists")
    ap.add_argument("--consumer", action="append", default=[], help="a Yax requirements file (JSON): report whether the export serves it")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a consumer is not served")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list or not a.machine:
        print("\n".join(registry().names()))
        return 0
    spec = features_from_file(a.features) if a.features else None
    include, exclude = selection([x for x in a.include.split(",") if x], [x for x in a.exclude.split(",") if x], a.full, spec)
    m, rin = machine(a.machine, lump=a.lump, include=include, exclude=exclude)
    served = True
    for rep in [rin, *write(m, a.to or f"{a.machine}.usda", also=a.also, consumers=a.consumer)]:
        print(rep)
        if isinstance(rep, features.Compatibility):
            served &= rep.ok
    print("exported:")
    print("  " + features.describe(m).replace("\n", "\n  "))
    return 1 if a.strict and not served else 0


if __name__ == "__main__":
    sys.exit(main())
