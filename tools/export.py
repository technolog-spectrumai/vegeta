"""Export a machine as standard files through Yax: the one path from a Vegeta design to anyone who simulates it.

    python tools/export.py drongo --to out/drongo.usda --also sdf urdf xml [--lump]
    python tools/export.py --list

A machine is what a lab registers in its ``designs/machines.py`` (``vegeta.labkit.machines``): its Chiron robot plus
the part data a simulator needs and MJCF cannot hold — rotor positions, spins and motor constants for Gazebo's motor models, from Ogma's laws and the
design's propulsion. The robot goes ``to_mjcf()`` → Yax (USD hub); the part data rides in the USD as ``data["ogma"]``
(Yax keeps it; formats that cannot hold it report it). ``--lump`` writes the body as one rigid link at q = 0 minus the
rotors (Gazebo's multicopters spin separate rotor links), with each rotor's own mass and inertia in the part data.
Who receives the files maps the names with ``yax sync`` (Piccolo: ``vehicles/<app>.map.json``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def designs_dirs() -> list[Path]:
    """Every lab's ``designs/`` (``labs/*/designs``; the old top-level ``designs/`` while it exists)."""
    return sorted(ROOT.glob("labs/*/designs")) + ([ROOT / "designs"] if (ROOT / "designs" / "machines.py").exists() else [])


def registry():
    from vegeta.labkit import machines
    machines.discover(*designs_dirs())
    return machines


# ----------------------------------------------------------------------------------------------------------- export
def machine(name: str, *, lump: bool = False):
    """The named machine as a ``yax.Machine`` (from its Chiron robot) with its part data in ``data["ogma"]``."""
    import yax
    from yax import hub
    entry = registry().get(name)
    robot, parts = entry.robot(), entry.parts
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"{name}.xml"
        p.write_text(robot.to_mjcf())
        m, rep = yax.read(p)
    m.name = name
    m.data.pop("mjcf", None)
    if parts:
        m.data["ogma"] = parts(m)
    m.producer = {"tool": "vegeta.tools.export", "machine": name, "lab": entry.lab, "revision": _revision(),
                  "lumped": lump}
    if lump:
        rot = m.data.get("ogma", {}).get("rotors", [])
        m = hub.lump(m)
        if rot:                                                                # the rotors spin as their own links
            _subtract(m.links[0], rot)
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


def export(name: str, to, *, also=(), lump: bool = False) -> list:
    """Write the machine at ``to`` (USD recommended) and in the formats ``also`` (suffixes) beside it; the reports."""
    import yax
    m, rin = machine(name, lump=lump)
    to = Path(to)
    reports = [rin, yax.save(m, to)]
    for ext in also:
        reports.append(yax.save(m, to.with_suffix("." + ext.lstrip("."))))
    return reports


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export a Vegeta machine as standard files through Yax.")
    ap.add_argument("machine", nargs="?")
    ap.add_argument("--to", help="the main file (.usda/.usdc recommended)")
    ap.add_argument("--also", nargs="*", default=[], help="more formats beside it: sdf urdf xml step stl")
    ap.add_argument("--lump", action="store_true", help="one rigid body minus the rotors (multicopter simulators)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list or not a.machine:
        print("\n".join(registry().names()))
        return 0
    for rep in export(a.machine, a.to or f"{a.machine}.usda", also=a.also, lump=a.lump):
        print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
