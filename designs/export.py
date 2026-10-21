"""Export a machine as standard files through Yax: the one path from a Vegeta design to anyone who simulates it.

    python designs/export.py drongo --to out/drongo.usda --also sdf urdf xml [--lump]
    python designs/export.py --list

A machine is its Chiron robot (``MACHINES``: the builders that already exist) plus the part data a simulator needs and
MJCF cannot hold — rotor positions, spins and motor constants for Gazebo's motor models, from Ogma's laws and the
design's propulsion. The robot goes ``to_mjcf()`` → Yax (USD hub); the part data rides in the USD as ``data["ogma"]``
(Yax keeps it; formats that cannot hold it report it). ``--lump`` writes the body as one rigid link at q = 0 minus the
rotors (Gazebo's multicopters spin separate rotor links), with each rotor's own mass and inertia in the part data.
Who receives the files maps the names with ``yax sync`` (Piccolo: ``vehicles/<app>.map.json``).
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# --------------------------------------------------------------------------------------------------------- machines
def _drongo():
    import drongo_robot as dr
    return dr.drongo()


def _drongo_parts(m) -> dict:
    """Drongo's rotors for a multicopter simulator: positions in the machine frame, spin (+1: drag torque +z on the
    frame), the propeller's mass and disk inertia, the motor model's constants. ω_max: 80 % of the no-load speed
    (KV × V) of the 2306 2400 KV on 4S (assumed); kF = T_max / ω_max²; kM = drag torque / thrust; τ = spin-up."""
    import drongo_robot as dr
    prop = dr.propulsion()
    pos, spin = dr.rotor_layout()
    g = dr.geometry()
    root = m.roots()[0]
    Rr = _R(root.quat)
    w_max = 0.8 * 2400.0 * 4 * 3.7 * 2 * math.pi / 60
    kg = dr.PARTS_G["propellers 5x4.3 (4x)"] / 4 / 1000.0
    r = g["prop_r"]
    rotors = []
    for p, s in zip(np.asarray(pos, float), np.asarray(spin, float)):
        rotors.append({"pos": (np.asarray(root.pos) + Rr @ p).tolist(), "spin": int(s), "kg": kg,
                       "inertia": [kg * r * r / 4, kg * r * r / 4, kg * r * r / 2], "radius": r,
                       "kF": prop.max_thrust_N / w_max ** 2, "kM": prop.torque_per_thrust_m, "w_max": w_max,
                       "tau_up": prop.spin_up_s, "tau_down": 2 * prop.spin_up_s, "T_max": prop.max_thrust_N})
    return {"rotors": rotors, "source": f"{prop.source}; ω_max = 0.8 × KV × V (2400 KV, 4S; assumed)"}


def _builder(module: str, fn: str):
    def build():
        import importlib
        return getattr(importlib.import_module(module), fn)()
    return build


MACHINES = {                                          # name → (Chiron robot builder, part data or None)
    "drongo": (_drongo, _drongo_parts),
    **{name: (_builder(mod, fn), None) for name, (mod, fn) in {
        "apheloria": ("apheloria_robot", "apheloria"), "falco": ("falco_robot", "falco_robot"),
        "myropod": ("myropod_robot", "cleopatra"), "nisus": ("nisus_robot", "nisus_robot"),
        "nisus_plus": ("nisus_plus_robot", "nisus_plus_robot"), "onager": ("onager_robot", "onager"),
        "onager_atlas": ("onager_atlas_robot", "atlas"), "onager_manus": ("onager_manus_robot", "manus"),
        "onager_sweeper": ("onager_sweeper_robot", "sweeper"), "pekari": ("pekari_rover_robot", "pekari"),
        "perseus": ("perseus_robot", "perseus"), "perseus_mini": ("perseus_mini_robot", "mini"),
        "robot_dog": ("robot_dog_robot", "dog_robot")}.items()},
}


def _R(q):
    from yax import hub
    return hub.R(q)


# ----------------------------------------------------------------------------------------------------------- export
def machine(name: str, *, lump: bool = False):
    """The named machine as a ``yax.Machine`` (from its Chiron robot) with its part data in ``data["ogma"]``."""
    import yax
    from yax import hub
    build, parts = MACHINES[name]
    robot = build()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"{name}.xml"
        p.write_text(robot.to_mjcf())
        m, rep = yax.read(p)
    m.name = name
    m.data.pop("mjcf", None)
    if parts:
        m.data["ogma"] = parts(m)
    m.producer = {"tool": "vegeta.designs.export", "machine": name, "revision": _revision(), "lumped": lump}
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
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
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
        print("\n".join(sorted(MACHINES)))
        return 0
    for rep in export(a.machine, a.to or f"{a.machine}.usda", also=a.also, lump=a.lump):
        print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
