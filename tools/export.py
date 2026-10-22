"""Export a machine as standard files through Yax: the one path from a Vegeta design to anyone who simulates it.

    python tools/export.py drongo --to out/drongo.yx2 --also sdf urdf xml [--lump]
    python tools/export.py drongo --to out/drongo.yx2 --drop 00strgth,00conn --consumer tools/consumers/piccolo.json
    python tools/export.py drongo --to out/drongo.yx2 --keep 00geom,00mass,00inert,00joint    # the structure alone
    python tools/export.py drongo --to out/drongo.yx2 --tag 'data:propulsion/*=boreas'        # a producer tag
    python tools/export.py drongo --to out/drongo.yx2 --tags export.yaml                      # keep/drop/tag from a file
    python tools/export.py drongo --to out/drongo.yx2 --full                                  # everything, said explicitly
    python tools/export.py pekari --to out/pekari.yx2 --no-visual                       # without the CAD looks
    python tools/export.py --list

The main file is a **YX2** (``.yx2``, the Yax 2 exchange file: a validated USD subset with the units, the frame, stable
component ids, the tag table and a canonical text — ``vendor/yax/docs/yx2.md``); ``.usda`` gives the plain USD stage
as before. Vegeta → YX2 → Yax → YX2 keeps every selected property byte for byte (``tools/tests/test_export.py``).

A machine is what a lab registers in its ``labs/aero/designs/machines.py`` (``vegeta.labkit.machines``): its Chiron robot plus
the part data a simulator needs and MJCF cannot hold — rotor positions, spins and motor constants for Gazebo's motor models
(``data["ogma"]``), the propeller maps (``data["propulsion"]``, ``vegeta.labkit.propulsion``), a materials catalogue, and the
provenance (the robot's ``sources`` and ``notes``: where every number comes from). The robot goes ``to_mjcf()`` → Yax (USD
hub); the part data rides in the file (Yax keeps it; JSBSim gets the propellers as propeller files, SDF gets the reduction
reported, the other formats report the loss). ``--lump`` writes the body as one rigid link at q = 0 minus the rotors
(Gazebo's multicopters spin separate rotor links), with each rotor's own mass and inertia in the part data.

**What leaves is chosen by tag** (Yax's contract, ``vendor/yax/docs/interchange.md``). Every property has a path
(``link:drongo/mass``, ``joint:jaw_L/drive``, ``data:propulsion/propellers/apc``, ``producer``) and carries tags: codes of
1 to 8 alphanumerics. Yax assigns the **system tags** (``00core`` the skeleton, ``00geom``, ``00mass``, ``00inert``, ``00joint``,
``00motor``, ``00matl``, ``00strgth``, ``00conn``, ``00aero``, ``00sensor``, ``00prov``, ``00fmt``) from what a property is;
the designs add their own through their parts functions (``boreas`` on the maps a BEMT made, ``nb08`` on notebook 08's
numbers) and the user adds more with ``--tag PATTERN=CODE``. ``--keep`` and ``--drop`` (or a YAML/JSON ``--tags`` file with
``keep``, ``drop`` and ``tag`` entries) choose by tag; ``--full`` says everything, which is also the default so existing calls
are unchanged. A dropped property reaches no file, no metadata and no side file (Yax's ``select``: withheld, never
invented; the file records the table and what was withheld). What Vegeta has no data for is *absent*, not invented: the
inventory printed after the export says which tags hold how many properties. ``--consumer FILE`` (repeatable; a Yax
requirements file of tag codes, e.g. ``tools/consumers/piccolo.json``) checks whether the exported files serve that
consumer, per tag and per format — a report, never a change to what was exported; ``--strict`` exits 1 when a consumer is
not served. Who receives the files maps the names with ``yax sync`` (Piccolo: ``vehicles/<app>.map.json``).
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


# ------------------------------------------------------------------------------------------------------------- tags
def tags_from_file(path) -> dict:
    """``{"keep": [...], "drop": [...], "tag": {pattern: code | [codes]}}`` from a YAML (needs PyYAML) or JSON file;
    any key may be absent."""
    text = Path(path).read_text()
    try:
        spec = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        spec = yaml.safe_load(text)
    spec = spec or {}
    unknown = set(spec) - {"keep", "drop", "tag"} if isinstance(spec, dict) else {"?"}
    if unknown:
        raise ValueError(f"{path}: a tags file holds 'keep' and 'drop' lists and a 'tag' mapping, not {sorted(unknown)}")
    tag = {k: ([v] if isinstance(v, str) else list(v)) for k, v in (spec.get("tag") or {}).items()}
    return {"keep": list(spec.get("keep") or []), "drop": list(spec.get("drop") or []), "tag": tag}


def selection(keep=None, drop=(), full: bool = False, spec: dict | None = None, tag=()) -> tuple:
    """The (keep, drop, tag) a user asked for: ``full`` is everything; the file's entries join the flags'. ``tag`` is a
    list of ``PATTERN=CODE`` strings; the result's third item is ``[(pattern, code), …]``."""
    from yax import tags as tg
    k = list(keep or []) + list((spec or {}).get("keep") or [])
    d = list(drop or []) + list((spec or {}).get("drop") or [])
    if full and (k or d):
        raise ValueError("--full says everything: it cannot go with --keep or --drop")
    for c in (*k, *d):
        tg.check_code(c, system_ok=True)
    pairs = []
    for s in tag:
        pattern, sep, code = s.rpartition("=")
        if not sep or not pattern:
            raise ValueError(f"--tag wants PATTERN=CODE, got {s!r}")
        pairs.append((pattern, tg.check_code(code)))
    for pattern, codes in ((spec or {}).get("tag") or {}).items():
        pairs += [(pattern, tg.check_code(c)) for c in codes]
    return (tuple(k) if k else None), tuple(d), pairs


# ----------------------------------------------------------------------------------------------------------- export
def machine(name: str, *, lump: bool = False, keep=None, drop=(), tag=(), visual: bool = True):
    """The named machine as a ``yax.Machine`` (from its Chiron robot) with its part data, its tags (the system tags,
    the design's, and ``tag``'s ``(pattern, code)`` pairs) and its CAD design as a visual mesh (``visual``), holding
    only the properties whose tags meet ``keep`` (default: all) and none of ``drop`` — the rest withheld by Yax's
    ``select``, never invented."""
    import yax
    from yax import hub
    from yax import tags as tg
    entry = registry().get(name)
    robot, parts = entry.robot(), entry.parts
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"{name}.xml"
        p.write_text(robot.to_mjcf())
        m, rep = yax.read(p)
    m.name = name
    m.data.pop("mjcf", None)
    from vegeta.labkit import motors as lm
    lm.attach(m, robot)                                                       # the servos as motor blocks (00motor), not just PD drives
    design_tags = {}
    if parts:
        part = parts(m)                                                       # {"ogma": …, "propulsion": …, "materials": …, "tags": …}
        design_tags = part.pop("tags", {}) or {}
        m.data.update(part)
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
    if visual and entry.visual is not None:                                   # the CAD design, the machine's looks
        attach_visual(m, entry.visual)
    for pattern, codes in design_tags.items():                                # the design's own tags
        for code in ([codes] if isinstance(codes, str) else codes):
            tg.tag(m, pattern, code)
    for pattern, code in tag:                                                 # the user's
        tg.tag(m, pattern, code)
    if keep is not None or drop:
        m = tg.select(m, keep, drop)
    return m, rep


def attach_visual(m, design, max_triangles: int | None = None):
    """The design's CAD as the root link's visual mesh ``cad`` (``vegeta.labkit.visual``: light, placed over the
    robot's shapes); Yax tags it ``00geom``."""
    from yax import hub
    from vegeta.labkit import visual as lv
    pts, faces = lv.mesh(design, max_triangles or lv.MAX_TRIANGLES)
    uri = design if isinstance(design, str) else getattr(design, "__name__", type(design).__name__)
    m.links[0].shapes.append(hub.Shape(name="cad", kind="mesh", role="visual", points=lv.place(pts, m), faces=faces,
                                       uri=f"vegeta designs:{uri}"))
    return m


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


def export(name: str, to, *, also=(), lump: bool = False, keep=None, drop=(), tag=(), consumers=(),
           visual: bool = True) -> list:
    """Write the machine at ``to`` (``.yx2`` recommended) and in the formats ``also`` (suffixes) beside it, holding only the
    properties chosen by tag; the reports, then — when ``consumers`` (requirements files) are given — one ``Compatibility``
    verdict per consumer and format, about the files as written."""
    m, rin = machine(name, lump=lump, keep=keep, drop=drop, tag=tag, visual=visual)
    return [rin, *write(m, to, also=also, consumers=consumers)]


def write(m, to, *, also=(), consumers=()) -> list:
    """Save a built machine (``machine()``) as ``export`` does; the writing reports and the consumers' verdicts."""
    import yax
    from yax import tags as tg
    name = m.name
    to = Path(to)
    reports = [yax.save(m, to)]
    fmts = [yax.detect(to, writing=True)]
    for ext in also:
        ext = ext.lstrip(".")
        if ext == "jsbsim":                                                   # aircraft/<name>/<name>.xml, as JSBSim loads it
            reports.append(yax.save(m, to.parent / "jsbsim" / "aircraft" / name / f"{name}.xml", fmt="jsbsim"))
            fmts.append("jsbsim")
        else:
            reports.append(yax.save(m, to.with_suffix("." + ext)))
            fmts.append(yax.detect(to.with_suffix("." + ext), writing=True))
    for c in consumers:
        req = tg.Requirements.load(c)
        reports += [tg.check(m, req, fmt) for fmt in fmts]
    return reports


def main(argv=None) -> int:
    from yax import tags as tg
    ap = argparse.ArgumentParser(description="Export a Vegeta machine as standard files through Yax, property by property, by tag.")
    ap.add_argument("machine", nargs="?")
    ap.add_argument("--to", help="the main file (default <machine>.yx2; .usda for plain USD)")
    ap.add_argument("--also", nargs="*", default=[], help="more formats beside it: sdf urdf xml step stl jsbsim (jsbsim/aircraft/<name>/)")
    ap.add_argument("--lump", action="store_true", help="one rigid body minus the rotors (multicopter simulators)")
    ap.add_argument("--keep", default="", help=f"comma-separated tags to export (default: every tag; system tags: {', '.join(tg.SYSTEM)})")
    ap.add_argument("--drop", default="", help="comma-separated tags to withhold (wins over --keep)")
    ap.add_argument("--tag", action="append", default=[], metavar="PATTERN=CODE", help="tag the properties matching PATTERN with CODE (repeatable)")
    ap.add_argument("--full", action="store_true", help="export every property (the default, said explicitly)")
    ap.add_argument("--tags", help="a YAML or JSON file with keep and drop lists and a tag mapping")
    ap.add_argument("--consumer", action="append", default=[], help="a Yax requirements file (JSON): report whether the export serves it")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a consumer is not served")
    ap.add_argument("--no-visual", action="store_true", help="leave out the CAD design's visual mesh")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list or not a.machine:
        print("\n".join(registry().names()))
        return 0
    spec = tags_from_file(a.tags) if a.tags else None
    keep, drop, tag = selection([x for x in a.keep.split(",") if x], [x for x in a.drop.split(",") if x], a.full, spec, a.tag)
    m, rin = machine(a.machine, lump=a.lump, keep=keep, drop=drop, tag=tag, visual=not a.no_visual)
    served = True
    for rep in [rin, *write(m, a.to or f"{a.machine}.yx2", also=a.also, consumers=a.consumer)]:
        print(rep)
        if isinstance(rep, tg.Compatibility):
            served &= rep.ok
    print("exported:")
    print("  " + tg.describe(m).replace("\n", "\n  "))
    return 1 if a.strict and not served else 0


if __name__ == "__main__":
    sys.exit(main())
