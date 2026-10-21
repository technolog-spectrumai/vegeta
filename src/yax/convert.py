"""Reading, writing and converting machine files: every conversion goes through a USD stage (the hub).

    yax.convert("rover.xml", "rover.urdf")            # MJCF → USD → URDF; returns a Report of what URDF cannot hold
    m, report = yax.read("c172x.xml")                 # → a hub.Machine (JSBSim detected from <fdm_config>)
    yax.save(m, "c172x.yx2")                          # the exchange file (a validated USD subset, docs/yx2.md); .usda: plain USD
"""
from __future__ import annotations

import importlib
import xml.etree.ElementTree as ET
from pathlib import Path

from . import hub
from .hub import Machine, Report

FORMATS = ("usd", "yx2", "mjcf", "urdf", "sdf", "jsbsim", "step", "stl", "package")
_MODULE = {"yx2": "yx2", "mjcf": "mjcf", "urdf": "urdf", "sdf": "sdf", "jsbsim": "jsbsim", "step": "step", "stl": "stlfile", "package": "package"}
_SUFFIX = {".usd": "usd", ".usda": "usd", ".usdc": "usd", ".yx2": "yx2", ".urdf": "urdf", ".sdf": "sdf", ".step": "step", ".stp": "step", ".stl": "stl"}
_TAG = {"mujoco": "mjcf", "fdm_config": "jsbsim", "robot": "urdf", "sdf": "sdf"}


def detect(path, *, writing: bool = False) -> str:
    """The format of ``path`` by its suffix (and, for ``.xml``, by its root element; a new ``.xml`` is MJCF)."""
    path = Path(path)
    if path.is_dir() and (path / "yax.json").exists():
        return "package"
    fmt = _SUFFIX.get(path.suffix.lower())
    if fmt:
        return fmt
    if path.suffix.lower() == ".xml":
        if writing or not path.exists():
            return "mjcf"
        tag = next(ET.iterparse(str(path), events=("start",)))[1].tag
        if tag in _TAG:
            return _TAG[tag]
        raise ValueError(f"{path}: an XML file whose root <{tag}> is no known machine format")
    raise ValueError(f"{path}: unknown format (give fmt= one of {FORMATS})")


def _module(fmt):
    return importlib.import_module(f".formats.{_MODULE[fmt]}", __package__)


def read(path, fmt: str | None = None, **opts) -> tuple[Machine, Report]:
    """A file as a ``hub.Machine`` and the report of what it could not carry into the hub."""
    fmt = fmt or detect(path)
    if fmt == "usd":
        from pxr import Usd
        return hub.from_stage(Usd.Stage.Open(str(path))), Report(str(path), "usd")
    return _module(fmt).read(path, **opts)


def save(m: Machine, path, fmt: str | None = None, *, report: Report | None = None, **opts) -> Report:
    """Write ``m`` in ``fmt`` (by default from the suffix); returns what the format could not hold."""
    fmt = fmt or detect(path, writing=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rep = report or Report("usd", str(path))
    if fmt == "usd":
        if path.exists():
            path.unlink()
        hub.to_stage(m, path)
        return rep
    from .features import report_excluded
    report_excluded(m, rep)                                                   # a withheld feature is a loss to know about
    return _module(fmt).write(m, path, rep, **opts)


def to_stage(path, fmt: str | None = None, **opts):
    """A file as a USD stage (in memory) and the reading report."""
    m, rep = read(path, fmt, **opts)
    return hub.to_stage(m), rep


def convert(src, dst, *, src_fmt: str | None = None, dst_fmt: str | None = None, read_opts: dict | None = None,
            write_opts: dict | None = None) -> Report:
    """``src`` → USD → ``dst``. The report lists what was dropped on the way in and on the way out."""
    m, rin = read(src, src_fmt, **(read_opts or {}))
    stage = hub.to_stage(m)
    m2 = hub.from_stage(stage)
    rout = Report(str(src), str(dst), dropped=[f"reading: {d}" for d in rin.dropped])
    save(m2, dst, dst_fmt, report=rout, **(write_opts or {}))
    return rout


def info(path, fmt: str | None = None) -> str:
    m, rep = read(path, fmt)
    lines = [f"{m.name}: {len(m.links)} links, {len(m.joints)} joints, {m.total_mass():.6g} kg"
             + (f"; data: {', '.join(sorted(m.data))}" if m.data else "")
             + (f"; excluded: {', '.join(m.features['excluded'])}" if m.features.get("excluded") else "")]
    for lk in m.links:
        lines.append(f"  link {lk.name}: {lk.mass if lk.mass is not None else '—'} kg, "
                     f"{len(lk.shapes)} shapes ({', '.join(s.kind for s in lk.shapes)})")
    for j in m.joints:
        lines.append(f"  joint {j.name}: {j.kind} {j.parent or 'world'} → {j.child}")
    if rep.dropped:
        lines.append(str(rep))
    return "\n".join(lines)
