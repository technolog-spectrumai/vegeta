"""Yax: design interchange between platforms through standard files.

Yax 2 converts machine descriptions between formats — USD (the hub and the universal file), MJCF, URDF, SDF, JSBSim,
STEP, STL — every conversion passing through a USD stage (docs/converter.md)::

    yax.convert("rover.xml", "rover.urdf")     # a Report of what the target cannot hold
    m, report = yax.read("c172x.xml")          # → yax.hub.Machine
    yax.save(m, "c172x.usda")

Yax 1 packages (``yax.json`` + ``body.stl``, docs/spec.md) are still written and read (``write``, ``load``, ``check``).
"""
from .manifest import BODY, NAME, VERSION, canonical, identity, validate
from .package import Design, Invalid, check, describe, load, write
from . import hub, stl, units
from .convert import FORMATS, convert, detect, info, read, save, to_stage
from .hub import Joint, Link, Machine, Report, Shape
from .sync import sync

__all__ = ["BODY", "NAME", "VERSION", "Design", "Invalid", "canonical", "check", "describe", "identity", "load",
           "stl", "units", "validate", "write", "hub", "FORMATS", "convert", "detect", "info", "read", "save", "to_stage",
           "Joint", "Link", "Machine", "Report", "Shape", "sync"]
