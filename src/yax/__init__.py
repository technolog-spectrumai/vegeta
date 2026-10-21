"""Yax: design interchange. A package is a directory with ``yax.json`` (units, frame, geometry, mass properties)
and ``body.stl``; producers write it, consumers load it, nobody imports anybody. Specification: docs/spec.md."""
from .manifest import BODY, NAME, VERSION, canonical, identity, validate
from .package import Design, Invalid, check, describe, load, write
from . import stl, units

__all__ = ["BODY", "NAME", "VERSION", "Design", "Invalid", "canonical", "check", "describe", "identity", "load",
           "stl", "units", "validate", "write"]
