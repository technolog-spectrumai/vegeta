"""Consistent unit systems. Talos never converts silently; the engineer chooses one explicitly."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitSystem:
    name: str
    length: str
    force: str
    stress: str
    density: str
    acceleration: str
    occ_unit: str  # unit Gmsh/OpenCascade converts STEP geometry into


UNIT_SYSTEMS = {
    "mm-N-MPa": UnitSystem("mm-N-MPa", "mm", "N", "MPa", "t/mm^3", "mm/s^2", "MM"),
    "m-N-Pa": UnitSystem("m-N-Pa", "m", "N", "Pa", "kg/m^3", "m/s^2", "M"),
}


def get_units(name: str) -> UnitSystem:
    if name not in UNIT_SYSTEMS:
        raise ValueError(f"unknown unit system {name!r}; choose one of {list(UNIT_SYSTEMS)}")
    return UNIT_SYSTEMS[name]
