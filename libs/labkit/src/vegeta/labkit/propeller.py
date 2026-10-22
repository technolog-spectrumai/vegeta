"""Propellers, in air or water: Dedalus' blade (``dedalus.examples.Propeller``) with the dimensions the notebooks give
each one. A right-hand propeller about +Z (it pushes fluid toward -Z); ``handed="left"`` mirrors it for the
counter-rotating one of a pair. The CAD numbers are the ones notebooks 08, 09a, 12, 13 and 25 typed into their
``dedalus.examples.Propeller(...)`` calls."""
from __future__ import annotations

import dataclasses

from vegeta.dedalus import Parameter
from vegeta.dedalus.examples import Propeller as _Blade


def _with(**defaults):
    """Dedalus' propeller parameters with other defaults, plus the handedness."""
    out = [dataclasses.replace(p, default=defaults.pop(p.name)) if p.name in defaults else p for p in _Blade.parameters]
    assert not defaults, f"unknown propeller parameters {sorted(defaults)}"
    return out + [Parameter("handed", "right", choices=("right", "left"),
                            description="right: pushes toward -Z turning by the right-hand rule about +Z; left: mirrored")]


class Propeller(_Blade):
    """Any propeller: Dedalus' defaults (a 5-inch two-blade)."""

    parameters = _with()

    def build(self, p):
        prop = super().build(p)
        return prop.mirror("XZ") if p["handed"] == "left" else prop


class Quad5x43(Propeller):
    """5x4.3 tri-blade: the quadcopter of notebook 08."""

    parameters = _with(diameter=127.0, pitch=109.22, hub_diameter=12.0, hub_height=7.0, bore=5.0, chord_root=10.0,
                       chord_max=16.0, chord_tip=6.0, thickness=0.1, camber=0.05, blades=3)


class Electric9x6(Propeller):
    """9x6 electric: the fixed-wing drone of notebook 09a (and its rotor disks in ``scenarios/``)."""

    parameters = _with(diameter=228.6, pitch=152.4, hub_diameter=16.0, hub_height=9.0, bore=5.0, chord_root=14.0,
                       chord_max=22.0, chord_tip=6.0, thickness=0.09, camber=0.04, blades=2)


class Electric10x6(Propeller):
    """10x6, 2 blades: MERLIN's tractor and pusher propeller (notebook 25)."""

    parameters = _with(diameter=254.0, pitch=152.4, hub_diameter=20.0, hub_height=10.0, bore=5.0, chord_root=18.0,
                       chord_max=26.0, chord_tip=10.0, thickness=0.1, camber=0.04, stations=8, blades=2)


class Marine60(Propeller):
    """60 mm 3-blade marine: the survey boat of notebook 12."""

    parameters = _with(diameter=60.0, pitch=50.0, hub_diameter=12.0, hub_height=8.0, bore=4.0, chord_root=10.0,
                       chord_max=18.0, chord_tip=8.0, thickness=0.12, camber=0.05, stations=8, blades=3)


class Marine120(Propeller):
    """120 mm 3-blade: the submarine of notebook 13."""

    parameters = _with(diameter=120.0, pitch=100.0, hub_diameter=24.0, hub_height=16.0, bore=8.0, chord_root=18.0,
                       chord_max=30.0, chord_tip=12.0, thickness=0.12, camber=0.05, stations=8, blades=3)


CATALOGUE = {"5x4.3 tri-blade": Quad5x43, "9x6 electric": Electric9x6, "10x6, 2 blades": Electric10x6,
             "60 mm 3-blade marine": Marine60, "120 mm 3-blade": Marine120}


def blade(name: str, handed: str = "right", **overrides):
    """The ``name`` propeller of ``CATALOGUE`` as a shape (axis +Z, hub centre at the origin) and its parameters."""
    design = CATALOGUE[name]()
    return design.generate(**overrides, handed=handed).shape, design.resolve(**overrides, handed=handed)
