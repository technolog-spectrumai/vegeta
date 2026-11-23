"""Outrunner motor, as a stand-in volume: the stator base on the mount, the bell above it and the shaft. Enough for
clearances, frontal area and a picture; no windings, magnets or bolt holes. Base on z = 0, shaft along +Z."""
from __future__ import annotations

import cadquery as cq
from vegeta.dedalus import Design, Parameter


class Outrunner(Design):
    """A 2306-class motor by default (notebook 08's 2306-2400KV: 29 mm can, 19 mm tall, 5 mm shaft)."""

    parameters = [
        Parameter("diameter", 29.0, "mm", min=5, description="bell diameter"),
        Parameter("height", 19.0, "mm", min=3, description="base to the top of the bell"),
        Parameter("base_height", 3.0, "mm", min=0.5, description="stator base below the bell"),
        Parameter("base_diameter", 25.0, "mm", min=3),
        Parameter("shaft_diameter", 5.0, "mm", min=1),
        Parameter("shaft_length", 10.0, "mm", min=0, description="shaft above the bell (the propeller sits on it)"),
    ]

    def build(self, p):
        if p["base_height"] >= p["height"]:
            raise ValueError("base_height must be smaller than height")
        base = cq.Workplane("XY").circle(p["base_diameter"] / 2).extrude(p["base_height"])
        bell = cq.Workplane("XY").workplane(offset=p["base_height"]).circle(p["diameter"] / 2) \
            .extrude(p["height"] - p["base_height"])
        motor = base.union(bell)
        if p["shaft_length"] > 0:
            motor = motor.union(cq.Workplane("XY").workplane(offset=p["height"]).circle(p["shaft_diameter"] / 2)
                                .extrude(p["shaft_length"]))
        return motor
