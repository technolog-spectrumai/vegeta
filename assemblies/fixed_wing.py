"""The twin-motor fixed-wing drone of notebook 09a: the airframe and a 9x6 propeller ahead of each nacelle, placed as
``scenarios/run_scenario.py`` places its rotor disks (12 mm ahead of the nacelle nose, the -y one right-hand about the
flight direction, the +y one left-hand). Angle of attack turns the whole aircraft, propellers included."""
from __future__ import annotations

from vegeta.dedalus import Parameter

from components.fixed_wing import FixedWing
from components.propeller import CATALOGUE, blade

from .base import Assembly, component_parameters, pick, turned


class FixedWingDrone(Assembly):
    """Airframe (``FixedWing`` part='aircraft', its parameters) + 2 tractor propellers."""

    parameters = component_parameters(FixedWing) + [
        Parameter("propeller", "9x6 electric", choices=tuple(CATALOGUE), description="components.propeller.CATALOGUE"),
        Parameter("propeller_gap", 12.0, "mm", min=0, description="propeller plane ahead of the nacelle nose"),
    ]

    def parts(self, p):
        airframe = FixedWing().generate(**pick(p, FixedWing, part="aircraft")).shape     # already at angle of attack
        right, _ = blade(p["propeller"], "right")
        left, _ = blade(p["propeller"], "left")
        x = -p["nacelle_forward"] - p["propeller_gap"]            # nose toward -x: thrust (+Z of the blade) to -x
        out = {"airframe": airframe}
        for name, shape, y in (("propeller_right", right, -p["nacelle_y"]), ("propeller_left", left, p["nacelle_y"])):
            prop = turned(shape, "-x").translate((x, y, 0.0))
            if p["angle_of_attack_deg"]:
                prop = prop.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])    # as FixedWing turns the airframe
            out[name] = prop
        return out
