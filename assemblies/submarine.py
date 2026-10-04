"""The AUV of notebook 13: the vehicle (``Submarine`` part='vehicle') and its 120 mm 3-blade propeller 40 mm behind
the tail tip, as ``scenarios/run_scenario.py`` places the rotor disk. Tail tip at x = 0, nose toward +x."""
from __future__ import annotations

from vegeta.dedalus import Parameter

from components.propeller import CATALOGUE, blade
from components.submarine import Submarine

from .base import Assembly, component_parameters, pick, turned


class SubmarineAssembly(Assembly):
    """Vehicle + propeller. Vehicle parameters as ``Submarine``."""

    parameters = component_parameters(Submarine) + [
        Parameter("propeller", "120 mm 3-blade", choices=tuple(CATALOGUE), description="components.propeller.CATALOGUE"),
        Parameter("propeller_gap", 40.0, "mm", min=0, description="propeller plane behind the tail tip"),
    ]

    def parts(self, p):
        vehicle = Submarine().generate(**pick(p, Submarine, part="vehicle")).shape
        right, _ = blade(p["propeller"])
        return {"vehicle": vehicle, "propeller": turned(right, "+x").translate((-p["propeller_gap"], 0.0, 0.0))}
