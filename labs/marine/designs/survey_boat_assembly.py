"""The survey boat of notebook 12: hull, deck and motor-pod bracket (``SurveyBoat`` part='boat') and the 60 mm marine
propeller 10 mm behind the pod, on the pod axis, as ``tools/run_scenario.py`` places it. Bow toward +x."""
from __future__ import annotations

from vegeta.dedalus import Parameter

from vegeta.labkit.propeller import CATALOGUE, blade
from survey_boat import SurveyBoat

from vegeta.labkit.assembly import Assembly, component_parameters, pick, turned

POD_LENGTH = 90.0           # mm: SurveyBoat._bracket extrudes the pod 90 mm aft of the bracket plate


class SurveyBoatAssembly(Assembly):
    """Boat + propeller. Boat parameters as ``SurveyBoat``."""

    parameters = component_parameters(SurveyBoat) + [
        Parameter("propeller", "60 mm 3-blade marine", choices=tuple(CATALOGUE), description="vegeta.labkit.propeller.CATALOGUE"),
        Parameter("propeller_gap", 10.0, "mm", min=0, description="propeller plane behind the pod"),
    ]

    def parts(self, p):
        boat = SurveyBoat().generate(**pick(p, SurveyBoat, part="boat")).shape
        right, _ = blade(p["propeller"])
        x = -p["bracket_thickness"] - POD_LENGTH - p["propeller_gap"]
        z = 0.55 * p["depth"] - p["bracket_height"]               # SurveyBoat lifts the bracket by 0.55 depth
        return {"boat": boat, "propeller": turned(right, "+x").translate((x, 0.0, z))}
