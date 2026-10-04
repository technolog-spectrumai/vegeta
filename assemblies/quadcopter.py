"""The quadcopter of notebook 08: the X frame, a motor on each pad and a propeller on each motor. Diagonal motors turn
the same way, neighbours the other (props in, as a betaflight quad): motor 1 at 45 deg turns right-hand about +Z."""
from __future__ import annotations

import math

from vegeta.dedalus import Parameter

from components.motor import Outrunner
from components.propeller import CATALOGUE, blade
from components.quad_frame import QuadFrame

from .base import Assembly, component_parameters, pick


class Quadcopter(Assembly):
    """Frame + 4 motors + 4 propellers. Frame parameters as ``QuadFrame``; motor ones prefixed ``motor_``."""

    parameters = component_parameters(QuadFrame) + [
        Parameter("propeller", "5x4.3 tri-blade", choices=tuple(CATALOGUE), description="components.propeller.CATALOGUE"),
        Parameter("motor_diameter", 29.0, "mm", min=5),
        Parameter("motor_height", 19.0, "mm", min=3, description="pad to the top of the bell"),
        Parameter("motor_shaft_length", 10.0, "mm", min=0, description="the propeller hub sits on the shaft"),
    ]

    def parts(self, p):
        frame = QuadFrame().generate(**pick(p, QuadFrame)).shape
        motor = Outrunner().generate(diameter=p["motor_diameter"], height=p["motor_height"],
                                     base_diameter=0.85 * p["motor_diameter"], shaft_length=p["motor_shaft_length"]).shape
        right, prop = blade(p["propeller"], "right")
        left, _ = blade(p["propeller"], "left")
        z_pad = p["arm_height"]                                             # pads are arm_height thick, from z = 0
        z_hub = z_pad + p["motor_height"] + prop["hub_height"] / 2
        out = {"frame": frame}
        for k in range(4):
            ang = math.radians(45 + 90 * k)
            x, y = p["wheelbase"] / 2 * math.cos(ang), p["wheelbase"] / 2 * math.sin(ang)
            out[f"motor_{k + 1}"] = motor.translate((x, y, z_pad))
            out[f"propeller_{k + 1}"] = (right if k % 2 == 0 else left).translate((x, y, z_hub))
        return out
