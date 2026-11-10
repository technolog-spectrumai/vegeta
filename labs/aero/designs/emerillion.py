"""EMERILLION (the merlin's French name): MERLIN as a tail-sitter — it takes off and lands vertically on its four tail
fins, on Dolphin's deck pad (``labs/marine/designs/dolphin.py``), and flies level like MERLIN between. One airframe
change: a larger EDF (120 mm, 2.4 kW) so it hovers at 1.3 times its weight with the 200 g pearl capsule, four fin
legs with pads and the latch cone's socket in the tail, folding wings (hinged at the root) for the hangar.

``part``: ``aircraft`` (MERLIN's, with the legs), ``legs``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Parameter

from merlin import Merlin

G = 9.81
#: masses [kg] over MERLIN's EDF variant (2.33 kg): the 120 mm EDF + ESC in place of the 90 mm, the legs, the wing hinges
EXTRA_MASS = {"EDF 120 mm + ESC (replaces the 90 mm)": 0.25, "4 fin legs with pads + latch socket": 0.14, "wing hinges (fold back along the fuselage)": 0.08,
              "capsule bay + release": 0.06, "battery 6S 10000 mAh in place of 8000": 0.26}
BATTERY_WH = 222.0
MERLIN_EDF_MASS = 2.326
EDF_120 = {"static_thrust_N": 44.0, "power_W": 2400.0}       # a 120 mm 12-blade EDF on 6S at full power (class values: 4.5 kgf)
PEARL_CAPSULE_KG = 0.2


def mass_kg(payload: float = PEARL_CAPSULE_KG) -> float:
    return MERLIN_EDF_MASS + sum(EXTRA_MASS.values()) + payload


def hover(payload: float = PEARL_CAPSULE_KG) -> dict:
    """Thrust over weight and the power to hover (the EDF's static thrust at full power, P ~ T^1.5 down from it)."""
    w = mass_kg(payload) * G
    tw = EDF_120["static_thrust_N"] / w
    p_hover = EDF_120["power_W"] * (w / EDF_120["static_thrust_N"]) ** 1.5
    return {"mass_kg": mass_kg(payload), "weight_N": w, "thrust_to_weight": tw, "hover_W": p_hover, "hover_ok": tw >= 1.3}


def pearl_run(distance_km: float = 10.0, cruise_m_s: float = 22.0, cruise_W: float = 350.0, hover_s: float = 60.0,
              battery_wh: float = BATTERY_WH, payload: float = PEARL_CAPSULE_KG) -> dict:
    """Out to the vessel and back with the hovers (lift-off, a vertical landing on the vessel's pad, the landing on
    Dolphin: ~20 s each): energy, time, reserve, and the farthest vessel with the 20 % reserve."""
    t_cruise = 2 * distance_km * 1000 / cruise_m_s
    wh_hover = hover(payload)["hover_W"] * hover_s / 3600
    wh = cruise_W * t_cruise / 3600 + wh_hover
    usable = 0.8 * battery_wh
    km_per_wh = cruise_m_s * 3.6 / cruise_W                                   # km of cruise per Wh
    return {"minutes": (t_cruise + hover_s) / 60, "wh": wh, "reserve": usable / wh - 1,
            "range_km_reserve": (usable - wh_hover) * km_per_wh / 2}


class Emerillion(Merlin):
    parameters = [q for q in Merlin.parameters if q.name not in ("part", "propulsion", "edf_diameter")] + [
        Parameter("part", "aircraft", choices=("aircraft", "legs"), description="what to build"),
        Parameter("propulsion", "edf", choices=("edf",)),
        Parameter("edf_diameter", 120.0, "mm", min=60, max=130, description="the tail-sitter's larger fan"),
        Parameter("leg_length", 230.0, "mm", min=100, description="the fin legs from the tail end to the pads"),
        Parameter("leg_spread", 200.0, "mm", min=80, description="the pads' radius from the axis"),
    ]

    def _legs(self, p):
        """Four fin legs at 45° from the tail end: a tapered blade each with a pad, and the latch socket on the axis."""
        x_tail = p["aft_length"]
        z_axis = -p["fuselage_diameter"] / 2
        out = None
        for a in (45, 135, 225, 315):
            blade = (cq.Workplane("XY").polyline([(0, 0), (p["leg_length"], p["leg_spread"] - 30), (p["leg_length"], p["leg_spread"]), (-40, 40), (-40, 0)])
                     .close().extrude(5, both=True).translate((x_tail - p["tail_cone"] * 0.3, 0, z_axis))
                     .rotate((x_tail, 0, z_axis), (x_tail + 1, 0, z_axis), a))
            pad = cq.Workplane("XY").circle(14).extrude(6).translate((x_tail - p["tail_cone"] * 0.3 + p["leg_length"], p["leg_spread"] - 15, z_axis)) \
                .rotate((x_tail, 0, z_axis), (x_tail + 1, 0, z_axis), a)
            leg = blade.union(pad)
            out = leg if out is None else out.union(leg)
        socket = cq.Workplane("YZ").workplane(offset=x_tail + 20).center(0, z_axis).circle(22).circle(14).extrude(60)
        return out.union(socket)

    def build(self, p):
        if p["part"] == "legs":
            return self._legs(p)
        base = super().build({**p, "part": "aircraft"})
        return base.union(self._legs(p))


if __name__ == "__main__":
    h, r = hover(), pearl_run()
    print(f"Emerillion {h['mass_kg']:.2f} kg with the capsule, T/W {h['thrust_to_weight']:.2f}, hovers at {h['hover_W']:.0f} W; "
          f"a 10 km pearl run {r['minutes']:.0f} min, {r['wh']:.0f} Wh, reserve {r['reserve']:+.2f}; range with reserve {r['range_km_reserve']:.0f} km each way")
