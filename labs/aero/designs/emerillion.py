"""EMERILLION (the merlin's French name): the pearl drone of Dolphin (``labs/marine/designs/dolphin.py``), in the
X-wing shape the user drew — a slender fuselage with a canopy, a straight mid wing with a pod at each tip, and a
cruciform X tail. It is a **tail-sitter**: it stands on the four tail fins (their tips carry pads and the latch
socket is in the tail), takes off and lands vertically on Dolphin's deck pad, and flies level between.

The two wing-tip pods are the lift fans: an 80 mm EDF each (22 N, 1.1 kW), together 1.3× the weight with the
200 g pearl capsule aboard; in level flight they are the cruise propulsion too. The fuselage holds the battery
(6S 10 000 mAh) and the capsule bay under the canopy; the wings fold at the root for Dolphin's hangar.

Frame: the nose at the origin, the fuselage along +X (the free stream), span along Y, up +Z; millimetres.
``part``: ``aircraft``, ``fuselage``, ``wing`` (with the pods), ``tail`` (the X fins with their pads), ``canopy``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Design, Parameter

G = 9.81
MASS = {"airframe (printed shells, carbon spars)": 0.70, "2 x EDF 80 mm + ESC": 0.52, "battery 6S 10000 mAh": 1.31,
        "flight controller + GPS + radio": 0.09, "servos (4)": 0.05, "fin pads + latch socket": 0.10, "wing hinges": 0.08,
        "capsule bay + release": 0.06}
POD_EDF = {"static_thrust_N": 22.0, "power_W": 1100.0}        # an 80 mm EDF on 6S (class values)
PEARL_CAPSULE_KG = 0.2
BATTERY_WH = 222.0
WING_AREA_M2 = 0.70 * 0.16
CL_MAX = 1.2


def mass_kg(payload: float = PEARL_CAPSULE_KG) -> float:
    return sum(MASS.values()) + payload


def hover(payload: float = PEARL_CAPSULE_KG) -> dict:
    """Thrust over weight and the power to hover on the two pods (P ~ T^1.5 down from full power)."""
    w = mass_kg(payload) * G
    t_max = 2 * POD_EDF["static_thrust_N"]
    tw = t_max / w
    p_hover = 2 * POD_EDF["power_W"] * (w / t_max) ** 1.5
    return {"mass_kg": mass_kg(payload), "weight_N": w, "thrust_to_weight": tw, "hover_W": p_hover, "hover_ok": tw >= 1.3}


def stall_speed(payload: float = PEARL_CAPSULE_KG) -> float:
    return math.sqrt(2 * mass_kg(payload) * G / (1.225 * WING_AREA_M2 * CL_MAX))


def pearl_run(distance_km: float = 10.0, cruise_m_s: float = 28.0, cruise_W: float = 420.0, hover_s: float = 60.0,
              battery_wh: float = BATTERY_WH, payload: float = PEARL_CAPSULE_KG) -> dict:
    """Out to the vessel and back with the hovers (lift-off, the vertical landing on the vessel's pad, the landing on
    Dolphin, ~20 s each): energy, time, reserve, the farthest vessel with the 20 % reserve. The small X-wing cruises
    fast (28 m/s, the stall at ~18) and spends more per km than MERLIN's 1.4 m wing: that is the shape's price."""
    t_cruise = 2 * distance_km * 1000 / cruise_m_s
    wh_hover = hover(payload)["hover_W"] * hover_s / 3600
    wh = cruise_W * t_cruise / 3600 + wh_hover
    usable = 0.8 * battery_wh
    km_per_wh = cruise_m_s * 3.6 / cruise_W
    return {"minutes": (t_cruise + hover_s) / 60, "wh": wh, "reserve": usable / wh - 1, "range_km_reserve": (usable - wh_hover) * km_per_wh / 2,
            "stall_m_s": stall_speed(payload)}


class Emerillion(Design):
    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "fuselage", "wing", "tail", "canopy"), description="what to build"),
        Parameter("length", 1050.0, "mm", min=300),
        Parameter("fuselage_d", 90.0, "mm", min=30),
        Parameter("span", 700.0, "mm", min=200, description="tip to tip, pods included"),
        Parameter("chord", 160.0, "mm", min=40),
        Parameter("wing_x", 0.45, "", min=0.2, max=0.8, description="the wing's leading edge along the length"),
        Parameter("pod_d", 95.0, "mm", min=30, description="the tip pods: the 80 mm EDFs' nacelles"),
        Parameter("pod_length", 260.0, "mm", min=80),
        Parameter("fin_span", 150.0, "mm", min=50, description="each X fin from the fuselage's surface to its tip"),
        Parameter("fin_chord", 130.0, "mm", min=40),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Emerillion.parameters} | (over or {})

    def _fuselage(self, p):
        L, d = p["length"], p["fuselage_d"]
        # a slender body of revolution: an ogive nose, a cylinder, a tapered tail; built from a spline revolved about X
        pts = [(0, 0), (L * 0.06, d * 0.25), (L * 0.16, d * 0.44), (L * 0.30, d * 0.5), (L * 0.62, d * 0.5), (L * 0.85, d * 0.33), (L, d * 0.14)]
        prof = cq.Workplane("XY").spline(pts, includeCurrent=False).lineTo(L, 0).close()
        return prof.revolve(360, (0, 0, 0), (1, 0, 0))

    def _canopy(self, p):
        L, d = p["length"], p["fuselage_d"]
        c = cq.Workplane("XY").add(cq.Solid.makeSphere(d * 0.38, cq.Vector(L * 0.30, 0, d * 0.42)))
        c = c.union(cq.Workplane("XY").add(cq.Solid.makeSphere(d * 0.30, cq.Vector(L * 0.40, 0, d * 0.40))))
        return c.intersect(cq.Workplane("XY").box(L, d * 2, d).translate((L * 0.35, 0, d * 0.5 + d * 0.3)))

    def _wing(self, p):
        L, d, b, c = p["length"], p["fuselage_d"], p["span"], p["chord"]
        x0 = p["wing_x"] * L
        pod_d, pod_l = p["pod_d"], p["pod_length"]
        half = b / 2 - pod_d / 2
        wing = cq.Workplane("XY").box(c, 2 * half, c * 0.11).edges("|Y").fillet(c * 0.05).translate((x0 + c / 2, 0, 0))
        for sy in (1, -1):
            pod = cq.Workplane("XY").add(cq.Solid.makeCylinder(pod_d / 2, pod_l * 0.6, cq.Vector(x0 + c / 2 - pod_l * 0.3, sy * half, 0), cq.Vector(1, 0, 0)))
            nose = cq.Workplane("XY").add(cq.Solid.makeCone(pod_d / 2, 8, pod_l * 0.2, cq.Vector(x0 + c / 2 - pod_l * 0.3, sy * half, 0), cq.Vector(-1, 0, 0)))
            tail = cq.Workplane("XY").add(cq.Solid.makeCone(pod_d / 2, pod_d * 0.3, pod_l * 0.2, cq.Vector(x0 + c / 2 + pod_l * 0.3, sy * half, 0), cq.Vector(1, 0, 0)))
            wing = wing.union(pod).union(nose).union(tail)
        return wing

    def _tail(self, p):
        L, d = p["length"], p["fuselage_d"]
        fs, fc = p["fin_span"], p["fin_chord"]
        x0 = L - fc - 20
        out = None
        for a in (45, 135, 225, 315):
            fin = (cq.Workplane("XZ").polyline([(0, d * 0.2), (fc, d * 0.2), (fc * 0.75, d * 0.2 + fs), (fc * 0.25, d * 0.2 + fs)]).close()
                   .extrude(3, both=True).translate((x0, 0, 0)).rotate((0, 0, 0), (1, 0, 0), a))
            pad = cq.Workplane("XY").add(cq.Solid.makeCylinder(10, 6, cq.Vector(x0 + fc * 0.6, 0, d * 0.2 + fs - 3), cq.Vector(1, 0, 0))) \
                .rotate((0, 0, 0), (1, 0, 0), a)
            f = fin.union(pad)
            out = f if out is None else out.union(f)
        socket = cq.Workplane("YZ").workplane(offset=L - 50).circle(16).circle(10).extrude(50)
        return out.union(socket)

    def build(self, p):
        parts = {"fuselage": self._fuselage, "canopy": self._canopy, "wing": self._wing, "tail": self._tail}
        if p["part"] in parts:
            return parts[p["part"]](p)
        out = None
        for name in parts:
            s = parts[name](p)
            out = s if out is None else out.union(s)
        return out


if __name__ == "__main__":
    h, r = hover(), pearl_run()
    print(f"Emerillion {h['mass_kg']:.2f} kg with the capsule, T/W {h['thrust_to_weight']:.2f} on the two pods, hovers at {h['hover_W']:.0f} W; "
          f"stall {r['stall_m_s']:.0f} m/s; a 10 km pearl run {r['minutes']:.0f} min, {r['wh']:.0f} Wh, reserve {r['reserve']:+.2f}; range with reserve {r['range_km_reserve']:.0f} km each way")
