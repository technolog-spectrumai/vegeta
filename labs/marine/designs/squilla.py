"""Squilla (after Squilla mantis, the mantis shrimp: raptorial claws that strike fast, and the most elaborate eyes in
the sea): Nephro's intervention variant. The same body of revolution, housings, thrusters and autonomy, but built to
work at the site rather than to survey 100 km of cable: shorter, a smaller battery, eight tunnel thrusters for a stiff
hover, wider skids for landed work, and **two arms** — a 6-DOF work arm with a fast "raptorial" cutter-gripper and a
4-DOF hold arm that clamps the vehicle to the cable, mooring or pile it works on.

Requirements (``REQUIREMENTS``, from the design questions): better than Nephro at intervention, without Nephro's
range; cut and clear (ropes, nets, fishing gear, debris on cables, moorings and piles); grab, tag and retrieve (markers,
tools, sensors, up to ~8 kg in water); close inspection by touch (UT thickness, CP stab, coating tests). The vision is
chosen from the sensor configurations in ``SENSOR_CONFIGS`` (``compare_sensors``).

x forward, y left, z up; the axis at z = 0. ``part``: ``vehicle``, ``body``, ``arms``, ``skids``, ``sensors``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Parameter

from nephro import Nephro
from scampi import FOAM_RHO, RHO_SW

G = 9.81
REQUIREMENTS = {
    "jobs": ["cut and clear: ropes, nets, fishing gear, debris", "grab, tag, retrieve (<= 8 kg in water)",
             "close inspection by touch: UT, CP stab, coating"],
    "depth_m": 300.0,
    "transit_km": 20.0,             # out and back from the vessel: a site, not a route
    "work_h": 4.0,                  # hover + landed work per dive
    "current_m_s": 1.0,             # nose into the current while working
    "lift_kg_in_water": 8.0,
    "reserve": 0.15,
}
#: The base vehicle's parts (mass kg, volume L, power W): navigation and comms, thrusters, arms. Class values.
PARTS = {
    "Jetson AGX Orin 64GB + carrier": (0.9, 0.0, 40.0),
    "INS (MEMS-FOG class) + depth": (1.8, 0.7, 15.0),
    "DVL 600 kHz": (1.6, 0.6, 15.0),
    "acoustic modem + USBL transponder": (1.2, 0.5, 5.0),
    "2 stern thrusters (T500 class)": (2 * 1.0, 2 * 0.35, 0.0),
    "8 tunnel thrusters (T200 class): 4 vertical, 4 lateral": (8 * 0.35, 8 * 0.15, 0.0),
    "6-DOF work arm, raptorial cutter-gripper": (10.0, 4.0, 0.0),
    "4-DOF hold arm, clamp jaw": (5.5, 2.2, 0.0),
    "contact probe set: UT, CP stab, coating (on the hold arm)": (0.8, 0.4, 5.0),
    "fins, wide skids, wet-mate connectors, cables": (7.0, 3.5, 0.0),
}
#: Vision components: (mass kg, volume L, power W, price EUR, Jetson load in TOPS) — class values.
SENSORS = {
    "stereo camera pair + 4 LED lights": (3.0, 1.2, 90.0, 6000.0, 20.0),
    "polarisation camera": (0.8, 0.3, 6.0, 4500.0, 15.0),
    "imaging sonar (dual-frequency, 1.2/2.1 MHz)": (1.5, 0.8, 20.0, 9000.0, 10.0),
    "laser line scanner (3-D for grasping)": (2.5, 1.2, 35.0, 30000.0, 25.0),
    "hyperspectral camera (push-broom)": (2.0, 1.0, 25.0, 40000.0, 45.0),
}
#: What each sensor contributes to (clear water, turbid water, finding fibres/ropes, 3-D grasping, material ID): 0..2.
CAPABILITY = {
    "stereo camera pair + 4 LED lights": (2, 0, 1, 1, 0),
    "polarisation camera": (1, 1, 2, 0, 1),
    "imaging sonar (dual-frequency, 1.2/2.1 MHz)": (1, 2, 1, 1, 0),
    "laser line scanner (3-D for grasping)": (1, 1, 1, 2, 0),
    "hyperspectral camera (push-broom)": (1, 0, 1, 0, 2),
}
ASPECTS = ("clear water", "turbid water", "ropes and fibres", "3-D grasping", "material ID")
SENSOR_CONFIGS = {
    "A stereo + polarisation": ["stereo camera pair + 4 LED lights", "polarisation camera"],
    "B stereo + imaging sonar": ["stereo camera pair + 4 LED lights", "imaging sonar (dual-frequency, 1.2/2.1 MHz)"],
    "C stereo + polarisation + imaging sonar": ["stereo camera pair + 4 LED lights", "polarisation camera",
                                                "imaging sonar (dual-frequency, 1.2/2.1 MHz)"],
    "D stereo + imaging sonar + laser scanner": ["stereo camera pair + 4 LED lights",
                                                 "imaging sonar (dual-frequency, 1.2/2.1 MHz)",
                                                 "laser line scanner (3-D for grasping)"],
    "E stereo + hyperspectral + imaging sonar": ["stereo camera pair + 4 LED lights", "hyperspectral camera (push-broom)",
                                                 "imaging sonar (dual-frequency, 1.2/2.1 MHz)"],
    "F all five": list(SENSORS),
}
JETSON_TOPS = 275.0                  # AGX Orin 64GB (sparse INT8); the autonomy stack itself takes ~120 (assumed)
AUTONOMY_TOPS = 120.0
DEFAULT_SENSORS = "C stereo + polarisation + imaging sonar"


class Squilla(Nephro):
    parameters = [q for q in Nephro.parameters if q.name not in (
        "length", "battery_kwh", "battery_housing_length", "antenna_span", "arm_reach", "arm_pose", "part")] + [
        Parameter("part", "vehicle", choices=("vehicle", "body", "arms", "skids", "sensors"), description="what to build"),
        Parameter("length", 2200.0, "mm", min=1000),
        Parameter("battery_kwh", 4.5, "kWh", min=0.5, description="a 4 h work dive with every sensor set (compare_sensors)"),
        Parameter("battery_housing_length", 550.0, "mm", min=200),
        Parameter("arm_reach", 1000.0, "mm", min=300, description="work arm, shoulder to the claw tips"),
        Parameter("hold_reach", 700.0, "mm", min=300),
        Parameter("arm_pose", "work", choices=("stowed", "work")),
        Parameter("sensors", DEFAULT_SENSORS, choices=tuple(SENSOR_CONFIGS), description="the vision configuration"),
        Parameter("strike_ms", 150.0, "ms", min=20, description="cutter close time: the raptorial strike (spring-assisted)"),
        Parameter("cut_force_N", 3000.0, "N", min=100, description="cutter blade force: 20 mm polyester rope, 10 mm wire"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Squilla.parameters} | (over or {})

    @classmethod
    def sizing(cls, over=None):
        """Housings, mass and buoyancy (foam to neutral, plus foam for the lift), hover thrust against the current and
        the lift, a dive (transit + work) against the battery, and the vision configuration's cost."""
        p = cls.p(over)
        hb = cls._housing(p, p["battery_housing_length"])
        he = cls._housing(p, p["electronics_housing_length"])
        cells = p["battery_kwh"] * 1000 / p["cell_wh_kg"]
        D = p["diameter"] / 1000
        fairing_kg = 0.95 * math.pi * D * p["length"] / 1000 * 0.008 * 1000
        items = {"fairing + internal frame": (fairing_kg + 7.0, (fairing_kg + 7.0) / 0.95),
                 "battery housing + cells + BMS": (hb["mass_kg"] + cells + 1.5, hb["volume_L"]),
                 "electronics housing": (he["mass_kg"] + 2.5, he["volume_L"])}
        items.update({k: (m, v) for k, (m, v, _) in PARTS.items()})
        sens = SENSOR_CONFIGS[p["sensors"]]
        items.update({k: SENSORS[k][:2] for k in sens})
        m_dry = sum(m for m, _ in items.values())
        v_dry = sum(v for _, v in items.values()) / 1000
        net_up = 1.0
        V_n = (m_dry + net_up - RHO_SW * v_dry) / (RHO_SW - FOAM_RHO)
        V_lift = REQUIREMENTS["lift_kg_in_water"] * 0.5 / (RHO_SW - FOAM_RHO)   # half the lift by foam, half by thrust
        V_f = V_n + V_lift
        m = m_dry + FOAM_RHO * V_f
        mid = (p["length"] - p["nose_length"] - p["tail_length"]) / 1000
        inside = math.pi * (D / 2 - 0.01) ** 2 * (mid + 0.5 * (p["nose_length"] + p["tail_length"]) / 1000) * 0.75
        need = (hb["volume_L"] + he["volume_L"]) / 1000 + V_f + 0.012
        # hover: nose into the current, the hold arm off; vertical thrust for the lift's other half and the arm's reactions
        A = math.pi * D * D / 4
        drag_axial = 0.5 * RHO_SW * p["cd"] * A * REQUIREMENTS["current_m_s"] ** 2
        T200 = 50.0                                                              # N per T200-class tunnel thruster (~5 kgf)
        vert, lat, surge = 4 * T200, 4 * T200, 2 * 157.0
        need_vert = REQUIREMENTS["lift_kg_in_water"] * 0.5 * G + 60.0            # + the cutter's and arm's reactions
        # power: hotel + vision; transit at 1.5 m/s; work: hover holding station in the current + arms
        hotel = sum(w for _, _, w in PARTS.values())
        vision_W = sum(SENSORS[k][2] for k in sens)
        v = 1.5
        transit_W = hotel + vision_W * 0.3 + 0.5 * RHO_SW * p["cd"] * A * v ** 3 / p["eta_propulsive"]
        work_W = hotel + vision_W + drag_axial * REQUIREMENTS["current_m_s"] / 0.3 + 220.0 + 180.0   # station keeping, tunnels, arms
        e = (REQUIREMENTS["transit_km"] / (v * 3.6)) * transit_W + REQUIREMENTS["work_h"] * work_W
        usable = p["battery_kwh"] * 1000 * (1 - REQUIREMENTS["reserve"])
        # the raptorial strike: blade energy and the close speed
        stroke = 0.03
        strike_W = p["cut_force_N"] * stroke / (p["strike_ms"] / 1000)
        tops = AUTONOMY_TOPS + sum(SENSORS[k][4] for k in sens)
        return {"housings": {"battery": hb, "electronics": he}, "mass_in_air_kg": m, "foam_L": V_f * 1000,
                "inside_L": inside * 1000, "needed_L": need * 1000, "fits": need <= inside, "items": items,
                "thrust_N": {"vertical": vert, "lateral": lat, "surge": surge}, "vertical_margin": vert / need_vert - 1,
                "surge_margin": surge / (2 * drag_axial) - 1, "drag_axial_N": drag_axial,
                "transit_W": transit_W, "work_W": work_W, "dive_Wh": e, "usable_Wh": usable,
                "battery_margin": usable / e - 1, "strike_W": strike_W, "jetson_tops": tops,
                "jetson_ok": tops <= JETSON_TOPS, "vision_W": vision_W,
                "vision_eur": sum(SENSORS[k][3] for k in sens), "sensors": p["sensors"]}

    @classmethod
    def compare_sensors(cls):
        """Every vision configuration: mass, power, cost, Jetson load, the work hours a 4 kWh pack still gives, and its
        score per aspect (the best sensor of the set counts) and in total."""
        rows = {}
        for name, comps in SENSOR_CONFIGS.items():
            s = cls.sizing({"sensors": name})
            score = [max(CAPABILITY[c][k] for c in comps) for k in range(len(ASPECTS))]
            rows[name] = {"mass kg": sum(SENSORS[c][0] for c in comps), "power W": s["vision_W"], "price EUR": s["vision_eur"],
                          "Jetson TOPS": s["jetson_tops"], "Jetson ok": s["jetson_ok"], "battery margin": s["battery_margin"],
                          "vehicle kg": s["mass_in_air_kg"], **dict(zip(ASPECTS, score)), "score": sum(score),
                          "score per kEUR": sum(score) / (s["vision_eur"] / 1000)}
        return rows

    # ------------------------------------------------------------------------------------------- CAD
    def _arms(self, p):
        L, R = p["length"], p["diameter"] / 2
        out = None
        for side, reach, links in ((-1, p["arm_reach"], 4), (1, p["hold_reach"], 3)):
            a = cq.Vector(L * 0.22, side * R * 0.45, -R * 0.55)
            if p["arm_pose"] == "stowed":
                dirs = [cq.Vector(-0.55, 0, 0), cq.Vector(0.45, 0, 0), cq.Vector(-0.2, 0, 0), cq.Vector(0.1, 0, 0)][:links]
            else:
                dirs = [cq.Vector(0.30, side * 0.05, -0.20), cq.Vector(0.32, 0, -0.10), cq.Vector(0.20, 0, -0.06),
                        cq.Vector(0.10, 0, -0.02)][:links]
            for k, d in enumerate(dirs):
                d = d * reach
                seg = cq.Workplane("XY").add(cq.Solid.makeCylinder(30 - 5 * k, d.Length, a, d)).union(
                    cq.Workplane("XY").add(cq.Solid.makeSphere(34 - 5 * k, a)))
                out = seg if out is None else out.union(seg)
                a = a + d
            tip = a
            for s in (1, -1):                                                    # the raptorial claw / the clamp jaw
                claw = cq.Vector(0.10, s * 0.05, -0.04 if side < 0 else 0.0) * reach
                out = out.union(cq.Workplane("XY").add(cq.Solid.makeCylinder(9, claw.Length, tip, claw)))
        return out

    def _skids(self, p):
        L, R = p["length"], p["diameter"] / 2
        out = None
        for sy in (1, -1):                                                       # wider and longer than Nephro's
            sk = (cq.Workplane("YZ").center(sy * R * 1.1, -R - 80).circle(20).extrude(L * 0.32, both=True)
                  .union(cq.Workplane("XY").box(30, R * 0.6, 30).translate((L * 0.2, sy * R * 0.8, -R - 80)))
                  .union(cq.Workplane("XY").box(30, R * 0.6, 30).translate((-L * 0.2, sy * R * 0.8, -R - 80)))
                  .union(cq.Workplane("XY").box(30, 30, 120).translate((L * 0.2, sy * R * 0.55, -R - 30)))
                  .union(cq.Workplane("XY").box(30, 30, 120).translate((-L * 0.2, sy * R * 0.55, -R - 30))))
            out = sk if out is None else out.union(sk)
        return out

    def _sensors(self, p):
        """The vision pods in the nose's chin: one per component of the configuration."""
        L, R = p["length"], p["diameter"] / 2
        comps = SENSOR_CONFIGS[p["sensors"]]
        out = None
        for k, c in enumerate(comps):
            y = (k - (len(comps) - 1) / 2) * 70
            pod = cq.Workplane("YZ").center(y, -R * 0.45).circle(26).extrude(70).translate((L / 2 - p["nose_length"] * 0.55, 0, 0))
            out = pod if out is None else out.union(pod)
        return out

    def build(self, p):
        if cq is None:
            raise ImportError("Squilla's CAD needs CadQuery")
        parts = {"body": self._body, "arms": self._arms, "skids": self._skids, "sensors": self._sensors}
        if p["part"] in parts:
            return parts[p["part"]](p)
        out = self._body(p)
        for k in ("arms", "skids", "sensors"):
            out = out.union(parts[k](p))
        return out


if __name__ == "__main__":
    s = Squilla.sizing()
    print(f"[{s['sensors']}] mass {s['mass_in_air_kg']:.0f} kg, foam {s['foam_L']:.0f} L, needs {s['needed_L']:.0f} of {s['inside_L']:.0f} L "
          f"({'fits' if s['fits'] else 'DOES NOT FIT'})")
    print(f"thrust {s['thrust_N']}, vertical margin {s['vertical_margin']:+.2f}, surge margin in {REQUIREMENTS['current_m_s']} m/s {s['surge_margin']:+.2f}")
    print(f"dive: {s['dive_Wh']:.0f} Wh (transit {s['transit_W']:.0f} W, work {s['work_W']:.0f} W) vs {s['usable_Wh']:.0f} usable: margin {s['battery_margin']:+.2f}")
    print(f"strike {s['strike_W']:.0f} W peak (spring-assisted), Jetson {s['jetson_tops']:.0f} of {JETSON_TOPS:.0f} TOPS")
    import pandas as pd
    print(pd.DataFrame(Squilla.compare_sensors()).T.round(2).to_string())
