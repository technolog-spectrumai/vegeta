"""Emerita (Emerita analoga, the Pacific mole crab: it burrows in the swash zone, where cables come ashore, and sweeps
the water with feathery antennae): a hybrid cable AUV. It cruises a torpedo hull along a buried cable for 60-100 km
on one dive, tracking it with a magnetic coil array on the nose (the antennae), and hovers or lands to inspect, mark
and lift it at a fault. Scampi's sibling (``scampi.py``): the same housings, Jetson autonomy and water model.

Requirements (``REQUIREMENTS``, from the design questions): wind export and array cables first, then telecom; route
survey with depth of burial, exposures and free spans, fault localisation, light intervention (lift and tag the cable,
clear debris, place markers); rated to 300 m; 60-100 km per dive.

Layout (x forward, y left, z up; the axis at z = 0): a body of revolution (elliptic nose, parallel middle body,
tapered tail) — free-flooding fairing over two pressure housings (battery, electronics) and syntactic foam; the coil
array as two swept "antennae" ahead of the nose; two stern thrusters (surge, yaw by differential thrust) behind
cruciform fins; four tunnel thrusters through the hull (two vertical, two lateral) to hover; a belly bay with a
folding 5-DOF arm; two low skids to land beside the cable.

``part``: ``vehicle``, ``body``, ``antennae``, ``arm``, ``skids``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Design, Parameter

from scampi import FOAM_RHO, RHO_SW, Scampi

G = 9.81
REQUIREMENTS = {
    "cables": "wind export (30-100 km) and array cables first, then telecom landings and shelf routes",
    "jobs": ["route survey + depth of burial", "exposure and free-span checks", "fault localisation", "light intervention"],
    "depth_m": 300.0,
    "range_km": (60.0, 100.0),
    "survey_speed_m_s": 1.5,
    "survey_altitude_m": 2.0,       # the coil array over the seabed: depth of burial to ~3 m
    "intervention_h": 1.0,          # hover and landed work kept in reserve on every dive
    "reserve": 0.15,
}
#: Bought parts: (mass kg, volume L, average power W while surveying) — class values, not quotes.
PARTS = {
    "Jetson AGX Orin 64GB + carrier": (0.9, 0.0, 40.0),
    "INS (FOG class) + depth": (2.2, 0.9, 20.0),
    "DVL 600 kHz": (1.6, 0.6, 15.0),
    "cable tracker: 3-axis coil array + electronics (active tone / passive field)": (4.0, 2.5, 25.0),
    "side-scan sonar": (2.0, 1.0, 20.0),
    "multibeam (bathymetry, free spans)": (3.0, 1.4, 40.0),
    "sub-bottom profiler (burial check, soft sediment)": (3.0, 1.5, 30.0),
    "cameras (2) + lights, survey duty": (2.5, 1.0, 25.0),
    "acoustic modem + USBL transponder": (1.2, 0.5, 5.0),
    "2 stern thrusters (T500 class)": (2 * 1.0, 2 * 0.35, 0.0),
    "4 tunnel thrusters (T200 class)": (4 * 0.35, 4 * 0.15, 0.0),
    "5-DOF folding arm, cable gripper + tagger": (8.0, 3.2, 0.0),
    "fins, skids, wet-mate connectors, cables": (6.0, 3.0, 0.0),
}


class Emerita(Design):
    parameters = [
        Parameter("part", "vehicle", choices=("vehicle", "body", "antennae", "arm", "skids"), description="what to build"),
        Parameter("length", 3000.0, "mm", min=1000),
        Parameter("diameter", 400.0, "mm", min=150),
        Parameter("nose_length", 450.0, "mm", min=50),
        Parameter("tail_length", 900.0, "mm", min=50),
        Parameter("cd", 0.25, "", min=0.05, description="drag on the frontal area, with fins, tunnels, antennae, skids (assumed)"),
        Parameter("eta_propulsive", 0.45, "", min=0.1, description="electrical to thrust power at cruise"),
        Parameter("battery_kwh", 8.0, "kWh", min=1),
        Parameter("cell_wh_kg", 250.0, "Wh/kg", min=100),
        Parameter("housing_od", 280.0, "mm", min=100, description="both pressure housings"),
        Parameter("housing_wall", 10.0, "mm", min=2, description="Al 6061-T6, buckling at 300 m x 1.5"),
        Parameter("battery_housing_length", 1000.0, "mm", min=200),
        Parameter("electronics_housing_length", 550.0, "mm", min=200),
        Parameter("cap_thickness", 22.0, "mm", min=5),
        Parameter("antenna_span", 900.0, "mm", min=200, description="coil array span (two swept arms)"),
        Parameter("arm_reach", 800.0, "mm", min=200),
        Parameter("arm_pose", "stowed", choices=("stowed", "deployed")),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Emerita.parameters} | (over or {})

    @staticmethod
    def _housing(p, length):
        q = {"housing_od": p["housing_od"], "housing_wall": p["housing_wall"], "cap_thickness": p["cap_thickness"]}
        return Scampi.housing_check(q, length, REQUIREMENTS["depth_m"])

    @classmethod
    def sizing(cls, over=None):
        """Housings, mass and buoyancy, drag and power at survey speed, range against the battery (an hour of hover
        and landed work and the reserve kept back), hover thrust in a current."""
        p = cls.p(over)
        hb = cls._housing(p, p["battery_housing_length"])
        he = cls._housing(p, p["electronics_housing_length"])
        cells = p["battery_kwh"] * 1000 / p["cell_wh_kg"]
        D = p["diameter"] / 1000
        fairing_kg = 0.95 * math.pi * D * p["length"] / 1000 * 0.008 * 1000      # 8 mm HDPE/GRP shell
        items = {"fairing + internal frame": (fairing_kg + 8.0, (fairing_kg + 8.0) / 0.95),
                 "battery housing + cells + BMS": (hb["mass_kg"] + cells + 2.0, hb["volume_L"]),
                 "electronics housing": (he["mass_kg"] + 2.5, he["volume_L"])}
        items.update({k: (m, v) for k, (m, v, _) in PARTS.items()})
        m_dry = sum(m for m, _ in items.values())
        v_dry = sum(v for _, v in items.values()) / 1000
        net_up = 1.0
        V_f = (m_dry + net_up - RHO_SW * v_dry) / (RHO_SW - FOAM_RHO)
        m = m_dry + FOAM_RHO * V_f
        # the hull's inside: is there room for the housings and the foam? (75 % of the cylindrical part usable)
        mid = (p["length"] - p["nose_length"] - p["tail_length"]) / 1000
        inside = math.pi * (D / 2 - 0.01) ** 2 * (mid + 0.5 * (p["nose_length"] + p["tail_length"]) / 1000) * 0.75
        need = hb["volume_L"] / 1000 + he["volume_L"] / 1000 + V_f + 0.012
        v = REQUIREMENTS["survey_speed_m_s"]
        A = math.pi * D * D / 4
        drag = 0.5 * RHO_SW * p["cd"] * A * v * v
        prop_W = drag * v / p["eta_propulsive"]
        hotel = sum(w for _, _, w in PARTS.values())
        survey_W = hotel + prop_W
        hover_W = hotel + 450.0                                                   # tunnel thrusters + arm, in a slack current
        usable = p["battery_kwh"] * 1000 * (1 - REQUIREMENTS["reserve"]) - REQUIREMENTS["intervention_h"] * hover_W
        hours = usable / survey_W
        range_km = hours * v * 3.6
        speeds = {}
        for u in (1.0, 1.5, 2.0):
            dr = 0.5 * RHO_SW * p["cd"] * A * u * u
            w = hotel + dr * u / p["eta_propulsive"]
            speeds[u] = {"power_W": w, "Wh_per_km": w / (u * 3.6), "range_km": usable / w * u * 3.6}
        return {"housings": {"battery": hb, "electronics": he}, "mass_in_air_kg": m, "dry_mass_kg": m_dry,
                "foam_L": V_f * 1000, "inside_L": inside * 1000, "needed_L": need * 1000, "fits": need <= inside,
                "items": items, "drag_N": drag, "propulsion_W": prop_W, "hotel_W": hotel, "survey_W": survey_W,
                "hover_W": hover_W, "usable_survey_Wh": usable, "survey_hours": hours, "range_km": range_km,
                "speeds": speeds, "cells_kg": cells}

    # ------------------------------------------------------------------------------------------- CAD
    def _radius(self, p, x):
        """Hull radius at x [mm] from the tail tip (0) to the nose tip (length)."""
        L, R, ln, lt = p["length"], p["diameter"] / 2, p["nose_length"], p["tail_length"]
        if x >= L - ln:
            s = (L - x) / ln
            return R * math.sqrt(max(0.0, 1 - (1 - s) ** 2))
        if x <= lt:
            return 0.25 * R + 0.75 * R * (x / lt) ** 1.4
        return R

    def _body(self, p):
        L = p["length"]
        xs = [L * k / 60 for k in range(61)]
        pts = [(x - L / 2, max(self._radius(p, x), 0.5)) for x in xs]
        prof = cq.Workplane("XY").polyline([(pts[0][0], 0)] + pts + [(pts[-1][0], 0)]).close()
        body = prof.revolve(360, (0, 0, 0), (1, 0, 0))
        R = p["diameter"] / 2
        for x0 in (L * 0.18, -L * 0.22):                                          # tunnels: vertical and lateral
            body = body.cut(cq.Workplane("XY").center(x0, 0).circle(55).extrude(R * 2.2, both=True))
            body = body.cut(cq.Workplane("XZ").center(x0 + 160, 0).circle(55).extrude(R * 2.2, both=True))
        bay = cq.Workplane("XY").box(p["arm_reach"] * 0.55, R * 0.9, R * 0.6).translate((L * 0.02, 0, -R * 0.75))
        body = body.cut(bay)
        xt = -L / 2 + 40
        for a in range(4):                                                        # cruciform fins
            ang = 45 + 90 * a
            fin = (cq.Workplane("XY").polyline([(xt, 0), (xt + 260, 0), (xt + 220, R * 1.05), (xt + 60, R * 1.05)]).close()
                   .extrude(6, both=True).rotate((0, 0, 0), (1, 0, 0), ang))
            body = body.union(fin)
        yt = R * 0.25 + 90                                                        # the two stern thrusters, clear of
        strut = cq.Workplane("XY").box(40, 2 * yt + 60, 16).translate((xt + 40, 0, 0))   # the tail cone, on a strut
        body = body.union(strut)
        for sy in (1, -1):
            body = body.union(cq.Workplane("YZ").center(sy * yt, 0).circle(60).circle(53).extrude(110)
                              .translate((xt - 15, 0, 0)))
        return body

    def _antennae(self, p):
        L, span = p["length"], p["antenna_span"]
        out = None
        for sy in (1, -1):
            a = cq.Vector(L / 2 - 60, sy * 40, 0)
            for k, d in enumerate((cq.Vector(260, sy * span * 0.25, -60), cq.Vector(240, sy * span * 0.25, -40))):
                seg = cq.Workplane("XY").add(cq.Solid.makeCylinder(14 - 4 * k, d.Length, a, d))
                coil = cq.Workplane("XY").add(cq.Solid.makeSphere(28 - 6 * k, a + d))  # the coil pods
                out = seg.union(coil) if out is None else out.union(seg).union(coil)
                a = a + d
        return out

    def _arm(self, p):
        L, R, reach = p["length"], p["diameter"] / 2, p["arm_reach"]
        a = cq.Vector(L * 0.02 + reach * 0.25, 0, -R * 0.6)
        dirs = ([cq.Vector(-0.5, 0, 0), cq.Vector(0.45, 0, -0.05), cq.Vector(-0.05, 0, 0)] if p["arm_pose"] == "stowed"
                else [cq.Vector(0.35, 0, -0.12), cq.Vector(0.35, 0, -0.06), cq.Vector(0.15, 0, -0.03)])   # to the seabed
        out = None
        for k, d in enumerate(dirs):
            d = d * reach
            seg = cq.Workplane("XY").add(cq.Solid.makeCylinder(28 - 5 * k, d.Length, a, d)).union(
                cq.Workplane("XY").add(cq.Solid.makeSphere(32 - 5 * k, a)))
            out = seg if out is None else out.union(seg)
            a = a + d
        return out

    def _skids(self, p):
        L, R = p["length"], p["diameter"] / 2
        out = None
        for sy in (1, -1):
            sk = (cq.Workplane("YZ").center(sy * R * 0.75, -R - 70).circle(18).extrude(L * 0.25, both=True)
                  .union(cq.Workplane("XY").box(30, 30, 110).translate((L * 0.15, sy * R * 0.65, -R - 30)))
                  .union(cq.Workplane("XY").box(30, 30, 110).translate((-L * 0.15, sy * R * 0.65, -R - 30))))
            out = sk if out is None else out.union(sk)
        return out

    def build(self, p):
        if cq is None:
            raise ImportError("Emerita's CAD needs CadQuery")
        parts = {"body": self._body, "antennae": self._antennae, "arm": self._arm, "skids": self._skids}
        if p["part"] in parts:
            return parts[p["part"]](p)
        out = self._body(p)
        for k in ("antennae", "arm", "skids"):
            out = out.union(parts[k](p))
        return out


if __name__ == "__main__":
    s = Emerita.sizing()
    for k, h in s["housings"].items():
        print(f"{k:12s} housing: buckling {h['buckling_MPa']:.1f} MPa vs {h['design_pressure_MPa']:.1f} (margin {h['buckling_margin']:+.2f}), "
              f"{h['mass_kg']:.1f} kg, {h['volume_L']:.0f} L")
    print(f"mass in air {s['mass_in_air_kg']:.0f} kg (dry {s['dry_mass_kg']:.0f}, cells {s['cells_kg']:.0f}), foam {s['foam_L']:.0f} L; "
          f"needs {s['needed_L']:.0f} L of {s['inside_L']:.0f} L inside ({'fits' if s['fits'] else 'DOES NOT FIT'})")
    print(f"survey 1.5 m/s: drag {s['drag_N']:.0f} N, propulsion {s['propulsion_W']:.0f} W + hotel {s['hotel_W']:.0f} W = {s['survey_W']:.0f} W")
    print(f"usable for survey {s['usable_survey_Wh']:.0f} Wh -> {s['survey_hours']:.1f} h, {s['range_km']:.0f} km")
    for u, r in s["speeds"].items():
        print(f"  {u} m/s: {r['power_W']:.0f} W, {r['Wh_per_km']:.0f} Wh/km, {r['range_km']:.0f} km")
