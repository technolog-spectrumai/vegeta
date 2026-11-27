"""FULMAR — the long-haul cargo fixed-wing for the drone carrier (Ketos Clams stage 7): 5 kg of clams (or a pearl
case) from the bed to a port 30-60 km away, launched by the carrier's pneumatic catapult and caught by its recovery
net. Named after *Fulmarus glacialis*, the stiff-winged petrel that glides for hours behind the boats.

Reused, not redesigned: the airframe is the lab's parametric twin-motor ``fixed_wing.FixedWing`` (its NACA wing with
dihedral, the streamlined fuselage, two wing nacelles, the flat-plate tail), scaled to 3.6 m and given a belly cargo
pod; the sizing here is what is new. Why not FALCO / NISUS+ (2.4 m, mountain survey): their wing loading and pack are
right for range, but they land on their bellies on a skid (no deck) and their pod has no bay for 5 kg; why not
Gannet for the long legs: a quad pays ~29 Wh/km with the crate, Fulmar ~8.5 (``sizing``) - three times the reach, twice the speed.

Sizing (first order):

* lift and drag: CL = W / (q S), CD = CD0 + CL^2 / (pi e AR), P = D V / eta_prop + avionics;
* the stall at CL max 1.35 (flaps none) -> the catapult's end speed 1.25 V_s and its stroke at 6 g;
* the net: a 6 x 4 m net on two masts at the stern, the approach at 1.15 V_s into the wind over the deck;
  the carrier steams into the wind at its speed so the closing speed is V - U_wind - U_ship;
* the pack's usable energy over Wh/km -> range, the radius against a head wind (out and back).

Frame (as ``FixedWing``): wing root leading edge at the origin, chord along +X (aft), span along Y, up +Z; mm.
``part``: as ``FixedWing``'s (``aircraft``, ``wing``, ``fuselage``, ``nacelle``), ``pod`` (the cargo pod) and ``gimbal``
(the EO/IR pod: ``payload`` ``eoir`` draws it instead of the cargo pod).
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None

from vegeta.dedalus import Parameter

try:
    from fixed_wing import FixedWing
except Exception:                         # no CadQuery: the sizing still works
    FixedWing = None

RHO = 1.225
G = 9.81

MASS = {"wing (foam-core glass, carbon spar)": 3.1, "fuselage + tail": 1.9, "motors + props 2x": 2 * 0.55, "ESCs 2x": 0.3,
        "flight stack + GNSS/INS + 4G/LTE + 900 MHz link": 0.6, "camera + ADS-B in": 0.2, "servos 6x": 0.3,
        "cargo pod + hook": 0.7, "net-capture hook + nose guard": 0.25, "wiring": 0.35}

DEFAULTS = {"span": 3600.0, "root_chord": 400.0, "taper": 0.6, "dihedral_deg": 4.0, "camber": 0.04, "camber_pos": 0.4,
            "thickness": 0.14, "fuselage_length": 2100.0, "fuselage_diameter": 200.0, "nose_length": 600.0,
            "nacelle_y": 600.0, "nacelle_diameter": 80.0, "nacelle_length": 240.0, "nacelle_forward": 140.0,
            "tail_span": 1100.0, "tail_chord": 280.0, "fin_height": 380.0, "tail_thickness": 10.0}


class Fulmar(FixedWing if FixedWing is not None else object):
    parameters = ([q for q in FixedWing.parameters if q.name != "part"] if FixedWing is not None else []) + [
        Parameter("part", "aircraft", choices=("aircraft", "wing", "fuselage", "nacelle", "pod", "gimbal")),
        Parameter("payload", "cargo", choices=("cargo", "eoir"), description="the cargo pod (cargo) or the surveillance pod (eoir_payload.py)"),
        Parameter("battery_wh", 1600.0, "Wh", description="12S Li-ion, 250 Wh/kg at pack level"),
        Parameter("pack_wh_kg", 250.0, "Wh/kg"),
        Parameter("payload_kg", 5.0, "kg"),
        Parameter("cruise", 26.0, "m/s"),
        Parameter("cd0", 0.032, "", description="pod, nacelles and the hook in"),
        Parameter("oswald", 0.8, ""),
        Parameter("cl_max", 1.35, ""),
        Parameter("eta_prop", 0.62, "", description="propeller x motor x ESC"),
        Parameter("avionics_w", 30.0, "W"),
        Parameter("catapult_g", 6.0, "g"),
        Parameter("price_eur", 21000.0, "EUR"),
    ]

    @staticmethod
    def p(over=None):
        base = {q.name: q.default for q in Fulmar.parameters} | DEFAULTS
        return base | (over or {})

    @staticmethod
    def wing_area(p):
        return p["span"] / 1000 * p["root_chord"] / 1000 * (1 + p["taper"]) / 2

    @staticmethod
    def empty_kg(p):
        return sum(MASS.values()) + p["battery_wh"] / p["pack_wh_kg"]

    @classmethod
    def power(cls, p, V, payload=True):
        m = cls.empty_kg(p) + (p["payload_kg"] if payload else 0.0)
        S, b = cls.wing_area(p), p["span"] / 1000
        AR = b * b / S
        q = 0.5 * RHO * V * V
        CL = m * G / (q * S)
        CD = p["cd0"] + CL * CL / (math.pi * p["oswald"] * AR)
        return q * S * CD * V / p["eta_prop"] + p["avionics_w"]

    @classmethod
    def sizing(cls, over=None):
        p = cls.p(over)
        empty = cls.empty_kg(p)
        mtow = empty + p["payload_kg"]
        S = cls.wing_area(p)
        AR = (p["span"] / 1000) ** 2 / S
        vs = math.sqrt(2 * mtow * G / (RHO * S * p["cl_max"]))
        v_launch = 1.25 * vs
        stroke = v_launch ** 2 / (2 * p["catapult_g"] * G)
        pc, pce = cls.power(p, p["cruise"]), cls.power(p, p["cruise"], payload=False)
        usable = p["battery_wh"] * 0.8 * 0.85
        wh_km, wh_km_e = pc / p["cruise"] / 3.6, pce / p["cruise"] / 3.6
        climb_wh = mtow * G * 150.0 / 0.5 / 3600.0                       # to 150 m at 50 % climb efficiency
        radius = (usable - climb_wh) / (wh_km + wh_km_e)
        V = p["cruise"]
        hw = 10.0                                                       # a 10 m/s head wind out, tail wind back
        radius_wind = (usable - climb_wh) / (pc / (V - hw) / 3.6 + pce / (V + hw) / 3.6)
        speeds = [v * 0.5 for v in range(int(2 * vs) + 2, 80)]
        best_range = min(speeds, key=lambda v: cls.power(p, v) / v)
        ld = mtow * G / ((pc - p["avionics_w"]) * p["eta_prop"] / V)
        return {"empty_kg": round(empty, 2), "mtow_kg": round(mtow, 2), "battery_kg": round(p["battery_wh"] / p["pack_wh_kg"], 2),
                "wing_area_m2": round(S, 3), "aspect_ratio": round(AR, 1), "wing_loading_N_m2": round(mtow * G / S, 1),
                "stall_m_s": round(vs, 1), "launch_m_s": round(v_launch, 1), "catapult_stroke_m": round(stroke, 2),
                "net_closing_m_s": round(1.15 * vs - 6.0, 1),                # approach at 1.15 Vs into 6 m/s wind over deck
                "cruise_W": round(pc), "cruise_empty_W": round(pce), "L_D": round(ld, 1),
                "wh_per_km": round(wh_km, 2), "wh_per_km_empty": round(wh_km_e, 2), "usable_Wh": round(usable),
                "range_km": round(usable / wh_km, 1), "radius_km": round(radius, 1), "radius_10ms_headwind_km": round(radius_wind, 1),
                "endurance_min": round(usable / pc * 60, 1), "best_range_m_s": best_range,
                "wind_limit_m_s": 15.0, "deck_hs_limit_m": 3.0,               # the net does not need the deck still
                "price_eur": p["price_eur"]}

    @classmethod
    def eoir_sizing(cls, over=None):
        """Fulmar with the surveillance pod (``eoir_payload.POD["fulmar"]``) in the cargo bay: its mass, drag and power
        on the same drag polar; the stall and the catapult's speed; the endurance at its loiter speed (least power, not
        under 1.2 V_s) and at cruise; the range; the time on station 30 km from the carrier; the sensors' DRI from its
        altitudes, the field of view, the datalink; the radar option weighed. Wind and deck limits are the cargo
        Fulmar's (the net does not see the payload)."""
        import eoir_payload as ep
        pod = ep.pod_totals("fulmar")
        base = cls.p(over)
        p = base | {"payload_kg": pod["mass_kg"], "cd0": base["cd0"] + ep.POD["fulmar"]["cd0_delta"], "avionics_w": base["avionics_w"] + pod["power_W"]}
        mtow = cls.empty_kg(p) + p["payload_kg"]
        vs = math.sqrt(2 * mtow * G / (RHO * cls.wing_area(p) * p["cl_max"]))
        usable = p["battery_wh"] * 0.8 * 0.85
        speeds = [v * 0.5 for v in range(int(2 * 1.2 * vs) + 1, 80)]
        loiter_v = min(speeds, key=lambda v: cls.power(p, v))
        p_l, p_c = cls.power(p, loiter_v), cls.power(p, p["cruise"])
        wh_km = p_c / p["cruise"] / 3.6
        climb = mtow * G * 900.0 / 0.5 / 3600.0                         # to its patrol altitude
        d = 30.0
        on_station = (usable - climb - 2 * d * wh_km) / p_l * 60
        cargo = cls.sizing(over)
        return {"payload": "eoir", "pod": pod, "pod_parts": {k: {"kg": m, "W": w, "eur": e} for k, (m, w, e) in ep.POD["fulmar"]["parts"].items()},
                "mtow_kg": round(mtow, 2), "stall_m_s": round(vs, 1), "launch_m_s": round(1.25 * vs, 1), "loiter_m_s": loiter_v,
                "loiter_W": round(p_l), "cruise_W": round(p_c), "endurance_loiter_min": round(usable / p_l * 60, 1),
                "endurance_cruise_min": round(usable / p_c * 60, 1), "wh_per_km": round(wh_km, 2), "range_km": round(usable / wh_km, 1),
                "on_station_min_at_30km": round(on_station, 1), "wind_limit_m_s": cargo["wind_limit_m_s"], "deck_hs_limit_m": cargo["deck_hs_limit_m"],
                "rain": "light to moderate rain (<= 4 mm/h; sealed avionics, IP54-class motors and gimbal, assumed)",
                "fov": ep.fov("fulmar"), "lrf_km": ep.POD["fulmar"]["lrf_km"], "jitter_urad": ep.POD["fulmar"]["jitter_urad"],
                "dri_km": {t: {f"{a:.0f}m": ep.ground_dri("fulmar", t, a) for a in ep.POD["fulmar"]["altitudes_m"]} for t in ep.TARGETS},
                "datalink": {f"{a:.0f}m": ep.link("fulmar", a) for a in ep.POD["fulmar"]["altitudes_m"]},
                "ais_km": round(min(4.12 * (math.sqrt(ep.POD["fulmar"]["altitudes_m"][0]) + math.sqrt(3.0)), 70.0), 1),
                "radar_option": ep.RADAR_OPTION | ep.radar_fits_fulmar(base["payload_kg"])}

    # --- CAD: FixedWing's aircraft plus a cargo pod under the wing --------------------------------------------
    def _gimbal(self, p):
        """The EO/IR pod in the cargo bay: a flush fairing and the gimbal's ball below the fuselage, forward of the wing."""
        r = p["fuselage_diameter"] / 2
        fair = (cq.Workplane("XY").box(360, 220, 90).edges("|X").fillet(40).translate((-p["nose_length"] * 0.35, 0, -r - 20)))
        ball = cq.Workplane("XY").sphere(95).translate((-p["nose_length"] * 0.35, 0, -r - 110))
        return fair.union(ball)

    def _pod(self, p):
        r = p["fuselage_diameter"] / 2
        return (cq.Workplane("XY").box(520, 300, 200).edges("|X").fillet(80).edges("|Y").fillet(60)
                .translate((p["root_chord"] * 0.3, 0, -2 * r - 90)))

    def build(self, p):
        if cq is None:
            raise ImportError("Fulmar's CAD needs CadQuery")
        if p["part"] == "pod":
            return self._pod(p)
        if p["part"] == "gimbal":
            return self._gimbal(p)
        base = super().build(p)
        if p["part"] == "aircraft":
            return base.union(self._gimbal(p) if p.get("payload", "cargo") == "eoir" else self._pod(p))
        return base


if __name__ == "__main__":
    import json
    print(json.dumps(Fulmar.sizing(), indent=1))
