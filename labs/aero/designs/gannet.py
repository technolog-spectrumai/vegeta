"""GANNET — a cargo quadcopter for the drone carrier (Ketos Clams stage 6): it lifts a 5 kg crate of clams (or a
pearl case) off a 2.5 m pad on a moving deck, flies it a few kilometres to a shore station, lands, drops it, and
flies back to the deck for a battery swap. Named after *Morus bassanus*, the seabird that lives on the wind over the
fishing grounds and comes home to a crowded ledge.

Why a new design (the aero lab's survey, ``carrier_survey`` below): the lab's vertical take-off machines are the
Kestrel tailsitters (1.9 kg, a 150 g capsule, 39 Wh), Drongo (a small X-quad) and Émerillon (a speed quad with no
payload bay); the fixed-wings (FALCO, NISUS+, MERLIN, AGUYA) land on their bellies or by parachute. None lifts a few
kilograms, and a tailsitter's wing is a sail on a rolling deck in 12 m/s of wind. So Gannet is the plain heavy-lift
X-quad: four 28-inch propellers on 2216-class-scaled 6 kW-peak motors, a 12S Li-ion pack in a quick-swap sled under
the body (the carrier's swap robot pulls it in 90 s), a crate hook between the legs, wide skids with a harpoon probe
that locks into the pad's grid on touch-down (the deck can heave and roll; the probe holds 3 g).

Sizing (first order, momentum theory; the numbers in ``sizing``):

* hover: T = W, v_h = sqrt(T / 2 rho A), P = T v_h / (FM eta); forward flight: Glauert's induced velocity
  v_i^4 + V^2 v_i^2 = v_h^4, P = (T v_i + D V) / (FM eta) + profile (15 % of hover);
* energy: the pack's usable Wh (80 % depth, 15 % reserve) over the out-and-back leg with the crate out and empty
  back, two hovers at each end (take-off, landing);
* wind: the pitch it needs to hold station (tan theta = D / W) under 25 deg with thrust to spare -> the wind limit;
* deck: the skids' and the probe's grip against the deck's acceleration -> the landing sea-state limit (Hs).

Frame: x forward, y left, z up, the skids' bottom at z = 0; mm. ``part``: ``aircraft``, ``body``, ``arms``,
``rotors`` (the discs, thin), ``skids``, ``crate``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None

from vegeta.dedalus import Design, Parameter

RHO = 1.225
G = 9.81

#: Mass budget (kg): the parts that are bought or printed (class values, assumed).
MASS = {"frame + arms (carbon tube)": 2.4, "motors 4x": 4 * 0.42, "propellers 4x 28 in": 4 * 0.14, "ESCs 4x": 4 * 0.15,
        "flight stack + RTK GNSS + radio + 4G": 0.55, "downward lidar + camera (deck tracking)": 0.25,
        "crate hook + harpoon probe": 0.45, "wiring": 0.35}

#: What the lab already has, and why each does not do this job (the survey of labs/aero/designs and the branches).
SURVEY = {
    "kestrel (4Q-3 tailsitter)": "VTOL from a 1 m pad, but 1.9 kg with a 150 g capsule and 39 Wh: 1/30 of the crate",
    "drongo (X-quad)": "the X-quad layout Gannet keeps, but a 250-class frame: no payload, minutes of hover",
    "emerillon (speed quad, malakai/epic-sagan-7jp75y)": "a 6S bullet for top speed, no bay, stands on its fins",
    "falco / nisus_plus": "2.4 m mountain survey wings, belly landing on a skid: no deck recovery, a lidar payload",
    "merlin / aguya": "fast samplers (electric, turbojet), parachute or belly: no cargo, no deck",
    "fixed_wing (twin-motor)": "the parametric twin-motor wing: Fulmar (the long-haul fixed-wing) is built on it",
}


class Gannet(Design):
    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "body", "arms", "rotors", "skids", "crate")),
        Parameter("prop_in", 28.0, "in", description="propeller diameter"),
        Parameter("arm", 760.0, "mm", description="motor axis from the centre (diagonal)"),
        Parameter("body", 420.0, "mm", description="body square"),
        Parameter("battery_wh", 1150.0, "Wh", description="12S Li-ion (21700, 250 Wh/kg at pack level)"),
        Parameter("pack_wh_kg", 250.0, "Wh/kg"),
        Parameter("payload_kg", 5.0, "kg", description="a 5 kg crate of clams on ice (or a 0.6 kg pearl case)"),
        Parameter("cruise", 15.0, "m/s"),
        Parameter("drag_area", 0.14, "m2", description="CdA forward, crate on"),
        Parameter("fm", 0.68, "", description="propeller figure of merit"),
        Parameter("eta", 0.85, "", description="motor x ESC"),
        Parameter("avionics_w", 25.0, "W"),
        Parameter("skid_span", 900.0, "mm"),
        Parameter("price_eur", 9800.0, "EUR"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Gannet.parameters} | (over or {})

    @staticmethod
    def disc_area(p):
        return 4 * math.pi * (p["prop_in"] * 0.0254 / 2) ** 2

    @classmethod
    def power(cls, p, V, payload=True):
        """Electrical power (W) in level flight at V m/s (0: hover), with or without the crate."""
        m = cls.empty_kg(p) + (p["payload_kg"] if payload else 0.0)
        T = m * G
        A = cls.disc_area(p)
        vh = math.sqrt(T / (2 * RHO * A))
        cda = p["drag_area"] if payload else 0.7 * p["drag_area"]
        D = 0.5 * RHO * V * V * cda
        Tt = math.hypot(T, D)
        vh_t = math.sqrt(Tt / (2 * RHO * A))
        # Glauert: v^4 + V^2 v^2 - vh^4 = 0
        vi = math.sqrt((-V * V + math.sqrt(V ** 4 + 4 * vh_t ** 4)) / 2)
        hover = T * vh / (p["fm"] * p["eta"])
        return (Tt * vi + D * V) / (p["fm"] * p["eta"]) + 0.15 * hover + p["avionics_w"]

    @staticmethod
    def empty_kg(p):
        return sum(MASS.values()) + p["battery_wh"] / p["pack_wh_kg"]

    @classmethod
    def sizing(cls, over=None):
        p = cls.p(over)
        empty = cls.empty_kg(p)
        mtow = empty + p["payload_kg"]
        A = cls.disc_area(p)
        ph, pe = cls.power(p, 0.0), cls.power(p, 0.0, payload=False)
        pc, pce = cls.power(p, p["cruise"]), cls.power(p, p["cruise"], payload=False)
        usable = p["battery_wh"] * 0.8 * 0.85                      # 80 % depth, 15 % reserve
        wh_km, wh_km_e = pc / p["cruise"] / 3.6, pce / p["cruise"] / 3.6
        hover_s = 90.0                                             # take-off, deck approach, landing: per end
        hover_wh = (ph + pe) * hover_s * 2 / 3600.0                # loaded at both ends of the out leg, empty back
        radius = (usable - hover_wh) / (wh_km + wh_km_e)
        speeds = [v * 0.5 for v in range(2, 50)]
        best_range = min(speeds, key=lambda v: cls.power(p, v) / v)
        # wind: tilt to hold station tan(theta) = D/W; limit at 25 deg with 20 % thrust margin
        w_lim = math.sqrt(math.tan(math.radians(25.0)) * mtow * G / (0.5 * RHO * p["drag_area"]))
        thrust_max = (2 * RHO * A) \
             ** (1 / 3) * (4 * 3000.0 * p["fm"] * p["eta"]) ** (2 / 3)
        # deck: the harpoon holds 3 g; the touch-down needs the pad's vertical acceleration under 0.25 g and its
        # roll under 6 deg. A 26 m catamaran's heave acceleration ~ 0.125 g per metre Hs (beam seas, 6 s): Hs 2.0 m
        deck_hs = 0.25 / 0.125
        return {"empty_kg": round(empty, 2), "mtow_kg": round(mtow, 2), "battery_kg": round(p["battery_wh"] / p["pack_wh_kg"], 2),
                "disc_loading_N_m2": round(mtow * G / A, 1), "hover_W": round(ph), "hover_empty_W": round(pe),
                "cruise_W": round(pc), "cruise_empty_W": round(pce), "wh_per_km": round(wh_km, 2), "wh_per_km_empty": round(wh_km_e, 2),
                "usable_Wh": round(usable), "radius_km": round(radius, 1), "range_km": round(usable / wh_km, 1),
                "endurance_hover_min": round(usable / ph * 60, 1), "best_range_m_s": best_range,
                "thrust_weight": round(thrust_max / (mtow * G), 2), "wind_limit_m_s": round(min(w_lim, 14.0), 1),
                "deck_hs_limit_m": round(deck_hs, 1), "swap_s": 90.0, "price_eur": p["price_eur"]}

    # --- CAD ---------------------------------------------------------------------------------------------------
    def _rotor_at(self, p):
        a = p["arm"] / math.sqrt(2)
        return [(a, a), (a, -a), (-a, a), (-a, -a)]

    def _body(self, p):
        b = p["body"]
        body = cq.Workplane("XY").box(b, b * 0.8, 160).translate((0, 0, 420)).edges("|Z").fillet(60)
        sled = cq.Workplane("XY").box(b * 0.9, b * 0.5, 110).translate((0, 0, 285)).edges("|X").fillet(30)
        dome = cq.Workplane("XY").sphere(110).translate((b * 0.25, 0, 500))
        return body.union(sled).union(dome)

    def _arms(self, p):
        solids = []
        for x, y in self._rotor_at(p):
            L = math.hypot(x, y)
            solids.append(cq.Solid.makeCylinder(22, L, cq.Vector(0, 0, 440), cq.Vector(x / L, y / L, 0)))
            solids.append(cq.Solid.makeCylinder(48, 70, cq.Vector(x, y, 440)))              # the motor
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])

    def _rotors(self, p):
        r = p["prop_in"] * 25.4 / 2
        solids = [cq.Solid.makeCylinder(r, 6, cq.Vector(x, y, 520)) for x, y in self._rotor_at(p)]
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])

    def _skids(self, p):
        s = p["skid_span"] / 2
        solids = []
        for y in (s, -s):
            solids.append(cq.Solid.makeCylinder(14, 700, cq.Vector(-350, y, 14), cq.Vector(1, 0, 0)))
            for x in (-200, 200):
                d = cq.Vector(0, -y * 0.55, 330).normalized()
                solids.append(cq.Solid.makeCylinder(12, math.hypot(y * 0.55, 330), cq.Vector(x, y, 14), d))
        solids.append(cq.Solid.makeCylinder(10, 230, cq.Vector(0, 0, 0)))                    # the harpoon probe
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])

    def _crate(self, p):
        return cq.Workplane("XY").box(400, 300, 200).translate((0, 0, 130)).edges("|Z").fillet(15)

    def build(self, p):
        if cq is None:
            raise ImportError("Gannet's CAD needs CadQuery")
        parts = {"body": self._body, "arms": self._arms, "rotors": self._rotors, "skids": self._skids, "crate": self._crate}
        if p["part"] in parts:
            return parts[p["part"]](p)
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound([f(p).val() for f in parts.values()])])


if __name__ == "__main__":
    import json
    print(json.dumps(Gannet.sizing(), indent=1))
