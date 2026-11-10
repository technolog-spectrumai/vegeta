"""Dolphin: an autonomous electric submarine that works a pearl farm with Walrus (``walrus.md``) and flies the pearls
out with Emerillion drones (``labs/aero/designs/emerillion.py``: a tail-sitter Merlin that lands vertically).

The loop: Walrus harvests the farm's bottom-cultured pearl oysters and releases its baskets on float lines; Dolphin
swims the farm's lanes, picks each basket's line with its belly hook into the wet-well, and its pearl cell opens and
sorts the oysters, dropping each pearl into a 200 g capsule; when a capsule is full Dolphin surfaces, opens the hangar
hatch, raises an Emerillion nose-up on the deck pad, and the drone flies the capsule to the farm's small vessel and
comes back to land vertically on the pad; the lift lowers it, the hatch closes, Dolphin dives again. Two drones: one
flying, one charging; the pearls never wait for the boat, and the boat never has to find the submarine.

Requirements (``REQUIREMENTS``): a day on the farm on one charge (40 km of lanes, 12 h); 60 m depth; two drones;
launch and recovery up to sea state 2; the vessel within 15 km; under EUR 250k.

Layout (x forward, y left, z up; z = 0 the keel; the waterline when surfaced at ``freeboard`` below the deck): a
torpedo hull Ø ``hull_d`` with an elliptical nose and a tapered tail, one main propeller in a duct with cross
tunnel thrusters fore and aft, a sail that carries the mast (GPS, radio, camera) and the deck pad; the hangar bay
behind the sail with its hatch and lift; the wet-well under the belly with the hook and the basket rack; the pearl
cell and the battery amidships.

``part``: ``vehicle``, ``hull``, ``sail``, ``hangar``, ``pad``, ``wetwell``, ``propulsor``, ``drone``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Design, Parameter

RHO_SW, G = 1025.0, 9.81
REQUIREMENTS = {
    "farm": "a bottom-culture pearl farm: 20-60 m, lanes of seeded oysters; a small vessel within 15 km",
    "jobs": ["collect Walrus's baskets", "open and sort the oysters: the pearls into capsules", "fly the capsules to the vessel by drone",
             "the farm's survey as it goes"],
    "endurance": "a day on the farm on one charge: 40 km of lanes, 12 h",
    "depth_m": 60.0,
    "drones": 2,
    "sea_state_max": 2,
    "vessel_km": 15.0,
    "price_eur": 250000.0,
}
#: The Emerillion as Dolphin sees it (``labs/aero/designs/emerillion.py`` is the design): the numbers the hangar,
#: the pad and the flight plan use.
EMERILLION = {"mass_kg": 3.11, "span_m": 0.7, "folded_span_m": 0.25, "length_m": 1.05, "payload_kg": 0.2,
              "cruise_m_s": 28.0, "hover_W": 1270.0, "cruise_W": 420.0, "battery_wh": 222.0, "hover_s": 60.0,
              "charge_h": 1.0, "price_eur": 4800.0}      # the X-wing: two 80 mm EDFs in the wing-tip pods
#: Bought and built parts: (mass kg, volume L, average power W, price EUR) — class values, not quotes.
PARTS = {
    "hull (GRP, Ø600 x 3600, 60 m) + frames + foam": (118.0, 175.0, 0.0, 40000.0),
    "main propulsor (ducted, 2 kW) + 2 tunnel thrusters + rudders/planes": (28.0, 12.0, 0.0, 14000.0),
    "LiFePO4 battery 12 kWh, 48 V + BMS": (95.0, 55.0, 0.0, 9000.0),
    "Jetson AGX Orin + navigation (INS, DVL, USBL, GPS on the mast)": (6.0, 3.0, 60.0, 22000.0),
    "sail + mast (GPS, 4G/LoRa radio, camera) + deck pad with latch": (18.0, 14.0, 10.0, 7000.0),
    "hangar bay (dry, hatch, lift for two drones) + chargers": (24.0, 70.0, 20.0, 16000.0),
    "wet-well + belly hook + basket rack (4 baskets)": (14.0, 0.0, 15.0, 6000.0),
    "pearl cell (opener, camera sorter, capsule carousel)": (22.0, 30.0, 100.0, 28000.0),
    "cameras + lights + sonar": (5.0, 2.0, 30.0, 9000.0),
    "ballast tanks + pump (surfacing)": (12.0, 0.0, 10.0, 5000.0),
    "assembly, cabling, test": (8.0, 2.0, 0.0, 30000.0),
}


class Dolphin(Design):
    parameters = [
        Parameter("part", "vehicle", choices=("vehicle", "hull", "sail", "hangar", "bay", "hatch", "pad", "wetwell", "propulsor", "drone"),
                  description="what to build"),
        Parameter("length", 3600.0, "mm", min=1500),
        Parameter("hull_d", 600.0, "mm", min=300),
        Parameter("freeboard", 250.0, "mm", min=100, description="deck above the waterline when surfaced (the pad dry)"),
        Parameter("cruise", 1.5, "m/s", min=0.3),
        Parameter("lanes_km", 40.0, "km", min=1, description="the farm's lanes in a day"),
        Parameter("oysters_per_h", 300.0, "per h", min=10, description="what Walrus delivers (pearl oysters, one pearl each)"),
        Parameter("pearl_g", 1.0, "g", min=0.1),
        Parameter("vessel_km", 10.0, "km", min=1, description="the flight to the vessel"),
        Parameter("drones_out", 2, "", min=1, max=2, description="drones in the rotation (one flies while the other charges)"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Dolphin.parameters} | (over or {})

    @classmethod
    def sizing(cls, over=None):
        """Mass and buoyancy (surfaced with the deck dry, submerged neutral), speed and power, the day's energy, the
        pearl flow and the flights it needs, the hangar's fit, the price."""
        p = cls.p(over)
        m_parts = sum(a for a, _, _, _ in PARTS.values())
        m = m_parts + REQUIREMENTS["drones"] * EMERILLION["mass_kg"]
        V = sum(b for _, b, _, _ in PARTS.values()) / 1000 + 0.02                                 # the drones' bay volume is in the hangar
        L, D = p["length"] / 1000, p["hull_d"] / 1000
        hull_V = 0.72 * math.pi * (D / 2) ** 2 * L                                                  # a torpedo's prismatic volume
        trim_submerged = RHO_SW * hull_V - m                                                        # >0: ballast water to carry neutral
        ballast_for_freeboard = RHO_SW * (0.72 * math.pi * (D / 2) ** 2 * L * (p["freeboard"] / p["hull_d"]) * 1.2)   # what the tanks blow to lift the deck
        # drag and power: a torpedo hull, Cd 0.1 on the frontal area plus the sail and the duct
        cda = 0.12 * math.pi * (D / 2) ** 2 + 0.35 * 0.08
        drag = 0.5 * RHO_SW * cda * p["cruise"] ** 2
        prop_W = drag * p["cruise"] / 0.55
        hotel = sum(w for _, _, w, _ in PARTS.values())
        lanes_h = p["lanes_km"] * 1000 / p["cruise"] / 3600
        day_wh = (prop_W + hotel) * lanes_h + hotel * (12.0 - lanes_h) + REQUIREMENTS["drones"] * EMERILLION["battery_wh"] * 4 / 0.9
        battery_wh = 12000.0 * 0.9
        energy_margin = battery_wh / day_wh - 1
        # pearls and flights
        pearls_h = p["oysters_per_h"]
        capsule_pearls = EMERILLION["payload_kg"] * 1000 / p["pearl_g"]
        capsule_h = capsule_pearls / pearls_h
        flight_min = 2 * p["vessel_km"] * 1000 / EMERILLION["cruise_m_s"] / 60 + EMERILLION["hover_s"] / 60 + 3.0   # out, back, the hovers, the hand-over on the vessel's pad
        flight_wh = EMERILLION["cruise_W"] * (2 * p["vessel_km"] * 1000 / EMERILLION["cruise_m_s"]) / 3600 + EMERILLION["hover_W"] * EMERILLION["hover_s"] / 3600
        flight_reserve = EMERILLION["battery_wh"] * 0.8 / flight_wh - 1
        cycle_h = max(capsule_h, (flight_min / 60 + EMERILLION["charge_h"]) / p["drones_out"])
        flights_day = 12.0 / cycle_h
        pearls_day = min(pearls_h * 12.0, flights_day * capsule_pearls)
        surfacing_min = 2 * REQUIREMENTS["depth_m"] / 0.5 / 60 + 4.0 + 4.0                        # up, hatch+lift out, lift in+hatch, down
        hangar_fit = (EMERILLION["folded_span_m"] + 0.1 < D - 0.1) and (EMERILLION["length_m"] + 0.2 < 0.4 * L)
        price = sum(c for _, _, _, c in PARTS.values()) + REQUIREMENTS["drones"] * EMERILLION["price_eur"]
        return {"mass_kg": m, "hull_volume_L": hull_V * 1000, "trim_submerged_kg": trim_submerged, "ballast_blow_kg": ballast_for_freeboard,
                "drag_N": drag, "prop_W": prop_W, "hotel_W": hotel, "lanes_h": lanes_h, "day_wh": day_wh, "energy_margin": energy_margin,
                "capsule_pearls": capsule_pearls, "capsule_h": capsule_h, "flight_min": flight_min, "flight_reserve": flight_reserve,
                "cycle_h": cycle_h, "flights_day": flights_day, "pearls_day": pearls_day, "surfacing_min": surfacing_min,
                "hangar_fit": hangar_fit, "price_eur": price}

    # ------------------------------------------------------------------------------------------- CAD
    def _hull(self, p):
        L, D = p["length"], p["hull_d"]
        z = D / 2
        # the parallel body is 0.6 L, the hemispherical nose D/2, the tail cone 0.3 L: 3.54 m of the 3.6 m length
        body = cq.Workplane("YZ").center(0, z).circle(D / 2).extrude(L * 0.3, both=True)
        nose = cq.Workplane("XY").add(cq.Solid.makeSphere(D / 2, cq.Vector(0, 0, 0))).val().scale(1.0)
        nose = cq.Workplane("XY").add(nose).translate((L * 0.3, 0, z))
        nose = nose.intersect(cq.Workplane("XY").box(L, D * 2, D * 2).translate((L * 0.3 + L / 2, 0, z)))
        tail = cq.Workplane("XY").add(cq.Solid.makeCone(D / 2, 60, L * 0.3, cq.Vector(-L * 0.3, 0, z), cq.Vector(-1, 0, 0)))
        return body.union(nose).union(tail)

    def _sail(self, p):
        L, D = p["length"], p["hull_d"]
        sail = cq.Workplane("XY").box(500, 180, 300).edges("|Z").fillet(80).translate((L * 0.12, 0, D + 150))
        mast = cq.Workplane("XY").circle(25).extrude(400).translate((L * 0.12 + 150, 0, D + 300))
        return sail.union(mast)

    def _bay(self, p):
        L, D = p["length"], p["hull_d"]
        return cq.Workplane("XY").box(1250, 420, 330).translate((-L * 0.2, 0, D - 165))

    def _hatch(self, p):
        """The hangar hatch, hinged along its +Y edge (Ketos swings it about that line)."""
        L, D = p["length"], p["hull_d"]
        return cq.Workplane("XY").box(1250, 440, 20).translate((-L * 0.2, 0, D + 10))

    def _hangar(self, p):
        return self._bay(p).union(self._hatch(p))

    def _pad(self, p):
        L, D = p["length"], p["hull_d"]
        pad = cq.Workplane("XY").circle(220).extrude(16).translate((-L * 0.2, 0, D + 20))
        cone = cq.Workplane("XY").add(cq.Solid.makeCone(60, 20, 80, cq.Vector(-L * 0.2, 0, D + 36), cq.Vector(0, 0, 1)))
        return pad.union(cone)

    def _wetwell(self, p):
        L, D = p["length"], p["hull_d"]
        well = cq.Workplane("XY").box(900, 360, 180).translate((L * 0.05, 0, 0))
        hook = cq.Workplane("XY").box(40, 20, 160).translate((L * 0.05 + 380, 0, -80))
        return well.union(hook)

    def _propulsor(self, p):
        L, D = p["length"], p["hull_d"]
        duct = cq.Workplane("YZ").workplane(offset=-L * 0.6 - 60).center(0, D / 2).circle(150).circle(135).extrude(130)
        hub = cq.Workplane("YZ").workplane(offset=-L * 0.6 - 40).center(0, D / 2).circle(40).extrude(90)
        fins = None
        for a in (0, 90, 180, 270):
            f = cq.Workplane("XY").box(220, 10, 180).translate((-L * 0.55, 0, D / 2 + 150)).rotate((-L * 0.55, 0, D / 2), (-L * 0.55 + 1, 0, D / 2), a)
            fins = f if fins is None else fins.union(f)
        return duct.union(hub).union(fins)

    def _drone(self, p):
        """An Emerillion standing nose-up on the pad (a stand-in for the X-wing: fuselage, the 0.7 m wing with its
        two tip pods, four X fins as the legs)."""
        L, D = p["length"], p["hull_d"]
        x0, z0 = -L * 0.2, D + 36
        fus = cq.Workplane("XY").circle(45).extrude(1050).translate((x0, 0, z0 + 40))
        wing = cq.Workplane("XY").box(160, 600, 18).translate((x0, 0, z0 + 560))
        for sy in (1, -1):
            wing = wing.union(cq.Workplane("XY").circle(47).extrude(260).translate((x0, sy * 302, z0 + 440)))
        legs = None
        for a in (45, 135, 225, 315):
            leg = cq.Workplane("XY").box(130, 6, 180).translate((x0 + 110, 0, z0 + 130)).rotate((x0, 0, 0), (x0, 0, 1), a)
            legs = leg if legs is None else legs.union(leg)
        return fus.union(wing).union(legs)

    def build(self, p):
        parts = {"hull": self._hull, "sail": self._sail, "hangar": self._hangar, "pad": self._pad, "wetwell": self._wetwell,
                 "propulsor": self._propulsor, "drone": self._drone}
        if p["part"] in ("bay", "hatch"):
            return getattr(self, "_" + p["part"])(p)
        if p["part"] in parts:
            return parts[p["part"]](p)
        out = None
        for name in parts:
            s = parts[name](p)
            out = s if out is None else out.union(s)
        return out


if __name__ == "__main__":
    s = Dolphin.sizing()
    print(f"{s['mass_kg']:.0f} kg, hull {s['hull_volume_L']:.0f} L (carries {s['trim_submerged_kg']:.0f} kg of ballast water neutral; blows {s['ballast_blow_kg']:.0f} kg to surface); "
          f"{s['drag_N']:.0f} N at cruise -> {s['prop_W']:.0f} W + {s['hotel_W']:.0f} W hotel; the day {s['day_wh']/1000:.1f} kWh (margin {s['energy_margin']:+.2f}); "
          f"a {s['capsule_pearls']:.0f}-pearl capsule every {s['capsule_h']*60:.0f} min, a flight {s['flight_min']:.0f} min (reserve {s['flight_reserve']:+.2f}), "
          f"{s['flights_day']:.1f} flights and {s['pearls_day']:.0f} pearls a day, {s['surfacing_min']:.0f} min on the surface each; hangar fits {s['hangar_fit']}; EUR {s['price_eur']:.0f}")
