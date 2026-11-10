"""Pinna: an autonomous drone-carrier catamaran for the clam bed (Ketos Clams stages 6-8), named after *Pinna
nobilis*, the noble pen shell of the Mediterranean that grows pearls and spins sea silk. It replaces the crewed
``clam_boat`` (the farm workboat with davits): an **uncrewed surface vessel** that lies over the bed on dynamic
positioning, runs four Walrus crawlers on automatic davits and tether reels, hauls their baskets with a robot,
grades and packs the clams into 5 kg iced crates, and sends the crates ashore by air - Gannet cargo quads
(``labs/aero/designs/gannet.py``) from four landing pads, Fulmar fixed-wings (``fulmar.py``) off a pneumatic
catapult, caught by a recovery net. One operator ashore supervises it (and others) over satcom / 4G / UHF.

What is on it, aft to forward (x forward, y left, z up from the waterline; mm in the CAD, m in the card):

* two slender demihulls (24 m, 2.8 m beam, 1.5 m draught, 10 m overall beam), diesel-electric: two 250 kW
  gensets in the hulls, two 280 kW pods (DP and transit), a bow tunnel thruster per hull, a 350 kWh buffer battery
  (silent DP at night, the drones' charging peaks);
* the **Walrus deck** aft: four automatic davits (two a side) with their tether reels and the USBL pole between the
  hulls; the **basket robot** (a gantry with a float-line grab and a winch: 90 s a basket against the crew's 3 min);
* the **processing house** amidships: a wash/grade conveyor, a vision grader, the crate packer (a 5 kg crate with
  ice every 60 s, labelled with the harvest time and cell for traceability), the chilled crate store (2 t);
* the **hangar** under the flight deck: 8 Gannets, 3 Fulmars, a battery room with 24 + 9 packs on 1C chargers,
  the swap robot on a rail between the pads;
* the **flight deck** on the hangar roof (12 x 9.5 m): four 2.6 m VTOL pads with a harpoon grid (centring and
  locking: the quad's probe locks on touch-down), a 7 m pneumatic catapult rail on the centreline launching forward,
  and the **recovery net** (6 x 4 m between two masts at the flight deck's aft edge, the Fulmar flies into it from
  astern while the boat heads into the wind);
* the **autonomy mast** forward: X- and S-band radar, AIS, a 360 deg lidar ring and day/IR cameras, dual RTK
  GNSS + INS, satcom (LEO) and 4G/LTE domes, UHF and the drones' 900 MHz / LTE links, ADS-B in for the air picture,
  the horn and the lights of a vessel not under command. COLREGs collision avoidance in the autopilot; no
  wheelhouse - a small transit console in the mast's base for a delivery crew in port.

The numbers (``sizing``; class values for 24 m workcats and USVs, assumed, no vendor's vessel): displacement from
the demihulls' block coefficient, the resistance by a total coefficient at the service speed, the electrical
budget on the bed, the deck's launch and recovery rates, the cost.

``part``: ``vessel`` (all), ``hulls``, ``deck``, ``house``, ``flight_deck``, ``pads``, ``catapult``, ``net``, ``mast``,
``davits``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None

from vegeta.dedalus import Design, Parameter

RHO_SW = 1025.0
G = 9.81
KN = 0.5144

#: The lightship mass budget (t): class values (assumed).
LIGHTSHIP = {"hulls + cross deck (aluminium)": 38.0, "gensets 2x250 kW + pods 2x280 kW + tunnels": 14.0,
             "buffer battery 350 kWh": 3.5, "house + hangar + flight deck": 9.0, "davits 4x + reels + basket robot": 4.0,
             "processing: conveyor, grader, packer, chiller": 3.0, "catapult + net + swap robot + pads": 2.5,
             "autonomy mast + sensors + comms": 1.2, "outfit, piping, cabling": 6.0}
#: Deadweight on the bed (t): fuel for 10 days, ice, crates, the drones, the Walruses.
DEADWEIGHT = {"fuel (diesel, 10 days)": 9.0, "fresh water + ice": 3.0, "crate store (2 t of clams)": 2.0,
              "crates + packaging": 0.6, "drones: 8 Gannet, 3 Fulmar, spare packs": 0.3, "4 Walruses + tethers": 0.25}
#: The electrical budget on the bed (kW): what runs while the Walruses work and the drones fly.
LOADS = {"DP (two pods, a tunnel, 0.5 m/s tide, 8 m/s wind)": 45.0, "4 Walruses on their tethers": 4 * 0.371 + 0.5,
         "davits + reels (average)": 2.0, "basket robot": 1.5, "processing line + vision grader": 6.0, "chiller + ice": 5.0,
         "drone chargers (12 at 1C)": 16.0, "autonomy: radars, lidar, cameras, compute, comms": 2.5, "hotel": 2.0}
#: The autonomy suite: (what, why).
AUTONOMY = {"radar": "X- and S-band ARPA: targets out to 12 nm, the collision avoidance's tracks",
            "ais": "class A transponder: the others see it, it sees them",
            "lidar_cameras": "360 deg lidar ring, day and IR cameras: small craft, buoys, swimmers, the drones on deck",
            "gnss_ins": "dual RTK GNSS + fibre-optic INS: DP over the bed, the pads' motion for the drones",
            "comms": "LEO satcom + 4G/LTE + UHF to shore; 900 MHz + LTE to the drones; ADS-B in",
            "colregs": "COLREGs rules 8-17 in the autopilot: give way / stand on, a closest-approach limit of 0.3 nm",
            "supervision": "one shore operator per 3 vessels: alarms, a remote takeover (teleop) when a fault asks"}


class Pinna(Design):
    parameters = [
        Parameter("part", "vessel", choices=("vessel", "hulls", "deck", "house", "flight_deck", "pads", "catapult", "net", "mast", "davits")),
        Parameter("length", 24000.0, "mm"),
        Parameter("beam", 10000.0, "mm", description="overall"),
        Parameter("hull_beam", 2800.0, "mm", description="each demihull"),
        Parameter("draught", 1500.0, "mm"),
        Parameter("hull_depth", 3200.0, "mm", description="keel to the main deck"),
        Parameter("cb", 0.55, "", description="demihull block coefficient"),
        Parameter("service_kn", 10.0, "kn"),
        Parameter("ct", 0.0065, "", description="total resistance coefficient on wetted area at the service speed"),
        Parameter("pods_kw", 2 * 280.0, "kW"),
        Parameter("gensets_kw", 2 * 250.0, "kW"),
        Parameter("battery_kwh", 350.0, "kWh"),
        Parameter("pads", 4, ""),
        Parameter("hangar_gannets", 8, ""),
        Parameter("hangar_fulmars", 3, ""),
        Parameter("gannet_packs", 24, ""),
        Parameter("fulmar_packs", 9, ""),
        Parameter("davits", 4, ""),
        Parameter("crate_kg", 5.0, "kg"),
        Parameter("crate_store_kg", 2000.0, "kg"),
        Parameter("pack_s", 60.0, "s", description="the packer: one 5 kg crate with ice"),
        Parameter("basket_haul_s", 90.0, "s", description="the basket robot: grab the float line, winch, empty, back"),
        Parameter("pad_cycle_s", 170.0, "s", description="a Gannet on a pad: land and lock 40, crate on 40, swap 90"),
        Parameter("catapult_cycle_s", 240.0, "s", description="a Fulmar loaded, armed and launched"),
        Parameter("net_cycle_s", 300.0, "s", description="a Fulmar out of the net, the net reset"),
        Parameter("deck_hs_limit", 2.0, "m", description="the Gannets' landing limit on its pads (their card)"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Pinna.parameters} | (over or {})

    @classmethod
    def sizing(cls, over=None):
        p = cls.p(over)
        L, B, T = p["length"] / 1000, p["hull_beam"] / 1000, p["draught"] / 1000
        lwl = 0.95 * L
        vol = 2 * lwl * B * T * p["cb"]
        disp_t = vol * RHO_SW / 1000
        light, dwt = sum(LIGHTSHIP.values()), sum(DEADWEIGHT.values())
        wetted = 2 * lwl * (B + 2 * T) * 0.85
        V = p["service_kn"] * KN
        R = 0.5 * RHO_SW * wetted * p["ct"] * V * V
        p_service = R * V / 0.6 / 1000
        # the top speed the pods allow (Ct held, cubic)
        v_max = V * (p["pods_kw"] * 0.95 / p_service) ** (1 / 3) / KN
        load = sum(LOADS.values())
        fuel_l_h = (load / 0.85) * 0.25 / 0.84                       # 250 g/kWh at the shaft, 0.84 kg/L
        fuel_days = DEADWEIGHT["fuel (diesel, 10 days)"] * 1000 / 0.84 / fuel_l_h / 24
        pad_rate = p["pads"] * 3600.0 / p["pad_cycle_s"]
        price = {"hulls + machinery": 2.3e6, "autonomy suite (sensors, compute, comms, COLREGs, certification)": 0.55e6,
                 "flight deck automation (pads, swap robot, catapult, net)": 0.42e6, "processing + packing + chiller": 0.26e6,
                 "4 davits + reels + basket robot": 0.24e6, "drones: 8 Gannet, 3 Fulmar, packs": 8 * 9800 + 3 * 21000 + 24 * 450 + 9 * 620}
        return {"length_m": L, "beam_m": p["beam"] / 1000, "draught_m": T, "displacement_t": round(disp_t, 1),
                "lightship_t": round(light, 1), "deadweight_t": round(dwt, 2), "draught_margin_t": round(disp_t - light - dwt, 1),
                "wetted_m2": round(wetted, 1), "resistance_kN": round(R / 1000, 1), "service_kn": p["service_kn"],
                "service_kW": round(p_service), "max_kn": round(v_max, 1), "bed_load_kW": round(load, 1),
                "fuel_l_h_on_bed": round(fuel_l_h, 1), "endurance_days": round(fuel_days, 1),
                "gensets_margin": round(p["gensets_kw"] / max(p_service, load) - 1, 2),
                "deck_spots": {"vtol_pads": p["pads"], "catapult": 1, "net": 1},
                "vtol_recoveries_per_h": round(pad_rate, 1), "vtol_launches_per_h": round(p["pads"] * 3600 / 60.0, 1),
                "fixed_wing_launches_per_h": round(3600 / p["catapult_cycle_s"], 1), "fixed_wing_recoveries_per_h": round(3600 / p["net_cycle_s"], 1),
                "crates_per_h": round(3600 / p["pack_s"]), "pack_kg_h": round(3600 / p["pack_s"] * p["crate_kg"]),
                "baskets_per_h": round(3600 / p["basket_haul_s"]), "air_kg_h_vtol_deck_limit": round(pad_rate * p["crate_kg"]),
                "charging_kW": LOADS["drone chargers (12 at 1C)"], "crew": 0, "operators_ashore": 1 / 3,
                "price": price, "price_eur": round(sum(price.values())), "day_rate_eur": 1400.0}

    # --- CAD ---------------------------------------------------------------------------------------------------
    def _hulls(self, p):
        L, b, H, y = p["length"], p["hull_beam"], p["hull_depth"], p["beam"] / 2 - p["hull_beam"] / 2
        T = p["draught"]
        out = []
        for s in (1, -1):
            body = cq.Workplane("XY").box(L * 0.9, b, H).translate((-L * 0.05, s * y, H / 2 - T)).edges("|Z").fillet(b * 0.45)
            bow = (cq.Workplane("XZ").polyline([(L * 0.4, -T), (L * 0.5, H - T), (L * 0.4, H - T)]).close()
                   .extrude(b / 2, both=True).translate((0, s * y, 0)))
            out.append(body.union(bow).val())
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(out)])

    def _deck(self, p):
        L, B, z = p["length"], p["beam"], p["hull_depth"] - p["draught"]
        return cq.Workplane("XY").box(L * 0.92, B, 400).translate((-L * 0.04, 0, z + 200))

    def _z_deck(self, p):
        return p["hull_depth"] - p["draught"] + 400

    def _house(self, p):
        z = self._z_deck(p)
        proc = cq.Workplane("XY").box(7000, 7600, 2600).translate((3500, 0, z + 1300))         # processing house
        hangar = cq.Workplane("XY").box(12000, 9200, 3600).translate((-3000, 0, z + 1800))    # hangar under the flight deck
        return proc.union(hangar)

    def _flight_deck(self, p):
        z = self._z_deck(p) + 3600
        return cq.Workplane("XY").box(12500, 9600, 150).translate((-3000, 0, z + 75))

    def _pad_at(self, p):
        return [(-6200, 2400), (-6200, -2400), (-2600, 2400), (-2600, -2400)][:int(p["pads"])]

    def _pads(self, p):
        z = self._z_deck(p) + 3600 + 150
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound([cq.Solid.makeCylinder(1300, 40, cq.Vector(x, y, z)) for x, y in self._pad_at(p)])])

    def _catapult(self, p):
        z = self._z_deck(p) + 3600 + 150
        rail = cq.Workplane("XY").box(7000, 300, 250).translate((-200, 0, z + 500))
        legs = [cq.Solid.makeBox(200, 200, 500, cq.Vector(x, -100, z)) for x in (-3400, 0, 3000)]
        ram = cq.Solid.makeCylinder(180, 2800, cq.Vector(-3600, 0, z + 300), cq.Vector(1, 0, 0))
        return rail.union(cq.Workplane("XY").newObject([cq.Compound.makeCompound(legs + [ram])]))

    def _net(self, p):
        z = self._z_deck(p) + 3600 + 150
        x = -9100
        masts = [cq.Solid.makeCylinder(110, 5200, cq.Vector(x, y, z)) for y in (3200, -3200)]
        net = cq.Solid.makeBox(60, 6000, 4000, cq.Vector(x - 30, -3000, z + 1000))
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(masts + [net])])

    def _mast(self, p):
        z = self._z_deck(p)
        base = cq.Workplane("XY").box(2400, 2400, 2600).translate((8200, 0, z + 1300)).edges("|Z").fillet(300)
        pole = cq.Workplane("XY").add(cq.Solid.makeCylinder(220, 5200, cq.Vector(8200, 0, z + 2600)))
        radar = cq.Workplane("XY").box(400, 3200, 220).translate((8200, 0, z + 6400))
        domes = [cq.Solid.makeSphere(380, cq.Vector(8200, y, z + 7300)) for y in (900, -900)]
        lidar = cq.Solid.makeCylinder(260, 300, cq.Vector(8200, 0, z + 7800))
        return base.union(pole).union(radar).union(cq.Workplane("XY").newObject([cq.Compound.makeCompound(domes + [lidar])]))

    def _davits(self, p):
        z = self._z_deck(p)
        y = p["beam"] / 2 - 400
        solids = []
        for x in (-9500, -11500):
            for s in (1, -1):
                solids.append(cq.Solid.makeBox(300, 300, 2600, cq.Vector(x - 150, s * y - 150, z)))
                solids.append(cq.Solid.makeBox(250, 1500, 250, cq.Vector(x - 125, s * y - (0 if s > 0 else 1500) + (0 if s > 0 else 0), z + 2400)))
                solids.append(cq.Solid.makeCylinder(500, 900, cq.Vector(x, s * (y - 1800), z + 500), cq.Vector(0, 1, 0)))
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])

    def build(self, p):
        if cq is None:
            raise ImportError("Pinna's CAD needs CadQuery")
        parts = {"hulls": self._hulls, "deck": self._deck, "house": self._house, "flight_deck": self._flight_deck, "pads": self._pads,
                 "catapult": self._catapult, "net": self._net, "mast": self._mast, "davits": self._davits}
        if p["part"] in parts:
            return parts[p["part"]](p)
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound([f(p).val() for f in parts.values()])])


if __name__ == "__main__":
    import json
    print(json.dumps(Pinna.sizing(), indent=1))
