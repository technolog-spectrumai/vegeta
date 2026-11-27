"""Esmeralda: the guard-duty variant of Zarco (``zarco.py``, the unmanned drone-carrier catamaran, formerly Pinna).
The hull, the propulsion, the autonomy mast, the flight deck (4 VTOL pads, catapult, recovery net, swap robot) and
the deck robots are Zarco's, unchanged. What changes:

* **removed**: the clam processing and packing line (the processing house, the packing cell, the chiller and the
  crate store) and the Walrus deck (4 compensated davits, tether reels, the basket gantry, the USBL pole);
* **added aft, in their place: a boat bay** down the centreline between the demihulls, from the transom to the
  middle of the hangar, for **Barracuda** (``barracuda.py``, the interceptor): an 8 deg stern ramp with a capture
  hook for the boat's bow eye, a haul-in winch and a cradle with keel guides at the ramp's top, under the flight
  deck; the cross deck is cut for it and the hangar's walls and the flight deck bridge the slot (a reinforced box
  girder, its mass counted);
* **effectors**: a long-range acoustic hailer and two searchlights on the mast, a water cannon (monitor) on the bow
  with an electric pump off the gensets' bus;
* **sensors**: Zarco's radars, AIS, lidar ring and day/IR cameras kept; a long-range gimballed EO/IR turret on the
  mast added; a diver-detection sonar considered and not fitted (``SONAR_OPTION``);
* **the hangar re-planned for guard duty**: the bay takes its centre, so two side bays of 2.5 m: port, 4 Gannets
  with the EO/IR payload and the payload workshop; starboard, 2 Fulmars (wings along the bay) and the battery room.
  Chargers, packs and the deck's rates re-sized for keeping a drone of each kind in the air.

A **flare or marker launcher is not fitted**: whether a civilian unmanned vessel may fire pyrotechnics at or near
another craft (distress signals are reserved by COLREGs Annex IV; illuminating flares are a rules-of-engagement
question for the flag state and the platform's operator) is the owner's decision (``FLARE_LAUNCHER``).

The sizing re-runs Zarco's with the changed weights and loads, and adds what Zarco's had no need of: each weight's
position, so the trim and the stability (catamaran waterplane, class coefficients) are checked against Zarco's.
Class values throughout (assumed), as in Zarco. x forward from midship, y left, z up from the waterline.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Parameter

import sensing
import zarco
from barracuda import Barracuda
from zarco import G, RHO_SW, Zarco

#: Zarco's weights with their centres (t, x m from midship +forward, z m above the waterline). The masses are
#: ``zarco.LIGHTSHIP`` / ``zarco.DEADWEIGHT``; the centres are class estimates from Zarco's CAD layout (assumed).
ZARCO_CENTRES = {
    "hulls + cross deck (aluminium)": (-1.0, 0.6), "gensets 2x250 kW + pods 2x280 kW + tunnels": (-5.0, -0.6),
    "buffer battery 350 kWh": (0.0, -0.5), "house + hangar + flight deck": (-1.2, 4.0),
    "davits 4x (compensated) + reels + basket robot": (-10.5, 3.2), "processing: conveyor, grader, packer, chiller": (3.5, 2.8),
    "catapult + net + swap robot + pads": (-4.0, 6.5), "autonomy mast + sensors + comms": (8.2, 6.5),
    "outfit, piping, cabling": (-0.5, 1.5),
    "fuel (diesel, 10 days)": (-1.0, -0.8), "fresh water + ice": (2.0, 1.0), "crate store (2 t of clams)": (3.5, 2.8),
    "crates + packaging": (3.5, 2.8), "drones: 8 Gannet, 3 Fulmar, spare packs": (-3.0, 3.5), "4 Walruses + tethers": (-10.5, 2.4),
}
#: What Esmeralda takes off Zarco (t), and what of Zarco's house goes with the processing house (its share of the
#: house + hangar + flight deck item, class).
REMOVED = ("davits 4x (compensated) + reels + basket robot", "processing: conveyor, grader, packer, chiller",
           "crate store (2 t of clams)", "crates + packaging", "4 Walruses + tethers", "drones: 8 Gannet, 3 Fulmar, spare packs")
HOUSE_PROCESSING_SHARE_T = 2.5
RAMP_FOOT_Z = -0.3                     # the ramp's foot at the transom, below the waterline (m)

#: The long-range EO/IR turret on the mast (class values, assumed): a cooled MWIR 1280 and an HD day camera on long
#: lenses, a laser rangefinder, a ship-stabilised gimbal (10 urad residual).
TURRET = {"what": "long-range EO/IR turret: HD day 1920 px (0.5 deg narrow), cooled MWIR 1280 px (0.9 deg narrow), eye-safe laser rangefinder 20 km",
          "channels": [{"band": "visible", "pixels_h": 1920, "hfov_narrow_deg": 0.5, "hfov_wide_deg": 30.0, "zoom": 60},
                       {"band": "ir", "pixels_h": 1280, "hfov_narrow_deg": 0.9, "hfov_wide_deg": 25.0, "zoom": 28}],
          "lrf_km": 20.0, "jitter_urad": 10.0, "mass_t": 0.12, "power_kW": 0.6, "price_eur": 450000.0}
#: Zarco's own situational cameras (kept): a ring of day 4K and uncooled LWIR 640 cameras, 60 deg each (class).
RING = {"channels": [{"band": "visible", "pixels_h": 3840, "hfov_narrow_deg": 60.0}, {"band": "ir", "pixels_h": 640, "hfov_narrow_deg": 60.0, "netd_mK": 50.0}]}
RADARS = {"X": {"ref_range_km": 14.0, "ref_rcs_m2": 10.0, "what": "X-band ARPA, 8 ft array: small craft, the collision avoidance's tracks"},
          "S": {"ref_range_km": 12.0, "ref_rcs_m2": 10.0, "what": "S-band ARPA: through rain and heavy clutter, ships at range"}}
#: The guard effectors on the carrier (mass t, power kW while in use, price EUR; class values, assumed).
EFFECTORS = {
    "hailer": {"what": "long-range acoustic hailer, 160 dB class at 1 m, 15 deg beam at 2 kHz, on the mast's pan-tilt", "spl_1m_db": 160.0,
               "mass_t": 0.10, "power_kW": 1.0, "price_eur": 90000.0},
    "searchlights": {"what": "two LED searchlights on the mast's wings, remote pan-tilt, slaved to the turret", "count": 2, "lumens": 35000.0,
                     "beam_deg": 2.5, "mass_t": 0.12, "power_kW": 1.6, "price_eur": 50000.0},
    "water_cannon": {"what": "bow monitor on a 1.5 m pedestal; electric pump off the gensets' bus in the starboard hull's fore peak, its own sea chest",
                     "flow_l_min": 3000.0, "pressure_bar": 12.0, "mass_t": 1.25, "price_eur": 120000.0},
}
FLARE_LAUNCHER = {"fitted": False, "space_reserved": "a 0.5 x 0.5 m foundation on the mast house roof, 60 kg",
                  "why_not": "pyrotechnics from a civilian unmanned vessel: distress signals are reserved (COLREGs Annex IV); illuminating "
                             "or warning flares near a craft are a rules-of-engagement question for the flag state and the platform's "
                             "operator - not assumed. Non-pyrotechnic alternative: self-locating datum marker buoys (a light and an AIS/GNSS "
                             "beacon) dropped to mark a contact's track - also not fitted until agreed."}
#: What the guard duty adds to Zarco's autonomy (``zarco.AUTONOMY``).
AUTONOMY_ADDED = {"guard": "the guard picture: radar, turret, ring cameras, AIS and the drones' tracks fused; contacts classified with "
                           "the operator; the escalation ladder's interlocks (no step above a hail without a live operator)",
                  "boat_control": "Barracuda's MANET and UHF command link from the mast, its takeover and lost-link rendezvous"}
#: A hull-mounted diver-detection sonar, considered (class values, assumed).
SONAR_OPTION = {"fitted": False, "what": "hull-mounted diver-detection sonar, ~70 kHz, 360 deg",
                "open_circuit_diver_m": 700, "closed_circuit_diver_m": 350, "small_uuv_m": 400,
                "on_dp_near_the_jacket": "thrusters' noise and the jacket's reverberation roughly halve those ranges",
                "mass_t": 0.6, "power_kW": 1.5, "price_eur": 380000.0,
                "verdict": "not worth it on the carrier: it moves with the guard picture and works on DP, noisy, beside a steel jacket that "
                           "clutters it; the underwater layer is Squilla's (resident at the jacket). If the platform's owner wants diver "
                           "detection it belongs fixed on the jacket (task 5's dock is a candidate), not on Esmeralda."}
#: The hangar after the re-plan: the boat bay's centre and two side bays.
HANGAR_BAYS = {"port": {"gannet": 4, "plus": "the payload workshop (EO/IR and cargo pods), the payload loader's track"},
               "starboard": {"fulmar": 2, "plus": "the battery room with the chargers"}}
#: The robots: Zarco's kept (swap robot, deck tug, cleaner; the crate loader as the payload loader), its harvest robots
#: removed (davits, basket gantry, packing cell), the boat ramp added.
ROBOTS = {k: dict(v) for k, v in zarco.ROBOTS.items() if k in ("crate_loader", "swap_robot", "deck_tug", "cleaner")}
ROBOTS["crate_loader"] |= {"what": "the payload loader: an arm on a rail along the flight deck and down the lift - an EO/IR pod or a cargo crate into a Gannet's hook or a Fulmar's bay"}
ROBOTS["boat_ramp"] = {"count": 1, "what": "the stern ramp's capture hook, haul-in winch and cradle lashings; the refuelling coupling and shore power for Barracuda",
                       "reach_m": 11.0, "payload_kg": 6400.0, "cycle_s": 240.0, "power_kW": 22.0, "faults_per_day": 0.1, "fix_min": 15.0, "price_eur": 260000.0}


def eoir_pods(p) -> dict:
    """The drones' surveillance pods (``labs/aero/designs/eoir_payload.py``): one per drone aboard."""
    import os
    import sys
    aero = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "aero", "designs")
    if aero not in sys.path:
        sys.path.insert(0, aero)
    import eoir_payload as ep
    from fulmar import Fulmar
    from gannet import Gannet
    g, f = ep.pod_totals("gannet"), ep.pod_totals("fulmar")
    return {"endurance_min": {"gannet": Gannet.eoir_sizing()["endurance_loiter_min"], "fulmar": Fulmar.eoir_sizing()["endurance_loiter_min"]},
            "count": {"gannet": p["hangar_gannets"], "fulmar": p["hangar_fulmars"]},
            "mass_kg": p["hangar_gannets"] * g["mass_kg"] + p["hangar_fulmars"] * f["mass_kg"],
            "price_eur": p["hangar_gannets"] * g["price_eur"] + p["hangar_fulmars"] * f["price_eur"]}


class Esmeralda(Zarco):
    parameters = [q for q in Zarco.parameters if q.name not in ("part", "davits", "hangar_gannets", "hangar_fulmars", "gannet_packs", "fulmar_packs")] + [
        Parameter("part", "vessel", choices=("vessel", "hulls", "deck", "house", "flight_deck", "pads", "catapult", "net", "mast", "robots",
                                             "ramp", "boat", "effectors", "turret")),
        Parameter("hangar_gannets", 4, "", description="VTOL quads with the EO/IR payload"),
        Parameter("hangar_fulmars", 2, ""),
        Parameter("gannet_packs", 12, ""),
        Parameter("fulmar_packs", 6, ""),
        Parameter("bay_width", 3600.0, "mm", description="the boat bay between the hangar's side bays (the ramp 3.4 m + clearance)"),
        Parameter("bay_length", 14000.0, "mm", description="from the transom forward, into the hangar's centre"),
        Parameter("ramp_length", 13500.0, "mm", description="longer than Barracuda's minimum (its length + 2 m) so the stowed boat's transom keel sits clear of the water"),
        Parameter("bay_structure_t", 3.0, "t", description="the box girder that bridges the cross deck's slot (assumed)"),
        Parameter("turret_z", 9.0, "m", description="the turret above the waterline (on the mast)"),
        Parameter("radar_z", 8.5, "m"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Esmeralda.parameters} | (over or {})

    # ----------------------------------------------------------------------------- weights
    @classmethod
    def weights(cls, over=None):
        """Esmeralda's weights with their centres (t, x, z): Zarco's less what is removed, plus the bay, the ramp,
        the boat with its fuel, the effectors, the turret, the guard drones."""
        p = cls.p(over)
        w = {k: (m, *ZARCO_CENTRES[k]) for k, m in {**zarco.LIGHTSHIP, **zarco.DEADWEIGHT}.items() if k not in REMOVED}
        m, x, z = w["house + hangar + flight deck"]
        x_new = (m * x - HOUSE_PROCESSING_SHARE_T * 3.5) / (m - HOUSE_PROCESSING_SHARE_T)
        z_new = (m * z - HOUSE_PROCESSING_SHARE_T * 3.4) / (m - HOUSE_PROCESSING_SHARE_T)
        w["house + hangar + flight deck"] = (m - HOUSE_PROCESSING_SHARE_T, x_new, z_new)
        m, x, z = w["fresh water + ice"]
        w["fresh water + ice"] = (1.5, x, z)                                       # no ice: fresh water only
        bs = Barracuda.sizing()
        bay_x0 = -p["length"] / 2000 * 0.92 - 0.96                                   # the transom: the deck's aft edge
        ramp = bs["carrier"]
        ta = math.tan(math.radians(ramp["ramp_deg"]))
        surface = lambda x: RAMP_FOOT_Z + (x - bay_x0) * ta                     # noqa: E731  the ramp's top surface
        rl = p["ramp_length"] / 1000
        boat_x = bay_x0 + rl * math.cos(math.radians(ramp["ramp_deg"])) - bs["length_overall_m"] / 2 - 0.3
        bow_keel = surface(boat_x + bs["length_overall_m"] / 2)
        pods = eoir_pods(p)
        drones_t = (p["hangar_gannets"] * 11.44 + p["hangar_fulmars"] * 15.2 + p["gannet_packs"] * 4.6 + p["fulmar_packs"] * 6.4
                    + pods["mass_kg"]) / 1000
        w |= {"boat bay box girder (reinforcement)": (p["bay_structure_t"], bay_x0 + p["bay_length"] / 2000, 3.5),
              "stern ramp, cradle, winch, capture hook": (ramp["system_mass_kg"] / 1000 * rl / ramp["ramp_length_m"], bay_x0 + rl / 2, 1.0),
              "Barracuda (full load)": (bs["mass_full_kg"] / 1000, boat_x, surface(boat_x) + 0.9),
              "hailer + searchlights (mast)": (EFFECTORS["hailer"]["mass_t"] + EFFECTORS["searchlights"]["mass_t"], 8.2, 7.6),
              "bow water cannon: pump, motor, sea chest, monitor, piping": (EFFECTORS["water_cannon"]["mass_t"], 8.5, 0.6),
              "long-range EO/IR turret": (TURRET["mass_t"], 8.2, p["turret_z"]),
              "guard drones: Gannets, Fulmars, packs, EO/IR payloads": (drones_t, -3.0, 3.5)}
        return w, {"bay_x0_m": round(bay_x0, 2), "boat_x_m": round(boat_x, 2), "boat_transom_keel_z_m": round(surface(boat_x - bs["length_overall_m"] / 2), 2),
                   "cradle_keel_z_m": round(bow_keel, 2), "ramp_foot_z_m": RAMP_FOOT_Z}

    @classmethod
    def hydrostatics(cls, weights, over=None):
        """Draught, trim and transverse GM from the weights: two demihulls (waterplane coefficient 0.78, longitudinal
        inertia coefficient 0.055 of B L^3 each; class values). The hull's LCB is where Zarco's loaded LCG is (Zarco
        floats level: an assumption); KB 0.55 T; Zarco's design draught at its 108 t."""
        p = cls.p(over)
        L = p["length"] / 1000 * 0.95
        B = p["hull_beam"] / 1000
        y = (p["beam"] - p["hull_beam"]) / 2000
        aw = 2 * L * B * 0.78
        W = sum(m for m, _, _ in weights.values())
        T = p["draught"] / 1000 - (108.0 - W) / (RHO_SW / 1000 * aw)
        lcg = sum(m * x for m, x, _ in weights.values()) / W
        kg = sum(m * (z + T) for m, _, z in weights.values()) / W                  # above the keel
        vol = W / (RHO_SW / 1000)
        i_l = 2 * 0.055 * B * L ** 3
        i_t = 2 * (L * B * 0.78 * y * y + 0.6 * L * B ** 3 / 12)
        zw = {k: (m, *ZARCO_CENTRES[k]) for k, m in {**zarco.LIGHTSHIP, **zarco.DEADWEIGHT}.items()}
        lcb = sum(m * x for m, x, _ in zw.values()) / sum(m for m, _, _ in zw.values())
        gml = 0.55 * T + i_l / vol - kg
        trim_rad = (lcg - lcb) / gml
        gmt = 0.55 * T + i_t / vol - kg
        return {"mass_t": round(W, 2), "draught_m": round(T, 3), "lcg_m": round(lcg, 2), "lcb_m": round(lcb, 2), "kg_m": round(kg, 2),
                "gm_t_m": round(gmt, 2), "gm_l_m": round(gml, 1), "trim_deg": round(math.degrees(trim_rad), 2),
                "trim_m": round(L * trim_rad, 2), "trim_by": "bow" if trim_rad > 0 else "stern",
                "fuel_to_level_t_m": round(W * (lcg - lcb), 1)}

    @classmethod
    def sizing(cls, over=None):
        p = cls.p(over)
        z = Zarco.sizing()
        w, geo = cls.weights(over)
        hz = cls.hydrostatics({k: (m, *ZARCO_CENTRES[k]) for k, m in {**zarco.LIGHTSHIP, **zarco.DEADWEIGHT}.items()})
        he = cls.hydrostatics(w, over)
        fuel_t = zarco.DEADWEIGHT["fuel (diesel, 10 days)"]
        level_shift = he["fuel_to_level_t_m"] / fuel_t                                  # the fuel's centre moved this far fixes it
        bs = Barracuda.sizing()
        ramp = dict(bs["carrier"]) | {"ramp_length_m": p["ramp_length"] / 1000,
                                       "system_mass_kg": round(bs["carrier"]["system_mass_kg"] * p["ramp_length"] / 1000 / bs["carrier"]["ramp_length_m"]),
                                       "boat_transom_keel_z_m": geo["boat_transom_keel_z_m"], "boat_bow_keel_z_m": geo["cradle_keel_z_m"]}
        # electrical loads on station (kW): Zarco's DP, autonomy and hotel; the guard's sensors, the drones' chargers,
        # the ramp; and the peak with the water cannon, the lights and the hailer on.
        g_end, f_end = eoir_pods(p)["endurance_min"]["gannet"], eoir_pods(p)["endurance_min"]["fulmar"]
        g_chargers = math.ceil(60 / g_end * 1.1) + 1                                   # packs a drone on station uses an hour x 1.1 h charge, + 1
        f_chargers = math.ceil(60 / f_end * 1.1) + 1
        chargers_kW = (g_chargers * 1.15 + f_chargers * 1.6) / 0.92
        loads = {"DP (two pods, a tunnel, 0.5 m/s tide, 8 m/s wind)": zarco.LOADS["DP (two pods, a tunnel, 0.5 m/s tide, 8 m/s wind)"],
                 f"drone chargers ({g_chargers} Gannet + {f_chargers} Fulmar at 1C)": round(chargers_kW, 2),
                 "autonomy: radars, lidar, cameras, compute, comms": zarco.LOADS["autonomy: radars, lidar, cameras, compute, comms"],
                 "long-range EO/IR turret": TURRET["power_kW"], "boat ramp, refuelling, Barracuda's shore power (average)": 2.5,
                 "effectors on standby": 0.4, "hotel": zarco.LOADS["hotel"]}
        cannon = sensing.water_cannon(EFFECTORS["water_cannon"]["flow_l_min"], EFFECTORS["water_cannon"]["pressure_bar"], pump_eff=0.75)
        cannon_kW = cannon["shaft_kW"] / 0.95                                           # the motor
        peak = sum(loads.values()) + cannon_kW + EFFECTORS["searchlights"]["power_kW"] * 2 + EFFECTORS["hailer"]["power_kW"] + ramp["power_kW"]
        load = sum(loads.values())
        fuel_l_h = (load / 0.85) * 0.25 / 0.84
        days = fuel_t * 1000 / 0.84 / fuel_l_h / 24
        # sensors
        sk, fb = sensing.TARGETS["skiff_6m"], sensing.TARGETS["fishing_20m"]
        turret = TURRET | {"z_m": p["turret_z"], "dri_skiff_km": sensing.eoir_dri(TURRET, sk, p["turret_z"], TURRET["jitter_urad"] * 1e-6),
                           "dri_fishing_km": sensing.eoir_dri(TURRET, fb, p["turret_z"], TURRET["jitter_urad"] * 1e-6)}
        ring = {"what": "Zarco's ring of day 4K and uncooled LWIR 640 cameras, 60 deg each (kept)",
                "dri_skiff_km": sensing.eoir_dri(RING, sk, p["radar_z"] - 1.0, 20e-6)}
        radars = {k: v | {"antenna_m": p["radar_z"],
                          "skiff_km_by_hs": {hs: round(sensing.radar_range(v["ref_range_km"], v["ref_rcs_m2"], sk["rcs_m2"], p["radar_z"], sk["height_m"], hs), 1) for hs in (0.5, 1.0, 2.0, 2.5)},
                          "fishing_km_by_hs": {hs: round(sensing.radar_range(v["ref_range_km"], v["ref_rcs_m2"], fb["rcs_m2"], p["radar_z"], fb["height_m"], hs), 1) for hs in (0.5, 1.0, 2.0, 2.5)}}
                  for k, v in RADARS.items()}
        sensors = {"turret": turret, "camera_ring": ring, "radars": radars,
                   "ais_km": {"class_a": round(sensing.horizon_km(p["radar_z"] + 1, 15.0, "radar"), 1),
                              "class_b": round(min(sensing.horizon_km(p["radar_z"] + 1, 3.0, "radar"), 15.0), 1)},
                   "lidar_m": {"skiff": 200, "swimmer": 60, "note": "the 360 deg lidar ring's class ranges (assumed)"},
                   "sonar": SONAR_OPTION}
        effectors = {"hailer": EFFECTORS["hailer"] | sensing.hailer(EFFECTORS["hailer"]["spl_1m_db"]),
                     "searchlights": EFFECTORS["searchlights"] | sensing.searchlight(EFFECTORS["searchlights"]["lumens"], EFFECTORS["searchlights"]["beam_deg"])
                     | {"haze_range_m": sensing.searchlight(EFFECTORS["searchlights"]["lumens"], EFFECTORS["searchlights"]["beam_deg"], 5.0)["range_m"]},
                     "water_cannon": EFFECTORS["water_cannon"] | cannon | {"motor_kW": round(cannon_kW, 1)},
                     "flare_launcher": FLARE_LAUNCHER, "vhf_hail": "DSC and voice on ch 16, AIS safety messages (Zarco's radios)"}
        # deck and hangar
        pads = p["pads"]
        pad_rate = pads * 3600.0 / p["pad_cycle_s"]
        bay = {"bay_width_m": p["bay_width"] / 1000, "bay_length_m": p["bay_length"] / 1000, "tunnel_width_m": round((p["beam"] - 2 * p["hull_beam"]) / 1000, 2),
               "ramp_fits_tunnel": ramp["ramp_width_m"] <= (p["beam"] - 2 * p["hull_beam"]) / 1000 and ramp["ramp_width_m"] <= p["bay_width"] / 1000,
               "clear_height_over_cradle_m": round(cls._z_deck(cls, p) / 1000 + 3.6 - geo["cradle_keel_z_m"], 2),
               "boat_needs_m": ramp["clear_height_m"], "boat_mast_up_top_m": round(geo["cradle_keel_z_m"] + bs["draught_m"] + bs["air_draught_mast_up_m"], 2),
               "flight_deck_underside_m": round(cls._z_deck(cls, p) / 1000 + 3.6, 2), "boat_mast": "folded in the bay: mast up it would hit the flight deck"}
        bay["boat_fits"] = bay["ramp_fits_tunnel"] and bay["clear_height_over_cradle_m"] >= ramp["clear_height_m"]
        side = ((p["beam"] - 800) / 1000 - p["bay_width"] / 1000 - 0.6) / 2               # the hangar's 9.2 m less the bay and its walls
        hangar = {"side_bay_width_m": round(side, 2), "length_m": 12.0, "bays": HANGAR_BAYS,
                  "gannet_fits": 1.79 + 0.3 <= side and p["hangar_gannets"] * (1.79 + 0.3) <= 12.0 - 3.0,
                  "fulmar_fits": 2.1 + 0.3 <= side and p["hangar_fulmars"] * (3.6 + 0.4) <= 12.0 - 3.0,
                  "fulmar_stowage": f"wing along the bay, fuselage athwartships (2.1 m in a {side:.1f} m bay)"}
        price = dict(z["price"])
        robots_eur = sum(r["count"] * r["price_eur"] for r in ROBOTS.values())
        price["robots aboard"] = robots_eur
        for k in ("processing line + chiller", "4 tether reels"):
            price.pop(k)
        price.pop("drones: 8 Gannet, 3 Fulmar, packs")
        price |= {"drones: 4 Gannet, 2 Fulmar, packs": p["hangar_gannets"] * 9800 + p["hangar_fulmars"] * 21000 + p["gannet_packs"] * 450 + p["fulmar_packs"] * 620,
                  "EO/IR payloads (one per drone)": eoir_pods(p)["price_eur"], "boat bay (cut and box girder)": 180000.0,
                  "long-range EO/IR turret": TURRET["price_eur"], **{f"effector: {k}": v["price_eur"] for k, v in EFFECTORS.items()}}
        price_eur = round(sum(price.values()))
        return {"length_m": z["length_m"], "beam_m": z["beam_m"], "displacement_t": z["displacement_t"],
                "mass_t": he["mass_t"], "draught_m": he["draught_m"], "trim": he, "zarco_trim": hz,
                "level_by_moving_fuel_m": round(-level_shift, 2), "weights": {k: [round(m, 3), round(x, 2), round(zz, 2)] for k, (m, x, zz) in w.items()},
                "layout": geo, "service_kn": z["service_kn"], "service_kW": z["service_kW"], "max_kn": z["max_kn"],
                "station_load_kW": round(load, 1), "loads_kW": loads, "peak_kW": round(peak, 1), "gensets_kW": p["gensets_kw"],
                "gensets_margin_at_peak": round(p["gensets_kw"] / peak - 1, 2), "fuel_l_h_on_station": round(fuel_l_h, 1),
                "endurance_days": round(days, 1), "chargers": g_chargers + f_chargers, "charging_kW": round(chargers_kW, 2),
                "vtol_launches_per_h": round(pads * 3600 / 60.0, 1), "vtol_recoveries_per_h": round(pad_rate, 1),
                "fixed_wing_launches_per_h": round(3600 / p["catapult_cycle_s"], 1), "fixed_wing_recoveries_per_h": round(3600 / p["net_cycle_s"], 1),
                "boat": {"name": "barracuda", "mass_kg": bs["mass_full_kg"], "launch_s": ramp["launch_s"], "recovery_s": ramp["recovery_s"],
                         "hs_limit_m": ramp["hs_limit_m"], "ramp": ramp, "refuel_from_ship": True},
                "bay": bay, "hangar": hangar, "deck_hs_limit_m": p["deck_hs_limit"], "sensors": sensors, "effectors": effectors,
                "robots": ROBOTS, "crew": 0, "operators_ashore": z["operators_ashore"],
                "price": price, "price_eur": price_eur, "price_with_barracuda_eur": price_eur + bs["price_eur"],
                "changed_vs_zarco": {"mass_t": [round(hz["mass_t"], 2), he["mass_t"]], "draught_m": [hz["draught_m"], he["draught_m"]],
                                     "trim_deg": [hz["trim_deg"], he["trim_deg"]], "gm_t_m": [hz["gm_t_m"], he["gm_t_m"]],
                                     "station_load_kW": [z["bed_load_kW"], round(load, 1)], "endurance_days": [z["endurance_days"], round(days, 1)],
                                     "hangar": [{"gannet": 8, "fulmar": 3}, {"gannet": p["hangar_gannets"], "fulmar": p["hangar_fulmars"]}],
                                     "chargers": [12, g_chargers + f_chargers], "davits": [4, 0], "price_eur": [z["price_eur"], price_eur],
                                     "unchanged": "hulls, propulsion (service 10 kn, max 14.6 kn), autonomy mast, flight deck (4 pads, catapult, net), deck closure Hs 2.0 m"}}

    # ----------------------------------------------------------------------------- CAD
    def _bay_box(self, p):
        x0 = -p["length"] * 0.46 - 960
        z = self._z_deck(p)
        return cq.Workplane("XY").box(p["bay_length"], p["bay_width"], 4400).translate((x0 + p["bay_length"] / 2, 0, z - 400 + 2200 - 600))

    def _deck(self, p):
        return Zarco._deck(self, p).cut(self._bay_box(p))

    def _house(self, p):
        """The hangar only (no processing house), its centre cut for the boat bay: two side bays under the flight deck."""
        z = self._z_deck(p)
        hangar = cq.Workplane("XY").box(12000, 9200, 3600).translate((-3000, 0, z + 1800))
        inner = cq.Workplane("XY").box(12000, p["bay_width"], 3600).translate((-3000, 0, z + 1800 - 300))
        return hangar.cut(inner)

    def _ramp(self, p):
        """The stern ramp at 8 deg from the transom forward, the cradle at its top, the winch, the capture hook."""
        r = Barracuda.sizing()["carrier"]
        x0 = -p["length"] * 0.46 - 960
        a = math.radians(r["ramp_deg"])
        L = p["ramp_length"]
        plate = (cq.Workplane("XY").box(L, r["ramp_width_m"] * 1000, 120).translate((L / 2, 0, -60))
                 .rotate((0, 0, 0), (0, 1, 0), -math.degrees(a)).translate((x0, 0, RAMP_FOOT_Z * 1000)))
        top = cq.Vector(x0 + L * math.cos(a), 0, RAMP_FOOT_Z * 1000 + L * math.sin(a))
        winch = cq.Workplane("XY").add(cq.Solid.makeCylinder(350, 900, top + cq.Vector(300, -450, 300), cq.Vector(0, 1, 0)))
        hook = cq.Workplane("XY").box(400, 300, 600).translate((top.x - 200, 0, top.z + 500))
        guides = None
        for s in (1, -1):
            g = (cq.Workplane("XY").box(L * 0.6, 150, 300).translate((L * 0.7, s * 600, 150))
                 .rotate((0, 0, 0), (0, 1, 0), -math.degrees(a)).translate((x0, 0, RAMP_FOOT_Z * 1000)))
            guides = g if guides is None else guides.union(g)
        return plate.union(winch).union(hook).union(guides)

    def _boat(self, p):
        """Barracuda on the cradle, its mast folded (drawn without the mast), keel on the ramp's top."""
        b = Barracuda()
        bp = Barracuda.p()
        _, geo = self.weights(p)
        hull = b.build(bp | {"part": "hull"})
        out = hull
        for k in ("collar", "jets", "console", "cannon", "effectors"):
            out = out.union(b.build(bp | {"part": k}))
        a = Barracuda.sizing()["carrier"]["ramp_deg"]
        half = (bp["length"] + bp["collar_d"] * 1.2) / 2
        return out.rotate((0, 0, 0), (0, 1, 0), -a).translate((geo["boat_x_m"] * 1000 - half, 0, geo["boat_transom_keel_z_m"] * 1000))

    def _effectors(self, p):
        """On the mast: the hailer's panel and the two searchlights on wings; on the bow: the monitor on its pedestal."""
        z = self._z_deck(p)
        hailer = cq.Workplane("XY").box(300, 1200, 900).translate((8700, 0, z + 5300))
        lights = cq.Workplane("XY").newObject([cq.Compound.makeCompound(
            [cq.Solid.makeCylinder(250, 500, cq.Vector(8300, y, z + 4600), cq.Vector(1, 0, 0)) for y in (1500, -1500)]
            + [cq.Solid.makeBox(100, 1300, 100, cq.Vector(8200, -1500, z + 4500))])])
        ped = cq.Workplane("XY").add(cq.Solid.makeCylinder(200, 1500, cq.Vector(10300, 0, z)))
        mon = cq.Workplane("XY").add(cq.Solid.makeCylinder(90, 1400, cq.Vector(10300, 0, z + 1600),
                                                           cq.Vector(math.cos(math.radians(15)), 0, math.sin(math.radians(15)))))
        return hailer.union(lights).union(ped).union(mon)

    def _turret(self, p):
        z = p["turret_z"] * 1000
        return (cq.Workplane("XY").add(cq.Solid.makeSphere(330, cq.Vector(8700, 0, z)))
                .union(cq.Workplane("XY").box(600, 200, 300).translate((8500, 0, z - 400))))

    def _robots(self, p):
        """Zarco's deck robots without the basket gantry: the payload loader's rail, the swap gantry, the tug, the cleaner."""
        zf = self._z_deck(p) + 3600 + 150
        s = [cq.Solid.makeBox(12000, 200, 150, cq.Vector(-9000, -4600, zf)), cq.Solid.makeBox(500, 400, 500, cq.Vector(-4500, -4700, zf + 150)),
             cq.Solid.makeCylinder(90, 1400, cq.Vector(-4250, -4500, zf + 650), cq.Vector(0.3, 0.6, 0.75))]
        for x in (-8000, -800):
            for y in (4500, -4300):
                s.append(cq.Solid.makeBox(160, 160, 2400, cq.Vector(x - 80, y - 80, zf)))
        s += [cq.Solid.makeBox(7400, 200, 200, cq.Vector(-8000, 4400, zf + 2400)), cq.Solid.makeBox(7400, 200, 200, cq.Vector(-8000, -4400, zf + 2400)),
              cq.Solid.makeBox(200, 9000, 250, cq.Vector(-4500, -4500, zf + 2400)), cq.Solid.makeBox(1200, 800, 400, cq.Vector(-1000, 3200, zf)),
              cq.Solid.makeBox(600, 500, 200, cq.Vector(2500, -2000, zf))]
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(s)])

    def build(self, p):
        if cq is None:
            raise ImportError("Esmeralda's CAD needs CadQuery")
        parts = {"robots": self._robots, "hulls": self._hulls, "deck": self._deck, "house": self._house, "flight_deck": self._flight_deck,
                 "pads": self._pads, "catapult": self._catapult, "net": self._net, "mast": self._mast, "ramp": self._ramp,
                 "boat": self._boat, "effectors": self._effectors, "turret": self._turret}
        if p["part"] in parts:
            return parts[p["part"]](p)
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound([f(p).val() for f in parts.values()])])


if __name__ == "__main__":
    import json
    s = Esmeralda.sizing()
    print(json.dumps({k: v for k, v in s.items() if k not in ("weights", "price", "robots")}, indent=1, default=str))
