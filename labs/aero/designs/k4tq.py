"""K4TQ — compact convertible tilt-wing medical transport UAV (plan: ``notebooks/k4_plan.md``, ``notebooks/k4_work.md``).

Four tandem wings, each turning 90° about its own spanwise spar (the tilt) and folding back on a separate vertical
hinge (the fold), one propeller near each wingtip, a level fuselage with an insulated medical cartridge at the CG.
Cruise: chords horizontal, propellers forward. Hover: chords vertical, leading edges up, discs horizontal.

This module is the single source of truth every stage reads (sizing, CAD, CFD, FEA, BOM):
- ``REQUIREMENTS``: R1–R12 as agreed;
- ``CATALOGUE``: the bought parts with mass, price (EUR, estimate) and source — every entry "to verify" until Stage 4;
- ``VARIANTS`` / ``params``: configurations A (equal tandem, the baseline), B (larger front wing), C (compact);
- ``layout``: stations, axes, clearances, the propeller pose at any tilt and fold angle;
- ``mass_table``: the itemised budget (``og.mass``), the battery placed for the CG target.

Axes: x aft from the nose tip, y to the right, z up; metres, radians (``_deg`` / ``_mm`` say otherwise).
Tilt 0 = cruise, 90° = hover (leading edge up). Fold 0 = deployed, + = the wing swung aft about the vertical hinge.
"""
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from ogma import mass as og_mass
from ogma.units import G, RHO0 as RHO  # noqa: F401  (RHO re-exported for the sizing)

# ================================================================================================= requirements
STANDARD_MARKING = "BE CAREFUL – FRAGILE – TEMPERATURE SENSITIVE LOAD"
REQUIREMENTS = {
    "R1 geometry": {"fuselage_length_m": 1.0, "span_m": 1.4, "front_wing_station": 0.30, "rear_wing_station": 0.75},
    "R2 payload": {"standard_kg": 1.0, "extended_min_kg": 0.5, "extended_max_kg": 0.75},
    "R3 mission": {"radius_km": 10.0, "hover_total_s": 120.0, "reserve_fraction": 0.20, "wind_m_s": 8.0, "cruise_altitude_m": 120.0},
    "R4 modes": "cruise: chords horizontal, propellers forward; hover: chords vertical, discs horizontal; fuselage level",
    "R5 transport": "wings fold to pack into a tube; partial fold accepted",
    "R6 launch": ["hand release (hover stance)", "from the transport tube (the case as the pad)", "hospital roof"],
    "R7 temperature logging": "mandatory: two independent sensors in the cartridge, a continuous timestamped log on board "
                              "that survives a power loss, telemetry stream, alarm thresholds per load type",
    "R8 marking": STANDARD_MARKING + " (text only: no symbol, no red cross)",
    "R9 directional stability": "a vertical fin is allowed where the analysis needs one",
    "R10 transition": "checked with CFD (cases prepared, run manually)",
    "R11 propulsion": "exactly four motors and four propellers; parts from Poland / the EU preferred",
    "R12 phases": "everything mocked in the early phases (DEBUG = True)",
}
MISSION = REQUIREMENTS["R3 mission"]

# ================================================================================================= the bought parts
# Masses and prices are datasheet / retailer values, prices EUR incl. VAT at EU retailers (estimates). Every row is
# "to verify" until the Stage 4 BOM fixes the supplier. Sources: see each "source".
CATALOGUE = {
    "motor": {
        "MN4014-400": {"name": "T-Motor MN4014 KV400 (Navigator)", "kv": 400.0, "R_ohm": 0.067, "I0_A": 1.0, "I_max_A": 30.0, "mass_kg": 0.150,
                       "price_eur": 95.0, "source": "retailer bench data: 67 mOhm, 1.3 A at 10 V no-load, 15x5 CF 26 N at 18.7 A on 6S; mass with cable to verify"},
        "X4110S-400": {"name": "SunnySky X4110S KV400", "kv": 400.0, "R_ohm": 0.112, "I0_A": 0.9, "I_max_A": 26.0, "mass_kg": 0.149,
                       "price_eur": 70.0, "source": "retailer listings: 112 mOhm, 149 g, 26 A / 30 s, 6S"},
        "MN5006-450": {"name": "T-Motor MN5006 KV450 (Antigravity)", "kv": 450.0, "R_ohm": 0.060, "I0_A": 1.2, "I_max_A": 30.0, "mass_kg": 0.106,
                       "price_eur": 110.0, "cells_max": 6, "off_the_shelf": True,
                       "source": "retailer listings: 60 mOhm, 106 g with cable, 4-6S, P14-18; I0 and I_max estimated (to verify)"},
        "MN501S-300": {"name": "T-Motor MN501-S KV300 (Navigator)", "kv": 300.0, "R_ohm": 0.063, "I0_A": 1.0, "I_max_A": 40.0, "mass_kg": 0.170,
                       "price_eur": 150.0, "cells_max": 8, "off_the_shelf": True,
                       "source": "retailer listings: 63 mOhm, 170 g, 40 A / 180 s, 1000 W, 6-8S; I0 estimated (to verify)"},
        "MN505S-380": {"name": "T-Motor MN505-S KV380 (Navigator)", "kv": 380.0, "R_ohm": 0.028, "I0_A": 2.0, "I_max_A": 50.0, "mass_kg": 0.225,
                       "price_eur": 180.0, "cells_max": 6, "off_the_shelf": True,
                       "source": "RobotShop / T-Motor store: 28 mOhm, 2.0 A idle at 15 V, 50 A / 180 s, 1200 W, 6S, 225 g, 17–18\" props"},
        # K4TQ-B "Bertha" (the scaled design, 12S): the large multirotor / VTOL class — every value "to verify" on the datasheet
        "U8L-150": {"name": "T-Motor U8 Lite KV150", "kv": 150.0, "R_ohm": 0.115, "I0_A": 0.7, "I_max_A": 30.0, "mass_kg": 0.244, "price_eur": 270.0,
                    "source": "T-Motor U8 Lite class: 12S, 26–28\" props, ~244 g; R and I0 estimated (to verify)"},
        "MN801S-150": {"name": "T-Motor MN801-S KV150 (Navigator)", "kv": 150.0, "R_ohm": 0.050, "I0_A": 1.4, "I_max_A": 60.0, "mass_kg": 0.480, "price_eur": 330.0,
                       "cells_max": 12, "off_the_shelf": True,
                       "source": "T-Motor store / LIGPower datasheet: 6–12S, 50 mΩ, idle 1.4 A at 15 V, 60 A / 2800 W (180 s), 480 g with cable, 26–27\" props; price to verify"},
        "U10II-100": {"name": "T-Motor U10 II KV100", "kv": 100.0, "R_ohm": 0.080, "I0_A": 0.9, "I_max_A": 45.0, "mass_kg": 0.410, "price_eur": 400.0,
                      "source": "T-Motor U10 II class: 12S, 28–30\" props, ~410 g; R and I0 estimated (to verify)"},
        "MN601S-180": {"name": "T-Motor MN601-S KV180 (Navigator)", "kv": 180.0, "R_ohm": 0.105, "I0_A": 0.7, "I_max_A": 30.0, "mass_kg": 0.200, "price_eur": 220.0,
                       "source": "T-Motor MN601-S class: 12S, 20–22\" props, ~200 g; R and I0 estimated (to verify)"},
        "MN701S-135": {"name": "T-Motor MN701-S KV135 (Navigator)", "kv": 135.0, "R_ohm": 0.095, "I0_A": 0.7, "I_max_A": 35.0, "mass_kg": 0.265, "price_eur": 260.0,
                       "source": "T-Motor MN701-S class: 12S, 22–26\" props, ~265 g; R and I0 estimated (to verify)"},
        "U12II-100": {"name": "T-Motor U12 II KV100", "kv": 100.0, "R_ohm": 0.040, "I0_A": 1.4, "I_max_A": 60.0, "mass_kg": 0.790, "price_eur": 620.0,
                      "source": "T-Motor U12 II class: 12S, 28–32\" props, ~790 g, ~2.5 kW peak; R and I0 estimated (to verify)"},
        "MN1005-90": {"name": "T-Motor MN1005 KV90 (Navigator)", "kv": 90.0, "R_ohm": 0.050, "I0_A": 1.2, "I_max_A": 50.0, "mass_kg": 0.560, "price_eur": 520.0,
                      "source": "T-Motor MN1005 class: 12S, 28–32\" props, ~560 g; R and I0 estimated (to verify)"},
        "U12II-120": {"name": "T-Motor U12 II KV120", "kv": 120.0, "R_ohm": 0.035, "I0_A": 1.6, "I_max_A": 60.0, "mass_kg": 0.790, "price_eur": 620.0,
                      "source": "T-Motor U12 II class (KV120 winding): 12S, 28–30\" props, ~790 g; R and I0 estimated (to verify)"},
        # off the shelf (2026-10-21, T-Motor store datasheets): the 12S Navigator motors for 18–22" propellers
        "MN601S-170": {"name": "T-Motor MN601-S KV170 (Navigator)", "kv": 170.0, "R_ohm": 0.115, "I0_A": 0.8, "I_max_A": 30.0, "mass_kg": 0.250, "price_eur": 200.0,
                       "cells_max": 12, "off_the_shelf": True,
                       "source": "T-Motor store: 6–12S, 115 mΩ, 30 A / 1500 W (180 s), 250 g with cable, 20–21\" props; I0 and price to verify"},
        "MN605S-170": {"name": "T-Motor MN605-S KV170 (Navigator)", "kv": 170.0, "R_ohm": 0.065, "I0_A": 1.0, "I_max_A": 35.0, "mass_kg": 0.350, "price_eur": 250.0,
                       "cells_max": 12, "off_the_shelf": True,
                       "source": "T-Motor store: 6–12S, 65 mΩ, 35 A / 1700 W (180 s), 350 g with cable (a reseller: 327 g), 20–22\" props, 8.6 kg max thrust; I0 and price to verify"},
        "U8II-170": {"name": "T-Motor U8 II KV170 (class)", "kv": 170.0, "R_ohm": 0.085, "I0_A": 0.8, "I_max_A": 35.0, "mass_kg": 0.250, "price_eur": 280.0,
                     "off_the_shelf": False,
                     "source": "an assumed U8 II class — no KV170 winding is listed (the U8 II comes as KV85/100/150/190); the real near-equivalent is the MN601-S KV170"},
    },
    "propeller": {   # diameter, pitch [in]; folding carbon-reinforced blades with a hub (mass per propeller incl. hub)
        "13x6.5": {"d_in": 13, "p_in": 6.5, "mass_kg": 0.032, "price_eur": 18.0, "source": "T-Motor T13x6.5 class (23.7 g fixed); folding hub +8 g"},
        "13x8": {"d_in": 13, "p_in": 8.0, "mass_kg": 0.035, "price_eur": 20.0, "source": "APC 13x8E / folding 13x8 class; folding hub included"},
        "14x7": {"d_in": 14, "p_in": 7.0, "mass_kg": 0.038, "price_eur": 22.0, "source": "folding 14x7 class"},
        "15x5": {"d_in": 15, "p_in": 5.0, "mass_kg": 0.040, "price_eur": 25.0, "source": "T-Motor P15x5 CF (multirotor)"},
        "15x8": {"d_in": 15, "p_in": 8.0, "mass_kg": 0.045, "price_eur": 26.0, "source": "folding 15x8 class"},
        "14x8.5": {"d_in": 14, "p_in": 8.5, "mass_kg": 0.040, "price_eur": 22.0, "source": "APC 14x8.5E class, folding hub included"},
        "15x10": {"d_in": 15, "p_in": 10.0, "mass_kg": 0.048, "price_eur": 26.0, "source": "APC 15x10E class, folding hub included"},
        "16x10": {"d_in": 16, "p_in": 10.0, "mass_kg": 0.055, "price_eur": 30.0, "source": "APC 16x10E class, folding hub included"},
        "17x10": {"d_in": 17, "p_in": 10.0, "mass_kg": 0.062, "price_eur": 34.0, "source": "APC 17x10E class, folding hub included"},
        # K4TQ-F3 / G3 (the fast designs): higher pitch for the pitch speed the 34.5 m/s top speed needs — "to verify"
        "16x12": {"d_in": 16, "p_in": 12.0, "mass_kg": 0.058, "price_eur": 32.0, "source": "APC 16x12E class, folding hub included"},
        "16x13": {"d_in": 16, "p_in": 13.0, "mass_kg": 0.060, "price_eur": 34.0, "source": "folding 16x13 carbon class (Mejzlik / Aeronaut CAM folding class), to verify"},
        "17x12": {"d_in": 17, "p_in": 12.0, "mass_kg": 0.065, "price_eur": 36.0, "source": "APC 17x12E class, folding hub included"},
        "20x13": {"d_in": 20, "p_in": 13.0, "mass_kg": 0.090, "price_eur": 95.0, "source": "Mejzlik 20x13 carbon class + folding hub, to verify"},
        "20x14": {"d_in": 20, "p_in": 14.0, "mass_kg": 0.092, "price_eur": 98.0, "source": "Mejzlik 20x14 carbon class + folding hub, to verify"},
        "22x14": {"d_in": 22, "p_in": 14.0, "mass_kg": 0.105, "price_eur": 110.0, "source": "Mejzlik 22x14 carbon class + folding hub, to verify"},
        # off the shelf, folding (2026-10-21): Aeronaut CAM Carbon (Z) blades on the 42 mm centre piece (EU: Lindinger, eBay.de —
        # stock varies) and T-Motor's folding VTOL props; mass per propeller with the hub and the spinner (estimates)
        "CAM16x13": {"d_in": 16, "p_in": 13.0, "mass_kg": 0.062, "price_eur": 75.0, "folding": True, "off_the_shelf": True,
                     "source": "Aeronaut CAM Carbon 16x13 folding blades + 42 mm hub + spinner (listed at eBay.de, stock varies)"},
        "CAM17x11": {"d_in": 17, "p_in": 11.0, "mass_kg": 0.066, "price_eur": 80.0, "folding": True, "off_the_shelf": True,
                     "source": "Aeronaut CAM Carbon 17x11 folding blades + 42 mm hub + spinner"},
        "CAM17x13": {"d_in": 17, "p_in": 13.0, "mass_kg": 0.068, "price_eur": 80.0, "folding": True, "off_the_shelf": True,
                     "source": "Aeronaut CAM Carbon 17x13 folding blades + 42 mm hub + spinner"},
        "CAM18x13": {"d_in": 18, "p_in": 13.0, "mass_kg": 0.075, "price_eur": 85.0, "folding": True, "off_the_shelf": True,
                     "source": "Aeronaut CAM Carbon 18x13 folding blades + 42 mm hub + spinner"},
        "CAM18x9": {"d_in": 18, "p_in": 9.0, "mass_kg": 0.070, "price_eur": 85.0, "folding": True, "off_the_shelf": True,
                    "source": "Aeronaut CAM Carbon Z 18x9 folding blades + 42 mm hub + spinner"},
        "MF2009": {"d_in": 20.4, "p_in": 7.3, "mass_kg": 0.075, "price_eur": 60.0, "folding": True, "off_the_shelf": True,
                   "source": "T-Motor MF2009 folding polymer + CF, 20.4x7.3, 59 g + the folding base"},
        "FA22.2": {"d_in": 22.2, "p_in": 7.2, "mass_kg": 0.085, "price_eur": 90.0, "folding": True, "off_the_shelf": True,
                   "source": "T-Motor carbon folding 22.2x7.2, 69 g + the folding base"},
        # off the shelf, rigid (K4TQ-H3 / J3): APC thin electric — the stopped rear pair cannot fold (``folding`` False: its drag ×4)
        "APC20x10E": {"d_in": 20, "p_in": 10.0, "mass_kg": 0.095, "price_eur": 40.0, "folding": False, "off_the_shelf": True,
                      "source": "APC 20x10E (LP20010E), rigid; in stock at Galaxus (EU) 2026-10"},
        "APC22x10E": {"d_in": 22, "p_in": 10.0, "mass_kg": 0.115, "price_eur": 48.0, "folding": False, "off_the_shelf": True,
                      "source": "APC 22x10E (LP22010E), rigid; Gator-RC / Motion RC (stock varies)"},
        "APC22x12E": {"d_in": 22, "p_in": 12.0, "mass_kg": 0.120, "price_eur": 50.0, "folding": False, "off_the_shelf": True,
                      "source": "APC 22x12E (LP22012E), rigid; Gator-RC (stock varies)"},
        "FA24.2": {"d_in": 24.2, "p_in": 7.9, "mass_kg": 0.110, "price_eur": 110.0, "folding": True, "off_the_shelf": True,
                   "source": "T-Motor carbon folding 24.2x7.9 (getfpv.com), + the folding base"},
        "FA29.2": {"d_in": 29.2, "p_in": 9.5, "mass_kg": 0.160, "price_eur": 150.0, "folding": True, "off_the_shelf": True,
                   "source": "T-Motor FA29.2x9.5 carbon folding (mabrobotics.pl, EU), + the folding base"},
        "12x8": {"d_in": 12, "p_in": 8.0, "mass_kg": 0.030, "price_eur": 16.0, "source": "APC 12x8E class, folding hub included"},
        "13x5.5": {"d_in": 13, "p_in": 5.5, "mass_kg": 0.030, "price_eur": 18.0, "source": "multirotor 13x5.5 CF class (quadcopter reference)"},
        "17x5.8": {"d_in": 17, "p_in": 5.8, "mass_kg": 0.050, "price_eur": 35.0, "source": "T-Motor P17x5.8 (the MN4014 KV400's recommended 6S propeller)"},
        # K4TQ-B "Bertha": large carbon propellers on a folding hub (Mejzlik, CZ, makes this class in the EU) — "to verify"
        "20x10": {"d_in": 20, "p_in": 10.0, "mass_kg": 0.085, "price_eur": 90.0, "source": "Mejzlik 20x10 carbon class + folding hub, to verify"},
        "22x11": {"d_in": 22, "p_in": 11.0, "mass_kg": 0.100, "price_eur": 105.0, "source": "Mejzlik 22x11 carbon class + folding hub, to verify"},
        "24x12": {"d_in": 24, "p_in": 12.0, "mass_kg": 0.115, "price_eur": 120.0, "source": "Mejzlik 24x12 carbon class + folding hub (~35 g), to verify"},
        "26x13": {"d_in": 26, "p_in": 13.0, "mass_kg": 0.135, "price_eur": 140.0, "source": "Mejzlik 26x13 carbon class + folding hub, to verify"},
        "27x14": {"d_in": 27, "p_in": 14.0, "mass_kg": 0.145, "price_eur": 150.0, "source": "Mejzlik 27x14 carbon class + folding hub, to verify"},
        "28x14": {"d_in": 28, "p_in": 14.0, "mass_kg": 0.155, "price_eur": 160.0, "source": "Mejzlik 28x14 carbon class + folding hub, to verify"},
        "24x15": {"d_in": 24, "p_in": 15.0, "mass_kg": 0.120, "price_eur": 125.0, "source": "Mejzlik 24x15 carbon class (cruise pitch, p/D 0.63 as K4TQ's 16x10) + folding hub, to verify"},
        "26x16": {"d_in": 26, "p_in": 16.0, "mass_kg": 0.140, "price_eur": 145.0, "source": "Mejzlik 26x16 carbon class + folding hub, to verify"},
        "28x17": {"d_in": 28, "p_in": 17.0, "mass_kg": 0.160, "price_eur": 165.0, "source": "Mejzlik 28x17 carbon class + folding hub, to verify"},
        "30x18": {"d_in": 30, "p_in": 18.0, "mass_kg": 0.180, "price_eur": 185.0, "source": "Mejzlik 30x18 carbon class + folding hub, to verify"},
        "32x16": {"d_in": 32, "p_in": 16.0, "mass_kg": 0.200, "price_eur": 200.0, "source": "Mejzlik 32x16 carbon class + folding hub, to verify"},
        "30x15": {"d_in": 30, "p_in": 15.0, "mass_kg": 0.175, "price_eur": 180.0, "source": "Mejzlik 30x15 carbon class + folding hub, to verify"},
    },
    "esc": {"name": "40 A 6S BLHeli_32 / AM32 (e.g. T-Motor AIR 40A)", "mass_kg": 0.028, "price_eur": 35.0, "source": "AIR 40A class, in the nacelle"},
    "esc_8s": {"name": "60 A 8S BLHeli_32 / AM32 (e.g. T-Motor F55A PRO II 8S class)", "mass_kg": 0.040, "price_eur": 60.0,
               "source": "8S 60 A class in the nacelle (K4TQ-F3), to verify"},
    "esc_hv": {"name": "60 A 12S FOC ESC (e.g. T-Motor ALPHA 60A HV / Hobbywing XRotor Pro HV)", "mass_kg": 0.075, "price_eur": 150.0,
               "source": "12S 60 A class with the cooling fins in the nacelle's flow (K4TQ-B), to verify"},
    "battery": {   # Li-ion packs of Molicel P45B (og.batteries.CELLS: 4.5 Ah, 70 g per cell); +7 % for the BMS, wrap, leads
        "6S2P-P45B": {"cells_s": 6, "cells_p": 2, "capacity_ah": 9.0, "cell_v": 3.6, "mass_kg": 12 * 0.070 * 1.07, "max_A": 90.0, "price_eur": 160.0,
                      "source": "Molicel P45B (21700, 4.5 Ah, 45 A); pack assembly (spot welded, BMS balance leads)"},
        "6S3P-P45B": {"cells_s": 6, "cells_p": 3, "capacity_ah": 13.5, "cell_v": 3.6, "mass_kg": 18 * 0.070 * 1.07, "max_A": 135.0, "price_eur": 230.0,
                      "source": "the extended-range pack (R2): same cells, 3P"},
        "6S1P-LiPo5000": {"cells_s": 6, "cells_p": 1, "capacity_ah": 5.0, "cell_v": 3.7, "mass_kg": 0.76, "max_A": 125.0, "price_eur": 110.0,
                          "source": "a 6S 5000 mAh 25C LiPo (the comparison: 111 Wh for the mass of 194 Wh Li-ion)"},
    },
    "avionics": {
        "flight controller": {"mass_kg": 0.040, "price_eur": 250.0, "source": "ArduPilot H7 board (Pixhawk 6C mini class), QuadPlane tilt-wing support"},
        "GNSS + compass": {"mass_kg": 0.030, "price_eur": 70.0, "source": "u-blox M10 module"},
        "telemetry (LTE + 868 MHz)": {"mass_kg": 0.045, "price_eur": 150.0, "source": "LTE modem + 868 MHz link (EU band)"},
        "RC receiver": {"mass_kg": 0.008, "price_eur": 40.0, "source": "ELRS 868 MHz"},
        "power module + BEC": {"mass_kg": 0.030, "price_eur": 45.0, "source": "6S power module, 5 V / 7.4 V BECs (servos)"},
        "airspeed sensor": {"mass_kg": 0.010, "price_eur": 45.0, "source": "MS4525 pitot (transition schedule needs airspeed)"},
        "remote ID + nav/status lights": {"mass_kg": 0.020, "price_eur": 60.0, "source": "EU direct remote ID, blue/white LEDs"},
    },
    # K4TQ-A2 — the fully autonomous variant's companion computer and perception (estimates; Jetson power by its nvpmodel mode)
    "autonomy": {
        "companion computer (Jetson Orin NX 16 GB + carrier + heatsink/fan)": {"mass_kg": 0.150, "power_W": 20.0, "price_eur": 950.0,
            "source": "NVIDIA Jetson Orin NX 16 GB module (10 / 15 / 25 W modes, up to 40 W MAXN Super) on a compact carrier with a fan heatsink; "
                      "20 W average assumed (perception + planning); to verify on the bench"},
        "forward stereo depth camera": {"mass_kg": 0.061, "power_W": 3.0, "price_eur": 300.0,
            "source": "global-shutter stereo depth camera (OAK-D Lite / Pro class, ~61 g, 2.5–4.5 W), behind a window in the nose"},
        "downward camera + rangefinder": {"mass_kg": 0.030, "power_W": 2.0, "price_eur": 150.0,
            "source": "optical flow / downward camera + lidar rangefinder for the vertical landing on an unprepared pad"},
        "12 V DC-DC for the companion": {"mass_kg": 0.025, "power_W": 0.0, "price_eur": 40.0, "efficiency": 0.92,
            "source": "6S → 12 V 5 A step-down"},
    },
    "cargo": {
        "insulated cartridge": {"mass_kg": 0.200, "price_eur": 90.0, "source": "VIP/EPP-lined removable shell 240 x 120 x 110 mm, ~1.2 l inner, latch"},
        "shock isolators (4)": {"mass_kg": 0.016, "price_eur": 12.0, "source": "silicone grommet isolators"},
        "temperature sensors (2)": {"mass_kg": 0.006, "price_eur": 30.0, "source": "TMP117 class (±0.1 °C), two independent channels (R7)"},
        "temperature logger": {"mass_kg": 0.020, "price_eur": 60.0, "source": "independent logger with its own cell and flash, GNSS-timestamped, survives power loss (R7)"},
    },
    "root": {   # one wing root: the tilt drive and the fold hinge (Stage 2 designs it; these are the Stage 1 estimates)
        "tilt servo": {"mass_kg": 0.062, "price_eur": 90.0, "torque_Nm": 2.9, "source": "high-torque metal-gear digital servo, 30 kg·cm class (~2.9 N·m), standard 40 x 20 x 38 mm case"},
        "spur pair 16:48 m1.0": {"mass_kg": 0.018, "price_eur": 18.0, "source": "steel pinion on the servo spline + 7075/POM gear on the stub (catalogue spur gears, bored)"},
        "bearings 6804-2RS (2)": {"mass_kg": 0.036, "price_eur": 10.0, "C0_N": 2470.0, "source": "20 x 32 x 7 deep groove, C 4.0 kN, C0 2.47 kN"},
        "stub shaft + bearing block": {"mass_kg": 0.045, "price_eur": 45.0, "source": "7075 stub (Ø20 seats, Ø12 bore for the cables) + 7075 block bolted to the frame"},
        "fold knuckles + pins": {"mass_kg": 0.035, "price_eur": 30.0, "pin_d_m": 0.006, "source": "7075 inner/outer knuckles, Ø6 steel hinge pin, Ø6 hardened locking pin with spring and witness mark"},
        "encoder": {"mass_kg": 0.004, "price_eur": 12.0, "source": "AS5048 magnetic encoder on the stub's inner end"},
        "fasteners + stops": {"mass_kg": 0.010, "price_eur": 5.0, "source": "hard stops −5° / 95°, M3/M4 A2-70, spar socket bond"},
    },
    "root_M": {   # the middle size (K4TQ-G "Ginna", or Bertha where it holds): estimates, to verify
        "tilt servo": {"mass_kg": 0.110, "price_eur": 160.0, "torque_Nm": 6.5,
                       "source": "high-torque digital servo, ~65 kg·cm class (~6.5 N·m), metal gears, 12S-capable BEC rail"},
        "spur pair 16:48 m1.25": {"mass_kg": 0.035, "price_eur": 28.0, "source": "steel pinion + 7075 gear, m1.25"},
        "bearings 6806-2RS (2)": {"mass_kg": 0.060, "price_eur": 14.0, "C0_N": 3600.0, "source": "30 x 42 x 7 deep groove, C0 ~3.6 kN (to verify)"},
        "stub shaft + bearing block": {"mass_kg": 0.100, "price_eur": 65.0, "source": "7075 stub (Ø30 seats, Ø16 bore) + 7075 block"},
        "fold knuckles + pins": {"mass_kg": 0.080, "price_eur": 45.0, "pin_d_m": 0.008, "source": "7075 knuckles, Ø8 hinge pin, Ø8 hardened locking pin"},
        "encoder": {"mass_kg": 0.004, "price_eur": 12.0, "source": "AS5048 magnetic encoder"},
        "fasteners + stops": {"mass_kg": 0.018, "price_eur": 8.0, "source": "hard stops, M4/M5 A4-80"},
    },
    "root_L": {   # one wing root of K4TQ-B "Bertha": the same architecture, the next hardware size up (estimates, to verify)
        "tilt servo": {"mass_kg": 0.180, "price_eur": 250.0, "torque_Nm": 14.0,
                       "source": "industrial brushless servo, ~140 kg·cm class (~14 N·m), CAN/RS485, metal case"},
        "spur pair 16:48 m1.5": {"mass_kg": 0.060, "price_eur": 40.0, "source": "steel pinion + 7075 gear, m1.5"},
        "bearings 6808-2RS (2)": {"mass_kg": 0.090, "price_eur": 20.0, "C0_N": 4500.0, "source": "40 x 52 x 7 deep groove, C0 ~4.5 kN (to verify)"},
        "stub shaft + bearing block": {"mass_kg": 0.190, "price_eur": 90.0, "source": "7075 stub (Ø40 seats, Ø22 bore) + 7075 block"},
        "fold knuckles + pins": {"mass_kg": 0.150, "price_eur": 70.0, "pin_d_m": 0.010, "source": "7075 knuckles, Ø10 hinge pin, Ø10 hardened locking pin"},
        "encoder": {"mass_kg": 0.004, "price_eur": 12.0, "source": "AS5048 magnetic encoder"},
        "fasteners + stops": {"mass_kg": 0.030, "price_eur": 12.0, "source": "hard stops, M5/M6 A4-80"},
    },
    "safety": {
        "recovery parachute": {"mass_kg": 0.160, "price_eur": 250.0, "source": "1 m² canopy + spring ejector (the four-rotor motor-out case; SORA mitigation)"},
    },
}

CATALOGUE["battery"]["6S4P-P45B"] = {"cells_s": 6, "cells_p": 4, "capacity_ah": 18.0, "cell_v": 3.6, "mass_kg": 24 * 0.070 * 1.07, "max_A": 180.0,
                                     "price_eur": 300.0, "source": "Molicel P45B 6S4P: the bigger pack for a bigger drive (K4TQ-F3 option), to verify"}
for _n in range(2, 5):                                           # K4TQ-F3 "Freddy 3": 8S·nP of the same cells
    CATALOGUE["battery"][f"8S{_n}P-P45B"] = {"cells_s": 8, "cells_p": _n, "capacity_ah": 4.5 * _n, "cell_v": 3.6, "mass_kg": 8 * _n * 0.070 * 1.07,
                                             "max_A": 45.0 * _n, "price_eur": 55.0 * _n + 30.0, "source": f"Molicel P45B 8S{_n}P (BMS, wrap, leads), to verify"}
for _n in range(2, 9):                                           # K4TQ-B "Bertha": 12S·nP of the same cells
    CATALOGUE["battery"][f"12S{_n}P-P45B"] = {"cells_s": 12, "cells_p": _n, "capacity_ah": 4.5 * _n, "cell_v": 3.6, "mass_kg": 12 * _n * 0.070 * 1.07,
                                              "max_A": 45.0 * _n, "price_eur": 75.0 * _n + 40.0, "source": f"Molicel P45B 12S{_n}P (BMS, wrap, leads), to verify"}

# areal and linear densities of the made parts (estimates, Stage 2 CAD replaces them)
WING_AREAL_KG_M2 = 0.70           # glass/carbon skins on XPS foam, 12 % thick, both surfaces (without the spar)
FUSELAGE_AREAL_KG_M2 = 0.60       # glass/carbon shell incl. frames' share
FIN_AREAL_KG_M2 = 0.55
NACELLE_KG = 0.022                # printed PA-CF nacelle with the ESC bay
LEG_KG = 0.020
WIRING_KG_M = 0.060               # one 6S power pair (2 × 14 AWG silicone) + a signal lead, per metre
CARBON_TUBE = {"E": 80e9, "G": 4.5e9, "sigma_allow": 400e6, "rho": 1550.0, "source": "pultruded/roll-wrapped, as kestrel_structure.CARBON; G for torsion"}
TUBES_MM = [(12, 10), (14, 12), (16, 14), (18, 16), (20, 18), (22, 20), (25, 22), (28, 25), (30, 27), (32, 29), (35, 32), (40, 36)]
GROWTH_MARGIN = 0.07              # on the empty mass

# ================================================================================================= the configurations
BASE = {
    "variant": "E", "title": "equal tandem, 1.4 m, 1.2 m fuselage",
    # fuselage
    "fuselage_length": 1.20, "fuselage_width": 0.14, "fuselage_height": 0.16,   # 1.2 m: the length sweep (disc gap 159 mm), approved 2026-10-21
    "fuselage_section_n": 5.0,        # superellipse exponent of the mid-body section (Stage 2: room for the bearing blocks near the sides)
    # wings: pivot stations (fraction of the fuselage length), heights (front low, rear high), planform
    "x_pivot_front": 0.30, "x_pivot_rear": 0.75, "z_front": -0.040, "z_rear": 0.040,   # low front, high rear wing: gap 0.08 m (Stage 2: the stubs' bearing blocks inside the fuselage)
    "span": 1.40, "root_chord_front": 0.20, "tip_chord_front": 0.15, "root_chord_rear": 0.20, "tip_chord_rear": 0.15,
    "pivot_chord": 0.30,              # the spar (tilt axis) at this fraction of the local chord, straight across the span
    # the wing root (Stage 2.3/2.4): the tilt drive inside the fuselage (stub shaft in two bearings, servo through 16:48
    # spur gears, encoder), the fold hinge just outside the skin, turning with the wing, its pin along the chord
    "root_gap": 0.012,                # fuselage side to the fold hinge axis
    "root_length": 0.018,             # fold hinge axis to the first wing rib (the outer knuckle; 18 mm: the root plate swings clear of the inner clevis)
    "bearing_spacing": 0.032,         # the two 6804 bearings of the stub, centre to centre
    "stub_od": 0.020,                 # 7075 stub shaft at the bearing seats (6804 bore)
    "gear_module": 1.0, "z_pinion": 16, "z_gear": 48,    # 3:1 spur reduction, the pinion aft of the stub
    "hinge_pin_offset": 0.010,        # the fold pin sits this far toward the upper surface (aft in the hover): the wing folds aft
    "lock_couple": 0.020,             # the locking pin this far from the hinge pin (both along the chord): the couple for the lift moment
    "knuckle_spread": 0.028,          # the outer lugs' centres apart along the chord: the couple for the thrust moment
    "fold_transport_deg": 85.0,       # the packed fold (Stage 2.7: the fold path is clear 0–85°; at 88° the front tips reach the rear roots)
    "section": "NACA 2412", "decalage_deg": 1.5,
    "section_front": None, "section_rear": None,      # one section per wing (None: "section"); K4TQ-F: NACA 4415 / NACA 2415 (k4tq_sections)
    # propulsion
    # Stage 2.8 revision (CAD masses, 6.9 kg): the MN4014 / 6S2P of Stage 1 no longer close the mission or the hover
    # margin — the motor × propeller × pack sweep on the CAD masses (data/k4tq/stage2_motor_sweep.csv) chose these
    "motor": "MN501S-300", "prop": "16x10", "motor_y": 0.62, "prop_ahead_of_le": 0.065,
    "battery": "6S3P-P45B", "payload_kg": 1.0,
    "cad_mass": "auto",               # the CAD's masses (data/k4tq/cad_masses.json, written by k4tq_cad) replace the estimates when current
    # stations along the fuselage (fractions of its length): the cargo at the CG, the battery bay, the parachute
    # Stage 2 packing (k4tq_sizing.pack): the battery at the aft end of the nose bay with the avionics above it, the
    # cartridge where it puts the CG on the target, the parachute right behind the cartridge (its line near the CG)
    "cargo_station": 0.46, "battery_bay": (0.12, 0.21), "parachute_station": 0.68, "avionics_station": 0.21,
    "cargo_box": (0.240, 0.120, 0.110), "battery_box": (0.135, 0.047, 0.074),   # outer sizes [m]: the insulated cartridge, the 6S2P 21700 pack
    # undercarriage, fin, safety
    "leg_bottom_z": -0.225,           # leg feet below the fuselage axis: clears the wing trailing edges (hover stance) and the discs (cruise stance)
    "fin": True, "fin_area": None,    # None: the fin is sized by k4tq_sizing.directional (R9)
    "parachute": True,
    "autonomy": False,                # K4TQ-A2: the companion computer and the perception (CATALOGUE["autonomy"])
    # K4TQ-B "Bertha": the scale of the airframe against K4TQ (``scaled``) and the hardware it takes
    "scale": 1.0, "esc": "esc", "root_set": "root",
    "v_max_design": 30.0,             # m/s: the speed the actuator's aerodynamic torque is sized for (K4TQ-F3 / G3: their top speed + 10 %)
    "motor_rear": None, "prop_rear": None,   # K4TQ-H3 / J3: the rear pair's own drive (hover and transition only; stopped and folded in cruise)
    "v_map_max": None,                # m/s: the drive maps' airspeed range (None: 32, the scaled designs 44)
    "made_calibration": None,         # {item prefix: CAD / estimate} — the made parts' estimates corrected by K4TQ-F1's CAD (Bertha has no CAD yet)
}
# Names (2026-10-21): the aircraft is K4TQ-A, "Cathy" (the larger rear wing — Stage 1's "B"); K4TQ-A2 is Cathy fully autonomous
# (Stage 2's "AS"); E (equal tandem — Stage 1's "A") and C (compact) are the Stage 1 study configurations.
VARIANTS = {
    "E": {},
    "A1": {"title": "K4TQ-A1 \"Cathy\": unequal tandem (larger rear wing, canard-like), NACA 2412 on both wings", "root_chord_front": 0.16, "tip_chord_front": 0.12,
           "root_chord_rear": 0.24, "tip_chord_rear": 0.18},
    "C": {"title": "compact, 1.2 m span, 0.9 m fuselage, 0.5 kg", "span": 1.20, "fuselage_length": 0.90, "root_chord_front": 0.18, "tip_chord_front": 0.13,
          "root_chord_rear": 0.18, "tip_chord_rear": 0.13, "motor_y": 0.53, "prop": "13x8", "payload_kg": 0.5},
}
VARIANTS["A"] = VARIANTS["A1"]                                  # the earlier name of Cathy
VARIANTS["A2"] = {**VARIANTS["A1"], "title": "K4TQ-A2: Cathy fully autonomous (Jetson Orin NX companion, stereo + downward perception)", "autonomy": True}
VARIANTS["F1"] = {**VARIANTS["A1"], "title": "K4TQ-F1 \"Freddy\": Cathy's airframe with the sections of the NeuralFoil study (front NACA 4415, rear NACA 2415)",
                  "section_front": "NACA 4415", "section_rear": "NACA 2415"}
VARIANTS["F"] = VARIANTS["F1"]                                  # the earlier name of Freddy
VARIANTS["F2"] = {**VARIANTS["F1"], "title": "K4TQ-F2: Freddy fully autonomous (Jetson Orin NX companion, stereo + downward perception)", "autonomy": True}
MAIN = "A1"                        # Cathy: the larger rear wing at 1.2 m, approved 2026-10-21
NICKNAME = {"A1": "Cathy", "A": "Cathy", "A2": "Cathy autonomous", "F1": "Freddy", "F": "Freddy", "F2": "Freddy autonomous"}


def section_of(p: dict, which: str) -> str:
    """The section of one wing: ``section_front`` / ``section_rear``, else ``section``."""
    return p.get(f"section_{which}") or p["section"]


GEOMETRY_KEYS = ("fuselage_length", "fuselage_width", "fuselage_height", "fuselage_section_n", "x_pivot_front", "x_pivot_rear", "z_front", "z_rear",
                 "span", "root_chord_front", "tip_chord_front", "root_chord_rear", "tip_chord_rear", "pivot_chord", "root_gap", "root_length",
                 "bearing_spacing", "stub_od", "gear_module", "z_pinion", "z_gear", "hinge_pin_offset", "lock_couple", "motor_y", "prop_ahead_of_le",
                 "leg_bottom_z", "section")
CAD_MASSES = Path(__file__).resolve().parent / "data" / "k4tq" / "cad_masses.json"


def geometry_fingerprint(p: dict) -> str:
    import hashlib
    import json
    g = {k: p.get(k) for k in GEOMETRY_KEYS}
    if p.get("section_front") or p.get("section_rear"):              # per-wing sections (K4TQ-F): part of the geometry
        g["sections"] = (section_of(p, "front"), section_of(p, "rear"))
    return hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()[:16]


def params(variant: str = MAIN, **over) -> dict:
    """The parameters of a configuration: ``BASE`` with the variant's changes and ``over``. ``cad_mass = "auto"``
    loads the CAD's masses when ``data/k4tq/cad_masses.json`` was written for this geometry (``cad_mass_status`` says
    "CAD", "stale: estimates used" or "none: estimates used")."""
    import json
    p = deepcopy(BASE)
    p.update(VARIANTS[variant])
    p.update(over)
    p["variant"] = variant if not over else p.get("variant", variant)
    p["variant"] = {"A": "A1", "F": "F1"}.get(p["variant"], p["variant"])
    if p.get("cad_mass") == "auto":
        p["cad_mass"] = None
        p["cad_mass_status"] = "none: estimates used"
        if CAD_MASSES.exists():
            doc = json.loads(CAD_MASSES.read_text())
            designs = doc.get("designs") or ({doc["fingerprint"]: doc["cad_mass"]} if "fingerprint" in doc else {})
            fp = geometry_fingerprint(p)
            if fp in designs:
                p["cad_mass"], p["cad_mass_status"] = designs[fp], "CAD"
            else:
                p["cad_mass_status"] = "stale: estimates used"
    elif p.get("cad_mass"):
        p.setdefault("cad_mass_status", "CAD (given)")
    return p


LENGTH_KEYS = ("fuselage_length", "fuselage_width", "fuselage_height", "z_front", "z_rear", "span", "root_chord_front", "tip_chord_front",
               "root_chord_rear", "tip_chord_rear", "root_gap", "root_length", "bearing_spacing", "stub_od", "gear_module", "hinge_pin_offset",
               "lock_couple", "knuckle_spread", "motor_y", "prop_ahead_of_le", "leg_bottom_z", "cargo_box", "battery_box")


def scaled(base_variant: str, s: float, **over) -> dict:
    """The variant's changes with every length × ``s`` (the stations stay fractions of the fuselage): the geometry of
    K4TQ-B "Bertha" and K4TQ-G "Ginna". The hardware (motor, propeller, pack, ESC, root parts) is chosen by the caller in
    ``over``."""
    base = {**BASE, **VARIANTS[base_variant]}
    out = {k: (tuple(x * s for x in base[k]) if isinstance(base[k], tuple) else base[k] * s) for k in LENGTH_KEYS}
    return {**VARIANTS[base_variant], **out, "scale": s, **over}


# K4TQ-B "Bertha" and K4TQ-G "Ginna" (2026-10-21): Freddy scaled to a take-off mass limit (k4tq_scaled.size; ≤ 24 kg: the
# 25 kg C-class line with a 1 kg margin; ≤ 12.5 kg) — the most payload on R3 with the hover rules and Freddy's stall speed
MADE_CALIBRATION = {"wing root (tilt + fold)": 1.525, "wing panel": 1.026, "nacelle": 1.491, "fuselage shell + frames + hatch": 1.817, "fin (dorsal + ventral)": 1.612}      # K4TQ-F1's CAD / estimate for the made parts (no CAD of the scaled designs yet)
SCALED = {
    "B": dict(s=1.85, motor="U12II-100", prop="28x17", battery="12S4P-P45B", root_set="root_M", spar_tube_mm=(40, 36), payload_kg=5.15),
    "G": dict(s=1.35, motor="U8II-170", prop="20x10", battery="12S3P-P45B", root_set="root", spar_tube_mm=(25, 22), payload_kg=2.6),
}
for _k, _v in SCALED.items():
    _v = dict(_v)
    _s = _v.pop("s")
    VARIANTS[_k] = scaled("F1", _s, esc="esc_hv", made_calibration=MADE_CALIBRATION, variant=_k,
                          title=f"K4TQ-{_k} \"{'Bertha' if _k == 'B' else 'Ginna'}\": K4TQ-F1 × {_s:g}, {_v['motor']} + {_v['prop']}, {_v['battery']}", **_v)
_g2 = {k: v for k, v in SCALED["G"].items() if k not in ("s", "payload_kg")}
VARIANTS["G2"] = scaled("F2", SCALED["G"]["s"], esc="esc_hv", made_calibration=MADE_CALIBRATION, variant="G2", payload_kg=2.3,   # Ginna + the Jetson: the payload refilled to 12.5 kg
                        title="K4TQ-G2: Ginna fully autonomous (the Jetson Orin NX + perception), Ginna's drive, 12.5 kg", **_g2)
NICKNAME.update({"B": "Bertha", "G": "Ginna", "G2": "Ginna autonomous"})

# K4TQ-F3 "Freddy 3" and K4TQ-G3 "Ginna 3" (2026-10-21): the Jetson (as A2) and bigger motors, to match or exceed the
# 15-inch FPV quad's top speed with a payload (34.5 m/s) and keep the range of F2 / G2 (k4tq_fast: the off-the-shelf
# motor × propeller × pack sweep, data/k4tq/sweep_*.csv)
FAST = {
    # off the shelf (2026-10-21): T-Motor Navigator motors within their rated voltage, Aeronaut CAM Carbon folding blades
    "F3": dict(motor="MN505S-380", prop="CAM16x13", battery="6S3P-P45B", esc="esc", payload_kg=1.0, v_top=45.0),
    "G3": dict(motor="MN605S-170", prop="CAM17x13", battery="12S3P-P45B", esc="esc_hv", payload_kg=2.1, v_top=40.0, s=1.32),   # 2.1 kg (MUST, 2026-10-22): × 1.32 keeps 12.5 kg and the stall at 18 m/s
}
for _k, _v in FAST.items():
    _v = dict(_v)
    _top, _s = _v.pop("v_top"), _v.pop("s", 1.0)
    _common = dict(autonomy=True, v_map_max=48, v_max_design=round(1.1 * _top, 1), root_set="root", **_v)
    _nick = "Freddy 3" if _k == "F3" else "Ginna 3"
    _title = f"K4TQ-{_k} \"{_nick}\": {'Freddy' if _k == 'F3' else 'Ginna'} + the Jetson, {_v['motor']} + {_v['prop']}, {_v['battery']} (top {_top:g} m/s)"
    if _s == 1.0:
        VARIANTS[_k] = {**VARIANTS["F2"], **_common, "spar_tube_mm": (16, 14), "variant": _k, "title": _title}
    else:
        VARIANTS[_k] = scaled("F2", _s, made_calibration=MADE_CALIBRATION, spar_tube_mm=(25, 22), variant=_k, title=_title, **_common)
NICKNAME.update({"F3": "Freddy 3", "G3": "Ginna 3"})

# K4TQ-H3 "Hanna 3" (3.5 kg) and K4TQ-J3 "Joanna 3" (4.5 kg), 2026-10-22: Ginna 3 grown for payload and hover, not speed
# (k4tq_heavy: the sweep, data/k4tq/sweep_H3/J3.csv). Off the shelf, two drives: the front pair cruises and hovers on rigid
# APC 22x12E, the rear pair hovers only on T-Motor FA24.2 folding (stopped and folded in cruise); MN801-S KV150 all round
HEAVY = {
    "H3": dict(s=1.74, payload_kg=3.5, battery="12S6P-P45B", root_set="root_M", spar_tube_mm=(32, 29)),
    "J3": dict(s=1.77, payload_kg=4.5, battery="12S6P-P45B", root_set="root_M", spar_tube_mm=(35, 32)),
}
for _k, _v in HEAVY.items():
    _v = dict(_v)
    _s = _v.pop("s")
    _nick = "Hanna 3" if _k == "H3" else "Joanna 3"
    VARIANTS[_k] = scaled("F2", _s, motor="MN801S-150", prop="APC22x12E", motor_rear="MN801S-150", prop_rear="FA24.2", esc="esc_hv",
                          autonomy=True, v_map_max=48, v_max_design=40.0, made_calibration=MADE_CALIBRATION, variant=_k,
                          title=f"K4TQ-{_k} \"{_nick}\": Ginna 3 grown × {_s / 1.32:.2f} for {_v['payload_kg']:g} kg — MN801-S + APC 22x12E front, "
                                f"MN801-S + FA24.2 folding rear (hover only), {_v['battery']}", **_v)
NICKNAME.update({"H3": "Hanna 3", "J3": "Joanna 3"})


def avionics_power(p: dict) -> float:
    """The electrical load besides the motors [W]: the flight stack, telemetry, lights (12 W), the temperature logger and the
    cartridge fan (3 W), the tilt servos (4 W) — and for K4TQ-A2 the companion computer and the perception through its DC-DC."""
    base = 19.0
    if p.get("autonomy"):
        a = CATALOGUE["autonomy"]
        eta = a["12 V DC-DC for the companion"]["efficiency"]
        base += sum(v["power_W"] for v in a.values()) / eta
    return base


def battery(p: dict) -> dict:
    b = dict(CATALOGUE["battery"][p["battery"]])
    b["key"] = p["battery"]
    b["voltage"] = b["cells_s"] * b["cell_v"]
    b["energy_wh"] = b["voltage"] * b["capacity_ah"]
    b["usable_wh"] = b["energy_wh"] * 0.85 * 0.95          # to 3.0 V/cell under load × ageing and cold allowance
    return b


def propeller(p: dict) -> dict:
    pr = dict(CATALOGUE["propeller"][p["prop"]])
    pr["key"] = p["prop"]
    pr["D"] = pr["d_in"] * 0.0254
    pr["R"] = pr["D"] / 2
    pr["pitch"] = pr["p_in"] * 0.0254
    pr["A"] = math.pi * pr["R"] ** 2
    return pr


# ================================================================================================= the layout
def _wing(p: dict, which: str) -> dict:
    yf = p["fuselage_width"] / 2
    y_hinge = yf + p["root_gap"]
    y0 = y_hinge + p["root_length"]
    y1 = p["span"] / 2
    cr, ct = p[f"root_chord_{which}"], p[f"tip_chord_{which}"]
    b = y1 - y0
    S = 0.5 * (cr + ct) * b
    lam = ct / cr
    mac = 2 / 3 * cr * (1 + lam + lam ** 2) / (1 + lam)
    y_mac = y0 + b / 3 * (1 + 2 * lam) / (1 + lam)
    pc = p["pivot_chord"]
    le_sweep = math.atan((pc * (cr - ct)) / b)         # the spar is straight across: the leading edge sweeps back by the taper
    x_piv = p[f"x_pivot_{which}"] * p["fuselage_length"]
    c_at = lambda y: cr + (ct - cr) * (y - y0) / b       # noqa: E731
    return {"which": which, "x_pivot": x_piv, "z": p[f"z_{which}"], "y_hinge": y_hinge, "y0": y0, "y1": y1, "semi_span": b, "root_chord": cr,
            "tip_chord": ct, "taper": lam, "S_panel": S, "S_pair": 2 * S, "mac": mac, "y_mac": y_mac, "le_sweep": le_sweep, "pivot_chord": pc,
            "x_ac_mac": x_piv - (pc - 0.25) * mac, "chord_at_motor": c_at(p["motor_y"]), "AR_pair": (2 * y1) ** 2 / (2 * S), "c_at": c_at}


def wing_frame(L: dict, which: str, side: int, tilt: float, fold: float = 0.0):
    """The map from the wing's own frame to body axes. The wing frame: origin on the spar at the fuselage centreline,
    x along the chord (+ aft), y along the spar (outboard), z the section's thickness (+ upper). The fold turns the
    wing about the hinge pin — along the chord at (y_hinge, z = hinge_pin_offset), the span toward +z — then the tilt
    turns everything about the spar (+ = leading edge up), then the side mirrors y. Folding is done at tilt 90°, where
    the pin is vertical and the wing swings aft. Returns ``f(points (n, 3)) -> (n, 3)``."""
    w = L["wings"][which]
    yh, zh = w["y_hinge"], L["p"]["hinge_pin_offset"]
    cf, sf, ct, st = math.cos(fold), math.sin(fold), math.cos(tilt), math.sin(tilt)

    def f(P):
        P = np.atleast_2d(np.asarray(P, float))
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        outer = y > yh - 1e-12
        r, h = y - yh, z - zh
        y1 = np.where(outer, yh + r * cf - h * sf, y)
        z1 = np.where(outer, zh + r * sf + h * cf, z)
        x2, z2 = x * ct + z1 * st, -x * st + z1 * ct
        return np.c_[w["x_pivot"] + x2, side * y1, w["z"] + z2]
    return f


def wing_point(L: dict, which: str, side: int, y: float, xi: float, tilt: float, fold: float = 0.0) -> np.ndarray:
    """A point of a wing in body axes: spanwise station ``y`` (>0) on ``side`` (±1), chord fraction ``xi`` (0 leading edge,
    1 trailing edge), on the chord line, at the tilt (rad, about the spar; + = leading edge up) and the fold (rad, about the
    hinge pin; + = swung aft when folded at tilt 90°)."""
    w = L["wings"][which]
    d = (xi - w["pivot_chord"]) * w["c_at"](y)
    return wing_frame(L, which, side, tilt, fold)([[d, y, 0.0]])[0]


def prop_pose(L: dict, which: str, side: int, tilt: float, fold: float = 0.0) -> dict:
    """The propeller hub and thrust axis at the tilt and fold angles: the thrust line lies on the chord line through
    the spar (no thrust moment about the tilt axis), the disc ahead of the leading edge by ``prop_ahead_of_le``."""
    w = L["wings"][which]
    c = w["c_at"](L["p"]["motor_y"])
    arm = w["pivot_chord"] * c + L["p"]["prop_ahead_of_le"]
    f = wing_frame(L, which, side, tilt, fold)
    hub, ahead = f([[-arm, L["p"]["motor_y"], 0.0], [-arm - 1.0, L["p"]["motor_y"], 0.0]])
    return {"hub": hub, "axis": ahead - hub, "arm": arm}


def layout(p: dict | None = None) -> dict:
    p = p or params()
    pr = propeller(p)
    wf, wr = _wing(p, "front"), _wing(p, "rear")
    L = {"p": p, "prop": pr, "wings": {"front": wf, "rear": wr}}
    Lf = p["fuselage_length"]
    S_ref = wf["S_pair"] + wr["S_pair"]
    stagger = wr["x_pivot"] - wf["x_pivot"]
    L.update({"S_ref": S_ref, "stagger": stagger, "gap": p["z_rear"] - p["z_front"], "c_ref": 0.5 * (wf["mac"] + wr["mac"]),
              "AR_total": p["span"] ** 2 / S_ref, "fuselage_length": Lf})
    # propeller stations in cruise and hover
    poses = {}
    for which in ("front", "rear"):
        for side in (-1, 1):
            poses[(which, side)] = {"cruise": prop_pose(L, which, side, 0.0), "hover": prop_pose(L, which, side, math.pi / 2)}
    L["poses"] = poses
    R = pr["R"]
    hf, hr = poses[("front", 1)]["hover"]["hub"], poses[("rear", 1)]["hover"]["hub"]
    cf, cr_ = poses[("front", 1)]["cruise"]["hub"], poses[("rear", 1)]["cruise"]["hub"]
    te_low_hover = min(wing_point(L, w, 1, wf["y0"], 1.0, math.pi / 2)[2] for w in ("front", "rear"))
    L["clearance"] = {
        "hover disc to disc (front-rear) [m]": float(math.hypot(hr[0] - hf[0], hr[2] - hf[2]) - 2 * R) if abs(hr[2] - hf[2]) < 2 * R else float(abs(hr[0] - hf[0])),
        "hover disc separation along x [m]": float(hr[0] - hf[0] - 2 * R),
        "disc to fuselage side [m]": float(p["motor_y"] - R - p["fuselage_width"] / 2),
        "disc beyond the wingtip [m]": float(p["motor_y"] + R - p["span"] / 2),
        "rear disc (cruise) to the front wing trailing edge [m]": float(cr_[0] - wing_point(L, "front", 1, p["motor_y"], 1.0, 0.0)[0]),
        "hover stance: wing trailing edge above the feet [m]": float(te_low_hover - p["leg_bottom_z"]),
        "cruise stance: lowest disc edge above the feet [m]": float(min(cf[2], cr_[2]) - R - p["leg_bottom_z"]),
    }
    L["width_overall"] = 2 * (p["motor_y"] + R)
    L["cruise_disc_height"] = {"front": float(cf[2]), "rear": float(cr_[2])}
    L["hover_disc_height"] = {"front": float(hf[2]), "rear": float(hr[2])}
    L["thrust_x"] = {"front": float(hf[0]), "rear": float(hr[0])}           # hover thrust stations along x
    return L


def checks(L: dict) -> list[str]:
    c = L["clearance"]
    out = []
    if c["hover disc separation along x [m]"] < 0.05:
        out.append(f"hover discs front to rear {c['hover disc separation along x [m]'] * 1e3:.0f} mm apart along x (< 50 mm)")
    if c["disc to fuselage side [m]"] < 0.05:
        out.append("discs within 50 mm of the fuselage")
    if c["rear disc (cruise) to the front wing trailing edge [m]"] < 0.10:
        out.append("the rear disc is within 100 mm behind the front wing's trailing edge in cruise")
    if c["hover stance: wing trailing edge above the feet [m]"] < 0.02:
        out.append("hover stance: the wing trailing edges touch the ground")
    if c["cruise stance: lowest disc edge above the feet [m]"] < -0.06:
        out.append("cruise stance: even the folded blades reach the ground")
    return out


def folded_envelope(L: dict, fold_deg: float, tilt_deg: float = 90.0) -> dict:
    """The packed size at a fold angle with the wings at ``tilt_deg`` (90: edge-on): length, width, height of the box that
    holds the fuselage, the folded wings and the nacelles (propeller blades folded along the nacelles, legs off). A first
    estimate for the tube; Stage 2 replaces it with the CAD's bounding box."""
    p, pr = L["p"], L["prop"]
    fold, tilt = math.radians(fold_deg), math.radians(tilt_deg)
    pts = [np.array([0, 0, 0]), np.array([L["fuselage_length"], 0, 0])]
    for which in ("front", "rear"):
        w = L["wings"][which]
        for side in (-1, 1):
            for y in (w["y0"], w["y1"]):
                for xi in (0.0, 1.0):
                    pts.append(wing_point(L, which, side, y, xi, tilt, fold))
            pose = prop_pose(L, which, side, tilt, fold)
            pts.append(pose["hub"] + 0.05 * pose["axis"])                 # spinner
            pts.append(pose["hub"] - 0.06 * pose["axis"])                 # motor can behind the hub
    P = np.array(pts)
    half_w = max(np.abs(P[:, 1]).max(), p["fuselage_width"] / 2) + 0.025      # + nacelle radius
    z_lo, z_hi = min(P[:, 2].min(), -p["fuselage_height"] / 2), max(P[:, 2].max(), p["fuselage_height"] / 2)
    x_lo, x_hi = P[:, 0].min(), P[:, 0].max()
    width, height = 2 * half_w, z_hi - z_lo
    return {"fold_deg": fold_deg, "tilt_deg": tilt_deg, "length [m]": float(x_hi - x_lo), "width [m]": float(width), "height [m]": float(height),
            "tube inner diameter needed [m]": float(math.hypot(width, height)), "nacelle diameter [m]": 0.05}


# ================================================================================================= masses
def spar_tube(p: dict) -> tuple:
    return tuple(t / 1000 for t in p.get("spar_tube_mm", (16, 14)))


def mass_table(L: dict, *, payload_kg: float | None = None, battery_key: str | None = None, x_cg_target: float | None = None) -> list[dict]:
    """Every item with mass, position and group; the battery placed for ``x_cg_target`` (default: 45 % of the way from
    the front to the rear pivot — the sizing replaces it with the stability's target), clipped to its bay."""
    p, pr = L["p"], L["prop"]
    cat = CATALOGUE
    items = []

    def add(name, kg, x, y=0.0, z=0.0, group="", source=""):
        items.append({"item": name, "mass_kg": float(kg), "x": float(x), "y": float(y), "z": float(z), "group": group, "source": source})

    mot = cat["motor"][p["motor"]]
    sc = p.get("scale", 1.0)                                      # K4TQ-B: the small made parts and the offsets grow with the airframe
    esc = cat[p.get("esc", "esc")]
    root = cat[p.get("root_set", "root")]
    od, idd = spar_tube(p)
    tube_kg_m = CARBON_TUBE["rho"] * math.pi / 4 * (od ** 2 - idd ** 2)
    root_kg = sum(v["mass_kg"] for v in root.values())
    for which in ("front", "rear"):
        w = L["wings"][which]
        for side in (-1, 1):
            s = "L" if side < 0 else "R"
            h = L["poses"][(which, side)]["cruise"]["hub"]
            mo_w = cat["motor"][p["motor_rear"]] if (which == "rear" and p.get("motor_rear")) else mot
            pr_w = cat["propeller"][p["prop_rear"]] if (which == "rear" and p.get("prop_rear")) else pr
            add(f"motor {which} {s}", mo_w["mass_kg"], h[0] + 0.03 * sc, h[1], h[2], "propulsion", mo_w["name"])
            add(f"propeller {which} {s}", pr_w["mass_kg"], h[0], h[1], h[2], "propulsion", pr_w["source"])
            add(f"ESC {which} {s}", esc["mass_kg"], h[0] + 0.07 * sc, h[1], h[2], "propulsion", esc["name"])
            add(f"nacelle {which} {s}", NACELLE_KG * sc ** 2, h[0] + 0.06 * sc, h[1], h[2], "structure", "printed PA-CF")
            y_c = w["y_mac"]
            add(f"wing panel {which} {s}", WING_AREAL_KG_M2 * w["S_panel"], w["x_pivot"] + (0.42 - w["pivot_chord"]) * w["mac"], side * y_c, w["z"], "structure",
                f"{WING_AREAL_KG_M2} kg/m² composite")
            spar_len = w["semi_span"] + p["root_length"] * 0.5
            add(f"spar {which} {s}", tube_kg_m * spar_len, w["x_pivot"], side * (w["y0"] + w["semi_span"] / 2), w["z"], "structure",
                f"carbon tube Ø{od * 1e3:.0f}×{idd * 1e3:.0f}")
            add(f"wing root (tilt + fold) {which} {s}", root_kg, w["x_pivot"], side * (w["y_hinge"] + p["root_length"] / 2), w["z"], "roots",
                " + ".join(root))
            add(f"power wiring {which} {s}", WIRING_KG_M * (p["motor_y"] + 0.25 * sc), 0.5 * (w["x_pivot"] + L["fuselage_length"] * 0.5), side * p["motor_y"] / 2, 0.0, "systems")
    # fuselage: an ellipsoid-ended cylinder's wetted area
    Lf, bw, bh = L["fuselage_length"], p["fuselage_width"], p["fuselage_height"]
    S_wet_f = math.pi * math.sqrt((bw ** 2 + bh ** 2) / 2) * Lf * 0.78
    add("fuselage shell + frames + hatch", FUSELAGE_AREAL_KG_M2 * S_wet_f, 0.47 * Lf, 0, 0, "structure", f"{FUSELAGE_AREAL_KG_M2} kg/m² on {S_wet_f:.2f} m²")
    for (x, side) in ((L["wings"]["front"]["x_pivot"], -1), (L["wings"]["front"]["x_pivot"], 1), (L["wings"]["rear"]["x_pivot"], -1), (L["wings"]["rear"]["x_pivot"], 1)):
        add(f"leg {'front' if x < 0.5 * Lf else 'rear'} {'L' if side < 0 else 'R'}", LEG_KG * sc ** 2, x, side * 0.06 * sc, 0.5 * (p["leg_bottom_z"] - bh / 2), "structure")
    if p["fin"]:
        S_fin = p["fin_area"] if p["fin_area"] else 0.030 * sc ** 2
        add("fin (dorsal + ventral)", FIN_AREAL_KG_M2 * S_fin, 0.93 * Lf, 0, 0.0, "structure", "sized by the directional stability")
    x_cargo = p["cargo_station"] * Lf
    for k, v in cat["cargo"].items():
        add(k, v["mass_kg"] * (sc ** 2 if k == "insulated cartridge" else 1.0), x_cargo, 0, 0.005, "cargo", v["source"])
    add("payload", p["payload_kg"] if payload_kg is None else payload_kg, x_cargo, 0, 0.005, "payload", "medical load (R2)")
    for i, (k, v) in enumerate(cat["avionics"].items()):
        add(k, v["mass_kg"], p["avionics_station"] * Lf + 0.012 * sc * (i - 3), 0, 0.0, "avionics", v["source"])       # the tail bay behind the rear station (Stage 2 packing)
    add("servo and signal wiring", 0.050 * sc, 0.5 * Lf, 0, 0, "systems")
    if p.get("autonomy"):                                         # K4TQ-A2
        auto = cat["autonomy"]
        x_f = L["wings"]["front"]["x_pivot"]
        stations = {"companion computer (Jetson Orin NX 16 GB + carrier + heatsink/fan)": (p.get("companion_station", 0.85) * Lf, 0.0),   # the tail bay (CG), a NACA inlet
                    "forward stereo depth camera": (0.105 * Lf / 1.2, 0.0), "downward camera + rangefinder": (x_f + 0.045, -p["fuselage_height"] / 2 + 0.012),
                    "12 V DC-DC for the companion": (0.175 * Lf / 1.2, 0.03)}
        for k, v in auto.items():
            add(k, v["mass_kg"], stations[k][0], 0, stations[k][1], "autonomy", v["source"])
    if p["parachute"]:
        add("recovery parachute", cat["safety"]["recovery parachute"]["mass_kg"] * sc ** 3, p["parachute_station"] * Lf, 0, 0.0, "safety", cat["safety"]["recovery parachute"]["source"])
    cad = p.get("cad_mass") or {}
    if cad:                                                      # Stage 2.8: the CAD's masses replace the estimates, item by item
        for i in items:
            for prefix, kg in cad.items():
                if i["item"].startswith(prefix):
                    i["source"] = f"CAD ({i['mass_kg'] * 1e3:.0f} g estimated → {kg * 1e3:.0f} g): " + i.get("source", "")
                    i["mass_kg"] = float(kg)
    elif p.get("made_calibration"):                              # K4TQ-B: the estimates corrected as K4TQ-F1's CAD corrected its own
        for i in items:
            for prefix, f in p["made_calibration"].items():
                if i["item"].startswith(prefix):
                    i["source"] = f"estimate × {f:.2f} (CAD/estimate of K4TQ-F1): " + i.get("source", "")
                    i["mass_kg"] *= float(f)
    empty = sum(i["mass_kg"] for i in items if i["group"] not in ("payload",))
    add("growth margin", GROWTH_MARGIN * empty, x_cargo, 0, 0, "margin", f"{GROWTH_MARGIN:.0%} of the empty mass")
    b = CATALOGUE["battery"][battery_key or p["battery"]]
    if p.get("battery_x"):                                       # Stage 2: two parallel half-packs at stations set by k4tq_sizing.pack
        hx, hy, hz = battery_half_box(battery_key or p["battery"], across_of(p))
        for name, fx in zip(("A (nose)", "B (mid)"), p["battery_x"]):
            items.append({"item": f"battery {name} {battery_key or p['battery']} half", "mass_kg": b["mass_kg"] / 2, "x": fx * Lf, "y": 0.0,
                          "z": -p["fuselage_height"] / 2 + 0.010 + hz / 2, "group": "battery", "source": b["source"] + " — half of the pack, in parallel",
                          "clipped": False, "box": (hx, hy, hz)})
        return items
    target = x_cg_target if x_cg_target is not None else L["wings"]["front"]["x_pivot"] + 0.45 * L["stagger"]
    m = sum(i["mass_kg"] for i in items)
    mx = sum(i["mass_kg"] * i["x"] for i in items)
    bay = (p["battery_bay"][0] * Lf, p["battery_bay"][1] * Lf)
    xb, clipped = og_mass.place_mass_for_cg(m, mx, b["mass_kg"], target, bay)
    items.append({"item": f"battery {battery_key or p['battery']}", "mass_kg": b["mass_kg"], "x": xb, "y": 0.0, "z": -0.035 * sc, "group": "battery",
                  "source": b["source"], "clipped": clipped})
    return items


def across_of(p: dict) -> int:
    """Cells side by side across the fuselage floor: three in K4TQ, as many as the wider floor takes when scaled."""
    return max(3, int((p["fuselage_width"] - 0.075) / 0.0215)) if p.get("scale", 1.0) > 1.0 else 3


def battery_half_box(key: str, across: int = 3) -> tuple:
    """The outer size [m] of one half-pack: its cells standing (21700: Ø21.5 × 70 mm, + wrap and BMS lead), ``across``
    side by side."""
    bt = CATALOGUE["battery"][key]
    n = bt["cells_s"] * bt["cells_p"] // 2
    nx = math.ceil(n / across)
    return (0.0215 * nx + 0.004, 0.0215 * across + 0.004, 0.074)


def cg_inertia(items: list[dict]) -> dict:
    """Mass, CG and inertia (``og.mass.Budget``, point masses — the wing panels and spars as rods along y)."""
    bud = og_mass.Budget()
    for i in items:
        shape = og_mass.Rod(length=0.6, axis="y") if i["item"].startswith(("wing panel", "spar")) else None
        bud.add(og_mass.Item(i["item"], i["mass_kg"], (i["x"], i["y"], i["z"]), shape, i["group"]))
    mp = bud.properties()
    return {"mass_kg": mp.mass, "x_cg": float(mp.com[0]), "y_cg": float(mp.com[1]), "z_cg": float(mp.com[2]), "Ixx": float(mp.inertia[0, 0]),
            "Iyy": float(mp.inertia[1, 1]), "Izz": float(mp.inertia[2, 2]), "by_group": bud.by_group()}


def groups(items: list[dict]) -> dict:
    out = {}
    for i in items:
        out[i["group"]] = out.get(i["group"], 0.0) + i["mass_kg"]
    return out


def tilt_wing_tax(items: list[dict]) -> float:
    """What the wings and their mechanisms cost against a quadcopter's arms: wing panels, spars, roots, fin, nacelles."""
    return float(sum(i["mass_kg"] for i in items if i["item"].startswith(("wing panel", "spar", "wing root", "fin", "nacelle"))))


def price(items: list[dict], p: dict) -> dict:
    """The bought parts' cost (EUR, estimates) — the made parts in Stage 4."""
    c = CATALOGUE
    mr, prr = p.get("motor_rear") or p["motor"], p.get("prop_rear") or p["prop"]
    rows = {"motors": 2 * c["motor"][p["motor"]]["price_eur"] + 2 * c["motor"][mr]["price_eur"],
            "propellers (+2 spares)": 3 * c["propeller"][p["prop"]]["price_eur"] + 3 * c["propeller"][prr]["price_eur"],
            "ESCs": 4 * c[p.get("esc", "esc")]["price_eur"], "battery": c["battery"][p["battery"]]["price_eur"],
            "avionics": sum(v["price_eur"] for v in c["avionics"].values()), "cargo + temperature logging": sum(v["price_eur"] for v in c["cargo"].values()),
            **({"autonomy (K4TQ-A2)": sum(v["price_eur"] for v in c["autonomy"].values())} if p.get("autonomy") else {}),
            "wing roots (4)": 4 * sum(v["price_eur"] for v in c[p.get("root_set", "root")].values())}
    if p["parachute"]:
        rows["parachute"] = c["safety"]["recovery parachute"]["price_eur"]
    rows["total bought"] = sum(rows.values())
    return rows
