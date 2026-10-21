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
        "12x8": {"d_in": 12, "p_in": 8.0, "mass_kg": 0.030, "price_eur": 16.0, "source": "APC 12x8E class, folding hub included"},
        "13x5.5": {"d_in": 13, "p_in": 5.5, "mass_kg": 0.030, "price_eur": 18.0, "source": "multirotor 13x5.5 CF class (quadcopter reference)"},
        "17x5.8": {"d_in": 17, "p_in": 5.8, "mass_kg": 0.050, "price_eur": 35.0, "source": "T-Motor P17x5.8 (the MN4014 KV400's recommended 6S propeller)"},
    },
    "esc": {"name": "40 A 6S BLHeli_32 / AM32 (e.g. T-Motor AIR 40A)", "mass_kg": 0.028, "price_eur": 35.0, "source": "AIR 40A class, in the nacelle"},
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
    "cargo": {
        "insulated cartridge": {"mass_kg": 0.200, "price_eur": 90.0, "source": "VIP/EPP-lined removable shell 240 x 120 x 110 mm, ~1.2 l inner, latch"},
        "shock isolators (4)": {"mass_kg": 0.016, "price_eur": 12.0, "source": "silicone grommet isolators"},
        "temperature sensors (2)": {"mass_kg": 0.006, "price_eur": 30.0, "source": "TMP117 class (±0.1 °C), two independent channels (R7)"},
        "temperature logger": {"mass_kg": 0.020, "price_eur": 60.0, "source": "independent logger with its own cell and flash, GNSS-timestamped, survives power loss (R7)"},
    },
    "root": {   # one wing root: the tilt drive and the fold hinge (Stage 2 designs it; these are the Stage 1 estimates)
        "tilt servo": {"mass_kg": 0.062, "price_eur": 90.0, "source": "high-torque metal-gear digital servo, 30 kg·cm class (~2.9 N·m)"},
        "reduction 2:1": {"mass_kg": 0.015, "price_eur": 15.0, "source": "spur pair, anti-backlash, or a toothed belt"},
        "bearings 6804-2RS (2)": {"mass_kg": 0.036, "price_eur": 10.0, "source": "20 x 32 x 7 deep groove, C 4.0 kN, C0 2.47 kN"},
        "housing": {"mass_kg": 0.040, "price_eur": 40.0, "source": "7075 machined core + printed cover"},
        "fold hinge + structural latch": {"mass_kg": 0.030, "price_eur": 25.0, "source": "steel pin in bronze bushes, Ø6 hardened locking pin with spring and witness mark"},
        "encoder": {"mass_kg": 0.004, "price_eur": 12.0, "source": "AS5048 magnetic encoder on the spar axis"},
        "fasteners + clamp": {"mass_kg": 0.010, "price_eur": 5.0, "source": "spar clamp, M3/M4 A2-70"},
    },
    "safety": {
        "recovery parachute": {"mass_kg": 0.160, "price_eur": 250.0, "source": "1 m² canopy + spring ejector (the four-rotor motor-out case; SORA mitigation)"},
    },
}

# areal and linear densities of the made parts (estimates, Stage 2 CAD replaces them)
WING_AREAL_KG_M2 = 0.70           # glass/carbon skins on XPS foam, 12 % thick, both surfaces (without the spar)
FUSELAGE_AREAL_KG_M2 = 0.60       # glass/carbon shell incl. frames' share
FIN_AREAL_KG_M2 = 0.55
NACELLE_KG = 0.022                # printed PA-CF nacelle with the ESC bay
LEG_KG = 0.020
WIRING_KG_M = 0.060               # one 6S power pair (2 × 14 AWG silicone) + a signal lead, per metre
CARBON_TUBE = {"E": 80e9, "G": 4.5e9, "sigma_allow": 400e6, "rho": 1550.0, "source": "pultruded/roll-wrapped, as kestrel_structure.CARBON; G for torsion"}
TUBES_MM = [(12, 10), (14, 12), (16, 14), (18, 16), (20, 18)]
GROWTH_MARGIN = 0.07              # on the empty mass

# ================================================================================================= the configurations
BASE = {
    "variant": "A", "title": "equal tandem, 1.4 m, 1.2 m fuselage",
    # fuselage
    "fuselage_length": 1.20, "fuselage_width": 0.14, "fuselage_height": 0.15,   # 1.2 m: the length sweep (disc gap 159 mm), approved 2026-10-21
    # wings: pivot stations (fraction of the fuselage length), heights (front low, rear high), planform
    "x_pivot_front": 0.30, "x_pivot_rear": 0.75, "z_front": -0.055, "z_rear": 0.055,   # low front wing, shoulder rear wing: gap 0.11 m
    "span": 1.40, "root_chord_front": 0.20, "tip_chord_front": 0.15, "root_chord_rear": 0.20, "tip_chord_rear": 0.15,
    "pivot_chord": 0.30,              # the spar (tilt axis) at this fraction of the local chord, straight across the span
    "root_gap": 0.012,                # fuselage side to the fold hinge axis
    "root_length": 0.045,             # fold hinge axis to the first wing rib (the tilt cartridge)
    "section": "NACA 2412", "decalage_deg": 1.5,
    # propulsion
    "motor": "MN4014-400", "prop": "15x10", "motor_y": 0.62, "prop_ahead_of_le": 0.065,   # 15x10: the Stage 1.8 sweep's choice
    "battery": "6S2P-P45B", "payload_kg": 1.0,
    # stations along the fuselage (fractions of its length): the cargo at the CG, the battery bay, the parachute
    "cargo_station": 0.46, "battery_bay": (0.10, 0.70), "parachute_station": 0.56,
    # undercarriage, fin, safety
    "leg_bottom_z": -0.225,           # leg feet below the fuselage axis: clears the wing trailing edges (hover stance) and the discs (cruise stance)
    "fin": True, "fin_area": None,    # None: the fin is sized by k4tq_sizing.directional (R9)
    "parachute": True,
}
VARIANTS = {
    "A": {},
    "B": {"title": "unequal tandem (larger rear wing, canard-like)", "root_chord_front": 0.16, "tip_chord_front": 0.12, "root_chord_rear": 0.24, "tip_chord_rear": 0.18},
    "C": {"title": "compact, 1.2 m span, 0.9 m fuselage, 0.5 kg", "span": 1.20, "fuselage_length": 0.90, "root_chord_front": 0.18, "tip_chord_front": 0.13,
          "root_chord_rear": 0.18, "tip_chord_rear": 0.13, "motor_y": 0.53, "prop": "13x8", "payload_kg": 0.5},
}
MAIN = "B"                         # the larger rear wing at 1.2 m: approved 2026-10-21


def params(variant: str = MAIN, **over) -> dict:
    """The parameters of a configuration: ``BASE`` with the variant's changes and ``over``."""
    p = deepcopy(BASE)
    p.update(VARIANTS[variant])
    p.update(over)
    p["variant"] = variant if not over else p.get("variant", variant)
    return p


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


def wing_point(L: dict, which: str, side: int, y: float, xi: float, tilt: float, fold: float = 0.0) -> np.ndarray:
    """A point of a wing in body axes: spanwise station ``y`` (>0) on ``side`` (±1), chord fraction ``xi`` (0 leading edge,
    1 trailing edge) at the wing's tilt (rad, about the spar; + = leading edge up) and fold (rad, about the vertical hinge;
    + = swung aft)."""
    w = L["wings"][which]
    c = w["c_at"](y)
    d = (xi - w["pivot_chord"]) * c                     # along the chord from the spar, + aft
    x_rel, z_rel = d * math.cos(tilt), -d * math.sin(tilt)   # LE (d<0) goes up when the tilt is +
    r = y - w["y_hinge"]                                # from the fold hinge, outboard
    # fold: rotate (x_rel, r) about the hinge's vertical axis, the tip swinging aft
    xf = x_rel * math.cos(fold) + r * math.sin(fold)
    yf = -x_rel * math.sin(fold) + r * math.cos(fold)
    return np.array([w["x_pivot"] + xf, side * (w["y_hinge"] + yf), w["z"] + z_rel])


def prop_pose(L: dict, which: str, side: int, tilt: float, fold: float = 0.0) -> dict:
    """The propeller hub and thrust axis at the tilt and fold angles: the thrust line lies on the chord line through
    the spar (no thrust moment about the tilt axis), the disc ahead of the leading edge by ``prop_ahead_of_le``."""
    w = L["wings"][which]
    c = w["c_at"](L["p"]["motor_y"])
    xi = -(L["p"]["prop_ahead_of_le"] / c)               # ahead of the leading edge, as a chord fraction (< 0)
    hub = wing_point(L, which, side, L["p"]["motor_y"], xi, tilt, fold)
    axis = np.array([-math.cos(tilt) * math.cos(fold), side * math.cos(tilt) * math.sin(fold), math.sin(tilt)])   # thrust direction
    return {"hub": hub, "axis": axis, "arm": w["pivot_chord"] * c + L["p"]["prop_ahead_of_le"]}


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
    od, idd = spar_tube(p)
    tube_kg_m = CARBON_TUBE["rho"] * math.pi / 4 * (od ** 2 - idd ** 2)
    root_kg = sum(v["mass_kg"] for v in cat["root"].values())
    for which in ("front", "rear"):
        w = L["wings"][which]
        for side in (-1, 1):
            s = "L" if side < 0 else "R"
            h = L["poses"][(which, side)]["cruise"]["hub"]
            add(f"motor {which} {s}", mot["mass_kg"], h[0] + 0.03, h[1], h[2], "propulsion", mot["name"])
            add(f"propeller {which} {s}", pr["mass_kg"], h[0], h[1], h[2], "propulsion", pr["source"])
            add(f"ESC {which} {s}", cat["esc"]["mass_kg"], h[0] + 0.07, h[1], h[2], "propulsion", cat["esc"]["name"])
            add(f"nacelle {which} {s}", NACELLE_KG, h[0] + 0.06, h[1], h[2], "structure", "printed PA-CF")
            y_c = w["y_mac"]
            add(f"wing panel {which} {s}", WING_AREAL_KG_M2 * w["S_panel"], w["x_pivot"] + (0.42 - w["pivot_chord"]) * w["mac"], side * y_c, w["z"], "structure",
                f"{WING_AREAL_KG_M2} kg/m² composite")
            spar_len = w["semi_span"] + p["root_length"] * 0.5
            add(f"spar {which} {s}", tube_kg_m * spar_len, w["x_pivot"], side * (w["y0"] + w["semi_span"] / 2), w["z"], "structure",
                f"carbon tube Ø{od * 1e3:.0f}×{idd * 1e3:.0f}")
            add(f"wing root (tilt + fold) {which} {s}", root_kg, w["x_pivot"], side * (w["y_hinge"] + p["root_length"] / 2), w["z"], "roots",
                " + ".join(cat["root"]))
            add(f"power wiring {which} {s}", WIRING_KG_M * (p["motor_y"] + 0.25), 0.5 * (w["x_pivot"] + L["fuselage_length"] * 0.5), side * p["motor_y"] / 2, 0.0, "systems")
    # fuselage: an ellipsoid-ended cylinder's wetted area
    Lf, bw, bh = L["fuselage_length"], p["fuselage_width"], p["fuselage_height"]
    S_wet_f = math.pi * math.sqrt((bw ** 2 + bh ** 2) / 2) * Lf * 0.78
    add("fuselage shell + frames + hatch", FUSELAGE_AREAL_KG_M2 * S_wet_f, 0.47 * Lf, 0, 0, "structure", f"{FUSELAGE_AREAL_KG_M2} kg/m² on {S_wet_f:.2f} m²")
    for (x, side) in ((L["wings"]["front"]["x_pivot"], -1), (L["wings"]["front"]["x_pivot"], 1), (L["wings"]["rear"]["x_pivot"], -1), (L["wings"]["rear"]["x_pivot"], 1)):
        add(f"leg {'front' if x < 0.5 * Lf else 'rear'} {'L' if side < 0 else 'R'}", LEG_KG, x, side * 0.06, 0.5 * (p["leg_bottom_z"] - bh / 2), "structure")
    if p["fin"]:
        S_fin = p["fin_area"] if p["fin_area"] else 0.030
        add("fin (dorsal + ventral)", FIN_AREAL_KG_M2 * S_fin, 0.93 * Lf, 0, 0.0, "structure", "sized by the directional stability")
    x_cargo = p["cargo_station"] * Lf
    for k, v in cat["cargo"].items():
        add(k, v["mass_kg"], x_cargo, 0, 0.005, "cargo", v["source"])
    add("payload", p["payload_kg"] if payload_kg is None else payload_kg, x_cargo, 0, 0.005, "payload", "medical load (R2)")
    for i, (k, v) in enumerate(cat["avionics"].items()):
        add(k, v["mass_kg"], 0.14 + 0.03 * i, 0, 0.03, "avionics", v["source"])
    add("servo and signal wiring", 0.050, 0.5 * Lf, 0, 0, "systems")
    if p["parachute"]:
        add("recovery parachute", cat["safety"]["recovery parachute"]["mass_kg"], p["parachute_station"] * Lf, 0, 0.05, "safety", cat["safety"]["recovery parachute"]["source"])
    empty = sum(i["mass_kg"] for i in items if i["group"] not in ("payload",))
    add("growth margin", GROWTH_MARGIN * empty, x_cargo, 0, 0, "margin", f"{GROWTH_MARGIN:.0%} of the empty mass")
    b = CATALOGUE["battery"][battery_key or p["battery"]]
    target = x_cg_target if x_cg_target is not None else L["wings"]["front"]["x_pivot"] + 0.45 * L["stagger"]
    m = sum(i["mass_kg"] for i in items)
    mx = sum(i["mass_kg"] * i["x"] for i in items)
    bay = (p["battery_bay"][0] * Lf, p["battery_bay"][1] * Lf)
    xb, clipped = og_mass.place_mass_for_cg(m, mx, b["mass_kg"], target, bay)
    items.append({"item": f"battery {battery_key or p['battery']}", "mass_kg": b["mass_kg"], "x": xb, "y": 0.0, "z": -0.035, "group": "battery",
                  "source": b["source"], "clipped": clipped})
    return items


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
    rows = {"motors": 4 * c["motor"][p["motor"]]["price_eur"], "propellers (+2 spares)": 6 * c["propeller"][p["prop"]]["price_eur"],
            "ESCs": 4 * c["esc"]["price_eur"], "battery": c["battery"][p["battery"]]["price_eur"],
            "avionics": sum(v["price_eur"] for v in c["avionics"].values()), "cargo + temperature logging": sum(v["price_eur"] for v in c["cargo"].values()),
            "wing roots (4)": 4 * sum(v["price_eur"] for v in c["root"].values())}
    if p["parachute"]:
        rows["parachute"] = c["safety"]["recovery parachute"]["price_eur"]
    rows["total bought"] = sum(rows.values())
    return rows
