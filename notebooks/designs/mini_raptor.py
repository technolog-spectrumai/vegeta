"""Mini raptors — Apus Mini, Alcotan, Gavilan (notebook 49 ``49_mini_raptor``): three first-guess tailsitters on a
stock 5-inch FPV racing stack, by first principles and empirical correlations only — no CFD, no FEA, no optimisation,
no long simulation. Conceptual geometry, not validated structure.

The stack every design reuses where it makes sense: four 2207-class motors, 5-inch three-blade propellers (5x4.3; the
chaser takes the 5x5), the 4-in-1 ESC and the flight controller, the FPV camera and video link, a 4S 1300 / 4S 1500 /
6S 1300 pack. The reference is the racing quad itself on the same parts (``quad``), not the table's rounded numbers.

Three independent layouts:
- **Apus Mini** — economy at the quad's speed and manoeuvre: one wing a third of a gate wide (0.50 × 0.17 m), two of
  the stack's motors on it, the hover pitch from the elevons in the slipstream (as every two-motor tailsitter);
  4S 1500. It began as a biplane cell and passed through a cruciform (``ALTERNATIVES``): the question "is the biplane
  — or any four-motor cell — the right one?" (``biplane_question``) moved it to a twin-motor monoplane.
- **Alcotan** — the chaser: a cruciform (+) tailsitter, a small highly loaded wing and two fins carrying the four motors
  at the arm tips (no posts, nothing between the discs), 6S and the steeper 5x5 for the pitch speed, a slender pod.
- **Gavilan** — tight spaces: the racing frame's own motor square (155 mm) wrapped in a short biplane cell (span 0.31 m,
  a fifth of a gate; the only cell that carries four 5-inch discs on that square — a cruciform needs 0.36 m), faired
  posts, the lightest pack, the biggest thrust-to-weight and the lowest inertia; the wing is mostly a brake and a
  glider in a hangar course, and the model says so.

``performance`` gives for each: the polar from a drag build-up at the chord Reynolds number and the induced factor
(Prandtl's biplane factor or the monoplane's), power / Wh per km / endurance / range against speed, the stall, the top
speed where the drives' full thrust meets the drag, the load factor (wing, thrust, structure) and the banked-turn radius
against speed, the hover (power, thrust-to-weight, the pitch and roll accelerations), the braking from 20 m/s with the
wing as an air brake, the hover reorientation time, the transition margins. ``quad`` is the same for the racer: tilted
flight, the tilt-limited brake, the yaw and tilt reorientation. The drives come from Boreas once and are cached in
``data/mini_raptor_drive_<key>.json`` (~2 min each when missing). The CAD (``MiniRaptor``, Dedalus / CadQuery) builds
each design with its component envelopes and propeller discs; ``parts`` gives the named, coloured solids for the
labelled renders and ``ortho_views`` the dimensioned orthographic views.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import kestrel
import kestrel_flight as kf
import kestrel_systems as ks

RHO, NU, G = 1.225, 1.46e-5, 9.81
DATA = Path(__file__).resolve().parent / "data"
CAD_DIR = DATA / "cad" / "mini_raptor"
GATE_WIDTH = 1.5                     # a racing gate [m]
FPV_W = 6.0                          # the camera, the video link and the flight controller
R_PROP = 0.0635                      # 5 inch
DISC_CLEARANCE = 0.015               # between two discs, and disc to pod [m]
V_GRID = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0, 24.0, 28.0, 32.0, 36.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0)

# ------------------------------------------------------------------------------------------------- the racing stack
STACK = {
    "motor": {"name": "2207-class racing outrunner", "mass_kg": 0.031, "R_ohm": 0.06, "I0_A": 1.0, "I_max_A": 40.0, "d_m": 0.028, "l_m": 0.020},
    "propeller": {"name": "5-inch three-blade racing propeller", "mass_kg": 0.004, "chord_root_m": 0.010, "chord_max_m": 0.016, "chord_tip_m": 0.006},
    "stack": {"name": "flight controller + 4-in-1 ESC (30.5 mm stack)", "mass_kg": 0.030, "size_m": (0.040, 0.040, 0.025)},
    "fpv": {"name": "FPV camera + video transmitter + antenna", "mass_kg": 0.035, "size_m": (0.025, 0.020, 0.020), "vtx_size_m": (0.035, 0.025, 0.010)},
    "receiver": {"name": "receiver + GNSS puck (the tailsitter firmware wants a heading)", "mass_kg": 0.012},
    "wiring": {"name": "wiring, connectors, straps", "mass_kg": 0.015},
}
PACKS = {"4S 1300": {"cells": 4, "capacity_ah": 1.3, "mass_kg": 0.150, "size_m": (0.075, 0.035, 0.030)},
         "4S 1500": {"cells": 4, "capacity_ah": 1.5, "mass_kg": 0.170, "size_m": (0.080, 0.036, 0.032)},
         "6S 1300": {"cells": 6, "capacity_ah": 1.3, "mass_kg": 0.210, "size_m": (0.080, 0.040, 0.045)}}
DRIVES = {"4S 5x4.3": {"pitch_in": 4.3, "kv": 2400.0, "pack": "4S 1500"},
          "4S 5x4.3 light": {"pitch_in": 4.3, "kv": 2400.0, "pack": "4S 1300"},
          "6S 5x4.3": {"pitch_in": 4.3, "kv": 1750.0, "pack": "6S 1300"},
          "6S 5x5": {"pitch_in": 5.0, "kv": 1750.0, "pack": "6S 1300"}}
# the structure's assumed densities (a foam-core, carbon-skinned mini panel; printed pod; 6 mm carbon tubes)
WING_AREAL_KG_M2 = 0.55
SPAR_KG_M = 0.020
POD_KG_PER_M = 0.16                  # the printed pod per metre of length at ~50 mm width
POSTS_KG = 0.018                     # the two posts with their printed streamlined sleeves and fittings
POST_FAIRING_CHORD = 0.016           # the sleeve's chord [m] (a 5 mm tube inside a 16 × 5.5 mm section)
POST_CD_FAIRED = 0.08                # on the sleeve's chord × length: a 35 % thick streamlined strut at this Reynolds number
FEET_KG = 0.008
NACELLE_KG = 0.006                   # a printed motor mount / nacelle, each
QUAD_FRAME_KG = 0.105                # a 5-inch carbon racing frame with its hardware (the baseline's structure)
PACK_C_CONT = 30.0                   # a racing pack's honest continuous rating (the label says 75–100C burst)
MOTOR_I_CONT_A = 22.0                # a 2207's continuous current without a heat problem
TILT_MAX_DEG = 70.0                  # how far a pilot tilts the thrust in a brake or a turn (the quad and the tailsitter alike)
UNCERTAINTY = {"structure mass": 0.20, "Cd0": 0.30, "cl_max": 0.10, "drive power": 0.15, "quad CdA": 0.30}

# ------------------------------------------------------------------------------------------------- the designs
VARIANTS = {
    "Apus Mini": {"layout": "monoplane", "n_motors": 2, "title": "economy: one wing a third of a gate wide, two of the stack's motors, the hover pitch by the elevons in the slipstream", "drive": "4S 5x4.3",
                  "span": 0.50, "chord": 0.17, "motor_y": 0.14, "cl_max": 1.0, "n_struct": 8.0, "camber": 0.03, "pod_length": 0.26, "pod_width": 0.050, "colour": "#2a9d8f"},
    "Alcotan": {"layout": "cruciform", "title": "the chaser: sustained speed on 6S and the 5x5, the motors at the tips of a +", "drive": "6S 5x5",
                "span": 0.42, "chord": 0.075, "arm": 0.14, "fin_chord": 0.065, "cl_max": 0.95, "n_struct": 10.0, "camber": 0.02, "pod_length": 0.30, "pod_width": 0.050, "colour": "#e76f51"},
    "Gavilan": {"layout": "biplane", "title": "tight spaces: the racing frame's motor square in a short cell, the wing as a brake", "drive": "4S 5x4.3 light",
                "span": 0.31, "chord": 0.100, "gap": 0.155, "cl_max": 1.1, "n_struct": 8.0, "camber": 0.04, "pod_length": 0.22, "pod_width": 0.046, "colour": "#264653"},
}
MAIN = "Apus Mini"
# the question "is the biplane the right cell?": the same briefs on one wing (the cruciform carries the four motors on
# its fins and wing tips; a monoplane with four motors in a row on the wing needs 4 × 142 mm + the pod = 0.62 m of span)
ALTERNATIVES = {
    "Apus as a biplane cell": {"layout": "biplane", "title": "the same area on two wings of 0.085 m chord, a 160 mm gap, faired posts: the first guess before the question", "drive": "4S 5x4.3",
                               "span": 0.50, "chord": 0.085, "gap": 0.16, "cl_max": 1.0, "n_struct": 8.0, "camber": 0.03, "pod_length": 0.26, "pod_width": 0.050, "colour": "#8ab17d"},
    "Apus with the span a gate forbids": {"layout": "cruciform", "title": "the same area at AR 5.9: span 0.71, half a gate — what span would buy if the course allowed it", "drive": "4S 5x4.3",
                                                     "span": 0.71, "chord": 0.12, "arm": 0.14, "fin_chord": 0.065, "cl_max": 1.0, "n_struct": 8.0, "camber": 0.03, "pod_length": 0.26, "pod_width": 0.050, "colour": "#b5c99a"},
    "Apus flat: four motors in a row": {"layout": "monoplane", "n_motors": 4, "title": "one flat wing, the four discs in a row (0.62 m of span needed), the same area at AR 4.9", "drive": "4S 5x4.3",
                                        "span": 0.64, "chord": 0.13, "cl_max": 1.0, "n_struct": 8.0, "camber": 0.03, "pod_length": 0.26, "pod_width": 0.050, "colour": "#f4a261"},
    "Apus cruciform: four motors on wing and fins": {"layout": "cruciform", "title": "the same wing with all four motors at the tips of a +: the full hover authority, two more motors and two fins to carry", "drive": "4S 5x4.3",
                                                     "span": 0.50, "chord": 0.17, "arm": 0.14, "fin_chord": 0.065, "cl_max": 1.0, "n_struct": 8.0, "camber": 0.03, "pod_length": 0.26, "pod_width": 0.050, "colour": "#e9c46a"},
    "Alcotan twin: two motors on one wing": {"layout": "monoplane", "n_motors": 2, "title": "the chaser with two motors: half the hover thrust, the same pack, the 5x5", "drive": "6S 5x5",
                                             "span": 0.42, "chord": 0.075, "motor_y": 0.13, "cl_max": 0.95, "n_struct": 10.0, "camber": 0.02, "pod_length": 0.30, "pod_width": 0.050, "colour": "#f4a261"},
    "X-C mini for the hangar": {"layout": "xwing", "phi_deg": 30.0, "motor_r": 0.15, "title": "Kestrel-X-C's cell on the 5-inch stack, the smallest that clears the discs: panels 0.16 m, a 0.40 m projected span", "drive": "4S 5x4.3 light",
                                "chord": 0.07, "span": 0.0, "cl_max": 1.0, "n_struct": 8.0, "camber": 0.03, "pod_length": 0.22, "pod_width": 0.046, "colour": "#9b5de5"},
    "X-C mini for the race": {"layout": "xwing", "phi_deg": 30.0, "motor_r": 0.22, "title": "the same cell with Apus's span: panels 0.28 m, chord 0.09, 4S 1500", "drive": "4S 5x4.3",
                              "chord": 0.09, "span": 0.0, "cl_max": 1.0, "n_struct": 8.0, "camber": 0.03, "pod_length": 0.26, "pod_width": 0.050, "colour": "#7b2cbf"},
    "Gavilan on one wing (cruciform)": {"layout": "cruciform", "title": "the smallest cruciform that clears the discs (arms 0.10): span 0.36, over the hangar's 0.31", "drive": "4S 5x4.3 light",
                                        "span": 0.36, "chord": 0.17, "arm": 0.10, "fin_chord": 0.065, "cl_max": 1.1, "n_struct": 8.0, "camber": 0.04, "pod_length": 0.22, "pod_width": 0.046, "colour": "#6c757d"},
}


# ------------------------------------------------------------------------------------------------- the drives
def propulsion(key: str):
    """Boreas's propeller + motor + pack for one 5-inch drive (the blade section as prop_lab calibrated it)."""
    from vegeta import boreas
    d = DRIVES[key]
    pk = PACKS[d["pack"]]
    pr = STACK["propeller"]
    dia, pitch = boreas.inches(5.0, d["pitch_in"])
    prop = boreas.Propeller.from_pitch(f"5x{d['pitch_in']:g}x3", dia, pitch, blades=3, chord_root_m=pr["chord_root_m"], chord_max_m=pr["chord_max_m"],
                                       chord_tip_m=pr["chord_tip_m"], mass_kg=pr["mass_kg"], rotor_mass_kg=pr["mass_kg"] + 0.010)
    m = STACK["motor"]
    motor = boreas.Motor(f"2207 {d['kv']:.0f} KV", kv_rpm_per_volt=d["kv"], resistance_ohm=m["R_ohm"], no_load_current_a=m["I0_A"], max_current_a=m["I_max_A"], mass_kg=m["mass_kg"])
    bat = boreas.Battery(d["pack"], cells=pk["cells"], capacity_ah=pk["capacity_ah"], usable_fraction=0.8, mass_kg=pk["mass_kg"])
    return boreas.Propulsion(prop, ks.blade_airfoil(), motor, bat, rho=RHO)


def drive_table(key: str, *, recompute: bool = False) -> ks.DriveTable:
    """The drive's table out to 85 m/s (past the 5x5's pitch speed), cached as a data file (Boreas, ~2 min when missing)."""
    f = DATA / f"mini_raptor_drive_{key.replace(' ', '_').replace('.', 'p')}.json"
    if not recompute and f.exists():
        doc = json.loads(f.read_text())
        if tuple(doc["table"]["V"]) == V_GRID:
            d = doc["table"]
            return ks.DriveTable(**{k: (tuple(map(tuple, v)) if k == "P" else tuple(v) if isinstance(v, list) else v) for k, v in d.items()})
    tab = ks.compute_drive_table.__wrapped__(RHO, V=V_GRID, sys_=propulsion(key))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"key": key, "drive": DRIVES[key], "pack": PACKS[DRIVES[key]["pack"]], "table": asdict(tab)}, indent=1))
    return tab


def energy_wh(key: str) -> float:
    pk = PACKS[DRIVES[key]["pack"]]
    return pk["cells"] * 3.7 * pk["capacity_ah"] * 0.8


def sustained_power_w(key: str, n_motors: int = 4) -> float:
    """The electrical power the pack and the four motors sustain: the lesser of the pack's continuous C rating and
    the motors' continuous current, at the nominal voltage."""
    pk = PACKS[DRIVES[key]["pack"]]
    V = pk["cells"] * 3.7
    return min(PACK_C_CONT * pk["capacity_ah"] * V, n_motors * MOTOR_I_CONT_A * V)


def drive_summary(key: str) -> dict:
    tab = drive_table(key)
    pk = PACKS[DRIVES[key]["pack"]]
    return {"pack": DRIVES[key]["pack"], "pack mass [kg]": pk["mass_kg"], "usable energy [Wh]": energy_wh(key), "propeller": f"5x{DRIVES[key]['pitch_in']:g} three-blade", "KV": DRIVES[key]["kv"],
            "static thrust per motor [N]": tab.T_max[0], "thrust at 20 m/s [N]": float(tab.t_max(20.0)), "thrust at 40 m/s [N]": float(tab.t_max(40.0)), "thrust at 60 m/s [N]": float(tab.t_max(60.0)),
            "power at 1.2 N static [W]": tab.power(1.2, 0.0), "power at 0.5 N, 20 m/s [W]": tab.power(0.5, 20.0), "electrical ceiling per motor [W]": tab.P_cap}


# ------------------------------------------------------------------------------------------------- layout
def layout(variant: str = MAIN) -> dict:
    v = VARIANTS.get(variant) or ALTERNATIVES[variant]
    b, c = v["span"], v["chord"]
    x_le = 0.095 if v["layout"] == "biplane" else 0.11
    L = {"variant": variant, "layout": v["layout"], "title": v["title"], "drive": v["drive"], "pack": DRIVES[v["drive"]]["pack"], "span": b, "chord": c, "R_prop": R_PROP,
         "pod_length": v["pod_length"], "pod_width": v["pod_width"], "x_le": x_le, "x_te": x_le + c, "x_ac": x_le + 0.25 * c, "x_spar": x_le + 0.30 * c, "prop_x": 0.045,
         "cl_max": v["cl_max"], "n_struct": v["n_struct"], "camber": v["camber"], "colour": v["colour"], "span_over_gate": b / GATE_WIDTH, "Re_20": 20 * c / NU,
         "section": f"thin cambered section ({v['camber']:.0%} camber, 10 % thick); the elevons' reflex trims it"}
    if v["layout"] == "biplane":
        gap = v["gap"]
        motor_y = 0.5 * b - R_PROP - DISC_CLEARANCE
        L.update(gap=gap, S_ref=2 * b * c, AR=b ** 2 / (2 * b * c), motor_y=motor_y, motor_z=gap / 2, gap_over_span=gap / b, n_wings=2,
                 disc_gap_span=2 * motor_y - 2 * R_PROP, disc_gap_vertical=gap - 2 * R_PROP, disc_to_pod=motor_y - R_PROP - v["pod_width"] / 2,
                 x_feet=x_le + c + 0.05, x_pod_nose=0.0 if motor_y - R_PROP - v["pod_width"] / 2 >= DISC_CLEARANCE else L["prop_x"] + 0.03,
                 pitch_arm=gap / 2, roll_arm=motor_y, fin_area=0.0)
        if L["x_pod_nose"] > 0:                                   # the discs would cut the pod: the pod starts behind the propeller plane
            L["disc_to_pod"] = math.inf
            L["x_le"], L["x_te"], L["x_ac"], L["x_spar"] = x_le + 0.0, x_le + c, x_le + 0.25 * c, x_le + 0.30 * c
    elif v["layout"] == "cruciform":
        r = v["arm"]
        L.update(arm=r, fin_chord=v["fin_chord"], S_ref=b * c, AR=b / c, motor_y=r, motor_z=r, n_wings=1, fin_area=2 * (r - v["pod_width"] / 2) * v["fin_chord"],
                 disc_gap_span=r * math.sqrt(2) - 2 * R_PROP, disc_gap_vertical=r * math.sqrt(2) - 2 * R_PROP, disc_to_pod=r - R_PROP - v["pod_width"] / 2, x_pod_nose=0.0,
                 x_feet=x_le + c + 0.06, pitch_arm=r, roll_arm=r, gap_over_span=None)
    elif v["layout"] == "xwing":                                  # Kestrel-X-C's cell: four panels at ±phi, the discs at the tips
        phi = math.radians(v["phi_deg"])
        r_m = v["motor_r"]
        r1 = r_m + R_PROP + DISC_CLEARANCE
        r0 = v["pod_width"] / 2
        S_panels = 4 * (r1 - r0) * c
        S_eff = S_panels * math.cos(phi) ** 2                      # the vertical lift of panels at ±phi
        b_proj = 2 * r1 * math.cos(phi)
        L.update(phi=phi, phi_deg=v["phi_deg"], motor_r=r_m, r0=r0, r1=r1, span=b_proj, span_over_gate=b_proj / GATE_WIDTH, height=2 * r1 * math.sin(phi), S_panels=S_panels, S_ref=S_eff,
                 AR=b_proj ** 2 / S_eff, motor_y=r_m * math.cos(phi), motor_z=r_m * math.sin(phi), n_wings=1, fin_area=0.0,
                 disc_gap_span=2 * r_m * math.cos(phi) - 2 * R_PROP, disc_gap_vertical=2 * r_m * math.sin(phi) - 2 * R_PROP, disc_to_pod=r_m - R_PROP - r0, x_pod_nose=0.0,
                 x_feet=x_le + c + 0.06, pitch_arm=r_m * math.sin(phi), roll_arm=r_m * math.cos(phi), gap_over_span=None)
        L["cl_max"] = v["cl_max"] * math.cos(phi)                   # the aircraft's vertical CL max on S_eff: each panel at its own cl max
    else:                                                        # "monoplane": the motors in a row on the one wing, no fins, no posts
        n = v["n_motors"]
        if n == 4:
            y_in = R_PROP + DISC_CLEARANCE / 2 + 0.0                 # the inner pair just clear of each other on the centre line
            y_out = y_in + 2 * R_PROP + DISC_CLEARANCE
            stations = [-y_out, -y_in, y_in, y_out]
            disc_to_pod = y_in - R_PROP - v["pod_width"] / 2
        else:
            y_in = v.get("motor_y", 0.5 * b - R_PROP - DISC_CLEARANCE)
            stations = [-y_in, y_in]
            disc_to_pod = y_in - R_PROP - v["pod_width"] / 2
        L.update(S_ref=b * c, AR=b / c, motor_y=stations[-1], motor_z=0.0, n_wings=1, fin_area=0.0, stations_y=stations, n_motors=n,
                 disc_gap_span=min(b2 - b1 for b1, b2 in zip(stations[:-1], stations[1:])) - 2 * R_PROP, disc_gap_vertical=math.inf,
                 disc_to_pod=disc_to_pod if disc_to_pod >= DISC_CLEARANCE else math.inf, x_pod_nose=0.0 if disc_to_pod >= DISC_CLEARANCE else 0.045 + 0.03,
                 x_feet=x_le + c + 0.06, pitch_arm=0.0, roll_arm=stations[-1], gap_over_span=None)
        if stations[-1] + R_PROP + DISC_CLEARANCE > b / 2:
            L["span_needed"] = 2 * (stations[-1] + R_PROP + DISC_CLEARANCE)
    L.setdefault("n_motors", 4)
    return L


def checks(L: dict) -> list[str]:
    out = []
    if L["disc_gap_span"] < DISC_CLEARANCE:
        out.append(f"the discs on one wing are {L['disc_gap_span'] * 1e3:.0f} mm apart")
    if L["disc_gap_vertical"] < DISC_CLEARANCE:
        out.append(f"the upper and lower discs are {L['disc_gap_vertical'] * 1e3:.0f} mm apart: two 5-inch discs need {(2 * R_PROP + DISC_CLEARANCE) * 1e3:.0f} mm")
    if L["disc_to_pod"] < DISC_CLEARANCE:
        out.append(f"the discs are {L['disc_to_pod'] * 1e3:.0f} mm from the pod")
    if L["span_over_gate"] > 0.5:
        out.append(f"the span is {L['span_over_gate']:.0%} of a racing gate")
    if L.get("span_needed"):
        out.append(f"the discs in a row need {L['span_needed'] * 1e3:.0f} mm of span")
    if L["layout"] == "monoplane":
        out.append("no vertical thrust lever: the hover pitch comes from the elevons in the slipstream only (estimated, not flown)")
    if L["Re_20"] < 8e4:
        out.append(f"the chord Reynolds number at 20 m/s is {L['Re_20']:.0f}: thin sections lose cl max and gain drag below ~100 000")
    return out


# ------------------------------------------------------------------------------------------------- masses
def mass_table(L: dict) -> list[dict]:
    """Every item with its mass and position [m]: the stack's parts, the pack, the structure by its densities."""
    pk = PACKS[L["pack"]]
    rows = []

    def add(item, m, x, y, z, group, note=""):
        rows.append({"item": item, "mass_kg": m, "x": x, "y": y, "z": z, "group": group, "note": note, "rod": None})
    if L["layout"] == "biplane":
        stations = [(sy * L["motor_y"], sz * L["motor_z"]) for sy, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1))]
    elif L["layout"] == "cruciform":
        stations = [(-L["arm"], 0.0), (L["arm"], 0.0), (0.0, L["arm"]), (0.0, -L["arm"])]
    elif L["layout"] == "xwing":
        stations = [(sy * L["motor_y"], sz * L["motor_z"]) for sy, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1))]
    else:
        stations = [(y, 0.0) for y in L["stations_y"]]
    for k, (y, z) in enumerate(stations):
        add(f"motor {k}", STACK["motor"]["mass_kg"], L["prop_x"] + 0.015, y, z, "propulsion", "2207-class")
        add(f"propeller {k}", STACK["propeller"]["mass_kg"], L["prop_x"], y, z, "propulsion", "5-inch tri-blade")
        add(f"nacelle / mount {k}", NACELLE_KG, L["prop_x"] + 0.03, y, z, "structure", "printed")
    if L["layout"] == "biplane":
        for name, z in (("upper wing", L["gap"] / 2), ("lower wing", -L["gap"] / 2)):
            add(name, WING_AREAL_KG_M2 * L["span"] * L["chord"] + SPAR_KG_M * L["span"], L["x_le"] + 0.40 * L["chord"], 0.0, z, "structure", f"{WING_AREAL_KG_M2} kg/m² + a 6 mm tube")
        add("posts and fittings", POSTS_KG, L["x_spar"], 0.0, 0.0, "structure", "two carbon posts at the motor stations")
    elif L["layout"] == "monoplane":
        add("wing", WING_AREAL_KG_M2 * L["span"] * L["chord"] + SPAR_KG_M * L["span"], L["x_le"] + 0.40 * L["chord"], 0.0, 0.0, "structure", f"{WING_AREAL_KG_M2} kg/m² + a 6 mm tube")
    elif L["layout"] == "xwing":
        add("panels (4)", WING_AREAL_KG_M2 * L["S_panels"] + SPAR_KG_M * 4 * (L["r1"] - L["r0"]), L["x_le"] + 0.40 * L["chord"], 0.0, 0.0, "structure", "four cantilever panels, a 6 mm tube each")
    else:
        add("wing", WING_AREAL_KG_M2 * L["span"] * L["chord"] + SPAR_KG_M * L["span"], L["x_le"] + 0.40 * L["chord"], 0.0, 0.0, "structure", f"{WING_AREAL_KG_M2} kg/m² + a 6 mm tube")
        add("fins (2)", WING_AREAL_KG_M2 * L["fin_area"] * 1.1 + SPAR_KG_M * 2 * L["arm"], L["x_le"] + 0.40 * L["fin_chord"], 0.0, 0.0, "structure", "the vertical arms carrying two motors")
    add("pod", POD_KG_PER_M * L["pod_length"] + 0.010, L["x_pod_nose"] + 0.45 * L["pod_length"], 0.0, 0.0, "structure", "printed, with the bays and the hatch")
    add("feet", FEET_KG, L["x_feet"], 0.0, 0.0, "structure", "four tips")
    add("battery", pk["mass_kg"], L["x_ac"] - 0.005, 0.0, -0.008, "energy", L["pack"])
    add("flight controller + ESC", STACK["stack"]["mass_kg"], L["x_ac"] + 0.035, 0.0, 0.012, "systems", "30.5 mm stack")
    add("FPV camera + VTX", STACK["fpv"]["mass_kg"], L["x_pod_nose"] + 0.03, 0.0, 0.0, "systems")
    add("receiver + GNSS", STACK["receiver"]["mass_kg"], L["x_ac"] + 0.08, 0.0, 0.015, "systems")
    add("wiring", STACK["wiring"]["mass_kg"], L["x_ac"], 0.0, 0.0, "systems")
    return rows


def budget(L: dict) -> pd.DataFrame:
    t = mass_table(L)
    df = pd.DataFrame(t)[["item", "mass_kg", "group", "note"]]
    df["mass_g"] = (df["mass_kg"] * 1e3).round(1)
    return df.drop(columns="mass_kg")


def quad_mass_table() -> list[dict]:
    rows = []

    def add(item, m, x, y, z, group, note=""):
        rows.append({"item": item, "mass_kg": m, "x": x, "y": y, "z": z, "group": group, "note": note, "rod": None})
    for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        add(f"motor {k}", STACK["motor"]["mass_kg"], sx * 0.078, sy * 0.078, 0.0, "propulsion")
        add(f"propeller {k}", STACK["propeller"]["mass_kg"], sx * 0.078, sy * 0.078, 0.01, "propulsion")
    add("frame", QUAD_FRAME_KG, 0.0, 0.0, 0.0, "structure", "5-inch carbon frame, arms, hardware")
    add("battery", PACKS["4S 1500"]["mass_kg"], 0.0, 0.0, 0.03, "energy", "4S 1500 on top")
    add("flight controller + ESC", STACK["stack"]["mass_kg"], 0.0, 0.0, 0.0, "systems")
    add("FPV camera + VTX", STACK["fpv"]["mass_kg"], 0.05, 0.0, 0.01, "systems")
    add("receiver", 0.005, -0.03, 0.0, 0.01, "systems")
    add("wiring", STACK["wiring"]["mass_kg"], 0.0, 0.0, 0.0, "systems")
    return rows


# ------------------------------------------------------------------------------------------------- aerodynamics
def drag_buildup(L: dict, *, V: float = 20.0, t: float = 0.10) -> dict:
    """The parts' drag areas [m²] at speed ``V``: the lifting surfaces (flat plate at their chord Reynolds number with
    the thickness form factor), the fins, the pod (its fineness ratio), the four motors in their mounts, the posts, the
    camera; times the interference allowance. ``UNCERTAINTY['Cd0']`` says how far to trust it."""
    S, c = L["S_ref"], L["chord"]
    rows = {}
    ff = 1 + 2 * t + 100 * t ** 4
    rows["wings"] = kf._cf(V * c / NU) * ff * 2 * (L["S_panels"] if L["layout"] == "xwing" else S)
    if L["layout"] == "xwing":
        rows["panel-pod junctions (4, filleted)"] = 4 * 0.0012 * (c / 0.14) ** 2 * 0.5         # Kestrel-X-C's allowance, scaled with the chord
    if L["layout"] == "cruciform":
        rows["fins (2)"] = kf._cf(V * L["fin_chord"] / NU) * ff * 2 * L["fin_area"]
    lp, dp = L["pod_length"], L["pod_width"]
    fr = lp / dp
    rows["pod"] = kf._cf(V * lp / NU) * (1 + 60 / fr ** 3 + fr / 400) * math.pi * dp * lp * 0.8
    dm = STACK["motor"]["d_m"]
    nm = L.get("n_motors", 4)
    rows[f"motors in their mounts ({nm})"] = nm * (0.25 * math.pi * dm ** 2 / 4 + kf._cf(V * 0.05 / NU) * 1.3 * math.pi * dm * 0.05)
    if L["layout"] == "biplane":
        rows["posts (2), faired"] = 2 * POST_CD_FAIRED * POST_FAIRING_CHORD * (L["gap"] - 0.01)      # printed streamlined sleeves on the tubes
    rows["camera and antenna"] = 0.6 * 0.0006
    total = sum(rows.values()) * kf.INTERFERENCE
    return {"parts_m2": rows, "CdA_m2": total, "Cd0": total / S}


def induced_factor(L: dict) -> dict:
    """The induced factor on the lifting area: the biplane's (1 + σ) / 2 of a monoplane of the same span (Prandtl), the
    monoplane's 1 / (π AR); both over the viscous Oswald factor."""
    if L["layout"] == "biplane":
        sig = kf.prandtl_sigma(L["gap_over_span"])
        k = (1 + sig) / 2 / (math.pi * L["AR"]) / kf.OSWALD_VISCOUS
    elif L["layout"] == "xwing":
        sig = None
        k = 1 / (math.pi * L["AR"] * 0.85) / kf.OSWALD_VISCOUS          # the X's lattice gave e ≈ 0.85 on the projected span (kestrel_xc)
    else:
        sig = None
        k = 1 / (math.pi * L["AR"]) / kf.OSWALD_VISCOUS
    return {"sigma": sig, "k": k, "e": 1 / (math.pi * L["AR"] * k)}


# ------------------------------------------------------------------------------------------------- performance
def performance(variant: str = MAIN, *, speeds=np.arange(6.0, 86.0, 1.0), fpv_W: float = FPV_W, cd0_scale: float = 1.0, cl_max_scale: float = 1.0, mass_scale: float = 1.0) -> dict:
    """Level flight and the manoeuvre against speed; the hover; the brake; the reorientation; the transitions. The
    three scales are the sensitivity knobs (the build-up's Cd0, the section's cl max, the structure's mass)."""
    L = dict(layout(variant))
    L["cl_max"] *= cl_max_scale
    tab = drive_table(L["drive"])
    ci = ks.cg_inertia(mass_table(L))
    m = ci["mass_kg"] * mass_scale
    W = m * G
    S = L["S_ref"]
    k = induced_factor(L)["k"]
    E = energy_wh(L["drive"])
    cd90 = kf.viterna_cd90(L["AR"])
    nm = L.get("n_motors", 4)
    rows = []
    for V in speeds:
        q = 0.5 * RHO * V * V
        CL = W / (q * S)
        d = drag_buildup(L, V=V)
        cd0 = d["Cd0"] * cd0_scale
        if CL > L["cl_max"]:
            rows.append({"V [m/s]": V, "feasible": False, "note": "stalled"})
            continue
        CD = cd0 + k * CL ** 2
        D = q * S * CD
        T = nm * float(tab.t_max(V))
        if D > T:
            rows.append({"V [m/s]": V, "feasible": False, "note": "no thrust left"})
            continue
        P = nm * tab.power(D / nm, V) + fpv_W
        n_aero = L["cl_max"] * q * S / W
        n_sus = math.sqrt(max((T / (q * S) - cd0) / k, 0.0)) * q * S / W
        n = min(n_aero, n_sus, L["n_struct"])
        r = V * V / (G * math.sqrt(n * n - 1)) if n > 1.02 else float("inf")
        rows.append({"V [m/s]": V, "feasible": True, "note": "", "CL": CL, "Cd0": cd0, "CD": CD, "L/D": CL / CD, "drag [N]": D, "thrust available [N]": T, "thrust margin": T / D,
                     "electrical [W]": P, "Wh/km": P / (V * 3.6), "endurance [min]": E / P * 60, "range [km]": E / P * V * 3.6, "n wing": n_aero, "n sustained": n_sus, "n": n,
                     "turn radius [m]": r, "turn rate [deg/s]": math.degrees(G * math.sqrt(max(n * n - 1, 0)) / V), "limited by": "wing" if n == n_aero else "thrust" if n == n_sus else "structure"})
    df = pd.DataFrame(rows).set_index("V [m/s]")
    ok = df[df["feasible"]]
    T0 = nm * tab.T_max[0]
    P_h = nm * tab.power(W / nm, 0.0) + fpv_W
    dT = tab.T_max[0] - W / nm                                        # what one motor can add above its hover share
    alpha_pitch = (nm / 2) * dT * L["pitch_arm"] / max(ci["Iyy"], 1e-6)       # one half up, the other down to the idle (0 for a flat monoplane: elevons only)
    alpha_roll = (nm / 2) * dT * L["roll_arm"] / max(ci["Ixx"], 1e-6)
    elev = elevon_pitch_authority(L, W, ci["Iyy"])
    if L["layout"] == "monoplane":
        alpha_pitch = elev["elevon pitch acceleration in the hover [rad/s²]"]
    V_s = math.sqrt(W / (0.5 * RHO * S * L["cl_max"]))
    at = lambda col, V: float(ok.loc[V, col]) if V in ok.index else np.nan
    summary = {"layout": L["layout"] + (f" ({nm} motors)" if nm != 4 else ""), "drive": L["drive"], "mass [kg]": m, "wing area [m²]": S, "span [m]": L["span"], "wing loading [N/m²]": W / S, "aspect ratio": L["AR"],
               "Cd0 at 20 m/s": float(drag_buildup(L)["Cd0"]) * cd0_scale, "CdA at 20 m/s [cm²]": float(drag_buildup(L)["CdA_m2"]) * 1e4 * cd0_scale, "induced k": k, "stall speed [m/s]": V_s,
               "hover thrust / weight": T0 / W, "hover power [W]": P_h, "hover endurance [min]": E / P_h * 60, "pitch acceleration in the hover [rad/s²]": alpha_pitch,
               "roll acceleration in the hover [rad/s²]": alpha_roll, "hover pitch by": "elevons in the wash" if L["layout"] == "monoplane" else "differential thrust (+ elevons)",
               "elevon pitch acceleration in the hover [rad/s²]": elev["elevon pitch acceleration in the hover [rad/s²]"], "power at 20 m/s [W]": at("electrical [W]", 20.0), "Wh/km at 20 m/s": at("Wh/km", 20.0), "L/D at 20 m/s": at("L/D", 20.0),
               "range at 20 m/s [km]": at("range [km]", 20.0), "best range [km]": float(ok["range [km]"].max()), "at [m/s]": float(ok["range [km]"].idxmax()), "least Wh/km": float(ok["Wh/km"].min()),
               "best endurance [min]": float(ok["endurance [min]"].max()), "top speed [m/s]": float(ok.index.max()), "top speed [km/h]": float(ok.index.max()) * 3.6,
               "power at top speed [W]": float(ok["electrical [W]"].iloc[-1]), "full-throttle endurance at top speed [s]": E / float(ok["electrical [W]"].iloc[-1]) * 3600,
               "sustained power [W]": sustained_power_w(L["drive"], nm), "sustained top speed [m/s]": float(ok[ok["electrical [W]"] <= sustained_power_w(L["drive"], nm)].index.max()),
               "sustained top speed [km/h]": float(ok[ok["electrical [W]"] <= sustained_power_w(L["drive"], nm)].index.max()) * 3.6, "n at 20 m/s (banked)": at("n", 20.0), "turn radius at 20 m/s (banked) [m]": at("turn radius [m]", 20.0),
               "turn radius at 40 m/s (banked) [m]": at("turn radius [m]", 40.0), "least banked turn radius [m]": float(ok["turn radius [m]"].min()), "at speed [m/s]": float(ok["turn radius [m]"].idxmin()),
               "span / gate": L["span_over_gate"], "energy [Wh]": E, "cd90": cd90}
    out = {"L": L, "table": df, "summary": summary, "cg": ci, "drive": tab, "mass_table": mass_table(L)}
    summary.update(vertical_climb(L, tab, m, float(drag_buildup(L)["CdA_m2"]) * cd0_scale, n_motors=nm, fpv_W=fpv_W, P_sustain=sustained_power_w(L["drive"], nm)))
    summary["zoom climb from the top speed [m]"] = summary["top speed [m/s]"] ** 2 / (2 * G)
    summary.update(braking(out))
    summary.update(reorientation(out))
    summary.update(transition(out))
    return out


def elevon_pitch_authority(L: dict, W: float, Iyy: float, *, deflection_deg: float = 30.0, cl_per_rad_flap: float = 2.5) -> dict:
    """The hover pitch moment from the elevons in the propellers' wash: the wake's dynamic pressure at the wing (the
    disc's induced speed from momentum theory at the hover thrust, taken at 1.4 × the disc value where the elevon
    sits), the washed strip (one disc width × the chord) per motor, a flap lift slope of 2.5 per radian of deflection on
    the 30 % elevon, the arm from the quarter chord to the elevon's centre of pressure. A first estimate; the biplane
    and the cruciform have it in addition to their differential thrust."""
    nm = L.get("n_motors", 4)
    T_each = W / nm
    A = math.pi * R_PROP ** 2
    v_i = math.sqrt(T_each / (2 * RHO * A))
    q = 0.5 * RHO * (1.4 * v_i) ** 2
    strip = 2 * R_PROP * L["chord"]
    n_washed = len([1 for st in (L.get("stations_y") or []) ]) or (2 if L["layout"] == "cruciform" else 4)     # the discs whose wash crosses a wing's elevon
    dCL = cl_per_rad_flap * math.radians(deflection_deg) * 0.75                                   # a part-span flap's lift on the washed strip
    F = q * strip * dCL
    arm = 0.60 * L["chord"]                                                                       # the elevon's lift behind the CG near the quarter chord
    M = n_washed * F * arm
    return {"elevon pitch moment in the hover [N·m]": M, "elevon pitch acceleration in the hover [rad/s²]": M / max(Iyy, 1e-6), "wash speed at the elevon [m/s]": 1.4 * v_i}


def braking(perf: dict, *, V0: float = 20.0, V_end: float = 4.0, tilt_max_deg: float = TILT_MAX_DEG) -> dict:
    """The tailsitter's brake from ``V0``: it pitches to the vertical and beyond — the thrust carries the weight and,
    tilted back by up to ``tilt_max_deg``, decelerates at g tan θ; the whole aircraft is an air brake too — the wing
    near 90° (Viterna's cd90 on the lifting area, cos² θ of it as the nose goes past the vertical), the fins, the pod,
    the motors. At each speed the pitch that brakes hardest is taken; integrated down to ``V_end`` where the hover
    takes over. No thrust reversal."""
    L, m, tab = perf["L"], perf["summary"]["mass [kg]"], perf["drive"]
    W = m * G
    CdA_wing = perf["summary"]["cd90"] * L["S_ref"] + 1.0 * L["fin_area"]
    nm = L.get("n_motors", 4)
    CdA_rest = 1.1 * math.pi * L["pod_width"] * L["pod_length"] * 0.7 + nm * 0.6 * math.pi * STACK["motor"]["d_m"] ** 2 / 4 + 2 * (L["pitch_arm"] + L["roll_arm"]) * 0.005 * 1.0
    thetas = np.radians(np.linspace(0.0, tilt_max_deg, 15))

    def decel(V):
        q = 0.5 * RHO * V * V
        best = 0.0
        for th in thetas:
            n_need = 1 / math.cos(th)
            if nm * float(tab.t_max(V * math.sin(th))) / W < n_need:
                break
            a = G * math.tan(th) + q * (CdA_wing * math.cos(th) ** 2 + CdA_rest) / m
            best = max(best, a)
        return best
    V, sdist, t, dt = V0, 0.0, 0.0, 0.005
    a0 = decel(V0)
    while V > V_end and t < 30:
        V -= decel(V) * dt
        sdist += V * dt
        t += dt
    return {"brake drag area, wing at 90° [m²]": CdA_wing + CdA_rest, "brake deceleration at 20 m/s [g]": a0 / G, "brake 20→4 m/s distance [m]": sdist, "brake 20→4 m/s time [s]": t,
            "brake drag-only deceleration at 20 m/s [g]": 0.5 * RHO * V0 * V0 * (CdA_wing + CdA_rest) / m / G}


def reorientation(perf: dict, *, angle_deg: float = 90.0, use: float = 0.5) -> dict:
    """Pointing the nose elsewhere in the hover (not a banked turn): a bang-bang rotation through ``angle_deg`` with
    ``use`` of the differential thrust the drives have (the controller keeps the rest for height and disturbances)."""
    th = math.radians(angle_deg)
    ap, ar = perf["summary"]["pitch acceleration in the hover [rad/s²]"] * use, perf["summary"]["roll acceleration in the hover [rad/s²]"] * use
    return {"hover pitch-over 90° [s]": 2 * math.sqrt(th / ap) if ap > 0 else np.nan, "hover roll 90° [s]": 2 * math.sqrt(th / ar)}


def transition(perf: dict) -> dict:
    """The two transitions by their margins: out of the hover, the drives' thrust at 1.2 × the stall speed against the
    weight (a tailsitter with thrust / weight well above 2 climbs out at any pitch); into the hover, the pull-up from
    the economical speed at the load factor available and the radius it needs; the thrust / weight at the stall."""
    L, s, tab = perf["L"], perf["summary"], perf["drive"]
    W = s["mass [kg]"] * G
    nm = L.get("n_motors", 4)
    V_t = 1.2 * s["stall speed [m/s]"]
    ok = perf["table"][perf["table"]["feasible"]]
    V_e = float(max(math.ceil(1.4 * s["stall speed [m/s]"]), s["at [m/s]"]))
    n = float(ok.loc[V_e, "n"]) if V_e in ok.index else np.nan
    return {"transition speed 1.2 Vs [m/s]": V_t, "thrust / weight at 1.2 Vs": nm * float(tab.t_max(V_t)) / W, "thrust / weight at Vs": nm * float(tab.t_max(s["stall speed [m/s]"])) / W,
            "pull-up speed [m/s]": V_e, "pull-up radius into the hover [m]": V_e ** 2 / (G * (n - 1)) if n > 1 else np.inf, "pull-up n": n,
            "transitions": ("both with thrust to spare" if nm * float(tab.t_max(V_t)) / W > 2 else "marginal: the climb-out leans on the wing") + ("; the hover pitch by elevons only" if L["layout"] == "monoplane" else "")}



def vertical_climb(L: dict, tab, m: float, CdA_endon: float, *, n_motors: int = 4, fpv_W: float = FPV_W, P_sustain: float | None = None) -> dict:
    """The escape upwards: the steady vertical speed where the drives' full thrust at that inflow equals the weight
    plus the drag end-on (``CdA_endon``: the airframe seen along its axis — the tailsitter's small nose-on area, the
    quad's whole frame from below), the power there, the vertical speed the sustainable power allows, the initial
    acceleration from the hover, and the zoom climb (the kinetic height from the top speed, less nothing for drag)."""
    W = m * G
    V_steady, P_steady, V_sus = 0.0, np.nan, 0.0
    for V in np.arange(0.0, 60.0, 0.5):
        need = W + 0.5 * RHO * V * V * CdA_endon
        if n_motors * float(tab.t_max(V)) < need:
            break
        V_steady = V
        P_steady = n_motors * tab.power(need / n_motors, V) + fpv_W
        if P_sustain is None or P_steady <= P_sustain:
            V_sus = V
    return {"vertical climb, thrust-limited [m/s]": V_steady, "power in that climb [W]": P_steady, "vertical climb at the sustainable power [m/s]": V_sus,
            "initial vertical acceleration from the hover [g]": n_motors * tab.T_max[0] / W - 1, "end-on drag area [m²]": CdA_endon}


def quad(*, drive: str = "4S 5x4.3", CdA_m2: float = 0.016, speeds=np.arange(2.0, 86.0, 1.0), fpv_W: float = FPV_W, tilt_max_deg: float = TILT_MAX_DEG) -> dict:
    """The racing quad on the same drives and stack (``quad_mass_table``): tilted flight against speed (the body's drag
    area ``CdA_m2`` — a 5-inch frame at racing tilt, ± ``UNCERTAINTY['quad CdA']``), the power through the drives at
    the axial inflow, Wh/km, range and endurance, the top speed where the tilted thrust runs out or the tilt limit
    holds; the banked turn by the thrust-to-weight (n = T / W at the tilt the thrust allows), the brake by the
    backward tilt plus the body's drag, the yaw (torque) and tilt (differential thrust) reorientations."""
    tab = drive_table(drive)
    ci = ks.cg_inertia(quad_mass_table())
    m = ci["mass_kg"]
    W = m * G
    E = energy_wh(drive)
    rows = []
    for V in speeds:
        D = 0.5 * RHO * V * V * CdA_m2
        T = math.hypot(W, D)
        tilt = math.atan2(D, W)
        vin = V * math.sin(tilt)
        if T / 4 > float(tab.t_max(vin)) or math.degrees(tilt) > tilt_max_deg:
            break
        P = 4 * tab.power(T / 4, vin) + fpv_W
        rows.append({"V [m/s]": V, "tilt [deg]": math.degrees(tilt), "thrust [N]": T, "electrical [W]": P, "Wh/km": P / (V * 3.6), "endurance [min]": E / P * 60, "range [km]": E / P * V * 3.6})
    df = pd.DataFrame(rows).set_index("V [m/s]")
    T0 = 4 * tab.T_max[0]
    P_h = 4 * tab.power(W / 4, 0.0) + fpv_W
    dT = tab.T_max[0] - W / 4
    alpha_tilt = 2 * dT * 0.078 / ci["Iyy"]
    kap = float(tab.kap(0.0))
    alpha_yaw = 2 * kap * (W / 4) / ci["Izz"]                           # the diagonal pairs' torque difference with ± the hover thrust
    # the banked turn at speed V: the thrust available at the inflow, tilted: n = T(V) / W, limited by the tilt
    def n_at(V):
        return min(4 * float(tab.t_max(V * 0.7)) / W, 1 / math.cos(math.radians(tilt_max_deg)))
    def r_at(V):
        n = n_at(V)
        return V * V / (G * math.sqrt(n * n - 1))
    # the brake from 20 m/s: tilt back to the limit (a = g tan tilt) plus the body's drag, down to 4 m/s
    V, s, t, dt = 20.0, 0.0, 0.0, 0.005
    a_tilt = G * math.tan(math.radians(tilt_max_deg))
    while V > 4.0 and t < 30:
        a = min(a_tilt, G * math.sqrt(max(n_at(V) ** 2 - 1, 0))) + 0.5 * RHO * V * V * CdA_m2 / m
        V -= a * dt
        s += V * dt
        t += dt
    summary = {"layout": "quad", "drive": drive, "mass [kg]": m, "hover thrust / weight": T0 / W, **vertical_climb({}, tab, m, 2.2 * CdA_m2, fpv_W=fpv_W, P_sustain=sustained_power_w(drive)),
               "zoom climb from the top speed [m]": float(df.index.max()) ** 2 / (2 * G), "hover power [W]": P_h, "hover endurance [min]": E / P_h * 60,
               "pitch acceleration in the hover [rad/s²]": alpha_tilt, "roll acceleration in the hover [rad/s²]": alpha_tilt, "yaw acceleration in the hover [rad/s²]": alpha_yaw,
               "power at 20 m/s [W]": float(df.loc[20.0, "electrical [W]"]), "Wh/km at 20 m/s": float(df.loc[20.0, "Wh/km"]), "range at 20 m/s [km]": float(df.loc[20.0, "range [km]"]),
               "best range [km]": float(df["range [km]"].max()), "at [m/s]": float(df["range [km]"].idxmax()), "least Wh/km": float(df["Wh/km"].min()), "best endurance [min]": float(df["endurance [min]"].max()),
               "top speed [m/s]": float(df.index.max()), "top speed [km/h]": float(df.index.max()) * 3.6, "power at top speed [W]": float(df["electrical [W]"].iloc[-1]),
               "full-throttle endurance at top speed [s]": E / float(df["electrical [W]"].iloc[-1]) * 3600, "sustained power [W]": sustained_power_w(drive),
               "sustained top speed [m/s]": float(df[df["electrical [W]"] <= sustained_power_w(drive)].index.max()), "sustained top speed [km/h]": float(df[df["electrical [W]"] <= sustained_power_w(drive)].index.max()) * 3.6,
               "n at 20 m/s (banked)": n_at(20.0), "turn radius at 20 m/s (banked) [m]": r_at(20.0), "turn radius at 20 m/s, 45° bank [m]": 400 / G, "turn radius at 40 m/s (banked) [m]": r_at(40.0),
               "least banked turn radius [m]": 0.0, "span / gate": 0.25 / GATE_WIDTH, "energy [Wh]": E, "stall speed [m/s]": 0.0,
               "brake deceleration at 20 m/s [g]": (min(a_tilt, G * math.sqrt(n_at(20.0) ** 2 - 1)) + 0.5 * RHO * 400 * CdA_m2 / m) / G, "brake 20→4 m/s distance [m]": s, "brake 20→4 m/s time [s]": t,
               "hover pitch-over 90° [s]": 2 * math.sqrt(math.pi / 2 / (0.5 * alpha_tilt)), "hover yaw 90° [s]": 2 * math.sqrt(math.pi / 2 / (0.5 * alpha_yaw)), "transitions": "none needed"}
    return {"table": df, "summary": summary, "drive": tab, "cg": ci, "mass_table": quad_mass_table(), "CdA_m2": CdA_m2}


COMPARE_KEYS = ["layout", "mass [kg]", "span / gate", "top speed [km/h]", "power at top speed [W]", "full-throttle endurance at top speed [s]", "sustained power [W]", "sustained top speed [km/h]", "power at 20 m/s [W]", "Wh/km at 20 m/s", "least Wh/km", "range at 20 m/s [km]", "best range [km]", "at [m/s]",
                "best endurance [min]", "stall speed [m/s]", "hover thrust / weight", "hover power [W]", "hover endurance [min]", "n at 20 m/s (banked)", "turn radius at 20 m/s (banked) [m]",
                "turn radius at 40 m/s (banked) [m]", "least banked turn radius [m]", "brake deceleration at 20 m/s [g]", "brake 20→4 m/s distance [m]", "brake 20→4 m/s time [s]", "vertical climb, thrust-limited [m/s]", "vertical climb at the sustainable power [m/s]", "power in that climb [W]", "initial vertical acceleration from the hover [g]", "zoom climb from the top speed [m]", "hover pitch-over 90° [s]",
                "pitch acceleration in the hover [rad/s²]", "hover pitch by", "thrust / weight at 1.2 Vs", "pull-up speed [m/s]", "pull-up radius into the hover [m]", "transitions", "energy [Wh]"]


def compare(perf: dict, base: dict) -> pd.DataFrame:
    cols = {"5-inch racing quad": base["summary"]} | {name: p["summary"] for name, p in perf.items()}
    return pd.DataFrame({k: {kk: v.get(kk, np.nan) for kk in COMPARE_KEYS} for k, v in cols.items()})


def design_rows(perf: dict) -> pd.DataFrame:
    keys = ("layout", "title", "drive", "pack", "span", "chord", "S_ref", "AR", "motor_y", "motor_z", "disc_gap_span", "disc_gap_vertical", "disc_to_pod", "span_over_gate", "Re_20", "cl_max", "n_struct", "section")
    return pd.DataFrame({name: {k: p["L"].get(k) for k in keys} | {"rules": "; ".join(checks(p["L"])) or "all hold"} for name, p in perf.items()})


def targets(perf: dict, base: dict) -> pd.DataFrame:
    """Each design against its own brief, honestly: met / not met with the margin."""
    b = base["summary"]
    rows = {}
    s = perf["Apus Mini"]["summary"]
    rows["Apus Mini: less energy at the quad's 20 m/s"] = {"target": "Wh/km at 20 m/s < half the quad's", "value": f"{s['Wh/km at 20 m/s']:.2f} vs {b['Wh/km at 20 m/s']:.2f}", "met": s["Wh/km at 20 m/s"] < 0.5 * b["Wh/km at 20 m/s"]}
    rows["Apus Mini: comparable speed"] = {"target": "sustained top speed ≥ the quad's", "value": f"{s['sustained top speed [km/h]']:.0f} vs {b['sustained top speed [km/h]']:.0f} km/h", "met": s["sustained top speed [km/h]"] >= b["sustained top speed [km/h]"]}
    rows["Apus Mini: hover pitch with elevons only"] = {"target": "a 90° pitch-over in the hover under 0.5 s (the quad: ~0.1 s)", "value": f"{s['hover pitch-over 90° [s]']:.2f} s at {s['pitch acceleration in the hover [rad/s²]']:.0f} rad/s²", "met": s["hover pitch-over 90° [s]"] < 0.5}
    rows["Apus Mini: comparable manoeuvre"] = {"target": "banked turn radius at 20 m/s ≤ the quad's at 45° bank (41 m)", "value": f"{s['turn radius at 20 m/s (banked) [m]']:.0f} m", "met": s["turn radius at 20 m/s (banked) [m]"] <= 41}
    s = perf["Alcotan"]["summary"]
    rows["Alcotan: faster sustained"] = {"target": "sustained top speed (pack 30C, motors 22 A) ≥ 1.25 × the quad's", "value": f"{s['sustained top speed [km/h]']:.0f} vs {b['sustained top speed [km/h]']:.0f} km/h", "met": s["sustained top speed [km/h]"] >= 1.25 * b["sustained top speed [km/h]"]}
    rows["Alcotan: burst speed"] = {"target": "thrust-limited top speed ≥ 1.25 × the quad's", "value": f"{s['top speed [km/h]']:.0f} vs {b['top speed [km/h]']:.0f} km/h ({s['full-throttle endurance at top speed [s]']:.0f} s of pack)", "met": s["top speed [km/h]"] >= 1.25 * b["top speed [km/h]"]}
    rows["Alcotan: a chase at 40 m/s"] = {"target": "endurance at 40 m/s ≥ the quad's at its top speed, with a turn", "value": f"{float(perf['Alcotan']['table'].loc[40.0, 'endurance [min]']):.1f} min, r = {s['turn radius at 40 m/s (banked) [m]']:.0f} m", "met": True}
    s = perf["Gavilan"]["summary"]
    rows["Gavilan: span for a hangar"] = {"target": "span ≤ a fifth of a gate (0.30 m)", "value": f"{s['span [m]']:.2f} m", "met": s["span [m]"] <= 0.31}
    rows["Gavilan: the quad's hover agility"] = {"target": "pitch acceleration in the hover ≥ the quad's", "value": f"{s['pitch acceleration in the hover [rad/s²]']:.0f} vs {b['pitch acceleration in the hover [rad/s²]']:.0f} rad/s²", "met": s["pitch acceleration in the hover [rad/s²]"] >= b["pitch acceleration in the hover [rad/s²]"]}
    rows["Gavilan: brakes harder than the quad"] = {"target": "brake distance 20→4 m/s ≤ the quad's", "value": f"{s['brake 20→4 m/s distance [m]']:.1f} vs {b['brake 20→4 m/s distance [m]']:.1f} m", "met": s["brake 20→4 m/s distance [m]"] <= b["brake 20→4 m/s distance [m]"]}
    rows["Gavilan: flies slower than the quad can"] = {"target": "stall speed ≤ 12 m/s (a hangar's straights)", "value": f"{s['stall speed [m/s]']:.1f} m/s", "met": s["stall speed [m/s]"] <= 12}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------- the CAD, first guess
def cad_params(variant: str = MAIN) -> dict:
    L = layout(variant)
    v = VARIANTS.get(variant) or ALTERNATIVES[variant]
    pk = PACKS[L["pack"]]
    return dict(layout=L["layout"], n_motors=L.get("n_motors", 4), motor_y_inner=(L["stations_y"][len(L["stations_y"]) // 2] if L["layout"] == "monoplane" else 0.0) * 1e3,
                span=L["span"] * 1e3, chord=L["chord"] * 1e3, gap=L.get("gap", 0.0) * 1e3, arm=L.get("arm", 0.0) * 1e3, fin_chord=L.get("fin_chord", 0.0) * 1e3,
                motor_y=L["motor_y"] * 1e3, motor_z=L["motor_z"] * 1e3, wing_le_x=L["x_le"] * 1e3, prop_x=L["prop_x"] * 1e3, pod_length=L["pod_length"] * 1e3, pod_width=L["pod_width"] * 1e3,
                pod_nose_x=L["x_pod_nose"] * 1e3, camber=v["camber"], battery_l=pk["size_m"][0] * 1e3, battery_w=pk["size_m"][1] * 1e3, battery_h=pk["size_m"][2] * 1e3,
                battery_x=(L["x_ac"] - 0.005) * 1e3 - pk["size_m"][0] * 1e3 / 2, x_feet=L["x_feet"] * 1e3)


class MiniRaptor(kestrel.Design):
    """A mini raptor, conceptual: the pod (an elliptical loft), the lifting surfaces (a thin cambered NACA section),
    the posts (biplane) or the fins (cruciform), the four motors in their mounts at the stations, the propeller discs,
    the tip feet, and the component envelopes inside the pod (battery, flight stack, camera, video transmitter).
    ``part``: ``aircraft`` (everything), ``airframe`` (no discs, no envelopes), ``envelopes`` (the components only)."""
    from vegeta.dedalus import Parameter as _P
    parameters = [
        _P("part", "aircraft", choices=("aircraft", "airframe", "envelopes")),
        _P("layout", "biplane", choices=("biplane", "cruciform", "monoplane")), _P("n_motors", 4, "", min=2, max=4), _P("motor_y_inner", 0.0, "mm", min=0),
        _P("span", 500.0, "mm", min=150), _P("chord", 85.0, "mm", min=30), _P("gap", 160.0, "mm", min=0), _P("arm", 140.0, "mm", min=0), _P("fin_chord", 65.0, "mm", min=0),
        _P("motor_y", 170.0, "mm", min=0), _P("motor_z", 80.0, "mm", min=0), _P("wing_le_x", 100.0, "mm", min=0), _P("prop_x", 45.0, "mm", min=0),
        _P("pod_length", 260.0, "mm", min=100), _P("pod_width", 50.0, "mm", min=20), _P("pod_nose_x", 0.0, "mm", min=0),
        _P("thickness", 0.10, "", min=0.06, max=0.2), _P("camber", 0.03, "", min=0, max=0.06), _P("camber_pos", 0.3, "", min=0.1, max=0.9),
        _P("prop_diameter", 127.0, "mm", min=50), _P("motor_diameter", 28.0, "mm", min=10), _P("post_diameter", 5.0, "mm", min=2),
        _P("battery_l", 80.0, "mm", min=0), _P("battery_w", 36.0, "mm", min=0), _P("battery_h", 32.0, "mm", min=0), _P("battery_x", 70.0, "mm", min=0),
        _P("x_feet", 235.0, "mm", min=0), _P("show_props", True, ""),
    ]

    def build(self, p):
        import cadquery as cq
        shape = None
        for name, (solid, _c) in parts(p).items():
            if p["part"] == "airframe" and name in ("propeller discs", "battery", "flight stack", "camera", "video transmitter"):
                continue
            if p["part"] == "envelopes" and name not in ("battery", "flight stack", "camera", "video transmitter"):
                continue
            shape = solid if shape is None else shape.union(solid)
        return shape


COLOURS = {"pod": "#c9d3dc", "wings": "#8fb3c9", "fins": "#8fb3c9", "posts": "#5b6770", "motors": "#37434c", "propeller discs": "#9aa5ae", "feet": "#5b6770",
           "battery": "#e9c46a", "flight stack": "#2a9d8f", "camera": "#e76f51", "video transmitter": "#8338ec"}


def parts(p: dict) -> dict:
    """The named solids of a design (mm): {name: (CadQuery solid, colour)}, in the frame x aft from the nose, y right,
    z up, the pod's axis on x. The same dict the Design unions."""
    import cadquery as cq
    p = {**{q.name: q.default for q in MiniRaptor.parameters}, **p}            # the Design's defaults for what the caller left out
    out = {}
    Lp, w, x0 = p["pod_length"], p["pod_width"], p["pod_nose_x"]
    st = [(0.0, 0.22), (0.10, 0.65), (0.25, 1.0), (0.70, 1.0), (0.88, 0.6), (1.0, 0.25)]
    wp = cq.Workplane("YZ").workplane(offset=x0)
    xp = x0
    for fx, f in st:
        x = x0 + fx * Lp
        wp = wp.workplane(offset=x - xp).ellipse(max(f * w / 2, 1.0), max(f * w / 2 * 1.1, 1.0))
        xp = x
    out["pod"] = (wp.loft(ruled=False, combine=True), COLOURS["pod"])
    up, lo = kestrel.naca4(p["camber"], p["camber_pos"], p["thickness"])

    def section(c, x_le, z0=0.0):
        return [(x_le + c * u, z0 + c * v) for u, v in up] + [(x_le + c * u, z0 + c * v) for u, v in lo[::-1][1:]]
    c, x_le = p["chord"], p["wing_le_x"]
    R = p["prop_diameter"] / 2
    if p["layout"] == "biplane":
        wings = None
        for z in (p["gap"] / 2, -p["gap"] / 2):
            wg = cq.Workplane("XZ").polyline(section(c, x_le, z)).close().extrude(p["span"] / 2, both=True)
            wings = wg if wings is None else wings.union(wg)
        out["wings"] = (wings, COLOURS["wings"])
        posts = None
        for y in (p["motor_y"], -p["motor_y"]):
            ps = cq.Workplane("XY").workplane(offset=-p["gap"] / 2).center(x_le + 0.3 * c, y).circle(p["post_diameter"] / 2).extrude(p["gap"])
            posts = ps if posts is None else posts.union(ps)
        out["posts"] = (posts, COLOURS["posts"])
        stations = [(sy * p["motor_y"], sz * p["motor_z"]) for sy, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1))]
    elif p["layout"] == "monoplane":
        out["wings"] = (cq.Workplane("XZ").polyline(section(c, x_le)).close().extrude(p["span"] / 2, both=True), COLOURS["wings"])
        stations = [(-p["motor_y"], 0.0), (p["motor_y"], 0.0)] if p["n_motors"] == 2 else [(-p["motor_y"], 0.0), (-p["motor_y_inner"], 0.0), (p["motor_y_inner"], 0.0), (p["motor_y"], 0.0)]
    else:
        out["wings"] = (cq.Workplane("XZ").polyline(section(c, x_le)).close().extrude(p["span"] / 2, both=True), COLOURS["wings"])
        fc = p["fin_chord"]
        fin = cq.Workplane("XY").polyline([(x, y) for x, y in section(fc, x_le + 0.25 * c - 0.25 * fc)]).close().extrude(p["arm"] + 10.0, both=True)
        out["fins"] = (fin, COLOURS["fins"])
        stations = [(-p["arm"], 0.0), (p["arm"], 0.0), (0.0, p["arm"]), (0.0, -p["arm"])]
    motors, discs, feet = None, None, None
    rm = p["motor_diameter"] / 2
    for y, z in stations:
        mt = cq.Workplane("YZ").workplane(offset=p["prop_x"] + 5.0).center(y, z).circle(rm).extrude(22.0)
        mount = cq.Workplane("YZ").workplane(offset=p["prop_x"] + 27.0).center(y, z).circle(rm * 0.6).extrude(x_le + 0.35 * c - p["prop_x"] - 27.0 + 10.0)
        mt = mt.union(mount)
        motors = mt if motors is None else motors.union(mt)
        d = cq.Workplane("YZ").workplane(offset=p["prop_x"] - 1.0).center(y, z).circle(R).extrude(2.0)
        discs = d if discs is None else discs.union(d)
        ft = cq.Workplane("YZ").workplane(offset=x_le + c - 5.0).center(y, z).circle(2.0).extrude(p["x_feet"] - (x_le + c - 5.0))
        feet = ft if feet is None else feet.union(ft)
    out["motors"] = (motors, COLOURS["motors"])
    out["feet"] = (feet, COLOURS["feet"])
    out["propeller discs"] = (discs, COLOURS["propeller discs"])
    bl, bw, bh = p["battery_l"], p["battery_w"], p["battery_h"]
    out["battery"] = (cq.Workplane("XY").box(bl, bw, bh).translate((p["battery_x"] + bl / 2, 0, -0.08 * w)), COLOURS["battery"])
    sl, sw, sh = (v * 1e3 for v in STACK["stack"]["size_m"])
    out["flight stack"] = (cq.Workplane("XY").box(sl, sw * 0.8, sh * 0.6).translate((p["battery_x"] + bl + 5.0 + sl / 2, 0, 0.1 * w)), COLOURS["flight stack"])
    cl_, cw, ch = (v * 1e3 for v in STACK["fpv"]["size_m"])
    out["camera"] = (cq.Workplane("XY").box(cl_, cw, ch).translate((x0 + 0.06 * Lp + cl_ / 2, 0, 0)), COLOURS["camera"])
    vl, vw, vh = (v * 1e3 for v in STACK["fpv"]["vtx_size_m"])
    out["video transmitter"] = (cq.Workplane("XY").box(vl, vw, vh).translate((x0 + 0.78 * Lp, 0, 0.05 * w)), COLOURS["video transmitter"])
    return out


def _tess(solid, tol=0.6):
    verts, tris = solid.val().tessellate(tol, 0.3)
    V = np.array([v.toTuple() for v in verts])
    return V[np.array(tris)]


def render3d(prt: dict, ax=None, *, elev: float = 22.0, azim: float = -55.0, title: str = "", extent: float | None = None, labels: bool = True, cutaway: bool = False):
    """A labelled 3-D render of ``parts(...)`` (matplotlib, orthographic, each part in its colour, a legend);
    ``cutaway`` makes the pod and the wings translucent so the component envelopes show."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
    allv = []
    for name, (solid, col) in prt.items():
        F = _tess(solid)
        allv.append(F.reshape(-1, 3))
        alpha = 0.35 if name == "propeller discs" else (0.18 if cutaway and name in ("pod", "wings", "fins") else 1.0)
        ax.add_collection3d(Poly3DCollection(F, facecolors=col, linewidths=0, shade=not cutaway or name not in ("pod", "wings", "fins"), alpha=alpha))
    P = np.vstack(allv)
    ctr = 0.5 * (P.max(0) + P.min(0))
    ext = extent or 0.55 * (P.max(0) - P.min(0)).max()
    ax.set(xlim=(ctr[0] - ext, ctr[0] + ext), ylim=(ctr[1] - ext, ctr[1] + ext), zlim=(ctr[2] - ext, ctr[2] + ext))
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off(); ax.set_proj_type("ortho")
    if labels:
        ax.legend(handles=[Patch(color=c, label=n) for n, (_, c) in prt.items()], loc="lower left", fontsize=8, frameon=False)
    ax.set_title(title)
    return ax


def ortho_views(prt: dict, L: dict, *, figsize=(16, 5.5), title: str = ""):
    """Dimensioned orthographic views: front (y–z), top (x–y), side (x–z) of the parts, with the span, the gap or arm,
    the length, the chord and the disc diameter dimensioned [mm]."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    views = (("front (looking aft): y–z", 1, 2), ("top: x–y", 0, 1), ("side: x–z", 0, 2))
    tess = {n: (_tess(s), c) for n, (s, c) in prt.items()}
    order = ["propeller discs", "wings", "fins", "posts", "motors", "feet", "pod", "battery", "flight stack", "camera", "video transmitter"]
    for ax, (name, a, b) in zip(axes, views):
        for n in order:
            if n not in tess:
                continue
            F, col = tess[n]
            # paint back to front along the third axis
            third = [i for i in (0, 1, 2) if i not in (a, b)][0]
            sign = 1 if name.startswith("front") else -1
            idx = np.argsort(sign * F[:, :, third].mean(1))
            ax.add_collection(PolyCollection(F[idx][:, :, [a, b]], facecolors=col, edgecolors="none", alpha=0.3 if n == "propeller discs" else 0.95))
        ax.set_aspect("equal"); ax.autoscale(); ax.set_title(name); ax.set_xlabel("mm"); ax.grid(alpha=0.2)
    mm = lambda v: v * 1e3

    def dim(ax, p0, p1, text, off=(0, 0), **kw):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color="k", lw=0.9))
        ax.text(0.5 * (p0[0] + p1[0]) + off[0], 0.5 * (p0[1] + p1[1]) + off[1], text, ha="center", va="bottom", fontsize=9, **kw)
    b, c = mm(L["span"]), mm(L["chord"])
    f, t, s = axes
    if L["layout"] == "biplane":
        g = mm(L["gap"])
        dim(f, (-b / 2, g / 2 + 90), (b / 2, g / 2 + 90), f"span {b:.0f}")
        dim(f, (b / 2 + 60, -g / 2), (b / 2 + 60, g / 2), f"gap {g:.0f}", off=(10, 0), rotation=90)
        dim(f, (mm(L["motor_y"]) - 63.5, -g / 2 - 100), (mm(L["motor_y"]) + 63.5, -g / 2 - 100), "disc 127")
        dim(s, (mm(L["x_le"]), g / 2 + 70), (mm(L["x_te"]), g / 2 + 70), f"chord {c:.0f}")
    elif L["layout"] == "cruciform":
        r = mm(L["arm"])
        dim(f, (-b / 2, r + 90), (b / 2, r + 90), f"span {b:.0f}")
        dim(f, (b / 2 + 60, -r), (b / 2 + 60, r), f"fin tips {2 * r:.0f}", off=(10, 0), rotation=90)
        dim(f, (r - 63.5, -r - 100), (r + 63.5, -r - 100), "disc 127")
        dim(s, (mm(L["x_le"]), r + 70), (mm(L["x_te"]), r + 70), f"chord {c:.0f}")
    else:
        y = mm(L["motor_y"])
        dim(f, (-b / 2, 110), (b / 2, 110), f"span {b:.0f}")
        dim(f, (-y, -100), (y, -100), f"motors {2 * y:.0f} apart")
        dim(f, (y - 63.5, -140), (y + 63.5, -140), "disc 127")
        dim(s, (mm(L["x_le"]), 90), (mm(L["x_te"]), 90), f"chord {c:.0f}")
    x0 = mm(L["x_pod_nose"])
    dim(t, (x0, -b / 2 - 70), (x0 + mm(L["pod_length"]), -b / 2 - 70), f"pod {mm(L['pod_length']):.0f}")
    dim(t, (mm(L["prop_x"]), b / 2 + 70), (mm(L["x_feet"]), b / 2 + 70), f"overall {mm(L['x_feet']) - mm(L['prop_x']):.0f}")
    for ax in axes:
        ax.relim(); ax.autoscale_view(); ax.margins(0.15)
    fig.suptitle(title or L["variant"], y=1.02)
    fig.tight_layout()
    return fig


def export_cad(variant: str, *, out_dir=CAD_DIR) -> dict:
    """STEP and STL of the aircraft and the airframe (conceptual geometry) into ``data/cad/mini_raptor/``."""
    from vegeta import dedalus
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    design = dedalus.load_design(str(Path(__file__).resolve()) + ":MiniRaptor")
    tag = variant.lower().replace(" ", "_")
    paths = {}
    g = design.generate(**cad_params(variant), part="aircraft")
    paths["aircraft.step"] = str(g.export_step(out_dir / f"{tag}_aircraft.step"))
    paths["aircraft.stl"] = str(g.export_stl(out_dir / f"{tag}_aircraft.stl", tolerance=0.3))
    g2 = design.generate(**cad_params(variant), part="airframe")
    paths["airframe.stl"] = str(g2.export_stl(out_dir / f"{tag}_airframe.stl", tolerance=0.3))
    bb = g.shape.BoundingBox()
    paths["bbox_mm"] = (round(bb.xlen, 1), round(bb.ylen, 1), round(bb.zlen, 1))
    return paths


def sensitivity(variant: str) -> pd.DataFrame:
    """What the uncertain inputs do to the headline numbers: Cd0 + 30 %, cl max − 10 %, structure + 20 % (as mass + 7 %)."""
    keys = ["power at 20 m/s [W]", "Wh/km at 20 m/s", "best range [km]", "sustained top speed [km/h]", "stall speed [m/s]", "turn radius at 20 m/s (banked) [m]", "hover power [W]"]
    cases = {"as modelled": {}, "Cd0 + 30 %": {"cd0_scale": 1.3}, "cl max − 10 %": {"cl_max_scale": 0.9}, "mass + 7 % (structure + 20 %)": {"mass_scale": 1.07}, "all three": {"cd0_scale": 1.3, "cl_max_scale": 0.9, "mass_scale": 1.07}}
    return pd.DataFrame({c: {k: performance(variant, **kw)["summary"][k] for k in keys} for c, kw in cases.items()})


def export_design(perf: dict, base: dict, cad: dict, path=DATA / "mini_raptor_design.json") -> Path:
    """The notebook's JSON: the layouts, the budgets, the summaries, the comparison, the targets, the CAD paths."""
    def plain(x):
        if isinstance(x, (np.floating, float)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, dict):
            return {str(k): plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        if isinstance(x, pd.DataFrame):
            return {str(k): plain(v) for k, v in x.to_dict(orient="index").items()}
        return x
    doc = {"schema": 1, "notebook": "49_mini_raptor", "method": "first principles and empirical correlations; no CFD, no FEA; conceptual CAD", "uncertainty": UNCERTAINTY,
           "stack": STACK, "packs": PACKS, "drives": DRIVES, "quad baseline": {"summary": base["summary"], "budget": base["mass_table"], "CdA_m2": base["CdA_m2"]},
           "designs": {v: {"layout": p["L"], "rules": checks(p["L"]), "budget": p["mass_table"], "summary": p["summary"], "cad": cad.get(v)} for v, p in perf.items()},
           "comparison": compare(perf, base), "targets": targets(perf, base)}
    path = Path(path)
    path.write_text(json.dumps(plain(doc), indent=1, ensure_ascii=False))
    return path


def biplane_question(perf: dict) -> pd.DataFrame:
    """Apus and Gavilan against the same briefs on one wing (``ALTERNATIVES``): what the biplane cell costs and buys."""
    keys = ["layout", "span [m]", "span / gate", "mass [kg]", "wing area [m²]", "aspect ratio", "induced k", "Cd0 at 20 m/s", "CdA at 20 m/s [cm²]", "L/D at 20 m/s", "power at 20 m/s [W]", "Wh/km at 20 m/s",
            "best range [km]", "stall speed [m/s]", "turn radius at 20 m/s (banked) [m]", "hover thrust / weight", "hover power [W]", "hover pitch by", "pitch acceleration in the hover [rad/s²]", "hover pitch-over 90° [s]", "sustained top speed [km/h]", "top speed [km/h]"]
    cols = {"Apus Mini (twin monoplane, as designed)": perf["Apus Mini"]["summary"], "Alcotan (cruciform, as designed)": perf["Alcotan"]["summary"], "Gavilan (biplane, as designed)": perf["Gavilan"]["summary"]}
    for a in ALTERNATIVES:
        cols[a] = performance(a)["summary"]
    df = pd.DataFrame({k: {kk: v.get(kk) for kk in keys} for k, v in cols.items()})
    names = ("Apus Mini", "Alcotan", "Gavilan", *ALTERNATIVES)
    df.loc["disc rules"] = ["; ".join(checks(layout(v))) or "all hold" for v in names]
    df.loc["parts to build"] = [{"cruciform": "pod, 1 wing, 2 fins, 4 mounts", "biplane": "pod, 2 wings, 2 posts + sleeves, 4 fittings, 4 mounts", "monoplane": f"pod, 1 wing, {layout(v).get('n_motors', 4)} mounts, big elevons", "xwing": "pod, 4 panels, 4 mounts"}[layout(v)["layout"]] for v in names]
    return df
