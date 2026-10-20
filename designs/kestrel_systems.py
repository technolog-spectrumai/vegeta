"""Kestrel-4Q — the parts, the masses, the centre of gravity, the inertia and the propulsion (notebook 40).

- ``COMPONENTS``: the bought parts (motor, propeller, ESC, battery, flight stack, camera) with the values the study
  assumes. They are datasheet-class numbers for parts of this size, not a sourced bill of materials: the propulsion
  is a 2216-class 880 KV motor turning a 9x6 propeller on 4S.
- ``mass_table``: every item with its mass and position (the structure from the layout's lengths and areas and
  assumed areal and linear densities); the battery slides in the pod to put the CG at ``x_cg_target``.
- ``cg_inertia``: the CG and the inertia tensor about it (point masses, the wings and spars as rods along y, the posts
  along z).
- ``propulsion``: Boreas (BEMT, the motor's first-order model and the battery) for one motor; ``drive_table`` turns it
  into the tables the transition model reads at every step — full-throttle thrust, the torque-to-thrust ratio (the
  rotor's reaction torque per newton, the only yaw lever in a hover) and the electrical power over thrust — against the
  axial inflow. Boreas's generic section reads ~25 % low on static thrust (``docs/boreas.md``): the control margins
  here are on the safe side of it.

Units SI (kg, m, N, W). Frame: ``kestrel``'s (x aft from the nose tip, y right, z up in level flight).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import kestrel

G = 9.80665
RHO0 = 1.225

# ------------------------------------------------------------------------------------------------- the bought parts
COMPONENTS = {
    "motor": {"name": "2216-class outrunner, 880 KV", "kv": 880.0, "R_ohm": 0.10, "I0_A": 0.5, "I_max_A": 25.0, "mass_kg": 0.066,
              "source": "datasheet-class values for a 2216 880 KV motor (assumed, not a sourced part)"},
    "propeller": {"name": "9x6 two-blade (APC 9x6E class)", "diameter_in": 9.0, "pitch_in": 6.0, "blades": 2, "mass_kg": 0.014,
                  "chord_root_m": 0.014, "chord_max_m": 0.021, "chord_tip_m": 0.008},
    "esc": {"name": "30 A BLHeli ESC, one per motor, in the nacelle (slipstream cooling)", "mass_kg": 0.014},
    "battery": {"name": "4S 3300 mAh LiPo", "cells": 4, "capacity_ah": 3.3, "mass_kg": 0.360, "length_m": 0.140,
                "width_m": 0.044, "height_m": 0.032, "usable": 0.80},
    "flight_stack": {"name": "flight controller, GNSS + compass, receiver, telemetry radio, power module", "mass_kg": 0.085, "power_W": 4.0},
    "camera": {"name": "compact forward-looking camera with its video link", "mass_kg": 0.045, "power_W": 3.0},
    "mission_computer": {"name": "optional imaging / AI computer (payload slot)", "mass_kg": 0.060, "power_W": 6.0},
    "capsule": {"name": "removable insulated payload capsule (medical supplies: gel, liquids, pills, a small kit)", "shell_kg": 0.090,
                "contents_nominal_kg": 0.300, "contents_max_kg": 0.500, "volume_L": 0.6, "source": "a 72 x 72 x 160 mm vacuum-panel box (assumed)"},
}

# assumed densities of the structure
WING_AREAL_KG_M2 = 0.70          # balsa-composite panel with its skin, ribs and local reinforcements, per m² of planform
SPAR_KG_M = 0.035                # one 10 mm carbon tube per wing
POST_KG_M = 0.030                # 8 x 6 mm carbon tube
DIAGONAL_KG_M = 0.022            # 6 x 4 mm carbon tube
JOINT_KG = 0.006                 # one printed or machined truss joint (12 in the truss)
POD_SHELL_KG = 0.120             # the pod's composite shell with its bulkheads and the battery tray
FIN_KG_M2 = 0.55                 # 4 mm balsa-composite plate
FOOT_ROD_KG_M = 0.020            # 6 mm carbon rod
NACELLE_KG = 0.012               # motor mount, can and fairing, each
SERVO_KG = 0.012                 # one 9 g micro servo with its horn, pushrod and hinge tape (Part II: the elevons)
SLEEVE_KG_M = {"post": 0.006, "diagonal": 0.004}   # a strut's foam-and-film fairing sleeve per metre (Part II)
WIRING_KG = 0.075                # power wiring to four nacelles, signal wiring, connectors
MARGIN_KG = 0.060                # growth margin of a concept

DRIVE_JSON = Path(__file__).resolve().parent / "data" / "kestrel_drive.json"
X_CG_TARGET_FRAC = 0.10          # the CG this fraction of the chord aft of the leading edge (checked against the neutral point)


def _parts(p):
    L = kestrel.Kestrel4Q.layout(p)
    c = COMPONENTS
    x_motor = L["prop_x"] + 0.025
    x_esc = L["x_le"] + 0.02
    items = []

    def add(name, m, x, y, z, group, rod=None):
        items.append({"item": name, "mass_kg": m, "x": x, "y": y, "z": z, "group": group, "rod": rod})

    for k, s in L["motors"].items():
        add(f"motor {k}", c["motor"]["mass_kg"], x_motor, s["y"], s["z"], "propulsion")
        add(f"propeller {k}", c["propeller"]["mass_kg"], L["prop_x"], s["y"], s["z"], "propulsion")
        add(f"ESC {k}", c["esc"]["mass_kg"], x_esc, s["y"], s["z"], "propulsion")
        add(f"nacelle {k}", NACELLE_KG, 0.5 * (L["prop_x"] + L["x_le"] + 0.6 * L["chord"]), s["y"], s["z"], "structure")
    for nm, z in (("upper", L["h"]), ("lower", -L["h"])):
        add(f"wing {nm}", WING_AREAL_KG_M2 * L["S_wing"], L["x_le"] + 0.42 * L["chord"], 0.0, z, "structure", ("y", L["span"]))
        add(f"spar {nm}", SPAR_KG_M * L["span"], L["x_spar"], 0.0, z, "structure", ("y", L["span"]))
    sleeve = bool(p.get("strut_fairing"))
    for s in (-1, 1):
        if L["posts"]:
            add(f"post {'left' if s < 0 else 'right'}", (POST_KG_M + (SLEEVE_KG_M["post"] if sleeve else 0.0)) * L["post_length"],
                L["x_spar"], s * L["motor_y"], 0.0, "structure", ("z", L["gap"]))
        if L["diagonals"]:
            add(f"diagonals {'left' if s < 0 else 'right'}", 2 * (DIAGONAL_KG_M + (SLEEVE_KG_M["diagonal"] if sleeve else 0.0)) * L["diagonal_length"],
                L["x_spar"], s * 0.5 * (L["motor_y"] + L["pod_width"] / 2), 0.0, "structure")
    n_joints = {4: 12, 0: 4 if L["posts"] else 0}[L["diagonals"]]
    if n_joints:
        add("truss joints", n_joints * JOINT_KG, L["x_spar"], 0.0, 0.0, "structure")
    if L["elevons"]:
        y_mid = 0.5 * (L["elevon_y_in"] + L["span"] / 2)
        for nm, z in (("upper", L["h"]), ("lower", -L["h"])):
            for s in (-1, 1):
                add(f"elevon servo {nm} {'L' if s < 0 else 'R'}", SERVO_KG, L["x_hinge"] - 0.02, s * y_mid, z, "systems")
    add("pod shell", POD_SHELL_KG, 0.45 * L["pod_length"], 0.0, 0.0, "structure")
    fin_area = (L["fin_height"] + L["ventral_height"]) * 0.5 * (L["x_feet"] - L["fin_root_x"] + L["fin_tip_chord"]) + L["gap"] * (L["x_feet"] - L["pod_length"])
    z_fin = 0.5 * (L["fin_height"] - L["ventral_height"]) * 0.5
    add("fin plate (dorsal + ventral)", FIN_KG_M2 * fin_area, 0.5 * (L["fin_root_x"] + L["x_feet"]) + 0.02, 0.0, z_fin, "structure")
    add("foot rods (4)", 4 * FOOT_ROD_KG_M * L["foot_rod_length"], L["x_feet"] - 0.5 * L["foot_rod_length"], 0.0, 0.0, "structure")
    add("wiring", WIRING_KG, L["x_le"], 0.0, 0.0, "systems")
    bays = kestrel.Kestrel4Q.bays(p)
    mid = lambda name: (0.5 * (bays[name][0] + bays[name][1]) / 1000, 0.5 * (bays[name][4] + bays[name][5]) / 1000)
    add("flight stack", c["flight_stack"]["mass_kg"], mid("flight stack")[0], 0.0, mid("flight stack")[1], "systems")
    add("camera", c["camera"]["mass_kg"], 0.02, 0.0, 0.0, "payload")
    add("mission computer", c["mission_computer"]["mass_kg"], mid("mission computer")[0], 0.0, mid("mission computer")[1], "payload")
    add("growth margin", MARGIN_KG, L["x_ac"], 0.0, 0.0, "margin")
    return items, L


def battery_range(p=None) -> tuple[float, float]:
    """The battery centre's travel in the pod [m]: under the bays, from the end of the nose taper to the start of the
    tail taper (it slides along the pod's floor)."""
    p = kestrel.resolve(p)
    half = COMPONENTS["battery"]["length_m"] / 2
    return p["nose_length"] / 1000 + half, (p["pod_length"] - p["tail_length"]) / 1000 - half


def mass_table(p=None, *, battery_x: float | None = None, x_cg_target: float | None = None, payload: bool = True,
               capsule: float | None = None) -> list[dict]:
    """Every item with mass and position. The battery sits at ``battery_x``, or where it puts the CG at
    ``x_cg_target`` (default ``X_CG_TARGET_FRAC`` of the chord aft of the leading edge); it is clipped to the pod's
    battery bay, and the row ``battery`` says so when it is (``clipped``). ``payload=False`` takes the mission computer
    out (the camera stays). ``capsule``: the medical capsule's contents [kg] (its shell comes with it; None: no capsule, Part I's
    aircraft; 0.0: the empty capsule) at the bay's centre."""
    p = kestrel.resolve(p)
    items, L = _parts(p)
    if not payload:
        items = [i for i in items if i["item"] != "mission computer"]
    if capsule is not None:
        bay = kestrel.Kestrel4Q.bays(p)["capsule bay"]
        xc, zc = 0.5 * (bay[0] + bay[1]) / 1000, 0.5 * (bay[4] + bay[5]) / 1000
        items.append({"item": "capsule shell", "mass_kg": COMPONENTS["capsule"]["shell_kg"], "x": xc, "y": 0.0, "z": zc, "group": "payload", "rod": None})
        if capsule > 0:
            items.append({"item": "capsule contents (medical supplies)", "mass_kg": float(capsule), "x": xc, "y": 0.0, "z": zc, "group": "payload", "rod": None})
    mb = COMPONENTS["battery"]["mass_kg"]
    lo, hi = battery_range(p)
    clipped = False
    if battery_x is None:
        target = L["x_le"] + X_CG_TARGET_FRAC * L["chord"] if x_cg_target is None else x_cg_target
        m = sum(i["mass_kg"] for i in items)
        mx = sum(i["mass_kg"] * i["x"] for i in items)
        battery_x = (target * (m + mb) - mx) / mb
        if not lo <= battery_x <= hi:
            clipped = True
            battery_x = float(np.clip(battery_x, lo, hi))
    z_batt = 0.5 * (kestrel.Kestrel4Q.bays(p)["battery"][4] + kestrel.Kestrel4Q.bays(p)["battery"][5]) / 1000
    items.append({"item": "battery", "mass_kg": mb, "x": battery_x, "y": 0.0, "z": z_batt, "group": "propulsion", "rod": None,
                  "clipped": clipped})
    return items


def cg_inertia(table) -> dict:
    """Mass, CG and the inertia tensor about the CG [kg m²] (point masses; rods along their axis: m l² / 12), computed
    by Ogma (``og.mass``) from the rows."""
    from ogma import mass as om
    props = om.Budget([om.Item(i["item"], i["mass_kg"], (i["x"], i["y"], i["z"]), om.Rod(*i["rod"]) if i.get("rod") else None)
                       for i in table]).properties()
    m, cg, I = props.mass, props.com, props.inertia
    return {"mass_kg": m, "x_cg": cg[0], "y_cg": cg[1], "z_cg": cg[2], "I": I, "Ixx": I[0, 0], "Iyy": I[1, 1], "Izz": I[2, 2],
            "battery_x": next(i["x"] for i in table if i["item"] == "battery"),
            "battery_clipped": next(i.get("clipped", False) for i in table if i["item"] == "battery")}


def groups(table) -> dict:
    out = {}
    for i in table:
        out[i["group"]] = out.get(i["group"], 0.0) + i["mass_kg"]
    return out


# ------------------------------------------------------------------------------------------------- propulsion
PROP_LAB_JSON = Path(__file__).resolve().parent / "data" / "prop_lab.json"   # notebook 38's calibration of the 9x6 (prop_lab.export)


def prop_lab_calibration() -> dict:
    """prop_lab's measured corrections for the 9x6 (``data/prop_lab.json``): ``cl_scale`` and ``cd_scale`` on the blade
    section, ``cn_factor`` on the propeller normal force; unit scales and ``on_file`` False before the campaign ran."""
    if not PROP_LAB_JSON.exists():
        return {"cl_scale": 1.0, "cd_scale": 1.0, "cn_factor": 1.0, "on_file": False, "meta": {}}
    doc = json.loads(PROP_LAB_JSON.read_text())
    cal = doc.get("calibration", {})
    if "applied" in cal:                                   # prop_lab.export put the fit through its gates
        ap = cal["applied"]
    else:                                                  # an older record: the same gates here, so a bad fit on file is never applied
        sec, nf = cal.get("section_fit") or {}, cal.get("normal_force_fit") or {}
        ap = {}
        cl, cd = cal.get("cl_scale"), cal.get("cd_scale")
        if cl is not None and cd is not None and (sec.get("rms_log_ratio_after") or 1.0) <= 0.15 and 0.7 <= cl <= 1.3 and 0.5 <= cd <= 2.0 and sec.get("n_cases", 0) >= 4:
            ap.update(cl_scale=cl, cd_scale=cd)
        cn = cal.get("cn_factor")
        if cn is not None and nf.get("azimuths_averaged") and 0.3 <= cn <= 3.0:
            ap["cn_factor"] = cn
    return {"cl_scale": ap.get("cl_scale", 1.0), "cd_scale": ap.get("cd_scale", 1.0), "cn_factor": ap.get("cn_factor", 1.0),
            "on_file": bool(ap), "rejected": cal.get("rejected", []), "meta": doc.get("meta", {})}


def blade_airfoil(calibrated: bool = True):
    """Boreas's generic thin cambered section for the 9-inch blade — with prop_lab's ``cl_scale`` / ``cd_scale`` when
    ``calibrated`` and the campaign is on file."""
    from vegeta import boreas
    base = boreas.Airfoil()
    cal = prop_lab_calibration() if calibrated else None
    if not cal or not cal["on_file"]:
        return base
    return boreas.Airfoil(name=f"{base.name}, prop_lab x{cal['cl_scale']:.3f} cl / x{cal['cd_scale']:.3f} cd", cl_alpha=base.cl_alpha * cal["cl_scale"],
                          alpha0_deg=base.alpha0_deg, cl_max=base.cl_max, cd0=base.cd0 * cal["cd_scale"], k=base.k)


def propulsion(rho: float = RHO0):
    """Boreas's propeller + motor + battery for one of the four drives (the blade section as prop_lab calibrated it,
    when its campaign is on file)."""
    from vegeta import boreas
    c = COMPONENTS
    d, pitch = boreas.inches(c["propeller"]["diameter_in"], c["propeller"]["pitch_in"])
    prop = boreas.Propeller.from_pitch(c["propeller"]["name"], d, pitch, blades=c["propeller"]["blades"],
                                       chord_root_m=c["propeller"]["chord_root_m"], chord_max_m=c["propeller"]["chord_max_m"],
                                       chord_tip_m=c["propeller"]["chord_tip_m"], mass_kg=c["propeller"]["mass_kg"],
                                       rotor_mass_kg=c["propeller"]["mass_kg"] + 0.030)
    airfoil = blade_airfoil()
    m = c["motor"]
    motor = boreas.Motor(m["name"], kv_rpm_per_volt=m["kv"], resistance_ohm=m["R_ohm"], no_load_current_a=m["I0_A"],
                         max_current_a=m["I_max_A"], mass_kg=m["mass_kg"])
    b = c["battery"]
    bat = boreas.Battery(b["name"], cells=b["cells"], capacity_ah=b["capacity_ah"], usable_fraction=b["usable"], mass_kg=b["mass_kg"])
    return boreas.Propulsion(prop, airfoil, motor, bat, rho=rho)


from ogma.propulsion import DriveTable  # noqa: E402  the multirotor drive table: Ogma

def _drive_key() -> dict:
    cal = prop_lab_calibration()
    return {**{k: COMPONENTS[k] for k in ("motor", "propeller", "battery")}, "V_grid_max": 36.0,
            "prop_lab": {k: round(cal[k], 4) for k in ("cl_scale", "cd_scale")} if cal["on_file"] else None}


def drive_table(rho: float = RHO0, *, recompute: bool = False, save: bool = False) -> DriveTable:
    """The drive's tables: read from ``data/kestrel_drive.json`` when it was made from the same motor, propeller and
    battery at sea level, else (or with ``recompute``) built with Boreas (``compute_drive_table``, ~40 s); ``save``
    writes the file."""
    if not recompute and rho == RHO0 and DRIVE_JSON.exists():
        doc = json.loads(DRIVE_JSON.read_text())
        if doc.get("components") == json.loads(json.dumps(_drive_key())):
            d = doc["table"]
            return DriveTable(**{k: (tuple(map(tuple, v)) if k == "P" else tuple(v) if isinstance(v, list) else v) for k, v in d.items()})
    tab = compute_drive_table(rho)
    if save:
        DRIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
        DRIVE_JSON.write_text(json.dumps({"components": _drive_key(), "rho": rho, "table": asdict(tab)}, indent=1))
    return tab


def drive_table_for(sys_, key: str, *, rho: float = RHO0, recompute: bool = False) -> DriveTable:
    """A drive table for another propulsion system (``kestrel_fast.propulsion(candidate)``: a 10x7, a 4000 mAh pack),
    cached in ``data/kestrel_drive_<key>.json`` (a cache: rebuilt with Boreas in ~40 s when missing)."""
    f = DRIVE_JSON.with_name(f"kestrel_drive_{key}.json")
    if not recompute and rho == RHO0 and f.exists():
        d = json.loads(f.read_text())["table"]
        return DriveTable(**{k: (tuple(map(tuple, v)) if k == "P" else tuple(v) if isinstance(v, list) else v) for k, v in d.items()})
    tab = compute_drive_table.__wrapped__(rho, sys_=sys_)        # past the cache: a Propulsion is not hashable
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"key": key, "rho": rho, "table": asdict(tab)}, indent=1))
    return tab


@lru_cache(maxsize=4)
def compute_drive_table(rho: float = RHO0, V=(0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 28.0, 32.0, 36.0), sys_=None) -> DriveTable:
    """Boreas's points for one drive (~40 s: BEMT and the motor model at every row), out to the inflow where the 9x6 at
    full throttle has no thrust left (its pitch speed): the thrust-limited maximum speed lives there. ``sys_``:
    another ``boreas.Propulsion`` than the standard drive's."""
    sys_ = sys_ or propulsion(rho)
    t_max, kap = [], []
    t_grid = np.linspace(0.0, 9.0, 19)
    P = []
    for v in V:
        full = sys_.at_throttle(1.0, v)
        if full.current_limited:                      # hold the motor at its current limit: lower the throttle
            lo, hi = 0.3, 1.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if sys_.at_throttle(mid, v).current > sys_.motor.max_current_a:
                    hi = mid
                else:
                    lo = mid
            full = sys_.at_throttle(lo, v)
        tm = max(full.thrust, 0.0)
        t_max.append(tm)
        ref = min(4.0, 0.6 * tm) if tm > 0.5 else None
        if ref:
            pt = sys_.for_thrust(ref, v)
            kap.append(pt.aero.torque / pt.aero.thrust)
        else:
            kap.append(kap[-1] if kap else 0.012)
        row = []
        for t in t_grid:
            if t <= 0.0:
                row.append(0.0)
            elif t <= tm:
                row.append(sys_.for_thrust(float(t), v).electrical_power)
            else:
                row.append(full.electrical_power * (t / tm) ** 1.5 if tm > 0 else float("nan"))
        P.append(row)
    hover = sys_.for_thrust(4.0, 0.0)
    return DriveTable(tuple(V), tuple(t_max), tuple(kap), tuple(t_grid), tuple(map(tuple, P)), diameter=sys_.prop.diameter,
                      rpm_hover=hover.rpm, source="Boreas BEMT (generic section) + first-order motor + 4S battery",
                      P_cap=sys_.battery.voltage * sys_.motor.max_current_a)


def rpm_for(thrust: float, v_axial: float, rho: float = RHO0) -> float:
    """The rpm one propeller turns for ``thrust`` [N] at the axial inflow (Boreas): the CFD's rotor-disk speeds."""
    sys_ = propulsion(rho)
    return float(sys_.for_thrust(max(thrust, 0.3), max(v_axial, 0.0)).rpm)


def hover_point(mass_kg: float, rho: float = RHO0) -> dict:
    """One drive in the hover at a quarter of the weight: rpm, current, electrical power, figure of merit; and the
    thrust-to-weight ratio at full throttle."""
    sys_ = propulsion(rho)
    t = mass_kg * G / 4
    pt = sys_.for_thrust(t, 0.0)
    tab = drive_table(rho)
    return {"thrust_N": t, "rpm": pt.rpm, "current_A": pt.current, "electrical_W": pt.electrical_power, "throttle": pt.throttle,
            "figure_of_merit": pt.aero.figure_of_merit, "T_max_N": tab.T_max[0], "T_W": 4 * tab.T_max[0] / (mass_kg * G),
            "kappa_m": pt.aero.torque / pt.aero.thrust, "total_W": 4 * pt.electrical_power}
