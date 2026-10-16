"""Kestrel-4Q — the parts, the masses, the centre of gravity, the inertia and the propulsion (notebook 38).

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
    add("flight stack", c["flight_stack"]["mass_kg"], 0.60 * L["pod_length"], 0.0, 0.03, "systems")
    add("camera", c["camera"]["mass_kg"], 0.03, 0.0, -0.01, "payload")
    add("mission computer", c["mission_computer"]["mass_kg"], 0.30 * L["pod_length"], 0.0, 0.03, "payload")
    add("growth margin", MARGIN_KG, L["x_ac"], 0.0, 0.0, "margin")
    return items, L


def battery_range(p=None) -> tuple[float, float]:
    """The battery centre's travel in the pod [m]: behind the camera (90 mm) to 60 mm ahead of the pod's tail."""
    L = kestrel.layout(p)
    half = COMPONENTS["battery"]["length_m"] / 2
    return 0.090 + half, L["pod_length"] - 0.060 - half


def mass_table(p=None, *, battery_x: float | None = None, x_cg_target: float | None = None, payload: bool = True) -> list[dict]:
    """Every item with mass and position. The battery sits at ``battery_x``, or where it puts the CG at
    ``x_cg_target`` (default ``X_CG_TARGET_FRAC`` of the chord aft of the leading edge); it is clipped to the pod's
    battery bay, and the row ``battery`` says so when it is (``clipped``). ``payload=False`` takes the mission computer
    out (the camera stays)."""
    p = kestrel.resolve(p)
    items, L = _parts(p)
    if not payload:
        items = [i for i in items if i["item"] != "mission computer"]
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
    items.append({"item": "battery", "mass_kg": mb, "x": battery_x, "y": 0.0, "z": 0.0, "group": "propulsion", "rod": None,
                  "clipped": clipped})
    return items


def cg_inertia(table) -> dict:
    """Mass, CG and the inertia tensor about the CG [kg m²] (point masses; rods along their axis: m l² / 12)."""
    m = sum(i["mass_kg"] for i in table)
    r = np.array([[i["x"], i["y"], i["z"]] for i in table])
    w = np.array([i["mass_kg"] for i in table])
    cg = (w[:, None] * r).sum(0) / m
    d = r - cg
    I = np.zeros((3, 3))
    for wi, di, it in zip(w, d, table):
        I += wi * (np.dot(di, di) * np.eye(3) - np.outer(di, di))
        if it.get("rod"):
            ax, ln = it["rod"]
            k = wi * ln ** 2 / 12
            I += k * (np.eye(3) - np.diag([ax == "x", ax == "y", ax == "z"]).astype(float))
    return {"mass_kg": m, "x_cg": cg[0], "y_cg": cg[1], "z_cg": cg[2], "I": I, "Ixx": I[0, 0], "Iyy": I[1, 1], "Izz": I[2, 2],
            "battery_x": next(i["x"] for i in table if i["item"] == "battery"),
            "battery_clipped": next(i.get("clipped", False) for i in table if i["item"] == "battery")}


def groups(table) -> dict:
    out = {}
    for i in table:
        out[i["group"]] = out.get(i["group"], 0.0) + i["mass_kg"]
    return out


# ------------------------------------------------------------------------------------------------- propulsion
def propulsion(rho: float = RHO0):
    """Boreas's propeller + motor + battery for one of the four drives."""
    from vegeta import boreas
    c = COMPONENTS
    d, pitch = boreas.inches(c["propeller"]["diameter_in"], c["propeller"]["pitch_in"])
    prop = boreas.Propeller.from_pitch(c["propeller"]["name"], d, pitch, blades=c["propeller"]["blades"],
                                       chord_root_m=c["propeller"]["chord_root_m"], chord_max_m=c["propeller"]["chord_max_m"],
                                       chord_tip_m=c["propeller"]["chord_tip_m"], mass_kg=c["propeller"]["mass_kg"],
                                       rotor_mass_kg=c["propeller"]["mass_kg"] + 0.030)
    airfoil = boreas.Airfoil()
    m = c["motor"]
    motor = boreas.Motor(m["name"], kv_rpm_per_volt=m["kv"], resistance_ohm=m["R_ohm"], no_load_current_a=m["I0_A"],
                         max_current_a=m["I_max_A"], mass_kg=m["mass_kg"])
    b = c["battery"]
    bat = boreas.Battery(b["name"], cells=b["cells"], capacity_ah=b["capacity_ah"], usable_fraction=b["usable"], mass_kg=b["mass_kg"])
    return boreas.Propulsion(prop, airfoil, motor, bat, rho=rho)


@dataclass(frozen=True)
class DriveTable:
    """One drive tabulated against the axial inflow ``V`` [m/s] (negative inflow is read as 0, beyond the table as its
    last row): full-throttle thrust ``T_max`` [N] (current-limited rows held at the motor's current limit), the
    torque-to-thrust ratio ``kappa`` [m] (reaction torque per newton of thrust, near the hover thrust), and the
    electrical power over a thrust grid ``P[i, j]`` at ``V[i]``, ``T[j]``. ``T_min`` is the least thrust the controller
    may command (the motor kept spinning: an ESC's idle)."""
    V: tuple
    T_max: tuple
    kappa: tuple
    T: tuple
    P: tuple
    T_min: float = 0.25
    diameter: float = 0.2286
    rpm_hover: float = 0.0
    source: str = ""

    def t_max(self, v_axial):
        return np.interp(np.maximum(v_axial, 0.0), self.V, self.T_max)

    def kap(self, v_axial):
        return np.interp(np.maximum(v_axial, 0.0), self.V, self.kappa)

    def power(self, thrust, v_axial):
        """Electrical power [W] of one drive at ``thrust`` [N] and axial inflow (bilinear in the table)."""
        V, T, P = np.asarray(self.V), np.asarray(self.T), np.asarray(self.P)
        v = float(np.clip(v_axial, V[0], V[-1]))
        i = min(int(np.searchsorted(V, v, side="right")) - 1, len(V) - 2)
        a = (v - V[i]) / (V[i + 1] - V[i])
        row = (1 - a) * P[i] + a * P[i + 1]
        return float(np.interp(max(float(thrust), 0.0), T, row))


def _drive_key() -> dict:
    return {k: COMPONENTS[k] for k in ("motor", "propeller", "battery")}


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


@lru_cache(maxsize=4)
def compute_drive_table(rho: float = RHO0, V=(0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0)) -> DriveTable:
    """Boreas's points for one drive (~40 s: BEMT and the motor model at every row)."""
    sys_ = propulsion(rho)
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
                      rpm_hover=hover.rpm, source="Boreas BEMT (generic section) + first-order motor + 4S battery")


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
