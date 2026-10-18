"""The reference racer: a conventional 5-inch FPV racing quad — a carbon X-frame, four 2207-class motors, 5x4.3
three-blade propellers, a 4-in-1 ESC, a flight controller, an FPV camera and video transmitter, a 6S 1300 mAh pack.

The all-up mass comes from the component budget (``BUDGET``); the motors are 1750 KV, the voltage 6S (22.2 V nominal,
25.2 V full) — a 2400 KV motor is a 4S motor and would over-speed the 5x4.3 on 6S. The propulsion is the existing
Boreas blade-element table for this exact drive (``notebooks/designs/data/mini_raptor_drive_6S_5x4p3.json``: thrust
limit and electrical power against axial inflow, made by ``mini_raptor.drive_table``); when the file is missing, an
analytic fallback of the same shape is used and said so. The MJCF body (``mjcf_body``) carries the frame, the arms,
the motors and the propeller discs as collision geometry, an onboard FPV camera tilted up and a chase camera.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

G, RHO = 9.81, 1.225
ROOT = Path(__file__).resolve().parents[1]
DRIVE_JSON = ROOT / "notebooks" / "designs" / "data" / "mini_raptor_drive_6S_5x4p3.json"

#: the component budget [g], realistic catalogue values
BUDGET = {
    "motors 2207 1750 KV (4)": 4 * 36.0, "propellers 5x4.3x3 (4)": 4 * 4.5, "4-in-1 ESC 45 A": 15.0, "flight controller": 8.0,
    "FPV camera": 9.0, "video transmitter": 10.0, "antenna": 5.0, "receiver": 2.0, "carbon X-frame 5 inch (220 mm)": 105.0,
    "hardware, wiring, straps, capacitor": 25.0, "battery 6S 1300 mAh": 230.0,
}
MASS_KG = sum(BUDGET.values()) / 1e3
MOTOR = {"name": "2207 1750 KV", "kv": 1750.0, "cells": 6, "v_nominal": 22.2, "v_full": 25.2, "I_max_A": 42.0, "I_cont_A": 22.0}
PACK = {"cells": 6, "capacity_ah": 1.3, "usable": 0.8, "C_cont": 30.0, "mass_kg": 0.230}
PROP = {"diameter_m": 0.127, "pitch_in": 4.3, "blades": 3}
FRAME = {"diagonal_m": 0.220, "arm_r_m": 0.110, "plate_half_m": (0.075, 0.03, 0.002), "motor_r_m": 0.014, "motor_h_m": 0.032, "height_m": 0.035}
CDA_FORWARD_M2 = 0.016              # the frame at racing tilt, ±30 %
CDA_FLAT_M2 = 0.035                 # the frame from below
FPV_W = 6.0
SPIN_UP_S = 0.035
ENERGY_WH = PACK["cells"] * 3.7 * PACK["capacity_ah"] * PACK["usable"]
P_SUSTAIN_W = min(PACK["C_cont"] * PACK["capacity_ah"] * MOTOR["v_nominal"], 4 * MOTOR["I_cont_A"] * MOTOR["v_nominal"])


def rotor_layout(arm_r: float = FRAME["arm_r_m"]) -> tuple:
    """Rotor positions (4, 3) in the body frame (x forward between two arms, z up) and their spin signs."""
    pos = np.array([(arm_r * math.cos(math.radians(a)), arm_r * math.sin(math.radians(a)), FRAME["height_m"]) for a in (45, 135, 225, 315)])
    spin = np.array([1.0, -1.0, 1.0, -1.0])
    return pos, spin


@dataclass
class Drive:
    """One drive against the axial inflow ``V`` [m/s]: full-throttle thrust ``T_max`` [N], torque per newton
    ``kappa`` [m], the electrical power ``P[i, j]`` at ``V[i]``, thrust ``T[j]``; ``P_cap`` the motor's ceiling."""
    V: np.ndarray
    T_max: np.ndarray
    kappa: np.ndarray
    T: np.ndarray
    P: np.ndarray
    P_cap: float
    source: str = ""

    def t_max(self, v):
        return float(np.interp(max(float(v), 0.0), self.V, self.T_max))

    def kap(self, v=0.0):
        return float(np.interp(max(float(v), 0.0), self.V, self.kappa))

    def power(self, thrust, v=0.0):
        v = float(np.clip(v, self.V[0], self.V[-1]))
        i = min(int(np.searchsorted(self.V, v, side="right")) - 1, len(self.V) - 2)
        a = (v - self.V[i]) / (self.V[i + 1] - self.V[i])
        row = np.minimum((1 - a) * self.P[i] + a * self.P[i + 1], self.P_cap)
        return float(min(np.interp(max(float(thrust), 0.0), self.T, row), self.P_cap))


def drive(path: Path = DRIVE_JSON) -> Drive:
    """The Boreas table for the 6S 5x4.3 2207 drive, or the analytic fallback."""
    if Path(path).exists():
        t = json.loads(Path(path).read_text())["table"]
        return Drive(np.array(t["V"]), np.array(t["T_max"]), np.array(t["kappa"]), np.array(t["T"]), np.array(t["P"]), float(t["P_cap"]), f"Boreas table {Path(path).name}")
    V = np.arange(0.0, 86.0, 5.0)
    T_max = 18.4 * np.clip(1 - V / 72.0, 0.0, 1.0) ** 1.2
    T = np.linspace(0.0, 9.0, 19)
    P = np.array([[0.0 if tt == 0 else 1.35 * tt ** 1.5 * (1 + 0.02 * v) * (22.2 / 10.0) / 2.2 for tt in T] for v in V])
    return Drive(V, T_max, np.full_like(V, 0.011), T, P, 932.0, "analytic fallback (no Boreas table on disk)")


@dataclass
class Propulsion:
    """What Drongo's ``Flight`` controller and the rotor hook read: the static maximum thrust, the spin-up time
    constant, the torque per thrust, and the power of one rotor at a thrust (at the given inflow)."""
    drive: Drive
    max_thrust_N: float
    spin_up_s: float = SPIN_UP_S
    torque_per_thrust_m: float = 0.011
    battery_Wh: float = PACK["cells"] * 3.7 * PACK["capacity_ah"]
    usable_fraction: float = PACK["usable"]
    source: str = ""

    def electrical_power(self, thrust, v_axial: float = 0.0):
        T = np.atleast_1d(np.asarray(thrust, dtype=float))
        return np.array([self.drive.power(t, v_axial) for t in T])

    @property
    def usable_Wh(self) -> float:
        return self.battery_Wh * self.usable_fraction


def propulsion() -> Propulsion:
    d = drive()
    return Propulsion(d, d.t_max(0.0), SPIN_UP_S, d.kap(0.0), source=d.source)


def summary() -> dict:
    p = propulsion()
    W = MASS_KG * G
    return {"all-up mass [kg]": MASS_KG, "motor": MOTOR["name"], "voltage": f"{MOTOR['cells']}S ({MOTOR['v_nominal']} V nominal)", "propeller": "5x4.3 three-blade", "pack": "6S 1300 mAh",
            "static thrust per motor [N]": p.max_thrust_N, "hover thrust / weight": 4 * p.max_thrust_N / W, "hover power [W]": float(4 * p.electrical_power(W / 4)[0] + FPV_W),
            "usable energy [Wh]": ENERGY_WH, "sustainable power [W]": P_SUSTAIN_W, "forward drag area [m²]": CDA_FORWARD_M2, "propulsion source": p.source}


def mjcf_body(name: str = "quad", pos=(0.0, 0.0, 0.1)) -> str:
    """The quad as an MJCF body with a free joint: collision geometry (plate, arms, motors, propeller discs), an
    onboard FPV camera tilted 30° up, a chase camera that tracks the body."""
    pos_r, _ = rotor_layout()
    px, py, pz = FRAME["plate_half_m"]
    arm_len = FRAME["arm_r_m"]
    parts = [f'<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">', '  <freejoint name="quad_free"/>',
             f'  <geom name="plate" type="box" size="{px} {py} {pz}" pos="0 0 0.01" rgba="0.15 0.15 0.17 1" mass="{(BUDGET["carbon X-frame 5 inch (220 mm)"] * 0.35 + BUDGET["4-in-1 ESC 45 A"] + BUDGET["flight controller"] + BUDGET["hardware, wiring, straps, capacitor"]) / 1e3:.4f}"/>',
             f'  <geom name="battery" type="box" size="0.041 0.0205 0.0245" pos="0 0 {0.012 + 0.0245}" rgba="0.95 0.75 0.1 1" mass="{PACK["mass_kg"]:.4f}"/>',
             f'  <geom name="camera_housing" type="box" size="0.012 0.012 0.012" pos="{px + 0.012} 0 0.03" rgba="0.1 0.1 0.1 1" mass="{(BUDGET["FPV camera"] + BUDGET["video transmitter"] + BUDGET["antenna"] + BUDGET["receiver"]) / 1e3:.4f}"/>']
    m_arm = BUDGET["carbon X-frame 5 inch (220 mm)"] * 0.65 / 4 / 1e3
    for k, (x, y, z) in enumerate(pos_r):
        a = math.radians(45 + 90 * k)
        parts.append(f'  <geom name="arm{k}" type="box" size="{arm_len / 2:.4f} 0.008 0.003" pos="{x / 2:.4f} {y / 2:.4f} 0.012" quat="{math.cos(a / 2):.5f} 0 0 {math.sin(a / 2):.5f}" rgba="0.15 0.15 0.17 1" mass="{m_arm:.4f}"/>')
        parts.append(f'  <geom name="motor{k}" type="cylinder" size="{FRAME["motor_r_m"]} {FRAME["motor_h_m"] / 2}" pos="{x:.4f} {y:.4f} {0.015 + FRAME["motor_h_m"] / 2:.4f}" rgba="0.08 0.08 0.09 1" mass="{BUDGET["motors 2207 1750 KV (4)"] / 4 / 1e3:.4f}"/>')
        parts.append(f'  <geom name="prop{k}" type="cylinder" size="{PROP["diameter_m"] / 2} 0.001" pos="{x:.4f} {y:.4f} {z:.4f}" rgba="0.55 0.75 0.95 0.35" mass="{BUDGET["propellers 5x4.3x3 (4)"] / 4 / 1e3:.4f}" contype="1" conaffinity="1"/>')
    parts.append(f'  <camera name="onboard" pos="{px + 0.02} 0 0.04" xyaxes="0 -1 0 0.5 0 0.866" fovy="110"/>')      # looking forward (+x), tilted 30° up
    parts.append('  <camera name="chase" mode="track" pos="-3.0 0 1.2" xyaxes="0 -1 0 0.37 0 0.93" fovy="70"/>')
    parts.append('  <site name="com" pos="0 0 0.02" size="0.005"/>')
    parts.append('</body>')
    return "\n".join(parts)
