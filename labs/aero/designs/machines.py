"""The machines this lab exports (``tools/export.py``): name → the Chiron robot builder, and the part data a simulator
needs that MJCF cannot hold (``vegeta.labkit.machines``)."""
from __future__ import annotations

import math

import numpy as np

from vegeta.labkit import machines


def _drongo_parts(m) -> dict:
    """Drongo's rotors for a multicopter simulator: positions in the machine frame, spin (+1: drag torque +z on the
    frame), the propeller's mass and disk inertia, the motor model's constants. ω_max: 80 % of the no-load speed
    (KV × V) of the 2306 2400 KV on 4S (assumed); kF = T_max / ω_max²; kM = drag torque / thrust; τ = spin-up."""
    import drongo_robot as dr
    from yax import hub
    prop = dr.propulsion()
    pos, spin = dr.rotor_layout()
    g = dr.geometry()
    root = m.roots()[0]
    Rr = hub.R(root.quat)
    w_max = 0.8 * 2400.0 * 4 * 3.7 * 2 * math.pi / 60
    kg = dr.PARTS_G["propellers 5x4.3 (4x)"] / 4 / 1000.0
    r = g["prop_r"]
    rotors = []
    for p, s in zip(np.asarray(pos, float), np.asarray(spin, float)):
        rotors.append({"pos": (np.asarray(root.pos) + Rr @ p).tolist(), "spin": int(s), "kg": kg,
                       "inertia": [kg * r * r / 4, kg * r * r / 4, kg * r * r / 2], "radius": r,
                       "kF": prop.max_thrust_N / w_max ** 2, "kM": prop.torque_per_thrust_m, "w_max": w_max,
                       "tau_up": prop.spin_up_s, "tau_down": 2 * prop.spin_up_s, "T_max": prop.max_thrust_N})
    return {"rotors": rotors, "source": f"{prop.source}; ω_max = 0.8 × KV × V (2400 KV, 4S; assumed)"}


machines.register("drongo", "drongo_robot:drongo", parts=_drongo_parts)
machines.register("falco", "falco_robot:falco_robot")
machines.register("nisus", "nisus_robot:nisus_robot")
machines.register("nisus_plus", "nisus_plus_robot:nisus_plus_robot")
