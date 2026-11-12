"""Scaling an ion-ring thruster (``docs/plasma.md``, P4): a parametric family of Halkas — diameter, ring pitch,
the grounds' axial offset, the number of stages, the voltage — with the 1-D corona model *calibrated on the 2-D
solve* (``Calibration``: the current factor and the onset factor fitted on a grid of ``ehd`` solutions), the
masses of the thruster, its supply and a battery, and the two questions a demonstrator must answer:

* **hover**: thrust over the weight of everything that must fly (the thruster, the supply, the battery, a payload);
* **winged flight**: the thrust a glider of lift-to-drag ``ld`` needs is its weight over ``ld``, so the power
  and the masses that go with it — the route MIT's 2018 ion plane took (2.45 kg, 5 m span, ~500 W, 3.2 N).

Physics from P3: thrust ``T = I dz / mu``, thrust per watt ``dz / (mu V)``; the current of a ring family
``I = k C V (V - V0)`` with ``C`` per metre of emitter edge (``corona.Gap``), ``k`` calibrated; the voltage at the
spark margin from ``breakdown``; stages add (each stage's ions are its own; the air speed through the later ones is
ignored — a few m/s against ion drifts of hundreds).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .breakdown import propagation_voltage, surface_flashover
from .corona import Gap
from .gas import Air, DRY
from .supply import HVSupply

G = 9.81


@dataclass(frozen=True)
class Family:
    """A Halka scaled: ``diameter`` [m] of the outer ground ring, ``pitch`` [m] between rings, ``dz`` [m] the
    grounds' axial offset, ``stages`` axial copies, ``hub`` [m] the open bore; the rings alternate ground/emitter
    from the outside in (an emitter between every two grounds)."""

    diameter: float
    pitch: float = 0.014
    dz: float = 0.012
    stages: int = 1
    hub: float = 0.02
    edge_r: float = 0.1e-3
    strip_h: float = 6e-3
    creepage: float = 0.018

    def rings(self) -> list:
        """[(radius, role, z)] of one stage."""
        out, r, role = [], self.diameter / 2, "ground"
        while r > self.hub + self.pitch / 2:
            out.append((r, role, self.dz if role == "ground" else 0.0))
            r -= self.pitch
            role = "emitter" if role == "ground" else "ground"
        if out and out[-1][1] == "emitter":                          # an emitter needs a ground inside it
            out.pop()
        return out

    @property
    def n_emitters(self) -> int:
        return sum(1 for _, role, _ in self.rings() if role == "emitter")

    @property
    def emitter_length(self) -> float:
        return sum(2 * math.pi * r for r, role, _ in self.rings() if role == "emitter")

    @property
    def gap(self) -> float:
        return math.hypot(self.pitch, self.dz)

    @property
    def exit_area(self) -> float:
        return math.pi * ((self.diameter / 2) ** 2 - self.hub ** 2)

    @property
    def stage_length(self) -> float:
        return self.dz + self.strip_h + 0.012


@dataclass
class Calibration:
    """The 1-D model's fitted factors from 2-D solves: ``k_current`` (the collection factor of ``corona.Gap``,
    the 2-D current over the 1-D one at collection 1) and ``k_onset`` (the 2-D Laplace onset over the hyperboloid
    one for the same edge); ``spread`` the factors' relative spread over the grid."""

    k_current: float = 0.6
    k_onset: float = 1.0
    spread: float = 0.0
    points: list = field(default_factory=list)

    @staticmethod
    def fit(points: list) -> "Calibration":
        """``points``: [{"pitch", "dz", "V", "I_2d", "onset_2d", "edge_r", "length"}] from 2-D solves."""
        ks, kos = [], []
        for p in points:
            g = Gap(d=math.hypot(p["pitch"], p["dz"]), r=p["edge_r"], length=p["length"], collection=1.0)
            v0 = g.onset_voltage() * (p["onset_2d"] / g.onset_voltage())
            ko = p["onset_2d"] / g.onset_voltage()
            i1 = g.current_constant() * p["V"] * (p["V"] - v0) if p["V"] > v0 else 0.0
            if i1 > 0 and p["I_2d"] > 0:
                ks.append(p["I_2d"] / i1)
                kos.append(ko)
        k = float(np.exp(np.mean(np.log(ks)))) if ks else 0.6
        spread = float(np.std(np.log(ks))) if len(ks) > 1 else 0.0
        return Calibration(k, float(np.mean(kos)) if kos else 1.0, spread, list(points))


def estimate(f: Family, V: float, cal: Calibration, air: Air = DRY) -> dict:
    """Current, thrust, power and the spark margin of the family at ``V``, every stage alike."""
    g = Gap(d=f.gap, r=f.edge_r, length=f.emitter_length, collection=cal.k_current, axial=f.dz / f.gap)
    v0 = g.onset_voltage(air) * cal.k_onset
    I = 0.0 if V <= v0 else g.current_constant(air) * V * (V - v0)
    T = I * f.dz / air.mobility()
    v_gap, v_surf = propagation_voltage(f.gap, air), surface_flashover(f.creepage, air)
    vb = min(v_gap, v_surf)
    return {"V": V, "onset": v0, "I": I * f.stages, "thrust": T * f.stages, "power": V * I * f.stages,
            "thrust_per_watt": f.dz / (air.mobility() * V), "margin": vb / V, "breakdown": vb,
            "limited_by": "propagation" if v_gap <= v_surf else "surface", "stages": f.stages}


def at_margin(f: Family, cal: Calibration, air: Air = DRY, margin: float = 1.2) -> dict:
    """The estimate at the highest voltage with ``margin`` to the predicted spark (a design margin, not the edge)."""
    e = estimate(f, 1.0, cal, air)
    return estimate(f, e["breakdown"] / margin, cal, air)


# --- masses
BUILDS = {
    # the printed replica (Halka's masses: resin frame, plated resin grounds) and a flight build (carbon/foam frame,
    # 3 mm aluminium tube grounds, the same bronze emitters); g per metre of ring, g per m^2 of frame disc, g per stage
    "replica": dict(ground_g_per_m=120.0, emitter_g_per_m=10.0, frame_g_per_m2=4300.0, stage_g=20.0),
    "flight": dict(ground_g_per_m=25.0, emitter_g_per_m=10.0, frame_g_per_m2=600.0, stage_g=8.0),
}


def thruster_mass(f: Family, build: str = "flight") -> dict:
    b = BUILDS[build]
    rings = f.rings()
    ground_len = sum(2 * math.pi * r for r, role, _ in rings if role == "ground")
    m_rings = (ground_len * b["ground_g_per_m"] + f.emitter_length * b["emitter_g_per_m"]) * f.stages
    m_frame = b["frame_g_per_m2"] * math.pi * (f.diameter / 2) ** 2 + b["stage_g"] * f.stages
    return {"rings": m_rings / 1000, "frame": m_frame / 1000, "total": (m_rings + m_frame) / 1000}


def flight_masses(f: Family, e: dict, *, build: str = "flight", supply: HVSupply | None = None, minutes: float = 2.0,
                  battery_wh_per_kg: float = 150.0, payload: float = 0.0) -> dict:
    """What must fly: the thruster, the supply sized for the power, the battery for ``minutes``, the payload."""
    s = supply or HVSupply(v_set=e["V"])
    p_in = e["power"] / s.efficiency
    m_t = thruster_mass(f, build)["total"]
    m_s = s.mass(e["power"])
    m_b = p_in * minutes / 60 / battery_wh_per_kg
    total = m_t + m_s + m_b + payload
    return {"thruster": m_t, "supply": m_s, "battery": m_b, "payload": payload, "total": total,
            "thrust_to_weight": e["thrust"] / (total * G), "power_in": p_in}


def winged(f: Family, e: dict, masses: dict, ld: float = 10.0) -> dict:
    """A glider of lift-to-drag ``ld`` carrying the lot: the thrust it needs and whether this thruster gives it."""
    need = masses["total"] * G / ld
    return {"thrust_needed": need, "thrust": e["thrust"], "flies": e["thrust"] >= need, "margin": e["thrust"] / need}
