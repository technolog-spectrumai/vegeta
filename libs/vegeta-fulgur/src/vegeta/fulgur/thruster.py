"""Ring thrusters in the 1-D model: concentric emitter and ground rings, each emitter facing one or two grounds
across a gap; the presets are Plasma Channel's Mark IV, 4B and 4C with their geometry *estimated* from the video's
screenshots (``docs/plasma.md``: every number marked estimated until measured).

``RingThruster.estimate(V, air)`` gives current, thrust, power, mean exit velocity, the velocity at the focal point
(the convergence gathers the jet: the anemometer there reads ``focal_gain`` times the mean, 1.4 estimated — the
2-D jet model replaces it) and the spark margin; ``at_power`` and ``at_thrust`` find the voltage that gives a
measured power or thrust, so the video's numbers can be checked against each other through the model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .breakdown import SparkMargin, spark_margin
from .corona import Gap, jet_velocity, thrust_from_velocity
from .gas import Air, DRY


@dataclass(frozen=True)
class Ring:
    radius: float                 # m, the ring's mean radius
    role: str                     # "emitter" | "ground"
    z: float = 0.0                # m, axial position (+ downstream): the convergence is the grounds sitting behind
    tip_radius: float = 50e-6     # m, the emitter's tooth tip (serrated bronze ~ 30-80 um); ground: the section/2
    edges: int = 1                # how many sharp edges face a ground (a serrated strip: 1; a loop with teeth: 1)


@dataclass(frozen=True)
class RingThruster:
    name: str
    rings: tuple
    hub_radius: float             # m, the open bore at the centre (0 for none)
    outer_radius: float           # m, the frame's inner wall: the exit area's edge
    creepage: float               # m, the shortest resin path between an emitter and a ground
    collection: float = 0.6       # the 1-D model's geometry factor (see corona.Gap)
    focal_gain: float = 1.4       # velocity at the focal point over the mean exit velocity (estimate)
    mass_kg: float = 0.0          # the video's scale reading
    notes: str = ""
    estimated: tuple = field(default_factory=tuple)

    @property
    def exit_area(self) -> float:
        return math.pi * (self.outer_radius ** 2 - self.hub_radius ** 2)

    def gaps(self) -> list[Gap]:
        """Each emitter's gap to its nearest ground (radial and axial distance combined), one ``Gap`` per pair."""
        out = []
        grounds = [r for r in self.rings if r.role == "ground"]
        for e in self.rings:
            if e.role != "emitter":
                continue
            near = sorted(grounds, key=lambda g: math.hypot(g.radius - e.radius, g.z - e.z))[:2]
            for g in near:
                d = math.hypot(g.radius - e.radius, g.z - e.z) - g.tip_radius
                out.append(Gap(d=d, r=e.tip_radius, length=2 * math.pi * e.radius * e.edges,
                               collection=self.collection / len(near)))
        return out

    def estimate(self, V: float, air: Air = DRY) -> dict:
        gaps = self.gaps()
        I = sum(g.current(V, air) for g in gaps)
        T = sum(g.thrust(V, air) for g in gaps)
        d_min = min(g.d for g in gaps)
        r_min = min(g.r for g in gaps)
        sm = spark_margin(V, r_min, d_min, air, self.creepage)
        return {"name": self.name, "V": V, "I": I, "thrust": T, "power": V * I, "thrust_per_watt": T / (V * I) if I else 0.0,
                "v_exit": jet_velocity(T, self.exit_area, air),
                "v_focal": self.focal_gain * jet_velocity(T, self.exit_area, air),
                "onset": min(g.onset_voltage(air) for g in gaps),
                "margin": sm.margin, "limited_by": sm.limited_by, "breakdown": sm.breakdown, "gaps": len(gaps),
                "air": air}

    def _solve(self, key: str, target: float, air: Air) -> dict:
        lo, hi = 0.0, 2.0e5
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self.estimate(mid, air)[key] < target:
                lo = mid
            else:
                hi = mid
        return self.estimate(0.5 * (lo + hi), air)

    def at_power(self, power: float, air: Air = DRY) -> dict:
        """The estimate at the voltage that draws ``power`` [W]."""
        return self._solve("power", power, air)

    def at_thrust(self, thrust: float, air: Air = DRY) -> dict:
        return self._solve("thrust", thrust, air)

    def at_margin(self, margin: float = 1.0, air: Air = DRY) -> dict:
        """The estimate at the highest voltage with at least ``margin`` to the predicted spark."""
        lo, hi = 0.0, 2.0e5
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self.estimate(mid, air)["margin"] > margin:
                lo = mid
            else:
                hi = mid
        return self.estimate(0.5 * (lo + hi), air)


def _rings(radii_roles, dz=0.0, tip=50e-6, ground_section=3e-3):
    """Alternating rings from the outside in; the grounds sit ``dz`` downstream of the emitters (the convergence)."""
    out = []
    for radius, role in radii_roles:
        out.append(Ring(radius, role, z=dz if role == "ground" else 0.0,
                        tip_radius=tip if role == "emitter" else ground_section / 2))
    return tuple(out)


# --- Plasma Channel's generations (all dimensions estimated from the screenshots: the rings' radii from a ~150 mm
# Mark IV on a kitchen scale and the Form 4's 200 mm build for the 4B; the gaps from the ring pitch)
MARK4 = RingThruster(
    "mark4",
    _rings([(0.072, "ground"), (0.061, "emitter"), (0.050, "ground"), (0.039, "emitter"), (0.028, "ground")],
           dz=0.009, tip=50e-6, ground_section=6e-3),
    hub_radius=0.025, outer_radius=0.075, creepage=0.014, mass_kg=0.209,
    notes="dual convergence: 3 copper-plated printed grounds, 2 serrated bronze emitters; 50 W; 3.5 m/s at 78 % RH, 4.1 at 50 %",
    estimated=("radii", "gap", "tip_radius", "creepage", "hub"))

MARK4B = RingThruster(
    "mark4b",
    _rings([(0.096, "ground"), (0.082, "emitter"), (0.068, "ground"), (0.054, "emitter"), (0.040, "ground"),
            (0.030, "ground")], dz=0.008, tip=40e-6, ground_section=3e-3),
    hub_radius=0.010, outer_radius=0.100, creepage=0.018, mass_kg=0.513,
    notes="3x the flow area, 4 nickel-plated grounds, 2 loop emitters with longer sharper teeth, hollow-bolt gap adjustment; 3.6 m/s max",
    estimated=("radii", "gap", "tip_radius", "creepage", "hub"))

MARK4C = RingThruster(
    "mark4c", MARK4B.rings, hub_radius=0.010, outer_radius=0.100, creepage=0.018, mass_kg=0.484,
    notes="4B without the threading, less restriction; 3.7 m/s max; ~25 gf on the pendulum",
    estimated=MARK4B.estimated)

PRESETS = {t.name: t for t in (MARK4, MARK4B, MARK4C)}
