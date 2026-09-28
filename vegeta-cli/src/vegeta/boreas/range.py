"""Mission range and consumption of a small aircraft with four kinds of powerplant.

The aircraft is a drag polar (``Aircraft``); a powerplant gives the thrust it can deliver at a speed and what
that thrust costs — electrical power (``ElectricProp``, ``ElectricFan``) or fuel flow (``PistonProp``,
``Turbojet``). ``fly_mission`` flies a list of segments step by step: at every step lift equals the current
weight, the drag and the thrust follow, the energy or fuel is drawn, and **the mass of a liquid-fuel aircraft
falls as the fuel burns**, so its drag and consumption fall along the mission. ``max_range`` flies a cruise until
the energy is gone. Pure numpy; the propeller powerplants use Boreas BEMT through a precomputed map.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .airfoil import Airfoil
from .bemt import solve
from .motor import Battery, Motor
from .propeller import Propeller
from .system import Propulsion

G = 9.81


@dataclass(frozen=True)
class Aircraft:
    """A drag polar ``Cd = cd0 + k Cl^2`` on a wing area; the masses that do not burn."""

    name: str
    empty_kg: float                  # airframe, systems, powerplant dry mass (no fuel, no battery, no payload)
    payload_kg: float
    wing_area_m2: float
    cd0: float
    k: float
    cl_max: float = 1.0
    rho: float = 1.225

    def drag(self, v: float, mass_kg: float) -> tuple[float, float]:
        """``(drag N, Cl)`` in level flight at ``v`` carrying ``mass_kg`` (Cl above cl_max is the stall)."""
        q = 0.5 * self.rho * v * v
        cl = mass_kg * G / (q * self.wing_area_m2)
        return q * self.wing_area_m2 * (self.cd0 + self.k * cl * cl), cl

    def stall_speed(self, mass_kg: float) -> float:
        return math.sqrt(2 * mass_kg * G / (self.rho * self.wing_area_m2 * self.cl_max))

    def best_ld(self) -> tuple[float, float]:
        """``(L/D max, Cl at L/D max)`` of the polar."""
        cl = math.sqrt(self.cd0 / self.k)
        return cl / (self.cd0 + self.k * cl * cl), cl


class Powerplant(Protocol):
    name: str
    kind: str                        # "electric" | "liquid"

    def thrust_max(self, v: float) -> float: ...
    def consume(self, thrust: float, v: float) -> tuple[float, float]:
        """``(electrical power W, fuel flow kg/s)`` to hold ``thrust`` at ``v``; one of them is zero."""
        ...
    def energy_available(self) -> float:
        """Usable energy: Wh (electric) or kg of fuel (liquid)."""
        ...
    def store_mass(self) -> float:
        """Mass of the full energy store: battery (constant) or fuel (burns away)."""
        ...


class _ThrustMap:
    """thrust(setting, v) and cost(setting, v) on a grid, so a mission of thousands of steps stays fast."""

    def __init__(self, settings, speeds, thrust, cost):
        self.settings, self.speeds = np.asarray(settings, float), np.asarray(speeds, float)
        self.thrust, self.cost = np.asarray(thrust, float), np.asarray(cost, float)   # [n_settings, n_speeds]

    def _col(self, table, v):
        j = np.clip(np.searchsorted(self.speeds, v) - 1, 0, len(self.speeds) - 2)
        w = np.clip((v - self.speeds[j]) / (self.speeds[j + 1] - self.speeds[j]), 0.0, 1.0)
        return table[:, j] * (1 - w) + table[:, j + 1] * w

    def thrust_max(self, v):
        return float(self._col(self.thrust, v).max())

    def cost_for(self, thrust, v):
        t, c = self._col(self.thrust, v), self._col(self.cost, v)
        order = np.argsort(t)
        return float(np.interp(thrust, t[order], c[order]))


@dataclass
class ElectricProp:
    """Propeller + brushless motor + battery (a Boreas ``Propulsion``); mass constant, energy from the battery."""

    propulsion: Propulsion
    name: str = "electric propeller"
    kind: str = "electric"
    speeds: tuple = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0)
    _map: _ThrustMap | None = field(default=None, repr=False)

    def _table(self) -> _ThrustMap:
        if self._map is None:
            th = np.linspace(0.15, 1.0, 12)
            T = np.zeros((len(th), len(self.speeds))); P = np.zeros_like(T)
            for i, t in enumerate(th):
                for j, v in enumerate(self.speeds):
                    pt = self.propulsion.at_throttle(float(t), float(v))
                    T[i, j], P[i, j] = max(pt.thrust, 0.0), pt.electrical_power
            self._map = _ThrustMap(th, self.speeds, T, P)
        return self._map

    def thrust_max(self, v):
        return self._table().thrust_max(v)

    def consume(self, thrust, v):
        return self._table().cost_for(thrust, v), 0.0

    def energy_available(self):
        return self.propulsion.battery.usable_wh

    def store_mass(self):
        return self.propulsion.battery.mass_kg


@dataclass
class PistonProp:
    """The same propeller on a small petrol engine: shaft power from BEMT, fuel from a specific consumption."""

    prop: Propeller
    airfoil: Airfoil
    power_max_w: float               # engine shaft power
    bsfc_kg_per_kwh: float = 0.50    # small two-stroke; a four-stroke is ~0.35
    fuel_kg: float = 1.0
    rpm_max: float = 9000.0
    rho: float = 1.225
    name: str = "petrol propeller"
    kind: str = "liquid"
    speeds: tuple = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0)
    _map: _ThrustMap | None = field(default=None, repr=False)

    def _table(self) -> _ThrustMap:
        if self._map is None:
            rpms = np.linspace(0.2 * self.rpm_max, self.rpm_max, 14)
            T = np.zeros((len(rpms), len(self.speeds))); P = np.zeros_like(T)
            for i, r in enumerate(rpms):
                for j, v in enumerate(self.speeds):
                    op = solve(self.prop, self.airfoil, float(r), float(v), self.rho)
                    T[i, j], P[i, j] = max(op.thrust, 0.0), max(op.power, 0.0)
            T[P > self.power_max_w] = 0.0                       # beyond the engine: not available
            self._map = _ThrustMap(rpms, self.speeds, T, P)
        return self._map

    def thrust_max(self, v):
        return self._table().thrust_max(v)

    def consume(self, thrust, v):
        p_shaft = self._table().cost_for(thrust, v)
        return 0.0, self.bsfc_kg_per_kwh * p_shaft / 1000 / 3600

    def energy_available(self):
        return self.fuel_kg

    def store_mass(self):
        return self.fuel_kg


@dataclass
class ElectricFan:
    """An electric ducted fan by fan momentum theory: thrust = mdot (Ve - V) with the exit speed from the shaft power."""

    fan_area_m2: float
    power_max_w: float               # electrical
    battery: Battery
    efficiency: float = 0.55         # electrical -> jet kinetic power (motor x fan x duct)
    rho: float = 1.225
    name: str = "electric ducted fan"
    kind: str = "electric"

    def _thrust_at_power(self, p_elec: float, v: float) -> float:
        p = self.efficiency * p_elec
        if p <= 0:
            return 0.0
        lo, hi = v, v + 400.0                                   # exit speed: 0.5 mdot (Ve^2 - V^2) = p, mdot = rho A (V + Ve) / 2
        for _ in range(60):
            ve = 0.5 * (lo + hi)
            mdot = self.rho * self.fan_area_m2 * 0.5 * (v + ve)
            lo, hi = (ve, hi) if 0.5 * mdot * (ve * ve - v * v) < p else (lo, ve)
        ve = lo
        return self.rho * self.fan_area_m2 * 0.5 * (v + ve) * (ve - v)

    def thrust_max(self, v):
        return self._thrust_at_power(self.power_max_w, v)

    def consume(self, thrust, v):
        lo, hi = 0.0, self.power_max_w
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self._thrust_at_power(mid, v) < thrust else (lo, mid)
        return hi, 0.0

    def energy_available(self):
        return self.battery.usable_wh

    def store_mass(self):
        return self.battery.mass_kg


@dataclass
class Turbojet:
    """A micro turbojet: static thrust with a linear lapse with speed, fuel from a thrust-specific consumption."""

    thrust_static_n: float
    tsfc_kg_per_n_h: float = 0.15    # small turbojets: 0.12-0.20 kg / (N h)
    fuel_kg: float = 2.0
    lapse_per_m_s: float = 0.002     # thrust falls ~10 % per 50 m/s (ram drag minus ram recovery, small engines)
    idle_fraction: float = 0.08      # fuel flow at zero thrust, as a fraction of the full-thrust flow
    name: str = "micro turbojet"
    kind: str = "liquid"

    def thrust_max(self, v):
        return self.thrust_static_n * max(0.2, 1.0 - self.lapse_per_m_s * v)

    def consume(self, thrust, v):
        full = self.tsfc_kg_per_n_h * self.thrust_static_n / 3600
        return 0.0, self.idle_fraction * full + self.tsfc_kg_per_n_h * max(thrust, 0.0) / 3600 * (1 - self.idle_fraction)

    def energy_available(self):
        return self.fuel_kg

    def store_mass(self):
        return self.fuel_kg


@dataclass(frozen=True)
class Leg:
    """One mission leg at a constant airspeed: for a ``duration_s`` or a ``distance_m`` (ground), climbing at
    ``climb_m_s`` (negative descends; the powerplant may not deliver a steep climb — it is then flattened)."""

    name: str
    speed_m_s: float
    duration_s: float | None = None
    distance_m: float | None = None
    climb_m_s: float = 0.0

    def __post_init__(self):
        if (self.duration_s is None) == (self.distance_m is None):
            raise ValueError("a leg needs a duration or a distance, not both")
        if self.speed_m_s <= 0:
            raise ValueError("speed must be > 0")


@dataclass
class MissionLog:
    aircraft: str
    powerplant: str
    kind: str
    t: np.ndarray
    x: np.ndarray                    # ground distance [m]
    mass: np.ndarray
    thrust: np.ndarray
    drag: np.ndarray
    power_w: np.ndarray
    fuel_flow: np.ndarray
    energy_used: np.ndarray          # Wh (electric) or kg (liquid), cumulative
    legs: list
    stopped: str | None = None       # why the mission ended early

    @property
    def range_km(self) -> float:
        return float(self.x[-1] / 1000)

    @property
    def endurance_min(self) -> float:
        return float(self.t[-1] / 60)

    def summary(self) -> dict:
        energy = float(self.energy_used[-1])
        return {"powerplant": self.powerplant, "kind": self.kind, "range_km": self.range_km,
                "endurance_min": self.endurance_min, "mass_start_kg": float(self.mass[0]), "mass_end_kg": float(self.mass[-1]),
                "energy_used_" + ("Wh" if self.kind == "electric" else "kg"): energy,
                "mean_thrust_N": float(self.thrust.mean()), "stopped": self.stopped, "legs": self.legs}


def fly_mission(aircraft: Aircraft, powerplant: Powerplant, legs, *, wind_m_s: float = 0.0, dt: float = 2.0,
                reserve: float = 0.0) -> MissionLog:
    """Fly ``legs`` step by step. ``wind_m_s`` > 0 is a tailwind over the whole mission. The mission stops when
    the energy (minus ``reserve``, a fraction) is used up; the log says so in ``stopped``."""
    avail = powerplant.energy_available() * (1 - reserve)
    m = aircraft.empty_kg + aircraft.payload_kg + powerplant.store_mass()
    t = x = used = 0.0
    rows = []
    stopped = None
    legs_out = []
    for leg in legs:
        v = leg.speed_m_s
        gnd = v + wind_m_s
        leg_t0, leg_x0, leg_e0 = t, x, used
        while True:
            done = (t - leg_t0 >= leg.duration_s) if leg.duration_s is not None else (x - leg_x0 >= leg.distance_m)
            if done:
                break
            drag, cl = aircraft.drag(v, m)
            climb = leg.climb_m_s
            need = drag + m * G * climb / v
            tmax = powerplant.thrust_max(v)
            if need > tmax:                                     # flatten the climb to what the powerplant gives
                climb = max(-abs(climb), (tmax - drag) * v / (m * G))
                need = tmax
            need = max(need, 0.0)
            p_w, f = powerplant.consume(need, v)
            step = p_w * dt / 3600 if powerplant.kind == "electric" else f * dt
            if used + step > avail:
                stopped = f"out of {'energy' if powerplant.kind == 'electric' else 'fuel'} in leg '{leg.name}'"
                break
            rows.append((t, x, m, need, drag, p_w, f, used, cl, climb))
            used += step
            if powerplant.kind == "liquid":
                m -= f * dt
            t += dt
            x += gnd * dt
        legs_out.append({"name": leg.name, "speed_m_s": v, "duration_min": (t - leg_t0) / 60, "distance_km": (x - leg_x0) / 1000,
                         "energy": used - leg_e0, "mass_end_kg": m})
        if stopped:
            break
    if not rows:
        rows.append((0.0, 0.0, m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    a = np.array(rows, dtype=float).T
    return MissionLog(aircraft.name, powerplant.name, powerplant.kind, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                      legs_out, stopped)


def max_range(aircraft: Aircraft, powerplant: Powerplant, speed_m_s: float, *, wind_m_s: float = 0.0,
              reserve: float = 0.1, dt: float = 5.0, t_max_s: float = 6 * 3600) -> MissionLog:
    """Cruise at ``speed_m_s`` until the energy (minus the reserve) is gone: the still-air or headwind range."""
    return fly_mission(aircraft, powerplant, [Leg("cruise", speed_m_s, duration_s=t_max_s)], wind_m_s=wind_m_s, dt=dt, reserve=reserve)


def breguet_jet(v: float, ld: float, tsfc_kg_per_n_h: float, m0: float, m1: float) -> float:
    """Breguet range [m] of a jet at constant speed and L/D: V (L/D) / (c g) ln(m0/m1); c in kg per N per second."""
    c = tsfc_kg_per_n_h / 3600
    return v * ld / (c * G) * math.log(m0 / m1)


def breguet_prop(eta: float, bsfc_kg_per_kwh: float, ld: float, m0: float, m1: float) -> float:
    """Breguet range [m] of a propeller aircraft: eta / c_p (L/D) ln(m0/m1); c_p in kg per J of shaft work."""
    cp = bsfc_kg_per_kwh / 3.6e6
    return eta / (cp * G) * ld * math.log(m0 / m1)
