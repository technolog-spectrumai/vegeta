"""AGUYA in flight (notebook 29): the turbojet's map as a propulsor, the mass that burns off, and the race to the fire.

The engine is ``vegeta.boreas.microjet`` tabulated over airspeed and shaft speed (``JetUnit``: net thrust and fuel flow;
built from the cycle here or loaded from notebook 28's export). The mission is MERLIN's (stand launch, climb to 150 m,
dash to the fire, 3 min of crosswind passes, dash home, recovery) with two differences that change the numbers:
the aircraft gets lighter as it flies (kerosene), and the energy store is sized per mission (the fuel load), not fixed.

Phases (a point mass in the vertical plane, 0.5 s steps):

1. ``start``: engine start and warm-up on the launcher at idle (``start_s``), then the spool-up to full thrust.
2. ``climb``: catapult release at ``launch_margin`` x the stall speed, full thrust, climbing at the launch speed up to
   ``mission.altitude_m`` (the flight-path angle is what the excess thrust allows, at most 30 deg).
3. ``accelerate``: level, full thrust, to the dash speed (``dash_speed`` or the top speed less 2 %).
4. ``dash``: level at the dash speed, thrust = drag; the fuel flow at the shaft speed that gives it.
5. ``sample``: ``mission.sampling_s`` of passes at 1.3 x stall in 30 deg banked turns; idle when idle is already
   too much thrust (an air brake holds the speed).
6. ``return``: the dash speed back over the same distance.
7. ``recovery``: idle for ``recovery_s`` (descent, the parachute); the engine is shut down at deployment.

``fuel_for`` finds the smallest fuel load that leaves ``reserve`` of it in the tank at the end (bisection); the mass
budget adds the fuel to the dry mass (``dry_mass``). ``top_speed`` solves full-throttle thrust = drag.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.80665
RHO = 1.225


@dataclass
class JetUnit:
    """A turbojet tabulated over airspeed ``V`` [m/s] and shaft speed ``rpm``: ``thrust[i, j]`` net thrust [N],
    ``fuel[i, j]`` fuel flow [kg/s]. Throttle = shaft speed between idle (``rpm[0]``) and maximum (``rpm[-1]``)."""
    name: str
    V: np.ndarray
    rpm: np.ndarray
    thrust: np.ndarray
    fuel: np.ndarray
    mass_kg: float                    # engine + ECU, pump, valves (without tank and fuel)
    spool_time: float = 3.0

    def _row(self, V):
        V = float(np.clip(V, self.V[0], self.V[-1]))
        i = min(int(np.searchsorted(self.V, V, side="right")) - 1, len(self.V) - 2)
        t = (V - self.V[i]) / (self.V[i + 1] - self.V[i])
        return (1 - t) * self.thrust[i] + t * self.thrust[i + 1], (1 - t) * self.fuel[i] + t * self.fuel[i + 1]

    def full(self, V):
        T, f = self._row(V)
        return float(T[-1]), float(f[-1])

    def idle(self, V):
        T, f = self._row(V)
        return float(T[0]), float(f[0])

    def for_thrust(self, V, thrust):
        """(fuel flow, rpm, feasible) for a net thrust at airspeed V; idle when idle gives more than asked."""
        T, f = self._row(V)
        if thrust <= T[0]:
            return float(f[0]), float(self.rpm[0]), True
        if thrust > T[-1]:
            return float(f[-1]), float(self.rpm[-1]), False
        j = int(np.searchsorted(T, thrust))           # thrust rises with rpm along a row
        t = (thrust - T[j - 1]) / (T[j] - T[j - 1])
        return float((1 - t) * f[j - 1] + t * f[j]), float((1 - t) * self.rpm[j - 1] + t * self.rpm[j]), True


def jet_unit(engine, V=None, n_rpm=14, atmosphere=None) -> JetUnit:
    """Tabulate a ``Microjet`` from its cycle (a few hundred operating points, ~1 s)."""
    from vegeta.boreas import microjet as mj

    V = np.linspace(0.0, 220.0, 23) if V is None else np.asarray(V, float)
    rpm = engine.rpm_idle + (engine.rpm_max - engine.rpm_idle) * np.linspace(0, 1, n_rpm) ** 1.2
    m = mj.performance_map(engine, V, rpm, atmosphere)
    T = np.maximum.accumulate(m["thrust"], axis=1)            # keep each row monotonic (the inversion needs it)
    return JetUnit(engine.name, V, rpm, T, m["fuel_flow"], engine.mass_kg + engine.system_mass_kg, engine.spool_time)


def jet_unit_from_export(d: dict) -> JetUnit:
    """A ``JetUnit`` from notebook 28's export (``vegeta.boreas.microjet.load``)."""
    e, m = d["engine"], d["map"]
    return JetUnit(e.name, m["V"], m["rpm"], np.maximum.accumulate(m["thrust"], axis=1), m["fuel_flow"],
                   e.mass_kg + e.system_mass_kg, e.spool_time)


# ---------------------------------------------------------------------------------------------- the aircraft
DRY_MASS_KG = {"airframe (carbon/glass shells, fuel-proof tank)": 1.15, "flight controller + GPS + telemetry": 0.09,
               "servos (4x, fast metal gear)": 0.08, "gas sensor (CO/CO2/PM) + pump": 0.12, "parachute + hatch": 0.25,
               "avionics/ECU battery 2S 2200 mAh": 0.13, "fuel tank bladder + plumbing + filter": 0.12}


def dry_mass(unit: JetUnit, budget: dict | None = None) -> float:
    return sum((budget or DRY_MASS_KG).values()) + unit.mass_kg


@dataclass
class JetAirframe:
    """Drag polar of the aircraft, its dry mass and the fuel on board (set per mission)."""
    name: str
    dry_mass_kg: float
    wing_area_m2: float
    aspect_ratio: float
    cd0: float
    oswald: float = 0.8
    cl_max: float = 1.2
    tank_kg: float = 3.0              # the most fuel the tank holds

    @property
    def k(self):
        return 1.0 / (math.pi * self.aspect_ratio * self.oswald)

    def drag(self, V, mass_kg, rho=RHO, n=1.0):
        V = max(float(V), 1e-3)
        q = 0.5 * rho * V * V
        cl = n * mass_kg * G / (q * self.wing_area_m2)
        return q * self.wing_area_m2 * (self.cd0 + self.k * cl * cl)

    def stall_speed(self, mass_kg, rho=RHO, n=1.0):
        return math.sqrt(2 * n * mass_kg * G / (rho * self.wing_area_m2 * self.cl_max))


def airframe(p, unit: JetUnit, *, design_speed=150.0, budget=None, fuel_density=800.0, oswald=0.8, cl_max=1.2,
             extra_cd_area=0.0008) -> JetAirframe:
    """``JetAirframe`` from the CAD parameters: the drag build-up at the design speed (``aguya.drag_buildup``, with
    ``extra_cd_area`` for the intake lip spillage, antennas and the parachute hatch) and the tank's volume."""
    import aguya as ag

    a = 340.3
    d = ag.drag_buildup(p, design_speed, mach=design_speed / a)
    S = d["planform"]
    return JetAirframe("AGUYA", dry_mass(unit, budget), S, d["aspect_ratio"], (d["cd_area_m2"] + extra_cd_area) / S, oswald,
                       cl_max, ag.tank_volume_l(p) * fuel_density / 1000)


def top_speed(af: JetAirframe, unit: JetUnit, mass_kg: float, rho=RHO) -> float:
    lo, hi = af.stall_speed(mass_kg, rho) * 1.05, float(unit.V[-1])
    f = lambda V: unit.full(V)[0] - af.drag(V, mass_kg, rho)                # noqa: E731
    if f(hi) > 0:
        return hi
    if f(lo) < 0:
        return float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------------------------- the mission
@dataclass
class JetMission:
    altitude_m: float = 150.0
    sampling_s: float = 180.0
    sampling_bank_deg: float = 30.0
    start_s: float = 45.0             # start and warm-up at idle on the launcher
    recovery_s: float = 40.0          # descent and parachute at idle
    launch_margin: float = 1.25       # catapult release speed / stall speed
    reserve: float = 0.15             # fraction of the fuel load left at the end
    dash_speed: float | None = None   # None: 98 % of the top speed
    dt: float = 0.5


@dataclass
class Flight:
    distance_m: float
    fuel_kg: float
    takeoff_mass_kg: float
    time_to_fire_s: float
    return_time_s: float
    total_time_s: float
    dash_speed: float
    launch_speed: float
    fuel_left_kg: float
    phases: dict = field(default_factory=dict)       # name -> (duration s, fuel kg, distance m)
    track: dict = field(default_factory=dict)        # t, x, h, V, mass, fuel_flow arrays
    feasible: bool = True
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"distance [km]": self.distance_m / 1000, "fuel [kg]": round(self.fuel_kg, 3),
                "take-off mass [kg]": round(self.takeoff_mass_kg, 2), "launch [m/s]": round(self.launch_speed, 1),
                "dash [m/s]": round(self.dash_speed, 1), "to the fire [s]": round(self.time_to_fire_s, 1),
                "back home [s]": round(self.return_time_s, 1), "total [s]": round(self.total_time_s, 1),
                "fuel left [kg]": round(self.fuel_left_kg, 3), "feasible": self.feasible}


def fly(af: JetAirframe, unit: JetUnit, distance_m: float, fuel_kg: float, mission: JetMission | None = None,
        rho=RHO) -> Flight:
    """Fly the mission with ``fuel_kg`` on board (see the module text)."""
    ms = mission or JetMission()
    dt = ms.dt
    m0 = af.dry_mass_kg + fuel_kg
    mass, fuel, t, x, h = m0, fuel_kg, 0.0, 0.0, 0.0
    log = {k: [] for k in ("t", "x", "h", "V", "mass", "fuel_flow")}
    phases = {}
    notes = []

    def step(V, ff, dx, dh, name):
        nonlocal mass, fuel, t, x, h
        burn = ff * dt
        mass -= burn
        fuel -= burn
        t += dt
        x += dx
        h += dh
        d, f, s = phases.get(name, (0.0, 0.0, 0.0))
        phases[name] = (d + dt, f + burn, s + abs(dx))
        for k, v in (("t", t), ("x", x), ("h", h), ("V", V), ("mass", mass), ("fuel_flow", ff)):
            log[k].append(v)

    # 1. start, warm-up and spool-up on the launcher
    _, f_idle = unit.idle(0.0)
    _, f_full = unit.full(0.0)
    for _ in range(int(round(ms.start_s / dt))):
        step(0.0, f_idle, 0.0, 0.0, "start")
    for _ in range(int(round(unit.spool_time / dt))):
        step(0.0, 0.5 * (f_idle + f_full), 0.0, 0.0, "start")
    # 2. catapult release and climb at the launch speed
    V = ms.launch_margin * af.stall_speed(mass, rho)
    v_launch = V
    while h < ms.altitude_m:
        T, ff = unit.full(V)
        sin_g = min((T - af.drag(V, mass, rho)) / (mass * G), math.sin(math.radians(30)))
        if sin_g <= 0:
            notes.append("cannot climb at the launch speed: the thrust is below the drag")
            return _fail(distance_m, fuel_kg, m0, V, v_launch, phases, log, notes)
        step(V, ff, V * math.sqrt(1 - sin_g ** 2) * dt, V * sin_g * dt, "climb")
    h = ms.altitude_m
    # 3. accelerate level to the dash speed
    v_top = top_speed(af, unit, mass, rho)
    v_dash = min(ms.dash_speed or 0.98 * v_top, 0.98 * v_top)
    while V < v_dash and x < distance_m:
        T, ff = unit.full(V)
        a = (T - af.drag(V, mass, rho)) / mass
        if a <= 1e-3:
            break
        V = min(V + a * dt, v_dash)
        step(V, ff, V * dt, 0.0, "accelerate")
    v_dash = V
    # 4. dash to the fire
    while x < distance_m:
        ff, _, ok = unit.for_thrust(V, af.drag(V, mass, rho))
        step(V, ff, V * dt, 0.0, "dash")
    t_fire = t
    # 5. sampling passes: banked turns at 1.3 x stall (in the turn)
    n = 1.0 / math.cos(math.radians(ms.sampling_bank_deg))
    for _ in range(int(round(ms.sampling_s / dt))):
        Vs = 1.3 * af.stall_speed(mass, rho, n)
        ff, _, _ = unit.for_thrust(Vs, af.drag(Vs, mass, rho, n))
        step(Vs, ff, 0.0, 0.0, "sample")
    t_sampled = t
    # 6. back home at the dash speed (with a short re-acceleration at full thrust)
    V = 1.3 * af.stall_speed(mass, rho)
    while V < v_dash:
        T, ff = unit.full(V)
        a = (T - af.drag(V, mass, rho)) / mass
        if a <= 1e-3:
            break
        V = min(V + a * dt, v_dash)
        step(V, ff, -V * dt, 0.0, "return")
    while x > 0:
        ff, _, _ = unit.for_thrust(V, af.drag(V, mass, rho))
        step(V, ff, -V * dt, 0.0, "return")
    t_home = t
    # 7. recovery
    for _ in range(int(round(ms.recovery_s / dt))):
        step(0.5 * V, f_idle, 0.0, -ms.altitude_m / ms.recovery_s * dt, "recovery")
    feasible = fuel >= ms.reserve * fuel_kg - 1e-9
    if not feasible:
        notes.append(f"fuel left {fuel:.3f} kg, below the {ms.reserve:.0%} reserve")
    return Flight(distance_m, fuel_kg, m0, t_fire - ms.start_s - unit.spool_time, t_home - t_sampled, t,
                  v_dash, v_launch, fuel, phases, {k: np.array(v) for k, v in log.items()}, feasible, notes)


def _fail(distance_m, fuel_kg, m0, V, v_launch, phases, log, notes):
    return Flight(distance_m, fuel_kg, m0, float("nan"), float("nan"), float("nan"), V, v_launch, float("nan"), phases,
                  {k: np.array(v) for k, v in log.items()}, False, notes)


def fuel_for(af: JetAirframe, unit: JetUnit, distance_m: float, mission: JetMission | None = None, rho=RHO) -> Flight:
    """The flight with the smallest fuel load that keeps the reserve (bisection on the fuel); ``feasible`` is False
    when even a full tank does not."""
    ms = mission or JetMission()
    lo, hi = 0.05, af.tank_kg

    def margin(fk):
        f = fly(af, unit, distance_m, fk, ms, rho)
        return f, (f.fuel_left_kg - ms.reserve * fk) if f.feasible or not math.isnan(f.fuel_left_kg) else -1.0

    full, m_hi = margin(hi)
    if m_hi < 0:
        full.feasible = False
        full.notes.append(f"a full tank ({af.tank_kg:.2f} kg) does not reach {distance_m / 1000:g} km and back with the reserve")
        return full
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if margin(mid)[1] >= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.002:
            break
    return fly(af, unit, distance_m, hi, ms, rho)


def size_for(p, unit: JetUnit, distance_m: float, mission: JetMission | None = None, diameters=None, **airframe_kw):
    """The thinnest fuselage (from ``diameters``, mm) whose tank carries the fuel for ``distance_m`` and back with the
    reserve: returns ``(params, airframe, flight)``; the flight's ``feasible`` is False when none does."""
    import aguya as ag

    diameters = np.arange(100.0, 181.0, 5.0) if diameters is None else diameters
    out = None
    for d in diameters:
        q = ag.Aguya().resolve(**dict(p, fuselage_diameter=float(d)))
        a = airframe(q, unit, **airframe_kw)
        f = fuel_for(a, unit, distance_m, mission)
        out = (q, a, f)
        if f.feasible:
            break
    return out


def race_table(af: JetAirframe, unit: JetUnit, distances_km=(5, 8, 10, 20, 30), mission: JetMission | None = None):
    import pandas as pd

    rows = []
    for d in distances_km:
        f = fuel_for(af, unit, d * 1000.0, mission)
        rows.append(f.summary())
    return pd.DataFrame(rows).set_index("distance [km]")


__all__ = ["JetUnit", "jet_unit", "jet_unit_from_export", "DRY_MASS_KG", "dry_mass", "JetAirframe", "airframe",
           "top_speed", "JetMission", "Flight", "fly", "fuel_for", "size_for", "race_table"]
