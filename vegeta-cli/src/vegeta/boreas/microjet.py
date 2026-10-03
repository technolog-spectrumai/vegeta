"""Microjet (small single-spool turbojet) performance: a 1-D gas-path cycle on its operating line.

A model-aircraft turbojet is a radial compressor, an annular combustor, an axial turbine and a convergent nozzle on
one shaft. The model follows the gas along the path (stations: 0 free stream, 2 compressor face, 3 compressor exit,
4 turbine entry, 5 turbine exit, 8 nozzle exit), with total temperatures and pressures:

- Intake: ``T2 = T0 + V^2 / (2 cp)``, ``P2 = recovery P0 (T2 / T0)^(g / (g - 1))`` (isentropic ram, a recovery factor).
- Radial compressor: the Euler work is set by the impeller tip speed ``U = omega d2 / 2`` alone,
  ``w_c = slip power_input U^2`` (slip factor of the impeller blades, power input factor for disc friction and
  recirculation); ``T3 = T2 + w_c / cp``, ``P3 = P2 (1 + eta_c w_c / (cp T2))^(g / (g - 1))``. The compressor runs
  wherever the turbine and nozzle let it (vertical speed lines: no compressor map, no surge line). Its efficiency
  falls away from the design speed, ``eta_c (1 - eta_c_droop (1 - N / N_design)^2)``.
- Combustor: ``P4 = (1 - burner_loss (N / N_max)^2) P3`` (the loss scales with the dynamic head, which follows the
  shaft speed squared); fuel-air ratio from the energy balance with the burner efficiency,
  ``f = (cp_g T4 - cp T3) / (eta_b LHV - cp_g T4)``.
- Turbine: drives the compressor (and the bearings, ``eta_mech``): ``(1 + f) cp_g (T4 - T5) = w_c / eta_mech``;
  ``P5 = P4 (1 - (T4 - T5) / (eta_t T4))^(g_g / (g_g - 1))``.
- Turbine nozzle guide vanes (NGV) and the exhaust nozzle are two throats in series. The NGV passes the gas from
  ``P4, T4`` to the stage's mid pressure ``sqrt(P4 P5)`` through its throat area ``A4``; the convergent nozzle passes it
  from ``P5, T5`` to ambient through ``A8``. Each is a compressible orifice (choked above the critical ratio).
- Operating point at a shaft speed ``N`` and airspeed ``V``: the compressor fixes ``T3, P3`` and the turbine work; the
  only unknown is ``T4``. Hotter gas means less flow through the NGV and less turbine expansion for the same work
  (``P5`` rises), so more flow through the nozzle: the two flows cross once. That root gives the mass flow, the fuel
  and the thrust ``F = (1 + f) mdot V8 + (p8 - p0) A8 - mdot V`` (gross minus ram drag, with a nozzle velocity
  coefficient).

``A4`` and ``A8`` are sized from a design point (maximum speed, static, sea level): the design mass flow and the
turbine entry temperature. ``calibrate`` fits those two and the burner efficiency to a datasheet's maximum thrust,
fuel flow and exhaust temperature (EGT), so one catalogue point fixes the engine and the rest of the envelope follows
from the cycle. The burner efficiency then carries every loss the cycle leaves out (a datasheet's fuel flow is
usually 15-25 % above what a clean cycle burns).

Limits: no compressor map (a real speed line bends over, and the compressor surges at low flow), no turbine map
(``eta_t`` constant), no cooling or bleed flows, no Reynolds effects at idle, steady state only (spool-up time is
``spool_time``, a datasheet number). Fuel is kerosene/Jet-A1 by default (``lhv``). Expect the shape of the envelope
to be right to ~10 % and the calibrated point to be exact.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

__all__ = ["Microjet", "MicrojetPoint", "Atmosphere", "CATALOGUE", "isa", "solve", "calibrate", "from_catalogue",
           "performance_map", "rpm_for_thrust", "metal_temperatures", "export", "load"]

R = 287.05
G_AIR, CP_AIR = 1.4, 1005.0
G_GAS, CP_GAS = 1.333, 1150.0


@dataclass(frozen=True)
class Atmosphere:
    pressure: float = 101325.0     # Pa
    temperature: float = 288.15    # K

    @property
    def density(self) -> float:
        return self.pressure / (R * self.temperature)

    @property
    def speed_of_sound(self) -> float:
        return math.sqrt(G_AIR * R * self.temperature)


def isa(altitude_m: float = 0.0, delta_t: float = 0.0) -> Atmosphere:
    """International Standard Atmosphere (troposphere), with a temperature offset ``delta_t``."""
    t_std = 288.15 - 0.0065 * altitude_m
    return Atmosphere(101325.0 * (t_std / 288.15) ** 5.2559, t_std + delta_t)


@dataclass(frozen=True)
class Microjet:
    """A single-spool turbojet. Lengths in metres, speeds in rpm, temperatures in K.

    ``design_mass_flow`` and ``design_tit`` (air flow and turbine entry temperature at ``rpm_max``, static, sea level)
    size the NGV throat and the nozzle; ``calibrate`` fits them to a datasheet."""

    impeller_diameter: float                  # compressor exducer (tip) diameter d2
    rpm_max: float
    rpm_idle: float
    design_mass_flow: float                   # kg/s of air at rpm_max, static, ISA sea level
    design_tit: float = 1150.0                # K, turbine entry temperature at the design point
    slip: float = 0.88                        # impeller slip factor (radial blades ~0.9)
    power_input: float = 1.04                 # disc friction and recirculation on top of the Euler work
    eta_c: float = 0.72                       # compressor total-to-total isentropic efficiency at the design speed
    eta_c_droop: float = 0.25                 # efficiency loss toward zero speed: eta_c (1 - droop (1 - N/Nd)^2)
    eta_t: float = 0.80                       # turbine total-to-total isentropic efficiency
    eta_mech: float = 0.98                    # bearings and oil/fuel pump on the shaft
    eta_burner: float = 0.93                  # combustion efficiency
    burner_loss: float = 0.06                 # fraction of P3 lost in the combustor at rpm_max (scales with N^2)
    intake_recovery: float = 0.98             # total-pressure recovery of the intake
    nozzle_cv: float = 0.97                   # nozzle velocity coefficient
    lhv: float = 43.0e6                       # J/kg, kerosene / Jet-A1
    fuel_density: float = 800.0               # kg/m^3
    max_egt: float = 1050.0                   # K, the ECU's exhaust temperature limit (datasheet)
    spool_time: float = 3.0                   # s, idle to full thrust (datasheet / measured)
    mass_kg: float = 1.1                      # engine with starter
    system_mass_kg: float = 0.25              # ECU, pump, valves, tubing (without the tank and fuel)
    outer_diameter: float = 0.10              # casing, for the nacelle
    length: float = 0.25
    name: str = "microjet"
    a4: float = field(default=0.0, compare=False)   # NGV throat area (set by sizing; 0 = size from the design point)
    a8: float = field(default=0.0, compare=False)   # nozzle exit area

    def __post_init__(self):
        if self.impeller_diameter <= 0 or self.rpm_max <= 0 or not 0 < self.rpm_idle < self.rpm_max:
            raise ValueError("need impeller_diameter > 0 and 0 < rpm_idle < rpm_max")
        if self.design_mass_flow <= 0 or self.design_tit <= 400:
            raise ValueError("need design_mass_flow > 0 and a design_tit above 400 K")
        for name in ("slip", "eta_c", "eta_t", "eta_mech", "eta_burner", "intake_recovery", "nozzle_cv"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.a4 == 0.0 or self.a8 == 0.0:
            a4, a8 = _size(self)
            object.__setattr__(self, "a4", a4)
            object.__setattr__(self, "a8", a8)

    def tip_speed(self, rpm: float) -> float:
        return rpm * math.pi / 60 * self.impeller_diameter

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MicrojetPoint:
    rpm: float
    airspeed: float
    thrust: float               # N, net
    fuel_flow: float            # kg/s
    mass_flow: float            # kg/s of air
    pressure_ratio: float       # P3 / P2
    t2: float
    t3: float
    t4: float                   # turbine entry temperature
    t5: float                   # turbine exit (the EGT a thermocouple reads)
    p3: float
    p4: float
    p5: float
    jet_velocity: float         # nozzle exit velocity
    jet_temperature: float      # nozzle exit static temperature
    nozzle_choked: bool
    ngv_choked: bool
    tip_speed: float
    tip_mach: float             # impeller tip speed / speed of sound at the compressor face
    compressor_power: float     # W
    eta_c: float
    converged: bool
    atmosphere: Atmosphere = Atmosphere()

    @property
    def fuel_flow_g_min(self) -> float:
        return self.fuel_flow * 60e3

    @property
    def tsfc(self) -> float:
        """Thrust-specific fuel consumption, kg/(N h)."""
        return self.fuel_flow * 3600 / self.thrust if self.thrust > 0 else float("inf")

    @property
    def thrust_power(self) -> float:
        return self.thrust * self.airspeed

    def overall_efficiency(self, lhv: float = 43.0e6) -> float:
        return self.thrust_power / (self.fuel_flow * lhv) if self.fuel_flow > 0 else 0.0

    @property
    def propulsive_efficiency(self) -> float:
        v, vj = self.airspeed, self.jet_velocity
        return 2 * v / (v + vj) if vj + v > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["atmosphere"] = asdict(self.atmosphere)
        d.update(fuel_flow_g_min=self.fuel_flow_g_min, tsfc=self.tsfc)
        return d


# -- gas dynamics -------------------------------------------------------------------------------------------------
def _flow_function(p0: float, t0: float, p_back: float, g: float) -> tuple[float, bool]:
    """Mass flow per unit throat area of an isentropic orifice from total ``p0, t0`` to the back pressure, and whether
    it is choked."""
    crit = (2 / (g + 1)) ** (g / (g - 1))
    ratio = max(min(p_back / p0, 1.0), 0.0)
    if ratio <= crit:
        return p0 / math.sqrt(t0) * math.sqrt(g / R) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1))), True
    m = math.sqrt(max(2 / (g - 1) * (ratio ** (-(g - 1) / g) - 1), 0.0))
    return p0 / math.sqrt(t0) * math.sqrt(g / R) * m * (1 + (g - 1) / 2 * m * m) ** (-(g + 1) / (2 * (g - 1))), False


def _compressor(e: Microjet, rpm: float, v: float, atm: Atmosphere):
    t2 = atm.temperature + v * v / (2 * CP_AIR)
    p2 = e.intake_recovery * atm.pressure * (t2 / atm.temperature) ** (G_AIR / (G_AIR - 1))
    u = e.tip_speed(rpm)
    w = e.slip * e.power_input * u * u
    eta = e.eta_c * (1 - e.eta_c_droop * (1 - rpm / e.rpm_max) ** 2)
    t3 = t2 + w / CP_AIR
    p3 = p2 * (1 + eta * w / (CP_AIR * t2)) ** (G_AIR / (G_AIR - 1))
    return t2, p2, t3, p3, w, eta, u


def _hot_section(e: Microjet, t3: float, p3: float, w: float, t4: float, rpm: float):
    """Fuel-air ratio, turbine exit and the stage mid pressure for a turbine entry temperature ``t4``; ``None`` when
    the turbine cannot deliver the work from that temperature."""
    f = (CP_GAS * t4 - CP_AIR * t3) / (e.eta_burner * e.lhv - CP_GAS * t4)
    if f <= 0:
        return None
    p4 = (1 - e.burner_loss * (rpm / e.rpm_max) ** 2) * p3
    dt = w / (e.eta_mech * (1 + f) * CP_GAS)
    t5 = t4 - dt
    x = 1 - dt / (e.eta_t * t4)
    if x <= 0.05:
        return None
    p5 = p4 * x ** (G_GAS / (G_GAS - 1))
    return f, p4, t5, p5


def _nozzle(e: Microjet, p5: float, t5: float, p_amb: float):
    """Exit velocity, static temperature, static pressure and choking of the convergent nozzle."""
    crit = (2 / (G_GAS + 1)) ** (G_GAS / (G_GAS - 1))
    p8 = max(p_amb, crit * p5)
    t8 = t5 * (p8 / p5) ** ((G_GAS - 1) / G_GAS)
    v8 = e.nozzle_cv * math.sqrt(max(2 * CP_GAS * (t5 - t8), 0.0))
    return v8, t8, p8, p8 > p_amb * (1 + 1e-9)


def _size(e: Microjet) -> tuple[float, float]:
    """NGV throat and nozzle exit areas that pass the design mass flow at the design TIT (rpm_max, static, ISA SL)."""
    atm = Atmosphere()
    t2, p2, t3, p3, w, eta, u = _compressor(e, e.rpm_max, 0.0, atm)
    hot = _hot_section(e, t3, p3, w, e.design_tit, e.rpm_max)
    if hot is None:
        raise ValueError(f"design_tit {e.design_tit} K cannot drive the compressor at {e.rpm_max} rpm (tip speed {u:.0f} m/s): "
                         "raise design_tit or eta_t, or lower the tip speed")
    f, p4, t5, p5 = hot
    m_gas = e.design_mass_flow * (1 + f)
    ff4, _ = _flow_function(p4, e.design_tit, math.sqrt(p4 * p5), G_GAS)
    ff8, _ = _flow_function(p5, t5, atm.pressure, G_GAS)
    return m_gas / ff4, m_gas / ff8


def solve(e: Microjet, rpm: float, airspeed: float = 0.0, atmosphere: Atmosphere | None = None) -> MicrojetPoint:
    """The steady operating point at a shaft speed and airspeed (see the module text)."""
    atm = atmosphere or Atmosphere()
    t2, p2, t3, p3, w, eta, u = _compressor(e, rpm, airspeed, atm)

    def residual(t4):
        hot = _hot_section(e, t3, p3, w, t4, rpm)
        if hot is None:
            return None
        f, p4, t5, p5 = hot
        ff4, c4 = _flow_function(p4, t4, math.sqrt(p4 * p5), G_GAS)
        ff8, c8 = _flow_function(p5, t5, atm.pressure, G_GAS)
        m4, m8 = e.a4 * ff4, e.a8 * ff8
        return (m8 - m4) / m4, (f, p4, t5, p5, m4, c4, c8)

    # the residual rises with t4; scan up from the lowest temperature the turbine can work from
    grid = np.linspace(t3 + 20.0, 2600.0, 160)
    lo = hi = None
    prev = None
    for t in grid:
        r = residual(float(t))
        if r is None:
            continue
        if r[0] >= 0:
            hi, lo = float(t), prev
            break
        prev = float(t)
    converged = hi is not None and lo is not None
    if converged:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if residual(mid)[0] >= 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-4:
                break
        t4 = 0.5 * (lo + hi)
    else:
        t4 = float(grid[-1]) if hi is None else float(hi)
    out = residual(t4)
    if out is None:
        raise ValueError(f"no operating point at {rpm:.0f} rpm, {airspeed:.1f} m/s: the turbine cannot drive the compressor")
    _, (f, p4, t5, p5, m_gas, c4, c8) = out
    m_air = m_gas / (1 + f)
    v8, t8, p8, choked = _nozzle(e, p5, t5, atm.pressure)
    rho8 = p8 / (R * t8)
    a8_eff = m_gas / (rho8 * v8) if v8 > 0 else e.a8
    thrust = m_gas * v8 + (p8 - atm.pressure) * a8_eff - m_air * airspeed
    return MicrojetPoint(rpm=float(rpm), airspeed=float(airspeed), thrust=float(thrust), fuel_flow=float(f * m_air),
                         mass_flow=float(m_air), pressure_ratio=float(p3 / p2), t2=t2, t3=t3, t4=float(t4), t5=float(t5),
                         p3=p3, p4=p4, p5=p5, jet_velocity=float(v8), jet_temperature=float(t8), nozzle_choked=bool(choked),
                         ngv_choked=bool(c4), tip_speed=u, tip_mach=u / math.sqrt(G_AIR * R * t2),
                         compressor_power=float(w * m_air), eta_c=eta, converged=converged, atmosphere=atm)


def calibrate(e: Microjet, thrust: float, fuel_flow: float, egt: float, iterations: int = 40) -> Microjet:
    """``e`` with ``design_mass_flow``, ``design_tit`` and ``eta_burner`` fitted so that the maximum-speed static point
    at sea level gives the datasheet ``thrust`` [N], ``fuel_flow`` [kg/s] and exhaust temperature ``egt`` [K]
    (3-D Newton with finite differences)."""
    x = np.array([e.design_mass_flow, e.design_tit, e.eta_burner])

    def f(x):
        eng = replace(e, design_mass_flow=float(x[0]), design_tit=float(x[1]), eta_burner=float(x[2]), a4=0.0, a8=0.0)
        pt = solve(eng, eng.rpm_max)
        return np.array([pt.thrust / thrust - 1, pt.fuel_flow / fuel_flow - 1, pt.t5 / egt - 1]), eng

    for _ in range(iterations):
        r, eng = f(x)
        if np.max(np.abs(r)) < 1e-7:
            break
        J = np.empty((3, 3))
        for j, h in enumerate((1e-4 * x[0], 0.5, 1e-4)):
            dx = x.copy()
            dx[j] += h
            J[:, j] = (f(dx)[0] - r) / h
        step = np.linalg.solve(J, -r)
        step = np.clip(step, [-0.3 * x[0], -150.0, -0.1], [0.3 * x[0], 150.0, 0.1])
        x = x + step
        x[1], x[2] = max(x[1], 700.0), float(np.clip(x[2], 0.3, 1.0))
    r, eng = f(x)
    if np.max(np.abs(r)) > 1e-3:
        raise ValueError(f"calibration did not converge (thrust, fuel and EGT off by {r[0]:+.1%}, {r[1]:+.1%}, {r[2]:+.1%}): "
                         "check the datasheet numbers against the impeller size and speed")
    if not 0.6 <= eng.eta_burner <= 1.0:
        raise ValueError(f"the datasheet needs a burner efficiency of {eng.eta_burner:.2f}: the fuel flow does not fit "
                         "the thrust and EGT with these component efficiencies")
    return eng


# Approximate published figures for two common engine classes (hobby turbojets, 2020s datasheets). They are class
# values to start from, not one manufacturer's numbers: replace them with the datasheet of the engine you fly.
CATALOGUE = {
    "100 N class": dict(thrust_N=100.0, fuel_g_min=320.0, egt_K=973.0, rpm_max=154000.0, rpm_idle=33000.0, impeller_diameter=0.056,
                        mass_kg=1.08, system_mass_kg=0.25, outer_diameter=0.097, length=0.245, max_egt=1023.0, spool_time=3.0),
    "140 N class": dict(thrust_N=142.0, fuel_g_min=450.0, egt_K=973.0, rpm_max=125000.0, rpm_idle=35000.0, impeller_diameter=0.068,
                        mass_kg=1.25, system_mass_kg=0.28, outer_diameter=0.112, length=0.300, max_egt=1023.0, spool_time=3.5),
    "200 N class": dict(thrust_N=200.0, fuel_g_min=620.0, egt_K=973.0, rpm_max=112000.0, rpm_idle=33000.0, impeller_diameter=0.080,
                        mass_kg=1.75, system_mass_kg=0.30, outer_diameter=0.130, length=0.330, max_egt=1023.0, spool_time=4.0),
}


def from_catalogue(name: str, **overrides) -> Microjet:
    """A calibrated engine of a ``CATALOGUE`` class (``overrides`` change the cycle assumptions before fitting)."""
    c = dict(CATALOGUE[name])
    thrust, fuel, egt = c.pop("thrust_N"), c.pop("fuel_g_min") / 60e3, c.pop("egt_K")
    guess = thrust / 450.0                      # air flow from a ~450 m/s jet
    e = Microjet(design_mass_flow=guess, name=name, **dict(c, **overrides))
    return calibrate(e, thrust, fuel, egt)


def performance_map(e: Microjet, airspeeds, rpms, atmosphere: Atmosphere | None = None) -> dict:
    """Net thrust, fuel flow, air flow, TIT and EGT on a grid ``[airspeed, rpm]``."""
    V, N = np.asarray(airspeeds, float), np.asarray(rpms, float)
    keys = ("thrust", "fuel_flow", "mass_flow", "t4", "t5", "pressure_ratio", "jet_velocity")
    out = {k: np.zeros((len(V), len(N))) for k in keys}
    for i, v in enumerate(V):
        for j, n in enumerate(N):
            p = solve(e, float(n), float(v), atmosphere)
            for k in keys:
                out[k][i, j] = getattr(p, k)
    out.update(V=V, rpm=N)
    return out


def rpm_for_thrust(e: Microjet, thrust: float, airspeed: float = 0.0, atmosphere: Atmosphere | None = None):
    """The shaft speed for a net thrust (bisection between idle and maximum); ``None`` when out of range."""
    lo, hi = e.rpm_idle, e.rpm_max
    f = lambda n: solve(e, n, airspeed, atmosphere).thrust - thrust          # noqa: E731
    if f(hi) < 0 or f(lo) > 0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1.0:
            break
    return 0.5 * (lo + hi)


def metal_temperatures(p: MicrojetPoint, cooling: float = 0.15) -> dict:
    """Rough turbine-wheel metal temperatures for a thermal-stress check: the blade and rim run near the relative
    total temperature (between TIT and EGT), cooled a little by the leakage air (``cooling`` of the gas-to-air
    difference); the bore sits near the compressor delivery air that cools the rear bearing."""
    t_rel = p.t5 + 0.5 * (p.t4 - p.t5)
    rim = t_rel - cooling * (t_rel - p.t3)
    bore = p.t3 + 0.25 * (rim - p.t3)
    return {"blade_K": t_rel - 0.5 * cooling * (t_rel - p.t3), "rim_K": rim, "bore_K": bore}


def export(e: Microjet, path, airspeeds, rpms, atmosphere: Atmosphere | None = None, extra: dict | None = None) -> Path:
    """Write the engine and its map to JSON (the format ``load`` reads)."""
    m = performance_map(e, airspeeds, rpms, atmosphere)
    data = {"kind": "microjet", "engine": e.to_dict(), "atmosphere": asdict(atmosphere or Atmosphere()),
            "map": {k: np.asarray(v).tolist() for k, v in m.items()}, **(extra or {})}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return path


def load(path) -> dict:
    """An exported engine: ``engine`` (a :class:`Microjet`), ``map`` (numpy arrays) and the rest as written."""
    d = json.loads(Path(path).read_text())
    d["engine"] = Microjet(**d["engine"])
    d["map"] = {k: np.asarray(v) for k, v in d["map"].items()}
    return d
