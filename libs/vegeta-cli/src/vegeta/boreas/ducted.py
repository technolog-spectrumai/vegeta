"""Ducted fan (EDF) performance: 1-D duct momentum theory coupled with rotor blade elements.

A ducted fan is a rotor in a duct, usually with a stator row behind it and a nozzle at the exit. The model:

- Incompressible and steady, axial inflow. Far upstream the air moves at ``V`` (the airspeed) at ambient
  pressure; at the fan it crosses the annulus ``A_f = pi (R^2 - r_hub^2)`` with a uniform mean axial velocity
  ``V_fan``; it leaves the nozzle exit ``A_e = sigma A_f`` at ambient static pressure with ``V_exit``.
- Continuity: ``V_fan = sigma V_exit``, ``mdot = rho A_f V_fan``. The thrust of the whole unit (rotor + duct lip
  + stator + nozzle, the inlet lip suction included; the nacelle's external friction is ``nacelle_drag``) is
  the momentum balance of the streamtube: ``T = mdot (V_exit - V)``.
- Rotor blade elements at every radial station (no inlet swirl; ``c_theta`` is the swirl behind the rotor):
      mean relative tangential velocity   w_t = omega r - c_theta / 2,   axial velocity V_fan
      phi = atan2(V_fan, w_t),   W^2 = V_fan^2 + w_t^2,   alpha = beta - phi,   (cl, cd) = airfoil(alpha)
      Euler (blade torque = angular momentum flux):  B 1/2 rho W^2 c (cl sin phi + cd cos phi) = rho V_fan 2 pi r c_theta
  solved for ``c_theta`` station by station: the fixed point ``c_theta = rhs(c_theta)`` is found as the root of
  ``c_theta - rhs`` bracketed between no swirl and ``2 omega r``, by Illinois regula falsi (a plain fixed-point
  iteration diverges at a high-solidity hub, where ``d rhs / d c_theta`` reaches about -1.8 for a 12-blade fan).
- Useful total-pressure rise per station:
      dp0 = rho omega r c_theta                                  Euler work per unit volume (the ideal rise)
          - B 1/2 rho W^2 c cd W / (2 pi r V_fan)                profile drag power / volume flow of the annulus
          - k_s 1/2 rho c_theta^2                                swirl: k_s = stator_loss with a stator, 1 without
                                                                 (no stator: the swirl leaves with the jet)
          - 2 (tip_clearance / blade_height) |rho omega r c_theta|   tip clearance (empirical rule: ~2 % efficiency
                                                                 per 1 % clearance / blade height)
  mass-averaged over the annulus (``V_fan`` uniform, so area-weighted) to ``dp0_useful``.
- System balance of total pressure, far upstream to the nozzle exit:
      dp0_useful = 1/2 rho (V_exit^2 - V^2) + duct_loss 1/2 rho V_fan^2
  solved for ``V_exit``. On the working branch the residual (fan rise minus system demand) falls as ``V_exit``
  grows (more flow -> less incidence -> less rise, and more demand). Near zero flow it falls again (the profile
  loss per unit volume grows like 1/V_fan), so ``V_exit`` is scanned from ~0 up to ``V + 3 U_tip`` and the
  highest-flow crossing (the stable intersection of the fan and system curves) is refined by bracketed
  regula falsi (bisection with secant steps, Illinois variant).
- Shaft torque and power: ``Q = sum r rho V_fan 2 pi r c_theta dr``, ``P = omega Q``.
- Several stages (``stages`` = n > 1, each a rotor and its stator, on one shaft): the stator turns the flow axial
  again, so every rotor meets the same axial inflow at the same ``V_fan`` and does the same work; the rise is
  ``n dp0_useful``, the rotor thrust, torque and power ``n`` times one rotor's. Each stage after the first adds
  ``interstage_loss 1/2 rho V_fan^2`` to the system (the stator's wake and the gap in front of the next rotor). A
  multi-stage fan needs a stator in every stage.

With no profile drag, no duct, stator or clearance loss the result is the ideal ducted actuator disk:
``P = T (V_exit + V) / 2``, and in hover ``P = T^1.5 / (2 sqrt(rho sigma A_f))`` — for ``sigma = 1`` that is 1/sqrt(2)
of an ideal open rotor of the same area (the blades then carry at most half the thrust, the duct lip and the
stator the rest). The figure of merit
is defined as for an open rotor of the fan annulus, ``T^1.5 / (P sqrt(2 rho A_f))``, so an ideal duct reaches
``sqrt(2 sigma)`` and a real one can exceed 1.

Assumptions and limits: incompressible (tip Mach below ~0.5); uniform ``V_fan`` (no radial equilibrium, no
mixing loss of the radially non-uniform ``dp0``, which is mass-averaged as if mixed losslessly); isolated-aerofoil
section data (no cascade or solidity correction, although a 12-blade hub has a solidity above 1); no Reynolds
effects; no inlet separation (a sharp-lip static loss belongs in ``duct_loss``); axial inflow only. The stator is
a loss coefficient on the swirl, not a vane-row model. Without a stator the jet's swirl is counted as lost but
its effect on the exit static pressure is ignored.

If the fan cannot push air against the system at any flow (the residual is negative over the whole scan, e.g. a
huge ``duct_loss``), ``solve`` returns the scan point nearest a balance (the largest residual) with
``converged=False``; its numbers are not an operating point. Windmilling (``V_exit < V``, negative thrust and an
Euler rise below zero) is solved like any other point.

The default losses (``duct_loss = 0.06``, ``stator_loss = 0.1``, 0.5 mm gap) describe a clean, well-made unit: a
90 mm, 12-blade fan with them needs only ~26 % more shaft power than the ideal duct at 22 N static (figure of merit
~1.07). Catalogue claims for hobby 90 mm, 12-blade 6S units (about 3.0-3.8 kgf from 1.9-3.1 kW electrical, taken at
an assumed 85 % motor + ESC efficiency) give 0.75-1.0, i.e. ``duct_loss`` ~0.25-0.5 with a 0.9 mm gap. For a real
unit, fit ``duct_loss`` to a measured static thrust and shaft power with ``fit_duct_loss``; it then carries
everything the model leaves out (inlet lip and strut losses, low-Reynolds or rough blades, mixing).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .airfoil import Airfoil
from .propeller import Propeller

__all__ = ["DuctedFan", "DuctedPoint", "fit_duct_loss", "nacelle_drag", "rpm_for_thrust", "solve"]


@dataclass(frozen=True)
class DuctedFan:
    """A rotor in a duct. ``rotor`` gives the blades (count, ``r`` from hub to tip, chord, ``beta_deg``); the annulus
    between ``rotor.hub_radius`` and ``rotor.radius`` is the fan face.

    ``tip_clearance_m``: radial gap between blade tips and duct wall. ``exit_area_ratio`` (sigma): nozzle exit area /
    fan annulus area. ``stator_vanes``: 0 = no stator (the swirl leaves with the jet). ``stator_loss``: fraction of the
    swirl dynamic pressure 1/2 rho c_theta^2 lost while the stator turns the flow axial. ``duct_loss``: inlet + wall +
    nozzle total-pressure loss in units of 1/2 rho V_fan^2. ``external_wetted_area_m2`` and ``duct_length_m``: the
    nacelle outside, for its friction drag (``nacelle_drag``). ``mass_kg``: the whole unit (rotor, duct, stator, motor
    if you include it — say so in ``notes``).

    The default losses are those of a clean unit and flatter a typical hobby EDF (see the module docstring): give
    ``duct_loss`` ~0.25-0.5 and the real gap for one, or fit it to a measured point with ``fit_duct_loss``."""

    name: str
    rotor: Propeller
    tip_clearance_m: float = 0.0005
    exit_area_ratio: float = 0.9
    stator_vanes: int = 0
    stator_loss: float = 0.1
    duct_loss: float = 0.06
    external_wetted_area_m2: float = 0.0
    duct_length_m: float = 0.0
    mass_kg: float = 0.0
    notes: str = ""
    stages: int = 1
    interstage_loss: float = 0.03

    def __post_init__(self):
        if not isinstance(self.rotor, Propeller):
            raise ValueError("rotor must be a boreas.Propeller (the blade geometry from hub to tip)")
        if not self.exit_area_ratio > 0:
            raise ValueError("exit_area_ratio (nozzle exit / fan annulus area) must be > 0")
        if self.stator_vanes < 0 or int(self.stator_vanes) != self.stator_vanes:
            raise ValueError("stator_vanes must be a whole number >= 0 (0 = no stator)")
        if self.stator_loss < 0 or self.duct_loss < 0:
            raise ValueError("stator_loss and duct_loss must be >= 0")
        if not 0 <= self.tip_clearance_m < self.blade_height:
            raise ValueError(f"tip_clearance_m must be >= 0 and below the blade height {self.blade_height * 1000:.2f} mm")
        if self.external_wetted_area_m2 < 0 or self.duct_length_m < 0 or self.mass_kg < 0:
            raise ValueError("external_wetted_area_m2, duct_length_m and mass_kg must be >= 0")
        if self.stages < 1 or int(self.stages) != self.stages:
            raise ValueError("stages must be a whole number >= 1")
        if self.stages > 1 and self.stator_vanes == 0:
            raise ValueError("a multi-stage fan needs a stator in every stage (stator_vanes > 0)")
        if self.interstage_loss < 0:
            raise ValueError("interstage_loss must be >= 0")
        if self.external_wetted_area_m2 > 0 and self.duct_length_m <= 0:
            raise ValueError("a nacelle wetted area needs duct_length_m > 0 (the friction Reynolds number length)")

    @property
    def fan_area(self) -> float:
        """Annulus between hub and blade tips [m^2]."""
        return math.pi * (self.rotor.radius**2 - self.rotor.hub_radius**2)

    @property
    def exit_area(self) -> float:
        return self.exit_area_ratio * self.fan_area

    @property
    def blade_height(self) -> float:
        """Blade span, tip radius minus hub radius [m]."""
        return self.rotor.radius - self.rotor.hub_radius

    def describe(self) -> dict:
        return {"name": self.name, "rotor": self.rotor.describe(), "fan_area_m2": self.fan_area,
                "exit_area_m2": self.exit_area, "blade_height_m": self.blade_height,
                "tip_clearance_m": self.tip_clearance_m, "exit_area_ratio": self.exit_area_ratio,
                "stator_vanes": self.stator_vanes, "stator_loss": self.stator_loss, "duct_loss": self.duct_loss,
                "stages": self.stages, "interstage_loss": self.interstage_loss,
                "external_wetted_area_m2": self.external_wetted_area_m2, "duct_length_m": self.duct_length_m,
                "mass_kg": self.mass_kg, "notes": self.notes}


@dataclass
class DuctedPoint:
    """One (rpm, airspeed) solution of a ducted fan: the whole unit, the rotor share, and radial distributions."""

    rpm: float
    airspeed: float
    rho: float
    thrust: float               # N, whole unit: mdot (V_exit - V)
    rotor_thrust: float         # N, axial force on the blades (blade elements)
    duct_thrust: float          # N, thrust - rotor_thrust: duct lip, stator, hub and nozzle together
    torque: float               # N m
    power: float                # W (shaft)
    efficiency: float           # T V / P (0 at V = 0)
    figure_of_merit: float      # static: T^1.5 / (P sqrt(2 rho A_fan)); 0 in forward flight
    mass_flow: float            # kg/s
    fan_velocity: float         # m/s, mean axial velocity at the fan
    exit_velocity: float        # m/s, nozzle exit
    total_pressure_rise: float  # Pa, useful, mass-averaged over the annulus
    tip_mach: float
    converged: bool
    r: np.ndarray = field(repr=False)
    c_theta: np.ndarray = field(repr=False)      # swirl behind the rotor, m/s
    alpha_deg: np.ndarray = field(repr=False)
    cl: np.ndarray = field(repr=False)
    dp0: np.ndarray = field(repr=False)          # useful total-pressure rise per station, Pa

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                continue
            d[k] = bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
        d["radial"] = {k: np.asarray(getattr(self, k)).round(6).tolist()
                       for k in ("r", "c_theta", "alpha_deg", "cl", "dp0")}
        return d


def _swirl(B: int, r, chord, beta, omega: float, v_fan: float, airfoil: Airfoil, *,
           iterations: int = 100, tolerance: float = 1e-11):
    """``c_theta`` at every station for the axial velocity ``v_fan``: the root of
    ``g(c) = c - B c W^2 (cl sin phi + cd cos phi) / (4 pi r V_fan)``. ``g(0) < 0`` for a fan doing work (the root
    lies in ``[0, 2 omega r]``, where ``w_t`` reaches 0 and the blade lift reverses), ``g(0) > 0`` for a windmilling
    station (the root is negative); the far end of the bracket is pushed out until ``g`` changes sign. Returns
    ``(c_theta, converged)``."""
    def g(c):
        wt = omega * r - 0.5 * c
        phi = np.arctan2(v_fan, wt)
        cl, cd = airfoil.coefficients(beta - phi)
        return c - B * chord * (v_fan**2 + wt**2) * (cl * np.sin(phi) + cd * np.cos(phi)) / (4 * math.pi * r * v_fan)

    a = np.zeros_like(r)
    ga = g(a)
    b = np.where(ga < 0, 2 * omega * r + v_fan, -(omega * r + v_fan))
    gb = g(b)
    for _ in range(60):                                   # widen the bracket where g has not changed sign yet
        same = (np.sign(gb) == np.sign(ga)) & (ga != 0)
        if not same.any():
            break
        b = np.where(same, 2 * b, b)
        gb = np.where(same, g(b), gb)
    done = ga == 0                                        # exactly no swirl (a feathered station)
    b, gb = np.where(done, a, b), np.where(done, ga, gb)
    scale = tolerance * (omega * r + v_fan)
    converged = False
    for _ in range(iterations):
        den = np.where(gb != ga, gb - ga, 1.0)
        c = np.where(done, a, b - gb * (b - a) / den)    # secant step inside the bracket
        gc = g(c)
        flip = np.sign(gc) != np.sign(gb)                 # the root is now between b and c: the old b becomes a
        a = np.where(flip, b, a)
        ga = np.where(flip, gb, 0.5 * ga)                 # Illinois: halve the stale end so it cannot stall
        b, gb = c, gc
        if np.all((np.abs(gc) <= scale) | done):
            converged = True
            break
    return b, converged


def _stations(fan: DuctedFan, airfoil: Airfoil, omega: float, v_fan: float, rho: float, r, chord, beta, dr) -> dict:
    """Blade elements and the useful total-pressure rise at every station for the fan-face velocity ``v_fan``."""
    B = fan.rotor.blades
    c_theta, ok = _swirl(B, r, chord, beta, omega, v_fan, airfoil)
    wt = omega * r - 0.5 * c_theta
    phi = np.arctan2(v_fan, wt)
    alpha = beta - phi
    cl, cd = airfoil.coefficients(alpha)
    W2 = v_fan**2 + wt**2
    euler = rho * omega * r * c_theta                                                        # ideal rise, Pa
    profile = B * 0.5 * rho * W2 * chord * cd * np.sqrt(W2) / (2 * math.pi * r * v_fan)    # drag power / volume flow
    k_swirl = fan.stator_loss if fan.stator_vanes > 0 else 1.0
    swirl = k_swirl * 0.5 * rho * c_theta**2
    tip = 2.0 * (fan.tip_clearance_m / fan.blade_height) * np.abs(euler)
    dp0 = euler - profile - swirl - tip
    w = r * dr                                                                               # annulus weights (V_fan uniform)
    return {"c_theta": c_theta, "phi": phi, "alpha": alpha, "cl": cl, "cd": cd, "W2": W2, "dp0": dp0,
            "dp0_mean": float(np.sum(dp0 * w) / np.sum(w)), "ok": ok}


def solve(fan: DuctedFan, airfoil: Airfoil, rpm: float, airspeed: float = 0.0, rho: float = 1.225, *,
          n_stations: int = 30, speed_of_sound: float = 340.0) -> DuctedPoint:
    """Steady operating point of ``fan`` at ``rpm`` and axial ``airspeed`` [m/s] (see the module docstring)."""
    if rpm <= 0:
        raise ValueError("rpm must be > 0")
    if airspeed < 0:
        raise ValueError("airspeed must be >= 0 (axial inflow only)")
    if rho <= 0 or n_stations < 3:
        raise ValueError("rho must be > 0 and n_stations >= 3")
    omega = rpm * 2 * math.pi / 60
    r, chord, beta, dr = fan.rotor.stations(n_stations)
    sigma, V, A = fan.exit_area_ratio, float(airspeed), fan.fan_area
    u_tip = omega * fan.rotor.radius
    n = int(fan.stages)

    def residual(v_exit: float):
        """Fan rise minus system demand [Pa] at the exit velocity ``v_exit``, and the station state."""
        v_fan = sigma * v_exit
        st = _stations(fan, airfoil, omega, v_fan, rho, r, chord, beta, dr)
        demand = 0.5 * rho * (v_exit**2 - V**2) + (fan.duct_loss + (n - 1) * fan.interstage_loss) * 0.5 * rho * v_fan**2
        return n * st["dp0_mean"] - demand, st

    # scan the fan and system curves; the highest-flow sign change (+ -> -) is the stable operating point
    v_max = V + 3.0 * math.sqrt(n) * u_tip
    grid = v_max * np.geomspace(1e-4, 1.0, 40)
    f = np.array([residual(v)[0] for v in grid])
    up = np.nonzero((f[:-1] > 0) & (f[1:] <= 0))[0]
    converged = up.size > 0
    if converged:
        a, b = float(grid[up[-1]]), float(grid[up[-1] + 1])
        fa, fb = float(f[up[-1]]), float(f[up[-1] + 1])
        p_tol = 1e-11 * 0.5 * rho * (u_tip**2 + V**2)
        converged = False
        for _ in range(200):                              # bracketed regula falsi (Illinois), as in _swirl
            c = b - fb * (b - a) / (fb - fa) if fb != fa else 0.5 * (a + b)
            fc, _ = residual(c)
            if (fc > 0) != (fb > 0):
                a, fa = b, fb
            else:
                fa *= 0.5
            b, fb = c, fc
            if abs(fc) <= p_tol or abs(b - a) <= 1e-13 * b:
                converged = True
                break
        v_exit = b
    else:
        v_exit = float(grid[int(np.argmax(f))])           # the nearest the fan gets to the system curve
    _, st = residual(v_exit)
    converged = converged and st["ok"]

    v_fan = sigma * v_exit
    mdot = rho * A * v_fan
    thrust = mdot * (v_exit - V)
    c_theta, phi, cl, cd, W2 = st["c_theta"], st["phi"], st["cl"], st["cd"], st["W2"]
    B = fan.rotor.blades
    dT_rotor = B * 0.5 * rho * W2 * chord * (cl * np.cos(phi) - cd * np.sin(phi))           # axial blade force / radius
    rotor_thrust = n * float(np.sum(dT_rotor * dr))
    dQ = rho * v_fan * 2 * math.pi * r * c_theta * r                                         # angular momentum flux / radius
    torque = n * float(np.sum(dQ * dr))
    power = torque * omega
    fm = thrust**1.5 / (power * math.sqrt(2 * rho * A)) if V == 0 and power > 0 and thrust > 0 else 0.0
    return DuctedPoint(
        rpm=float(rpm), airspeed=V, rho=rho, thrust=thrust, rotor_thrust=rotor_thrust,
        duct_thrust=thrust - rotor_thrust, torque=torque, power=power,
        efficiency=(thrust * V / power) if power > 0 and V > 0 else 0.0, figure_of_merit=fm,
        mass_flow=mdot, fan_velocity=v_fan, exit_velocity=v_exit, total_pressure_rise=n * st["dp0_mean"],
        tip_mach=u_tip / speed_of_sound, converged=bool(converged),
        r=r, c_theta=c_theta, alpha_deg=np.degrees(st["alpha"]), cl=cl, dp0=st["dp0"],
    )


def rpm_for_thrust(fan: DuctedFan, airfoil: Airfoil, thrust: float, airspeed: float = 0.0, rho: float = 1.225,
                   rpm_max: float = 80000.0, *, n_stations: int = 30, speed_of_sound: float = 340.0) -> DuctedPoint:
    """The rpm at which ``fan`` gives ``thrust`` [N] at ``airspeed``: bisection on rpm between 100 rpm and
    ``rpm_max``, which assumes the thrust rises with rpm (it does on the working branch).

    Raises ValueError if the target lies outside the thrusts at the two ends of that range — above the thrust at
    ``rpm_max``, or below the one at 100 rpm (a static target of ~0 N, or more drag than the nearly stopped fan
    windmills with in forward flight) — or if the point found is not a converged operating point."""
    def at(n):
        return solve(fan, airfoil, n, airspeed, rho, n_stations=n_stations, speed_of_sound=speed_of_sound)

    lo, hi = 100.0, float(rpm_max)
    if not hi > lo:
        raise ValueError(f"rpm_max must be above {lo:.0f} rpm, where the search starts")
    top = at(hi)
    if top.thrust < thrust:
        raise ValueError(f"{thrust:.2f} N is not reachable below {rpm_max:.0f} rpm at {airspeed} m/s "
                         f"({top.thrust:.2f} N at {rpm_max:.0f} rpm)")
    bottom = at(lo)
    if bottom.thrust > thrust:                            # bisection would otherwise slide down to lo and return it
        raise ValueError(f"{thrust:.4g} N is not reachable above {lo:.0f} rpm at {airspeed} m/s "
                         f"({bottom.thrust:.4g} N at {lo:.0f} rpm, the slowest searched)")
    for _ in range(60):
        if hi - lo <= 1e-7 * hi:
            break
        mid = 0.5 * (lo + hi)
        if at(mid).thrust < thrust:
            lo = mid
        else:
            hi = mid
    op = at(hi)
    if not op.converged:
        raise ValueError(f"no converged operating point for {thrust:.2f} N at {airspeed} m/s "
                         f"(solve gave converged=False at {op.rpm:.0f} rpm)")
    return op


def fit_duct_loss(fan: DuctedFan, airfoil: Airfoil, thrust: float, power: float, airspeed: float = 0.0,
                  rho: float = 1.225, *, rpm_max: float = 80000.0, n_stations: int = 30,
                  tolerance: float = 1e-4) -> DuctedFan:
    """``fan`` with the ``duct_loss`` at which it needs the shaft ``power`` [W] to give ``thrust`` [N] at
    ``airspeed`` — a measured point, usually static. The blades, stator and clearance stay as given; the one
    coefficient takes up whatever the model leaves out (inlet lip and struts, low-Reynolds or rough blades, mixing).

    ``power`` is shaft power: a measured electrical power times the motor and ESC efficiencies (~0.8-0.9 together).
    At a fixed thrust the shaft power rises with ``duct_loss`` (almost linearly), so the root of
    ``P(duct_loss) - power`` is bracketed from 0 upwards and refined by Illinois regula falsi until it is within
    ``tolerance`` (relative) of ``power``. Raises ValueError if even ``duct_loss = 0`` needs more than ``power`` (the
    blade, swirl and clearance losses alone exceed it: check the section data, the gap or the motor efficiency), or
    if ``thrust`` stops being reachable below ``rpm_max`` before the power is matched."""
    if not (thrust > 0 and power > 0 and tolerance > 0):
        raise ValueError("thrust, power and tolerance must be > 0")

    def excess(k: float) -> float:
        op = rpm_for_thrust(replace(fan, duct_loss=k), airfoil, thrust, airspeed, rho, rpm_max, n_stations=n_stations)
        return op.power - power

    a, fa = 0.0, excess(0.0)
    if fa > 0:
        raise ValueError(f"{thrust:.2f} N at {airspeed} m/s needs {fa + power:.1f} W even with duct_loss = 0, "
                         f"more than the {power:.1f} W given")
    b = 0.5
    fb = excess(b)
    while fb < 0:                                         # widen the bracket; rpm_for_thrust raises when out of reach
        if b >= 64.0:
            raise ValueError(f"no duct_loss up to {b:.0f} uses {power:.1f} W for {thrust:.2f} N")
        a, fa = b, fb
        b *= 2.0
        fb = excess(b)
    for _ in range(50):                                   # bracketed regula falsi (Illinois), as in solve
        if abs(fb) <= tolerance * power:
            break
        c = b - fb * (b - a) / (fb - fa)
        fc = excess(c)
        if (fc > 0) != (fb > 0):
            a, fa = b, fb
        else:
            fa *= 0.5
        b, fb = c, fc
    else:
        raise ValueError("fit_duct_loss did not converge")
    return replace(fan, duct_loss=b)


def nacelle_drag(fan: DuctedFan, airspeed: float, rho: float = 1.225, nu: float = 1.5e-5) -> float:
    """Friction drag [N] of the nacelle's outside: a turbulent flat plate of area ``external_wetted_area_m2`` with the
    Reynolds number on ``duct_length_m``, Prandtl-Schlichting ``Cf = 0.455 / (log10 Re)^2.58``, ``D = 1/2 rho V^2 Cf S``.
    Turbulent from the leading edge (conservative for a short nacelle that stays partly laminar); the fit is meant for
    Re ~ 1e5..1e9, so Re is floored at 1e4. Subtract it from ``DuctedPoint.thrust`` for the net propulsive force.
    0 with no wetted area or no airspeed."""
    if airspeed < 0 or rho <= 0 or nu <= 0:
        raise ValueError("airspeed must be >= 0, rho and nu > 0")
    S, L = fan.external_wetted_area_m2, fan.duct_length_m
    if S <= 0 or airspeed == 0:
        return 0.0
    re = max(airspeed * L / nu, 1e4)
    cf = 0.455 / math.log10(re) ** 2.58
    return 0.5 * rho * airspeed**2 * cf * S
