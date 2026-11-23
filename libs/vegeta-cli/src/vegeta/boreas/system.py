"""Propeller + motor + battery: throttle to rpm, current and power; excitation summary for structures."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .airfoil import Airfoil
from .bemt import OperatingPoint, rpm_for_thrust, solve
from .motor import Battery, Motor
from .propeller import Propeller


@dataclass
class SystemPoint:
    throttle: float
    voltage: float
    rpm: float
    current: float
    electrical_power: float
    motor_efficiency: float
    aero: OperatingPoint
    current_limited: bool

    @property
    def thrust(self) -> float:
        return self.aero.thrust

    def to_dict(self) -> dict:
        d = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in self.__dict__.items()
             if k != "aero"}
        d["aero"] = {k: v for k, v in self.aero.to_dict().items() if k != "radial"}
        return d


class Propulsion:
    """One propeller on one motor fed by a battery. Steady state only."""

    def __init__(self, prop: Propeller, airfoil: Airfoil, motor: Motor, battery: Battery, rho: float = 1.225):
        self.prop, self.airfoil, self.motor, self.battery, self.rho = prop, airfoil, motor, battery, rho

    def at_throttle(self, throttle: float, airspeed: float = 0.0) -> SystemPoint:
        """Operating point at a throttle fraction (0..1 of battery voltage), axial airspeed [m/s]."""
        if not 0 < throttle <= 1:
            raise ValueError("throttle must be in (0, 1]")
        volts = throttle * self.battery.voltage
        lo, hi = 50.0, self.motor.kv_rpm_per_volt * volts       # no-load rpm is the upper bound
        limited = False
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            need, cur = self.motor.voltage_for(mid, solve(self.prop, self.airfoil, mid, airspeed, self.rho).torque)
            if need < volts:
                lo = mid
            else:
                hi = mid
        rpm = lo
        aero = solve(self.prop, self.airfoil, rpm, airspeed, self.rho)
        volts_used, current = self.motor.voltage_for(rpm, aero.torque)
        if current > self.motor.max_current_a:
            limited = True
        return SystemPoint(throttle, volts_used, rpm, current, volts_used * current,
                           self.motor.efficiency(rpm, aero.torque), aero, limited)

    def for_thrust(self, thrust: float, airspeed: float = 0.0) -> SystemPoint:
        """Throttle needed for ``thrust`` [N] (raises if full throttle is not enough): the rpm that gives the
        thrust, then the voltage the motor needs to hold that rpm against the propeller torque."""
        rpm_max = self.motor.kv_rpm_per_volt * self.battery.voltage          # no-load rpm at full voltage
        try:
            aero = rpm_for_thrust(self.prop, self.airfoil, thrust, airspeed, self.rho, rpm_max=rpm_max)
        except ValueError:
            aero = None
        volts, current = self.motor.voltage_for(aero.rpm, aero.torque) if aero is not None else (float("inf"), 0.0)
        if volts > self.battery.voltage:
            raise ValueError(f"{thrust:.2f} N not reachable at full throttle and {airspeed} m/s")
        return SystemPoint(volts / self.battery.voltage, volts, aero.rpm, current, volts * current,
                           self.motor.efficiency(aero.rpm, aero.torque), aero, current > self.motor.max_current_a)

    def sweep(self, throttles, airspeed: float = 0.0) -> list[SystemPoint]:
        return [self.at_throttle(t, airspeed) for t in throttles]


def unbalance_force(rotor_mass_kg: float, rpm: float, grade_mm_s: float = 6.3) -> float:
    """Rotating force [N] from a residual unbalance of ISO 21940 grade G ``grade_mm_s``
    (e * omega = G, so F = m e omega^2 = m G omega)."""
    omega = rpm * 2 * math.pi / 60
    return rotor_mass_kg * (grade_mm_s / 1000) * omega


def excitations(prop: Propeller, rpm: float, grade_mm_s: float = 6.3) -> dict:
    """Frequencies and forces a structure sees from this rotor at ``rpm``."""
    f1 = rpm / 60
    return {"rpm": rpm, "shaft_hz": f1, "blade_pass_hz": prop.blades * f1,
            "unbalance_force_n": unbalance_force(prop.rotor_mass_kg, rpm, grade_mm_s),
            "unbalance_grade_mm_s": grade_mm_s}
