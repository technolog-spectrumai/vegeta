"""Brushless DC motor as the usual first-order model: Kv, winding resistance, no-load current."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Motor:
    name: str
    kv_rpm_per_volt: float
    resistance_ohm: float
    no_load_current_a: float
    max_current_a: float
    mass_kg: float = 0.0
    source: str = "datasheet-style values"

    def __post_init__(self):
        if min(self.kv_rpm_per_volt, self.resistance_ohm, self.max_current_a) <= 0 or self.no_load_current_a < 0:
            raise ValueError("kv, resistance and max current must be > 0; no-load current >= 0")

    @property
    def kt(self) -> float:
        """Torque constant [N m / A] = 1 / Kv in rad/s per volt."""
        return 60 / (2 * math.pi * self.kv_rpm_per_volt)

    def voltage_for(self, rpm: float, torque: float) -> tuple[float, float]:
        """``(voltage, current)`` needed to hold ``rpm`` against a shaft ``torque`` [N m]."""
        current = self.no_load_current_a + torque / self.kt
        return rpm / self.kv_rpm_per_volt + current * self.resistance_ohm, current

    def efficiency(self, rpm: float, torque: float) -> float:
        v, i = self.voltage_for(rpm, torque)
        return (torque * rpm * 2 * math.pi / 60) / (v * i) if v * i > 0 else 0.0


@dataclass(frozen=True)
class Battery:
    name: str
    cells: int
    capacity_ah: float
    cell_voltage_nominal: float = 3.7
    usable_fraction: float = 0.8
    mass_kg: float = 0.0

    @property
    def voltage(self) -> float:
        return self.cells * self.cell_voltage_nominal

    @property
    def energy_wh(self) -> float:
        return self.voltage * self.capacity_ah

    @property
    def usable_wh(self) -> float:
        return self.energy_wh * self.usable_fraction
