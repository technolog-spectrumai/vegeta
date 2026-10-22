"""og.batteries — cells and packs: what they store, what they sag to, what they accept.

* ``LiIonCell`` (alias ``Cell``): open-circuit voltage against charge, DC resistance doubling every ``r_double_K``
  below 25 °C, the usable-capacity cold factor; ``CELLS`` the catalogue (sourced).
* ``LiIonPack`` (``Pack``): ``series`` x ``parallel`` cells with wiring resistance and packing mass — terminal voltage
  under load, the power limit, usable energy at a current and temperature, the charge (regeneration) limit.
* ``Battery``: a bought pack described by its label (cells, mAh, C, mass, size); ``BATTERIES`` the catalogue.

Moved from Vegeta's ``nisus_plus_systems`` (cells, packs) and ``nisus_systems`` (bought packs); reproduced exactly
(tests/test_batteries.py). Units as in the names: V, A, Ah/mAh, g, Wh, °C, Ω.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SNIPPET = "search-index snippet of the cited page (2026-10-10); page not opened; availability not confirmed"


@dataclass(frozen=True)
class LiIonCell:
    """A 21700 cell's electrical model: the open-circuit voltage against the state of charge (a generic high-power NMC
    curve — assumed shape, fitted to the datasheet's 4.2 V / 3.6 V nominal / 2.5 V ends), the DC resistance at 25 °C
    and its growth in the cold (doubling every ``r_double_K`` below 25 °C — assumed, the usual Arrhenius-like trend),
    the capacity's loss in the cold (assumed), the continuous discharge and the maximum charge currents."""
    name: str
    capacity_ah: float
    mass_g: float
    r_dc_25: float
    i_max_a: float
    i_charge_max_a: float
    v_nominal: float = 3.6
    v_min: float = 2.5
    r_double_K: float = 20.0
    source: str = ""
    url: str = ""

    SOC = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
    OCV = (2.80, 3.25, 3.40, 3.50, 3.57, 3.63, 3.69, 3.77, 3.86, 3.96, 4.07, 4.20)

    def ocv(self, soc):
        return np.interp(np.clip(soc, 0.0, 1.0), self.SOC, self.OCV)

    def r(self, T_C):
        return self.r_dc_25 * 2.0 ** ((25.0 - min(T_C, 25.0)) / self.r_double_K)

    def capacity_factor(self, T_C):
        """Usable capacity against 25 °C: 1.0 at ≥ 25 °C, 0.95 at 5 °C, 0.88 at −10 °C, 0.78 at −20 °C (assumed, NMC class)."""
        return float(np.interp(T_C, (-20.0, -10.0, 5.0, 25.0), (0.78, 0.88, 0.95, 1.0)))


CELLS = {
    "P45B": LiIonCell("Molicel INR21700-P45B", 4.5, 70.0, 0.015, 45.0, 13.5,
                      source=f"4500 mAh typ (4300 min), 45 A, charge 4.5 A std / 13.5 A max (Molicel sheet via imrbatteries.com); DC 15 mΩ at 10 A "
                             f"(Battery Junction); 70 g; {SNIPPET}", url="https://www.imrbatteries.com/content/molicel_p45b.pdf"),
    "P50B": LiIonCell("Molicel INR21700-P50B", 5.0, 70.0, 0.015, 60.0, 10.0,
                      source=f"5000 mAh, 60 A (akkuteile.de listing); resistance as P45B and charge 2C assumed; 70 g assumed; {SNIPPET}",
                      url="https://akkuteile.de/en/lithium-ionen-battery/size-21700/molicel/molicel-inr21700-p50b-5000mah-60a-3-6-3-7v-li-ion-battery_100638_3507"),
}
CHARGE_MIN_C = 5.0                    # no charging (regeneration) below this cell temperature: lithium plating (assumed margin over the 0 °C datasheet limit)


@dataclass(frozen=True)
class LiIonPack:
    """``series`` x ``parallel`` cells with the pack's own resistance (nickel strips, wires, connector) and its packing
    mass (strips, wrap, balance lead, XT90: ``pack_overhead`` of the cells' mass)."""
    key: str
    cell: LiIonCell
    series: int = 6
    parallel: int = 3
    wiring_r: float = 0.006
    pack_overhead: float = 0.08

    @property
    def name(self):
        return f"{self.series}S{self.parallel}P {self.cell.name}"

    @property
    def cells(self):
        return self.series

    @property
    def mass_g(self):
        return self.series * self.parallel * self.cell.mass_g * (1 + self.pack_overhead)

    @property
    def capacity_ah(self):
        return self.parallel * self.cell.capacity_ah

    @property
    def voltage(self):
        return self.series * self.cell.v_nominal

    @property
    def energy_wh(self):
        return self.voltage * self.capacity_ah

    @property
    def max_current_a(self):
        return self.parallel * self.cell.i_max_a

    @property
    def max_charge_a(self):
        return self.parallel * self.cell.i_charge_max_a

    @property
    def size_mm(self):
        """Cells lying across the pod (70 mm along y), ``series * parallel / 2`` along x, two layers."""
        n = self.series * self.parallel
        return (math.ceil(n / 2) * 21.5 + 6, 76.0, 2 * 21.5 + 4)

    def resistance(self, T_C=25.0):
        return self.series * self.cell.r(T_C) / self.parallel + self.wiring_r

    def ocv(self, soc):
        return self.series * self.cell.ocv(soc)

    def terminal_v(self, current_a, soc, T_C=25.0):
        """Loaded voltage [V] at a discharge current (negative: charging)."""
        return self.ocv(soc) - current_a * self.resistance(T_C)

    def max_power_w(self, soc, T_C=25.0):
        """The most electrical power the pack gives at the cells' current limit or at the cut-off voltage (2.9 V/cell under load)."""
        R, E = self.resistance(T_C), self.ocv(soc)
        I = min(self.max_current_a, (E - self.series * 2.9) / R)
        return max(I, 0.0) * (E - I * R)

    def usable_wh(self, T_C=25.0, current_a=20.0):
        """The energy delivered to the load from full to the cut-off at a constant current and temperature: the capacity's
        cold factor and the resistive loss (I² R) taken off; cut-off at 3.0 V/cell under load."""
        soc = np.linspace(1.0, 0.0, 400)
        v = self.terminal_v(current_a, soc, T_C)
        ok = v >= 3.0 * self.series
        if not ok.any():
            return 0.0
        dq = self.capacity_ah * self.cell.capacity_factor(T_C) / (len(soc) - 1)
        return float(np.sum(v[ok][1:] * current_a * dq / current_a))

    def charge_limit_a(self, T_C=25.0, soc=0.5):
        """The charge current the pack accepts: none below ``CHARGE_MIN_C`` or above 95 % charge, else the cells' maximum."""
        if T_C < CHARGE_MIN_C or soc > 0.95:
            return 0.0
        return self.max_charge_a


Cell = LiIonCell
Pack = LiIonPack


@dataclass(frozen=True)
class Battery:
    key: str
    name: str
    cells: int
    chemistry: str
    capacity_mah: float
    mass_g: float
    size_mm: tuple
    c_rating: float
    price: str
    url: str
    availability: str = "not confirmed"
    cell_v_nominal: float = 3.7
    note: str = ""

    @property
    def voltage(self):
        return self.cells * self.cell_v_nominal

    @property
    def energy_wh(self):
        return self.voltage * self.capacity_mah / 1000

    @property
    def max_current_a(self):
        return self.c_rating * self.capacity_mah / 1000


BATTERIES = [
    Battery("gens-ace-3s-2200", "Gens Ace 3S 2200 mAh 45C G-Tech", 3, "LiPo", 2200, 190.0, (105, 34, 23), 45, "€24.99 (gensace.de)",
            "https://gensace.de/collections/3s-lipo-battery", note=f"the reference pack; {SNIPPET}"),
    Battery("tattu-3s-2200", "Tattu 3S 2200 mAh 45C", 3, "LiPo", 2200, 190.0, (107, 36, 24), 45, "not stated",
            "https://www.motionew.com/shop/power-solution/battery-lipo-3s-2200mah-45c-tattu/", note=SNIPPET),
    Battery("gnb-3s-3000-lihv", "GNB 3S 3000 mAh 120C LiHV", 3, "LiHV", 3000, 213.0, (109, 34, 30), 120, "not stated",
            "https://www.gaoneng.shop/products/gaoneng-gnb-lihv-3s-11.4v-3000mah-120c-xt60-lipo-battery", cell_v_nominal=3.8,
            note=f"LiHV: 3.8 V nominal, 4.35 V charged — the ESC and the regulators must accept 13.05 V (they do: 2-8S); {SNIPPET}"),
    Battery("turnigy-3s-3000-nano", "Turnigy nano-tech 3S 3000 mAh 30C", 3, "LiPo", 3000, 215.0, (135, 44, 17), 30, "not stated",
            "https://hobbyking.com/turnigy-nano-tech-3000mah-3s-30c-lipo-pack-wxt60.html", note=SNIPPET),
    Battery("tattu-3s-3300", "Tattu 3S 3300 mAh 30C", 3, "LiPo", 3300, 248.0, (141, 44, 21), 30, "US$47.00",
            "https://www.motionew.com/shop/power-solution/battery-lipo-3s-3300mah-30c-tattu/", note=SNIPPET),
    Battery("gens-ace-3s-3300-60c", "Gens Ace 3S 3300 mAh 60C G-Tech", 3, "LiPo", 3300, 267.0, (136.5, 42.6, 21.6), 60, "not stated",
            "https://genstattu.com/gens-ace-g-tech-3300mah-3s-11-1v-60c-lipo-battery-pack-with-xt60-plug/", note=SNIPPET),
    Battery("turnigy-4s-2200", "Turnigy 4S 2200 mAh 40C", 4, "LiPo", 2200, 216.0, (106, 35, 29), 40, "not stated",
            "https://hobbyking.com/turnigy-2200mah-4s-40c-lipo-pack-w-xt60.html", note=f"4S: the motor needs a lower KV or a smaller propeller (the X2216 KV1250 on 9x6 at 14.8 V exceeds 40 A static); {SNIPPET}"),
    Battery("liion-3s2p-p42a", "Li-ion 3S2P Molicel P42A (assembled, US shop)", 3, "Li-ion", 8400, 450.0, (70, 48, 67), 10.7, "US$102",
            "https://nexusbatterysystems.com/products/3s2p-42a", cell_v_nominal=3.6,
            note=f"90 A pack claim (45 A per cell); 10.8 V nominal, 9.0 V at the end: the propulsion map at 11.1 V flatters it by ~5 %; no EU seller with a stated mass found; {SNIPPET}"),
]


def battery(key: str) -> Battery:
    return next(b for b in BATTERIES if b.key == key)


@dataclass(frozen=True)
class NominalPack:
    """A pack by its nominal numbers: cells in series × capacity, a usable fraction (Boreas's ``Battery``)."""
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


