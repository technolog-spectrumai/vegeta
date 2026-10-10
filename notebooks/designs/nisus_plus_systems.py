"""NISUS+ systems — the atmosphere, the components with their sources, the mass budget, the centre of gravity and
inertia, the electrical loads, the Li-ion pack (its sag and the cold), the drive over altitude with its brake and
regeneration region, and the mountain mission's energy (notebook 33).

Everything is a number with a stated origin, as in ``nisus_systems`` (whose dataclasses, materials and Jetson options
this module reuses): **calculated** (CAD volumes x densities, Boreas BEMT, the ISA), **sourced** (a published figure:
the component table carries the URL; the research of 2026-10-10 read search-index snippets only, **no availability is
confirmed** — ``data/nisus_plus_component_sources.md``) or **assumed** (an engineer's first estimate, labelled).

What changes against NISUS, and why:

- **the air thins with height**: ``atmosphere(h, dT)`` (the ISA troposphere of ``vegeta.boreas.microjet.isa`` with a
  temperature offset, plus Sutherland's viscosity) feeds every power, thrust and Reynolds number below;
- **the drive keeps its sign**: ``NisusPlusDrive`` holds the signed Boreas BEMT table of the 15x8 (thrust and torque go
  negative at low rpm and high airspeed: the windmill) and the fitted AT4125 KV540; at density ρ the propeller's thrust
  and torque scale with ρ/ρ0 at fixed (V, rpm) (Boreas has no Reynolds or Mach effect: stated, not hidden), the motor and
  ESC follow. Low throttle at speed **brakes**: the ESC's active freewheeling holds a voltage below the motor's back-EMF,
  the current reverses and charges the pack (**regeneration**), within the ESC's and the pack's charge-current limits
  and never into a cold pack (lithium plating below ``CHARGE_MIN_C``); throttle 0 is the short-circuit brake (all the
  energy into the windings);
- **the pack sags and gets cold**: ``LiIonPack`` (21700 cells in 6S nP): open-circuit voltage against the state of
  charge, resistance and capacity against temperature — the loaded voltage caps the full-throttle rpm;
- **the mission climbs and descends**: ``mission_energy`` climbs from the launch site to the work altitude step by
  step at that altitude's density, surveys at altitude, flies back, comes down fast (crow and the propeller brake:
  the regeneration in its own row) and lands; the return trigger knows the height is energy.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import nisus_plus
import nisus_systems as ns
from nisus_systems import Component, MATERIALS, REGULATOR_EFF, JETSON_OPTIONS

G = 9.81
RHO0 = 1.225
DATA = Path(__file__).resolve().parent / "data"
ACCESS = "2026-10-10"
SNIPPET = "search-index snippet of the cited page (2026-10-10); page not opened; availability not confirmed"
ZERO = ("Zero",)


# ================================================================================================= the atmosphere
def atmosphere(h_m: float = 0.0, dT: float = 0.0) -> dict:
    """The ISA troposphere at the geopotential altitude ``h_m`` [m] with a temperature offset ``dT`` [K] (a hot day:
    +15 to +25 in summer in the Alps): density, pressure, temperature, dynamic and kinematic viscosity (Sutherland),
    the speed of sound, σ = ρ/ρ0 and the density altitude."""
    from vegeta.boreas.microjet import isa
    a = isa(float(h_m), float(dT))
    T, rho = a.temperature, a.density
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    return {"h_m": float(h_m), "dT": float(dT), "T_K": T, "T_C": T - 273.15, "p_Pa": a.pressure, "rho": rho, "mu": mu, "nu": mu / rho,
            "a": a.speed_of_sound, "sigma": rho / RHO0, "density_altitude_m": density_altitude(rho)}


def density_altitude(rho: float) -> float:
    """The ISA altitude [m] with density ``rho`` (inverse of the standard troposphere)."""
    return 288.15 / 0.0065 * (1 - (rho / RHO0) ** (1 / 4.2559))


def atmosphere_table(heights=(0, 1000, 1500, 2000, 3000, 4000, 4500, 5000, 6000), dT=0.0) -> pd.DataFrame:
    rows = {f"{h:.0f} m": atmosphere(h, dT) for h in heights}
    df = pd.DataFrame(rows).T[["T_C", "p_Pa", "rho", "sigma", "nu", "a", "density_altitude_m"]]
    df.columns = ["T [°C]", "p [Pa]", "ρ [kg/m³]", "σ", "ν [m²/s]", "a [m/s]", "density altitude [m]"]
    return df


# ================================================================================================= CAD numbers
#: CAD volumes [mm³] of the default Nisus+ parts (``nisus_plus.NisusPlus().generate(part=...).measure()['volume']``); ``cad_numbers(
#: recompute=True)`` rebuilds them (about a minute).
CAD = {"aircraft": 22950563.0, "wing": 12783897.0, "spar": 174536.0, "rear_spar": 60363.0, "pod": 400388.0, "nose": 31160.0, "boom": 52527.0,
       "boom_fitting": 143789.0, "tail": 1721585.0, "tail_fitting": 31193.0, "motor_mount": 17681.0, "tray": 35216.0, "skid": 183580.0,
       "battery_tray": 71320.0, "flap": 224959.0, "aileron": 277388.0, "spar_joiner": 18871.0}


def cad_numbers(p: dict | None = None, *, recompute: bool = False) -> dict:
    if not recompute and not p:
        return dict(CAD)
    d = nisus_plus.NisusPlus()
    return {part: float(d.generate(**nisus_plus.overrides(p), part=part).measure()["volume"]) for part in CAD}


# ================================================================================================= components
#: the 6S pack's cells: the electrical model's data (``LiIonCell``); the packs are ``PACKS``
# positions: the aircraft frame of ``nisus_plus`` (x aft from the wing root LE, z up from the pod top), mm
COMPONENTS = [
    # --- propulsion
    Component("motor T-Motor AT4125 KV540", "propulsion/ESC/wiring", 355.0, ZERO, 387.0, 40.0, supply="battery", supply_v=21.6, price_eur=95.0,
              url="https://store.tmotor.com/goods.php?id=827",
              source=f"355 g with cable, 6S, 85 A / 2000 W for 180 s (T-Motor datasheet via ligpower.com / robotshop PDF); APC 15x8 at 100 %: "
                     f"21.36 V, 72.77 A, 1554 W, 8879 rpm, 5537 gf (T-Motor table, 2017 test platform); {SNIPPET}",
              note="the published full-throttle point fits the motor model (nisus_plus_systems.motor_model); price assumed"),
    Component("propeller APC 15x8E (fixed: it brakes)", "propulsion/ESC/wiring", 46.0, ZERO, 418.0, 40.0, price_eur=15.0,
              url="https://www.motionrc.com/products/apc-15x8-thin-electric-propeller-black-lpb15080e",
              source=f"47.9 g (Motion RC, black LPB15080E) / ~44 g (Gator-RC, LP15080E); 46 g taken; {SNIPPET}",
              note="not a folding propeller: a folding one folds back when braked and recovers nothing"),
    Component("ESC 80 A 3-8S, AM32 firmware (active freewheeling, brake)", "propulsion/ESC/wiring", 50.0, ZERO, 300.0, -30.0, supply="battery", supply_v=21.6,
              p_idle_w=0.5, p_typ_w=0.5, p_peak_w=0.5, price_eur=60.0, kind="assumed",
              url="https://www.rotorama.com/product/pilotix-esc-4in1-pilotix-80a-am32-8s-cz",
              source=f"80 A continuous / 100 A peak, 3-8S AM32 class (Pilotix listing — a 4-in-1 board: a single-motor 80 A AM32 ESC of the "
                     f"same class is assumed, 50 g with wires and cap); AM32's 'damped light' mode does regenerative braking (Hobbywing 65A "
                     f"AM32 listing) — the energy returned to the pack is NOT documented by any vendor: to measure; {SNIPPET}",
              note="8S rating: the regen voltage spikes on a 6S bus stay below its limit; quiescent 0.5 W assumed"),
    Component("wiring 10 AWG, XT90, bullets, leads, capacitor", "propulsion/ESC/wiring", 70.0, ZERO, 100.0, -70.0, kind="assumed",
              source="estimate: 10 AWG battery lead 2 x 250 mm, XT90 pair, 3 x 12 AWG motor leads 300 mm, bullets, a 1000 µF low-ESR bank"),
    Component("battery straps, velcro, foam, insulating sleeve", "battery/retention", 30.0, ZERO, -25.0, -110.0, kind="assumed",
              source="estimate: two 25 mm straps with buckles, velcro pads, 5 mm closed-cell foam sleeve (keeps the pack warm at altitude)"),
    # --- actuation: six wing/tail servos of the 20-25 g metal-gear class
    *[Component(f"servo 22 g metal gear ({name})", "servos/linkages", 22.0, ZERO, x, 5.0, y_mm=y, supply="servo", supply_v=7.4,
                p_idle_w=0.2, p_typ_w=1.0, p_peak_w=12.0, price_eur=25.0, kind="assumed",
                source="assumed: a 20-25 g metal-gear wing servo, 6-8 kg·cm at 7.4 V, 0.10 s/60° (KST / Emax class); not researched",
                note="typical 1.0 W = 20 % of the time moving; peak = stall 1.6 A at 7.4 V")
      for name, x, y in (("flap L", 230.0, -405.0), ("flap R", 230.0, 405.0), ("aileron L", 150.0, -880.0), ("aileron R", 150.0, 880.0),
                         ("elevator", 200.0, -180.0), ("rudders, torque rod", 200.0, 180.0))],
    Component("horns, clevises, pushrods, hinges", "servos/linkages", 40.0, ZERO, 400.0, -15.0, kind="assumed",
              source="estimate: 6 horns, 2 mm carbon pushrods (two inside the booms, 780 mm), the rudder cross rod, hinge tape"),
    Component("servo BEC 8 A, 7.4 V", "servos/linkages", 18.0, ZERO, 80.0, -80.0, kind="assumed",
              source="estimate: a separate 8 A switching BEC for six servos (the flight controller's rail would be marginal), 85 %"),
    # --- flight control
    Component("flight controller Matek H743-WING (ArduPilot)", "flight controller/receiver/GNSS/wiring", 28.0, ZERO, 35.0, -55.0,
              supply="battery", supply_v=21.6, p_idle_w=1.5, p_typ_w=1.8, p_peak_w=2.5, price_eur=80.0, kind="assumed",
              url="https://www.mateksys.com/?portfolio=h743-wing-v3",
              source="assumed: the H743-WING class (6-36 V input, 5 V/9 V BECs, current sensor, ArduPilot Plane with TECS and terrain following); not researched"),
    Component("GNSS + compass Matek M10Q-5883", "flight controller/receiver/GNSS/wiring", 8.0, ZERO, -330.0, -5.0, supply="5V", supply_v=5.3,
              p_idle_w=0.25, p_typ_w=0.25, p_peak_w=0.35, price_eur=25.0, url="https://www.unmannedtechshop.co.uk/products/matek-m10q-5883-gps-module",
              source="as NISUS's (nisus_component_sources.md)"),
    Component("airspeed sensor (pitot-static, I2C differential pressure)", "flight controller/receiver/GNSS/wiring", 12.0, ZERO, -395.0, -70.0, supply="5V", supply_v=5.3,
              p_idle_w=0.05, p_typ_w=0.05, p_peak_w=0.05, price_eur=40.0, kind="assumed",
              source="assumed: a pitot tube in the nose cone with an MS4525-class sensor — essential above 1500 m: the controller flies the "
                     "indicated (equivalent) airspeed, the stall and V_NE are EAS"),
    Component("lidar rangefinder (40 m) for the landing", "flight controller/receiver/GNSS/wiring", 50.0, ZERO, -360.0, -130.0, supply="5V", supply_v=5.3,
              p_idle_w=1.0, p_typ_w=1.0, p_peak_w=1.0, price_eur=80.0, kind="assumed",
              source="assumed: a Benewake TF02-Pro-class lidar (40 m, ~1 W) looking down: the height over the meadow in the flare (the "
                     "DEM is not good to a metre)"),
    Component("RC receiver ELRS 900 MHz", "flight controller/receiver/GNSS/wiring", 5.0, ZERO, 50.0, -40.0, supply="5V", supply_v=5.3,
              p_idle_w=0.5, p_typ_w=0.5, p_peak_w=0.8, price_eur=25.0, kind="assumed",
              source="assumed: 900 MHz diffracts over ridges better than 2.4 GHz"),
    Component("telemetry RFD900x (900 MHz, 1 W)", "flight controller/receiver/GNSS/wiring", 15.0, ZERO, 60.0, -40.0, supply="5V", supply_v=5.3,
              p_idle_w=0.4, p_typ_w=2.0, p_peak_w=5.0, price_eur=120.0, kind="assumed",
              source="assumed: RFD900x class, 14.5 g, up to 1 W output; typical 2 W at a 30 % transmit duty"),
    Component("signal wiring, connectors, antenna mounts", "flight controller/receiver/GNSS/wiring", 25.0, ZERO, 40.0, -40.0, kind="assumed",
              source="estimate"),
    # --- the computer (the Orin Nano Super installation: NISUS's JETSON_OPTIONS entry)
    Component("Jetson Orin Nano Super: module + carrier + heatsink/fan", "computer", 230.0, ZERO, -240.0, -50.0, supply="jetson", supply_v=12.0,
              p_idle_w=7.0, p_typ_w=15.0, p_peak_w=25.0, price_eur=None, url="https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit",
              source=f"NISUS's JETSON_OPTIONS['orin_nano_super']: 7/15/25 W modes, 9-19 V input, JetPack 6.2; mass not published "
                     f"(the kit with its heatsink ~250 g, 230 g taken without the case); {SNIPPET}",
              note="the installation's battery-side power is JETSON_INSTALLATION_W (15 W mode + camera + buck)"),
    Component("buck 6S → 12 V / 5 A for the Orin", "computer", 30.0, ZERO, -190.0, -100.0, kind="assumed",
              source="assumed: the Orin kit takes 9-19 V and a full 6S pack is 25.2 V: a 12 V / 5 A synchronous buck, 92 %"),
    Component("vision camera IMX477 (12 MP, CSI) + 6 mm lens + cable", "computer", 35.0, ZERO, -375.0, -40.0, supply="jetson", supply_v=5.0,
              p_idle_w=0.4, p_typ_w=0.7, p_peak_w=0.8, price_eur=60.0, kind="assumed", source="assumed: Raspberry Pi HQ camera class"),
    Component("data link (LTE stick + antenna)", "computer", 25.0, ZERO, 120.0, -10.0, supply="jetson", supply_v=5.0,
              p_idle_w=1.0, p_typ_w=2.0, p_peak_w=3.0, price_eur=40.0, kind="assumed", source="as NISUS's (the computer's results and a low-rate video)"),
    Component("storage (NVMe 256 GB) and the onboard DEM", "computer", 10.0, ZERO, -240.0, -50.0, kind="assumed",
              source="assumed: the terrain model of the mission area (a 30 m DEM clip) lives on the Orin; ArduPilot's terrain database on its SD card"),
    Component("Orin mounting frame (printed), cooling duct, vibration dampers", "joints/mounts/adhesive", 25.0, ZERO, -240.0, -80.0, kind="assumed",
              source="estimate: PETG rails behind the nose cone, an inlet duct from a NACA vent to the fan"),
]
JETSON = JETSON_OPTIONS["orin_nano_super"]
JETSON_INSTALLATION_W = 22.0          # battery-side: 15 W mode + camera + data link share + the buck at 92 % (assumed)
JETSON_INSTALLATION_RANGE_W = (15.0, 30.0)
ALLOWANCE_FRACTION = 0.08


def components() -> list:
    return list(COMPONENTS)


def component_table() -> pd.DataFrame:
    rows = [{"item": c.key, "group": c.group, "mass [g]": c.mass_g, "x [mm]": c.x("Zero"), "supply": c.supply, "P typ [W]": c.p_typ_w,
             "price [EUR]": c.price_eur, "kind": c.kind, "availability": c.availability, "url": c.url, "source": c.source}
            for c in COMPONENTS]
    return pd.DataFrame(rows).set_index("item")


# ================================================================================================= the Li-ion pack
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


PACKS = {
    "6s3p-p45b": LiIonPack("6s3p-p45b", CELLS["P45B"], 6, 3),
    "6s4p-p45b": LiIonPack("6s4p-p45b", CELLS["P45B"], 6, 4),
    "6s3p-p50b": LiIonPack("6s3p-p50b", CELLS["P50B"], 6, 3),
    "6s5p-p45b": LiIonPack("6s5p-p45b", CELLS["P45B"], 6, 5),
}
DEFAULT_PACK = "6s4p-p45b"


def pack(key: str = DEFAULT_PACK) -> LiIonPack:
    return PACKS[key]


def pack_table(T_C=(25.0, 5.0, -10.0)) -> pd.DataFrame:
    rows = {}
    for k, b in PACKS.items():
        r = {"pack": b.name, "cells": b.series * b.parallel, "V nom": b.voltage, "Ah": b.capacity_ah, "Wh nominal": b.energy_wh,
             "mass [g]": b.mass_g, "Wh/kg": b.energy_wh / b.mass_g * 1000, "max A (cells)": b.max_current_a, "max charge A": b.max_charge_a,
             "R at 25 °C [mΩ]": 1000 * b.resistance(25.0), "size [mm]": "x".join(f"{v:.0f}" for v in b.size_mm)}
        for T in T_C:
            r[f"usable Wh at {T:g} °C, 20 A"] = b.usable_wh(T, 20.0)
            r[f"sag at 70 A, {T:g} °C [V]"] = 70.0 * b.resistance(T)
        rows[k] = r
    return pd.DataFrame(rows).T


def battery_fits(b: LiIonPack, p=None) -> dict:
    p = nisus_plus.resolve(p)
    x0, x1, w, h = nisus_plus.NisusPlus.bays(p)["battery bay"]
    L_bay = x1 - x0
    dims = b.size_mm
    fits = dims[0] <= L_bay - 6 and dims[1] <= p["battery_tray_width"] and dims[2] <= h - 50      # the flight controller tray sits above the pack
    return {"bay [mm]": (L_bay, w, h), "pack [mm]": tuple(round(d) for d in dims), "fits": bool(fits)}


# ================================================================================================= mass table
TAIL_SPAR = dict(od=6.0, id_=5.0)
TIP_RODS = dict(od=8.0, id_=6.0, length=200.0, count=2)
PRINT_FILL = dict(ns.PRINT_FILL)


def structure_items(p=None, cad: dict | None = None) -> list:
    """The airframe's parts (item, group, mass_g, x, y, z, basis) from the CAD volumes and the materials — NISUS's
    recipe on NISUS+'s geometry: an XPS core (25 % lightened) with a 2 mm balsa D-box over the whole span (the bigger wing
    flies faster and twists more), the carbon spar, joiners and tip rods, film; the flaps and ailerons are foam with a
    balsa trailing edge."""
    p = nisus_plus.resolve(p)
    cad = cad or CAD
    L = nisus_plus.NisusPlus.layout(p)
    m = MATERIALS
    x_wing = L["x_mac_le"] + 0.42 * L["mac"]
    prof = nisus_plus.NisusPlus.pod_profile(p)
    per = math.pi * (prof[:, 1] + prof[:, 2])
    x_pod = float(np.trapezoid(prof[:, 0] * per, prof[:, 0]) / np.trapezoid(per, prof[:, 0]))
    z_pod = float(np.trapezoid(prof[:, 3] * per, prof[:, 0]) / np.trapezoid(per, prof[:, 0]))
    w = nisus_plus.wetted_areas(p)
    foam = cad["wing"] * 1e-3 * m["XPS foam 30"]["rho"] * (1 - ns.FOAM_LIGHTENING)
    balsa = 2 * 0.30 * 0.5 * (p["root_chord"] + p["tip_chord"]) * p["span"] * 2.0 * 1e-3 * m["balsa"]["rho"] * 1.05 + 25.0
    carbon = m["carbon tube"]["rho"]
    rods = TIP_RODS["count"] * math.pi / 4 * (TIP_RODS["od"] ** 2 - TIP_RODS["id_"] ** 2) * TIP_RODS["length"] * 1e-3 * carbon
    film_wing = m["covering film"]["areal_g_m2"] * w["wing"]
    x_tail = L["tail_le"] + 0.4 * p["tail_chord"]
    tail_foam = cad["tail"] * 1e-3 * m["XPS foam 30"]["rho"]
    tail_spar = math.pi / 4 * (TAIL_SPAR["od"] ** 2 - TAIL_SPAR["id_"] ** 2) * p["tail_span"] * 1e-3 * carbon
    film_tail = m["covering film"]["areal_g_m2"] * (w["tail"] + w["fins"])
    petg = m["PETG printed"]["rho"]
    fs = nisus_plus.NisusPlus().fitting_stations(p)
    items = [
        ("wing: XPS core (25 % lightened)", "wing/reinforcement/covering", foam, x_wing, 0.0, 8.0, "CAD volume x 30 kg/m³ x 0.75"),
        ("wing: carbon spar tube 20/17 x 2000 (three pieces)", "wing/reinforcement/covering", cad["spar"] * 1e-3 * carbon, L["spar_x_root"], 0.0, 6.0, "CAD volume x 1.55 g/cm³"),
        ("wing: spar joiners 16.8/13.5 x 240 (carbon), 2 x", "wing/reinforcement/covering", 2 * cad["spar_joiner"] * 1e-3 * carbon, L["spar_x_root"] + 15, 0.0, 12.0, "CAD volume x 1.55"),
        ("wing: tip rods 8/6 carbon, 2 x 200", "wing/reinforcement/covering", rods, L["spar_x_root"] + 30, 0.0, 50.0, "tube section x length x 1.55"),
        ("wing: balsa D-box 2 mm (LE to spar, full span), TE strips, servo bays, joint ribs", "wing/reinforcement/covering", balsa, 0.15 * L["mac"] + 10, 0.0, 8.0,
         "D-box area x 2 mm x 0.16 g/cm³ + 25 g strips and ribs"),
        (f"wing: rear carry-through tube {p['rear_spar_od']:g}/{p['rear_spar_id']:g} x {2 * p['rear_spar_half_length']:g} (carbon)", "wing/reinforcement/covering", cad["rear_spar"] * 1e-3 * carbon, fs["x_rear"], 0.0, 10.0, "CAD volume x 1.55"),
        ("wing: covering film", "wing/reinforcement/covering", film_wing, x_wing, 0.0, 8.0, "32 g/m² x wetted area"),
        ("wing: flap and aileron hinges, horns' hard points, tape", "wing/reinforcement/covering", 20.0, 0.82 * p["root_chord"], 0.0, 4.0, "assumed"),
        ("wing: panel joint pins, latches, root ribs (plywood)", "wing/reinforcement/covering", 30.0, 0.35 * p["root_chord"], 0.0, 10.0, "assumed"),
        ("pod: LW-PLA shell with nose cone (1.5 mm)", "pod", cad["pod"] * 1e-3 * m["LW-PLA printed"]["rho"], x_pod, 0.0, z_pod, "CAD shell volume x 0.55 g/cm³"),
        ("pod: PETG keel frame and wing saddle (bolts, spar bridge)", "pod", 70.0, 120.0, 0.0, -30.0, "assumed (NISUS's 25 g x the size)"),
        ("pod: hatch, latches, nose bayonet ring", "pod", 35.0, -200.0, 0.0, -10.0, "assumed"),
        ("pod: electronics tray (PETG)", "pod", cad["tray"] * 1e-3 * petg * PRINT_FILL["tray"], 35.0, 0.0, -82.0, "CAD x 1.25 x fill 0.5"),
        ("pod: battery tray (PETG)", "pod", cad["battery_tray"] * 1e-3 * petg * PRINT_FILL["battery_tray"], -25.0, 0.0, -125.0, "CAD x 1.25 x fill 0.6"),
        ("pod: vents, grommets, foam liner", "pod", 15.0, 200.0, 0.0, -70.0, "assumed"),
        (f"booms: carbon tube {p['boom_od']:g}/{p['boom_id']:g} x {p['boom_length']:g}, 2 x", "booms/tail", 2 * cad["boom"] * 1e-3 * carbon, 0.5 * (p["boom_x0"] + L["boom_x1"]), 0.0, p["boom_z"], "CAD volume x 1.55"),
        ("tail: foam stabiliser and fins (11 mm plates)", "booms/tail", tail_foam, x_tail, 0.0, 15.0, "CAD volume x 30 kg/m³"),
        ("tail: carbon spar 6/5", "booms/tail", tail_spar, L["tail_le"] + 0.3 * p["tail_chord"], 0.0, p["boom_z"], "tube section x length x 1.55"),
        ("tail: covering film", "booms/tail", film_tail, x_tail, 0.0, 15.0, "32 g/m² x wetted area"),
        ("tail: boom-end fittings (PETG), 2 x", "booms/tail", 2 * cad["tail_fitting"] * 1e-3 * petg * PRINT_FILL["tail_fitting"], L["tail_le"] + 90, 0.0, p["boom_z"], "CAD x 1.25 x fill 0.6"),
        ("tail: elevator/rudder hinges, torque rod", "booms/tail", 15.0, L["tail_le"] + 130, 0.0, 0.0, "assumed"),
        ("boom root fittings (PETG, clamp rings on both tubes), 2 x", "joints/mounts/adhesive", 2 * cad["boom_fitting"] * 1e-3 * petg * PRINT_FILL["boom_fitting"],
         0.5 * (p["boom_x0"] + fs["x_te"]), 0.0, -10.0, "CAD x 1.25 x fill 0.6"),
        ("motor mount (PETG, solid)", "joints/mounts/adhesive", cad["motor_mount"] * 1e-3 * petg * PRINT_FILL["motor_mount"], L["x_pod_end"], 0.0, p["motor_z"], "CAD x 1.25"),
        ("belly keel skid (TPU)", "joints/mounts/adhesive", cad["skid"] * 1e-3 * m["TPU 95A"]["rho"] * PRINT_FILL["skid"], 0.5 * (p["skid_x0"] + p["skid_x1"]), 0.0,
         -p["pod_height"] - 0.4 * p["skid_depth"], "CAD x 1.21 x fill 0.5"),
        ("GNSS mast, pitot mount, lidar window, antenna feed-throughs", "joints/mounts/adhesive", 20.0, -250.0, 0.0, -20.0, "assumed"),
        ("adhesive (epoxy, CA, hot glue), screws, inserts", "joints/mounts/adhesive", 50.0, 100.0, 0.0, -20.0, "assumed"),
    ]
    struct_mass = sum(i[2] for i in items)
    x_s = sum(i[2] * i[3] for i in items) / struct_mass
    z_s = sum(i[2] * i[5] for i in items) / struct_mass
    items.append(("construction allowance (8 % of structure and joints: paint, tape, filler)", "construction allowance",
                  ALLOWANCE_FRACTION * struct_mass, x_s, 0.0, z_s, f"{ALLOWANCE_FRACTION:.0%} of the structure, assumed"))
    return items


def mass_table(battery_key: str = DEFAULT_PACK, *, p=None, cad=None, cg_frac_mac: float = 0.28, battery_x: float | None = None) -> pd.DataFrame:
    """NISUS+'s mass table [g, mm]: the structure from the CAD, the components, the pack — placed along its bay so the
    centre of gravity sits at ``cg_frac_mac`` of the MAC (``attrs['ballast_note']`` when the bay's ends do not allow it)."""
    p = nisus_plus.resolve(p)
    L = nisus_plus.NisusPlus.layout(p)
    rows = [{"item": i[0], "group": i[1], "mass [g]": i[2], "x [mm]": i[3], "y [mm]": i[4], "z [mm]": i[5], "basis": i[6]}
            for i in structure_items(p, cad)]
    for c in COMPONENTS:
        rows.append({"item": c.key, "group": c.group, "mass [g]": c.mass_g, "x [mm]": c.x("Zero"), "y [mm]": c.y_mm, "z [mm]": c.z_mm,
                     "basis": f"{c.kind}: {c.source[:60]}"})
    df = pd.DataFrame(rows).set_index("item")
    b = pack(battery_key)
    x0, x1, _, _ = nisus_plus.NisusPlus.bays(p)["battery bay"]
    half = b.size_mm[0] / 2
    lo, hi = x0 + 4 + half, x1 - 4 - half
    z_b = -p["pod_height"] + p["pod_wall"] + 15.0 + b.size_mm[2] / 2
    note = ""
    if battery_x is None:
        target = L["x_mac_le"] + cg_frac_mac * L["mac"]
        m_rest, mx_rest = df["mass [g]"].sum(), (df["mass [g]"] * df["x [mm]"]).sum()
        battery_x = (target * (m_rest + b.mass_g) - mx_rest) / b.mass_g
        if battery_x < lo or battery_x > hi:
            edge = lo if battery_x < lo else hi
            cg = ((mx_rest + b.mass_g * edge) / (m_rest + b.mass_g) - L["x_mac_le"]) / L["mac"]
            note = f"battery at the bay's {'front' if edge == lo else 'rear'} ({edge:.0f} mm): the CG is {cg:.1%} MAC instead of {cg_frac_mac:.0%}"
            battery_x = edge
    df.loc[f"battery {b.name}"] = {"group": "battery/retention", "mass [g]": b.mass_g, "x [mm]": battery_x, "y [mm]": 0.0, "z [mm]": z_b,
                                   "basis": f"calculated: {b.series * b.parallel} cells x {b.cell.mass_g:g} g + {b.pack_overhead:.0%} packing"}
    df.attrs.update(variant="Zero", battery=b.key, battery_item=f"battery {b.name}", ballast_note=note, battery_x_range=(lo, hi), cg_target_frac=cg_frac_mac)
    return df


def cg_inertia(table: pd.DataFrame, p=None) -> dict:
    """NISUS's mass, CG and inertia arithmetic on NISUS+'s parameters (``nisus_systems.cg_inertia(design=NisusPlus())``)."""
    return ns.cg_inertia(table, p, design=nisus_plus.NisusPlus())


# ================================================================================================= electrical loads
PHASES = ("prelaunch", "launch+climb", "transit", "survey", "return", "descent", "approach+landing")
DUTY = {"servo": {"prelaunch": 0.2, "launch+climb": 0.6, "transit": 0.3, "survey": 0.6, "return": 0.3, "descent": 0.8, "approach+landing": 0.9},
        "default": {ph: 1.0 for ph in PHASES}}
REGULATOR = dict(REGULATOR_EFF, servo=0.85, jetson=0.92)


def electrical_loads(jetson_w: float = JETSON_INSTALLATION_W) -> pd.DataFrame:
    """Every electrical load with its supply, idle / typical / peak at the device, the survey duty, the regulator and the
    battery side; the Orin installation is one battery-side line (``jetson_w``)."""
    rows = []
    for c in COMPONENTS:
        if c.supply in ("none", "jetson") or c.p_peak_w <= 0:
            continue
        eff = REGULATOR[c.supply]
        duty = DUTY["servo" if c.supply == "servo" else "default"]["survey"]
        p_avg = duty * c.p_typ_w + (1 - duty) * c.p_idle_w
        rows.append({"load": c.key, "supply": c.supply, "V": c.supply_v, "idle [W]": c.p_idle_w, "typical [W]": c.p_typ_w, "peak [W]": c.p_peak_w,
                     "duty (survey)": duty, "regulator eff.": eff, "battery-side average [W]": p_avg / eff, "battery-side peak [W]": c.p_peak_w / eff,
                     "source": ("assumed" if c.kind == "assumed" else "sourced") + ": " + c.source[:90]})
    rows.append({"load": "Jetson Orin installation (15 W mode, camera, data link, buck)", "supply": "battery (12 V buck)", "V": 21.6,
                 "idle [W]": 0.5 * jetson_w, "typical [W]": jetson_w, "peak [W]": 30.0, "duty (survey)": 1.0, "regulator eff.": 1.0,
                 "battery-side average [W]": jetson_w, "battery-side peak [W]": 30.0,
                 "source": f"assumed: {jetson_w:g} W battery side, {JETSON_INSTALLATION_RANGE_W[0]:g}-{JETSON_INSTALLATION_RANGE_W[1]:g} W sensitivity"})
    df = pd.DataFrame(rows).set_index("load")
    df.loc["TOTAL (battery side)"] = {"supply": "", "V": np.nan, "idle [W]": df["idle [W]"].sum(), "typical [W]": df["typical [W]"].sum(),
                                      "peak [W]": df["peak [W]"].sum(), "duty (survey)": np.nan, "regulator eff.": np.nan,
                                      "battery-side average [W]": df["battery-side average [W]"].sum(),
                                      "battery-side peak [W]": df["battery-side peak [W]"].sum(), "source": "sum; servo peaks would not all coincide"}
    return df


def phase_power(jetson_w: float = JETSON_INSTALLATION_W) -> pd.DataFrame:
    """The electronics' battery-side average [W] per mission phase."""
    out = {}
    for ph in PHASES:
        total = 0.0
        for c in COMPONENTS:
            if c.supply in ("none", "jetson") or c.p_peak_w <= 0:
                continue
            duty = DUTY["servo" if c.supply == "servo" else "default"][ph]
            total += (duty * c.p_typ_w + (1 - duty) * c.p_idle_w) / REGULATOR[c.supply]
        total += jetson_w if ph != "prelaunch" else 0.8 * jetson_w
        out[ph] = total
    return pd.DataFrame({"electronics battery-side [W]": out})


def servo_check(p=None, *, V_ail_eas=40.0, V_fe_eas=22.0, flap_deg=55.0, ail_deg=25.0, ch_delta=0.40, servo_kgcm=6.0, linkage_ratio=1.0) -> pd.DataFrame:
    """Hinge moments ``H = q S_s c_s Ch_δ δ`` (NISUS's estimate; Ch_δ ~ 0.4/rad, capped at 40° of effective deflection —
    a plain flap's hinge moment saturates as it separates: assumed) against a 6 kg·cm servo: the aileron at V_NE (EAS),
    the flap in crow at the maximum flap-extended speed V_FE, the elevator and the rudders at V_NE."""
    p = nisus_plus.resolve(p)
    L = nisus_plus.NisusPlus.layout(p)
    rows = {}
    c_ail = p["aileron_chord_frac"] * 0.5 * (p["root_chord"] + p["tip_chord"]) / 1000
    c_flap = p["flap_chord_frac"] * L["chord_joint"] / 1000
    cases = {"aileron (each) at V_NE": (L["S_aileron_each"], c_ail, V_ail_eas, ail_deg),
             f"flap (each) in crow {flap_deg:g}° at V_FE": (L["S_flap_each"], c_flap, V_fe_eas, flap_deg),
             "elevator at V_NE": (L["S_elevator"], p["elevator_frac"] * p["tail_chord"] / 1000, V_ail_eas, 25.0),
             "rudders (both on one servo) at V_NE": (2 * L["S_rudder_each"], p["rudder_frac"] * p["fin_chord"] / 1000, V_ail_eas, 25.0)}
    for name, (S, c, V, d) in cases.items():
        q = 0.5 * RHO0 * V ** 2                 # EAS: the dynamic pressure is the sea-level one at the same EAS
        H = q * S * c * ch_delta * math.radians(min(d, 40.0))
        need = H / G * 100 / linkage_ratio
        rows[name] = {"area [cm²]": S * 1e4, "EAS [m/s]": V, "deflection [deg]": d, "hinge moment [N·cm]": H * 100,
                      "servo torque needed [kg·cm]": need, "servo [kg·cm]": servo_kgcm, "safety factor": servo_kgcm / need}
    return pd.DataFrame(rows).T


# ================================================================================================= the drive
PUBLISHED_STATIC = dict(voltage=21.36, current=72.77, power_w=1554.08, thrust_n=5.537 * G, rpm=8879.0, kv=540.0,
                        part_load=dict(voltage=22.38, current=11.21, thrust_n=1.674 * G, rpm=4943.0),
                        source="T-Motor AT4125 KV540 + APC 15x8, 100 % throttle on 6S (T-Motor's table, 2017 platform; ligpower.com / robotshop PDF snippets, 2026-10-10)")
MOTOR_SHORTLIST = [
    dict(motor="T-Motor AT4125 KV540", kv=540, mass_g=355.0, max_a=85.0, max_w=2000.0, burst="85 A / 180 s", cells="6S",
         point="APC 15x8, 21.36 V: 72.77 A, 1554 W, 8879 rpm, 5537 gf (maker's table)", thrust_g=5537.0, power_w=1554.08,
         fit="selected: a published full-throttle point with the rpm; 49 mm can", url="https://store.tmotor.com/goods.php?id=827"),
    dict(motor="SunnySky X4120 V3 KV430", kv=430, mass_g=np.nan, max_a=85.0, max_w=2100.0, burst="85 A / 30 s", cells="6S",
         point="no thrust table in the snippets", thrust_g=np.nan, power_w=np.nan, fit="41 mm stator; mass not found",
         url="https://www.readymaderc.com/products/details/86776-sunnysky-x-series-v3-x4120-430kv-brushless-motor"),
    dict(motor="SunnySky X4120 V3 KV480", kv=480, mass_g=np.nan, max_a=97.0, max_w=np.nan, burst="97 A / 30 s", cells="6S",
         point="no thrust table in the snippets", thrust_g=np.nan, power_w=np.nan, fit="as KV430",
         url="https://www.readymaderc.com/products/details/86775-sunnysky-x-series-v3-x4120-480kv-brushless-motor"),
    dict(motor="T-Motor AT4130 KV450", kv=450, mass_g=np.nan, max_a=np.nan, max_w=np.nan, burst="not found", cells="6-12S",
         point="the KV450 rows were cut off in the snippets", thrust_g=np.nan, power_w=np.nan, fit="heavier; for 8S+ with 16-17 inch",
         url="https://rcdrone.top/pl/products/tmotor-at4130"),
]


def motor_shortlist_table(t_max_required_n: float) -> pd.DataFrame:
    rows = []
    for m in MOTOR_SHORTLIST:
        ok = (m["thrust_g"] * G / 1000 >= t_max_required_n) if np.isfinite(m["thrust_g"]) else "no data"
        rows.append({"motor": m["motor"], "KV": m["kv"], "mass [g]": m["mass_g"], "max A": m["max_a"], "max W": m["max_w"], "burst": m["burst"],
                     "published point": m["point"], "static g/W": m["thrust_g"] / m["power_w"] if np.isfinite(m["power_w"]) else np.nan,
                     f"meets {t_max_required_n:.0f} N static": ok, "fit": m["fit"], "url": m["url"], "availability": "not confirmed (snippets)"})
    return pd.DataFrame(rows).set_index("motor")


def propeller_15x8():
    """The APC 15x8E as Boreas' generic planform (NISUS's 9x6 shape scaled by 15/9): constant geometric pitch 203 mm."""
    from vegeta import boreas
    d, pitch = boreas.inches(15, 8)
    return boreas.Propeller.from_pitch("APC 15x8E (generic planform)", d, pitch, blades=2, chord_root_m=0.023, chord_max_m=0.036,
                                       chord_tip_m=0.010, mass_kg=0.046, rotor_mass_kg=0.16,
                                       notes="generic planform for a 15x8 thin-electric (NISUS's 9x6 shape x 15/9); the real blade's chord and twist are not in the model")


def blade_section():
    from vegeta import boreas
    return boreas.Airfoil(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-2.5, cl_max=1.1, cd0=0.018, k=0.04,
                          source="assumed for a 15-inch blade at Re ~ 2.5e5 (NISUS's values, cd0 a little lower for the larger chord)")


def motor_model(no_load_a: float = 1.6, max_current_a: float = 85.0):
    """The AT4125 KV540 fitted to its published full-throttle point, which gives the rpm: R = (V − rpm/KV)/I; the shaft
    torque kt (I − I0) there against the BEMT's gives the torque calibration, the published thrust against the BEMT's the
    thrust calibration (both applied to the positive region of the table only: nothing was measured in the brake region).
    I0 assumed (the 41xx class at ~10 V: 1.4-1.8 A). Returns (motor, fit)."""
    from vegeta import boreas
    prop, af = propeller_15x8(), blade_section()
    P = PUBLISHED_STATIC
    rpm, I, V = P["rpm"], P["current"], P["voltage"]
    R = (V - rpm / P["kv"]) / I
    motor = boreas.Motor("T-Motor AT4125 KV540 (fitted)", kv_rpm_per_volt=P["kv"], resistance_ohm=R, no_load_current_a=no_load_a,
                         max_current_a=max_current_a, mass_kg=0.355, source=f"KV, the full-throttle point published ({P['source']}); R fitted (ESC and wires included), I0 assumed")
    op = boreas.solve(prop, af, rpm, 0.0, RHO0)
    Q_motor = motor.kt * (I - no_load_a)
    fit = {"rpm_at_point": rpm, "resistance_ohm": R, "no_load_a": no_load_a, "kt": motor.kt, "bemt_thrust_n": op.thrust, "published_thrust_n": P["thrust_n"],
           "thrust_correction": P["thrust_n"] / op.thrust, "bemt_torque_nm": op.torque, "motor_torque_nm": Q_motor, "torque_correction": Q_motor / op.torque,
           "shaft_power_w": Q_motor * rpm * math.pi / 30, "motor_efficiency_at_point": Q_motor * rpm * math.pi / 30 / (V * I),
           "note": "the corrections say how far the generic blade is from the real one at the static point"}
    # the part-load point is the check: the model's current at its rpm against the published 11.21 A
    pl = P["part_load"]
    op2 = boreas.solve(prop, af, pl["rpm"], 0.0, RHO0)
    fit["check_part_load_current_a"] = no_load_a + op2.torque * fit["torque_correction"] / motor.kt
    fit["check_part_load_thrust_n"] = op2.thrust * fit["thrust_correction"]
    fit["published_part_load"] = dict(pl)
    return motor, fit


PROPULSION_JSON = DATA / "nisus_plus_propulsion.json"
V_GRID = np.arange(0.0, 40.01, 2.5)
RPM_GRID = np.r_[150.0, 300.0, 500.0, 750.0, np.arange(1000.0, 2000.0, 250.0), np.arange(2000.0, 11001.0, 500.0)]


@dataclass
class NisusPlusDrive:
    """The drive over (airspeed V, rpm) at sea level — signed thrust [N] and shaft torque [N m] from Boreas BEMT on the
    15x8 (calibrated on the positive region) — and the fitted motor; at any density the propeller scales with σ and the
    motor's current and voltage follow (``rows``). The throttle is the ESC's applied voltage over the pack's loaded
    voltage; at low throttle and speed the voltage is below the back-EMF and the current reverses (``at``)."""
    V: np.ndarray
    rpm: np.ndarray
    thrust0: np.ndarray
    torque0: np.ndarray
    kv: float
    R: float
    I0: float
    max_current_a: float
    battery_v: float = 21.6
    esc_efficiency: float = 0.96
    motor_name: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def kt(self):
        return 60 / (2 * math.pi * self.kv)

    def _row(self, V, table):
        V = float(np.clip(V, self.V[0], self.V[-1]))
        i = min(int(np.searchsorted(self.V, V, side="right")) - 1, len(self.V) - 2)
        t = (V - self.V[i]) / (self.V[i + 1] - self.V[i])
        return (1 - t) * table[i] + t * table[i + 1]

    def rows(self, V, rho=RHO0) -> dict:
        """Over the rpm grid at airspeed V and density rho: thrust, torque, the motor's current and the voltage it needs."""
        s = rho / RHO0
        T = s * self._row(V, self.thrust0)
        Q = s * self._row(V, self.torque0)
        I = self.I0 + Q / self.kt
        Vn = self.rpm / self.kv + I * self.R
        return {"rpm": self.rpm, "thrust": T, "torque": Q, "current": I, "voltage": Vn, "shaft_power": Q * self.rpm * math.pi / 30}

    def _battery_w(self, Vn, I):
        P = Vn * I
        return P / self.esc_efficiency if P >= 0 else P * self.esc_efficiency

    def _point(self, r, n, mode):
        T = float(np.interp(n, r["rpm"], r["thrust"]))
        I = float(np.interp(n, r["rpm"], r["current"]))
        Vn = float(np.interp(n, r["rpm"], r["voltage"]))
        Ps = float(np.interp(n, r["rpm"], r["shaft_power"]))
        return {"rpm": float(n), "thrust": T, "current": I, "voltage": Vn, "shaft_power": Ps, "electrical": self._battery_w(max(Vn, 0.0), I), "mode": mode}

    def freewheel(self, V, rho=RHO0) -> dict:
        """The ESC off (or its motor output floating): the propeller windmills where the motor carries no current — its
        friction torque −kt I0 only (on the stable branch: above the rpm of the most negative torque); when even the
        lowest rpm's torque cannot overcome the friction, the propeller stops (``locked_drag``)."""
        r = self.rows(V, rho)
        Q_need = -self.kt * self.I0
        i0 = int(np.argmin(r["torque"]))
        if r["torque"][i0] < Q_need:
            n = float(np.interp(Q_need, r["torque"][i0:], r["rpm"][i0:]))
            p = self._point(r, n, "freewheel")
            p.update(current=0.0, voltage=n / self.kv, electrical=0.0)
            return p
        return {"rpm": 0.0, "thrust": -self.locked_drag(V, rho), "current": 0.0, "voltage": 0.0, "shaft_power": 0.0, "electrical": 0.0, "mode": "stopped"}

    def locked_drag(self, V, rho=RHO0) -> float:
        """A stopped two-blade propeller: the blades' area (~9 % of the disc for a 15-inch thin-electric) nearly flat to the
        flow at a flat plate's CD of 1.2 (assumed; Boreas cannot solve rpm = 0)."""
        R = self.notes.get("prop_radius_m", 0.1905)
        return 0.5 * rho * V * V * 1.2 * 0.09 * math.pi * R * R

    def at(self, V, throttle, rho=RHO0, v_batt=None, *, i_charge_max=math.inf, i_motor_max=None) -> dict:
        """The drive at airspeed V (axial), ``throttle`` 0..1 of the pack's loaded voltage ``v_batt`` (default nominal),
        density rho; ``throttle`` None: freewheel. Below the windmill's back-EMF the motor generates: the battery-side
        power is negative (``electrical``: charging, mode 'regen'). The motor current is held to ``i_motor_max`` and the
        charge current (battery side) to ``i_charge_max`` by moving the ESC's voltage back toward the free windmill (less
        braking: the ESC cannot dump the energy anywhere else)."""
        if throttle is None:
            return self.freewheel(V, rho)
        vb = self.battery_v if v_batt is None else float(v_batt)
        i_motor_max = self.max_current_a if i_motor_max is None else i_motor_max
        r = self.rows(V, rho)
        vn = np.maximum.accumulate(r["voltage"])
        volts = float(np.clip(throttle, 0.0, 1.0)) * vb
        if volts <= vn[0]:
            pt = self._point(r, float(r["rpm"][0]), "drive")
            if vn[0] > 0:
                f = volts / vn[0]
                pt.update(thrust=pt["thrust"] * f, current=pt["current"] * f, shaft_power=pt["shaft_power"] * f, electrical=pt["electrical"] * f, rpm=pt["rpm"] * f)
            return pt
        n = float(np.interp(volts, vn, r["rpm"]))
        pt = self._point(r, n, "regen" if np.interp(n, r["rpm"], r["current"]) < 0 else "drive")
        if pt["current"] > i_motor_max:
            pt = self._point(r, float(np.interp(i_motor_max, r["current"], r["rpm"])), "drive (current limited)")
        if pt["mode"] == "regen":
            fw = self.freewheel(V, rho)
            lim_motor = pt["current"] < -i_motor_max
            lim_charge = -pt["electrical"] / vb > i_charge_max
            if lim_motor or lim_charge:
                ns_ = np.linspace(n, max(fw["rpm"], n), 41)
                ok = np.array([(lambda q: q["current"] >= -i_motor_max and -q["electrical"] / vb <= i_charge_max + 1e-9)(self._point(r, k, "")) for k in ns_])
                k = ns_[int(np.argmax(ok))] if ok.any() else ns_[-1]
                pt = self._point(r, float(k), "regen (current limited)" if not lim_charge else "regen (charge limited)")
                if i_charge_max <= 0:                   # no charging at all: only the short circuit brakes (the windings take it all)
                    pt = self._point(r, self.short_rpm(V, rho), "short-circuit brake (no charging)")
                    pt["electrical"] = 0.0
        return pt

    def short_rpm(self, V, rho=RHO0) -> float:
        """The rpm where the motor's terminal voltage is zero: the short-circuit brake."""
        r = self.rows(V, rho)
        vn = np.maximum.accumulate(r["voltage"])
        if vn[0] >= 0:
            return float(r["rpm"][0])
        return float(np.interp(0.0, vn, r["rpm"]))

    def max_drag_rpm(self, V, rho=RHO0) -> float:
        """The rpm (between the short circuit and the free windmill) where the braked propeller's drag is largest: below
        it the blades stall and the drag falls again."""
        r = self.rows(V, rho)
        fw = self.freewheel(V, rho)
        if fw["mode"] == "stopped":
            return 0.0
        n_s = self.short_rpm(V, rho)
        ns_ = np.linspace(n_s, fw["rpm"], 60)
        T = np.interp(ns_, r["rpm"], r["thrust"])
        return float(ns_[int(np.argmin(T))])

    def brake(self, V, b, rho=RHO0, v_batt=None, *, i_charge_max=math.inf, i_motor_max=None) -> dict:
        """The brake command b in 0..1 at airspeed V: 0 = the free windmill (no current), 1 = the rpm of the largest drag
        (``max_drag_rpm``); between, the ESC holds a voltage below the back-EMF, the propeller slows, its drag and the
        charging current grow. Returns ``at``'s dict (with the limits)."""
        vb = self.battery_v if v_batt is None else float(v_batt)
        fw = self.freewheel(V, rho)
        if fw["mode"] == "stopped" or b <= 0:
            return fw
        r = self.rows(V, rho)
        n = fw["rpm"] - float(np.clip(b, 0, 1)) * (fw["rpm"] - self.max_drag_rpm(V, rho))
        th = max(float(np.interp(n, r["rpm"], np.maximum.accumulate(r["voltage"]))), 0.0) / vb
        return self.at(V, th, rho, vb, i_charge_max=i_charge_max, i_motor_max=i_motor_max)

    def max_thrust(self, V, rho=RHO0, v_batt=None):
        return self.at(V, 1.0, rho, v_batt)["thrust"]

    def throttle_for_thrust(self, V, T, rho=RHO0, v_batt=None):
        vb = self.battery_v if v_batt is None else float(v_batt)
        r = self.rows(V, rho)
        Tr = np.maximum.accumulate(r["thrust"])
        if T >= self.at(V, 1.0, rho, vb)["thrust"]:
            return 1.0
        n = float(np.interp(T, Tr, r["rpm"]))
        return float(np.clip(np.interp(n, r["rpm"], r["voltage"]) / vb, 0.0, 1.0))

    def electrical_for_thrust(self, V, T, rho=RHO0, v_batt=None):
        if T > self.max_thrust(V, rho, v_batt) + 1e-9:
            return math.inf
        return self.at(V, self.throttle_for_thrust(V, T, rho, v_batt), rho, v_batt)["electrical"]

    def map_at(self, rho=RHO0, battery_v=None) -> ns.PropulsionMap:
        """NISUS's ``PropulsionMap`` at density rho (the positive, driving region: for the envelope arithmetic of
        ``nisus_flight.MapUnit`` / ``merlin_flight.performance``)."""
        bv = self.battery_v if battery_v is None else battery_v
        T = np.zeros((len(self.V), len(self.rpm))); Q = np.zeros_like(T); I = np.zeros_like(T); Vn = np.zeros_like(T); P = np.zeros_like(T)
        for i, v in enumerate(self.V):
            r = self.rows(v, rho)
            T[i], Q[i], I[i], Vn[i], P[i] = r["thrust"], r["torque"], r["current"], r["voltage"], r["shaft_power"]
        keep = np.all(Vn > 0, axis=0)
        T = np.maximum.accumulate(np.maximum(T[:, keep], 0.0), axis=1) + np.arange(keep.sum()) * 1e-9
        return ns.PropulsionMap(self.V.copy(), self.rpm[keep].copy(), T, np.maximum(P[:, keep], 0), np.maximum(Q[:, keep], 0), np.maximum(I[:, keep], 0),
                                Vn[:, keep], bv, self.esc_efficiency, self.motor_name, dict(self.notes, rho=rho))

    def to_json(self) -> dict:
        return {"V": self.V.tolist(), "rpm": self.rpm.tolist(), "thrust0": self.thrust0.tolist(), "torque0": self.torque0.tolist(), "kv": self.kv,
                "R": self.R, "I0": self.I0, "max_current_a": self.max_current_a, "battery_v": self.battery_v, "esc_efficiency": self.esc_efficiency,
                "motor_name": self.motor_name, "rho0": RHO0, "notes": self.notes}

    @classmethod
    def from_json(cls, d: dict) -> "NisusPlusDrive":
        return cls(np.asarray(d["V"]), np.asarray(d["rpm"]), np.asarray(d["thrust0"]), np.asarray(d["torque0"]), d["kv"], d["R"], d["I0"],
                   d["max_current_a"], d["battery_v"], d["esc_efficiency"], d["motor_name"], d.get("notes", {}))


def build_drive(*, V=V_GRID, rpm=RPM_GRID, battery_v=21.6, esc_efficiency=0.96, correct: bool = True) -> NisusPlusDrive:
    """Boreas BEMT over (V, rpm) at sea level for the 15x8, signed (the windmill region kept), the calibration ratios on
    the positive region; ~400 solves."""
    from vegeta import boreas
    prop, af = propeller_15x8(), blade_section()
    motor, fit = motor_model()
    kT = fit["thrust_correction"] if correct else 1.0
    kQ = fit["torque_correction"] if correct else 1.0
    T = np.zeros((len(V), len(rpm))); Q = np.zeros_like(T)
    for i, v in enumerate(V):
        for j, n in enumerate(rpm):
            op = boreas.solve(prop, af, float(n), float(v), RHO0)
            T[i, j] = op.thrust * (kT if op.thrust > 0 else 1.0)
            Q[i, j] = op.torque * (kQ if op.torque > 0 else 1.0)
    notes = {"propeller": prop.describe()["notes"], "section": af.source, "motor": motor.source,
             "fit": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in fit.items()}, "prop_radius_m": prop.radius,
             "density": "sea level ISA (1.225); at density rho the propeller's thrust and torque are x rho/1.225 at the same (V, rpm) — Boreas has no "
                        "Reynolds or Mach effect (the blades' Re falls ~35 % at 4500 m: the real drag polar worsens; not modelled)",
             "brake region": "BEMT at low rpm and high V: negative thrust and torque (the windmill). Boreas drops the induced velocity when the "
                             "thrust is negative (the momentum balance is skipped): the braking is somewhat optimistic; the turbulent-windmill state "
                             "is not modelled. Calculated, low confidence; the calibration ratios are not applied there",
             "limits": "no installation effect (the pusher behind the pod); the pack's sag enters through v_batt; no motor heating model"}
    return NisusPlusDrive(np.asarray(V, float), np.asarray(rpm, float), T, Q, motor.kv_rpm_per_volt, motor.resistance_ohm, motor.no_load_current_a,
                      motor.max_current_a, battery_v, esc_efficiency, motor.name, notes)


def drive(path: Path = PROPULSION_JSON, rebuild: bool = False) -> NisusPlusDrive:
    """The cached drive (``data/nisus_plus_propulsion.json``), rebuilt when missing or ``rebuild``."""
    path = Path(path)
    if path.exists() and not rebuild:
        return NisusPlusDrive.from_json(json.loads(path.read_text()))
    d = build_drive()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d.to_json(), indent=1))
    return d


def drive_table(dr: NisusPlusDrive, heights=(0.0, 1500.0, 3000.0, 4500.0, 6000.0), V=(0.0, 15.0, 20.0, 25.0)) -> pd.DataFrame:
    """Full throttle against altitude: thrust, rpm, current, battery-side power at a few airspeeds (nominal voltage)."""
    rows = {}
    for h in heights:
        rho = atmosphere(h)["rho"]
        for v in V:
            a = dr.at(v, 1.0, rho)
            rows[(f"{h:.0f} m", f"{v:g} m/s")] = {"σ": rho / RHO0, "thrust [N]": a["thrust"], "rpm": a["rpm"], "motor current [A]": a["current"],
                                                   "battery side [W]": a["electrical"], "mode": a["mode"]}
    return pd.DataFrame(rows).T


def requirements(mass_kg, cd0, AR, oswald, S, *, targets=((0.0, 8.0, 16.0), (4500.0, 5.0, 18.0)), V_cruise_tas=20.0, V_max_eas=30.0,
                 launch_tw=0.70, eta_prop=0.60, eta_motor=0.80, eta_esc=0.96, dT=0.0) -> pd.DataFrame:
    """What the drive must give: cruise at altitude, the climb targets ((altitude, rate, EAS)), the dash, a launch T/W —
    as NISUS's ``requirements`` but at the altitude's density (TAS = EAS/√σ)."""
    W = mass_kg * G
    k = 1 / (math.pi * AR * oswald)

    def drag(V, rho, n=1.0):
        q = 0.5 * rho * V ** 2
        return q * S * (cd0 + k * (n * W / (q * S)) ** 2)

    rows = {}
    for h in (0.0, 3000.0, 4500.0):
        a = atmosphere(h, dT)
        rows[f"cruise {V_cruise_tas:g} m/s TAS at {h:.0f} m"] = dict(V=V_cruise_tas, h=h, thrust=drag(V_cruise_tas, a["rho"]))
    for h, roc, V_eas in targets:
        a = atmosphere(h, dT)
        V = V_eas / math.sqrt(a["sigma"])
        rows[f"climb {roc:g} m/s at {h:.0f} m ({V_eas:g} m/s EAS)"] = dict(V=V, h=h, thrust=drag(V, a["rho"]) + W * roc / V)
    a = atmosphere(3000.0, dT)
    Vd = V_max_eas / math.sqrt(a["sigma"])
    rows[f"dash {V_max_eas:g} m/s EAS at 3000 m"] = dict(V=Vd, h=3000.0, thrust=drag(Vd, a["rho"]))
    rows[f"launch: static T/W ≥ {launch_tw:g}"] = dict(V=0.0, h=3000.0, thrust=launch_tw * W)
    out = {}
    for name, r in rows.items():
        P_aero = r["thrust"] * r["V"]
        P_shaft = P_aero / eta_prop if r["V"] > 0 else np.nan
        out[name] = {"altitude [m]": r["h"], "TAS [m/s]": r["V"], "thrust [N]": r["thrust"], "T/W": r["thrust"] / W, "aero power T·V [W]": P_aero,
                     "shaft power [W]": P_shaft, "electrical [W]": P_shaft / (eta_motor * eta_esc) if r["V"] > 0 else np.nan}
    df = pd.DataFrame(out).T
    df.attrs["T_max_required_N"] = float(df["thrust [N]"].max())
    return df


# ================================================================================================= mission energy
@dataclass
class MissionPlan:
    """The sizing mission in the mountains: ground operation at the launch meadow, the launch, a climb from
    ``launch_alt_m`` to ``work_alt_m`` (spiralling over the valley at ``roc`` and ``V_climb_eas``), the transit to the
    survey area ``radius_m`` away, ``survey_s`` of terrain-following survey, the transit back at altitude, the fast
    descent over the meadow (``descent_rate`` m/s with crow and the propeller brake), the approach and landing."""
    ground_s: float = 300.0
    launch_alt_m: float = 1200.0
    work_alt_m: float = 4000.0
    dT: float = 0.0                  # ISA offset [K]
    pack_T_C: float = 15.0           # the pack's temperature (insulated sleeve, warmed by its own losses): assumed
    radius_m: float = 3000.0
    survey_s: float = 1200.0
    V_climb_eas: float = 16.0
    roc: float = 6.0
    V_cruise_eas: float = 18.0
    V_survey_eas: float = 17.0
    descent_rate: float = 10.0
    descent_regen_w: float = 0.0     # battery-side power recovered while descending (negative of a load): from nisus_plus_flight.descent_table
    approach_s: float = 90.0
    wind_m_s: float = 0.0
    ridge_m: float = 0.0             # height the return must climb over an intervening ridge
    derating: float = 0.90
    reserve_frac: float = 0.20
    uncertainty_frac: float = 0.15
    go_arounds: int = 1
    circuit_m: float = 1500.0


def _drag(mass_kg, cd0, AR, oswald, S, V, rho):
    W = mass_kg * G
    k = 1 / (math.pi * AR * oswald)
    q = 0.5 * rho * V ** 2
    return q * S * (cd0 + k * (W / (q * S)) ** 2)


def level_power(dr: NisusPlusDrive, mass_kg, cd0, AR, oswald, S, V_tas, h, roc=0.0, dT=0.0, v_batt=None):
    """Battery-side propulsion power [W] and thrust for level (or climbing) flight at TAS and altitude."""
    a = atmosphere(h, dT)
    T = _drag(mass_kg, cd0, AR, oswald, S, V_tas, a["rho"]) + mass_kg * G * roc / V_tas
    return dr.electrical_for_thrust(V_tas, T, a["rho"], v_batt), T


def max_roc(dr: NisusPlusDrive, mass_kg, cd0, AR, oswald, S, V_tas, h, dT=0.0, v_batt=None) -> float:
    a = atmosphere(h, dT)
    T = dr.max_thrust(V_tas, a["rho"], v_batt)
    return (T - _drag(mass_kg, cd0, AR, oswald, S, V_tas, a["rho"])) * V_tas / (mass_kg * G)


def climb_profile(dr: NisusPlusDrive, airframe: dict, plan: MissionPlan, *, step_m: float = 100.0, v_batt=None) -> pd.DataFrame:
    """The climb from the launch altitude to the work altitude in ``step_m`` steps: at each, the TAS for the climb's
    EAS, the climb rate (the plan's, or the most the drive gives), the thrust, the battery-side power, the time and
    energy of the step."""
    m, cd0, AR, e, S = airframe["mass_kg"], airframe["cd0"], airframe["AR"], airframe["oswald"], airframe["S"]
    rows = []
    hs = np.arange(plan.launch_alt_m, plan.work_alt_m, step_m)
    for h in hs:
        hm = h + 0.5 * min(step_m, plan.work_alt_m - h)
        a = atmosphere(hm, plan.dT)
        V = plan.V_climb_eas / math.sqrt(a["sigma"])
        roc_max = max_roc(dr, m, cd0, AR, e, S, V, hm, plan.dT, v_batt)
        roc = min(plan.roc, 0.97 * roc_max)
        P, T = level_power(dr, m, cd0, AR, e, S, V, hm, roc, plan.dT, v_batt)
        dh = min(step_m, plan.work_alt_m - h)
        t = dh / max(roc, 0.05)
        rows.append({"altitude [m]": hm, "σ": a["sigma"], "TAS [m/s]": V, "roc [m/s]": roc, "roc max [m/s]": roc_max, "thrust [N]": T,
                     "power [W]": P, "time [s]": t, "energy [Wh]": P * t / 3600, "horizontal [m]": V * t * math.cos(math.asin(min(roc / V, 0.99)))})
    return pd.DataFrame(rows).set_index("altitude [m]")


def return_energy(dr, airframe, plan: MissionPlan, distance_m, electronics: dict) -> dict:
    """From the survey area home: the transit at the work altitude against the wind (cruise EAS), a climb over
    ``ridge_m`` if there is one, the descent (electronics only, the regeneration credited), the approach, the
    go-arounds; times the uncertainty. The height is energy: the descent costs only the electronics."""
    m, cd0, AR, e, S = airframe["mass_kg"], airframe["cd0"], airframe["AR"], airframe["oswald"], airframe["S"]
    a = atmosphere(plan.work_alt_m, plan.dT)
    V = plan.V_cruise_eas / math.sqrt(a["sigma"])
    P_c, _ = level_power(dr, m, cd0, AR, e, S, V, plan.work_alt_m, 0.0, plan.dT)
    gs = max(V - plan.wind_m_s, 3.0)
    t_back = distance_m / gs
    E_back = (P_c + electronics["return"]) * t_back / 3600
    E_ridge = 0.0
    if plan.ridge_m > 0:
        P_r, _ = level_power(dr, m, cd0, AR, e, S, V, plan.work_alt_m, 3.0, plan.dT)
        E_ridge = (P_r - P_c) * plan.ridge_m / 3.0 / 3600
    t_desc = (plan.work_alt_m - plan.launch_alt_m) / plan.descent_rate
    E_desc = (electronics["descent"] - plan.descent_regen_w) * t_desc / 3600
    a0 = atmosphere(plan.launch_alt_m, plan.dT)
    V_app = 13.0 / math.sqrt(a0["sigma"])
    P_app, _ = level_power(dr, m, cd0, AR, e, S, V_app, plan.launch_alt_m, 0.0, plan.dT)
    E_land = (0.5 * P_app + electronics["approach+landing"]) * plan.approach_s / 3600
    V0 = plan.V_cruise_eas / math.sqrt(a0["sigma"])
    P_c0, _ = level_power(dr, m, cd0, AR, e, S, V0, plan.launch_alt_m, 0.0, plan.dT)
    E_ga = plan.go_arounds * (P_c0 + electronics["return"]) * plan.circuit_m / V0 / 3600
    E = E_back + E_ridge + E_desc + E_land + E_ga
    return {"t_back_s": t_back, "t_descent_s": t_desc, "E_back_wh": E_back, "E_ridge_wh": E_ridge, "E_descent_wh": E_desc, "E_land_wh": E_land,
            "E_go_around_wh": E_ga, "E_return_wh": E, "E_uncertainty_wh": plan.uncertainty_frac * E}


def mission_energy(pk: LiIonPack, dr: NisusPlusDrive, airframe: dict, plan: MissionPlan = MissionPlan(), *, jetson_w=JETSON_INSTALLATION_W) -> dict:
    """The mountain mission's energy phase by phase on one pack (see ``MissionPlan``): the electronics from
    ``phase_power``, the propulsion from the drive at each phase's density; the descent's regeneration in its own row
    (negative). Returns the phase table [Wh, kWh, % of nominal], the totals, the reserve, the return trigger."""
    m, cd0, AR, e, S = airframe["mass_kg"], airframe["cd0"], airframe["AR"], airframe["oswald"], airframe["S"]
    el = phase_power(jetson_w)["electronics battery-side [W]"]
    E_nom = pk.energy_wh
    E_avail = pk.usable_wh(plan.pack_T_C, 20.0) * plan.derating / 0.98        # the cold and the I²R loss in usable_wh; derating for age and the cut-off
    E_avail = min(E_avail, E_nom * plan.derating)
    E_reserve = plan.reserve_frac * E_avail
    cl = climb_profile(dr, airframe, plan)
    t_climb = float(cl["time [s]"].sum())
    E_climb_prop = float(cl["energy [Wh]"].sum())
    climb_dist = float(cl["horizontal [m]"].sum())
    a = atmosphere(plan.work_alt_m, plan.dT)
    V_c = plan.V_cruise_eas / math.sqrt(a["sigma"])
    V_s = plan.V_survey_eas / math.sqrt(a["sigma"])
    P_cruise, _ = level_power(dr, m, cd0, AR, e, S, V_c, plan.work_alt_m, 0.0, plan.dT)
    P_survey, _ = level_power(dr, m, cd0, AR, e, S, V_s, plan.work_alt_m, 0.0, plan.dT)
    P_survey *= 1.15                                     # turns and the terrain following's small climbs and descents: +15 % (assumption)
    gs_out = max(V_c - plan.wind_m_s, 3.0)
    t_out = plan.radius_m / gs_out                       # the climb spirals over the valley: the transit starts over the meadow
    ret = return_energy(dr, airframe, plan, plan.radius_m, el)
    phases = [
        ("ground operation (prelaunch)", plan.ground_s, 0.0, el["prelaunch"]),
        (f"launch + climb {plan.launch_alt_m:.0f} → {plan.work_alt_m:.0f} m", t_climb, E_climb_prop * 3600 / max(t_climb, 1e-9), el["launch+climb"]),
        ("transit", t_out, P_cruise, el["transit"]),
        ("survey (terrain following)", plan.survey_s, P_survey, el["survey"]),
        ("return transit", ret["t_back_s"], P_cruise, el["return"]),
        (f"descent {plan.descent_rate:g} m/s (crow + brake)", ret["t_descent_s"], 0.0, el["descent"]),
        ("descent: regeneration", ret["t_descent_s"], -plan.descent_regen_w, 0.0),
        ("approach + landing", plan.approach_s, ret["E_land_wh"] * 3600 / plan.approach_s - el["approach+landing"], el["approach+landing"]),
    ]
    rows = []
    for name, t, P_prop, P_el in phases:
        E_prop, E_el = P_prop * t / 3600, P_el * t / 3600
        rows.append({"phase": name, "time [s]": t, "propulsion [W]": P_prop, "electronics [W]": P_el, "propulsion [Wh]": E_prop, "electronics [Wh]": E_el,
                     "energy [Wh]": E_prop + E_el, "energy [kWh]": (E_prop + E_el) / 1000, "% of nominal": 100 * (E_prop + E_el) / E_nom})
    df = pd.DataFrame(rows).set_index("phase")
    E_used = df["energy [Wh]"].sum()
    left = E_avail - E_used
    trigger = ret["E_return_wh"] + ret["E_uncertainty_wh"] + E_reserve
    E_fixed = df.iloc[:3]["energy [Wh]"].sum()
    t_survey_feasible = max(E_avail - E_fixed - trigger, 0.0) * 3600 / (P_survey + el["survey"])
    potential_wh = m * G * (plan.work_alt_m - plan.launch_alt_m) / 3600
    return {"pack": pk.key, "mass_kg": m, "phases": df, "climb": cl, "E_nominal_wh": E_nom, "E_available_wh": E_avail, "E_reserve_wh": E_reserve,
            "E_used_wh": E_used, "E_used_kwh": E_used / 1000, "E_left_wh": left, "E_left_pct_available": 100 * left / E_avail,
            "reserve_kept": bool(left >= E_reserve - 1e-9), "return": ret, "trigger_wh": trigger, "t_climb_s": t_climb, "climb_horizontal_m": climb_dist,
            "E_climb_wh": float(df.iloc[1]["energy [Wh]"]), "potential_energy_wh": potential_wh,
            "climb_efficiency": potential_wh / max(E_climb_prop, 1e-9), "t_survey_feasible_s": t_survey_feasible,
            "survey_time_ok": bool(plan.survey_s <= t_survey_feasible + 1e-9), "P_cruise_w": P_cruise, "P_survey_w": P_survey,
            "electronics_w": dict(el), "plan": asdict(plan)}


def mission_table(dr, airframe_fn, plan: MissionPlan = MissionPlan(), packs=None) -> pd.DataFrame:
    """The battery iteration: one row per pack (mass → power → energy), the reserve, the feasible survey time."""
    rows = {}
    for k in (packs or PACKS):
        b = pack(k)
        af = airframe_fn(k)
        r = mission_energy(b, dr, af, plan)
        rows[b.name] = {"pack mass [g]": b.mass_g, "aircraft mass [kg]": af["mass_kg"], "nominal [Wh]": b.energy_wh, "available [Wh]": r["E_available_wh"],
                        "climb [Wh]": r["E_climb_wh"], "climb time [s]": r["t_climb_s"], "survey power [W]": r["P_survey_w"], "used [Wh]": r["E_used_wh"],
                        "left [% avail.]": r["E_left_pct_available"], "reserve kept": r["reserve_kept"], "feasible survey [min]": r["t_survey_feasible_s"] / 60,
                        f"{plan.survey_s / 60:.0f}-min survey ok": r["survey_time_ok"] and r["reserve_kept"], "fits the bay": battery_fits(b)["fits"]}
    return pd.DataFrame(rows).T


__all__ = ["G", "RHO0", "atmosphere", "density_altitude", "atmosphere_table", "CAD", "cad_numbers", "COMPONENTS", "components", "component_table",
           "JETSON", "JETSON_INSTALLATION_W", "LiIonCell", "CELLS", "CHARGE_MIN_C", "LiIonPack", "PACKS", "DEFAULT_PACK", "pack", "pack_table", "battery_fits",
           "structure_items", "mass_table", "cg_inertia", "PHASES", "DUTY", "electrical_loads", "phase_power", "servo_check", "PUBLISHED_STATIC",
           "MOTOR_SHORTLIST", "motor_shortlist_table", "propeller_15x8", "blade_section", "motor_model", "NisusPlusDrive", "build_drive", "drive", "drive_table",
           "requirements", "MissionPlan", "level_power", "max_roc", "climb_profile", "return_energy", "mission_energy", "mission_table"]
