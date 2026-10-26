"""Strato — the component data the airship studies assume (Stage V8). **Every number here is an estimate**: a
representative value for the technology class, chosen from the ranges reported for HAPS, stratospheric balloons and
solar aircraft, not a measured datasheet of a product. Each entry says what it stands for; replace an entry with a
datasheet and the studies follow. The physics that uses them (og.stratosphere, og.lifting_gas, og.solar,
og.envelope) is verified separately; ``STRATO.md`` keeps the verified/estimate table.

SI units: kg, m, W, J, K, Pa, N/m.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PV:
    """A flexible solar array as installed: cells, encapsulation, wiring and the bond to the envelope."""
    key: str
    name: str
    efficiency: float             # at 25 °C cell temperature, under the stratospheric (near-AM0) spectrum
    temp_coeff: float             # relative efficiency change per K (negative)
    areal_mass: float             # kg/m², installed
    absorptance: float            # solar absorptance of the array surface (what it does not convert heats it)
    emissivity: float             # thermal IR emissivity
    note: str


PVS = {
    "gaas": PV("gaas", "thin-film GaAs (flexible, encapsulated)", 0.26, -0.0020, 0.25, 0.90, 0.85,
               "estimate: flexible single/multi-junction GaAs on polymer, ~24–30 % near AM0, ~0.15–0.35 kg/m² installed"),
    "cigs": PV("cigs", "flexible CIGS (encapsulated)", 0.15, -0.0036, 0.45, 0.90, 0.85,
               "estimate: flexible CIGS on polymer/steel foil, ~13–17 % near AM0, ~0.3–0.6 kg/m² installed"),
}
MPPT_EFFICIENCY = 0.97            # estimate: maximum-power-point trackers
HARNESS_EFFICIENCY = 0.98         # estimate: array wiring losses


@dataclass(frozen=True)
class BatteryTech:
    key: str
    name: str
    specific_energy: float        # Wh/kg at pack level (cells, structure, BMS, thermal insulation)
    charge_efficiency: float
    discharge_efficiency: float
    max_dod: float                # usable fraction of the nameplate energy
    note: str


BATTERIES = {
    "li-ion": BatteryTech("li-ion", "Li-ion (high-energy NMC/NCA cells)", 220.0, 0.97, 0.97, 0.90,
                          "estimate: 260–300 Wh/kg cells, ~75–80 % cell-to-pack"),
    "li-s": BatteryTech("li-s", "Li–S (HAPS-grade)", 350.0, 0.93, 0.95, 0.85,
                        "estimate: 400–450 Wh/kg cells as flown on solar HAPS demonstrators; lower cycle life"),
}


@dataclass(frozen=True)
class Laminate:
    """An envelope laminate (load fibre + gas barrier + UV/thermal outer film)."""
    key: str
    name: str
    areal_mass: float             # kg/m² per ply
    strength: float               # N/m, ultimate uniaxial line load per ply at room temperature
    absorptance: float            # solar absorptance of the outer surface
    emissivity: float
    note: str


LAMINATES = {
    "vectran": Laminate("vectran", "Vectran woven laminate", 0.20, 80e3, 0.20, 0.80,
                        "estimate: high-altitude airship laminates of Vectran fabric, ~0.15–0.25 kg/m², 60–100 kN/m"),
    "zylon": Laminate("zylon", "Zylon (PBO) laminate", 0.13, 90e3, 0.20, 0.80,
                      "estimate: PBO-fibre laminates, ~0.1–0.15 kg/m²; PBO loses strength under UV and humidity"),
    "uhmwpe": Laminate("uhmwpe", "UHMWPE (Dyneema) laminate", 0.15, 60e3, 0.20, 0.80,
                       "estimate: Dyneema laminates; creep under sustained load is the limit for months aloft"),
}
ENVELOPE_SAFETY_FACTOR = 4.0      # estimate: creep, seams (~80 % efficiency), cold, UV ageing over months (3–5 used)
SEAM_AND_PATCH_FRACTION = 0.10    # estimate: seams, load patches, valves: + of the envelope's mass


@dataclass(frozen=True)
class Structure:
    fin_areal_mass: float = 0.35          # kg/m² of fin planform (inflated or foam-core fins with their skin)
    ballonet_areal_mass: float = 0.08     # kg/m² of ballonet film (vented / ballonet designs)
    ballonet_area_fraction: float = 0.45  # ballonet film area / hull envelope area
    nose_and_keel_fraction: float = 0.06  # nose battens, nose cap, keel and catenary: of the envelope mass
    gondola_fraction: float = 0.20        # gondola structure: of the mass it carries
    avionics_mass: float = 8.0            # kg: flight computer, sensors, radios, valves, ballast release
    avionics_power: float = 60.0          # W, continuous ("hotel")
    night_heater_fraction: float = 0.10   # of payload + hotel power, at night (batteries and payload kept warm)
    mass_margin: float = 0.10             # growth margin on everything but the payload
    note: str = "estimates; the fractions stand for parts not yet designed"


STRUCTURE = Structure()


@dataclass(frozen=True)
class Propulsion:
    specific_mass: float = 1.6            # kg per kW of rated shaft power: motor, gearbox, propeller, mount, ESC
    motor_efficiency: float = 0.92
    esc_efficiency: float = 0.97
    profile_efficiency: float = 0.85      # propeller efficiency / actuator-disc ideal efficiency
    disc_loading_limit: float = 25.0      # N/m²: ceiling on the stern disc's thrust per area (large slow props)
    note: str = "estimates: large, slow, low-Re stratospheric propellers; a Boreas BEMT design replaces them later"


PROPULSION = Propulsion()


@dataclass(frozen=True)
class Gas:
    gas: str = "helium"
    purity: float = 0.98                  # air diffuses in over weeks
    leak_per_day: float = 0.001           # fraction of the gas mass lost per day through the laminate and seams
    note: str = "estimate: good superpressure envelopes lose ~0.05–0.3 %/day"


GAS = Gas()

PAYLOAD_POWER_PER_KG = 10.0               # W/kg: a communications or observation payload (estimate, 5–20 W/kg)
