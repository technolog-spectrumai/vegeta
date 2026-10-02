"""Actuator, motor and joint library shared by the product notebooks (robot dog, Myropods, the Onager series).

A catalogue of explicit entries — every number is an engineer's input with a source, nothing is
fitted — plus two helpers: pick the lightest actuator that holds a torque with a safety factor, and
the electrical power of holding a torque. Notebooks select from here instead of inventing their own
motor dictionaries, so a change of catalogue reaches every machine.
"""
from dataclasses import dataclass, asdict

import pandas as pd


@dataclass(frozen=True)
class Actuator:
    """One actuator: a hobby servo, a serial-bus smart servo, or a quasi-direct-drive (QDD) module."""

    key: str
    kind: str                 # "servo", "smart servo", "sealed servo", "qdd", "industrial module", "hub motor", "linear (jaw-equivalent)", ...
    mass_g: float
    stall_Nm: float           # peak / stall torque
    rated_Nm: float           # continuous torque (thermal)
    stall_A: float            # current at stall
    voltage_V: float
    no_load_rpm: float
    self_locking: bool = False   # worm / lead-screw output or a holding brake: holds a load with the motor unpowered
    source: str = ""

    @property
    def name(self) -> str:
        return self.key

    def as_dict(self) -> dict:
        d = asdict(self); d["name"] = self.key
        return d

    def holding_power(self, torque_Nm: float, hold_current_fraction: float = 1.0) -> float:
        """Electrical power [W] to hold ``torque_Nm``: the current is proportional to the torque fraction of
        stall (a DC motor at rest); a self-locking drive holds for nothing."""
        if self.self_locking:
            return 0.0
        f = min(abs(torque_Nm) / self.stall_Nm, 1.0)
        return f * self.stall_A * self.voltage_V * hold_current_fraction

    def safety_factor(self, torque_Nm: float) -> float:
        return float("inf") if torque_Nm == 0 else self.stall_Nm / abs(torque_Nm)


@dataclass(frozen=True)
class Joint:
    """A joint type: degrees of freedom, pin, range. Pins are steel (C45, yield 400 MPa in the notebooks)."""

    key: str
    dof: int
    pin_diameter_mm: float
    range_deg: float          # per axis, +/- from neutral
    description: str = ""


CATALOG = [
    # hobby servos, 7.4 V, metal gears (datasheet-class values; the sizes the Persephone legs choose from)
    Actuator("sub-micro 9 g", "servo", 9.0, 0.22, 0.10, 0.9, 7.4, 110, False, "generic 9 g metal-gear micro servo datasheet"),
    Actuator("micro 13 g", "servo", 13.0, 0.32, 0.14, 1.1, 7.4, 100, False, "generic 13 g servo datasheet"),
    Actuator("micro-metal 20 g", "servo", 20.0, 0.55, 0.22, 1.4, 7.4, 90, False, "generic 20 g metal-gear servo datasheet"),
    Actuator("mini 32 g", "servo", 32.0, 1.10, 0.40, 2.0, 7.4, 80, False, "generic 32 g mini servo datasheet"),
    Actuator("standard 55 g", "servo", 55.0, 2.20, 0.80, 3.0, 7.4, 60, False, "generic 55 g standard servo datasheet"),
    Actuator("micro-metal 22 g worm", "servo", 22.0, 0.50, 0.22, 1.4, 7.4, 45, True, "assumed: a 20 g servo with a worm output stage (self-locking, slower)"),
    Actuator("mini 32 g worm", "servo", 36.0, 1.00, 0.40, 2.0, 7.4, 40, True, "assumed: a 32 g servo with a worm output stage (self-locking, slower)"),
    Actuator("standard 60 g worm", "servo", 60.0, 2.00, 0.80, 3.0, 7.4, 30, True, "assumed: a 55 g servo with a worm output stage (self-locking, slower)"),
    # serial-bus smart servos, 12 V (Cleopatra and Apheloria legs and joints)
    Actuator("smart servo 6 Nm", "smart servo", 70.0, 6.0, 2.0, 3.0, 12.0, 55, False, "serial-bus smart servo class, supplier datasheet"),
    Actuator("smart servo 12 Nm", "smart servo", 120.0, 12.0, 4.0, 4.5, 12.0, 45, False, "serial-bus smart servo class, supplier datasheet"),
    Actuator("smart servo 25 Nm", "smart servo", 250.0, 25.0, 8.0, 6.0, 24.0, 40, False, "serial-bus smart servo class, supplier datasheet"),
    # quasi-direct-drive modules (the robot dog's legs, Apheloria's joints)
    Actuator("qdd 24 Nm", "qdd", 480.0, 24.0, 8.0, 18.0, 24.0, 300, False, "quasi-direct-drive leg module class (planetary 1:6), supplier datasheet"),
    Actuator("qdd 60 Nm", "qdd", 650.0, 60.0, 20.0, 25.0, 48.0, 200, False, "quasi-direct-drive module class (planetary 1:9), supplier datasheet"),
    # industrial joint modules and hub motors, 48 V (the Onager series: 380 kg wheel-leg machines, notebook 20)
    Actuator("cycloidal 400 Nm, brake", "industrial module", 6500.0, 400.0, 150.0, 120.0, 48.0, 60, True,
             "assumed: 48 V BLDC with a 1:40 cycloidal stage and a holding brake, light-industrial joint-module class"),
    Actuator("cycloidal 800 Nm, brake", "industrial module", 11000.0, 800.0, 300.0, 180.0, 48.0, 40, True,
             "assumed: 48 V BLDC with a 1:60 cycloidal stage and a holding brake, light-industrial joint-module class"),
    Actuator("hub motor 3 kW", "hub motor", 9000.0, 240.0, 60.0, 150.0, 48.0, 500, False,
             "assumed: in-wheel BLDC hub motor class of light electric vehicles, 48 V, 3 kW peak (stall torque x no-load speed / 4)"),
    # manipulator joints and the pincer drive (Onager Manus, notebook 21)
    Actuator("harmonic 60 Nm, brake", "industrial module", 1600.0, 60.0, 25.0, 15.0, 48.0, 40, True,
             "assumed: 48 V BLDC with a 1:100 strain-wave gear and a holding brake, cobot wrist-joint class"),
    Actuator("harmonic 150 Nm, brake", "industrial module", 3200.0, 150.0, 60.0, 30.0, 48.0, 30, True,
             "assumed: 48 V BLDC with a 1:120 strain-wave gear and a holding brake, cobot elbow-joint class"),
    Actuator("jaw screw 6 kN", "linear (jaw-equivalent)", 2200.0, 360.0, 120.0, 20.0, 48.0, 9.5, False,
             "assumed: 48 V ball-screw linear actuator, 6 kN stall, 60 mm/s no-load, acting on the jaw at a 60 mm "
             "lever: given here at the jaw pivot (6 kN x 0.06 m = 360 N m, 1.0 rad/s = 9.5 rpm)"),
    # sealed underwater drives (Sikarian Lobster, notebook 24): oil-filled / potted servo cases rated for 10 m+
    Actuator("sealed servo 1.5 Nm", "sealed servo", 55.0, 1.5, 0.5, 1.6, 12.0, 60, False,
             "assumed: 12 V potted underwater micro servo class (shaft seal), rated 30 m"),
    Actuator("sealed servo 3 Nm", "sealed servo", 95.0, 3.0, 1.0, 2.5, 12.0, 50, False,
             "assumed: 12 V oil-compensated underwater servo class (aluminium case, shaft seal), rated 30 m"),
    Actuator("sealed servo 12 Nm, worm", "sealed servo", 300.0, 12.0, 5.0, 4.0, 12.0, 15, True,
             "assumed: 12 V underwater gripper drive class, worm output (holds the grip unpowered), rated 30 m"),
    # the street sweeper's broom and suction fan (Onager Sweeper, notebook 23)
    Actuator("broom drive 1.5 kW", "gearmotor", 9000.0, 120.0, 50.0, 60.0, 48.0, 240, False,
             "assumed: 48 V BLDC through a 1:12 planetary stage, brush-disc sweeper drive class (disc brooms run 100-200 rpm)"),
    Actuator("suction fan 4 kW", "fan motor", 8000.0, 16.0, 9.0, 110.0, 48.0, 6000, False,
             "assumed: 48 V outer-rotor BLDC blower motor class, 9 N m continuous at 4200 rpm (4 kW), direct drive on the impeller"),
]
JOINTS = [
    Joint("leg hip pin", 1, 5.0, 60.0, "a leg's hip pin in a printed boss (Persephone)"),
    Joint("two-axis body joint", 2, 5.0, 45.0, "pitch + yaw between two Persephone segments: tongue and fork, one pin per axis"),
    Joint("two-axis body joint, large", 2, 10.0, 45.0, "Cleopatra and Apheloria body joints"),
    Joint("dog hip / knee pin", 1, 10.0, 110.0, "robot dog leg pins in clevises"),
    Joint("onager shoulder / knee pin", 1, 40.0, 100.0, "Onager Sentinel leg pins (steel, in the actuator output flange)"),
    Joint("onager wheel axle", 1, 45.0, 360.0, "Onager Sentinel stub axle carrying the hub motor (continuous rotation)"),
    Joint("manus arm pin", 1, 30.0, 150.0, "Onager Manus arm joints (shoulder, elbow, wrist): pin in the module flange"),
    Joint("manus jaw pin", 1, 16.0, 60.0, "Onager Manus pincer: both jaws on one hardened pin"),
    Joint("sweeper broom spindle", 1, 40.0, 360.0, "Onager Sweeper disc broom: the drive's output spindle (continuous rotation)"),
    Joint("lobster leg pin", 1, 5.0, 60.0, "Sikarian Lobster leg hips (yaw, pitch): stainless pins in the servo horn"),
    Joint("lobster tail knuckle", 2, 6.0, 60.0, "Sikarian Lobster tail: yaw + pitch knuckle, a rubber boot over it"),
    Joint("lobster jaw pin", 1, 6.0, 60.0, "Sikarian Lobster pincer: both jaws on one hardened pin"),
]


@dataclass(frozen=True)
class LinearActuator:
    """A linear drive: force [N] and speed [mm/s] instead of torque and rpm (the Onager Atlas's lift and tilt)."""

    key: str
    kind: str
    mass_g: float
    stall_N: float
    rated_N: float
    no_load_mm_s: float
    stall_A: float
    voltage_V: float
    self_locking: bool = False
    source: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


LINEAR = [
    LinearActuator("lift screw 8 kN", "ball screw + brake", 12000.0, 8000.0, 3000.0, 200.0, 60.0, 48.0, True,
                   "assumed: 48 V BLDC on a 32 mm ball screw (10 mm lead) with a holding brake, forklift-retrofit class"),
    LinearActuator("tilt screw 12 kN", "ball screw + brake", 6000.0, 12000.0, 5000.0, 60.0, 40.0, 48.0, True,
                   "assumed: 48 V electric cylinder (5 mm lead) with a brake, mast-tilt class"),
]


def get_linear(key: str) -> LinearActuator:
    for a in LINEAR:
        if a.key == key:
            return a
    raise KeyError(f"no linear actuator {key!r}; known: {[a.key for a in LINEAR]}")


def get(key: str) -> Actuator:
    for a in CATALOG:
        if a.key == key:
            return a
    raise KeyError(f"no actuator {key!r}; known: {[a.key for a in CATALOG]}")


def joint(key: str) -> Joint:
    for j in JOINTS:
        if j.key == key:
            return j
    raise KeyError(f"no joint {key!r}; known: {[j.key for j in JOINTS]}")


def table(kind: str | None = None) -> pd.DataFrame:
    """The catalogue as a DataFrame indexed by key, lightest first (optionally one kind)."""
    rows = [a.as_dict() for a in CATALOG if kind is None or a.kind == kind]
    return pd.DataFrame(rows).set_index("key").drop(columns=["name"]).sort_values("mass_g")


def select(torque_Nm: float, sf: float = 1.5, kind: str | None = None, self_locking: bool | None = None, max_mass_g: float | None = None) -> Actuator:
    """The lightest actuator whose stall torque is at least ``sf × torque``."""
    cands = [a for a in CATALOG if (kind is None or a.kind == kind) and (self_locking is None or a.self_locking == self_locking)
             and (max_mass_g is None or a.mass_g <= max_mass_g) and a.stall_Nm >= sf * abs(torque_Nm)]
    if not cands:
        raise ValueError(f"no actuator in the catalogue holds {torque_Nm:.2f} Nm with SF {sf} (kind={kind}, self_locking={self_locking})")
    return min(cands, key=lambda a: a.mass_g)
