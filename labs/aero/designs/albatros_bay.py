"""ALBATROS's bay (notebook 50): the multirotor that carries the cargo from the hovering Albatros down to the deck,
and what the bay holds — one cargo hexacopter, or up to three small quadcopters.

The multirotor's flight is short: it is released above the vessel, flies down to the deck with the cargo and stays
there. So it is sized by lifting, not by endurance (``flight_s``, a few minutes with a go-around). ``size_multirotor``
closes its mass by fixed point:

- **rotors**: the hover disk loading ``disk_loading`` [N/m²] gives the propeller diameter (a folding airframe in a
  fuselage bay cannot have large disks); the ideal hover power is momentum theory's (``ogma.rotors``), the electric
  power divides it by the figure of merit and the drive efficiency;
- **drive**: motors, controllers and folding propellers as mass per newton of maximum thrust, the maximum thrust
  ``thrust_to_weight`` times the weight (2: a hexacopter then still hovers on five rotors with control left);
- **battery**: the larger of the energy for ``flight_s`` (specific energy of the pack, a usable fraction) and the
  hover power at the pack's continuous specific power;
- **frame**: a fraction of the take-off mass, plus a hinge per arm; **fixed**: flight controller, GNSS, radio link,
  the cargo hook and the lights — the mass every vehicle pays once, which is what three small ones pay three times.

Folded for the bay, the arms swing in the hub's plane to lie fore and aft (alternately above and below the hub
plate, the folding propellers along them) and the cargo box hangs under the hub: a flat package ``folded_m`` =
(length, width, height), an arm and the hub long, the hub and the motors wide. ``BayOption`` puts one or several of
them in a row along the fuselage and gives what the airframe needs from it: the mass dropped, the bay's length,
width and height, and the fuselage's cross-section around it (a rounded box, not a circle: the bay is flat).

``options`` builds the candidates for a cargo mass: one hexacopter, three quadcopters with a third of the cargo each,
and three ``CAMERA_QUAD`` (a small folding camera quadcopter as bought, with its published take-off margin as cargo).
``delivery`` gives what arrives on the deck when each vehicle succeeds with a probability (an assumption to vary).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ogma.rotors import ideal_hover_power
from ogma.units import G
from ogma.units import RHO0 as RHO

CARGO_DENSITY = 250.0            # kg/m³: boxed spares, medicine and tools as packed


@dataclass(frozen=True)
class Multirotor:
    name: str
    rotors: int
    cargo_kg: float
    empty_kg: float               # without the cargo (frame, drive, battery, fixed equipment)
    battery_kg: float
    prop_diameter_m: float
    hover_power_w: float          # electric, with the cargo
    flight_s: float
    folded_m: tuple               # (length, width, height) of the folded vehicle with its cargo
    arm_m: float                  # hub centre to motor axis
    rotor_out_ok: bool            # hovers with one rotor stopped
    budget: dict

    @property
    def mass_kg(self) -> float:
        return self.empty_kg + self.cargo_kg

    @property
    def span_m(self) -> float:    # unfolded, propeller tip to propeller tip
        return 2 * self.arm_m + self.prop_diameter_m


def size_multirotor(cargo_kg: float, rotors: int = 6, *, flight_s: float = 300.0, disk_loading: float = 110.0,
                    thrust_to_weight: float = 2.0, figure_of_merit: float = 0.65, drive_efficiency: float = 0.85,
                    pack_wh_kg: float = 160.0, usable: float = 0.8, pack_w_kg: float = 2500.0,
                    drive_kg_per_n: float = 0.0062, frame_fraction: float = 0.16, hinge_kg: float = 0.03,
                    fixed_kg: float = 0.32, name: str | None = None) -> Multirotor:
    """The lightest multirotor of ``rotors`` that carries ``cargo_kg`` for ``flight_s`` (see the module text)."""
    if rotors < 3 or cargo_kg < 0:
        raise ValueError("need at least 3 rotors and a cargo mass >= 0")
    m = cargo_kg + fixed_kg + 0.5
    for _ in range(200):
        W = m * G
        area = W / disk_loading
        p_el = rotors * ideal_hover_power(W / rotors, area / rotors, RHO) / (figure_of_merit * drive_efficiency)
        battery = max(p_el * flight_s / 3600 / (pack_wh_kg * usable), p_el / pack_w_kg)
        drive = drive_kg_per_n * thrust_to_weight * W
        frame = frame_fraction * m + hinge_kg * rotors
        new = cargo_kg + fixed_kg + battery + drive + frame
        if abs(new - m) < 1e-6:
            break
        m = 0.5 * (m + new)
    d = math.sqrt(4 * area / (rotors * math.pi))
    arm = 1.05 * d / (2 * math.sin(math.pi / rotors))            # tips 5 % of a diameter apart
    hub = max(0.28 * arm, 0.06)
    motor = 0.25 * d ** 0.5 * 0.2 + 0.02                          # motor can diameter, growing with the propeller
    length, width = arm + hub + 0.05, 2 * hub + rotors / 2 * motor + 0.02
    cargo_h = cargo_kg / CARGO_DENSITY / (0.9 * length * 0.9 * width)       # the box under the hub, 90 % of the footprint
    folded = (length, width, 0.12 + cargo_h)                                 # the stacked arms and the hub: 0.12 m
    # one rotor out: an even hexacopter (or more) stops the opposite rotor too and hovers on the rest; a quadcopter cannot
    out_ok = rotors >= 6 and (rotors - 2) / rotors * thrust_to_weight >= 1.15
    budget = {"cargo": cargo_kg, "battery": battery, "motors, controllers, propellers": drive, "frame and arm hinges": frame,
              "flight controller, GNSS, link, hook, lights": fixed_kg}
    return Multirotor(name or f"{'hexa' if rotors == 6 else 'quad' if rotors == 4 else str(rotors) + '-rotor '}copter for {cargo_kg:g} kg",
                      rotors, cargo_kg, m - cargo_kg, battery, d, p_el, flight_s, folded, arm, out_ok, budget)


# A small folding camera quadcopter as bought (the "DJI-like" class): typical published figures of the 0.9 kg class.
# Its cargo is the published take-off margin (maximum take-off mass less its own); it has no cargo hook.
CAMERA_QUAD = Multirotor("camera-class folding quadcopter (as bought)", 4, 0.15, 0.90, 0.34, 0.24, 110.0, 1500.0,
                         (0.23, 0.10, 0.14), 0.19, False,
                         {"cargo": 0.15, "the quadcopter as bought": 0.90})


@dataclass(frozen=True)
class BayOption:
    name: str
    vehicles: tuple                # the multirotors in the bay
    note: str = ""

    @property
    def count(self) -> int:
        return len(self.vehicles)

    @property
    def cargo_kg(self) -> float:
        return sum(v.cargo_kg for v in self.vehicles)

    @property
    def dropped_kg(self) -> float:
        return sum(v.mass_kg for v in self.vehicles)

    @property
    def largest_item_kg(self) -> float:
        return max(v.cargo_kg for v in self.vehicles)

    def bay(self, clearance: float = 0.04, release_kg: float = 0.6, door_kg_m2: float = 3.5) -> dict:
        """The bay the airframe must give [m] (vehicles in a row along the fuselage, ``clearance`` around each), the
        fuselage's cross-section around it (the bay's width plus the two side walls and frames; its height plus the
        wing's carry-through and the systems tunnel above it), and the mass of the release gear and doors."""
        length = sum(v.folded_m[0] + clearance for v in self.vehicles) + clearance
        width = max(v.folded_m[1] for v in self.vehicles) + 2 * clearance
        height = max(v.folded_m[2] for v in self.vehicles) + clearance
        return {"length_m": length, "width_m": width, "height_m": height, "fuselage_width_m": width + 0.06,
                "fuselage_height_m": height + 0.14, "volume_l": length * width * height * 1000,
                "fittings_kg": release_kg * self.count + door_kg_m2 * length * width}

    def summary(self) -> dict:
        b = self.bay()
        return {"vehicles": self.count, "rotors each": self.vehicles[0].rotors, "cargo delivered [kg]": round(self.cargo_kg, 2),
                "largest single item [kg]": round(self.largest_item_kg, 2), "each vehicle, empty [kg]": round(self.vehicles[0].empty_kg, 2),
                "dropped mass [kg]": round(self.dropped_kg, 2), "cargo / dropped": round(self.cargo_kg / self.dropped_kg, 2),
                "propeller [m]": round(self.vehicles[0].prop_diameter_m, 2), "unfolded span [m]": round(self.vehicles[0].span_m, 2),
                "bay L x W x H [m]": f"{b['length_m']:.2f} x {b['width_m']:.2f} x {b['height_m']:.2f}",
                "fuselage W x H [m]": f"{b['fuselage_width_m']:.2f} x {b['fuselage_height_m']:.2f}", "bay fittings [kg]": round(b["fittings_kg"], 1),
                "one rotor out": "flies on" if self.vehicles[0].rotor_out_ok else "falls"}


def options(cargo_kg: float = 10.0, **kw) -> dict:
    """The bay's candidates for ``cargo_kg``: ``hexacopter`` (one, all the cargo), ``three quadcopters`` (a third each),
    ``two hexacopters`` (half each: what a scaled-down hexacopter costs) and ``three camera quadcopters`` (as bought)."""
    hexa = size_multirotor(cargo_kg, 6, **kw)
    quad = size_multirotor(cargo_kg / 3, 4, **kw)
    half = size_multirotor(cargo_kg / 2, 6, **kw)
    return {"hexacopter": BayOption("one cargo hexacopter", (hexa,), "the whole cargo as one item"),
            "three quadcopters": BayOption("three cargo quadcopters", (quad,) * 3, "the cargo split in three"),
            "two hexacopters": BayOption("two half-size hexacopters", (half,) * 2, "the cargo split in two"),
            "three camera quadcopters": BayOption("three camera-class quadcopters", (CAMERA_QUAD,) * 3,
                                                  "small folding quadcopters as bought: the cargo is their take-off margin")}


def delivery(option: BayOption, p_vehicle: float = 0.95, p_rotor_out: float = 0.02) -> dict:
    """What arrives when each vehicle's flight to the deck succeeds with ``p_vehicle`` apart from rotor failures, and
    a rotor (motor, controller or propeller) fails with ``p_rotor_out`` per flight — which a vehicle that flies on
    with a rotor out survives. Both probabilities are assumptions: the comparison is the point, not the values."""
    v = option.vehicles[0]
    p = p_vehicle * (1.0 if v.rotor_out_ok else (1 - p_rotor_out) ** v.rotors)
    n = option.count
    pmf = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    return {"per vehicle": p, "all of the cargo": pmf[n], "at least part": 1 - pmf[0], "nothing": pmf[0],
            "expected cargo [kg]": p * option.cargo_kg}


def table(cargo_kg: float = 10.0, **kw):
    import pandas as pd

    rows = {}
    for o in options(cargo_kg, **kw).values():
        d = delivery(o)
        rows[o.name] = {**o.summary(), "P(all the cargo)": round(d["all of the cargo"], 3), "P(nothing)": round(d["nothing"], 4),
                        "expected cargo [kg]": round(d["expected cargo [kg]"], 2)}
    return pd.DataFrame(rows)


__all__ = ["Multirotor", "size_multirotor", "CAMERA_QUAD", "BayOption", "options", "delivery", "table", "CARGO_DENSITY"]
