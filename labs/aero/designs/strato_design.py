"""Strato — a design point: the inputs (``Design``, ``Mission``) and everything that follows from them without
simulating time (``configure``): hull, gas fill, envelope plies, mass budget, centre of gravity, trim, lift (V8.6).

Sizing logic (superpressure baseline):
1. The gas fill is chosen so the hull is just full, at the minimum superpressure, when the gas is coldest at float
   (night). Every warmer moment raises the superpressure; the lift does not change while the hull is full.
2. The minimum superpressure keeps the shape: above the nose's dynamic pressure (×1.2) at the fastest design speed,
   above the bending wrinkling pressure (og.envelope), and at least 50 Pa; ``night_superpressure`` adds a reserve:
   the gas a sealed hull can lose (leakage) before it goes slack at night and starts to sink.
3. The envelope carries the warmest (noon) superpressure plus bending with a safety factor: plies of laminate are
   added until it does; their mass goes into the budget.
4. Lift at float = displaced air − gas; what is left after the budget is surplus (ballast, or more payload).

A vented (zero-pressure) design fills to the same night volume, vents above ``valve_dp`` by day and needs a ballonet;
its lift is the same at the design point but it loses gas every day (strato_sim shows the consequence).
Masses and efficiencies come from ``strato_catalogue`` (estimates); the gas, atmosphere, envelope and mass physics from
Ogma (verified).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

import ogma.mass as om
from ogma import envelope as oe
from ogma import lifting_gas as lg
from ogma import stratosphere as st

import strato_aero as sa
import strato_catalogue as cat
import strato_geometry as sg

G0 = st.G0
MAX_PLIES = 6
GUST = 3.0                        # m/s, vertical gust for the aerodynamic bending estimate (estimate)


@dataclass(frozen=True)
class Design:
    """What the engineer (or the optimiser) chooses. SI units."""
    length: float = 15.0                   # m
    fineness: float = 15.0 / 3.5
    altitude: float = 20000.0              # m, float altitude (geometric)
    payload_mass: float = 5.0              # kg
    payload_power: float | None = None     # W by day; None = catalogue W/kg × payload mass
    payload_power_night: float | None = None   # W at night; None = the same as by day (e.g. 0 for a daylight camera)
    battery_energy: float = 3000.0         # Wh nameplate
    battery: str = "li-ion"
    pv: str = "gaas"
    solar_half_angle_deg: float = 55.0
    solar_x0: float = 0.18
    solar_x1: float = 0.72
    propulsion_power: float = 1500.0       # W, rated electrical power of all propulsion
    laminate: str = "vectran"
    gas_system: str = "sealed"             # "sealed" (superpressure) or "vented" (zero-pressure + ballonet)
    fin_scale: float = 1.0                 # fin span and chord × this (stability)
    valve_dp: float = 30.0                 # Pa, vented designs: the valve's opening superpressure
    night_superpressure: float = 0.0       # Pa above the shape minimum at night: the gas reserve against leakage

    def scaled(self, length: float) -> "Design":
        """The same design at another length: power, battery and payload scale with volume (a starting point)."""
        k = (length / self.length) ** 3
        return replace(self, length=length, payload_mass=self.payload_mass * k, battery_energy=self.battery_energy * k,
                       propulsion_power=self.propulsion_power * k)


@dataclass(frozen=True)
class Mission:
    lat: float = 50.0                      # °N
    lon: float = 10.0                      # °E
    month: int = 7
    year: int = 2026
    wind_percentile: float = 90.0          # the wind the station-keeping is sized for (daily speed percentile)
    station_radius: float = 20e3           # m: within this the payload serves its area
    battery_reserve: float = 0.10          # fraction of usable energy never planned to be used
    heading: float | None = None           # deg; None = into the month's mean wind


@dataclass
class Configuration:
    design: Design
    mission: Mission
    ship: sg.Strato
    air: st.State
    gas_mass: float
    dp_min: float
    dp_max: float
    superheat_day: float
    superheat_night: float
    plies: int
    hoop_line_load: float
    bending_moment: float
    budget: om.Budget
    x_cb: float
    cg: np.ndarray
    gross_lift: float
    net_lift: float
    system_mass: float
    surplus_lift: float
    v_design: float
    problems: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def pv(self) -> cat.PV:
        return cat.PVS[self.design.pv]

    @property
    def battery(self) -> cat.BatteryTech:
        return cat.BATTERIES[self.design.battery]

    @property
    def payload_power(self) -> float:
        d = self.design
        return d.payload_power if d.payload_power is not None else cat.PAYLOAD_POWER_PER_KG * d.payload_mass

    @property
    def payload_power_night(self) -> float:
        d = self.design
        return d.payload_power_night if d.payload_power_night is not None else self.payload_power

    @property
    def masses(self) -> dict:
        return self.budget.by_group()


def ship_of(d: Design) -> sg.Strato:
    lay = sg.Layout(solar_half_angle_deg=d.solar_half_angle_deg, solar_x0=d.solar_x0, solar_x1=d.solar_x1,
                    fin_root_chord=sg.Layout.fin_root_chord * d.fin_scale, fin_tip_chord=sg.Layout.fin_tip_chord * d.fin_scale,
                    fin_span=sg.Layout.fin_span * d.fin_scale)
    return sg.Strato(sg.Hull(d.length, d.length / d.fineness), lay)


def check_inputs(d: Design) -> list[str]:
    out = []
    for name in ("length", "fineness", "battery_energy", "propulsion_power"):
        if getattr(d, name) <= 0:
            out.append(f"{name} must be positive")
    if d.payload_mass < 0:
        out.append("payload mass must not be negative")
    if d.pv not in cat.PVS:
        out.append(f"unknown PV {d.pv!r}")
    if d.battery not in cat.BATTERIES:
        out.append(f"unknown battery {d.battery!r}")
    if d.laminate not in cat.LAMINATES:
        out.append(f"unknown laminate {d.laminate!r}")
    if d.gas_system not in ("sealed", "vented"):
        out.append("gas_system must be 'sealed' or 'vented'")
    if not 0.0 <= d.solar_x0 < d.solar_x1 <= 1.0:
        out.append("the array must satisfy 0 ≤ solar_x0 < solar_x1 ≤ 1")
    if not 5000.0 <= d.altitude <= 30000.0:
        out.append("float altitude outside 5–30 km")
    return out


def _static_bending(ship: sg.Strato, net_lift_density: float, point_masses: list, g: float) -> float:
    """Peak bending moment [N m] of the hull floating in equilibrium: lift ∝ section area, the envelope's weight ∝
    its local area, the rest as point loads (spread over 2 % of the length). Any imbalance (surplus lift) is spread
    like the buoyancy, as ballast or payload would be."""
    h = ship.hull
    x = np.linspace(0.0, h.length, 801)
    r = h.r(x)
    area_x = math.pi * r ** 2
    w = net_lift_density * g * area_x                                  # N/m up
    weight = np.zeros_like(x)
    spread = 0.02 * h.length
    for kg, xi in point_masses:
        weight += kg * g * np.exp(-0.5 * ((x - xi) / spread) ** 2) / (spread * math.sqrt(2 * math.pi))
    balance = np.trapezoid(w, x) - np.trapezoid(weight, x)
    w = w - weight - balance * area_x / np.trapezoid(area_x, x)
    _, m = oe.beam_loads(x, w)
    return float(np.max(np.abs(m)))


def configure(d: Design, m: Mission, superheat_day: float = 15.0, superheat_night: float = -5.0,
              v_design: float | None = None) -> Configuration:
    """The design point at float. ``superheat_*`` [K] come from the thermal model (strato_feasibility iterates);
    ``v_design`` [m/s] is the fastest airspeed the ship must keep its shape at (default: its propulsion's top speed)."""
    problems = check_inputs(d)
    if problems:
        raise ValueError("; ".join(problems))
    try:
        ship = ship_of(d)
    except ValueError as e:
        raise ValueError(str(e)) from None
    h, lay = ship.hull, ship.layout
    air = st.state(d.altitude)
    g = air.gravity
    lam = cat.LAMINATES[d.laminate]
    s = cat.STRUCTURE
    pv = cat.PVS[d.pv]
    bat = cat.BATTERIES[d.battery]
    if v_design is None:
        v_design = sa.max_speed(ship, d.propulsion_power, d.altitude)
    q = air.dynamic_pressure(v_design)

    # --- masses that do not depend on the envelope's plies (positions: body frame, x aft, z up)
    x_cb = h.centre_of_buoyancy
    r_cb = float(h.r([x_cb])[0])
    z_gondola = -(r_cb + lay.gondola_drop * h.diameter + 0.5 * lay.gondola_height * h.diameter)
    x_fin = lay.fin_trailing_edge * h.length - 0.5 * lay.fin_root_chord * h.length
    fin_area = ship.fin_planform_area * lay.fin_count
    fac = ship.facets(24, 24)
    a_arr = fac["area"] * fac["array"]
    arr_centre = (fac["centre"] * a_arr[:, None]).sum(0) / max(a_arr.sum(), 1e-9)
    battery_kg = d.battery_energy / bat.specific_energy
    prop_kg = cat.PROPULSION.specific_mass * d.propulsion_power / 1000.0
    x_stern = h.length
    x_side = lay.side_prop_station * h.length
    gondola_payload = d.payload_mass + battery_kg + s.avionics_mass

    # --- superpressure, bending, plies (two passes: the envelope's weight changes the bending a little)
    T_night = air.temperature + superheat_night
    T_day = air.temperature + superheat_day
    k1, k2, _ = sa.lamb_k(h.fineness)
    alpha_gust = math.atan2(GUST, max(v_design, 1.0))
    m_aero = q * h.volume * (k2 - k1) * math.sin(2 * alpha_gust)
    plies, envelope_kg, m_bend = 1, 0.0, 0.0
    for _ in range(3):
        envelope_kg = plies * lam.areal_mass * h.surface_area * (1 + cat.SEAM_AND_PATCH_FRACTION)
        rho_net = lg.net_lift_per_volume(air.pressure, air.temperature, 0.0, superheat_night, cat.GAS.gas, cat.GAS.purity)
        points = [(fin_area * s.fin_areal_mass, x_fin), (gondola_payload * (1 + s.gondola_fraction), x_cb),
                  (prop_kg * 0.6, x_stern), (prop_kg * 0.4, x_side), (pv.areal_mass * ship.solar_area, arr_centre[0])]
        m_static = _static_bending(ship, rho_net - envelope_kg / h.volume, points, g)
        m_bend = m_static + m_aero
        dp_shape = max(50.0, 1.2 * q, oe.wrinkling_pressure(m_bend, r_cb))
        dp_min = dp_shape + (d.night_superpressure if d.gas_system == "sealed" else 0.0)
        gas_mass = lg.fill_for_full(h.volume, air.pressure, T_night, dp_min, cat.GAS.gas, cat.GAS.purity)
        if d.gas_system == "sealed":
            dp_max = lg.superpressure(gas_mass, h.volume, air.pressure, T_day, cat.GAS.gas, cat.GAS.purity)
        else:
            dp_max = max(d.valve_dp, dp_min)
        x, r = h.profile(801)
        keep = (x > 0.02 * h.length) & (x < 0.95 * h.length)
        hoop = float(np.max(oe.membrane_forces(x[keep], r[keep], dp_max)["hoop"]))
        line = max(hoop, dp_max * r_cb / 2 + oe.bending_line_load(m_bend, r_cb))
        need = math.ceil(oe.required_strength(line, cat.ENVELOPE_SAFETY_FACTOR) / lam.strength)
        if need == plies:
            break
        plies = max(1, need)

    # --- the budget (the gondola, with payload, battery and avionics, is placed to trim the CG over the CB)
    b = om.Budget()
    b.add(om.Item("envelope", envelope_kg, (x_cb, 0.0, 0.0), group="envelope"))
    b.add(om.Item("fins", fin_area * s.fin_areal_mass, (x_fin, 0.0, 0.0), group="structure"))
    b.add(om.Item("nose and keel", s.nose_and_keel_fraction * envelope_kg, (0.45 * h.length, 0.0, -0.6 * r_cb), group="structure"))
    if d.gas_system == "vented":
        b.add(om.Item("ballonet", s.ballonet_areal_mass * s.ballonet_area_fraction * h.surface_area,
                      (x_cb, 0.0, -0.5 * r_cb), group="structure"))
    b.add(om.Item("solar array", pv.areal_mass * ship.solar_area, tuple(arr_centre), group="energy"))
    b.add(om.Item("stern propulsion", 0.6 * prop_kg, (x_stern, 0.0, 0.0), group="propulsion"))
    b.add(om.Item("vectoring propulsion", 0.4 * prop_kg, (x_side, 0.0, 0.0), group="propulsion"))
    fixed = om.Budget(list(b.items))
    margin_kg = s.mass_margin * (fixed.mass + battery_kg + s.avionics_mass + s.gondola_fraction * gondola_payload)
    b.add(om.Item("margin", margin_kg, (x_cb, 0.0, 0.0), group="margin"))
    pre = b.properties()
    gondola_kg = gondola_payload * (1 + s.gondola_fraction)
    moment = pre.mass * pre.com[0]
    x_g, _ = om.place_mass_for_cg(pre.mass, moment, gondola_kg, x_cb)
    b.add(om.Item("payload", d.payload_mass, (x_g, 0.0, z_gondola), group="payload"))
    b.add(om.Item("battery", battery_kg, (x_g, 0.0, z_gondola), group="energy"))
    b.add(om.Item("avionics", s.avionics_mass, (x_g, 0.0, z_gondola), group="systems"))
    b.add(om.Item("gondola structure", s.gondola_fraction * gondola_payload, (x_g, 0.0, z_gondola), group="structure"))
    props = b.properties()

    gross = air.density * h.volume
    net = gross - gas_mass
    problems, warnings = [], []
    if d.gas_system == "vented" and dp_shape > d.valve_dp:
        warnings.append(f"vented design: the shape needs {dp_shape:.0f} Pa, above the {d.valve_dp:.0f} Pa valve")
    if not 0.25 * h.length <= x_g <= 0.65 * h.length:
        problems.append(f"cannot trim: the gondola would have to sit at x = {x_g:.1f} m ({x_g / h.length:.0%} of L)")
    if props.com[2] >= 0.0:
        problems.append("the centre of gravity is not below the centre of buoyancy (no pendulum stability)")
    if plies > MAX_PLIES:
        problems.append(f"envelope needs {plies} plies of {lam.name} (> {MAX_PLIES}): superpressure {dp_max:.0f} Pa too high")
    surplus = net - props.mass
    if surplus < 0:
        problems.append(f"does not float at {d.altitude / 1000:.1f} km: lift {net:.1f} kg < mass {props.mass:.1f} kg")
    stab = sa.stability(ship, props.com[0])
    if stab["margin"] < 0:
        warnings.append(f"statically unstable in pitch/yaw (fin/Munk margin {stab['margin']:+.2f}): needs active control")
    return Configuration(d, m, ship, air, gas_mass, dp_min, dp_max, superheat_day, superheat_night, plies, line,
                         m_bend, b, x_cb, props.com, gross, net, props.mass, surplus, v_design, problems, warnings)


def max_float_altitude(c: Configuration) -> float:
    """The highest altitude [m] at which this ship (its mass, its hull, the same fill rule) still floats."""
    d, h = c.design, c.ship.hull

    def surplus(z):
        a = st.state(z)
        m_gas = lg.fill_for_full(h.volume, a.pressure, a.temperature + c.superheat_night, c.dp_min, cat.GAS.gas, cat.GAS.purity)
        return a.density * h.volume - m_gas - c.system_mass

    lo, hi = 0.0, 40000.0
    if surplus(lo) < 0:
        return float("nan")
    if surplus(hi) > 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if surplus(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
