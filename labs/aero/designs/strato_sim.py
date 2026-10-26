"""Strato — time-domain simulations of a configured ship (Stage V8.11): day/night energy and thermal cycles, station-
keeping against wind, multi-day endurance with gas leakage (sealed) or venting (vented), and the ascent to float.

Each simulation is deterministic for its seed and returns plain arrays (``dict`` of numpy arrays plus scalars) that
the studies plot and the tests check. Physics: Ogma (atmosphere, gas, sun), ``strato_aero`` (power), ``strato_thermal``
(superheat, cell temperature). The station-keeping rule is a budget, not an autopilot: fly at the airspeed that holds
the station (wind plus a pull back towards the centre) as far as the power allows; real guidance is Piccolo's.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

import ogma.wind as ow
from ogma import lifting_gas as lg
from ogma import solar as so
from ogma import stratosphere as st

import strato_aero as sa
import strato_catalogue as cat
import strato_design as sd
import strato_geometry as sg
import strato_thermal as sth
import strato_wind as sw

N_X, N_PHI = 30, 24               # facets for the solar and thermal models


class Ship:
    """What every simulation needs from a ``Configuration``, precomputed: facets, the thermal hull, the power curve."""

    def __init__(self, c: sd.Configuration):
        self.c = c
        d, ship = c.design, c.ship
        f = ship.facets(N_X, N_PHI)
        self.normals, self.area, self.array = f["normal"], f["area"], f["array"]
        pv, lam = c.pv, sd.cat.LAMINATES[d.laminate]
        absorp = self.array * pv.absorptance + (1 - self.array) * lam.absorptance
        emis = self.array * pv.emissivity + (1 - self.array) * lam.emissivity
        cap = (c.masses.get("envelope", 0.0) + pv.areal_mass * ship.solar_area) * sth.CP_LAMINATE
        self.thermal = sth.Hull(self.normals, self.area, self.array, absorp, emis, ship.hull.length, cap, c.gas_mass)
        self.v_max = sa.max_speed(ship, d.propulsion_power, d.altitude)
        self.v_grid = np.linspace(0.0, max(self.v_max, 1.0), 41)
        self.p_grid = np.array([sa.electric_power(ship, v, d.altitude)["power"] for v in self.v_grid])
        # loads by day; at night the payload's night power, plus heaters for the batteries and the payload
        self.loads = cat.STRUCTURE.avionics_power + c.payload_power
        self.loads_night = (cat.STRUCTURE.avionics_power + c.payload_power_night
                            + cat.STRUCTURE.night_heater_fraction * self.loads)
        self.night_extra = self.loads_night - self.loads
        bat = c.battery
        self.usable_wh = d.battery_energy * bat.max_dod
        self.reserve_wh = c.mission.battery_reserve * self.usable_wh

    def power_for(self, v):
        """Electrical propulsion power [W] at airspeed ``v`` (table; inf beyond the top speed)."""
        v = np.asarray(v, float)
        p = np.interp(v, self.v_grid, self.p_grid)
        return np.where(v > self.v_max + 1e-9, np.inf, p)

    def speed_for(self, power):
        return np.interp(power, self.p_grid, self.v_grid)


def _t0(m: sd.Mission) -> datetime:
    return datetime(m.year, m.month, 15, 0, 0, tzinfo=timezone.utc)


def heading_into_wind(m: sd.Mission, altitude: float) -> float:
    if m.heading is not None:
        return m.heading
    p = sw.profile(altitude, m.month, m.lat)
    u, v = float(p["u"]), float(p["v"])
    # the wind blows towards (u, v); the nose points into it: towards (−u, −v)
    return math.degrees(math.atan2(-u, -v)) % 360.0


def solar_step(sh: Ship, t: datetime, heading: float, thermal: sth.ThermalState, air: st.State, speed: float) -> dict:
    """Array output [W] (after MPPT and harness), mean array irradiance, cell temperature, and the facet irradiance."""
    c, m = sh.c, sh.c.mission
    R = sg.body_to_enu(heading)
    n_enu = sh.normals @ R.T
    E = so.irradiance_on(n_enu, t, m.lat, m.lon, air.pressure, c.design.altitude)
    a_arr = sh.area * sh.array
    e_mean = float(np.sum(E * a_arr) / max(a_arr.sum(), 1e-9))
    pv = c.pv
    if e_mean > 0:
        Tc = sth.cell_temperature(e_mean, pv.efficiency, pv.absorptance, pv.emissivity, air, speed, c.ship.hull.length,
                                  thermal.T_envelope)
    else:
        Tc = thermal.T_envelope
    eta = max(pv.efficiency * (1 + pv.temp_coeff * (Tc - 298.15)), 0.0)
    p = eta * float(np.sum(E * a_arr)) * cat.MPPT_EFFICIENCY * cat.HARNESS_EFFICIENCY
    return {"power": p, "irradiance": e_mean, "T_cell": Tc, "eta": eta, "E": E, "n_enu": n_enu}


def day_cycle(c: sd.Configuration, wind_speed: float, days: int = 2, dt: float = 600.0, ship: Ship | None = None) -> dict:
    """``days`` of hovering on station in a steady wind of ``wind_speed`` [m/s] (airspeed = wind speed) on the 15th of
    the mission's month: solar power, loads, battery state, cell temperature, gas superheat and superpressure."""
    sh = ship or Ship(c)
    d, m = c.design, c.mission
    air = c.air
    heading = heading_into_wind(m, d.altitude)
    n = int(days * 86400 / dt)
    t0 = _t0(m)
    th = sth.equilibrium_guess(air)
    p_prop_need = float(sh.power_for(min(wind_speed, sh.v_max)))
    soc = sh.usable_wh                                  # starts full at midnight
    out = {k: np.zeros(n) for k in ("t", "solar", "load", "prop", "soc", "T_cell", "eta", "superheat", "dp", "T_env",
                                    "speed", "irradiance", "sun_el")}
    gas = lg.GasCell(c.ship.hull.volume, c.gas_mass, cat.GAS.gas, cat.GAS.purity)
    for i in range(n):
        t = t0 + timedelta(seconds=i * dt)
        sol = solar_step(sh, t, heading, th, air, wind_speed)
        el, _ = so.position(t, m.lat, m.lon)
        night = el < -so.horizon_dip(d.altitude)
        load = sh.loads + (sh.night_extra if night else 0.0)
        net = sol["power"] - load - p_prop_need
        prop = p_prop_need
        if net >= 0:
            soc = min(sh.usable_wh, soc + net * dt / 3600 * c.battery.charge_efficiency)
        else:
            draw = -net * dt / 3600 / c.battery.discharge_efficiency
            if soc - draw < sh.reserve_wh:               # out of energy: the propulsion gets what is left
                avail = max(0.0, (soc - sh.reserve_wh) * c.battery.discharge_efficiency * 3600 / dt)
                prop = max(0.0, sol["power"] - load + avail)
                soc = max(sh.reserve_wh, soc - draw)
            else:
                soc -= draw
        th = sth.envelope_step(sh.thermal, th, sol["E"], sol["power"], sol["n_enu"], air, wind_speed, dt)
        state = gas.state(air.pressure, air.temperature, th.T_gas - air.temperature)
        for k, v in (("t", i * dt), ("solar", sol["power"]), ("load", load), ("prop", prop), ("soc", soc),
                     ("T_cell", sol["T_cell"]), ("eta", sol["eta"]), ("superheat", th.T_gas - air.temperature),
                     ("dp", state["superpressure"]), ("T_env", th.T_envelope), ("speed", float(sh.speed_for(prop))),
                     ("irradiance", sol["irradiance"]), ("sun_el", el)):
            out[k][i] = v
    last = slice(n - int(86400 / dt), n)
    out.update({"dt": dt, "heading": heading, "wind_speed": wind_speed, "p_prop_need": p_prop_need,
                "energy_solar_wh": float(out["solar"][last].sum() * dt / 3600),
                "energy_load_wh": float((out["load"][last] + p_prop_need).sum() * dt / 3600),
                "superheat_max": float(out["superheat"][last].max()), "superheat_min": float(out["superheat"][last].min()),
                "soc_min": float(out["soc"].min()), "held": bool(np.all(out["prop"] >= p_prop_need - 1e-6))})
    return out


def holds_station(sh: Ship, solar: np.ndarray, night: np.ndarray, dt: float, winds) -> np.ndarray:
    """For each steady wind speed in ``winds``: can the ship fly at that airspeed all day, every day? (the speed is
    within its top speed, and the battery, starting full, never reaches the reserve over two repetitions of the
    day's ``solar`` profile and ends the second day no lower than the first). Vectorised over ``winds``."""
    w = np.asarray(winds, float)
    p = sh.power_for(np.minimum(w, sh.v_max))
    ok_speed = w <= sh.v_max
    load = sh.loads + np.where(night, sh.night_extra, 0.0)
    soc = np.full(w.shape, sh.usable_wh)
    low = np.full(w.shape, np.inf)
    ends = []
    c = sh.c.battery
    for rep in range(2):
        for i in range(len(solar)):
            net = solar[i] - load[i] - p
            soc = np.where(net >= 0, np.minimum(sh.usable_wh, soc + net * dt / 3600 * c.charge_efficiency),
                           soc + net * dt / 3600 / c.discharge_efficiency)
            low = np.minimum(low, soc)
        ends.append(soc.copy())
    return ok_speed & (low >= sh.reserve_wh) & (ends[1] >= ends[0] - 1e-6 * sh.usable_wh)


def availability(c: sd.Configuration, cycle: dict | None = None, n_days: int = 2000, seed: int = 0,
                 ship: Ship | None = None) -> dict:
    """Fraction of the month's days on which the ship holds station all day (daily winds from ``strato_wind``), the
    wind it can hold indefinitely (energy) and at all (speed)."""
    sh = ship or Ship(c)
    d, m = c.design, c.mission
    cycle = cycle or day_cycle(c, 0.0, ship=sh)
    n_day = int(86400 / cycle["dt"])
    solar = cycle["solar"][-n_day:]
    night = cycle["sun_el"][-n_day:] < -so.horizon_dip(d.altitude)
    winds = sw.daily_samples(d.altitude, m.month, m.lat, n_days, seed)
    speeds = np.hypot(winds[:, 0], winds[:, 1])
    ok = holds_station(sh, solar, night, cycle["dt"], speeds)
    grid = np.linspace(0.0, sh.v_max, 121)
    holds = holds_station(sh, solar, night, cycle["dt"], grid)
    v_energy = float(grid[holds].max()) if holds.any() else 0.0
    return {"availability": float(ok.mean()), "v_energy": v_energy, "v_max": sh.v_max,
            "wind_p50": float(np.percentile(speeds, 50)), "wind_p90": float(np.percentile(speeds, 90)),
            "wind_p95": float(np.percentile(speeds, 95)), "calm_closes": bool(holds[0])}


# ------------------------------------------------------------------------------------------------ float equilibrium
def lift(c: sd.Configuration, gas_mass: float, z: float, superheat: float) -> dict:
    """Static lift [kg] of the hull with ``gas_mass`` at altitude ``z`` with gas ``superheat`` (full or slack)."""
    a = st.state(z)
    cell = lg.GasCell(c.ship.hull.volume, gas_mass, cat.GAS.gas, cat.GAS.purity)
    return cell.state(a.pressure, a.temperature, superheat)


def equilibrium_altitude(c: sd.Configuration, gas_mass: float, mass: float, superheat: float,
                         lo: float = 12000.0, hi: float = 32000.0) -> float:
    """The altitude [m] where the static lift equals ``mass`` [kg] (the hull full there); nan when the ship cannot
    float anywhere in [lo, hi] (slack lift below the mass: it sinks), ``hi`` when it would rise above it. ``lo`` is
    the tropopause: below it the constant-superheat assumption of the callers no longer holds, and a ship that
    sinks out of the stratosphere has lost its station anyway."""
    f = lambda z: lift(c, gas_mass, z, superheat)["net_lift"] - mass   # noqa: E731
    if f(lo) < 0:
        return float("nan")
    if f(hi) > 0:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------------------------- endurance
def endurance(c: sd.Configuration, days: int = 30, dt: float = 900.0, seed: int = 0, gain: float = 2e-4,
              ballast_policy: bool = True, ship: Ship | None = None) -> dict:
    """``days`` on station: daily winds from the climatology plus turbulence (og.wind.Wind, σ 1.5 m/s, τ 30 min), the
    station-keeping rule, the battery, the thermal cycle, gas leakage (sealed) or venting (vented), the float altitude
    and the position. The ship carries its surplus lift as droppable ballast; with ``ballast_policy`` it drops ballast
    when it would sink more than 300 m below the design altitude."""
    sh = ship or Ship(c)
    d, m = c.design, c.mission
    daily = sw.daily_samples(d.altitude, m.month, m.lat, days + 1, seed)
    gust = ow.Wind(sigma=1.5, tau=1800.0, seed=seed + 1)
    gust.reset()
    n = int(days * 86400 / dt)
    t0 = _t0(m)
    gas_mass = c.gas_mass
    ballast = max(c.surplus_lift, 0.0)
    mass_fixed = c.system_mass                          # without ballast
    z = d.altitude
    th = sth.equilibrium_guess(st.state(z))
    pos = np.zeros(2)
    soc = sh.usable_wh
    vented_total = 0.0
    keys = ("t", "z", "x", "y", "dist", "soc", "solar", "prop", "airspeed", "wind", "superheat", "dp", "gas", "ballast")
    out = {k: np.zeros(n) for k in keys}
    lost_at = None
    for i in range(n):
        t = t0 + timedelta(seconds=i * dt)
        day = int(i * dt // 86400)
        w = daily[day] + gust.at(i * dt)[:2]
        gust.step(dt)
        air = st.state(z)
        heading = math.degrees(math.atan2(-w[0], -w[1])) % 360.0
        speed_guess = float(np.hypot(*w))
        sol = solar_step(sh, t, heading, th, air, speed_guess)
        el, _ = so.position(t, m.lat, m.lon)
        load = sh.loads + (sh.night_extra if el < -so.horizon_dip(z) else 0.0)
        # station-keeping: the air velocity that cancels the wind and pulls back to the centre
        want = -w - gain * pos                         # gain [1/s]: 2e-4 pulls back at 4 m/s from 20 km out
        v_want = float(np.linalg.norm(want))
        p_avail = sol["power"] - load + max(0.0, (soc - sh.reserve_wh) * c.battery.discharge_efficiency * 3600 / dt)
        p_cap = min(d.propulsion_power, max(p_avail, 0.0))
        v_cap = float(sh.speed_for(p_cap))
        v_air = min(v_want, v_cap, sh.v_max)
        air_vel = want / v_want * v_air if v_want > 1e-9 else np.zeros(2)
        prop = float(sh.power_for(v_air))
        net = sol["power"] - load - prop
        if net >= 0:
            soc = min(sh.usable_wh, soc + net * dt / 3600 * c.battery.charge_efficiency)
        else:
            soc = max(sh.reserve_wh, soc + net * dt / 3600 / c.battery.discharge_efficiency)
        pos = pos + (w + air_vel) * dt
        th = sth.envelope_step(replace(sh.thermal, gas_mass=gas_mass), th, sol["E"], sol["power"], sol["n_enu"], air, v_air, dt)
        dT = th.T_gas - air.temperature
        # gas: leakage (both systems), venting above the valve (vented)
        gas_mass *= (1 - cat.GAS.leak_per_day) ** (dt / 86400)
        if d.gas_system == "vented":
            cell = lg.GasCell(c.ship.hull.volume, gas_mass, cat.GAS.gas, cat.GAS.purity, vented=True, valve_dp=d.valve_dp)
            vented_total += cell.vent(air.pressure, air.temperature, dT)
            gas_mass = cell.gas_mass
        z_eq = equilibrium_altitude(c, gas_mass, mass_fixed + ballast, dT)
        if ballast_policy and (math.isnan(z_eq) or z_eq < d.altitude - 300.0) and ballast > 0:
            # drop just enough ballast to float at the design altitude (or all of it)
            need = mass_fixed + ballast - lift(c, gas_mass, d.altitude, dT)["net_lift"]
            drop = min(ballast, max(need, 0.0))
            ballast -= drop
            z_eq = equilibrium_altitude(c, gas_mass, mass_fixed + ballast, dT)
        if math.isnan(z_eq):                           # it cannot float anywhere: it is coming down; stop here
            lost_at = i * dt
            out = {k: v[:i] for k, v in out.items()}
            break
        z = z_eq
        st_cell = lift(c, gas_mass, z, dT)
        for k, v in (("t", i * dt), ("z", z), ("x", pos[0]), ("y", pos[1]), ("dist", float(np.linalg.norm(pos))),
                     ("soc", soc), ("solar", sol["power"]), ("prop", prop), ("airspeed", v_air), ("wind", float(np.hypot(*w))),
                     ("superheat", dT), ("dp", st_cell["superpressure"]), ("gas", gas_mass), ("ballast", ballast)):
            out[k][i] = v
    if len(out["t"]) == 0:
        out = {k: np.zeros(1) for k in out}
    on_station = out["dist"] <= m.station_radius
    out.update({"dt": dt, "days": days, "on_station": float(on_station.mean()), "lost_at_s": lost_at,
                "vented_kg": vented_total, "gas_lost_kg": c.gas_mass - gas_mass, "max_dist_km": float(out["dist"].max() / 1000),
                "min_altitude": float(out["z"].min()), "ballast_left": ballast})
    return out


def gas_life_days(c: sd.Configuration, max_days: float = 3650.0) -> float:
    """Days until leakage (``cat.GAS.leak_per_day``) leaves too little gas to float at night at the design altitude
    with all the surplus lift dropped as ballast (sealed; the night is when the hull is slack first)."""
    lo, hi = 0.0, max_days
    rate = cat.GAS.leak_per_day

    def floats(days):
        g = c.gas_mass * (1 - rate) ** days
        return lift(c, g, c.design.altitude, c.superheat_night)["net_lift"] >= c.system_mass - 1e-9

    if not floats(0.0):
        return 0.0
    if floats(hi):
        return hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if floats(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------------------------------------------------ ascent
ASCENT_CD = 0.5                   # estimate: cross-flow drag coefficient of the hull moving broadside (on planform)


def ascent(c: sd.Configuration, free_lift: float = 0.05, dt: float = 2.0, t_max: float = 6 * 3600.0,
           tau_gas: float = 600.0, superheat: float = 0.0) -> dict:
    """Vertical ascent from the ground (z = 0) with the propellers off: the hull partly filled (the ballonet holds
    air), ``free_lift`` × the float net lift added as launch ballast removed at release (i.e. the ship is that much
    light at launch and drops nothing later). The gas cools adiabatically as it rises and relaxes towards air
    temperature + ``superheat`` with time constant ``tau_gas``; the hull's added mass (Lamb's k2, broadside) and
    cross-flow drag slow it. Sealed: the gas stays and the superpressure builds at the top. Vented: the valve releases
    gas above ``valve_dp``. The horizontal drift through the wind profile of the mission's month is integrated."""
    d, m, h = c.design, c.mission, c.ship.hull
    k1, k2, _ = sa.lamb_k(h.fineness)
    mass = c.system_mass + max(c.surplus_lift, 0.0) - free_lift * max(c.net_lift, 0.0)
    gas_mass = c.gas_mass
    plan = float(np.trapezoid(2 * h.r(np.linspace(0, h.length, 400)), np.linspace(0, h.length, 400)))
    z, w, x = 0.0, 0.0, 0.0
    a0 = st.state(0.0)
    Tg = a0.temperature + superheat
    n = int(t_max / dt)
    keys = ("t", "z", "w", "dp", "fill", "Tg_minus_Ta", "drift", "gas")
    out = {k: np.zeros(n) for k in keys}
    p_prev = a0.pressure
    vented = 0.0
    k_end = n
    for i in range(n):
        a = st.state(max(z, 0.0))
        Tg += lg.adiabatic_rate(Tg, a.pressure, (a.pressure - p_prev) / dt) * dt + (a.temperature + superheat - Tg) * dt / tau_gas
        p_prev = a.pressure
        cell = lg.GasCell(h.volume, gas_mass, cat.GAS.gas, cat.GAS.purity, vented=d.gas_system == "vented", valve_dp=d.valve_dp)
        vented += cell.vent(a.pressure, a.temperature, Tg - a.temperature)
        gas_mass = cell.gas_mass
        s = cell.state(a.pressure, a.temperature, Tg - a.temperature)
        buoy = (s["gross_lift"] - gas_mass - mass) * a.gravity
        drag = 0.5 * a.density * w * abs(w) * ASCENT_CD * plan
        inertia = mass + gas_mass + k2 * a.density * s["gas_volume"]
        w += (buoy - drag) / inertia * dt
        z = max(z + w * dt, 0.0)
        wind = sw.profile(z, m.month, m.lat)
        x += float(wind["speed"]) * dt
        for k, v in (("t", i * dt), ("z", z), ("w", w), ("dp", s["superpressure"]), ("fill", s["fill"]),
                     ("Tg_minus_Ta", Tg - a.temperature), ("drift", x), ("gas", gas_mass)):
            out[k][i] = v
        if i * dt > 1800 and abs(w) < 0.02 and abs(z - out["z"][max(i - int(600 / dt), 0)]) < 5.0:
            k_end = i + 1
            break
    out = {k: v[:k_end] for k, v in out.items()}
    reach = np.argmax(out["z"] >= 0.99 * d.altitude) if np.any(out["z"] >= 0.99 * d.altitude) else None
    out.update({"float_altitude": float(out["z"][-1]), "max_rate": float(out["w"].max()),
                "time_to_99pct_s": float(out["t"][reach]) if reach is not None else float("nan"),
                "vented_kg": vented, "dp_top": float(out["dp"][-1]), "drift_km": float(out["drift"][-1] / 1000)})
    return out

