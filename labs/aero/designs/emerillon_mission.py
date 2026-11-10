"""ÉMERILLON — the flights: a point-mass simulation of ER1 and ER2 through five scenarios (notebook 37, part 5).

A reduced flight model, honest about what it is: the aircraft is a point mass with a body axis. The four propellers
push along the axis (``emerillon_flight.drive`` tabulated over altitude, axial airspeed and throttle: ``envelope``);
the air pushes back with the axial drag of the build-up and a normal force on the body and the arms when the axis is
not along the airflow (so it flies on its body's lift at speed, and stops by turning broadside). The axis slews to
where the wanted force points, at a limited rate. There is no attitude dynamics, no controller tuning, no propeller
at an angle of attack, no vortex-ring state: what the scenarios give is **times, speeds, energy and margins**, not
handling. The pack is counted in watt-hours (its sag is in the drive, its state of charge is not), the windings heat
with their copper loss and cool in the propeller's wash.

- ``envelope(variant)`` — thrust, electrical power and copper loss over (altitude, axial airspeed, throttle).
- ``Sim`` — the integrator: guidance gives a wanted velocity, the wanted force follows, the axis and the throttle obey
  within their limits; wind, terrain and obstacles come from the scenario.
- ``SCENARIOS`` — ``wildfire`` (out to a fire and back, at the fastest speed the pack allows), ``mountain`` (catch a
  target circling a mountain, come back), ``sea`` (catch a boat in a crosswind with gusts, low over the water),
  ``climb`` (straight up to the variant's ceiling target — 4 km for ER1, 7 km for ER2 — and back down),
  ``palace`` (a fast climb over Warsaw, a swoop, and a slow landing on the roof of the Palace of Culture and Science).
- ``fly(variant, scenario)`` → a ``Flight`` (history, events, metrics, verdict); ``ceiling(variant)`` — the altitude
  question without the simulator: thrust and energy against height.
- ``plot(flight)`` — one figure per flight.

The time step is 0.02 s; the chases converge with it (the catch times move by about 1 % between 0.01 and 0.04 s) and
break above 0.04 s at ER2's speed.

World frame: x east, y north, z up [m], heights above the scenario's ground; ``alt0`` is the ground's altitude above
sea level (the air's density). Units SI.
"""
from __future__ import annotations

import bisect
import functools
import math
from dataclasses import dataclass, field

import numpy as np
from ogma import wind as og_wind

import emerillon_flight as ef
import emerillon_market as mk

G = ef.G
H_GRID = (0.0, 1500.0, 3000.0, 4500.0, 6000.0, 7500.0, 9000.0)
U_GRID = (0.15, 0.35, 0.55, 0.75, 0.90, 1.0)
N_V = 13
SLEW = 6.0                         # how fast the body axis turns [rad/s] (assumed: a racing quad turns faster)
ACC_MAX = 5.0 * G                  # the largest manoeuvre the guidance asks for (assumed: the arms' FEA load case)
TAU_V = 0.8                        # the guidance closes a velocity error in about this time [s]
REVERSE_THRUST = 0.5               # share of the static thrust counted when the propellers meet the air backwards (assumed)
BACKWARD_DRAG = 1.5                # the tail-first axial drag over the nose-first one (assumed)
COOL_0, COOL_V = 0.35, 0.045       # a motor's cooling [W/K] at rest and per m/s of axial flow (assumed)
T_MOTOR_MAX = 130.0                # winding limit [C]
RESERVE = 0.20                     # share of the pack a flight must land with
LANDING_SINK = 0.8                 # the last metres of a landing [m/s]


# =========================================================================================================== envelope
@functools.lru_cache(maxsize=None)
def envelope(variant: str) -> dict:
    """The drive of a variant over (altitude, axial airspeed, throttle): thrust [N], electrical power [W] and one
    motor's copper loss [W] as nested lists, with the grids, the mass, the drag area and the planform numbers the
    simulator needs. A few seconds; cached."""
    c = ef.VARIANTS[variant] if isinstance(variant, str) else variant          # a named variant, or any Config
    m = ef.mass_budget(c).mass
    v_top = max(ef.top_speed(c, m, h)["V"] for h in (0.0, 4500.0, 9000.0))
    vs = tuple(float(v) for v in np.linspace(0.0, 1.25 * v_top, N_V))
    T, P, Q = [], [], []
    for h in H_GRID:
        air = ef.air_at(h)
        Th, Ph, Qh = [], [], []
        for v in vs:
            row = [ef.drive(c, max(v, 0.5), air=air, throttle=u) for u in U_GRID]
            Th.append([max(float(d["thrust"]), 0.0) for d in row])
            Ph.append([float(d["electric_w"]) for d in row])
            Qh.append([float(d["copper_w"]) for d in row])
        T.append(Th), P.append(Ph), Q.append(Qh)
    g = ef.geometry(c)
    bat, motor = mk.part(c.battery), mk.part(c.motor)
    return {"variant": variant, "config": c, "mass": m, "v_top": float(ef.top_speed(c, m)["V"]), "V": vs, "T": T, "P": P, "Q": Q,
            "cda": [ef.drag_buildup(c, max(v, 5.0))["CdA"] for v in vs],
            "frontal": g["frontal"], "plan": g["d"] * (0.67 * g["nose"] + g["cyl"] + 0.55 * g["tail"]), "arms": 2 * g["arm_c"] * g["arm_l"],
            "wh": bat.cells * 3.7 * bat.ah * c.packs, "heat_capacity": 0.55 * motor.mass_g / 1e3 * 450.0}


def _cell(grid, x):
    x = min(max(x, grid[0]), grid[-1])
    i = min(max(bisect.bisect_right(grid, x) - 1, 0), len(grid) - 2)
    return i, (x - grid[i]) / (grid[i + 1] - grid[i])


def _tri(tab, ih, fh, iv, fv, iu, fu):
    out = 0.0
    for a, wa in ((ih, 1 - fh), (ih + 1, fh)):
        for b, wb in ((iv, 1 - fv), (iv + 1, fv)):
            out += wa * wb * ((1 - fu) * tab[a][b][iu] + fu * tab[a][b][iu + 1])
    return out


def thrust_at(env: dict, h: float, v_ax: float, u: float) -> tuple:
    """(thrust, electrical power, copper loss per motor) at altitude ``h``, axial airspeed ``v_ax`` ≥ 0, throttle ``u``."""
    ih, fh = _cell(H_GRID, h)
    iv, fv = _cell(env["V"], v_ax)
    u = min(max(u, 0.0), 1.0)
    if u < U_GRID[0]:                                               # below the table: straight to nothing
        k = u / U_GRID[0]
        return tuple(k * _tri(env[t], ih, fh, iv, fv, 0, 0.0) for t in ("T", "P", "Q"))
    iu, fu = _cell(U_GRID, u)
    return tuple(_tri(env[t], ih, fh, iv, fv, iu, fu) for t in ("T", "P", "Q"))


def throttle_for(env: dict, h: float, v_ax: float, thrust: float) -> float:
    """The throttle that gives ``thrust`` (1 when full throttle gives less)."""
    if thrust <= 0:
        return 0.0
    if thrust_at(env, h, v_ax, 1.0)[0] <= thrust:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(14):
        u = 0.5 * (lo + hi)
        lo, hi = (u, hi) if thrust_at(env, h, v_ax, u)[0] < thrust else (lo, u)
    return hi


# ============================================================================================================= ceiling
CLIMB_THROTTLES = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0)


def _steady_climb(env: dict, h: float, u: float) -> tuple:
    """(vertical speed, electrical power) of a steady vertical climb at altitude ``h`` and throttle ``u``."""
    W, rho = env["mass"] * G, ef.air_at(h)["rho"]
    if thrust_at(env, h, 0.0, u)[0] <= W:
        return 0.0, thrust_at(env, h, 0.0, u)[1]
    lo, hi = 0.0, env["V"][-1]
    for _ in range(26):
        v = 0.5 * (lo + hi)
        lo, hi = (v, hi) if thrust_at(env, h, v, u)[0] > W + 0.5 * rho * v * v * float(np.interp(v, env["V"], env["cda"])) else (lo, v)
    return lo, thrust_at(env, h, lo, u)[1]


def ceiling(variant, target_m: float | None = None, step: float = 250.0, mode: str = "economy") -> dict:
    """The altitude question without the simulator: a vertical climb from sea level in slices, each at its steady
    speed. ``mode='fast'`` holds full throttle; ``'economy'`` takes, per slice, the throttle with the fewest
    watt-hours per metre. Returns the profile, the time and energy to ``target_m`` (default: the variant's target),
    the landing's share, whether the pack covers both with the reserve, the height the pack reaches at all, and the
    static thrust over weight up there. ``variant`` is a name or any ``Config`` (a what-if with another pack)."""
    env = envelope(variant)
    m, W = env["mass"], env["mass"] * G
    target = ef.TARGET[variant]["ceiling_m"] if target_m is None else target_m
    hover_w = thrust_at(env, 0.0, 0.0, throttle_for(env, 0.0, 0.0, W))[1]
    land_wh = hover_w * ef.HOVER_S / 3600.0
    budget = (1.0 - RESERVE) * env["wh"] - land_wh
    rows, t, wh, reached, h = [], 0.0, 0.0, None, 0.0
    while h <= H_GRID[-1] + 3000.0:
        opts = [(u, *_steady_climb(env, h, u)) for u in (CLIMB_THROTTLES if mode == "economy" else (1.0,))]
        opts = [o for o in opts if o[1] > 3.0]
        if not opts:
            break
        u, v, P = min(opts, key=lambda o: o[2] / o[1])
        rows.append({"h": h, "static_tw": thrust_at(env, h, 0.0, 1.0)[0] / W, "throttle": u, "climb_ms": v, "power_w": P, "t": t, "wh": wh,
                     "wh_per_km": P / v / 3.6})
        if reached is None and wh > budget:
            reached = rows[-2]["h"] if len(rows) > 1 else 0.0
        t += step / v
        wh += P * step / v / 3600.0
        h += step
    at = min(rows, key=lambda r: abs(r["h"] - target))
    return {"variant": variant if isinstance(variant, str) else variant.label(), "mode": mode, "target_m": target, "profile": rows, "time_s": at["t"],
            "climb_wh": at["wh"], "land_wh": land_wh, "pack_wh": env["wh"], "budget_wh": budget, "mass_g": m * 1e3,
            "static_tw_at_target": at["static_tw"], "climb_ms_at_target": at["climb_ms"],
            "reaches_target": bool(at["wh"] <= budget and at["static_tw"] > 1.2 and at["h"] >= target - step),
            "energy_ceiling_m": reached if reached is not None else rows[-1]["h"],
            "potential_wh": m * G * target / 3600.0, "T_C_at_target": ef.air_at(target)["T_C"]}


def ceiling_options(variant: str) -> list[dict]:
    """The variant's ceiling target against the packs that could carry it there: the variant as built, then with each
    other pack of the catalogue, one or two in parallel. Rows of pack, mass, price, the economy climb's energy, the
    height the pack reaches, and whether the target is met."""
    import dataclasses
    base = ef.VARIANTS[variant]
    rows = []
    for bat in mk.BATTERIES:
        for n_par in (1, 2):
            c = dataclasses.replace(base, battery=bat.key, n_parallel=n_par)
            r = ceiling(c, ef.TARGET[variant]["ceiling_m"])
            rows.append({"pack": f"{n_par} x {bat.name}", "as built": c == base, "mass [g]": round(r["mass_g"]), "price [USD]": mk.price(c),
                         "top speed [km/h]": round(ef.top_speed(c)["kmh"]), "climb [Wh]": round(r["climb_wh"], 1), "landing [Wh]": round(r["land_wh"], 1),
                         "pack [Wh]": round(r["pack_wh"], 1), "reaches [m]": r["energy_ceiling_m"], "target met": r["reaches_target"],
                         "time to target [s]": round(r["time_s"])})
    return rows


# ============================================================================================================== the sim
@dataclass
class Leg:
    """One piece of a flight. ``kind``: ``goto`` (fly to ``to`` at ``speed``, arriving at ``speed_end``), ``climb``
    (straight up to ``to[2]``, at full throttle or at ``speed``), ``fall`` (a flat fall, motors stopped, down to ``to[2]``), ``chase`` (close on ``target(t)`` → position and velocity, until within
    ``radius``), ``hold`` (stay on ``target(t)`` for ``seconds``), ``land`` (down onto ``to``, the last metres slowly)."""
    kind: str
    to: tuple = (0.0, 0.0, 0.0)
    speed: float = 30.0
    speed_end: float | None = None
    radius: float = 15.0
    target: object = None
    seconds: float = 0.0
    name: str = ""
    offset: tuple = (0.0, 0.0, 0.0)
    floor: float | None = None                 # terrain clearance this leg keeps (None: the scenario's)
    full: bool = True                          # a climb at full throttle (False: at ``speed``)


@dataclass
class Scenario:
    key: str
    title: str
    legs: object                              # callable (variant, env, pace) -> list[Leg]
    start: tuple = (0.0, 0.0, 0.0)
    alt0: float = 0.0
    wind: object = None                       # callable (pos, t) -> wind vector [m/s], or None
    terrain: object = None                    # callable (x, y) -> ground height [m]
    obstacles: tuple = ()                     # vertical cylinders (x, y, radius, z_bottom, z_top)
    paced: bool = False                       # True: ``fly`` looks for the fastest pace the pack allows
    t_max: float = 900.0
    features: dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class Flight:
    variant: str
    scenario: str
    title: str
    hist: dict
    events: list
    metrics: dict
    ok: bool
    why: list
    pace: float
    features: dict


def _unit(v):
    n = float(np.linalg.norm(v))
    return (v / n, n) if n > 1e-9 else (np.zeros(3), 0.0)


class Sim:
    def __init__(self, variant: str, scn: Scenario, pace: float = 1.0, dt: float = 0.02, seed: int = 1, config=None):
        self.env = envelope(config or variant)                       # ``config``: the variant with other parts (a bigger pack)
        self.variant, self.scn, self.pace, self.dt = variant, scn, pace, dt
        self.m = self.env["mass"]
        self.p = np.array(scn.start, float)
        self.v = np.zeros(3)
        self.b = np.array([0.0, 0.0, 1.0])                          # it stands on its tail
        self.t = self.wh = 0.0
        self.temp = ef.air_at(scn.alt0)["T_C"] + 25.0                # the motors start warm from the hover check
        self.rng = np.random.default_rng(seed)
        self.gust = np.zeros(3)
        self.events, self.rows = [], []
        self.min_agl, self.min_clear, self.max_g, self.max_v = 1e9, 1e9, 0.0, 0.0

    # ------------------------------------------------------------------------------------------------ the world
    def ground(self, x, y) -> float:
        return float(self.scn.terrain(x, y)) if self.scn.terrain else 0.0

    def wind(self) -> np.ndarray:
        return np.asarray(self.scn.wind(self, self.p, self.t), float) if self.scn.wind else np.zeros(3)

    # ---------------------------------------------------------------------------------------------- one time step
    def step(self, v_want: np.ndarray, full_throttle: bool = False, flat: bool = False) -> None:
        """One time step toward the wanted velocity. ``flat``: a flat fall — the axis held level, the motors stopped."""
        env, dt, m = self.env, self.dt, self.m
        h = self.scn.alt0 + self.p[2]
        rho = ef.air_at(h)["rho"]
        va = self.v - self.wind()
        vhat, V = _unit(va)
        q = 0.5 * rho * V * V
        # the air on the body: axial drag along the axis, a normal force against the crossflow
        cos_a = float(np.dot(self.b, vhat)) if V > 1e-6 else 1.0
        sin_a = math.sqrt(max(1.0 - cos_a * cos_a, 0.0))
        cda = float(np.interp(V, env["V"], env["cda"]))
        axial = -q * cda * cos_a * abs(cos_a) * (1.0 if cos_a >= 0 else BACKWARD_DRAG)
        cna = (2 * env["frontal"] + 2.5 * env["arms"]) * sin_a * abs(cos_a) + 1.2 * env["plan"] * sin_a * sin_a
        nhat, _ = _unit(-(vhat - cos_a * self.b))
        aero = axial * self.b + q * cna * nhat
        # the wanted force, the axis toward it, the thrust along the axis
        a_want = (v_want - self.v) / TAU_V
        an = float(np.linalg.norm(a_want))
        if an > ACC_MAX:
            a_want *= ACC_MAX / an
        need = m * a_want + np.array([0.0, 0.0, m * G]) - aero
        want_dir, need_n = _unit(need)
        if flat:
            want_dir, need_n = _unit(np.array([self.b[0], self.b[1], 0.0]) if abs(self.b[2]) < 0.999 else np.array([1.0, 0.0, 0.0]))
        if need_n > 1e-6:
            ang = math.acos(min(max(float(np.dot(self.b, want_dir)), -1.0), 1.0))
            if ang > 1e-6:
                k = min(1.0, SLEW * dt / ang)
                self.b, _ = _unit(self.b + k * (want_dir - self.b) / max(1.0 - 0.5 * k * (1 - math.cos(ang)), 1e-3))
        v_ax = float(np.dot(va, self.b))
        T_max = thrust_at(env, h, max(v_ax, 0.0), 1.0)[0] * (1.0 if v_ax >= -2.0 else REVERSE_THRUST)
        T = T_max if full_throttle else min(max(float(np.dot(need, self.b)), 0.0), T_max)
        if flat:
            T = 0.0                                                  # motors stopped: no sideways push in a fall of minutes
        u = throttle_for(env, h, max(v_ax, 0.0), T / (1.0 if v_ax >= -2.0 else REVERSE_THRUST))
        _, P, Qc = thrust_at(env, h, max(v_ax, 0.0), u)
        acc = (T * self.b + aero) / m - np.array([0.0, 0.0, G])
        self.v = self.v + acc * dt
        self.p = self.p + self.v * dt
        gz = self.ground(self.p[0], self.p[1])
        if self.p[2] < gz:                                           # on the ground: it stands
            self.p[2] = gz
            self.v[:] = 0.0 if self.v[2] <= 0 else self.v
        self.t += dt
        self.wh += P * dt / 3600.0
        t_air = ef.air_at(h)["T_C"]
        self.temp += (Qc - (COOL_0 + COOL_V * max(v_ax, 0.0)) * (self.temp - t_air)) / env["heat_capacity"] * dt
        agl = self.p[2] - gz
        g_load = float(np.linalg.norm(acc + np.array([0.0, 0.0, G]))) / G
        self.max_g, self.max_v = max(self.max_g, g_load), max(self.max_v, float(np.linalg.norm(self.v)))
        if self.t > 3.0 and float(np.linalg.norm(self.v)) > 8.0:
            self.min_agl = min(self.min_agl, agl)
        for (ox, oy, r, z0, z1) in self.scn.obstacles:
            if z0 - 2.0 <= self.p[2] <= z1 + 2.0:
                self.min_clear = min(self.min_clear, math.hypot(self.p[0] - ox, self.p[1] - oy) - r)
        if len(self.rows) == 0 or self.t - self.rows[-1][0] >= 0.1 - 1e-9:
            self.rows.append((self.t, *self.p, float(np.linalg.norm(self.v)), self.v[2], V, self.wh, self.temp, u, g_load, agl,
                              math.degrees(math.acos(min(max(cos_a, -1.0), 1.0))) if V > 3 else 0.0, P))

    # ------------------------------------------------------------------------------------------------- guidance
    def _avoid(self, v_want: np.ndarray, floor: float) -> np.ndarray:
        """Keep ``floor`` metres over the terrain ahead and steer round the obstacles."""
        if self.scn.terrain:
            ahead = self.p + self.v * 2.5
            need_z = max(self.ground(self.p[0], self.p[1]), self.ground(ahead[0], ahead[1])) + floor
            z_soon = self.p[2] + min(self.v[2], 0.0) * 2.0           # where a descent puts it in two seconds
            if z_soon < need_z:
                v_want = v_want.copy()
                v_want[2] = max(v_want[2], min(2.0 * (need_z - z_soon), 0.8 * max(float(np.linalg.norm(v_want)), 15.0)))
        for (ox, oy, r, z0, z1) in self.scn.obstacles:
            d = np.array([self.p[0] - ox, self.p[1] - oy, 0.0])
            dist = float(np.linalg.norm(d))
            keep = r + 6.0
            if z0 - 5.0 <= self.p[2] <= z1 + 5.0 and dist < keep + 10.0:
                v_want = v_want + (d / max(dist, 1e-3)) * 0.8 * (keep + 10.0 - dist)
        return v_want

    def _toward(self, point, speed, speed_end, brake=7.0):
        d, dist = _unit(np.asarray(point, float) - self.p)
        v_cmd = min(speed, math.sqrt(max((speed_end or 0.0) ** 2 + 2 * brake * dist, 0.0)) if speed_end is not None else speed)
        return d * v_cmd, dist

    def run(self) -> None:
        legs = self.scn.legs(self.variant, self.env, self.pace)
        floor = self.scn.features.get("floor", 40.0)
        for leg in legs:
            t_leg, held, done = self.t, 0.0, False
            self.events.append({"t": round(self.t, 2), "leg": leg.name or leg.kind, "wh": round(self.wh, 2), "pos": [round(float(x), 1) for x in self.p]})
            while not done and self.t < self.scn.t_max:
                if leg.kind == "climb":
                    coast = self.v[2] ** 2 / (2 * 1.15 * G)              # it coasts up this much once the throttle is cut
                    if self.p[2] < leg.to[2] - coast:
                        self.step(np.array([0.0, 0.0, 400.0 if leg.full else leg.speed]), full_throttle=leg.full)
                    else:
                        self.step(np.array([0.0, 0.0, 0.0]), flat=True)
                    done = (self.p[2] >= leg.to[2] - 5.0 and self.v[2] < 3.0) or (self.t - t_leg > 20.0 and self.v[2] < 0.5)
                elif leg.kind == "fall":
                    self.step(np.array([0.0, 0.0, -100.0]), flat=True)
                    done = self.p[2] <= leg.to[2]
                elif leg.kind == "goto":
                    v_want, dist = self._toward(leg.to, leg.speed, leg.speed_end)
                    self.step(self._avoid(v_want, floor if leg.floor is None else leg.floor))
                    done = dist < leg.radius
                elif leg.kind in ("chase", "hold"):
                    tp, tv = leg.target(self.t)
                    tp = np.asarray(tp, float) + np.asarray(leg.offset, float)
                    rel = tp - self.p
                    dist = float(np.linalg.norm(rel))
                    closing = max(leg.speed - float(np.linalg.norm(tv)), 5.0)
                    lead = tp + np.asarray(tv, float) * min(dist / closing, 12.0)        # where it will be when we get there
                    d, _ = _unit(lead - self.p)
                    v_close = min(leg.speed, float(np.linalg.norm(tv)) + math.sqrt(2 * 6.0 * dist))
                    v_want = d * v_close if dist > 3.0 else np.asarray(tv, float) + rel / 1.0
                    self.step(self._avoid(v_want, floor if leg.kind == "chase" else 3.0))
                    if leg.kind == "chase":
                        done = dist < leg.radius
                    else:
                        held = held + self.dt if dist < leg.radius else 0.0
                        done = held >= leg.seconds
                else:                                                # land
                    over = np.array([leg.to[0], leg.to[1], self.p[2]])
                    err = over - self.p
                    height = self.p[2] - leg.to[2]
                    off = float(np.linalg.norm(err))
                    sink = min(leg.speed, LANDING_SINK + 0.45 * max(height - 2.0, 0.0))        # slower and slower, the last 2 m at the touchdown speed
                    if off > 2.0 and height < 8.0:
                        sink = 0.0                                   # not over the spot: do not come down yet
                    v_want = np.array([*(err[:2] / 1.2), -sink])
                    hn = float(np.linalg.norm(v_want[:2]))
                    if hn > 12.0:
                        v_want[:2] *= 12.0 / hn
                    v_before = self.v.copy()
                    self.step(v_want)
                    if height <= 0.02 or self.p[2] <= leg.to[2] + 1e-6:
                        self.events.append({"t": round(self.t, 2), "leg": "touchdown", "wh": round(self.wh, 2), "sink": round(max(-float(v_before[2]), 0.0), 2),
                                            "drift": round(float(np.linalg.norm(v_before[:2])), 2), "miss": round(off, 2),
                                            "pos": [round(float(x), 1) for x in self.p]})
                        self.v[:] = 0.0
                        done = True
            if not done:
                self.events.append({"t": round(self.t, 2), "leg": "TIME OUT in " + (leg.name or leg.kind), "wh": round(self.wh, 2),
                                    "pos": [round(float(x), 1) for x in self.p]})
                break


COLS = ("t", "x", "y", "z", "speed", "vz", "airspeed", "wh", "motor_C", "throttle", "g", "agl", "alpha_deg", "power_w")


def _flight(sim: Sim) -> Flight:
    env, scn = sim.env, sim.scn
    hist = {k: np.array([r[i] for r in sim.rows]) for i, k in enumerate(COLS)}
    touch = [e for e in sim.events if e["leg"] == "touchdown"]
    used = sim.wh / env["wh"]
    m = {"time_s": sim.t, "energy_wh": sim.wh, "pack_wh": env["wh"], "pack_used": used, "reserve": 1.0 - used, "max_kmh": sim.max_v * 3.6,
         "max_altitude_m": float(hist["z"].max()) + scn.alt0, "max_g": sim.max_g, "max_motor_C": float(hist["motor_C"].max()),
         "min_agl_m": None if sim.min_agl > 1e8 else sim.min_agl, "min_obstacle_clearance_m": None if sim.min_clear > 1e8 else sim.min_clear,
         "distance_km": float(np.sum(np.hypot(np.hypot(np.diff(hist["x"]), np.diff(hist["y"])), np.diff(hist["z"])))) / 1000.0}
    if touch:
        m.update(touchdown_sink_ms=touch[-1]["sink"], touchdown_drift_ms=touch[-1]["drift"], touchdown_miss_m=touch[-1]["miss"])
    for e in sim.events:
        if e["leg"] in ("caught", "at the fire", "top of climb", "over the roof"):
            m[f"t_{e['leg'].replace(' ', '_')}_s"] = e["t"]
    why = []
    if any(e["leg"].startswith("TIME OUT") for e in sim.events):
        why.append(next(e["leg"] for e in sim.events if e["leg"].startswith("TIME OUT")))
    if not touch:
        why.append("did not land")
    elif touch[-1]["sink"] > 1.5 or touch[-1]["drift"] > 1.0 or touch[-1]["miss"] > 2.5:
        why.append("hard or wide landing")
    if used > 1.0 - RESERVE:
        why.append(f"lands with {max(1 - used, 0) * 100:.0f} % in the pack (reserve {RESERVE:.0%})" if used < 1 else "the pack is empty before the end")
    if m["max_motor_C"] > T_MOTOR_MAX:
        why.append(f"windings at {m['max_motor_C']:.0f} C")
    if m["min_obstacle_clearance_m"] is not None and m["min_obstacle_clearance_m"] < 1.0:
        why.append("too close to an obstacle")
    if scn.terrain and m["min_agl_m"] is not None and m["min_agl_m"] < 5.0:
        why.append("too close to the terrain")
    return Flight(sim.variant, scn.key, scn.title, hist, sim.events, m, not why, why, sim.pace, scn.features)


def fly(variant: str, scenario: str, pace: float | None = None, dt: float = 0.02, config=None) -> Flight:
    """Fly a scenario. A paced scenario (the wildfire) is flown at the fastest pace (share of the top speed) that
    still lands with the reserve; ``pace`` fixes it instead. ``config``: the variant with other parts (``ef.HIGH``)."""
    scn = SCENARIOS[scenario]

    def one(p):
        s = Sim(variant, scn, p, dt, config=config)
        s.run()
        return _flight(s)

    if pace is not None or not scn.paced:
        return one(1.0 if pace is None else pace)
    best = None                                                      # energy per kilometre is not monotonic in speed: scan
    for p in PACES:
        s = Sim(variant, scn, p, max(dt, 0.05), config=config)
        s.run()
        f = _flight(s)
        if f.ok:
            best = p
            break
    return one(best if best is not None else PACES[len(PACES) // 2])


PACES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.42)                         # shares of the top speed, fastest first


# ========================================================================================================== scenarios
def _mark(name):
    return Leg("goto", name=name, radius=1e9)                        # a leg that ends at once: an event in the log


def _event_leg(sim_name):
    return _mark(sim_name)


# ---------------------------------------------------------------------------------------------------------- wildfire
FIRE = (2000.0, 300.0, 0.0)


def _wildfire(variant, env, pace):
    v = pace * env["v_top"]
    fx, fy, _ = FIRE
    d, _ = _unit(np.array([fx, fy, 0.0]))
    left = np.array([-d[1], d[0], 0.0])
    r, alt = 180.0, 90.0
    round_fire = [np.array([fx, fy, alt]) + r * (math.cos(a) * d + math.sin(a) * left) for a in np.radians([-150, -90, -30, 30, 90, 150])]
    return ([Leg("climb", to=(0, 0, alt), name="take-off"), Leg("goto", to=tuple(round_fire[0]), speed=v, speed_end=0.55 * v, radius=60, name="dash out"),
             _mark("at the fire")] +
            [Leg("goto", to=tuple(p), speed=0.55 * v, radius=60, name="round the smoke") for p in round_fire[1:]] +
            [Leg("goto", to=(0, 0, 40.0), speed=v, speed_end=4.0, radius=8, name="dash home"), Leg("land", to=(0, 0, 0), speed=6.0, name="land")])


# ---------------------------------------------------------------------------------------------------------- mountain
PEAK = {"xy": (1600.0, 0.0), "height": 900.0, "radius": 1300.0}
HARE_R, HARE_Z, HARE_V = 1000.0, 520.0, 38.0                         # a target circling the mountain at mid height


def _mountain_terrain(x, y):
    r = math.hypot(x - PEAK["xy"][0], y - PEAK["xy"][1])
    return PEAK["height"] * max(0.0, 1.0 - r / PEAK["radius"]) ** 1.3


def _hare(t):
    w = HARE_V / HARE_R
    a = math.pi + w * t                                              # it starts on the near side and goes round
    cx, cy = PEAK["xy"]
    return ((cx + HARE_R * math.cos(a), cy + HARE_R * math.sin(a), HARE_Z), (-HARE_V * math.sin(a), HARE_V * math.cos(a), 0.0))


def _mountain(variant, env, pace):
    v = pace * env["v_top"]
    return [Leg("climb", to=(0, 0, 120.0), name="take-off"), Leg("chase", target=_hare, speed=v, radius=12.0, name="chase round the mountain"),
            _mark("caught"), Leg("goto", to=(0, 0, 60.0), speed=0.6 * v, speed_end=4.0, radius=8, name="home"), Leg("land", to=(0, 0, 0), speed=6.0, name="land")]


# --------------------------------------------------------------------------------------------------------------- sea
SEA_WIND, SEA_GUST, BOAT_V = 14.0, 3.0, 25.0                         # a crosswind from the north with gusts; a 90 km/h boat


def _sea_wind(sim, pos, t):
    sim.gust = og_wind.ou_euler_step(sim.gust, sim.dt, SEA_GUST, 4.0, sim.rng)
    return og_wind.from_direction(og_wind.power_law(SEA_WIND, max(pos[2], 1.0), 10.0, 0.11), 90.0) + sim.gust


def _boat(t):
    """A boat running east from 1.2 km out, turning 35° left and right every 12 s."""
    x, y, tt, k = 1200.0, 300.0, 0.0, 0
    while tt + 12.0 < t:
        a = math.radians(35.0 if k % 2 == 0 else -35.0)
        x, y, tt, k = x + BOAT_V * math.cos(a) * 12.0, y + BOAT_V * math.sin(a) * 12.0, tt + 12.0, k + 1
    a = math.radians(35.0 if k % 2 == 0 else -35.0)
    return ((x + BOAT_V * math.cos(a) * (t - tt), y + BOAT_V * math.sin(a) * (t - tt), 0.0), (BOAT_V * math.cos(a), BOAT_V * math.sin(a), 0.0))


def _sea(variant, env, pace):
    v = pace * env["v_top"]
    return [Leg("climb", to=(0, 0, 25.0), name="take-off"), Leg("chase", target=_boat, speed=v, radius=10.0, offset=(0, 0, 12.0), name="chase the boat"),
            _mark("caught"), Leg("hold", target=_boat, speed=v, radius=6.0, seconds=5.0, offset=(0, 0, 12.0), name="hold over the boat"),
            Leg("goto", to=(0, 0, 30.0), speed=0.5 * v, speed_end=4.0, radius=8, name="home"), Leg("land", to=(0, 0, 0), speed=6.0, name="land")]


# ------------------------------------------------------------------------------------------------------------- climb
def _climb(variant, env, pace):
    top = ef.TARGET[variant]["ceiling_m"]
    eco = ceiling(env["config"], top)["profile"]
    v_eco = float(np.mean([r["climb_ms"] for r in eco if r["h"] <= top]))        # the economy climb's speed
    return [Leg("climb", to=(0, 0, top), speed=v_eco, full=False, name="climb"), _mark("top of climb"),
            Leg("fall", to=(0, 0, 220.0), name="flat fall"),
            Leg("goto", to=(0, 0, 40.0), speed=30.0, speed_end=5.0, radius=8, floor=0.0, name="catch"), Leg("land", to=(0, 0, 0), speed=6.0, name="land")]


# ------------------------------------------------------------------------------------------------------------ palace
# The Palace of Culture and Science, Warsaw. Heights as commonly quoted (verify before anything flies there): the
# tower's roof 187.68 m, the top of the spire 237 m, the terrace of the 30th floor 114 m; the ground about 115 m above
# sea level. The landing spot is on the roof beside the spire's base.
PALACE = {"roof": 187.68, "spire_top": 237.0, "terrace": 114.0, "tower_half": 20.0, "spire_radius": 4.0, "ground_asl": 115.0, "pad": (11.0, 0.0)}
PALACE_START = (-260.0, -120.0, 0.0)                                 # a spot on the square, 290 m from the tower's axis


def _palace_terrain(x, y):
    return PALACE["roof"] if (abs(x) <= PALACE["tower_half"] and abs(y) <= PALACE["tower_half"]) else 0.0


def _palace_wind(sim, pos, t):
    sim.gust = og_wind.ou_euler_step(sim.gust, sim.dt, 1.5, 3.0, sim.rng)
    return og_wind.from_direction(og_wind.power_law(6.0, max(pos[2], 2.0), 10.0, 0.25), 250.0) + sim.gust   # a city profile


def _palace(variant, env, pace):
    px, py = PALACE["pad"]
    roof = PALACE["roof"]
    sx, sy, _ = PALACE_START
    return [Leg("climb", to=(sx, sy, 1200.0), name="fast climb"), _mark("top of climb"),
            Leg("goto", to=(px + 480.0, py + 190.0, roof + 330.0), speed=env["v_top"], speed_end=55.0, radius=40, name="swoop"),
            Leg("goto", to=(px + 150.0, py + 60.0, roof + 110.0), speed=55.0, speed_end=20.0, radius=25, name="pull out and slow"),
            Leg("goto", to=(px, py, roof + 12.0), speed=14.0, speed_end=1.5, radius=1.5, floor=4.0, name="slow approach"), _mark("over the roof"),
            Leg("land", to=(px, py, roof), speed=2.5, name="land on the roof")]


SCENARIOS = {
    "wildfire": Scenario("wildfire", "to a wildfire 2 km out, round the smoke and back", _wildfire, paced=True, t_max=400.0,
                         features={"fire": FIRE, "floor": 30.0},
                         notes="flown at the fastest pace that still lands with the reserve"),
    "mountain": Scenario("mountain", "a chase round a mountain", _mountain, alt0=600.0, terrain=_mountain_terrain, t_max=400.0,
                         features={"peak": PEAK, "hare": _hare, "floor": 60.0},
                         notes="the target circles the 900 m peak at 520 m and 137 km/h; the base is 600 m above the sea"),
    "sea": Scenario("sea", "a chase over the sea in a crosswind", _sea, wind=_sea_wind, t_max=400.0,
                    features={"boat": _boat, "wind": (SEA_WIND, 90.0), "floor": 8.0},
                    notes=f"wind {SEA_WIND:g} m/s from the north with {SEA_GUST:g} m/s gusts; the boat runs at {BOAT_V * 3.6:.0f} km/h and zigzags"),
    "climb": Scenario("climb", "a climb to the ceiling target and back", _climb, t_max=1500.0, features={"floor": 0.0},
                      notes="straight up from sea level at the economy climb speed, a flat fall with the motors stopped, a catch and a landing"),
    "palace": Scenario("palace", "a fast climb over Warsaw, a swoop, a slow landing on the roof of the Palace of Culture", _palace,
                       start=PALACE_START, alt0=PALACE["ground_asl"], wind=_palace_wind, terrain=_palace_terrain,
                       obstacles=((0.0, 0.0, PALACE["spire_radius"], PALACE["roof"], PALACE["spire_top"]),), t_max=500.0,
                       features={"palace": PALACE, "floor": 25.0},
                       notes="6 m/s wind at 10 m rising with height, gusts; the spire is an obstacle; touchdown at 0.8 m/s"),
}


def fly_all(variants=("ER1", "ER2"), scenarios=None) -> dict:
    """Every variant through every scenario; the climb also with the variant's high-altitude pack (key ``ER1 high``)."""
    out = {(v, s): fly(v, s) for v in variants for s in (scenarios or SCENARIOS)}
    if "climb" in (scenarios or SCENARIOS):
        for v in variants:
            out[(f"{v} high", "climb")] = fly(v, "climb", config=ef.HIGH[v])
    return out


def table(flights: dict) -> list[dict]:
    """One row per flight: the verdict and the numbers worth reading."""
    rows = []
    for (v, s), f in flights.items():
        m = f.metrics
        rows.append({"variant": v, "scenario": s, "ok": f.ok, "why not": "; ".join(f.why), "pace": round(f.pace, 2), "time [s]": round(m["time_s"], 1),
                     "distance [km]": round(m["distance_km"], 2), "max [km/h]": round(m["max_kmh"]), "top altitude [m]": round(m["max_altitude_m"]),
                     "energy [Wh]": round(m["energy_wh"], 1), "pack left [%]": round(100 * m["reserve"]), "motors [C]": round(m["max_motor_C"]),
                     "max g": round(m["max_g"], 1), "caught at [s]": m.get("t_caught_s"),
                     "touchdown [m/s]": m.get("touchdown_sink_ms"), "miss [m]": m.get("touchdown_miss_m")})
    return rows


# ================================================================================================================ plot
def plot(f: Flight, path=None):
    """One figure of a flight: the ground track with the scenario's features, altitude, speed, and the pack with the
    motors' temperature."""
    import matplotlib.pyplot as plt
    h, ft = f.hist, f.features
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    a = ax[0]
    if "peak" in ft:
        cx, cy = ft["peak"]["xy"]
        for frac in (1.0, 0.75, 0.5, 0.25):
            a.add_patch(plt.Circle((cx, cy), ft["peak"]["radius"] * frac, color="#8c6d31", alpha=0.12))
        ts = np.linspace(0, f.metrics.get("t_caught_s") or h["t"][-1], 200)
        hp = np.array([ft["hare"](t)[0] for t in ts])
        a.plot(hp[:, 0], hp[:, 1], ":", color="k", label="the target")
    if "boat" in ft:
        ts = np.linspace(0, h["t"][-1], 300)
        bp = np.array([ft["boat"](t)[0] for t in ts])
        a.plot(bp[:, 0], bp[:, 1], ":", color="k", label="the boat")
        a.annotate("", xy=(0.12, 0.80), xytext=(0.12, 0.95), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#1f77b4"))
        a.text(0.14, 0.86, f"wind {ft['wind'][0]:g} m/s", transform=a.transAxes, color="#1f77b4", fontsize=8)
    if "fire" in ft:
        a.plot(ft["fire"][0], ft["fire"][1], "*", ms=16, color="#ff7f0e", label="the fire")
    if "palace" in ft:
        t_ = ft["palace"]["tower_half"]
        a.add_patch(plt.Rectangle((-t_, -t_), 2 * t_, 2 * t_, color="0.6", label="the tower"))
        a.add_patch(plt.Circle((0, 0), ft["palace"]["spire_radius"], color="k"))
    a.plot(h["x"], h["y"], color="#d62728", lw=1.6, label=f.variant)
    a.plot(h["x"][0], h["y"][0], "o", color="k", ms=5)
    a.set_aspect("equal", adjustable="datalim")
    a.set(xlabel="east [m]", ylabel="north [m]", title=f"{f.variant}: {f.title}"[:70])
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
    ax[1].plot(h["t"], h["z"], color="#d62728")
    if "palace" in ft:
        ax[1].axhline(ft["palace"]["roof"], color="0.5", ls="--", lw=0.8)
        ax[1].axhline(ft["palace"]["spire_top"], color="k", ls=":", lw=0.8)
    ax[1].set(xlabel="s", ylabel="height above the start's ground [m]", title="altitude")
    ax[2].plot(h["t"], h["speed"] * 3.6, color="#d62728", label="ground speed")
    ax[2].plot(h["t"], h["airspeed"] * 3.6, color="#1f77b4", lw=0.8, label="airspeed")
    ax[2].set(xlabel="s", ylabel="km/h", title=f"speed (max {f.metrics['max_kmh']:.0f} km/h)")
    ax[2].legend(fontsize=7)
    ax[3].plot(h["t"], 100 * (1 - h["wh"] / f.metrics["pack_wh"]), color="#2ca02c", label="pack [%]")
    ax[3].axhline(100 * RESERVE, color="#2ca02c", ls=":", lw=0.8)
    tw = ax[3].twinx()
    tw.plot(h["t"], h["motor_C"], color="#ff7f0e", label="windings [C]")
    tw.set_ylabel("C")
    ax[3].set(xlabel="s", ylabel="%", ylim=(0, 100), title=("OK" if f.ok else "NOT OK: " + "; ".join(f.why))[:60])
    ax[3].legend(fontsize=7, loc="lower left")
    for a_ in ax[1:]:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
