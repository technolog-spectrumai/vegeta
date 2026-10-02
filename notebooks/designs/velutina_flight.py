"""Velutina — the mountain medical-aid mission as a flight simulation, and its movie (notebook 24).

A reduced flight model, honest about what it is: a point mass with a body axis. The four pusher propellers thrust
along the body axis; the guidance tilts the body to point the thrust where the wanted acceleration is; the body and
fins give drag that depends on the angle between the axis and the relative wind (a two-coefficient model fitted to
the CFD or to the hand estimate). Air density follows the ISA with altitude; the propulsion comes from Boreas points
(hover power and the thrust limit per motor at sea level, scaled with density). Wind is a steady vector plus a
first-order gust process. Everything else — attitude dynamics, motor lag, the autopilot's estimator — is assumed
ideal ("the electronics is there").

Mission: a valley depot → a fast climb along a route of waypoints → a hover descent onto a small pad at a mountain
rescue site → the capsule is set down → the flight home → landing. The emergency parachute is a separate
analysis (`parachute_descent`), not part of the nominal flight.

Units SI (m, kg, s, N, W); altitudes are above sea level; the ground is the `Terrain` height.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

G = 9.80665
RHO0, T0, LAPSE = 1.225, 288.15, 0.0065


def isa_density(altitude_m):
    """ISA troposphere density [kg/m^3]."""
    t = T0 - LAPSE * np.asarray(altitude_m, dtype=float)
    return RHO0 * (t / T0) ** (G / (LAPSE * 287.05) - 1.0)


# ---------------------------------------------------------------------------------------------------------------
# the terrain: an abstract mountain valley (smooth, analytic), the depot low, the rescue site high
@dataclass(frozen=True)
class Terrain:
    """Height [m ASL] = base + a few smooth ridges. The depot stands at (0, 0), the site at ``site_xy``."""
    base: float = 600.0
    site_xy: tuple = (8000.0, 1500.0)
    site_altitude: float = 2400.0
    ridges: tuple = ((5200.0, -1800.0, 1400.0, 2400.0, 1500.0),      # (x, y, height, sx, sy): a ridge beside the route
                     (9800.0, 2800.0, 1900.0, 2600.0, 1800.0),       # the massif behind the site
                     (2600.0, 3600.0, 900.0, 2200.0, 1600.0))

    def height(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        h = np.full(np.broadcast(x, y).shape, self.base)
        for cx, cy, a, sx, sy in self.ridges:
            h = h + a * np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
        # the site itself: a shoulder that the route climbs onto
        sxy = self.site_xy
        shoulder = (self.site_altitude - self.base) * np.exp(-(((x - sxy[0]) / 1500.0) ** 2 + ((y - sxy[1]) / 1100.0) ** 2))
        h = np.maximum(h, self.base + shoulder)
        return h

    wall_normal_deg: float = 200.0      # hand-over mode: the rock face at the site faces this way (free air is this side)

    @property
    def wall_normal(self):
        a = math.radians(self.wall_normal_deg)
        return np.array([math.cos(a), math.sin(a), 0.0])

    @property
    def depot(self):
        return np.array([0.0, 0.0, float(self.height(0.0, 0.0))])

    @property
    def site(self):
        return np.array([self.site_xy[0], self.site_xy[1], float(self.height(*self.site_xy))])


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Aircraft:
    """What the flight model needs, all from the notebook's earlier sections."""
    mass_kg: float                      # with the capsule
    capsule_kg: float                   # set down at the site
    cda_axial_m2: float                 # drag area with the axis along the relative wind
    cda_cross_m2: float                 # drag area with the axis across the relative wind
    hover_power_w_sl: float             # electrical power of all motors at hover, sea level, with the capsule
    hover_thrust_n_sl: float            # the thrust at which that power was found (all motors)
    max_thrust_n_sl: float              # all motors, full throttle, static, sea level
    battery_wh: float                   # usable energy
    disk_area_m2: float = 4 * math.pi * 0.089 ** 2   # all rotors (7 inch default)
    max_tilt_deg: float = 80.0          # the body may fly almost horizontal (nose first)
    max_speed: float = 40.0             # the guidance's speed cap
    max_accel: float = 12.0             # horizontal acceleration cap [m/s^2]
    v_climb: float = 8.0                # vertical speed caps
    v_descend: float = 4.0
    v_final: float = 0.8                # last metres over the pad
    position_noise_m: float = 0.25      # the navigation solution's error (random walk, 1 sigma) seen by the guidance
    response_s: float = 1.2             # the velocity loop's time constant (attitude and motor lags lumped)
    hold_gain: float = 0.7              # hand mode: position-loop gain of the station keeping [1/s]

    def cda(self, cos_axis):
        """Drag area for the cosine of the angle between the body axis and the relative wind."""
        c2 = cos_axis ** 2
        return self.cda_axial_m2 * c2 + self.cda_cross_m2 * (1.0 - c2)

    def max_thrust(self, rho):
        return self.max_thrust_n_sl * rho / RHO0

    def power(self, thrust_n, airspeed, rho):
        """Electrical power [W] for ``thrust_n`` (all motors) at ``airspeed`` along the axis and density ``rho``:
        momentum-theory scaling of the hover point (P ∝ T^1.5 / sqrt(rho)) with a forward-flight relief (the
        induced part falls with the inflow) and the profile part proportional to T."""
        t = max(float(thrust_n), 1e-6)
        p_hover = self.hover_power_w_sl * (t / self.hover_thrust_n_sl) ** 1.5 * math.sqrt(RHO0 / rho)
        induced, profile = 0.75 * p_hover, 0.25 * p_hover * (t / self.hover_thrust_n_sl) ** -0.5
        v_i = math.sqrt(t / (2.0 * rho * self.disk_area_m2))
        relief = 1.0 / math.sqrt(1.0 + (0.6 * airspeed / v_i) ** 2)
        return induced * relief + profile          # the parasite power is in the thrust (drag) already


@dataclass
class Wind:
    """Steady wind [m/s] in the xy plane (from a direction, so the vector is ``-speed * (cos, sin)``) plus a
    first-order gust process with the given standard deviation and time constant; stronger with altitude."""
    speed: float = 8.0
    direction_deg: float = 240.0        # meteorological: blowing FROM 240° (south-west)
    gust_sigma: float = 2.5
    gust_tau_s: float = 4.0
    shear_per_km: float = 0.35          # +35 % per km above the valley
    seed: int = 0

    def steady(self, altitude_m, base_altitude):
        a = math.radians(self.direction_deg)
        f = 1.0 + self.shear_per_km * max(0.0, (altitude_m - base_altitude)) / 1000.0
        return -self.speed * f * np.array([math.cos(a), math.sin(a), 0.0])


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Plan:
    """The mission as phases. ``cruise_agl`` is the clearance above the terrain along the route."""
    cruise_agl: float = 120.0
    approach_agl: float = 25.0          # hover point above the pad
    hold_s: float = 6.0                 # on the pad: release the capsule
    waypoints_out: int = 6              # route points between depot and site (terrain-following)
    pad_radius: float = 1.0             # what counts as "on the pad"
    mode: str = "pad"                   # "pad": set the capsule down; "hand": hover at a wall, a person takes it
    standoff_m: float = 1.6             # hand mode: body axis to the rock face
    hand_height_m: float = 1.5          # hand mode: capsule height above the ledge the person stands on
    hand_hold_s: float = 12.0           # hand mode: station-keeping time for the hand-over
    hand_arrival_m: float = 0.4         # hand mode: how close to the hand-over point before the hold starts
    hand_arrival_max_s: float = 25.0    # ... and how long it may try before holding where it is
    hold_settle_s: float = 3.0          # the hold's first seconds are not counted in the station-keeping error


@dataclass
class Episode:
    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    axis: np.ndarray            # unit body axis (thrust direction)
    thrust: np.ndarray          # all motors [N]
    power: np.ndarray           # [W]
    energy_wh: np.ndarray       # consumed
    phase: list
    events: list = field(default_factory=list)
    touchdown_error_m: float = float("nan")
    hold_error_m: np.ndarray | None = None       # hand mode: position error during the hold
    min_wall_distance_m: float = float("nan")
    rho: np.ndarray | None = None
    wind: np.ndarray | None = None

    def phase_table(self):
        import pandas as pd
        rows, start, name = [], 0, self.phase[0]
        for i in range(1, len(self.phase) + 1):
            if i == len(self.phase) or self.phase[i] != name:
                sl = slice(start, i)
                d = np.linalg.norm(np.diff(self.pos[sl], axis=0), axis=1).sum() if i - start > 1 else 0.0
                rows.append({"phase": name, "start_s": self.t[start], "duration_s": self.t[i - 1] - self.t[start],
                             "distance_m": d, "max_speed_m_s": np.linalg.norm(self.vel[sl], axis=1).max(),
                             "energy_wh": self.energy_wh[i - 1] - self.energy_wh[start],
                             "max_thrust_N": self.thrust[sl].max(), "max_power_W": self.power[sl].max()})
                if i < len(self.phase):
                    start, name = i, self.phase[i]
        return pd.DataFrame(rows).set_index("phase")

    def summary(self):
        return {"duration_s": float(self.t[-1]), "energy_wh": float(self.energy_wh[-1]),
                "max_speed_m_s": float(np.linalg.norm(self.vel, axis=1).max()),
                "max_power_w": float(self.power.max()), "touchdown_error_m": float(self.touchdown_error_m),
                "hold_error_rms_m": float(np.sqrt(np.mean(self.hold_error_m ** 2))) if self.hold_error_m is not None and len(self.hold_error_m) else float("nan"),
                "hold_error_max_m": float(np.max(self.hold_error_m)) if self.hold_error_m is not None and len(self.hold_error_m) else float("nan"),
                "min_wall_distance_m": float(self.min_wall_distance_m), "events": list(self.events)}


def route(terrain: Terrain, a, b, n, agl):
    """Waypoints from ``a`` to ``b`` following the terrain ``agl`` metres above it (the straight line in plan)."""
    pts = []
    for s in np.linspace(0, 1, n + 2)[1:-1]:
        x, y = a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])
        pts.append(np.array([x, y, float(terrain.height(x, y)) + agl]))
    return pts


def simulate(ac: Aircraft, terrain: Terrain, wind: Wind, plan: Plan, *, dt=0.05, max_t=1500.0, seed=None) -> Episode:
    """Fly the mission. Guidance: a velocity command toward the current goal (speed-capped, slowing near it),
    an acceleration command limited by ``max_accel`` and the thrust available, the body tilted toward the needed
    thrust vector, drag from the relative wind. Returns the logged episode."""
    rng = np.random.default_rng(wind.seed if seed is None else seed)
    depot, site = terrain.depot, terrain.site
    out = route(terrain, depot, site, plan.waypoints_out, plan.cruise_agl)
    back = route(terrain, site, depot, plan.waypoints_out, plan.cruise_agl)
    hand_point = site + plan.standoff_m * terrain.wall_normal + [0, 0, plan.hand_height_m]
    if plan.mode == "hand":
        delivery = [("approach", hand_point + 3.0 * terrain.wall_normal + [0, 0, plan.approach_agl]),
                    ("final approach", hand_point), ("hand-over", None)]
    else:
        delivery = [("approach", site + [0, 0, plan.approach_agl]), ("final descent", site + [0, 0, 0.0]), ("set-down", None)]
    goals = ([("climb-out", depot + [0, 0, plan.cruise_agl])] + [("transit out", w) for w in out] + delivery
             + [("climb-out home", site + [0, 0, plan.cruise_agl])] + [("transit home", w) for w in back]
             + [("approach home", depot + [0, 0, plan.approach_agl]), ("landing", depot)])
    pos, vel = depot.copy(), np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    gust = np.zeros(3)
    nav_err = np.zeros(3)
    mass = ac.mass_kg
    disk_relief = None
    k = 0
    hold_t = 0.0
    log = {k_: [] for k_ in ("t", "pos", "vel", "axis", "thrust", "power", "energy", "phase", "rho", "wind")}
    energy = 0.0
    events = []
    touchdown_error = float("nan")
    hold_err, min_wall = [], float("inf")
    t_phase = 0.0
    t = 0.0
    while t < max_t and k < len(goals):
        name, goal = goals[k]
        alt = pos[2]
        rho = float(isa_density(alt))
        # wind: steady + gust (Ornstein–Uhlenbeck)
        gust += (-gust / wind.gust_tau_s) * dt + wind.gust_sigma * math.sqrt(2 * dt / wind.gust_tau_s) * rng.standard_normal(3) * np.array([1, 1, 0.4])
        w = wind.steady(alt, terrain.base) + gust
        # navigation error: a slow random walk the guidance cannot see (it steers the *estimated* position)
        nav_err += (-nav_err / 20.0) * dt + ac.position_noise_m * math.sqrt(2 * dt / 20.0) * rng.standard_normal(3) * np.array([1, 1, 0.5])
        pos_est = pos + nav_err
        # ---- guidance: velocity command
        if goal is None:                                    # set-down / hand-over: hold, release the capsule
            hand = name == "hand-over"
            v_cmd = (hand_point - pos_est) * ac.hold_gain if hand else np.zeros(3)     # hand: active station keeping
            hold_t += dt
            if hand:
                err = pos - hand_point
                if hold_t >= plan.hold_settle_s:
                    hold_err.append(float(np.linalg.norm(err)))
                min_wall = min(min_wall, plan.standoff_m + float(np.dot(err, terrain.wall_normal)))
            if hold_t >= (plan.hand_hold_s if hand else plan.hold_s):
                mass = ac.mass_kg - ac.capsule_kg
                events.append((round(t, 1), "capsule taken by hand at the wall" if hand else "capsule released at the site"))
                k += 1
                continue
        else:
            to_goal = goal - pos_est
            dist = np.linalg.norm(to_goal)
            final = name in ("final descent", "landing", "final approach")
            last = name in ("approach", "approach home") or final
            v_cap = ac.max_speed if not last else (ac.v_descend if final else 12.0)
            # slow down toward the goal so it can be reached: v = sqrt(2 a d), capped
            v_mag = min(v_cap, math.sqrt(2 * 0.5 * ac.max_accel * max(dist, 0.0)) + (ac.v_final if final else 1.5))
            v_cmd = to_goal / max(dist, 1e-6) * v_mag
            v_cmd[2] = float(np.clip(v_cmd[2], -ac.v_descend, ac.v_climb)) if not final else float(np.clip(v_cmd[2], -ac.v_descend, ac.v_climb))
            if final and dist < 3.0:
                v_cmd = to_goal / max(dist, 1e-6) * ac.v_final
            tol = plan.hand_arrival_m if name == "final approach" else (plan.pad_radius if last else 25.0)
            reached = dist < tol and (np.linalg.norm(vel) < (0.8 if name == "final approach" else 1.5) if last else True)
            if name == "final approach" and t_phase > plan.hand_arrival_max_s:
                reached = True
            if final and pos[2] <= goal[2] + 0.05 and np.linalg.norm(vel) < 1.2:
                reached = True
            if reached:
                if name == "final descent":
                    touchdown_error = float(np.linalg.norm((pos - site)[:2]))
                    events.append((round(t, 1), f"touchdown at the site, {touchdown_error:.2f} m from the pad centre"))
                if name == "final approach":
                    touchdown_error = float(np.linalg.norm(pos - hand_point))
                    events.append((round(t, 1), f"on station at the wall, {touchdown_error:.2f} m from the hand-over point"))
                if name == "landing":
                    events.append((round(t, 1), f"landed at the depot, {np.linalg.norm((pos - depot)[:2]):.2f} m off"))
                k += 1
                t_phase = 0.0
                if k >= len(goals):
                    break
                continue
        t_phase += dt
        # ---- acceleration command (velocity loop), limited
        a_cmd = (v_cmd - vel) / ac.response_s
        a_h = np.linalg.norm(a_cmd[:2])
        if a_h > ac.max_accel:
            a_cmd[:2] *= ac.max_accel / a_h
        a_cmd[2] = float(np.clip(a_cmd[2], -0.6 * G, 0.8 * G))
        # ---- forces: drag from the relative wind (axis from the previous step), then the thrust that gives a_cmd
        v_rel = vel - w
        v_rel_mag = np.linalg.norm(v_rel)
        cos_axis = float(np.dot(axis, v_rel) / v_rel_mag) if v_rel_mag > 1e-6 else 1.0
        drag = -0.5 * rho * ac.cda(cos_axis) * v_rel_mag * v_rel
        thrust_vec = mass * (a_cmd + np.array([0, 0, G])) - drag
        T = np.linalg.norm(thrust_vec)
        T_max = ac.max_thrust(rho) * max(0.35, 1.0 - 0.01 * max(0.0, v_rel_mag - 15.0))      # pusher thrust falls with speed
        if T > T_max:
            thrust_vec *= T_max / T
            T = T_max
        new_axis = thrust_vec / max(T, 1e-6)
        tilt = math.degrees(math.acos(float(np.clip(new_axis[2], -1, 1))))
        if tilt > ac.max_tilt_deg:                                      # tilt limit: keep the vertical component
            h = new_axis[:2] / max(np.linalg.norm(new_axis[:2]), 1e-9)
            new_axis = np.array([*(h * math.sin(math.radians(ac.max_tilt_deg))), math.cos(math.radians(ac.max_tilt_deg))])
        axis = new_axis
        acc = (axis * T + drag) / mass - np.array([0, 0, G])
        vel = vel + acc * dt
        pos = pos + vel * dt
        ground = float(terrain.height(pos[0], pos[1]))
        if pos[2] < ground:
            pos[2] = ground
            vel[2] = max(vel[2], 0.0)
        v_axis = float(np.dot(axis, v_rel))
        P = ac.power(T, max(v_axis, 0.0), rho)
        energy += P * dt / 3600.0
        t += dt
        for key, val in (("t", t), ("pos", pos.copy()), ("vel", vel.copy()), ("axis", axis.copy()), ("thrust", T),
                         ("power", P), ("energy", energy), ("phase", name), ("rho", rho), ("wind", w.copy())):
            log[key].append(val)
        if energy > ac.battery_wh and not any("battery" in e[1] for e in events):
            events.append((round(t, 1), "usable battery exhausted"))
    ep = Episode(np.array(log["t"]), np.array(log["pos"]), np.array(log["vel"]), np.array(log["axis"]),
                 np.array(log["thrust"]), np.array(log["power"]), np.array(log["energy"]), log["phase"], events,
                 touchdown_error, np.array(hold_err), min_wall if hold_err else float("nan"), np.array(log["rho"]), np.array(log["wind"]))
    return ep


def touchdown_statistics(ac, terrain, wind, plan, n=12, **kw):
    """The set-down error over ``n`` gust realisations (the same wind, different seeds)."""
    errs, energies, holds, walls = [], [], [], []
    for s in range(n):
        ep = simulate(ac, terrain, wind, plan, seed=s, **kw)
        errs.append(ep.touchdown_error_m)
        energies.append(ep.energy_wh[-1])
        if ep.hold_error_m is not None and len(ep.hold_error_m):
            holds.append(float(np.max(ep.hold_error_m))); walls.append(ep.min_wall_distance_m)
    errs = np.array(errs)
    out = {"n": n, "mean_m": float(np.nanmean(errs)), "p95_m": float(np.nanpercentile(errs, 95)), "max_m": float(np.nanmax(errs)),
           "energy_mean_wh": float(np.mean(energies)), "errors_m": errs.tolist()}
    if holds:
        out.update({"hold_error_max_m": float(np.max(holds)), "hold_error_max_mean_m": float(np.mean(holds)), "min_wall_distance_m": float(np.min(walls))})
    return out


# ---------------------------------------------------------------------------------------------------------------
def parachute_descent(mass_kg, altitude_m, ground_m, chute_cd_area_m2, wind: Wind, terrain_base, *, deploy_s=1.5,
                      shock_factor=1.8, v_initial=35.0, dt=0.05):
    """Emergency recovery from ``altitude_m``: the opening shock (``shock_factor`` × the steady drag at the speed at
    deployment, after ``deploy_s`` of free deceleration), then the terminal descent with the steady wind's drift.
    Returns the descent time, the ground speed at touchdown, the drift and the peak shock load [N]."""
    rho = float(isa_density(altitude_m))
    v_term = math.sqrt(2 * mass_kg * G / (rho * chute_cd_area_m2))
    v_open = v_initial * math.exp(-deploy_s * 0.5 * rho * 0.03 * v_initial / mass_kg)   # a little body drag first
    shock = shock_factor * 0.5 * rho * chute_cd_area_m2 * v_open ** 2
    h = altitude_m - ground_m
    t = h / v_term + deploy_s
    drift = np.linalg.norm(wind.steady(0.5 * (altitude_m + ground_m), terrain_base)[:2]) * t
    return {"terminal_speed_m_s": v_term, "descent_time_s": t, "drift_m": float(drift), "opening_shock_N": shock,
            "shock_g": shock / (mass_kg * G), "touchdown_vertical_m_s": math.sqrt(2 * mass_kg * G / (float(isa_density(ground_m)) * chute_cd_area_m2))}


# ---------------------------------------------------------------------------------------------------------------
# the movie: an abstract sky, the terrain as a shaded surface, the route, and the aircraft as a glyph
def _glyph(scale, arm_reach, body_length):
    """Points of the aircraft glyph in body coordinates (x along the axis, forward = nose = -axis? no: the thrust
    points along +axis and the nose is ahead of the arms: nose at +axis side)."""
    L, R = body_length * scale, arm_reach * scale
    body = np.array([[0.0, 0, L * 0.55], [0, 0, -L * 0.45]])
    arms = []
    for a in (45, 135, 225, 315):
        r = math.radians(a)
        arms.append(np.array([[0, 0, -L * 0.2], [R * math.cos(r), R * math.sin(r), -L * 0.2]]))
    return body, arms, R


def render_movie(ep: Episode, terrain: Terrain, path, *, fps=25, seconds=24.0, size=(960, 540), glyph_scale=320.0,
                 arm_reach=0.165, body_length=0.52, title="Velutina — medical aid to the mountains", progress=False):
    """Frames with matplotlib (3D), the Vegeta logo as a watermark, MP4 through OpenCV. ``seconds`` of movie cover the
    whole mission (time-lapse); the camera follows the aircraft from behind-above and the route is drawn as a trail."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from vegeta.aeromant._watermark import watermark

    n_frames = int(seconds * fps)
    idx = np.linspace(0, len(ep.t) - 1, n_frames).astype(int)
    xs = np.linspace(min(ep.pos[:, 0].min(), 0) - 1200, ep.pos[:, 0].max() + 2200, 90)
    ys = np.linspace(ep.pos[:, 1].min() - 2600, ep.pos[:, 1].max() + 2600, 90)
    X, Y = np.meshgrid(xs, ys)
    Z = terrain.height(X, Y)
    ls = LightSource(azdeg=315, altdeg=40)
    snow = np.clip((Z - 1700) / 900, 0, 1)
    base_rgb = np.stack([0.42 + 0.5 * snow, 0.50 + 0.45 * snow, 0.38 + 0.6 * snow], axis=-1)
    shaded = ls.shade_rgb(np.clip(base_rgb, 0, 1), Z, vert_exag=0.6, blend_mode="soft")
    w_px, h_px = size
    fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("#8fb6d9")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#8fb6d9")
    ax.computed_zorder = False
    ax.plot_surface(X, Y, Z, facecolors=shaded, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), 2.2 * (Z.max() - Z.min())), zoom=1.45)
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.set_pane_color((0.56, 0.71, 0.85, 1.0)); a_.line.set_color((1, 1, 1, 0)); a_.set_ticks([])
    ax.grid(False)
    d, s = terrain.depot, terrain.site
    ax.scatter([d[0]], [d[1]], [d[2] + 5], color="#1565c0", s=40, depthshade=False, zorder=11)
    ax.scatter([s[0]], [s[1]], [s[2] + 5], color="#c62828", s=60, marker="P", depthshade=False, zorder=11)
    ax.text(d[0], d[1], d[2] + 250, "depot", color="#0d2a4a", fontsize=8)
    ax.text(s[0], s[1], s[2] + 250, "rescue site", color="#7a0000", fontsize=8)
    if "hand-over" in ep.phase:
        hp = s + 1.6 * terrain.wall_normal + [0, 0, 1.5]
        ax.scatter([hp[0]], [hp[1]], [hp[2] + 5], color="#6a1b9a", s=30, marker="^", depthshade=False)
    trail, = ax.plot([], [], [], color="#ffeb3b", lw=1.4, zorder=10)
    body_line, = ax.plot([], [], [], color="#111", lw=3.0, zorder=12)
    arm_lines = [ax.plot([], [], [], color="#111", lw=1.5, zorder=12)[0] for _ in range(4)]
    rotor_pts = ax.scatter([], [], [], color="#ff7043", s=28, depthshade=False, zorder=13)
    txt = fig.text(0.02, 0.95, "", fontsize=10, color="#0d2a4a", family="monospace", va="top")
    fig.text(0.02, 0.985, title, fontsize=11, color="#0d2a4a", weight="bold", va="top")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    body0, arms0, R = _glyph(glyph_scale, arm_reach, body_length)
    frames = []
    it = range(n_frames)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="movie frames")
    for f in it:
        i = idx[f]
        p, ax_ = ep.pos[i], ep.axis[i]
        # body frame: z along the axis, x any perpendicular
        z = ax_ / np.linalg.norm(ax_)
        xref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        xb = np.cross(xref, z); xb /= np.linalg.norm(xb); yb = np.cross(z, xb)
        Rm = np.stack([xb, yb, z], axis=1)
        b = (Rm @ body0.T).T + p
        body_line.set_data_3d(b[:, 0], b[:, 1], b[:, 2])
        tips = []
        for line, arm in zip(arm_lines, arms0):
            a = (Rm @ arm.T).T + p
            line.set_data_3d(a[:, 0], a[:, 1], a[:, 2]); tips.append(a[1])
        tips = np.array(tips)
        rotor_pts._offsets3d = (tips[:, 0], tips[:, 1], tips[:, 2])
        trail.set_data_3d(ep.pos[:i + 1:4, 0], ep.pos[:i + 1:4, 1], ep.pos[:i + 1:4, 2])
        # camera: behind and above, looking along the route
        az = math.degrees(math.atan2(p[1] - (d[1] - 2500), p[0] - (d[0] - 4500)))
        ax.view_init(elev=28, azim=az - 150 + 40 * f / n_frames)
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_zlim(Z.min(), Z.max() + 400)
        v = np.linalg.norm(ep.vel[i]); agl = p[2] - float(terrain.height(p[0], p[1]))
        txt.set_text(f"t = {ep.t[i]:6.0f} s   {ep.phase[i]:<16s}  {v:5.1f} m/s   alt {p[2]:5.0f} m ASL   {agl:4.0f} m AGL\n"
                     f"thrust {ep.thrust[i]:5.1f} N   {ep.power[i]:5.0f} W   {ep.energy_wh[i]:5.1f} Wh used   rho {ep.rho[i]:.3f}")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frames.append(watermark(bgr))
    plt.close(fig)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    hh, ww = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (ww, hh))
    for fr in frames:
        out.write(fr)
    out.release()
    return path, frames


def profile_figure(ep: Episode, terrain: Terrain):
    """Altitude, speed, thrust and power against time, and the plan view over the terrain."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
    t = ep.t
    ground = terrain.height(ep.pos[:, 0], ep.pos[:, 1])
    ax = axes[0, 0]
    ax.fill_between(t, ground.min() - 100, ground, color="#9e9e9e", alpha=0.5, label="terrain under the track")
    ax.plot(t, ep.pos[:, 2], color="#1565c0", label="altitude ASL")
    ax.set(xlabel="time [s]", ylabel="m", title="altitude"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(t, np.linalg.norm(ep.vel, axis=1), label="ground speed"); ax.plot(t, np.linalg.norm(ep.vel - ep.wind, axis=1), label="airspeed", alpha=0.7)
    ax.plot(t, ep.vel[:, 2], label="vertical speed", alpha=0.7); ax.set(xlabel="time [s]", ylabel="m/s", title="speed"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(t, ep.thrust, color="#c62828", label="thrust, all motors [N]"); ax.plot(t, ep.power / 100, color="#ef6c00", label="electrical power / 100 [W]")
    ax.plot(t, ep.energy_wh, color="#2e7d32", label="energy used [Wh]"); ax.set(xlabel="time [s]", title="propulsion"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    xs = np.linspace(-1500, ep.pos[:, 0].max() + 2500, 120); ys = np.linspace(ep.pos[:, 1].min() - 3000, ep.pos[:, 1].max() + 3000, 120)
    X, Y = np.meshgrid(xs, ys)
    cs = ax.contourf(X, Y, terrain.height(X, Y), levels=14, cmap="terrain")
    ax.plot(ep.pos[:, 0], ep.pos[:, 1], color="#ffeb3b", lw=1.5); ax.plot(*terrain.depot[:2], "o", color="#1565c0"); ax.plot(*terrain.site[:2], "P", color="#c62828", ms=9)
    ax.set(xlabel="x [m]", ylabel="y [m]", title="plan view"); ax.set_aspect("equal"); fig.colorbar(cs, ax=ax, label="terrain [m ASL]")
    for a in axes.flat[:3]:
        last = None
        for i, ph in enumerate(ep.phase):
            if ph != last:
                a.axvline(t[i], color="#ddd", lw=0.6, zorder=0); last = ph
    fig.tight_layout()
    return fig
