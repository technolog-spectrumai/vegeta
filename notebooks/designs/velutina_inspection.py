"""Velutina v2 — inspecting the blades of an onshore wind turbine (notebook 24, §12–13).

The turbine is parked for the inspection (rotor braked, one blade pointing straight down along the tower, the other
two up in a Y). Velutina flies along each blade at a fixed standoff — leading edge, pressure side, trailing edge,
suction side — filming it, and **dwells** at points of suspected damage until the camera has enough sharp, steady
frames for a reliable judgement. The scan speed and the dwell time are not guesses: they come from a camera model
(pixel size on the blade, motion blur, frame overlap) and from the station-keeping error the gusts and the tower's
wind shadow leave.

Rudimentary aerodynamics: a power-law wind shear, the tower's wind shadow (potential flow around a cylinder upstream
and beside it, a Gaussian velocity deficit in its wake), gusts as in ``velutina_flight.Wind``. The parked blades make
no wake of their own. The drone is the same point-mass-with-an-axis model as ``velutina_flight.simulate``, here
following a path of waypoints with a speed limit and dwell times.

Units SI; the turbine stands at the origin, the tower axis is +z, the wind blows along +x (the rotor faces -x, so
the pressure side of a parked blade faces the wind and the suction side faces the tower).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from velutina_flight import G, Aircraft, Wind, isa_density

# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Turbine:
    """A half-real 2 MW class onshore turbine (V90-like numbers, rounded)."""
    hub_height: float = 80.0
    rotor_radius: float = 45.0
    blade_root_r: float = 1.5           # blade starts at this radius (the hub)
    hub_radius: float = 1.5
    tower_base_d: float = 4.2
    tower_top_d: float = 2.3
    nacelle: tuple = (10.0, 4.0, 4.0)   # length (along x), width, height
    overhang: float = 4.0               # rotor plane ahead (-x) of the tower axis
    tilt_deg: float = 5.0               # rotor axis tilt (nose up)
    cone_deg: float = 3.0               # blades coned away from the tower
    parked_azimuths_deg: tuple = (180.0, 60.0, 300.0)   # 0 = 12 o'clock, clockwise seen from upwind; blade 1 points down
    chord_root: float = 1.9
    chord_max: float = 3.5
    chord_max_r: float = 10.0
    chord_tip: float = 0.5
    site_altitude: float = 300.0

    def tower_diameter(self, z):
        f = np.clip(np.asarray(z, dtype=float) / self.hub_height, 0, 1)
        return self.tower_base_d + (self.tower_top_d - self.tower_base_d) * f

    def chord(self, r):
        r = np.asarray(r, dtype=float)
        rise = self.chord_root + (self.chord_max - self.chord_root) * np.clip((r - self.blade_root_r) / (self.chord_max_r - self.blade_root_r), 0, 1)
        fall = self.chord_max + (self.chord_tip - self.chord_max) * np.clip((r - self.chord_max_r) / (self.rotor_radius - self.chord_max_r), 0, 1)
        return np.where(r < self.chord_max_r, rise, fall)

    @property
    def hub(self):
        return np.array([-self.overhang, 0.0, self.hub_height])

    def blade_frame(self, k):
        """Unit vectors of blade ``k``: ``span`` (root → tip), ``chordwise`` (LE → TE, in the rotor plane) and
        ``normal`` (pressure side → suction side, i.e. downwind, +x)."""
        az = math.radians(self.parked_azimuths_deg[k])
        span = np.array([0.0, math.sin(az), math.cos(az)])          # 0° = up (+z), 90° = +y
        normal = np.array([1.0, 0.0, 0.0])
        # cone: tip leans upwind, away from the tower
        c = math.radians(self.cone_deg)
        span = span * math.cos(c) - normal * math.sin(c)
        span /= np.linalg.norm(span)
        chordwise = np.cross(normal, span)
        return span, chordwise, normal

    def blade_point(self, k, r, chord_frac=0.0, side=0.0):
        """A point on blade ``k`` at radius ``r``: ``chord_frac`` 0 = leading edge, 1 = trailing edge (the chord runs from
        the LE at -0.3 c to the TE at +0.7 c about the pitch axis); ``side`` in units of the local thickness, + = suction."""
        span, chordwise, normal = self.blade_frame(k)
        c = float(self.chord(r))
        thick = 0.22 * c
        return self.hub + span * r + chordwise * ((chord_frac - 0.3) * c) + normal * (side * thick / 2)

    def blade_outline(self, k, n=40):
        """Leading- and trailing-edge polylines of blade ``k`` (for drawing)."""
        rs = np.linspace(self.blade_root_r, self.rotor_radius, n)
        le = np.array([self.blade_point(k, r, 0.0) for r in rs])
        te = np.array([self.blade_point(k, r, 1.0) for r in rs])
        return le, te


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class SiteWind(Wind):
    """The turbine site's wind: hub-height speed with a power-law shear and the tower's shadow."""
    shear_exponent: float = 0.14
    hub_height: float = 80.0

    def mean(self, pos, turbine: Turbine):
        """Steady wind vector at ``pos`` (no gust): shear × tower shadow. Blowing along +x."""
        x, y, z = float(pos[0]), float(pos[1]), max(float(pos[2]), 2.0)
        u = self.speed * (z / self.hub_height) ** self.shear_exponent
        a = float(turbine.tower_diameter(z)) / 2
        r2 = x * x + y * y
        if z < turbine.hub_height - turbine.nacelle[2] / 2 and r2 > a * a:
            if x < 0.5 * a:                                      # upstream and beside: potential flow around the cylinder
                ux = u * (1.0 - a * a * (x * x - y * y) / (r2 * r2))
                uy = -u * 2 * a * a * x * y / (r2 * r2)
                return np.array([ux, uy, 0.0])
            # wake: Gaussian deficit, half-width growing with distance, deficit decaying
            dx = x
            half = a * (1.0 + 0.3 * dx / a)
            deficit = 0.6 * (1.0 + dx / (4 * a)) ** -0.7
            return np.array([u * (1.0 - deficit * math.exp(-(y / half) ** 2)), 0.0, 0.0])
        return np.array([u, 0.0, 0.0])

    wake_turbulence: float = 2.5        # gust sigma multiplier at the centre of the wake, decaying with the deficit

    def turbulence_factor(self, pos, turbine: Turbine):
        """How much stronger the gusts are at ``pos`` than in the free wind: 1 outside the wake, up to
        ``wake_turbulence`` where the tower's velocity deficit is largest (the wake is turbulent, not just slow)."""
        x, y, z = float(pos[0]), float(pos[1]), max(float(pos[2]), 2.0)
        a = float(turbine.tower_diameter(z)) / 2
        if z >= turbine.hub_height - turbine.nacelle[2] / 2 or x <= 0.5 * a:
            return 1.0
        half = a * (1.0 + 0.3 * x / a)
        deficit = 0.6 * (1.0 + x / (4 * a)) ** -0.7 * math.exp(-(y / half) ** 2)
        return 1.0 + (self.wake_turbulence - 1.0) * deficit / 0.6

    def field(self, turbine: Turbine, z, xs, ys):
        X, Y = np.meshgrid(xs, ys)
        U = np.zeros_like(X); V = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                w = self.mean((X[i, j], Y[i, j], z), turbine)
                U[i, j], V[i, j] = w[0], w[1]
        return X, Y, U, V


# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Camera:
    """The inspection camera, nose-mounted (a camera nose instead of the capsule)."""
    pixels_across: int = 4000
    sensor_width_mm: float = 6.17        # 1/2.3 inch
    focal_mm: float = 24.0               # a long lens: fine detail from a safe standoff
    fps: float = 20.0
    exposure_s: float = 1.0 / 4000
    crack_width_mm: float = 1.0          # the finest defect to detect
    pixels_per_crack: float = 3.0        # ... and how many pixels it must span
    overlap: float = 0.6                 # frame-to-frame overlap along the scan
    sharp_frames_needed: int = 30        # steady, sharp frames for one reliable judgement of a point
    position_tolerance_m: float = 0.10   # the camera must sit within this of the planned point for a frame to count

    def gsd_mm(self, standoff_m):
        """Size of one pixel on the blade [mm] at ``standoff_m``."""
        return standoff_m * 1000 * self.sensor_width_mm / (self.focal_mm * self.pixels_across)

    def max_standoff_m(self):
        """Standoff at which the finest crack still spans the required pixels."""
        return self.crack_width_mm / self.pixels_per_crack * self.focal_mm * self.pixels_across / (self.sensor_width_mm * 1000)

    def footprint_m(self, standoff_m):
        return standoff_m * self.sensor_width_mm / self.focal_mm

    def max_scan_speed(self, standoff_m):
        """Scan speed limited by motion blur (under one pixel during the exposure) and by frame overlap."""
        blur = self.gsd_mm(standoff_m) / 1000 / self.exposure_s
        overlap = (1 - self.overlap) * self.footprint_m(standoff_m) * self.fps
        return min(blur, overlap), {"blur_limit_m_s": blur, "overlap_limit_m_s": overlap}

    def dwell_time_s(self, hold_error_series, dt):
        """Time to collect ``sharp_frames_needed`` frames with the camera inside the tolerance, from a station-keeping
        error time series: the fraction of time inside the tolerance scales the frame rate."""
        inside = float(np.mean(np.asarray(hold_error_series) < self.position_tolerance_m)) if len(hold_error_series) else 0.0
        if inside <= 0.0:
            return float("inf"), inside
        return self.sharp_frames_needed / (self.fps * inside), inside


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class InspectionPlan:
    standoff_m: float = 4.0
    passes: tuple = (("leading edge", 0.0, 0.0, "LE"), ("pressure side", 0.5, -1.0, "PS"),
                     ("trailing edge", 1.0, 0.0, "TE"), ("suction side", 0.5, 1.0, "SS"))   # (name, chord_frac, side, tag)
    blades: tuple = (0, 1, 2)
    suspect_points: tuple = ((0, "leading edge", 38.0), (0, "pressure side", 22.0), (1, "trailing edge", 41.0), (2, "leading edge", 30.0))   # (blade, pass, radius)
    scan_speed: float | None = None     # None: the camera's limit
    transit_speed: float = 6.0
    waypoint_spacing_m: float = 2.0
    clearance_ground: float = 15.0


TOWER_CLEARANCE = 2.5      # the camera keeps at least this between itself and the tower's skin [m]


def pass_offset(turbine: Turbine, k, chord_frac, side, standoff, r=None, oblique_deg=60.0):
    """The camera's position relative to the blade surface point for a pass: standing off the surface it films.
    Where that would put the camera inside ``TOWER_CLEARANCE`` of the tower (the suction side of the downward blade,
    between blade and tower), it films obliquely instead: the offset turns ``oblique_deg`` from the normal toward the
    trailing edge, out beside the tower."""
    span, chordwise, normal = turbine.blade_frame(k)
    if side == 0.0:                                     # an edge: stand off along the chord direction, outward
        d = -chordwise if chord_frac == 0.0 else chordwise
    else:                                               # a side: stand off along the normal
        d = normal * (1.0 if side > 0 else -1.0)
    if r is not None and side != 0.0:
        cam = turbine.blade_point(k, r, chord_frac, side) + d * standoff
        if cam[2] < turbine.hub_height - turbine.nacelle[2] / 2 and math.hypot(cam[0], cam[1]) < turbine.tower_diameter(cam[2]) / 2 + TOWER_CLEARANCE:
            a = math.radians(oblique_deg)
            d = d * math.cos(a) + chordwise * math.sin(a)
    return d * standoff


def wake_point(turbine: Turbine, z=40.0, behind=6.0):
    """A reference point squarely in the tower's wake at height ``z`` (not on any pass; for the station-keeping comparison)."""
    return np.array([turbine.tower_diameter(z) / 2 + behind, 0.0, z])


def inspection_path(turbine: Turbine, plan: InspectionPlan, camera: Camera, depot_xy=(-120.0, 60.0), dwell_s=None):
    """Waypoints (position, speed limit, dwell time, label) for the whole inspection, blade by blade and pass by pass,
    root to tip then tip to root, with the suspect points' dwells. ``dwell_s`` is the dwell at a suspect point."""
    v_scan = plan.scan_speed or camera.max_scan_speed(plan.standoff_m)[0]
    dwell = dwell_s if dwell_s is not None else 0.0
    pts = []
    depot = np.array([depot_xy[0], depot_xy[1], 0.0])
    pts.append((depot + [0, 0, plan.clearance_ground], plan.transit_speed, 0.0, "climb-out"))
    for k in plan.blades:
        rs = np.arange(turbine.blade_root_r + 1.0, turbine.rotor_radius - 0.5, plan.waypoint_spacing_m)
        for i, (name, cf, side, tag) in enumerate(plan.passes):
            order = rs if i % 2 == 0 else rs[::-1]
            off0 = pass_offset(turbine, k, cf, side, plan.standoff_m, r=order[0])
            first = turbine.blade_point(k, order[0], cf, side) + off0
            # get to the start of the pass from where we are, at transit speed, outside the rotor's envelope
            pts.append((first + off0 * 2.0, plan.transit_speed, 0.0, f"blade {k + 1}: to {name}"))
            for r in order:
                off = pass_offset(turbine, k, cf, side, plan.standoff_m, r=r)
                p = turbine.blade_point(k, r, cf, side) + off
                sus = [s for s in plan.suspect_points if s[0] == k and s[1] == name and abs(s[2] - r) < plan.waypoint_spacing_m / 2]
                pts.append((p, v_scan, dwell if sus else 0.0, f"blade {k + 1}: {name}" + (" — suspect point" if sus else "")))
    pts.append((depot + [0, 0, plan.clearance_ground], plan.transit_speed, 0.0, "home"))
    pts.append((depot, 2.0, 0.0, "landing"))
    return pts, v_scan


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class InspectionEpisode:
    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    axis: np.ndarray
    thrust: np.ndarray
    power: np.ndarray
    energy_wh: np.ndarray
    label: list
    wind: np.ndarray
    track_error: np.ndarray            # distance to the current waypoint's line
    events: list = field(default_factory=list)

    def summary(self):
        return {"duration_s": float(self.t[-1]), "energy_wh": float(self.energy_wh[-1]), "max_power_w": float(self.power.max()),
                "track_error_rms_m": float(np.sqrt(np.mean(self.track_error[np.isfinite(self.track_error)] ** 2))),
                "track_error_p95_m": float(np.nanpercentile(self.track_error, 95)), "events": list(self.events)}

    def table(self):
        import pandas as pd
        rows, start, name = [], 0, self.label[0]
        for i in range(1, len(self.label) + 1):
            if i == len(self.label) or self.label[i] != name:
                sl = slice(start, i)
                rows.append({"phase": name, "start_s": self.t[start], "duration_s": self.t[i - 1] - self.t[start],
                             "energy_wh": self.energy_wh[i - 1] - self.energy_wh[start], "mean_power_W": float(np.mean(self.power[sl])),
                             "track_error_rms_m": float(np.sqrt(np.nanmean(self.track_error[sl] ** 2)))})
                if i < len(self.label):
                    start, name = i, self.label[i]
        return pd.DataFrame(rows).set_index("phase")


def follow_path(ac: Aircraft, turbine: Turbine, wind: SiteWind, waypoints, *, dt=0.05, max_t=4000.0, seed=0,
                arrive_m=0.6, hold_gain=None) -> InspectionEpisode:
    """Fly the waypoints with the point-mass model of ``velutina_flight``: a velocity command toward the next waypoint
    at its speed limit (slowing to arrive), a dwell where asked, the steady site wind + gusts, drag by attitude."""
    rng = np.random.default_rng(seed)
    rho = float(isa_density(turbine.site_altitude))
    pos = waypoints[-1][0].copy(); pos[2] = 0.0
    vel, axis, gust, nav = np.zeros(3), np.array([0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3)
    k_gain = hold_gain if hold_gain is not None else ac.hold_gain
    k, dwell_t, t = 0, 0.0, 0.0
    log = {n: [] for n in ("t", "pos", "vel", "axis", "thrust", "power", "energy", "label", "wind", "err")}
    energy, events = 0.0, []
    prev = pos.copy()
    while t < max_t and k < len(waypoints):
        goal, v_lim, dwell, label = waypoints[k]
        sigma = wind.gust_sigma * wind.turbulence_factor(pos, turbine)
        gust += (-gust / wind.gust_tau_s) * dt + sigma * math.sqrt(2 * dt / wind.gust_tau_s) * rng.standard_normal(3) * np.array([1, 1, 0.4])
        w = wind.mean(pos, turbine) + gust
        nav += (-nav / 20.0) * dt + ac.position_noise_m * math.sqrt(2 * dt / 20.0) * rng.standard_normal(3) * np.array([1, 1, 0.5])
        pos_est = pos + nav
        to_goal = goal - pos_est
        dist = np.linalg.norm(to_goal)
        # distance from the line prev → goal (the track error the camera sees)
        seg = goal - prev
        if np.linalg.norm(seg) > 1e-6:
            s = np.clip(np.dot(pos - prev, seg) / np.dot(seg, seg), 0, 1)
            err = float(np.linalg.norm(pos - (prev + s * seg)))
        else:
            err = float(np.linalg.norm(pos - goal))
        if dist < arrive_m:
            if dwell > 0 and dwell_t < dwell:
                dwell_t += dt
                v_cmd = to_goal * k_gain                                    # hold on the point
                err = float(np.linalg.norm(pos - goal))
            else:
                if dwell > 0:
                    events.append((round(t, 1), f"{label}: {dwell:.0f} s of steady video recorded"))
                prev = goal.copy(); k += 1; dwell_t = 0.0
                continue
        else:
            v_mag = min(v_lim, math.sqrt(2 * 0.4 * ac.max_accel * dist) + 0.3)
            v_cmd = to_goal / max(dist, 1e-6) * v_mag
        a_cmd = (v_cmd - vel) / ac.response_s
        a_h = np.linalg.norm(a_cmd[:2])
        if a_h > ac.max_accel:
            a_cmd[:2] *= ac.max_accel / a_h
        a_cmd[2] = float(np.clip(a_cmd[2], -0.6 * G, 0.8 * G))
        v_rel = vel - w
        v_rel_mag = np.linalg.norm(v_rel)
        cos_axis = float(np.dot(axis, v_rel) / v_rel_mag) if v_rel_mag > 1e-6 else 1.0
        drag = -0.5 * rho * ac.cda(cos_axis) * v_rel_mag * v_rel
        thrust_vec = ac.mass_kg * (a_cmd + np.array([0, 0, G])) - drag
        T = np.linalg.norm(thrust_vec)
        T_max = ac.max_thrust(rho) * max(0.35, 1.0 - 0.01 * max(0.0, v_rel_mag - 15.0))
        if T > T_max:
            thrust_vec *= T_max / T; T = T_max
        axis = thrust_vec / max(T, 1e-6)
        acc = (axis * T + drag) / ac.mass_kg - np.array([0, 0, G])
        vel = vel + acc * dt
        pos = pos + vel * dt
        if pos[2] < 0.0:
            pos[2] = 0.0; vel[2] = max(vel[2], 0.0)
        P = ac.power(T, max(float(np.dot(axis, v_rel)), 0.0), rho)
        energy += P * dt / 3600.0
        t += dt
        for key, val in (("t", t), ("pos", pos.copy()), ("vel", vel.copy()), ("axis", axis.copy()), ("thrust", T), ("power", P),
                         ("energy", energy), ("label", label), ("wind", w.copy()), ("err", err)):
            log[key].append(val)
    if k < len(waypoints):
        events.append((round(t, 1), "time limit reached before the path was flown"))
    return InspectionEpisode(np.array(log["t"]), np.array(log["pos"]), np.array(log["vel"]), np.array(log["axis"]), np.array(log["thrust"]),
                             np.array(log["power"]), np.array(log["energy"]), log["label"], np.array(log["wind"]), np.array(log["err"]), events)


def station_keeping_at_blade(ac: Aircraft, turbine: Turbine, wind: SiteWind, point, *, seconds=40.0, seed=0, dt=0.05, hold_gain=None):
    """Hold one point next to the blade for ``seconds`` (after a 5 s settle) and return the camera position error series."""
    wps = [(np.asarray(point, dtype=float), 3.0, seconds + 5.0, "hold")]
    ep = follow_path(ac, turbine, wind, [(point + np.array([0, 0, 0.0]), 3.0, 0.0, "to the point")] + wps + [(point, 3.0, 0.0, "done")],
                     dt=dt, seed=seed, max_t=600.0, hold_gain=hold_gain)
    hold = np.array(ep.label) == "hold"
    err = np.linalg.norm(ep.pos[hold] - np.asarray(point, dtype=float), axis=1)
    n_settle = int(5.0 / dt)
    return err[n_settle:] if len(err) > n_settle else err, ep


def flights_needed(ep: InspectionEpisode, battery_wh, reserve=0.2, n_blades=3):
    """How many battery charges the whole inspection takes: energy per blade from the episode's labels."""
    tab = ep.table()
    per_blade = {k: float(tab.loc[tab.index.str.startswith(f"blade {k + 1}"), "energy_wh"].sum()) for k in range(n_blades)}
    usable = battery_wh * (1 - reserve)
    overhead = float(tab.loc[~tab.index.str.startswith("blade"), "energy_wh"].sum())
    flights, load, n = 1, overhead, 0
    for k in range(n_blades):
        if load + per_blade[k] > usable and n > 0:
            flights += 1; load = overhead; n = 0
        load += per_blade[k]; n += 1
    return {"energy_per_blade_wh": per_blade, "overhead_wh": overhead, "usable_with_reserve_wh": usable, "flights": flights}


# ---------------------------------------------------------------------------------------------------------------
def render_movie(ep: InspectionEpisode, turbine: Turbine, path, *, fps=25, seconds=24.0, size=(960, 540), glyph_scale=6.0,
                 title="Velutina v2 — wind turbine blade inspection", progress=False):
    """The turbine (tower, nacelle, three parked blades), the flown path and the aircraft glyph, camera following."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from vegeta.aeromant._watermark import watermark
    from velutina_flight import _glyph

    n_frames = int(seconds * fps)
    idx = np.linspace(0, len(ep.t) - 1, n_frames).astype(int)
    w_px, h_px = size
    fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("#9ec1dd")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#9ec1dd"); ax.computed_zorder = False
    # ground
    gx, gy = np.meshgrid(np.linspace(-600, 600, 2), np.linspace(-600, 600, 2))
    ax.plot_surface(gx, gy, np.zeros_like(gx), color="#7f9a6a", alpha=1.0, shade=False, zorder=0)
    # tower
    zs = np.linspace(0, turbine.hub_height, 24)
    th = np.linspace(0, 2 * np.pi, 20)
    TH, ZS = np.meshgrid(th, zs)
    Rt = turbine.tower_diameter(ZS) / 2
    ax.plot_surface(Rt * np.cos(TH), Rt * np.sin(TH), ZS, color="#e8e8e8", shade=True, linewidth=0, zorder=1)
    # nacelle
    L, Wn, H = turbine.nacelle
    cx, cz = -turbine.overhang + L / 2 - 2.0, turbine.hub_height
    for dz in (-H / 2, H / 2):
        ax.add_collection3d(Poly3DCollection([[(cx - L / 2, -Wn / 2, cz + dz), (cx + L / 2, -Wn / 2, cz + dz), (cx + L / 2, Wn / 2, cz + dz), (cx - L / 2, Wn / 2, cz + dz)]],
                                             facecolors="#f2f2f2", edgecolors="#999", zorder=2))
    for dy in (-Wn / 2, Wn / 2):
        ax.add_collection3d(Poly3DCollection([[(cx - L / 2, dy, cz - H / 2), (cx + L / 2, dy, cz - H / 2), (cx + L / 2, dy, cz + H / 2), (cx - L / 2, dy, cz + H / 2)]],
                                             facecolors="#e0e0e0", edgecolors="#999", zorder=2))
    # blades as polygons
    for k in range(3):
        le, te = turbine.blade_outline(k, n=30)
        poly = np.vstack([le, te[::-1]])
        ax.add_collection3d(Poly3DCollection([poly.tolist()], facecolors="#fafafa", edgecolors="#777", linewidths=0.6, zorder=3))
    trail, = ax.plot([], [], [], color="#ffb300", lw=1.2, zorder=10)
    body_line, = ax.plot([], [], [], color="#111", lw=3.0, zorder=12)
    arm_lines = [ax.plot([], [], [], color="#111", lw=1.5, zorder=12)[0] for _ in range(4)]
    rotor_pts = ax.scatter([], [], [], color="#ff7043", s=26, depthshade=False, zorder=13)
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.set_pane_color((0.62, 0.76, 0.87, 1.0)); a_.line.set_color((1, 1, 1, 0)); a_.set_ticks([])
    ax.grid(False)
    ax.set_xlim(-75, 75); ax.set_ylim(-75, 75); ax.set_zlim(0, 135)
    ax.set_box_aspect((1, 1, 135 / 150), zoom=1.6)
    txt = fig.text(0.02, 0.95, "", fontsize=10, color="#0d2a4a", family="monospace", va="top")
    fig.text(0.02, 0.985, title, fontsize=11, color="#0d2a4a", weight="bold", va="top")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    body0, arms0, _ = _glyph(glyph_scale, 0.165, 0.52)
    frames = []
    it = range(n_frames)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="movie frames")
    for f in it:
        i = idx[f]
        p, ax_ = ep.pos[i], ep.axis[i]
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
        trail.set_data_3d(ep.pos[:i + 1:3, 0], ep.pos[:i + 1:3, 1], ep.pos[:i + 1:3, 2])
        ax.view_init(elev=14, azim=-140 + 80 * f / n_frames)
        v = np.linalg.norm(ep.vel[i]); wv = np.linalg.norm(ep.wind[i][:2])
        txt.set_text(f"t = {ep.t[i]:6.0f} s   {ep.label[i]:<34s} {v:4.1f} m/s   z {p[2]:5.1f} m   wind {wv:4.1f} m/s\n"
                     f"thrust {ep.thrust[i]:5.1f} N   {ep.power[i]:5.0f} W   {ep.energy_wh[i]:5.1f} Wh used   track error {ep.track_error[i]:4.2f} m")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(watermark(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
    plt.close(fig)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    hh, ww = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (ww, hh))
    for fr in frames:
        out.write(fr)
    out.release()
    return path, frames


def wind_figure(turbine: Turbine, wind: SiteWind, z=40.0):
    """The tower's wind shadow at height ``z`` (plan view) and the shear profile."""
    import matplotlib.pyplot as plt
    xs, ys = np.linspace(-12, 30, 85), np.linspace(-12, 12, 49)
    X, Y, U, V = wind.field(turbine, z, xs, ys)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.4, 1]})
    cs = ax[0].contourf(X, Y, np.hypot(U, V), levels=16, cmap="viridis")
    ax[0].streamplot(X, Y, U, V, color="w", density=0.9, linewidth=0.5, arrowsize=0.6)
    a = float(turbine.tower_diameter(z)) / 2
    ax[0].add_patch(plt.Circle((0, 0), a, color="#eee", zorder=5)); ax[0].set_aspect("equal")
    ax[0].set(xlabel="x [m] (wind → +x)", ylabel="y [m]", title=f"wind speed around the tower at z = {z:.0f} m"); fig.colorbar(cs, ax=ax[0], label="m/s")
    zs = np.linspace(2, 130, 60)
    ax[1].plot([wind.mean((-30.0, 0.0, zz), turbine)[0] for zz in zs], zs, label="free wind (shear)")
    ax[1].plot([wind.mean((turbine.tower_diameter(zz) / 2 + 4.0, 0.0, zz), turbine)[0] for zz in zs], zs, label="4 m behind the tower")
    ax[1].axhline(turbine.hub_height, color="#999", ls=":"); ax[1].set(xlabel="m/s", ylabel="z [m]", title="wind with height"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig
