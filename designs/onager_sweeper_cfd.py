"""The Onager Sweeper's suction (notebook 23 §4): the CFD case of the hood under the hull (Aeromant's
``suction_hood`` template, OpenFOAM), what the field says about picking up litter, and a debris movie.

* ``DEBRIS`` — the litter classes the sweeper is sized for (inputs: mass, frontal area, drag coefficient — handbook
  values for the shapes), their terminal velocities and the air speeds that slide them on the road and lift them;
* ``suction_case`` — the Aeromant case: the hood STL (``onager_sweeper.py`` part ``hood``: hood, duct, broom disc
  and housing on the road), the fan's flow, the road moving at the sweeping speed;
* ``sample_hood`` — the speeds and pressures the design needs, read off the solved field (pyvista): the inflow at
  the lips, the upward speed in the hood, the depression under the roof;
* ``SUCTION`` — those numbers as recorded from notebook 23 §4, the inputs of the MuJoCo scene's air drag
  (``onager_sweeper_scenario.Vacuum``) when the notebook is not re-run;
* ``DebrisTracks`` / ``debris_movie`` — litter particles (mass, drag) carried by the solved air field in the hood's
  frame: they ride the road into the hood at the sweeping speed, slide when the drag beats friction, lift when it
  beats their weight, and are counted collected when they enter the duct. A steady field shown as motion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["AIR", "DEBRIS", "SUCTION", "QUALITY", "terminal_velocity", "pickup_table", "suction_case", "sample_hood",
           "DebrisTracks", "debris_movie", "hood_frame"]

G = 9.81
#: Air at 20 °C.
AIR = {"density": 1.2, "kinematic_viscosity": 1.5e-5}
#: Litter classes (inputs): mass [kg], frontal area [m²] (as it lies, seen by an upward flow), drag coefficient,
#: size [m] (the height it lies at), friction on asphalt. Sources: handbook Cd of plates/cylinders/spheres; masses
#: weighed in the office.
DEBRIS = {
    "cigarette butt": {"mass": 0.3e-3, "area": 8e-3 * 25e-3, "cd": 1.0, "size": 0.008, "mu": 0.5, "note": "Ø 8 x 25 mm cylinder"},
    "dry leaf": {"mass": 0.2e-3, "area": 0.06 * 0.04, "cd": 1.2, "size": 0.003, "mu": 0.5, "note": "60 x 40 mm, flat"},
    "paper scrap": {"mass": 1.0e-3, "area": 0.10 * 0.10, "cd": 1.2, "size": 0.002, "mu": 0.5, "note": "100 x 100 mm, flat"},
    "bottle cap": {"mass": 2.0e-3, "area": math.pi / 4 * 0.03 ** 2, "cd": 1.1, "size": 0.012, "mu": 0.5, "note": "Ø 30 x 12 mm"},
    "gum packet": {"mass": 10e-3, "area": 0.07 * 0.02, "cd": 1.1, "size": 0.010, "mu": 0.5, "note": "70 x 20 x 10 mm box"},
    "crushed can": {"mass": 15e-3, "area": 0.10 * 0.066, "cd": 1.0, "size": 0.03, "mu": 0.5, "note": "Ø 66 can crushed to 30 mm"},
    "gravel chip 5 mm": {"mass": 2650 * math.pi / 6 * 0.005 ** 3, "area": math.pi / 4 * 0.005 ** 2, "cd": 0.8, "size": 0.005, "mu": 0.6, "note": "a 5 mm stone"},
    "pebble 15 mm": {"mass": 2650 * math.pi / 6 * 0.015 ** 3, "area": math.pi / 4 * 0.015 ** 2, "cd": 0.6, "size": 0.015, "mu": 0.6, "note": "a 15 mm stone"},
}
#: The suction numbers the MuJoCo scene uses (``onager_sweeper_scenario.Vacuum``) when notebook 23 §4 has not been
#: run on this machine: ``sample_hood`` on the ``fast`` case of the Sweeper's hood (OpenFOAM v2412, 110 k cells, 150
#: iterations, the road at 1 m/s, the fan at ``onager_sweeper_robot.FAN_FLOW``). Notebook 23 §4 recomputes them.
SUCTION = {
    "flow_rate_m3_s": 0.35, "duct_velocity_m_s": 17.9, "gap_velocity_m_s": 6.4, "mouth_velocity_m_s": 3.0,
    "hood_depression_Pa": 49.0, "fan_static_pressure_Pa": 328.0, "air_power_W": 115.0,
    "source": "notebook 23 §4: suction_hood CFD of the Sweeper hood, fast preset (recorded; re-run the notebook to refresh)",
}


def terminal_velocity(d: dict, rho: float = AIR["density"]) -> float:
    """Free-fall terminal speed [m/s]: drag = weight."""
    return math.sqrt(2 * d["mass"] * G / (rho * d["cd"] * d["area"]))


def pickup_table(suction: dict | None = None, rho: float = AIR["density"]) -> pd.DataFrame:
    """Per litter class: its terminal speed, the air speed that slides it on the road (drag = μ m g), whether the
    lip inflow slides it, the hood's upward flow lifts it and the duct carries it."""
    s = SUCTION if suction is None else suction
    rows = {}
    for name, d in DEBRIS.items():
        vt = terminal_velocity(d, rho)
        v_slide = math.sqrt(2 * d["mu"] * d["mass"] * G / (rho * d["cd"] * d["area"]))
        rows[name] = {"mass [g]": d["mass"] * 1e3, "terminal speed [m/s]": vt, "slides at [m/s]": v_slide,
                      "lips slide it": s["gap_velocity_m_s"] > v_slide, "hood lifts it": s["mouth_velocity_m_s"] > vt,
                      "duct carries it": s["duct_velocity_m_s"] > vt, "note": d["note"]}
    return pd.DataFrame(rows).T


#: Mesh and solver settings of ``suction_case``: ``fast`` (the notebook's default: ~100 k cells, 150 iterations, a few
#: minutes on 4 cores — the depression and the mean speeds within ~20 %, the gap under the lips two cells deep) and
#: ``fine`` (~0.9 M cells, 400 iterations, ~15 min on 4 cores: the gap resolved by 5 mm cells).
QUALITY = {
    "fast": dict(cells_per_length=8.0, surface_level=3, near_level=1, gap_level=2, margin_ahead=1.5, margin_behind=1.5,
                 margin_aside=1.5, iterations=150, residual_target=1e-3),
    "fine": dict(cells_per_length=12.0, surface_level=3, near_level=2, gap_level=3, iterations=400, residual_target=1e-4),
}


def suction_case(stl, workdir, p: dict, *, flow_rate: float, ground_speed: float = 1.0, environment=None,
                 quality: str = "fast", **overrides):
    """The Aeromant case of the hood: ``stl`` the ``hood`` part of the design (mm), ``p`` the design parameters,
    ``quality`` a ``QUALITY`` preset (``overrides`` go on top of it)."""
    from vegeta import aeromant

    mm = 1e-3
    params = dict(flow_rate=flow_rate, duct_center=(p["hood_x"] * mm, 0.0, 0.0), duct_inner=p["duct_inner"] * mm,
                  duct_wall=p["duct_wall"] * mm, floor_height=p["hull_bottom"] * mm, kinematic_viscosity=AIR["kinematic_viscosity"],
                  density=AIR["density"], reference_length=p["hood_width"] * mm, ground_speed=ground_speed)
    params.update(QUALITY[quality])
    params.update(overrides)
    return aeromant.CFDCase("suction_hood", stl, params, workdir=workdir, geometry_units="mm", environment=environment)


def hood_frame(p: dict) -> dict:
    """The hood's box [m] on the road (the CFD frame = the design frame scaled): x0, x1, y0, y1, lip, roof, duct."""
    mm = 1e-3
    return {"x0": (p["hood_x"] - p["hood_length"] / 2) * mm, "x1": (p["hood_x"] + p["hood_length"] / 2) * mm,
            "y0": -p["hood_width"] / 2 * mm, "y1": p["hood_width"] / 2 * mm, "lip": p["hood_gap"] * mm,
            "roof": p["hood_height"] * mm, "wall": p["hood_wall"] * mm, "duct": p["duct_inner"] * mm,
            "duct_x": p["hood_x"] * mm, "floor": p["hull_bottom"] * mm, "broom_x": p["broom_x"] * mm,
            "broom_r": p["broom_diameter"] / 2 * mm}


def sample_hood(case, p: dict, n: int = 400) -> dict:
    """Speeds and pressures read off the solved field (pyvista, inverse-distance over cell centres): the mean
    inflow speed through the lip gap (points around the hood's perimeter at the gap's mid-height, 10 mm outside the
    wall), the mean upward speed inside the hood at its mid-height, the mean depression under the roof, and the
    template's own metrics (the duct pressure, the flow drawn)."""
    from vegeta.aeromant import movie, read_case_results

    h = hood_frame(p)
    sampler = movie.openfoam_sampler(case)
    res = read_case_results(case.workdir if hasattr(case, "workdir") else case)
    # the lip gap: a ring of points 10 mm outside the walls at half the gap
    s = np.linspace(0, 1, n, endpoint=False)
    out, ox, oy = 0.01, h["x0"] - 0.01, h["x1"] + 0.01
    ring = []
    for t in s:
        u = t * 4
        if u < 1:
            ring.append((ox + (oy - ox) * u, h["y0"] - out))
        elif u < 2:
            ring.append((oy, h["y0"] - out + (h["y1"] - h["y0"] + 2 * out) * (u - 1)))
        elif u < 3:
            ring.append((oy - (oy - ox) * (u - 2), h["y1"] + out))
        else:
            ring.append((ox, h["y1"] + out - (h["y1"] - h["y0"] + 2 * out) * (u - 3)))
    ring = np.array([(x, y, h["lip"] / 2) for x, y in ring])
    u_ring, ok = sampler(ring)
    gap_speed = float(np.mean(np.linalg.norm(u_ring[ok], axis=1))) if ok.any() else float("nan")
    # inside the hood at mid-height: the upward speed
    xs = np.linspace(h["x0"] + h["wall"] + 0.01, h["x1"] - h["wall"] - 0.01, 12)
    ys = np.linspace(h["y0"] + h["wall"] + 0.01, h["y1"] - h["wall"] - 0.01, 30)
    grid = np.array([(x, y, (h["lip"] + h["roof"]) / 2) for x in xs for y in ys])
    u_in, ok_in = sampler(grid)
    w_mouth = float(np.mean(u_in[ok_in, 2])) if ok_in.any() else float("nan")
    m = res.metrics
    q = m.get("flow_rate_m3_s") or case.parameters["flow_rate"]
    return {"flow_rate_m3_s": q, "duct_velocity_m_s": q / case.parameters["duct_inner"] ** 2, "gap_velocity_m_s": gap_speed,
            "mouth_velocity_m_s": w_mouth, "hood_depression_Pa": -m.get("hood_wall_pressure_Pa", float("nan")),
            "fan_static_pressure_Pa": m.get("fan_static_pressure_Pa"), "air_power_W": m.get("air_power_W"),
            "gap_points_in_mesh": int(ok.sum()), "source": "sample_hood on " + str(case.workdir)}


# ----------------------------------------------------------------------------------------------- debris tracks
@dataclass
class DebrisTracks:
    """Litter particles in the hood's frame (the road moves at ``-ground_speed`` along x): each particle has a
    litter class (``DEBRIS``), rides the road until the air drag beats friction, lifts when the vertical drag beats
    its weight, flies with drag and gravity, lands without bounce, and is collected when it enters the duct (the
    square under the roof) or escapes when it leaves the domain. ``run(seconds)`` integrates with Heun steps and
    keeps ``frames`` (positions every ``frame_dt``) for the movie; ``summary()`` counts the outcomes."""

    sampler: object
    p: dict
    kinds: list
    x0: np.ndarray                   # start positions (N, 3)
    ground_speed: float = 1.0
    rho: float = AIR["density"]
    dt: float = 0.5e-3
    frame_dt: float = 0.02
    frames: list = field(default_factory=list)
    times: list = field(default_factory=list)

    def __post_init__(self):
        self.h = hood_frame(self.p)
        self.N = len(self.kinds)
        d = [DEBRIS[k] for k in self.kinds]
        self.mass = np.array([x["mass"] for x in d])
        self.k_drag = 0.5 * self.rho * np.array([x["cd"] * x["area"] for x in d])     # F = k |u−v| (u−v)
        self.size = np.array([x["size"] for x in d])
        self.mu = np.array([x["mu"] for x in d])
        self.x = np.asarray(self.x0, dtype=float).copy()
        self.x[:, 2] = np.maximum(self.x[:, 2], self.size / 2)
        self.v = np.zeros((self.N, 3))
        self.v[:, 0] = -self.ground_speed
        self.state = np.zeros(self.N, dtype=int)         # 0 on the road, 1 airborne, 2 collected, 3 escaped
        self.t = 0.0
        self.t_collected = np.full(self.N, np.nan)

    def _accel(self, x, v, idx=None):
        """Acceleration and air force of particles ``idx`` (all when None) at positions ``x``, velocities ``v``."""
        idx = slice(None) if idx is None else idx
        u, ok = self.sampler(x)
        u = np.where(ok[:, None], u, 0.0)
        rel = u - v
        f = self.k_drag[idx, None] * np.linalg.norm(rel, axis=1)[:, None] * rel
        return f / self.mass[idx, None] + np.array([0.0, 0.0, -G]), f

    def step(self):
        h = self.h
        active = self.state < 2
        if not active.any():
            return
        a1, f = self._accel(self.x, self.v)
        on_road = active & (self.state == 0)
        if on_road.any():
            # on the road: the vertical drag must beat the weight to lift; otherwise the particle slides only when
            # the horizontal drag beats friction, else it rides the road
            lift = f[:, 2] > self.mass * G
            self.state[on_road & lift] = 1
            idx = np.where(on_road & ~lift)[0]
            v_road = np.array([-self.ground_speed, 0.0, 0.0])
            for i in idx:
                fh = f[i, :2]
                v_rel = self.v[i, :2] - v_road[:2]
                fric = self.mu[i] * max(self.mass[i] * G - f[i, 2], 0.0)
                if np.linalg.norm(v_rel) < 1e-3 and np.linalg.norm(fh) <= fric:
                    self.v[i, :2] = v_road[:2]
                else:
                    direction = v_rel / max(np.linalg.norm(v_rel), 1e-9) if np.linalg.norm(v_rel) > 1e-3 else fh / max(np.linalg.norm(fh), 1e-9)
                    a = (fh - fric * direction) / self.mass[i]
                    self.v[i, :2] += a * self.dt
                    v_rel2 = self.v[i, :2] - v_road[:2]
                    if np.dot(v_rel2, direction) < 0:          # friction stopped it
                        self.v[i, :2] = v_road[:2]
                self.v[i, 2] = 0.0
                self.x[i, :2] += self.v[i, :2] * self.dt
        air = active & (self.state == 1)
        if air.any():
            idx = np.where(air)[0]
            xp = self.x[idx] + self.v[idx] * self.dt
            vp = self.v[idx] + a1[idx] * self.dt
            a2, _ = self._accel(xp, vp, idx)
            self.x[idx] += 0.5 * (self.v[idx] + vp) * self.dt
            self.v[idx] += 0.5 * (a1[idx] + a2) * self.dt
            # landing
            low = self.x[idx, 2] < self.size[idx] / 2
            for j in idx[low]:
                self.x[j, 2] = self.size[j] / 2
                self.v[j, 2] = 0.0
                self.state[j] = 0
        # outcomes
        half = h["duct"] / 2
        in_duct = ((self.x[:, 2] > h["roof"] - 0.02) & (abs(self.x[:, 0] - h["duct_x"]) < half) & (abs(self.x[:, 1]) < half)) | (self.x[:, 2] > h["floor"])
        newly = active & in_duct
        self.state[newly] = 2
        self.t_collected[newly] = self.t
        gone = active & ((self.x[:, 0] < h["x0"] - 1.0) | (abs(self.x[:, 1]) > 1.2) | (self.x[:, 0] > h["x1"] + 1.6))
        self.state[gone] = 3
        self.t += self.dt

    def run(self, seconds: float):
        every = max(1, int(round(self.frame_dt / self.dt)))
        n = int(round(seconds / self.dt))
        for k in range(n):
            if k % every == 0:
                self.frames.append(self.x.copy())
                self.times.append(self.t)
            self.step()
        self.frames.append(self.x.copy())
        self.times.append(self.t)
        return self

    def summary(self) -> pd.DataFrame:
        rows = {}
        for kind in dict.fromkeys(self.kinds):
            sel = np.array([k == kind for k in self.kinds])
            rows[kind] = {"particles": int(sel.sum()), "collected": int((self.state[sel] == 2).sum()),
                          "on the road": int((self.state[sel] == 0).sum()), "airborne": int((self.state[sel] == 1).sum()),
                          "escaped": int((self.state[sel] == 3).sum()),
                          "time to the duct [s]": float(np.nanmean(self.t_collected[sel])) if np.isfinite(self.t_collected[sel]).any() else float("nan")}
        return pd.DataFrame(rows).T


def seed_litter(p: dict, kinds=None, per_kind: int = 6, seed: int = 0, ahead=(0.10, 0.45), width: float = 0.22) -> tuple:
    """Litter on the road ahead of the hood (it rides the road into it): positions (N, 3) and their classes."""
    rng = np.random.default_rng(seed)
    h = hood_frame(p)
    kinds = list(DEBRIS) if kinds is None else list(kinds)
    names, pts = [], []
    for k in kinds:
        for _ in range(per_kind):
            names.append(k)
            pts.append((h["x1"] + rng.uniform(*ahead), rng.uniform(-width, width), DEBRIS[k]["size"] / 2))
    return np.array(pts), names


def debris_movie(case, p: dict, path, *, seconds: float = 2.5, fps: int = 25, ground_speed: float = 1.0, kinds=None,
                 per_kind: int = 6, seed: int = 0, size=(1280, 640)) -> tuple:
    """Litter carried into the hood by the solved field: a side view (x–z, the hood section) and a top view, the
    particles coloured by class, trails behind them; writes an mp4 and returns (path, DebrisTracks)."""
    import matplotlib
    import matplotlib.pyplot as plt
    from vegeta.aeromant import movie
    from vegeta.chiron.viz import to_video

    sampler = movie.openfoam_sampler(case)
    x0, names = seed_litter(p, kinds, per_kind, seed)
    tracks = DebrisTracks(sampler, p, names, x0, ground_speed=ground_speed).run(seconds)
    h = hood_frame(p)
    kinds_u = list(dict.fromkeys(names))
    cmap = matplotlib.colormaps["tab10"]
    colour = {k: cmap(i % 10) for i, k in enumerate(kinds_u)}
    fr = np.asarray(tracks.frames)
    imgs = []
    dpi = 100
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, gridspec_kw={"height_ratios": [1, 1]})
    xlim = (h["x0"] - 0.35, h["x1"] + 0.75)
    for k in range(len(fr)):
        for ax in (ax1, ax2):
            ax.clear()
        # side view: the hood section, the floor, the duct, the broom
        ax1.add_patch(plt.Rectangle((h["x0"], h["lip"]), h["x1"] - h["x0"], h["roof"] - h["lip"], fill=False, lw=2, color="#444"))
        ax1.add_patch(plt.Rectangle((h["duct_x"] - h["duct"] / 2, h["roof"]), h["duct"], h["floor"] - h["roof"], fill=False, lw=2, color="#444"))
        ax1.axhline(h["floor"], color="#888", lw=1); ax1.axhline(0, color="#222", lw=2)
        ax1.add_patch(plt.Rectangle((h["broom_x"] - h["broom_r"], 0.0), 2 * h["broom_r"], 0.08, color="#999", alpha=0.4))
        # top view: the hood footprint, the duct, the broom
        ax2.add_patch(plt.Rectangle((h["x0"], h["y0"]), h["x1"] - h["x0"], h["y1"] - h["y0"], fill=False, lw=2, color="#444"))
        ax2.add_patch(plt.Rectangle((h["duct_x"] - h["duct"] / 2, -h["duct"] / 2), h["duct"], h["duct"], fill=False, lw=1.5, color="#444", ls="--"))
        ax2.add_patch(plt.Circle((h["broom_x"], 0.0), h["broom_r"], color="#999", alpha=0.3))
        k0 = max(0, k - 12)
        for kind in kinds_u:
            sel = np.array([n == kind for n in names])
            ax1.plot(fr[k0:k + 1, sel, 0], fr[k0:k + 1, sel, 2], color=colour[kind], lw=0.8, alpha=0.5)
            ax2.plot(fr[k0:k + 1, sel, 0], fr[k0:k + 1, sel, 1], color=colour[kind], lw=0.8, alpha=0.5)
            ax1.scatter(fr[k, sel, 0], fr[k, sel, 2], s=14, color=colour[kind], label=kind, zorder=3)
            ax2.scatter(fr[k, sel, 0], fr[k, sel, 1], s=14, color=colour[kind], zorder=3)
        ax1.set(xlim=xlim, ylim=(-0.02, h["floor"] + 0.04), ylabel="z [m]", title=f"litter into the hood — t = {tracks.times[k]:.2f} s (the road moves at {ground_speed:g} m/s)")
        ax2.set(xlim=xlim, ylim=(-0.45, 0.45), xlabel="x [m] (forward)", ylabel="y [m]")
        for ax in (ax1, ax2):
            ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax1.legend(fontsize=7, loc="upper right", ncol=2)
        fig.tight_layout()
        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    out = to_video(imgs, Path(path), fps=fps)
    return out, tracks
