"""Particle movies of a propeller case (``rotor_mrf`` / ``rotor_mrf_static``) with OpenCV.

A few tracer particles are carried by the converged OpenFOAM velocity field and drawn in two views:
a side view (axial position against radius, the disc edge-on) and a view along the axis (the blades
turning, the swirl of the slipstream). Time is consistent slow motion: every frame advances the flow
by the time the rotor needs to turn ``degrees_per_frame``, so the particles and the blades move at the
same (slowed) clock. In an MRF case the field is steady and ``U`` is the absolute velocity, so the
swirl the particles show is the swirl the rotor puts into the flow.

The pieces are separate so they can be used (and tested) without OpenFOAM:

- ``RotorView``: centre, diameter, rpm, sense of rotation, blade count (``from_case`` reads the case record)
- ``openfoam_sampler(case)``: a function ``points -> (U, valid)`` over the case's last time step
- ``Tracer``: seeds, advects (Heun steps), re-seeds and keeps short trails
- ``render_frame``: one image (numpy BGR) of the tracer and the rotor
- ``make_movie``: the whole thing to an ``.mp4``

Air or water makes no difference here: the field is whatever the case solved. Needs ``opencv-python-headless``;
reading a case needs ``pyvista`` (and uses ``scipy`` for a fast nearest-cell lookup when it is installed).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

Sampler = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray]"]

ROTOR_TEMPLATES = ("rotor_mrf", "rotor_mrf_static")


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("particle movies need OpenCV: pip install opencv-python-headless") from e
    return cv2


@dataclass(frozen=True)
class RotorView:
    """What the movie needs to know about the rotor: axis +x through ``center`` [m], ``diameter`` [m],
    ``rpm`` and ``rotation`` (+1 / -1 about +x), ``blades`` to draw, ``inflow`` [m/s] (free stream)."""

    center: tuple
    diameter: float
    rpm: float
    rotation: float = 1.0
    blades: int = 2
    inflow: float = 0.0

    def __post_init__(self):
        if self.diameter <= 0 or self.rpm <= 0 or self.blades < 1 or self.rotation not in (1, -1, 1.0, -1.0):
            raise ValueError("diameter and rpm must be > 0, blades >= 1, rotation +1 or -1")

    @property
    def omega(self) -> float:
        """Signed angular speed about +x [rad/s]."""
        return self.rotation * self.rpm * 2 * math.pi / 60

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @classmethod
    def from_case(cls, case, blades: int) -> "RotorView":
        """Read centre, diameter, rpm, rotation and inflow from a prepared rotor case (``aeromant_case.json``)."""
        cdir = Path(case.workdir) if hasattr(case, "workdir") else Path(case)
        rec = cdir / "aeromant_case.json"
        if not rec.is_file():
            raise FileNotFoundError(f"{cdir} is not a prepared Aeromant case (no aeromant_case.json)")
        cfg = json.loads(rec.read_text())["config"]
        if cfg.get("template") not in ROTOR_TEMPLATES:
            raise ValueError(f"particle movies are for rotor cases {ROTOR_TEMPLATES}, not {cfg.get('template')!r}")
        p = cfg["parameters"]
        tip = p["rpm"] * 2 * math.pi / 60 * p["diameter"] / 2
        inflow = p["airspeed"] if "airspeed" in p else p.get("inflow_fraction", 0.0) * tip
        return cls(tuple(float(v) for v in p.get("center", (0.0, 0.0, 0.0))), float(p["diameter"]), float(p["rpm"]),
                   float(p.get("rotation", 1.0)), int(blades), float(inflow))


def time_step(rotor: RotorView, degrees_per_frame: float) -> float:
    """Flow time per frame [s] so that the rotor turns ``degrees_per_frame`` between frames."""
    if degrees_per_frame <= 0:
        raise ValueError("degrees_per_frame must be > 0")
    return degrees_per_frame / (360.0 * rotor.rpm / 60.0)


def openfoam_sampler(case, time: float | None = None) -> Sampler:
    """Velocity sampler over the case's cells (last time step unless ``time``): inverse-distance weights of
    the nearest cell centres. A point is invalid outside the mesh (beyond the domain, inside the body):
    farther from the nearest cell centre than that cell's size."""
    cdir = Path(case.workdir) if hasattr(case, "workdir") else Path(case)
    solved = sorted(t for t in (_time_value(d) for d in cdir.iterdir() if d.is_dir()) if t is not None and t > 0) if cdir.is_dir() else []
    if not solved:                                         # only 0/ (the initial field) or nothing: not run yet
        raise FileNotFoundError(f"no time directories in {cdir}: run the case first")
    import pyvista as pv

    foam = cdir / "case.foam"
    foam.touch(exist_ok=True)
    reader = pv.OpenFOAMReader(str(foam))
    reader.set_active_time_value(float(time) if time is not None else solved[-1])
    reader.cell_to_point_creation = False
    internal = reader.read()["internalMesh"]
    centres = np.asarray(internal.cell_centers().points, dtype=float)
    u = np.asarray(internal.cell_data["U"], dtype=float)
    size = np.cbrt(np.abs(np.asarray(internal.compute_cell_sizes(length=False, area=False)["Volume"], dtype=float)))
    return field_sampler(centres, u, size)


def _time_value(d: Path):
    try:
        return float(d.name)
    except ValueError:
        return None


def field_sampler(centres: np.ndarray, u: np.ndarray, size: np.ndarray, k: int = 4) -> Sampler:
    """Sampler over scattered cell values (``centres`` (N,3), ``u`` (N,3), cell ``size`` (N,)): the shared
    lookup used by ``openfoam_sampler``, usable on any cloud of velocity samples."""
    centres, u, size = np.asarray(centres, float), np.asarray(u, float), np.asarray(size, float)
    k = min(k, len(centres))
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centres)

        def query(pts):
            d, i = tree.query(pts, k=k)
            return d.reshape(len(pts), k), i.reshape(len(pts), k)
    except ImportError:  # pragma: no cover - brute force, fine for small clouds
        def query(pts):
            d2 = ((pts[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
            i = np.argsort(d2, axis=1)[:, :k]
            return np.sqrt(np.take_along_axis(d2, i, 1)), i

    def sample(points):
        pts = np.atleast_2d(np.asarray(points, float))
        d, i = query(pts)
        w = 1.0 / np.maximum(d, 1e-12)
        vel = (u[i] * w[..., None]).sum(1) / w.sum(1)[:, None]
        valid = d[:, 0] <= size[i[:, 0]]
        return vel, valid

    return sample


class Tracer:
    """``n`` particles fed from a region upstream of the disc (x in [-feed_upstream, -0.2] D from the
    centre, r < feed_radius D), advected through ``sampler`` with Heun steps, and re-seeded when they
    leave the view (``downstream`` / ``lateral`` diameters), fall outside the mesh, or stall (slower than
    ``stall_speed`` for more than ``stall_steps`` steps: a dead-water corner would otherwise hold them).
    ``trails`` keeps the last ``trail`` positions of each particle (a re-seeded particle starts a new trail)."""

    def __init__(self, sampler: Sampler, rotor: RotorView, n: int = 40, *, seed: int = 0, trail: int = 12,
                 feed_upstream: float = 1.2, feed_radius: float = 0.6, downstream: float = 3.0, lateral: float = 1.0,
                 stall_speed: float = 0.0, stall_steps: int = 48, fill: bool = True):
        if n < 1 or trail < 1:
            raise ValueError("n and trail must be >= 1")
        self.sampler, self.rotor, self.n = sampler, rotor, n
        self.rng = np.random.default_rng(seed)
        self.c = np.asarray(rotor.center, float)
        self.feed_upstream, self.feed_radius = feed_upstream, feed_radius
        self.downstream, self.lateral = downstream, lateral
        self.stall_speed, self.stall_steps = stall_speed, stall_steps
        # the first frame shows particles over the whole view, later ones come from the feed region
        self.pos = self._seed(n, x_range=(-feed_upstream, downstream) if fill else None)
        self.stalled = np.zeros(n, dtype=int)
        self.speed = np.linalg.norm(self.sampler(self.pos)[0], axis=1)
        self.trails = np.repeat(self.pos[None], trail, axis=0)
        self.reseeded = 0

    def _seed(self, k, x_range=None):
        D = self.rotor.diameter
        lo, hi = x_range or (-self.feed_upstream, -0.2)
        r = self.feed_radius * D * np.sqrt(self.rng.uniform(0, 1, k))
        a = self.rng.uniform(0, 2 * np.pi, k)
        x = self.rng.uniform(lo, hi, k) * D
        return self.c + np.column_stack([x, r * np.cos(a), r * np.sin(a)])

    def step(self, dt: float) -> np.ndarray:
        """Advance by ``dt`` [s]; returns the speed of every particle after the step."""
        u1, ok1 = self.sampler(self.pos)
        u2, ok2 = self.sampler(self.pos + dt * u1)
        new = self.pos + 0.5 * dt * (u1 + np.where(ok2[:, None], u2, u1))
        rel = new - self.c
        D = self.rotor.diameter
        lost = (~ok1 | (rel[:, 0] > self.downstream * D) | (rel[:, 0] < -2.5 * D)
                | (np.hypot(rel[:, 1], rel[:, 2]) > self.lateral * D))
        slow = np.linalg.norm(u1, axis=1) < self.stall_speed
        self.stalled = np.where(slow, self.stalled + 1, 0)
        lost |= self.stalled > self.stall_steps
        if lost.any():
            new[lost] = self._seed(int(lost.sum()))
            self.stalled[lost] = 0
            self.reseeded += int(lost.sum())
        self.trails = np.roll(self.trails, -1, axis=0)
        self.trails[-1] = new
        self.trails[:, lost] = new[lost]
        self.pos = new
        self.speed = np.linalg.norm(self.sampler(new)[0], axis=1)
        return self.speed


def _colour(values, vmax):
    cv2 = _cv2()
    idx = np.clip(np.asarray(values, float) / max(vmax, 1e-12) * 255, 0, 255).astype(np.uint8).reshape(-1, 1)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(idx, cmap).reshape(-1, 3)


def render_frame(tracer: Tracer, angle: float, *, size=(1280, 540), speed_max: float | None = None,
                 title: str = "", time_s: float | None = None, background=(250, 250, 250)) -> np.ndarray:
    """One frame (H, W, 3 BGR uint8): side view on the left (x downstream to the right, radius up; the
    particles nearer the viewer drawn larger), the view along the axis on the right (looking upstream
    from behind: y to the right, z up), blades at ``angle`` [rad] about +x, particles and trails coloured
    by speed."""
    cv2 = _cv2()
    W, H = size
    img = np.full((H, W, 3), background, dtype=np.uint8)
    rot, D, R = tracer.rotor, tracer.rotor.diameter, tracer.rotor.radius
    c = tracer.c
    vmax = speed_max or float(np.percentile(tracer.speed, 98)) or 1.0
    top, bottom = 40, 60
    w_side = int(W * 0.66)
    # --- side view: x in [-feed-0.1, downstream] D, y in [-lateral, lateral] D
    x0, x1 = -(tracer.feed_upstream + 0.1) * D, tracer.downstream * D
    y0, y1 = -tracer.lateral * D, tracer.lateral * D
    s1 = min((w_side - 40) / (x1 - x0), (H - top - bottom) / (y1 - y0))
    ox, oy = 20 - x0 * s1, top + (H - top - bottom) / 2

    def side(p):
        return np.column_stack([ox + (p[:, 0] - c[0]) * s1, oy - (p[:, 1] - c[1]) * s1]).astype(np.int32)

    # --- axial view: y, z in [-lateral, lateral] D
    s2 = min((W - w_side - 40) / (y1 - y0), (H - top - bottom) / (y1 - y0))
    ax_, ay_ = w_side + (W - w_side) / 2, oy

    def axial(p):
        return np.column_stack([ax_ + (p[:, 1] - c[1]) * s2, ay_ - (p[:, 2] - c[2]) * s2]).astype(np.int32)

    grey = (170, 170, 170)
    cv2.line(img, (int(ox), int(oy)), (w_side - 10, int(oy)), grey, 1, cv2.LINE_AA)                  # axis
    cv2.circle(img, (int(ax_), int(ay_)), int(R * s2), grey, 1, cv2.LINE_AA)                          # disc
    cv2.line(img, (w_side, top - 10), (w_side, H - bottom + 10), (220, 220, 220), 1)
    # blades: direction (cos, sin) in (y, z)
    hub = 0.12 * R
    for k in range(rot.blades):
        phi = angle + 2 * math.pi * k / rot.blades
        d = np.array([0.0, math.cos(phi), math.sin(phi)])
        root, tip = c + hub * d, c + R * d
        a, b = axial(np.array([root, tip]))
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (60, 60, 60), max(2, int(0.06 * D * s2)), cv2.LINE_AA)
        a, b = side(np.array([root, tip]))
        depth = math.sin(phi)                                                                          # z: towards the viewer
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (60, 60, 60) if depth >= 0 else (150, 150, 150),
                 max(2, int(0.03 * D * s1)), cv2.LINE_AA)
    cv2.circle(img, (int(ax_), int(ay_)), max(2, int(hub * s2)), (60, 60, 60), -1, cv2.LINE_AA)
    # trails, then particles (far ones first in the side view)
    cols = _colour(tracer.speed, vmax)
    L = len(tracer.trails)
    for j in range(tracer.n):
        tr = tracer.trails[:, j]
        col = tuple(int(v) for v in cols[j])
        for view in (side, axial):
            pts = view(tr)
            for i in range(1, L):
                if (pts[i] != pts[i - 1]).any():
                    fade = i / L
                    cc = tuple(int(bg + (v - bg) * fade) for v, bg in zip(col, background))
                    cv2.line(img, tuple(map(int, pts[i - 1])), tuple(map(int, pts[i])), cc, 1, cv2.LINE_AA)
    order = np.argsort(tracer.pos[:, 2] - c[2])
    ps, pa = side(tracer.pos), axial(tracer.pos)
    for j in order:
        col = tuple(int(v) for v in cols[j])
        z = (tracer.pos[j, 2] - c[2]) / (tracer.lateral * D)
        r_px = int(np.clip(4 + 2.5 * z, 2, 7))
        cv2.circle(img, tuple(map(int, ps[j])), r_px, col, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, pa[j])), 4, col, -1, cv2.LINE_AA)
    # labels and a speed bar
    txt = (30, 30, 30)
    cv2.putText(img, title or "tracer particles in the converged flow", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt, 1, cv2.LINE_AA)
    info = f"{rot.rpm:.0f} rpm, {rot.blades} blades, D {D * 1000:.0f} mm" + (f", t = {time_s * 1000:.1f} ms of flow" if time_s is not None else "")
    cv2.putText(img, info, (w_side + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt, 1, cv2.LINE_AA)
    cv2.putText(img, "side view: flow left to right", (12, H - bottom + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt, 1, cv2.LINE_AA)
    cv2.putText(img, "along the axis (from behind)", (w_side + 10, H - bottom + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt, 1, cv2.LINE_AA)
    bx, by, bw = 12, H - 22, 220
    bar = _colour(np.linspace(0, vmax, bw), vmax)
    for i in range(bw):
        cv2.line(img, (bx + i, by), (bx + i, by + 10), tuple(int(v) for v in bar[i]), 1)
    cv2.putText(img, f"|U| 0 .. {vmax:.3g} m/s", (bx + bw + 8, by + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt, 1, cv2.LINE_AA)
    return img


def make_movie(case, path, *, blades: int = 2, rotor: RotorView | None = None, sampler: Sampler | None = None,
               n: int = 40, seconds: float = 10.0, fps: int = 24, degrees_per_frame: float = 10.0, trail: int = 12,
               size=(1280, 540), seed: int = 0, title: str = "", speed_max: float | None = None) -> Path:
    """Write an ``.mp4`` of ``n`` tracer particles through a solved rotor case and return its path.

    ``case`` is a ``CFDCase`` or its directory (it provides the rotor record and the field); ``rotor`` and
    ``sampler`` replace what would be read from it (e.g. a stub field in a test, ``case`` may then be None).
    Each frame advances the flow by the time the rotor needs to turn ``degrees_per_frame``."""
    cv2 = _cv2()
    rotor = rotor or RotorView.from_case(case, blades)
    sampler = sampler or openfoam_sampler(case)
    dt = time_step(rotor, degrees_per_frame)
    frames = max(2, int(round(seconds * fps)))
    D, c = rotor.diameter, np.asarray(rotor.center, float)
    gx, gy = np.meshgrid(np.linspace(-1.0, 3.0, 41) * D, np.linspace(-0.8, 0.8, 17) * D)
    u, ok = sampler(c + np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]))   # the field around the disc
    sp = np.linalg.norm(u[ok], axis=1) if ok.any() else np.array([rotor.inflow or 1.0])
    speed_max = speed_max or float(np.percentile(sp, 98)) or 1.0              # one colour scale for the whole movie
    tracer = Tracer(sampler, rotor, n, seed=seed, trail=trail, stall_speed=0.02 * speed_max, stall_steps=2 * fps)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(size))
    if not out.isOpened():  # pragma: no cover
        raise RuntimeError(f"OpenCV could not open {path} for writing")
    try:
        for k in range(frames):
            t = k * dt
            out.write(render_frame(tracer, rotor.omega * t, size=size, speed_max=speed_max, title=title, time_s=t))
            tracer.step(dt)
    finally:
        out.release()
    return path
