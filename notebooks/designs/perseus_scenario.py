"""The Perseus rovers' branch-cutting job in ChironLab (notebook 37 §7, notebook 38 §7, ``scenarios/perseus_branches.py``
and ``perseus_mini_garden.py``): a tree (or a garden shrub) on uneven ground, branches to cut.

* **ground** — ``chiron.Rough`` soil (RMS and correlation length of the scene) with a cross-slope band and a few
  hummocks (``terrain``): a heightfield MuJoCo rolls the tracks over;
* **the tree** — a fixed trunk prop (a cylinder) and its branches: each a free capsule prop of green wood
  (``cutting.WOODS``: density, friction) held to the trunk by a ``Weld`` at its base;
* **the cut** — ``SawCutter``, a scene hook: every control step it reads the contact force of the robot's bar geom
  on each branch (``lab.contact_force``); while the bar presses with a feed in ``[F_min, F_max]`` the cut area grows
  at the saw's rate ``cutting.cut_rate_area`` (P_cut / (e_c · kerf)); when it reaches the branch's cross-section the
  weld is released (``lab.set_weld``) and the branch falls; a push above ``F_jam`` is a **jam** (the controller backs
  off while it lasts). Whether the branch parts is decided by the contact forces MuJoCo computes from the boom's
  servos and the cutting model's numbers.

The robot only knows where the trunk stands and where each branch leaves it (``Scene``); the mission is
``perseus_controller.branch_job``. ``run(lab, scene)`` returns the Episode with the mission's and the scene's
events; ``timeseries`` and ``phase_table`` give the numbers.

    import perseus_scenario as pss
    scene = pss.Scene(); lab = pss.make_lab(scene); ep = pss.run(lab, scene)
    scene = pss.garden_scene()                     # the Mini's shrub on a lawn bank
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vegeta import chiron as ch

import cutting
import perseus_controller as pcc
import perseus_robot as pbr

__all__ = ["Branch", "Scene", "SawCutter", "garden_scene", "terrain", "props", "make_lab", "run", "timeseries",
           "phase_table", "FEED_WINDOW"]

#: The feed window the cutter counts as engaged, as fractions of the saw's nominal feed force (``cutting.feed_force``
#: at the variant's ``k_feed``): below ``min`` the chain only scratches, above ``jam`` the bar is forced and binds.
FEED_WINDOW = {"min": 0.15, "max": 2.5, "jam": 4.0}


@dataclass
class Branch:
    """A branch leaving the trunk at height ``z`` [m] in direction ``azimuth_deg`` (0 = +x, towards the rover when
    it approaches from −x) and ``elevation_deg`` above horizontal, diameter ``d`` and length ``length`` [m], of
    ``wood`` (``cutting.WOODS`` key)."""

    name: str
    z: float
    azimuth_deg: float
    elevation_deg: float
    d: float
    length: float
    wood: str = "softwood (spruce, pine), green"
    base: tuple = (0.0, 0.0, 0.0)                     # set by the scene: where it leaves the trunk (world)

    @property
    def axis(self) -> tuple:
        a, e = math.radians(self.azimuth_deg), math.radians(self.elevation_deg)
        return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))

    def cut_point(self, s: float) -> tuple:
        """The point on the branch's axis ``s`` [m] from the trunk (clipped to the branch)."""
        s = min(max(s, 0.0), self.length)
        return tuple(b + s * a for b, a in zip(self.base, self.axis))

    @property
    def geom(self) -> str:
        return f"{self.name}_g"

    @property
    def weld(self) -> str:
        return f"{self.name} hold"

    @property
    def mass_kg(self) -> float:
        return cutting.branch_mass(self.d, self.length, self.wood)


@dataclass
class Scene:
    """Where things are (world, m): the trunk at ``x_tree`` on the centre line (radius ``r_trunk``, height
    ``h_trunk``), its branches (they leave the trunk sideways — azimuths near ±90° — so the bar, pointing ahead
    from the rover, crosses them), the rover's stand-off from the trunk (``stand_dist``: hull centre to trunk; None
    = found by ``perseus_controller.stand_distance``), the cut ``cut_s`` [m] out from the trunk, where the machine
    backs out to when done (``x_end``, behind it: the trunk is ahead); the ground: rough soil with a cross-slope
    band at ``bank``=(x0, x1, deg), hummocks, and an ``even_patch`` (x0, x1, factor) where the operator parks the
    machine (the roughness scaled down there: a rover on two rollers rocks and walks while its boom swings)."""

    x_tree: float = 6.5
    r_trunk: float = 0.18
    h_trunk: float = 3.2
    branches: list = field(default_factory=lambda: [
        Branch("branch_1", 1.20, 100.0, 12.0, 0.080, 1.6),
        Branch("branch_2", 1.50, -80.0, 15.0, 0.110, 1.8),
        Branch("branch_3", 1.10, 125.0, 5.0, 0.060, 1.4),
    ])
    stand_dist: float | None = None
    cut_s: float = 0.40
    cut_timeout: float = 40.0
    x_end: float = 1.0                                 # back out the way it came (the trunk is ahead)
    rough_rms: float = 0.03
    rough_len: float = 0.4
    bank: tuple = (2.0, 4.5, 8.0)                      # a side slope band: x0, x1, deg (0 = none)
    hummocks: tuple = ((3.2, 0.5, 0.12, 0.6), (1.4, -0.6, 0.10, 0.5))   # (x, y, height, radius)
    even_patch: tuple = (4.3, 6.4, 0.3)                # where the machine parks: x0, x1, roughness × factor
    seed: int = 5
    course_extent: tuple = (-2.0, 13.0, -4.0, 4.0)
    name: str = "a tree on rough ground"

    def __post_init__(self):
        for b in self.branches:
            b.base = (self.x_tree + self.r_trunk * 0.9 * math.cos(math.radians(b.azimuth_deg)),
                      self.r_trunk * 0.9 * math.sin(math.radians(b.azimuth_deg)), b.z)

    @property
    def trunk_base(self) -> tuple:
        return (self.x_tree, 0.0, 0.0)

    def ground(self, x, y) -> float:
        return float(terrain(self).height(x, y))


def garden_scene() -> Scene:
    """The Mini's job: a shrub (a 90 mm stem) on a lawn with a 12° bank, five branches of 25–45 mm, a kerb step."""
    return Scene(x_tree=4.5, r_trunk=0.045, h_trunk=1.8,
                 branches=[Branch("branch_1", 0.50, 95.0, 20.0, 0.030, 0.9, "garden shrub (hazel, privet, fruit), green"),
                           Branch("branch_2", 0.65, -85.0, 15.0, 0.045, 1.1, "garden shrub (hazel, privet, fruit), green"),
                           Branch("branch_3", 0.70, 110.0, 20.0, 0.025, 0.8, "garden shrub (hazel, privet, fruit), green"),
                           Branch("branch_4", 0.58, -110.0, 10.0, 0.040, 1.0, "garden shrub (hazel, privet, fruit), green"),
                           Branch("branch_5", 0.68, 75.0, 25.0, 0.028, 0.7, "garden shrub (hazel, privet, fruit), green")],
                 cut_s=0.25, cut_timeout=20.0, x_end=0.8, rough_rms=0.012, rough_len=0.25, bank=(1.5, 3.0, 12.0),
                 hummocks=((2.2, 0.3, 0.05, 0.35),), even_patch=(3.3, 4.6, 0.5), seed=11,
                 course_extent=(-1.5, 9.0, -3.0, 3.0),
                 name="a shrub on a lawn bank")


# ----------------------------------------------------------------------------------------------- ground and props
def terrain(scene: Scene):
    """Rough soil + a side-slope band (a smooth ramp in y) + hummocks, as a ``chiron.Custom``; flat when all zero."""
    if scene.rough_rms <= 0 and not scene.bank[2] and not scene.hummocks:
        return ch.Flat()
    x0, x1, y0, y1 = scene.course_extent
    rough = ch.Rough(scene.rough_rms, scene.rough_len, start=0.5, seed=scene.seed, extent=(x0, x1, y0, y1), cell=0.05)
    bx0, bx1, bdeg = scene.bank
    slope = math.tan(math.radians(bdeg))
    hums = scene.hummocks
    ex0, ex1, efac = scene.even_patch

    def ground(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        z = rough.height(x, y)
        if efac < 1.0:
            inside = np.clip((x - ex0) / 0.4, 0.0, 1.0) * np.clip((ex1 - x) / 0.4, 0.0, 1.0)
            z = z * (1.0 - (1.0 - efac) * inside)
        if bdeg:
            w = np.clip((x - bx0) / 0.6, 0.0, 1.0) * np.clip((bx1 - x) / 0.6, 0.0, 1.0)   # fades in and out over 0.6 m
            z = z + w * slope * y
        for hx, hy, hh, hr in hums:
            r2 = (x - hx) ** 2 + (y - hy) ** 2
            z = z + hh * np.exp(-r2 / (2 * (hr / 2) ** 2))
        return z
    return ch.Custom(ground, name=scene.name)


def props(scene: Scene) -> tuple:
    """The trunk (fixed) and the branches (free capsules of green wood) each welded to it."""
    out = [ch.Prop(ch.Link("trunk", pos=scene.trunk_base,
                           geoms=[ch.Geom("trunk_g", "cylinder", (scene.r_trunk, scene.h_trunk / 2), pos=(0, 0, scene.h_trunk / 2),
                                          friction=(0.5, 0.005, 0.0001), rgba=(0.42, 0.30, 0.18, 1.0))]), log=False)]
    welds = []
    for b in scene.branches:
        w = cutting.get_wood(b.wood)
        r = b.d / 2
        tip = tuple(b.length * a for a in b.axis)
        # a stiff contact for a thin bar pressing on a light branch (as the Manus wire): sub-mm penetration, a
        # crisp normal force to read; condim 3 (the normal is what the cutter reads)
        out.append(ch.Prop(ch.Link(b.name, pos=b.base,
                                   geoms=[ch.Geom(b.geom, "capsule", (r,), fromto=(0, 0, 0, *tip), mass=b.mass_kg,
                                                  friction=(w.mu_steel, 0.005, 0.0001), rgba=(0.36, 0.45, 0.2, 1.0))]),
                           free=True, condim=3, solref=(0.002, 1.0), solimp=(0.95, 0.99, 0.001)))
        welds.append(ch.Weld(b.weld, b.name, "trunk", solref=(0.005, 1.0)))
    return out, welds


# ----------------------------------------------------------------------------------------------- the cutter
class SawCutter:
    """Scene hook: a branch parts (its weld is released) once the bar has removed its cross-section.

    ``rate`` [m²/s] is the saw's ``cutting.cut_rate_area`` on each branch's wood; the bar is **engaged** while its
    normal contact force on the branch (low-passed over ``tau_filter``: a rigid contact chatters at the control
    rate) is within the feed window (fractions of the saw's nominal feed force), a force above the jam threshold is
    a **jam** (logged; ``jammed`` until the force is back in the window). ``history`` keeps [t, F_1, area_1, F_2,
    area_2, ...] (the filtered force, one pair per branch) from the first contact on; ``cut_at`` the times."""

    def __init__(self, scene: Scene, saw: cutting.Saw, k_feed: float = 1.5, bar_geom: str = "saw_bar",
                 window: dict | None = None, tau_filter: float = 0.02):
        self.scene, self.saw, self.bar = scene, saw, bar_geom
        self.tau_filter = float(tau_filter)                       # s: the force reading low-passed (contact chatter)
        self.branches = list(scene.branches)
        f_feed = cutting.feed_force(saw, None, k_feed)
        win = dict(FEED_WINDOW if window is None else window)
        self.F_min, self.F_max, self.F_jam = win["min"] * f_feed, win["max"] * f_feed, win["jam"] * f_feed
        self.rate = {b.name: cutting.cut_rate_area(saw, b.wood) for b in self.branches}
        self.area_full = {b.name: float(cutting.cut_area(b.d)) for b in self.branches}

    def reset(self, lab):
        self.area = {b.name: 0.0 for b in self.branches}
        self.cut_at, self.jammed, self.peak, self.history = {}, set(), 0.0, []
        self.engaged_s = {b.name: 0.0 for b in self.branches}
        self.force = {b.name: 0.0 for b in self.branches}            # the filtered reading (the controller's feed loop)
        self.raw = {b.name: 0.0 for b in self.branches}

    def __call__(self, lab):
        row = [lab.time]
        touched = False
        for b in self.branches:
            if b.name in self.cut_at:
                row += [0.0, self.area[b.name]]
                continue
            raw = lab.contact_force(self.bar, b.geom)[1]
            self.raw[b.name] = raw
            a = min(1.0, lab.control_dt / self.tau_filter)
            fn = self.force[b.name] + a * (raw - self.force[b.name])
            self.force[b.name] = fn
            touched = touched or raw > 0.5
            self.peak = max(self.peak, raw)
            if fn > self.F_jam and b.name not in self.jammed:
                self.jammed.add(b.name)
                lab.log_event("saw", f"jam in {b.name}: {fn:.0f} N > {self.F_jam:.0f} N")
            elif b.name in self.jammed and fn < self.F_max:               # hysteresis: freed once back in the window
                self.jammed.discard(b.name)
                lab.log_event("saw", f"{b.name}: freed at {fn:.0f} N")
            if b.name not in self.jammed:
                if self.F_min <= fn <= self.F_max:
                    self.area[b.name] = min(self.area_full[b.name], self.area[b.name] + self.rate[b.name] * lab.control_dt)
                    self.engaged_s[b.name] += lab.control_dt
                    if self.area[b.name] >= self.area_full[b.name] - 1e-12:
                        lab.set_weld(b.weld, False)
                        self.cut_at[b.name] = lab.time
                        lab.log_event("saw", f"cut {b.name} (Ø {b.d * 1e3:.0f} mm) at {lab.time:.2f} s after "
                                             f"{self.engaged_s[b.name]:.1f} s of feed, peak {self.peak:.0f} N")
            row += [fn, self.area[b.name]]
        if touched or self.history:
            self.history.append(row)

    @property
    def done(self) -> bool:
        return len(self.cut_at) == len(self.branches)


# ----------------------------------------------------------------------------------------------- the run
def make_lab(scene: Scene, robot=None, v: pbr.Variant = pbr.PERSEUS, **kwargs):
    """The machine on the scene's ground with the tree, the welds and the saw-cutter hook (``lab.cutter``)."""
    robot = pbr.perseus(v=v) if robot is None else robot
    v = robot.variant
    pr, welds = props(scene)
    lab = pbr.perseus_lab(terrain(scene), robot=robot, v=v, props=pr, welds=welds, course_extent=scene.course_extent,
                          **kwargs)
    lab.cutter = SawCutter(scene, robot.saw, k_feed=v.k_feed)
    lab.add_hook(lab.cutter)
    return lab


def run(lab, scene: Scene, duration: float = 90.0, seed: int = 0, **mission_kw):
    """The whole job; ``ep.log['mission']`` the controller's phase log, ``ep.log['events']`` the scene's, the
    cutter's history and the cut times."""
    mission = pcc.CutMission(pcc.branch_job(scene, lab.robot, **mission_kw), name="cut the branches")
    ep = lab.run(mission, duration=duration, rules=None, settle=0.5, seed=seed,
                 info={"controller": mission.name, "treatment": lab.robot.name})
    ep.log["mission"] = [list(x) for x in mission.log]
    ep.log["mission_finished"] = mission.finished
    n = len(scene.branches)
    ep.log["cutter_history"] = np.asarray(lab.cutter.history, dtype=float).reshape(-1, 1 + 2 * n)
    ep.log["cut_at"] = dict(lab.cutter.cut_at)
    ep.log["branches"] = [b.name for b in scene.branches]
    return ep


def timeseries(ep) -> pd.DataFrame:
    """Per log sample: hull x, y, speed, tilt, each branch's base height, the arm joint torques and angles."""
    log = ep.log
    names = list(log["joints"])
    tau, q = np.asarray(log["tau"]), np.asarray(log["q"])
    props_ = list(log["props"])
    pp = np.asarray(log["prop_pos"])
    bq = np.asarray(log["body_quat"])[:, 0]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (bq[:, 1] ** 2 + bq[:, 2] ** 2), -1, 1)))
    pos = np.asarray(log["body_pos"])[:, 0]
    df = pd.DataFrame({"t": log["t"], "x": pos[:, 0], "y": pos[:, 1], "v": np.asarray(log["com_vel"])[:, 0], "tilt_deg": tilt})
    for b in log.get("branches", []):
        if b in props_:
            df[f"{b}_z"] = pp[:, props_.index(b), 2]
    for j in pbr.ARM_JOINTS:
        df[f"tau_{j}"] = tau[:, names.index(j)]
        df[f"q_{j}_deg"] = np.degrees(q[:, names.index(j)])
    return df


def phase_table(ep, robot=None) -> pd.DataFrame:
    """Per mission phase: start, duration, hull x, and the peak |τ| of each arm joint against its stall (a cylinder
    drive's at its smallest lever)."""
    log = ep.log
    ts = timeseries(ep)
    starts = [(t, name) for t, name, note in log["mission"] if note == "start"]
    v = robot.variant if robot is not None else pbr.PERSEUS
    stall = {f"arm_{j}": pbr.joint_limits(j, None, None, v)["stall_Nm"] for j in ("yaw", "shoulder", "elbow", "wrist")}
    rows = {}
    for k, (t0, name) in enumerate(starts):
        t1 = starts[k + 1][0] if k + 1 < len(starts) else float(ts.t.iloc[-1])
        s = ts[(ts.t >= t0) & (ts.t <= t1)]
        row = {"start_s": t0, "duration_s": t1 - t0, "hull_x_m": float(s.x.iloc[-1]) if len(s) else float("nan")}
        for j in pbr.ARM_JOINTS:
            peak = float(s[f"tau_{j}"].abs().max()) if len(s) else float("nan")
            row[f"{j} / stall"] = peak / stall[j]
        rows[f"{k + 1:02d} {name}"] = row
    return pd.DataFrame(rows).T
