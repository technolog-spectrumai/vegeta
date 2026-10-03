"""The Onager Sweeper's street in ChironLab (notebook 23 §7, ``scenarios/onager_sweeper_street.py``): litter in
the sweeping path, a brick and a box the broom cannot take.

* **street** — asphalt (RMS 2 mm roughness), 25 m long;
* **litter** — crushed cans and gum packets (``onager_sweeper_cfd.DEBRIS``: 15 g and 10 g) scattered in the
  hood's path: free props with mass-scaled hard contacts (light things under a 590 kg machine). The **suction** is
  the ``Vacuum`` scene hook: under the hood and in a band around its lips it applies the air drag of the field the
  CFD gave (``onager_sweeper_cfd.SUCTION``: the inflow at the lips, the upward speed in the hood) to each piece —
  ½ ρ C_d A |u − v| (u − v) — and a piece that reaches the duct is collected (parked out of the world, an event);
* **brick** — a 2.3 kg clay brick (230 × 110 × 70 mm) lying across the street in the left arm's lane: the obstacle,
  the left pincer takes it over its 110 mm width (the jaws close along the arm, so the object must lie across it);
* **box** — a 0.6 kg cardboard box (300 × 200 × 150 mm) across the right arm's lane: the rubbish the right pincer
  takes over its 200 mm side;
* the **basket** on the roof receives them: physics decides whether they stay in the jaws and land in it.

``run(lab, scene)`` returns the Episode with the mission's and the scene's events; ``timeseries``, ``phase_table``
and ``collected`` give the numbers.

Promoted from ``notebooks/designs/onager_sweeper_scenario.py`` as it was proven there; the notebook copy may move on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vegeta import chiron as ch

from . import onager_manus_robot as omr
from . import onager_sweeper_cfd as cfd
from . import onager_sweeper_controller as osc
from . import onager_sweeper_robot as osr

__all__ = ["Scene", "Vacuum", "OBJECTS", "LITTER_CONTACT", "props", "terrain", "make_lab", "run", "timeseries",
           "phase_table", "collected"]

#: The pick-ups (inputs): size [m] (x, y, z as they lie), mass [kg], friction on the jaws and the road.
OBJECTS = {"brick": {"size": (0.23, 0.11, 0.07), "mass": 2.3, "mu": 0.6, "rgba": (0.6, 0.3, 0.22, 1.0)},
           "box": {"size": (0.30, 0.20, 0.15), "mass": 0.6, "mu": 0.5, "rgba": (0.72, 0.6, 0.42, 1.0)}}
#: Litter as MuJoCo props: a crushed can and a gum packet (boxes of the DEBRIS sizes), their drag class.
LITTER = {"can": {"kind": "crushed can", "half": (0.05, 0.033, 0.015), "rgba": (0.8, 0.1, 0.1, 1.0)},
          "packet": {"kind": "gum packet", "half": (0.035, 0.01, 0.005), "rgba": (0.1, 0.5, 0.2, 1.0)}}
#: Light props under heavy jaws and a heavy hood: a hard contact at the time-constant floor (as the Manus's wire).
LITTER_CONTACT = {"solref": (0.001, 1.0), "solimp": (0.99, 0.999, 0.001)}


@dataclass
class Scene:
    """Where things are (world, m). ``pickups``: (prop name, kind in ``OBJECTS``, the arm that takes it)."""

    x_brick: float = 6.0
    x_box: float = 13.0
    x_end: float = 20.0
    litter: list = field(default_factory=lambda: [("can", 3.0, 0.05), ("packet", 4.0, -0.08), ("can", 8.5, -0.06),
                                                  ("packet", 9.5, 0.1), ("can", 10.5, 0.0), ("packet", 15.5, -0.05),
                                                  ("can", 16.5, 0.08), ("packet", 17.5, 0.0)])
    pickups: list = field(default_factory=lambda: [("brick", "brick", "L"), ("box", "box", "R")])
    rough_rms: float = 0.002
    seed: int = 5
    capture_band: float = 0.12               # m outside the lips where the inflow still pulls (the CFD's field decays over ~a gap width)

    y_lane: float = 0.30                     # the pick-ups lie in line with the arm that takes them (the pedestals' y)

    def position(self, name: str) -> np.ndarray:
        x = {"brick": self.x_brick, "box": self.x_box}[name]
        side = next(sd for nm, _, sd in self.pickups if nm == name)
        y = omr.SIDES[side] * self.y_lane
        return np.array([x, y, self.ground(x, y)])

    def top_of(self, name: str) -> float:
        """The object's top over its prop frame (the frame sits on the road under its centre)."""
        return OBJECTS[name]["size"][2]

    def ground(self, x, y) -> float:
        return float(terrain(self).height(x, y))


def terrain(scene: Scene):
    if scene.rough_rms <= 0:
        return ch.Flat()
    return ch.Rough(scene.rough_rms, 0.3, start=1.0, seed=scene.seed, extent=(-3.0, 28.0, -3.0, 3.0), cell=0.05)


def props(scene: Scene) -> list:
    """The pick-ups and the litter, every one a free prop on the road."""
    out = []
    for name, kind, _ in scene.pickups:
        o = OBJECTS[kind]
        sx, sy, sz = o["size"]
        x, y, z = scene.position(name)
        q90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))        # the long side across the street
        out.append(ch.Prop(ch.Link(name, pos=(x, y, z + 0.002), quat=q90,
                                   geoms=[ch.Geom(f"{name}_g", "box", (sx / 2, sy / 2, sz / 2), pos=(0, 0, sz / 2), mass=o["mass"],
                                                  friction=(o["mu"], 0.005, 0.0001), rgba=o["rgba"])]), free=True))
    for k, (kind, x, y) in enumerate(scene.litter):
        lt = LITTER[kind]
        d = cfd.DEBRIS[lt["kind"]]
        hx, hy, hz = lt["half"]
        out.append(ch.Prop(ch.Link(f"{kind}_{k}", pos=(x, y, scene.ground(x, y) + hz + 0.002),
                                   geoms=[ch.Geom(f"{kind}_{k}_g", "box", (hx, hy, hz), mass=d["mass"],
                                                  friction=(d["mu"], 0.005, 0.0001), rgba=lt["rgba"])]),
                           free=True, solref=LITTER_CONTACT["solref"], solimp=LITTER_CONTACT["solimp"]))
    return out


class Vacuum:
    """Scene hook: the fan's air drag on the litter under the hood (``onager_sweeper_cfd.SUCTION``; ``suction``
    overrides it). In the hood's footprint the air rises at ``mouth_velocity`` (``duct_velocity`` on the duct's
    axis, a quadratic blend over 0.12 m: the field converges on the duct) and converges on it horizontally; in a band
    of ``capture_band`` around the lips it flows in at ``gap_velocity × gap / (gap + d)``. Litter that reaches the
    duct is collected: parked out of the world, logged (``collected``: name -> time). Runs only while ``lab.fan_on``."""

    def __init__(self, scene: Scene, litter: list, suction: dict | None = None, rho: float = cfd.AIR["density"]):
        self.scene, self.litter, self.rho = scene, list(litter), rho
        self.s = dict(cfd.SUCTION if suction is None else suction)

    def reset(self, lab):
        import mujoco

        m = lab.model
        self.collected = {}
        self.hull = lab._body_id("hull")
        sg = lab.robot.sweeper
        self.hood = sg["hood"]
        self.gap = sg["hood_world"][2]
        self.pieces = []
        for name, kind in self.litter:
            b = lab._body_id(name)
            j = int(m.body_jntadr[b])
            d = cfd.DEBRIS[kind]
            self.pieces.append((name, b, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), 0.5 * self.rho * d["cd"] * d["area"]))
        self.n_parked = 0
        self._mujoco = mujoco

    def __call__(self, lab):
        if not getattr(lab, "fan_on", False):
            return
        d = lab.data
        R = d.xmat[self.hull].reshape(3, 3)
        hull_pos = d.xpos[self.hull]
        x0, y0, z0, x1, y1, z1 = self.hood
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, u_gap, w_duct = self.s["mouth_velocity_m_s"], self.s["gap_velocity_m_s"], self.s["duct_velocity_m_s"]
        r_duct = 0.12                                                     # m: the duct's inflow region on the hood floor
        for name, b, qadr, dadr, k in self.pieces:
            if name in self.collected:
                continue
            p = R.T @ (d.xpos[b] - hull_pos)                              # the piece in the hull frame
            if p[2] > z1 + 0.05:
                continue
            inside = x0 < p[0] < x1 and y0 < p[1] < y1
            if inside:
                if p[2] > z1 - 0.04:                                      # at the roof: into the duct, collected
                    self.collected[name] = float(lab.time)
                    self.n_parked += 1
                    d.qpos[qadr:qadr + 3] = (-30.0 - 0.3 * self.n_parked, 6.0, 0.05)
                    d.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
                    d.qvel[dadr:dadr + 6] = 0.0
                    lab.body_force(name, None)
                    lab.log_event("vacuum", f"{name} collected")
                    continue
                to_duct = np.array([cx - p[0], cy - p[1], 0.0])
                n = np.linalg.norm(to_duct)
                # the flow converges on the duct: the duct's own speed on its axis, the hood's mean away from it
                w_here = w + (w_duct - w) * max(0.0, 1.0 - n / r_duct) ** 2
                u = np.array([0.0, 0.0, w_here]) + (0.5 * w / n) * to_duct if n > 1e-6 else np.array([0.0, 0.0, w_here])
            else:
                dx = max(x0 - p[0], 0.0, p[0] - x1)
                dy = max(y0 - p[1], 0.0, p[1] - y1)
                dist = math.hypot(dx, dy)
                if dist > self.scene.capture_band:
                    lab.body_force(name, None)
                    continue
                target = np.array([min(max(p[0], x0), x1), min(max(p[1], y0), y1), 0.0])
                to_lips = target - np.array([p[0], p[1], 0.0])
                n = np.linalg.norm(to_lips)
                u = (u_gap * self.gap / (self.gap + dist)) * to_lips / n if n > 1e-6 else np.zeros(3)
            u_world = R @ u
            v = d.qvel[dadr:dadr + 3]
            rel = u_world - v
            f = k * np.linalg.norm(rel) * rel
            lab.body_force(name, f)


def make_lab(scene: Scene, robot=None, suction: dict | None = None, **kwargs):
    """The Sweeper on the street with its props and the vacuum hook (``lab.vacuum``)."""
    pr = props(scene)
    lab = osr.sweeper_lab(terrain(scene), robot=robot, props=pr, course_extent=(-3.0, 28.0, -3.0, 3.0), **kwargs)
    litter = [(f"{kind}_{k}", LITTER[kind]["kind"]) for k, (kind, _, _) in enumerate(scene.litter)]
    lab.vacuum = Vacuum(scene, litter, suction)
    lab.add_hook(lab.vacuum)
    return lab


def run(lab, scene: Scene, duration: float = 90.0, seed: int = 0, **mission_kw):
    """The whole mission; ``ep.log['mission']`` the controller's phase log, ``ep.log['events']`` the scene's."""
    mission = osc.Mission(osc.sweep_and_clear(scene, **mission_kw), name="sweep the street, clear the brick and the box")
    ep = lab.run(mission, duration=duration, rules=None, settle=0.5, seed=seed,
                 info={"controller": mission.name, "treatment": "sweeper"})
    ep.log["mission"] = [list(x) for x in mission.log]
    ep.log["mission_finished"] = mission.i >= len(mission.phases)
    ep.log["collected"] = dict(lab.vacuum.collected)
    return ep


def collected(ep) -> pd.DataFrame:
    """Which litter the vacuum took and when."""
    c = ep.log.get("collected", {})
    return pd.DataFrame({"collected at [s]": c}).sort_values("collected at [s]") if c else pd.DataFrame(columns=["collected at [s]"])


def timeseries(ep) -> pd.DataFrame:
    """Per log sample: hull x and speed, tilt, the pick-ups' positions (world), the broom's speed, arm and jaw
    torques, the broom torque."""
    log = ep.log
    names = list(log["joints"])
    tau, qd = np.asarray(log["tau"]), np.asarray(log["qd"])
    props_ = list(log["props"])
    pp = np.asarray(log["prop_pos"])
    q = np.asarray(log["body_quat"])[:, 0]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
    df = pd.DataFrame({"t": log["t"], "x": np.asarray(log["body_pos"])[:, 0, 0], "v": np.asarray(log["com_vel"])[:, 0],
                       "tilt_deg": tilt, "broom_rpm": qd[:, names.index("broom")] * 60 / (2 * np.pi),
                       "tau_broom": tau[:, names.index("broom")]})
    for name in ("brick", "box"):
        if name in props_:
            i = props_.index(name)
            df[f"{name}_x"], df[f"{name}_y"], df[f"{name}_z"] = pp[:, i, 0], pp[:, i, 1], pp[:, i, 2]
    for side in omr.SIDES:
        for j in omr.arm_joints(side):
            df[f"tau_{j}"] = tau[:, names.index(j)]
    return df


def in_basket(ep, name: str) -> bool:
    """Is the pick-up inside the basket at the end (hull frame: the basket box)?"""
    log = ep.log
    props_ = list(log["props"])
    p = np.asarray(log["prop_pos"])[-1, props_.index(name)]
    hp, hq = np.asarray(log["body_pos"])[-1, 0], np.asarray(log["body_quat"])[-1, 0]
    w, x, y, z = hq
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    ph = R.T @ (p - hp)
    x0, y0, z0, x1, y1, z1 = osr.sweeper_geometry()["basket"]
    return bool(x0 < ph[0] < x1 and y0 < ph[1] < y1 and z0 - 0.02 < ph[2] < z1)


def phase_table(ep) -> pd.DataFrame:
    """Per mission phase: start, duration, hull x at its end, the peak arm-joint torques against stall (both arms),
    the broom's peak torque against its stall."""
    log = ep.log
    ts = timeseries(ep)
    starts = [(t, name) for t, name, note in log["mission"] if note == "start"]
    rows = {}
    stall = {j: omr.act.get(omr.ARM_ACTUATORS[j]).stall_Nm for j in omr.ARM_ACTUATORS}
    broom_stall = omr.act.get(osr.BROOM_DRIVE).stall_Nm
    for k, (t0, name) in enumerate(starts):
        t1 = starts[k + 1][0] if k + 1 < len(starts) else float(ts.t.iloc[-1])
        s = ts[(ts.t >= t0) & (ts.t <= t1)]
        row = {"start_s": t0, "duration_s": t1 - t0, "hull_x_m": float(s.x.iloc[-1]) if len(s) else float("nan"),
               "broom / stall": float(s.tau_broom.abs().max()) / broom_stall if len(s) else float("nan")}
        for side in omr.SIDES:
            for j, kind in zip(omr.arm_joints(side)[:5], ("yaw", "shoulder", "elbow", "wrist", "jaw")):
                peak = float(s[f"tau_{j}"].abs().max()) if len(s) else float("nan")
                row[f"{j} / stall"] = peak / stall[kind]
        rows[f"{k + 1:02d} {name}"] = row
    return pd.DataFrame(rows).T
