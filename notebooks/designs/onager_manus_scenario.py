"""The Onager Manus's mission in ChironLab (notebook 21 §8, ``scenarios/onager_manus_tasks.py``): a track blocked
first by a wire, then by a log.

* **wire** — high-tensile fence wire (Ø 3.15 mm, UTS 1200 MPa: the cutting force ``F_CUT`` = 0.8 × UTS × area =
  7.5 kN, an input) strung across the track at 1.15 m between two posts 2.8 m apart — it blocks the hull. It is two
  rigid halves, each hinged at its post (about the track's axis and the vertical), held together by a ``Weld`` exactly where the
  robot's cutting pincer meets it. A scene hook (``WireCutter``) releases the weld when **both** jaws squeeze the
  wire with at least ``F_CUT`` for ``T_SHEAR`` (the blade's travel through the wire): whether the wire parts is
  decided by the contact forces MuJoCo computes, i.e. by the jaw drive's torque and the notch's distance from the
  pin. The parted halves swing down on their hinges and the track is clear;
* **log** — spruce (450 kg/m³), Ø 150 mm × 1.8 m (14.3 kg), lying across the track (it spans both wheel tracks),
  its middle in line with the left arm: a free body the left pincer takes in the middle, lifts, swings over the
  side and puts down beside the track. Friction on
  the serrated jaws μ = 0.6 and a rolling resistance of 10 mm (bark and knots: it holds on a 7° slope) are inputs —
  a smooth cylinder rolls away on the gravel.

The robot only knows where the wire and the log are (``Scene``); the mission is ``onager_manus_controller
.cut_and_clear``. ``run(lab)`` returns the Episode with the mission's and the scene's events; ``timeseries`` and
``phase_table`` give the numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vegeta import chiron as ch

import onager_manus_controller as omc
import onager_manus_robot as omr

__all__ = ["Scene", "WireCutter", "WIRE", "LOG", "props", "terrain", "make_lab", "run", "timeseries", "phase_table"]

#: The wire (inputs): fence wire class, high tensile.
WIRE = {"diameter_m": 3.15e-3, "uts_MPa": 1200.0, "shear_over_uts": 0.8, "density_kg_m3": 7850.0,
        "contact_solref": (0.001, 1.0), "contact_solimp": (0.99, 0.999, 0.001)}
WIRE["area_mm2"] = math.pi / 4 * (WIRE["diameter_m"] * 1e3) ** 2
F_CUT = WIRE["shear_over_uts"] * WIRE["uts_MPa"] * WIRE["area_mm2"]           # N
T_SHEAR = 0.05                                                                # s at full force to part the wire
#: The log (inputs): spruce.
LOG = {"diameter_m": 0.15, "length_m": 1.8, "density_kg_m3": 450.0, "mu_on_jaws": 0.6, "rolling_m": 0.01}
LOG["mass_kg"] = LOG["density_kg_m3"] * math.pi / 4 * LOG["diameter_m"] ** 2 * LOG["length_m"]


@dataclass
class Scene:
    """Where things are (world, m): the wire across the track at ``x_wire``, height ``z_wire``, posts at
    ``±post_y``; the log's centre at ``x_log`` on the centre line; the mission ends at ``x_end``. ``rough_rms``:
    the gravel track (0 = flat)."""

    x_wire: float = 6.0
    z_wire: float = 1.15
    post_y: float = 1.4
    cut_y: float = -0.28                         # where the cutting pincer meets the wire (the right arm's plane)
    x_log: float = 12.0
    y_log: float = 0.28                          # the log's middle: in line with the left arm
    x_end: float = 18.0
    rough_rms: float = 0.006
    seed: int = 4
    WIRE_WELD: str = "wire joint"
    WIRE_GEOMS: tuple = ("wire_a_g", "wire_b_g")

    def wire_point(self, y: float) -> np.ndarray:
        return np.array([self.x_wire, y, self.z_wire])

    def ground(self, x, y) -> float:
        return float(terrain(self).height(x, y))

    def log_center(self) -> np.ndarray:
        """Where the log is placed (it settles a little: the mission reads its pose when it reaches for it)."""
        r = LOG["diameter_m"] / 2
        ys = self.y_log + np.linspace(-LOG["length_m"] / 2, LOG["length_m"] / 2, 9)
        z = max(self.ground(self.x_log, y) for y in ys) + r
        return np.array([self.x_log, self.y_log, z])


def terrain(scene: Scene):
    if scene.rough_rms <= 0:
        return ch.Flat()
    return ch.Rough(scene.rough_rms, 0.3, start=1.0, seed=scene.seed, extent=(-3.0, 25.0, -3.0, 3.0), cell=0.05)


def props(scene: Scene) -> tuple:
    """The posts, the two wire halves (hinged at the posts) with their weld, and the log."""
    r = WIRE["diameter_m"] / 2
    post_h = scene.z_wire + 0.05
    out = []
    for side, y0 in (("a", -scene.post_y), ("b", scene.post_y)):
        out.append(ch.Prop(ch.Link(f"post_{side}", pos=(scene.x_wire, y0, 0.0),
                                   geoms=[ch.Geom(f"post_{side}_g", "box", (0.04, 0.04, post_h / 2), pos=(0, 0, post_h / 2),
                                                  rgba=(0.45, 0.33, 0.2, 1.0))])))
    # a hard contact for a thin wire squeezed by heavy jaws: MuJoCo's soft contact scales with the lighter body's
    # mass (the wire), so the time constant is at its floor (2 time steps) and the impedance near 1: ~1 mm of
    # penetration at the cutting force (a 3.15 mm wire)
    hinge_y = scene.post_y - 0.06                                         # beside the post, not on it
    for side, y0, y1 in (("a", -hinge_y, scene.cut_y), ("b", hinge_y, scene.cut_y)):
        L = abs(y1 - y0)
        m = WIRE["density_kg_m3"] * math.pi * r * r * L
        # two hinges at the post (a wire end swings down, and aside when the robot pushes it)
        out.append(ch.Prop(ch.Link(f"wire_{side}", pos=(scene.x_wire, y0, scene.z_wire),
                                   joints=[ch.Joint(f"wire_{side}_hinge", axis=(1, 0, 0), damping=0.02),
                                           ch.Joint(f"wire_{side}_swing", axis=(0, 0, 1), damping=0.02)],
                                   geoms=[ch.Geom(f"wire_{side}_g", "capsule", (r,), fromto=(0, 0, 0, 0, y1 - y0, 0),
                                                  mass=m, friction=(0.3, 0.005, 0.0001), rgba=(0.75, 0.75, 0.78, 1.0))]),
                           solref=WIRE["contact_solref"], solimp=WIRE["contact_solimp"]))
    c = scene.log_center()
    q = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)          # cylinder axis z -> y: across the track
    out.append(ch.Prop(ch.Link("log", pos=tuple(c), quat=q,
                               geoms=[ch.Geom("log_g", "cylinder", (LOG["diameter_m"] / 2, LOG["length_m"] / 2),
                                              mass=LOG["mass_kg"], friction=(LOG["mu_on_jaws"], 0.01, LOG["rolling_m"]),
                                              rgba=(0.55, 0.38, 0.2, 1.0))]),
                       free=True, condim=6))
    welds = [ch.Weld(scene.WIRE_WELD, "wire_a", "wire_b")]
    return out, welds


class WireCutter:
    """Scene hook: the wire parts (its weld is released) once both jaws of a pincer have squeezed it with at
    least ``f_cut`` for ``t_shear`` (cumulative). ``history`` keeps [t, upper-jaw force, lower-jaw force]."""

    def __init__(self, scene: Scene, jaws=None, f_cut: float = F_CUT, t_shear: float = T_SHEAR):
        self.scene, self.f_cut, self.t_shear = scene, float(f_cut), float(t_shear)
        self.jaws = jaws or {s: omr.jaw_geoms(s) for s in omr.SIDES}

    def reset(self, lab):
        self.time_above, self.history, self.cut_at, self.peak = 0.0, [], None, 0.0

    def __call__(self, lab):
        if self.cut_at is not None:
            return
        wire = list(self.scene.WIRE_GEOMS)
        best = (0.0, 0.0)
        for upper, lower in self.jaws.values():
            fu = lab.contact_force(upper, wire)[1]
            fl = lab.contact_force(lower, wire)[1]
            if min(fu, fl) > min(best):
                best = (fu, fl)
        squeeze = min(best)
        self.peak = max(self.peak, squeeze)
        if squeeze > 1.0 or self.history:
            self.history.append([lab.time, best[0], best[1]])
        if squeeze >= self.f_cut:
            self.time_above += lab.control_dt
            if self.time_above >= self.t_shear - 1e-9:
                lab.set_weld(self.scene.WIRE_WELD, False)
                self.cut_at = lab.time
                lab.log_event("wire", f"cut: both jaws ≥ {self.f_cut:.0f} N for {self.t_shear * 1000:.0f} ms "
                                      f"(peak squeeze {self.peak:.0f} N)")


def make_lab(scene: Scene, robot=None, **kwargs):
    """The Manus on the scene's track with its props, weld and the wire-cutter hook (``lab.cutter``)."""
    pr, welds = props(scene)
    lab = omr.manus_lab(terrain(scene), robot=robot, props=pr, welds=welds,
                        course_extent=(-3.0, 25.0, -3.0, 3.0), **kwargs)
    lab.cutter = WireCutter(scene)
    lab.add_hook(lab.cutter)
    return lab


def run(lab, scene: Scene, duration: float = 75.0, seed: int = 0, **mission_kw):
    """The whole mission; ``ep.log['mission']`` the controller's phase log, ``ep.log['events']`` the scene's."""
    mission = omc.Mission(omc.cut_and_clear(scene, **mission_kw), name="cut the wire, clear the log")
    ep = lab.run(mission, duration=duration, rules=None, settle=0.5, seed=seed,
                 info={"controller": mission.name, "treatment": "manus"})
    ep.log["mission"] = [list(x) for x in mission.log]
    ep.log["mission_finished"] = mission.i >= len(mission.phases)
    ep.log["cutter_history"] = np.asarray(lab.cutter.history, dtype=float).reshape(-1, 3)
    ep.log["cut_at"] = lab.cutter.cut_at
    return ep


def timeseries(ep) -> pd.DataFrame:
    """Per log sample: hull x and speed, tilt, the log's height and lateral position, the wire's free-end heights,
    arm joint torques (both arms) and jaw torques."""
    log = ep.log
    names = list(log["joints"])
    tau = np.asarray(log["tau"])
    props_ = list(log["props"])
    pp, pq = np.asarray(log["prop_pos"]), np.asarray(log["prop_quat"])
    q = np.asarray(log["body_quat"])[:, 0]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
    df = pd.DataFrame({"t": log["t"], "x": np.asarray(log["body_pos"])[:, 0, 0], "v": np.asarray(log["com_vel"])[:, 0],
                       "tilt_deg": tilt, "log_z": pp[:, props_.index("log"), 2], "log_y": pp[:, props_.index("log"), 1]})
    for w in ("a", "b"):
        i = props_.index(f"wire_{w}")
        ang = 2 * np.arctan2(pq[:, i, 1], pq[:, i, 0])                        # about the hinge axis (x)
        df[f"wire_{w}_deg"] = np.degrees(ang)
    for side in omr.SIDES:
        for j in omr.arm_joints(side):
            df[f"tau_{j}"] = tau[:, names.index(j)]
    return df


def phase_table(ep) -> pd.DataFrame:
    """Per mission phase: start, duration, and the peak absolute torque of each arm joint against its stall."""
    log = ep.log
    ts = timeseries(ep)
    starts = [(t, name) for t, name, note in log["mission"] if note == "start"]
    rows = {}
    stall = {j: omr.act.get(omr.ARM_ACTUATORS[j]).stall_Nm for j in omr.ARM_ACTUATORS}
    for k, (t0, name) in enumerate(starts):
        t1 = starts[k + 1][0] if k + 1 < len(starts) else float(ts.t.iloc[-1])
        s = ts[(ts.t >= t0) & (ts.t <= t1)]
        row = {"start_s": t0, "duration_s": t1 - t0, "hull_x_m": float(s.x.iloc[-1]) if len(s) else float("nan")}
        for side in omr.SIDES:
            for j, kind in zip(omr.arm_joints(side)[:5], ("yaw", "shoulder", "elbow", "wrist", "jaw")):
                peak = float(s[f"tau_{j}"].abs().max()) if len(s) else float("nan")
                row[f"{j} / stall"] = peak / stall[kind]
        rows[f"{k + 1:02d} {name}"] = row
    return pd.DataFrame(rows).T
