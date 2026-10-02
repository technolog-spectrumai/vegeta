"""The Onager Atlas's job in ChironLab (notebook 21 §7, ``scenarios/onager_atlas_pallet.py``): pick up a loaded Euro
pallet, carry it along a gravel yard and set it down.

* **pallet** — a Euro pallet (1200 × 800 × 144 mm, 25 kg: three bottom boards, nine blocks, the top deck; the fork
  openings 250 mm wide either side of the middle row of blocks) with a 175 kg crate strapped on it (one rigid body:
  the strapping is not modelled), lying on the yard with its 1200 mm side along the track;
* **yard** — compacted gravel (RMS 4 mm);
* **mission** (``ForkliftMission`` on top of the Sentinel drive, ``onager_controller.PhasedMission``) — forks at
  travel height (120 mm: tines dragged through gravel tip the robot onto its front wheels), drive until the tine
  tips are 25 cm short of the pallet, forks to the openings (50 mm), creep in until the heels are 4 cm from it, lift the forks'
  undersides to 250 mm (the tines meet the deck and lift it), tilt the mast back 5°, drive to the drop point at
  1 m/s, tilt upright, lower, back out.
  The mission knows where the pallet is (ideal perception: its simulated pose when it starts the approach).

Nothing about the lift is scripted beyond the forks' motion: whether the pallet comes up, stays on the tines while
the robot drives and lands where it should is decided by MuJoCo's contacts, the screws' force limits and the
chassis's servo compliance. ``run`` returns the Episode; ``timeseries`` and ``phase_table`` give the numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vegeta import chiron as ch

import onager_atlas_robot as oar
import onager_controller as oc
import onager_robot as orb

__all__ = ["PALLET", "Scene", "terrain", "props", "make_lab", "ForkliftMission", "pallet_job", "run", "timeseries",
           "phase_table"]

#: The Euro pallet and its load (inputs).
PALLET = {"length": 1.2, "width": 0.8, "height": 0.144, "mass_kg": 25.0, "crate": (1.0, 0.7, 0.6), "crate_kg": 175.0,
          "mu_ground": 0.5, "block_y": 0.35, "block_w": 0.10}
PALLET["total_kg"] = PALLET["mass_kg"] + PALLET["crate_kg"]


@dataclass
class Scene:
    x_pallet: float = 6.0                 # pallet centre on the track [m]
    x_drop: float = 14.0                  # where its centre should end up
    rough_rms: float = 0.004
    seed: int = 7


def terrain(scene: Scene):
    if scene.rough_rms <= 0:
        return ch.Flat()
    return ch.Rough(scene.rough_rms, 0.3, start=1.0, seed=scene.seed, extent=(-3.0, 25.0, -3.0, 3.0), cell=0.05)


def props(scene: Scene) -> list:
    """The pallet with its crate: one free body, its frame at the pallet's centre on the ground."""
    P = PALLET
    L, W = P["length"], P["width"]
    wood, crate_rgba = (0.72, 0.58, 0.38, 1.0), (0.35, 0.42, 0.30, 1.0)
    fr = (P["mu_ground"], 0.005, 0.0001)
    geoms = []
    by, bw = P["block_y"], P["block_w"]
    for k, y in enumerate((-by, 0.0, by)):
        geoms.append(ch.Geom(f"pallet_board_{k}", "box", (L / 2, bw / 2, 0.011), pos=(0.0, y, 0.011), mass=2.0,
                             friction=fr, rgba=wood))
        for j, x in enumerate((-L / 2 + 0.05, 0.0, L / 2 - 0.05)):
            geoms.append(ch.Geom(f"pallet_block_{k}{j}", "box", (0.05, bw / 2, 0.05), pos=(x, y, 0.072), mass=1.0,
                                 friction=fr, rgba=wood))
    geoms.append(ch.Geom("pallet_deck", "box", (L / 2, W / 2, 0.011), pos=(0.0, 0.0, 0.133), mass=10.0, friction=fr,
                         rgba=wood))
    cl, cw, chh = P["crate"]
    geoms.append(ch.Geom("crate", "box", (cl / 2, cw / 2, chh / 2), pos=(0.0, 0.0, P["height"] + chh / 2),
                         mass=P["crate_kg"], friction=fr, rgba=crate_rgba))
    z = float(terrain(scene).height(scene.x_pallet, 0.0))
    return [ch.Prop(ch.Link("pallet", pos=(scene.x_pallet, 0.0, z + 0.002), geoms=geoms), free=True)]


def make_lab(scene: Scene, robot=None, **kwargs):
    return oar.atlas_lab(terrain(scene), robot=robot, props=props(scene), course_extent=(-3.0, 25.0, -3.0, 3.0),
                         **kwargs)


class ForkliftMission(oc.PhasedMission):
    """``PhasedMission`` with the forklift: ``self.lift`` (the carriage's travel, m, clamped to its range) and
    ``self.tilt`` (mast back, rad). ``fork_height(z)`` converts a wanted height of the forks' undersides above the
    ground into a lift, from the height measured when the forks hang free at the start (an operator's eye, once).
    MuJoCo's gravity torques of the mast and carriage (``qfrc_bias``) are fed forward."""

    def tool_reset(self, lab):
        f = lab.robot.forklift
        self.fork_t, self.lift_max = f["fork_t"], f["lift_max"]
        self.fork_gid = [lab._geom_id("fork_L"), lab._geom_id("fork_R")]
        self.lift, self.tilt = 0.0, 0.0
        self.z_at_lift0 = None

    def fork_underside(self) -> float:
        d = self.lab.data
        return float(np.mean(d.geom_xpos[self.fork_gid, 2])) - self.fork_t / 2

    def fork_height(self, obs, z: float) -> float:
        if self.z_at_lift0 is None:
            self.z_at_lift0 = self.fork_underside() - float(obs.q[self.idx["lift"]])
        return float(np.clip(z - self.z_at_lift0, 0.0, self.lift_max))

    def tool_command(self, obs, q_t, qd_t, ff):
        bias = self.lab.data.qfrc_bias
        q_t["lift"] = self.lift
        q_t["mast_tilt"] = self.tilt
        for j in ("lift", "mast_tilt"):
            qd_t[j] = 0.0
            ff[j] = float(bias[self.dof[j]])


def pallet_job(scene: Scene, *, v_carry: float = 1.0, fork_travel: float = 0.12, fork_insert: float = 0.05,
               fork_carry: float = 0.25, fork_set: float = 0.03, tilt_deg: float = 5.0, heel_gap: float = 0.04,
               tip_gap: float = 0.25) -> list:
    """The phases: forks to travel height, drive to a stand-off before the pallet, forks to the openings' height,
    creep in, lift, tilt back, carry, tilt upright, lower, back out."""
    P = PALLET
    rad = math.radians

    def pallet_now(m, obs):
        if m.memory.get("pallet") is None:
            m.memory["pallet"] = np.asarray(obs.prop_pos[m.lab.prop_bodies.index("pallet")], dtype=float).copy()
        return m.memory["pallet"]

    def heel_world(m):
        return m.lab.robot.forklift["heel_x"] + m.lab.robot.forklift["pivot"][0]

    def set_fork(z, T):
        def start(m, obs):
            m.state.update(l0=m.lift, l1=m.fork_height(obs, z))

        def update(m, obs, tau):
            m.lift = m.state["l0"] + (m.state["l1"] - m.state["l0"]) * oc.smooth(tau / T)
            return tau >= T
        return start, update

    def set_tilt(a, T):
        def start(m, obs):
            m.state["a0"] = m.tilt

        def update(m, obs, tau):
            m.tilt = m.state["a0"] + (a - m.state["a0"]) * oc.smooth(tau / T)
            return tau >= T
        return start, update

    def go(x_fn, v_max):
        def update(m, obs, tau):
            return m.drive_to(obs, x_fn(m, obs), v_max)
        return update

    def wait(T):
        return lambda m, obs, tau: tau >= T

    approach_x = lambda m, obs: pallet_now(m, obs)[0] - P["length"] / 2 - heel_gap - heel_world(m)   # noqa: E731
    drop_x = lambda m, obs: scene.x_drop - (pallet_now(m, obs)[0] - approach_x(m, obs))               # noqa: E731
    back_x = lambda m, obs: drop_x(m, obs) - (P["length"] + 0.6)                                       # noqa: E731

    # the stand-off: the tine tips tip_gap short of the pallet (the tines are as long as most of the pallet)
    stand_x = lambda m, obs: approach_x(m, obs) - (m.lab.robot.forklift["fork_L"] - heel_gap + tip_gap)   # noqa: E731

    phases = []
    add = lambda name, upd, start=None, timeout=30.0: phases.append(oc.Phase(name, upd, start, timeout))  # noqa: E731
    st, up = set_fork(fork_travel, 2.0)
    add("forks to travel height", up, st)
    add("drive to the pallet", go(stand_x, 1.0), lambda m, obs: pallet_now(m, obs), timeout=30.0)
    st, up = set_fork(fork_insert, 1.5)
    add("forks to the openings", up, st)
    add("creep in", go(approach_x, 0.3), timeout=15.0)
    add("settle", wait(0.5))
    st, up = set_fork(fork_carry, 3.0)
    add("lift", up, st)
    st, up = set_tilt(rad(tilt_deg), 2.0)
    add("tilt back", up, st)
    add("carry", go(drop_x, v_carry), timeout=40.0)
    add("settle", wait(0.5))
    st, up = set_tilt(0.0, 2.0)
    add("tilt upright", up, st)
    st, up = set_fork(fork_set, 3.0)
    add("lower", up, st)
    add("settle", wait(0.5))
    add("back out", go(back_x, 0.4), timeout=20.0)
    return phases


def run(lab, scene: Scene, duration: float = 60.0, seed: int = 0, accel: float = 0.4, **job_kw):
    """The whole job; ``accel`` [m/s²] the chassis's acceleration and braking (0.4: gentle with a load up)."""
    mission = ForkliftMission(pallet_job(scene, **job_kw), accel=accel, drive_kw={"load_ff": "axle"}, name="pallet job")
    ep = lab.run(mission, duration=duration, rules=None, settle=0.5, seed=seed,
                 info={"controller": mission.name, "treatment": "atlas"})
    ep.log["mission"] = [list(x) for x in mission.log]
    ep.log["mission_finished"] = mission.finished
    return ep


def timeseries(ep) -> pd.DataFrame:
    """Per log sample: hull x, speed, pitch and tilt; the pallet's position, height and tilt; the fork height (lift
    and mast tilt), the lift force and tilt torque; the wheel loads and the front/rear knee torques."""
    log = ep.log
    names = list(log["joints"])
    q, tau = np.asarray(log["q"]), np.asarray(log["tau"])
    bq = np.asarray(log["body_quat"])[:, 0]
    w, x, y, z = bq.T
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x ** 2 + y ** 2), -1, 1)))
    pi = log["props"].index("pallet")
    pp, pq = np.asarray(log["prop_pos"])[:, pi], np.asarray(log["prop_quat"])[:, pi]
    ptilt = np.degrees(np.arccos(np.clip(1 - 2 * (pq[:, 1] ** 2 + pq[:, 2] ** 2), -1, 1)))
    ff = np.asarray(log["foot_force"])
    df = pd.DataFrame({"t": log["t"], "x": np.asarray(log["body_pos"])[:, 0, 0], "v": np.asarray(log["com_vel"])[:, 0],
                       "pitch_deg": pitch, "tilt_deg": tilt, "pallet_x": pp[:, 0], "pallet_y": pp[:, 1],
                       "pallet_z": pp[:, 2], "pallet_tilt_deg": ptilt,
                       "lift_m": q[:, names.index("lift")], "mast_tilt_deg": np.degrees(q[:, names.index("mast_tilt")]),
                       "lift_force_N": tau[:, names.index("lift")], "tilt_torque_Nm": tau[:, names.index("mast_tilt")]})
    for i, wname in enumerate(log["feet"]):
        df[f"fz_{wname}"] = ff[:, i, 2]
    for leg in orb.LEGS:
        df[f"knee_{leg}_Nm"] = tau[:, names.index(f"{leg}_knee")]
        df[f"shoulder_{leg}_Nm"] = tau[:, names.index(f"{leg}_shoulder")]
    return df


def phase_table(ep) -> pd.DataFrame:
    """Per mission phase: duration, the pallet's height and tilt at its end, peak lift force and tilt torque, peak
    front knee torque against the module's stall, the smallest rear-wheel load (tipping margin)."""
    log = ep.log
    ts = timeseries(ep)
    starts = [(t, name) for t, name, note in log["mission"] if note == "start"]
    stall = orb.act.get(orb.LEG_ACTUATOR).stall_Nm
    rows = {}
    for k, (t0, name) in enumerate(starts):
        t1 = starts[k + 1][0] if k + 1 < len(starts) else float(ts.t.iloc[-1])
        s = ts[(ts.t >= t0) & (ts.t <= t1)]
        if not len(s):
            continue
        rows[f"{k + 1:02d} {name}"] = {
            "start_s": t0, "duration_s": t1 - t0, "hull_x_m": float(s.x.iloc[-1]),
            "pallet_z_m": float(s.pallet_z.iloc[-1]), "pallet_tilt_deg": float(s.pallet_tilt_deg.max()),
            "lift_force_max_N": float(s.lift_force_N.abs().max()), "tilt_torque_max_Nm": float(s.tilt_torque_Nm.abs().max()),
            "front_knee_max/stall": float(s[["knee_FL_Nm", "knee_FR_Nm"]].abs().max().max() / stall),
            "rear_wheel_min_N": float(s[["fz_RL", "fz_RR"]].min().min()),
        }
    return pd.DataFrame(rows).T
