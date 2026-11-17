"""The robot as the environment sees it: the MuJoCo model built from the design (``space_unit_robots.bare`` through
Chiron's ``Robot.to_mjcf``, on a terrain), the tables of actuated joints with their limits and servo data, the
contact geoms, the default poses (the quad pose from ``kerchak_motions``, the standing design pose), and a check
of what the model lacks for training (``missing_requirements``).

The model's actuators are Chiron's ``<general>`` servos: ``ctrl`` is a joint torque (gain 1) and the servo's
damping is a bias term; the PD loop that turns the policy's joint targets into torques lives here, in ``pd.py``,
not in the model. There are no MuJoCo sensors in the model: the observations are computed from the state
(``env.py``), which is what the physical robot's IMU, encoders and contact switches would give.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DESIGNS = ROOT / "labs" / "legged" / "designs"
if str(DESIGNS) not in sys.path:
    sys.path.insert(0, str(DESIGNS))

__all__ = ["JointTable", "RobotModel", "CONTACT_GEOMS", "BASE_BODY", "FREE_JOINT"]

BASE_BODY = "pelvis"
FREE_JOINT = "pelvis:free"
#: the contact geoms, in the order of the contact observation: left foot, right foot, left knuckle, right knuckle
CONTACT_GEOMS = ("L", "R", "L_knuckle", "R_knuckle")


@dataclass
class JointTable:
    """The actuated joints in actuator order."""
    names: list
    joint_ids: np.ndarray
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    stall: np.ndarray
    no_load_speed: np.ndarray
    armature: np.ndarray

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.lower + self.upper)

    @property
    def span(self) -> np.ndarray:
        return self.upper - self.lower


class RobotModel:
    """The compiled model plus its tables. ``terrain`` is a Chiron ``Terrain`` (None: flat), ``props`` Chiron props
    (a doorway, a wall), ``sim_hz`` the physics rate."""

    def __init__(self, terrain=None, sim_hz: int = 1000, casing: str = "none", props=(), extent=None):
        import space_unit_robots as sr
        from vegeta.chiron import SimOptions

        self.robot = sr.bare(casing=casing)
        opts = SimOptions(timestep=1.0 / sim_hz)
        if extent is not None:
            opts.course_extent = tuple(extent)
        self.xml = self.robot.to_mjcf(terrain=terrain, options=opts, props=list(props))
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.sim_hz = sim_hz
        m = self.model
        self.base_body = m.body(BASE_BODY).id
        self.free_qpos = m.jnt_qposadr[m.joint(FREE_JOINT).id]
        self.free_dof = m.jnt_dofadr[m.joint(FREE_JOINT).id]
        self.joints = self._joint_table()
        self.contact_geoms = {name: m.geom(name).id for name in CONTACT_GEOMS}
        self.terrain_geom = m.geom("chiron_terrain").id
        self.robot_geoms = self._robot_geom_ids()
        self.total_mass = float(sum(m.body_mass[b] for b in self._robot_bodies()))
        self.q_quad, self.q_stand = self._default_poses()

    # ---- tables
    def _joint_table(self) -> JointTable:
        m = self.model
        servos = {j.name: j.servo for j in self.robot.joints() if j.servo is not None}
        names, jids = [], []
        for a in range(m.nu):
            jid = int(m.actuator_trnid[a][0])
            names.append(m.joint(jid).name)
            jids.append(jid)
        jids = np.array(jids)
        lower = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -np.pi for j in jids])
        upper = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else np.pi for j in jids])
        kp = np.array([servos[n].kp for n in names])
        kd = np.array([servos[n].kd for n in names])
        stall = np.array([servos[n].stall_torque for n in names])
        w0 = np.array([servos[n].no_load_speed for n in names])
        arm = np.array([servos[n].armature for n in names])
        return JointTable(names, jids, m.jnt_qposadr[jids].copy(), m.jnt_dofadr[jids].copy(), lower, upper, kp, kd, stall, w0, arm)

    def _robot_bodies(self) -> list:
        m = self.model
        out = []
        for b in range(1, m.nbody):
            r = b
            while r != 0 and r != self.base_body:
                r = m.body_parentid[r]
            if r == self.base_body:
                out.append(b)
        return out

    def _robot_geom_ids(self) -> dict:
        m = self.model
        bodies = set(self._robot_bodies())
        out = {}
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in bodies and (m.geom_contype[g] or m.geom_conaffinity[g]):
                out[m.geom(g).name] = g
        return out

    def _default_poses(self):
        """The quad pose (``kerchak_motions.quad_pose``) and the standing design pose, in actuator order, with the
        pelvis's height and pitch for each."""
        import kerchak_motions as gm
        import motion_kit as mk
        kin = mk.Kin(self.robot)
        qq = gm.quad_pose(kin)
        order = [kin.index[n] for n in self.joints.names]
        quad = {"q": np.asarray(qq)[order], "height": gm.PELVIS_H, "pitch": gm.PELVIS_PITCH}
        import humanoid_motions as hum
        stand = {"q": np.asarray(kin.q0)[order], "height": hum.base_height(kin), "pitch": 0.0}
        return quad, stand

    # ---- helpers for the environment
    def body_collision_geoms(self, names) -> np.ndarray:
        """Geom ids of the parts that must not touch the ground (by name; missing names are ignored)."""
        return np.array([self.robot_geoms[n] for n in names if n in self.robot_geoms], dtype=int)

    def contact_flags(self, data: mujoco.MjData) -> np.ndarray:
        """Which of the four contacts touch the terrain (or a prop), in CONTACT_GEOMS order."""
        flags = np.zeros(len(CONTACT_GEOMS), dtype=bool)
        ids = [self.contact_geoms[n] for n in CONTACT_GEOMS]
        for i in range(data.ncon):
            c = data.contact[i]
            for k, g in enumerate(ids):
                if c.geom1 == g or c.geom2 == g:
                    flags[k] = True
        return flags

    def contact_forces(self, data: mujoco.MjData) -> np.ndarray:
        """The normal force on each of the four contacts [N] (summed over their contacts)."""
        out = np.zeros(len(CONTACT_GEOMS))
        ids = [self.contact_geoms[n] for n in CONTACT_GEOMS]
        f = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            for k, g in enumerate(ids):
                if c.geom1 == g or c.geom2 == g:
                    mujoco.mj_contactForce(self.model, data, i, f)
                    out[k] += abs(f[0])
        return out

    def touching(self, data: mujoco.MjData, geom_ids: np.ndarray) -> bool:
        """Whether any of ``geom_ids`` is in contact."""
        ids = set(int(g) for g in geom_ids)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 in ids or c.geom2 in ids:
                return True
        return False

    def missing_requirements(self) -> list:
        """What the model lacks for training and deployment, stated rather than invented."""
        out = []
        m = self.model
        if m.nsensor == 0:
            out.append("no MuJoCo sensors: the observations are computed from the state (base angular velocity, projected gravity, "
                       "joint positions and velocities, contacts); the physical robot needs an IMU in the pelvis, joint encoders "
                       "and contact switches on the soles and the knuckles to provide the same")
        if not any(n.endswith("knuckle") for n in self.robot_geoms):
            out.append("no knuckle contact geoms")
        if m.nu != self.joints.n:
            out.append("actuators do not match the joint table")
        if np.any(self.joints.no_load_speed <= 0):
            out.append("a servo has no no-load speed: the torque–speed line cannot be applied")
        if m.opt.timestep > 0.002:
            out.append(f"the physics step is {m.opt.timestep} s: Chiron's servos are integrated implicitly and need ≤ 1 ms for stability")
        return out

    def summary(self) -> dict:
        j = self.joints
        return {"joints": j.n, "mass_kg": round(self.total_mass, 1), "bodies": int(self.model.nbody), "geoms": int(self.model.ngeom),
                "contact_geoms": list(self.contact_geoms), "timestep": float(self.model.opt.timestep),
                "torque_limits_Nm": {n: float(s) for n, s in zip(j.names, j.stall)},
                "no_load_speed_rad_s": {n: float(s) for n, s in zip(j.names, j.no_load_speed)}}
