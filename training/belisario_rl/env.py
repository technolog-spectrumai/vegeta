"""The environment: Belisario's bare frame in MuJoCo, Gymnasium-compatible (``gymnasium.Env`` when gymnasium is
installed, the same interface otherwise), one robot per instance, vectorised by ``vec_env``.

Observations (the actor's, all available on the physical robot): the joint positions as offsets from the default
pose, the joint velocities, the pelvis's angular velocity and the projected gravity in its frame, an estimate of
the pelvis's linear velocity in its frame (a finite difference the deployed estimator would replace), the four
contact flags, the previous action, and the normalised command (``commands``). The critic's extra, privileged
observations (``info["critic_obs"]``): the true linear velocity, the height above the ground, the contact forces,
the current push, the sampled ground friction and mass scale, and the goal's offset in the base frame.

Actions: normalised joint-position offsets about the default pose (``pd.PDController``), applied for
``decimation`` physics steps per control step. Episodes end on a fall (height, tilt, a body collision), on the
time limit, or on reaching the scene's goal.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .commands import COMMAND_DIM, Command, CommandRanges, CommandSampler
from .config import Config
from .pd import PDController
from .randomization import Disturbances, DomainRandomizer
from .rewards import RewardComputer, StepState
from .robot_model import CONTACT_GEOMS, RobotModel
from .scenes import SceneSpec, build_scene

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:                                                        # the same interface without gymnasium
    gym, spaces = None, None

    class _Base:
        metadata = {}

__all__ = ["BelisarioEnv", "ObsLayout", "PRIVILEGED_DIM"]

PRIVILEGED_DIM = 3 + 1 + 4 + 3 + 1 + 1 + 3


class ObsLayout:
    """Where each block sits in the actor's observation vector."""

    def __init__(self, n_joints: int, n_act: int | None = None):
        n = n_joints
        self.blocks = [("joint_pos", n), ("joint_vel", n), ("ang_vel", 3), ("gravity", 3), ("lin_vel_est", 3),
                       ("contacts", len(CONTACT_GEOMS)), ("prev_action", n if n_act is None else n_act), ("command", COMMAND_DIM)]
        self.slices, k = {}, 0
        for name, size in self.blocks:
            self.slices[name] = slice(k, k + size)
            k += size
        self.dim = k


class BelisarioEnv(_Base):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Config, scene: SceneSpec | None = None, seed: int = 0, eval_mode: bool = False, render_mode: str | None = None,
                 command: Command | None = None):
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.scene = scene or build_scene("flat", cfg)
        self.robot = RobotModel(terrain=self.scene.terrain, sim_hz=cfg.sim.sim_hz, casing=cfg.sim.casing, props=self.scene.props,
                                extent=self.scene.extent)
        self.model, self.data = self.robot.model, mujoco.MjData(self.robot.model)
        self.joints = self.robot.joints
        self.n = self.joints.n
        self.decimation = max(1, int(round(cfg.sim.sim_hz / cfg.sim.control_hz)))
        self.dt = self.decimation * self.model.opt.timestep
        self.max_steps = int(round(cfg.sim.episode_seconds * cfg.sim.control_hz))
        self.default = self.robot.q_quad if cfg.action.default_pose == "quad" else self.robot.q_stand
        a = cfg.action
        self.pd = PDController(self.joints, self.default["q"], action_scale=a.scale, action_clip=a.clip, kp_fraction=a.kp_fraction,
                               kd_fraction=a.kd_fraction, torque_fraction=a.torque_fraction, velocity_limit_fraction=a.velocity_limit_fraction,
                               latency_steps=a.latency_steps)
        self.ranges = CommandRanges.from_config(cfg.commands)
        self.sampler = CommandSampler(self.ranges, cfg.commands.stop_probability, cfg.commands.hold_still_speed, self.rng)
        self.fixed_command = command
        self.randomizer = DomainRandomizer(cfg.randomization, self.model, self.robot._robot_bodies(), self.robot.terrain_geom, self.rng)
        self.disturbances = Disturbances(cfg.disturbance, self.rng, enabled=cfg.disturbance.enabled and not eval_mode)
        self.rewards = RewardComputer(cfg.reward, goal_weight=1.0 if self.scene.goal is not None else 0.0)
        self.body_geoms = self.robot.body_collision_geoms(cfg.termination.body_collision_geoms)
        hands = a.hands == "policy"
        self.act_index = np.array([i for i, nm in enumerate(self.joints.names) if hands or not ("finger" in nm or "thumb" in nm)])
        self.act_dim = len(self.act_index)
        self.layout = ObsLayout(self.n, self.act_dim)
        self.obs_dim = self.layout.dim
        if spaces is not None:
            self.observation_space = spaces.Box(-cfg.obs.clip, cfg.obs.clip, (self.obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), dtype=np.float32)
        self.reference = None                                              # set by the runner when imitation is on
        self.viewer = None
        self._reset_state()

    # ---- state
    def _reset_state(self):
        self.t, self.step_count = 0.0, 0
        self.command = Command()
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.action = np.zeros(self.act_dim, dtype=np.float32)
        self.contacts_prev = np.zeros(len(CONTACT_GEOMS), dtype=bool)
        self.qd_prev = np.zeros(self.n)
        self.prev_base_pos = np.zeros(3)
        self.push = np.zeros(3)
        self.goal_dist_prev = 0.0
        self.sample = {"mass_scale": 1.0, "friction": self.cfg.terrain.friction, "strength": 1.0, "latency": 0}
        self._next_cmd = 0.0

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.sampler.rng = self.rng
        self.randomizer.rng = self.rng
        self.disturbances.rng = self.rng

    # ---- the Gymnasium interface
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)
        options = options or {}
        self._reset_state()
        m, d, r = self.model, self.data, self.cfg.randomization
        self.sample = self.randomizer.apply(self.pd, self.joints.dof_adr, self.cfg.terrain.friction)
        self.disturbances.reset()
        mujoco.mj_resetData(m, d)
        q = self.default["q"].copy()
        noise = 0.0 if self.eval_mode else r.init_pose_noise
        if noise > 0:
            q = np.clip(q + self.rng.normal(0.0, noise, size=self.n), self.joints.lower, self.joints.upper)
        d.qpos[self.joints.qpos_adr] = q
        yaw = self.scene.start[2] + (0.0 if self.eval_mode else self.rng.uniform(-r.init_yaw, r.init_yaw))
        if self.scene.goal is not None:
            yaw = self.scene.start[2]                                     # face the task
        x0, y0 = self.scene.start[0], self.scene.start[1]
        ground = self._ground_height(x0, y0)
        d.qpos[self.robot.free_qpos:self.robot.free_qpos + 3] = [x0, y0, ground + self.default["height"] + 0.02]
        d.qpos[self.robot.free_qpos + 3:self.robot.free_qpos + 7] = self._quat(yaw, self.default["pitch"])
        if not self.eval_mode and r.init_vel_noise > 0:
            d.qvel[self.robot.free_dof:self.robot.free_dof + 6] = self.rng.normal(0.0, r.init_vel_noise, size=6)
        mujoco.mj_forward(m, d)
        self.command = options.get("command", self.fixed_command) or self.sampler.sample()
        self._next_cmd = self.cfg.commands.resample_seconds
        self.prev_base_pos = d.qpos[self.robot.free_qpos:self.robot.free_qpos + 3].copy()
        self.contacts_prev = self.robot.contact_flags(d)
        self.qd_prev = d.qvel[self.joints.dof_adr].copy()
        self.goal_dist_prev = self._goal_distance()
        obs, critic = self._observe()
        return obs, {"critic_obs": critic, "command": self.command, "sample": self.sample}

    def step(self, action):
        self.prev_action = self.action
        self.action = np.clip(np.asarray(action, dtype=np.float32), -self.cfg.action.clip, self.cfg.action.clip)
        m, d = self.model, self.data
        full = np.zeros(self.n, dtype=np.float32)                            # the joints the policy does not drive stay at the default pose
        full[self.act_index] = self.action
        self.pd.command(full)
        if not self.eval_mode and self.fixed_command is None and self.t >= self._next_cmd:
            self.command = self.sampler.sample()
            self._next_cmd = self.t + self.cfg.commands.resample_seconds
        self.push = self.disturbances.force(self.t)
        base = self.robot.base_body
        for _ in range(self.decimation):
            q = d.qpos[self.joints.qpos_adr]
            qd = d.qvel[self.joints.dof_adr]
            d.ctrl[:] = self.pd.torques(q, qd)
            d.xfrc_applied[base, 0:3] = self.push
            mujoco.mj_step(m, d)
        d.xfrc_applied[base, :] = 0.0
        self.t += self.dt
        self.step_count += 1
        state = self._state()
        reward, terms = self.rewards.compute(state)
        terminated = state.fell or state.goal_reached
        truncated = self.step_count >= self.max_steps and not terminated
        obs, critic = self._observe()
        self.contacts_prev = state.contacts
        self.qd_prev = state.qd.copy()
        self.prev_base_pos = d.qpos[self.robot.free_qpos:self.robot.free_qpos + 3].copy()
        info = {"critic_obs": critic, "reward_terms": terms, "command": self.command, "success": bool(state.goal_reached), "fell": bool(state.fell),
                "height": state.height, "lin_vel": state.lin_vel.copy(), "contacts": state.contacts.copy()}
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode != "human":
            return None
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ---- the state
    @staticmethod
    def _quat(yaw, pitch):
        cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])             # yaw about z, then pitch about the body's y

    def _rot(self):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[self.robot.free_qpos + 3:self.robot.free_qpos + 7])
        return R.reshape(3, 3)

    def _ground_height(self, x, y):
        """The terrain under (x, y): a ray straight down from 5 m up (the plane or the height field)."""
        geomid = np.array([-1], dtype=np.int32)
        pnt, vec = np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0])
        dist = mujoco.mj_ray(self.model, self.data, pnt, vec, None, 1, -1, geomid)
        return 5.0 - dist if dist >= 0 else 0.0

    def _commanded_pitch(self):
        hq, hs, pq = self.robot.q_quad["height"], self.robot.q_stand["height"], self.robot.q_quad["pitch"]
        f = np.clip((hs - self.command.height) / max(hs - hq, 1e-6), 0.0, 1.0)
        return pq * f

    def _goal_distance(self):
        if self.scene.goal is None:
            return 0.0
        p = self.data.qpos[self.robot.free_qpos:self.robot.free_qpos + 3]
        g = np.asarray(self.scene.goal)
        return float(np.linalg.norm(g[:2] - p[:2]) + (abs(g[2] - p[2]) if self.scene.task == "climb" else 0.0))

    def _state(self) -> StepState:
        d, R = self.data, self._rot()
        p = d.qpos[self.robot.free_qpos:self.robot.free_qpos + 3]
        v_world = d.qvel[self.robot.free_dof:self.robot.free_dof + 3]
        w_world = d.qvel[self.robot.free_dof + 3:self.robot.free_dof + 6]
        lin_vel, ang_vel = R.T @ v_world, R.T @ w_world
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        height = float(p[2] - self._ground_height(p[0], p[1]))
        pitch_cmd = self._commanded_pitch()
        up_cmd = np.array([math.sin(pitch_cmd), 0.0, math.cos(pitch_cmd)])
        tilt = math.acos(max(-1.0, min(1.0, float(R[:, 2] @ (self._yaw_rot() @ up_cmd)))))
        pitch_err = float(math.atan2(-R[2, 0], R[2, 2]) - pitch_cmd)
        q = d.qpos[self.joints.qpos_adr].copy()
        qd = d.qvel[self.joints.dof_adr].copy()
        qdd = (qd - self.qd_prev) / self.dt
        contacts = self.robot.contact_flags(d)
        forces = self.robot.contact_forces(d)
        slip = self._slip(contacts)
        collided = bool(len(self.body_geoms) and self.robot.touching(d, self.body_geoms))
        fell = height < self.cfg.termination.min_height or math.degrees(tilt) > self.cfg.termination.max_tilt_deg or \
            (collided and self.cfg.termination.body_collision_terminates)
        gd = self._goal_distance()
        progress = self.goal_dist_prev - gd if self.scene.goal is not None else 0.0
        reached = self.scene.goal is not None and gd < self.scene.goal_radius
        self.goal_dist_prev = gd
        c = self.command.effective()
        q_ref = self.reference.walk_at(self.t) if (self.reference is not None and not c.stop) else None
        return StepState(self.dt, lin_vel, ang_vel, c.vx, c.vy, c.yaw_rate, c.height, c.stop, height, gravity, pitch_err, contacts,
                         self.contacts_prev, forces, slip, self.pd.last_torque.copy(), q, qd, qdd, self.joints.lower, self.joints.upper,
                         self.action, self.prev_action, collided, fell, self.pd.saturation(qd), q_ref=q_ref, goal_progress=progress,
                         goal_reached=reached, speed_norm=max(abs(self.ranges.vx[0]), abs(self.ranges.vx[1])))

    def _yaw_rot(self):
        R = self._rot()
        yaw = math.atan2(R[1, 0], R[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _slip(self, contacts):
        """Tangential speed of each contact geom's centre while in contact [m/s]."""
        d, m = self.data, self.model
        out = np.zeros(len(CONTACT_GEOMS))
        vel = np.zeros(6)
        for k, name in enumerate(CONTACT_GEOMS):
            if not contacts[k]:
                continue
            gid = self.robot.contact_geoms[name]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, gid, vel, 0)
            out[k] = float(np.linalg.norm(vel[3:5]))
        return out

    def _observe(self):
        d, R = self.data, self._rot()
        o = self.cfg.obs
        q = d.qpos[self.joints.qpos_adr] - self.default["q"]
        qd = d.qvel[self.joints.dof_adr]
        w = R.T @ d.qvel[self.robot.free_dof + 3:self.robot.free_dof + 6]
        g = R.T @ np.array([0.0, 0.0, -1.0])
        q, qd, w, g = self.randomizer.noise(o, q, qd, w, g)
        p = d.qpos[self.robot.free_qpos:self.robot.free_qpos + 3]
        if o.lin_vel_estimate == "finite_difference" and self.step_count > 0:
            v_est = R.T @ ((p - self.prev_base_pos) / self.dt)
        else:
            v_est = np.zeros(3)
        contacts = self.robot.contact_flags(d).astype(np.float32)
        cmd = self.ranges.normalize(self.command)
        obs = np.concatenate([q * o.joint_pos_scale, qd * o.joint_vel_scale, w * o.ang_vel_scale, g, v_est * o.lin_vel_scale, contacts,
                              self.action, cmd]).astype(np.float32)
        obs = np.clip(obs, -o.clip, o.clip)
        v_true = R.T @ d.qvel[self.robot.free_dof:self.robot.free_dof + 3]
        height = p[2] - self._ground_height(p[0], p[1])
        forces = self.robot.contact_forces(d) / max(self.robot.total_mass * 9.81, 1.0)
        goal = np.zeros(3) if self.scene.goal is None else R.T @ (np.asarray(self.scene.goal) - p)
        critic = np.concatenate([v_true * o.lin_vel_scale, [height], forces, self.push / 200.0, [self.sample["friction"]],
                                 [self.sample["mass_scale"]], np.clip(goal, -5.0, 5.0)]).astype(np.float32)
        return obs, critic

    # ---- for the controllers and the comparisons
    def joint_targets(self) -> np.ndarray:
        return self.pd.last_target.copy()

    def base_state(self):
        d = self.data
        return d.qpos[self.robot.free_qpos:self.robot.free_qpos + 7].copy(), d.qvel[self.robot.free_dof:self.robot.free_dof + 6].copy()
