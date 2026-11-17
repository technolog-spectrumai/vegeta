"""Domain randomization and disturbances. ``DomainRandomizer`` samples, at every reset, body masses, joint friction
and damping, motor strength (the PD's kp and torque scale), actuator latency and the ground's friction, writing
them into the MuJoCo model in place from a saved nominal copy, and adds sensor noise to observations when asked.
``Disturbances`` schedules pushes at the pelvis. Both do nothing until enabled in the configuration, so the early
curriculum stages train on the nominal model."""
from __future__ import annotations

import numpy as np

__all__ = ["DomainRandomizer", "Disturbances"]


class DomainRandomizer:
    def __init__(self, cfg, model, robot_bodies, terrain_geom: int, rng=None):
        self.cfg, self.model, self.rng = cfg, model, rng or np.random.default_rng()
        self.bodies = np.array(list(robot_bodies), dtype=int)
        self.terrain_geom = int(terrain_geom)
        self.nominal = {"body_mass": model.body_mass.copy(), "body_inertia": model.body_inertia.copy(),
                        "dof_frictionloss": model.dof_frictionloss.copy(), "dof_damping": model.dof_damping.copy(),
                        "geom_friction": model.geom_friction.copy()}
        self.sample = {"mass_scale": 1.0, "friction": float(model.geom_friction[self.terrain_geom][0]), "strength": 1.0, "latency": 0}

    def restore(self):
        m = self.model
        m.body_mass[:] = self.nominal["body_mass"]
        m.body_inertia[:] = self.nominal["body_inertia"]
        m.dof_frictionloss[:] = self.nominal["dof_frictionloss"]
        m.dof_damping[:] = self.nominal["dof_damping"]
        m.geom_friction[:] = self.nominal["geom_friction"]

    def apply(self, pd, dof_adr, ground_friction_nominal: float) -> dict:
        """Sample and write; returns the sample (for the critic's privileged observations)."""
        self.restore()
        c, m, rng = self.cfg, self.model, self.rng
        if not c.enabled:
            pd.strength[:] = 1.0
            pd.reset(latency_steps=None)
            self.sample = {"mass_scale": 1.0, "friction": ground_friction_nominal, "strength": 1.0, "latency": pd.latency}
            return dict(self.sample)
        scale = rng.uniform(*c.mass_scale, size=len(self.bodies))
        m.body_mass[self.bodies] = self.nominal["body_mass"][self.bodies] * scale
        m.body_inertia[self.bodies] = self.nominal["body_inertia"][self.bodies] * scale[:, None]
        fr = rng.uniform(*c.joint_friction, size=len(dof_adr))
        m.dof_frictionloss[dof_adr] = fr
        dsc = rng.uniform(*c.joint_damping_scale, size=len(dof_adr))
        m.dof_damping[dof_adr] = self.nominal["dof_damping"][dof_adr] * dsc
        strength = float(rng.uniform(*c.motor_strength))
        pd.strength[:] = strength
        latency = int(rng.integers(c.latency_steps[0], c.latency_steps[1] + 1))
        pd.reset(latency_steps=latency)
        mu = float(rng.uniform(*c.ground_friction))
        m.geom_friction[self.terrain_geom, 0] = mu
        self.sample = {"mass_scale": float(np.mean(scale)), "friction": mu, "strength": strength, "latency": latency}
        return dict(self.sample)

    def noise(self, obs_cfg, joint_pos, joint_vel, ang_vel, gravity):
        """Sensor noise on the deployable observations (in place copies returned)."""
        if not (self.cfg.enabled and self.cfg.sensor_noise):
            return joint_pos, joint_vel, ang_vel, gravity
        rng = self.rng
        jp = joint_pos + rng.normal(0.0, obs_cfg.noise_joint_pos, size=joint_pos.shape)
        jv = joint_vel + rng.normal(0.0, obs_cfg.noise_joint_vel, size=joint_vel.shape)
        av = ang_vel + rng.normal(0.0, obs_cfg.noise_ang_vel, size=3)
        g = gravity + rng.normal(0.0, obs_cfg.noise_gravity, size=3)
        g = g / max(np.linalg.norm(g), 1e-6)
        return jp, jv, av, g


class Disturbances:
    """Pushes at the pelvis: a horizontal force of the configured magnitude in a random direction for
    ``duration_seconds``, every ``interval_seconds`` (uniform)."""

    def __init__(self, cfg, rng=None, enabled: bool | None = None):
        self.cfg, self.rng = cfg, rng or np.random.default_rng()
        self.enabled = cfg.enabled if enabled is None else enabled
        self.reset()

    def reset(self):
        self._next = float(self.rng.uniform(*self.cfg.interval_seconds)) if self.enabled else np.inf
        self._until, self._force = -1.0, np.zeros(3)

    def force(self, t: float) -> np.ndarray:
        """The force to apply at time ``t`` [N] (zeros between pushes)."""
        if not self.enabled:
            return np.zeros(3)
        if t >= self._next:
            a = self.rng.uniform(0.0, 2 * np.pi)
            self._force = self.cfg.force_xy * np.array([np.cos(a), np.sin(a), 0.0])
            self._until = t + self.cfg.duration_seconds
            self._next = self._until + float(self.rng.uniform(*self.cfg.interval_seconds))
        return self._force if t < self._until else np.zeros(3)
