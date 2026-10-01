"""ChironLab — the reusable environment: any Chiron robot, on any Chiron terrain, with any controller.

    lab = ChironLab(robot, terrain, timestep=0.001, control_dt=0.001, log_dt=0.01)
    obs = lab.reset(seed=0)
    obs = lab.step(Command(q_target={...}))              # one control step
    ep  = lab.run(controller, rules=FailureRules(course_m=1.5, v_target=0.2), settle=0.5, seed=0)
    ep.outcome, ep.log["com"], ep.save("trial.npz"), ep.to_result()

**Simulation.** MuJoCo with the options of ``SimOptions`` (1 ms, implicit-fast integration by default). Every
actuated joint is driven by its servo law (``servo.py``) at every physics step: a PD loop on the controller's
targets, clipped to the motor's torque–speed line, vectorised over joints. The controller is called every
``control_dt`` with an ``Observation`` and returns a ``Command`` (zero-order hold between calls).

**Servo integration.** Each actuator is a MuJoCo ``general`` actuator with an affine velocity bias ``b·q̇``
(b [N·m·s/rad]); every physics step its control is set to ``τ − b·q̇``, so the applied torque is exactly the
clipped servo torque τ [N·m], and ``b`` is τ's velocity derivative in the joint's current regime
(``servo.implicit_slope``), which ``implicitfast`` (and ``implicit``) integrate implicitly: ``−kd`` for an unclipped
PD joint; ``−τ_stall/ω₀``, the torque–speed line's slope, for a joint saturated while pulling; 0 at or beyond the
no-load speed (τ ≡ 0); 0 (explicit) for a joint saturated while braking, where the line's slope is positive —
unless the torque would stop the joint within the step (judged with its articulated inertia from MuJoCo's
factorisation of M), when it gets the line's slope of the far side of zero speed. A saturated joint is thus
accelerated by τ/I (I [kg·m²]), not by τ/(I + h·kd) as when −kd stayed implicit whatever the clip; unclipped joints
are integrated bit for bit as before. ``reset`` restores the compiled ``b = −kd``. (With ``Euler`` or ``RK4``, b
only cancels out of the applied torque.)

**Timing of observations.** ``q``, ``qd`` and ``t`` are the current state. Body poses, velocities, contact forces
and Jacobians come from MuJoCo's last forward pass — the state one physics step earlier (≤ 1 ms), as from a real
sensor pipeline; right after ``reset`` they are current. The **log** is fully consistent: each sample holds the
state at its time ``t`` together with the contact forces and the torques applied over the step from ``t``.

**Episode log.** See ``Episode`` and docs: the dict format shared with ``vegeta.chiron.metrics``.

**Failure rules.** ``FailureRules`` implements the protocol's §5 generically (progress along +x).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .result import Result
from .robot import TERRAIN_NAME, Robot, RobotMeta, SimOptions, terrain_geometry
from .terrain import Flat, Terrain

__all__ = ["ChironLab", "Command", "Disturbance", "FailureRules", "Observation", "Episode"]

LOG_FORMAT_VERSION = 1
#: Recorded in every Episode's meta: how the servo law is integrated (module docstring, "Servo integration").
SERVO_INTEGRATION = "implicit-slope/1"


# ----------------------------------------------------------------------------------------------- small types
@dataclass
class Command:
    """What a controller asks of the servos until its next call.

    ``q_target``: dict joint → angle [rad] (joints left out keep their previous target) or an array over
    ``lab.actuated_joints``; ``qd_target`` [rad/s] and ``tau_ff`` [N·m] likewise (None = 0). ``leg_phase``
    (cycle fraction per foot, in ``lab.feet`` order) and ``leg_stance`` (commanded stance per foot) are optional
    and only logged (``leg_phase``, ``leg_stance_cmd``)."""

    q_target: object
    qd_target: object = None
    tau_ff: object = None
    leg_phase: object = None
    leg_stance: object = None


@dataclass
class Disturbance:
    """A force on a body's centre of mass: ``impulse`` [N·s] delivered over ``duration`` [s] (constant force
    ``impulse/duration``) or a constant ``force`` [N], along ``direction`` (world, normalised), starting at
    walking time ``t_start`` [s]. The delivered impulse is exact: the force is applied for a whole number of
    physics steps and scaled to it."""

    body: str
    t_start: float
    duration: float
    impulse: float | None = None
    force: float | None = None
    direction: tuple = (0.0, 1.0, 0.0)

    def __post_init__(self):
        if (self.impulse is None) == (self.force is None):
            raise ValueError("give exactly one of impulse or force")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        n = np.linalg.norm(self.direction)
        if not n > 0:
            raise ValueError("direction must be non-zero")

    @property
    def unit(self) -> np.ndarray:
        d = np.asarray(self.direction, dtype=float)
        return d / np.linalg.norm(d)

    @property
    def impulse_Ns(self) -> float:
        return float(self.impulse) if self.impulse is not None else float(self.force) * self.duration


@dataclass
class FailureRules:
    """Exactly one outcome per trial (the pre-registered stability protocol in docs/, §5;
    generic: the COM progresses along +x).

    * ``success`` — the whole-robot COM passes x = ``course_m``;
    * ``fall`` — any logged body tilts more than ``max_tilt_deg`` from vertical, or the COM stays lower than
      ``min_height_fraction`` × the nominal hip height above the ground beneath it for ``low_height_time`` s;
    * ``stall`` — over any ``stall_window`` s window that starts after the first ``stall_grace`` s of walking,
      the COM advances less than ``stall_fraction × v_target × stall_window``;
    * ``off_course`` — |COM y| > ``lateral_limit``;
    * ``timeout`` — none of the above by ``timeout`` (default ``2 × course_m / v_target + 2`` s).

    Rules are evaluated at every log sample; when several fire at the same sample the order is fall,
    off_course, success, stall, timeout. ``nominal_hip_height`` None = the lab's (from the robot's standing pose).
    """

    course_m: float
    max_tilt_deg: float = 60.0
    min_height_fraction: float = 0.4
    low_height_time: float = 0.5
    stall_window: float = 3.0
    stall_fraction: float = 0.1
    stall_grace: float = 2.0
    lateral_limit: float = 0.5
    timeout: float | None = None
    v_target: float | None = None
    nominal_hip_height: float | None = None

    def timeout_s(self) -> float:
        if self.timeout is not None:
            return float(self.timeout)
        if not self.v_target:
            raise ValueError("FailureRules needs v_target or an explicit timeout")
        return 2.0 * self.course_m / self.v_target + 2.0


# ----------------------------------------------------------------------------------------------- observation
_OBS_GROUPS = {
    "q": "joints", "qd": "joints", "tau": "joints", "q_act": "joints", "qd_act": "joints",
    "body_pos": "bodies", "body_quat": "bodies", "body_angvel": "bodies", "body_linvel": "bodies",
    "base_pos": "base", "base_quat": "base",
    "com": "com", "com_vel": "com", "ang_mom": "com",
    "foot_pos": "foot_pos",
    "foot_force": "contacts", "foot_normal": "contacts", "foot_normal_force": "normal_force",
    "foot_contact_pos": "contacts", "foot_in_contact": "contacts", "belly_contact": "contacts",
    "foot_jac": "foot_jac",
}


class Observation:
    """A snapshot of the lab for a controller; fields are computed on first access.

    ``t`` walking time [s]; ``q``, ``qd``, ``tau`` (J,) every joint (``lab.joint_names``), ``q_act``/``qd_act``
    the actuated ones (``lab.actuated_joints``); ``body_pos`` (B,3) COM, ``body_quat`` (B,4) w,x,y,z,
    ``body_angvel`` (B,3) body frame, ``body_linvel`` (B,3) world, for ``lab.bodies``; ``base_pos``/``base_quat``
    of the root; ``com``, ``com_vel``, ``ang_mom`` (about the COM, world); ``foot_pos`` (F,3) pad centres,
    ``foot_force`` (F,3) contact force on each foot (world, N), ``foot_normal`` (F,3) from the ground into the
    foot (nan without contact), ``foot_normal_force`` (F,) [N], ``foot_contact_pos`` (F,3), ``foot_in_contact``
    (F,), ``foot_jac`` list of (3, n_leg) Jacobians, ``belly_contact`` (B,).

    Valid until the lab steps again (reading a new field afterwards raises); ``as_dict()`` freezes everything.
    """

    __slots__ = ("_lab", "_stamp", "_cache", "t")

    def __init__(self, lab: "ChironLab"):
        self._lab = lab
        self._stamp = lab._stamp
        self._cache = {}
        self.t = lab.time

    def _get(self, key):
        c = self._cache
        if key not in c:
            if self._lab._stamp != self._stamp:
                raise RuntimeError("stale Observation: the lab has stepped since it was taken (use as_dict())")
            c.update(self._lab._compute(_OBS_GROUPS[key]))
        return c[key]

    def __getattr__(self, key):
        if key in _OBS_GROUPS:
            return self._get(key)
        raise AttributeError(key)

    def joint(self, name: str) -> float:
        return float(self.q[self._lab._joint_index[name]])

    def as_dict(self) -> dict:
        for key in _OBS_GROUPS:
            self._get(key)
        out = dict(self._cache)
        out["t"] = self.t
        return out

    def __dir__(self):
        return sorted(set(_OBS_GROUPS) | {"t", "joint", "as_dict"})


# ----------------------------------------------------------------------------------------------- episode
class Episode:
    """One run: ``log`` (the episode log dict), ``outcome`` (success, reason, t_end, distance_m, x_end, detail)
    and ``meta`` (settings, timing, versions). ``save``/``load`` use a compressed .npz; ``to_result`` gives the
    common Vegeta Result (kind ``chiron.episode``).

    The log (T samples every ``log_dt`` of walking time; B logged bodies, F feet, J hinge/slide joints; world z up,
    the route along +x from x = 0):

    * ``t`` (T,) walking time [s]; ``bodies``, ``body_group`` (B names); ``body_pos`` (T,B,3) body COM;
      ``body_quat`` (T,B,4) w,x,y,z of the body frame (x forward, y left, z up); ``body_angvel`` (T,B,3) in the
      body frame; ``body_linvel`` (T,B,3) COM velocity, world;
    * ``com``, ``com_vel``, ``ang_mom`` (T,3) whole robot (angular momentum about the COM, world); ``total_mass``;
      ``gravity`` (3,); ``terrain_height_under_com`` (T,);
    * ``feet`` (F names), ``foot_body`` (F,) index into ``bodies``, ``foot_group``, ``foot_joints``; ``foot_pos``
      (T,F,3) pad centre; ``foot_force`` (T,F,3) total contact force ON the foot (world, N; 0 without contact);
      ``foot_normal`` (T,F,3) unit normal from the ground into the foot and ``foot_contact_pos`` (T,F,3) (both
      force-weighted over the foot's contacts; nan without contact); ``foot_mu`` (F,); ``foot_jac`` (T,F,3,n)
      d(foot centre)/d(q) of the foot's joints (zero-padded to the longest leg);
    * ``joints``, ``joint_kind`` (tags), ``joint_active`` (J,), ``joint_leg`` (J,) foot index or -1, ``q_range``
      (J,2) (nan = unlimited); ``q``, ``qd``, ``tau`` (T,J) angle, speed, applied servo torque (0 if passive);
      ``tau_stall``, ``tau_rated``, ``qd_noload``, ``i_stall``, ``voltage`` (J,) servo data (nan if passive);
    * ``belly_contact`` (T,B) any 'body' geom of that body touching the terrain;
    * ``leg_phase`` (T,F), ``leg_stance_cmd`` (T,F) when the controller's Command reports them;
    * ``v_target``, ``course_m``, ``nominal_hip_height``, ``disturbances`` (t_start in walking time), ``robot``,
      ``treatment``, ``controller``, ``terrain`` (spec dict), ``seed``; ``geom_pose`` with ``log_geoms=True``.

    Each sample is consistent: the state at ``t`` with the contact forces and torques of the step from ``t``.
    ``log=False`` runs keep only ``t``, bodies, COM and terrain height (plus the static fields).
    """

    def __init__(self, log: dict, outcome: dict, meta: dict):
        self.log, self.outcome, self.meta = log, outcome, meta
        self.path: Path | None = None

    @property
    def success(self) -> bool:
        return bool(self.outcome.get("success"))

    @property
    def reason(self) -> str:
        return self.outcome.get("reason", "")

    def __repr__(self) -> str:
        o = self.outcome
        return (f"Episode({self.log.get('robot', '?')}: {o.get('reason')} at t={o.get('t_end', float('nan')):.2f} s, "
                f"distance {o.get('distance_m', float('nan')):.3f} m, {len(self.log.get('t', []))} samples)")

    def save(self, path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays, plain = {}, {}

        def put(prefix, obj, sink):
            for k, v in obj.items():
                if isinstance(v, np.ndarray) and v.dtype != object:
                    arrays[prefix + k] = v
                elif isinstance(v, dict) and any(isinstance(x, np.ndarray) for x in v.values()):
                    sub = {}
                    sink[k] = {"__nested__": prefix + k + "/"}
                    put(prefix + k + "/", v, sub)
                    sink[k]["plain"] = sub
                else:
                    sink[k] = v

        put("log/", self.log, plain)
        header = {"format": "chiron.episode", "version": LOG_FORMAT_VERSION, "log": plain,
                  "outcome": self.outcome, "meta": self.meta}
        arrays["__json__"] = np.array(json.dumps(header, default=_json_default))
        np.savez_compressed(path, **arrays)
        self.path = path
        return path

    @classmethod
    def load(cls, path) -> "Episode":
        path = Path(path)
        with np.load(path, allow_pickle=False) as f:
            header = json.loads(str(f["__json__"]))
            arrays = {k: f[k] for k in f.files if k != "__json__"}

        def get(prefix, plain):
            out = {}
            for k, v in plain.items():
                if isinstance(v, dict) and "__nested__" in v:
                    out[k] = get(v["__nested__"], v.get("plain", {}))
                else:
                    out[k] = v
            for key, arr in arrays.items():
                if key.startswith(prefix) and "/" not in key[len(prefix):]:
                    out[key[len(prefix):]] = arr
            return out

        ep = cls(get("log/", header["log"]), header["outcome"], header["meta"])
        ep.path = path
        return ep

    def to_result(self) -> Result:
        o, log = self.outcome, self.log
        t_end = o.get("t_end")
        dist = o.get("distance_m")
        metrics = {
            "outcome": o.get("reason"),
            "success": o.get("success"),
            "t_end_s": t_end,
            "distance_m": dist,
            "mean_speed_m_s": (dist / t_end) if (t_end and dist is not None and t_end > 0) else None,
            "total_mass_kg": log.get("total_mass"),
            "n_samples": int(len(log.get("t", []))),
            "wall_time_s": self.meta.get("wall_time_s"),
            "realtime_factor": self.meta.get("realtime_factor"),
        }
        msgs = [o.get("detail", "")] if o.get("detail") else []
        res = Result(kind="chiron.episode", metrics=metrics, messages=msgs,
                     duration_s=float(self.meta.get("wall_time_s") or 0.0),
                     metadata={"robot": log.get("robot"), "treatment": log.get("treatment"),
                               "controller": log.get("controller"), "terrain": log.get("terrain"),
                               "seed": log.get("seed"), "units": "SI (m, kg, s, N, N·m, rad)", **self.meta})
        if self.path is not None:
            res.artifacts["episode"] = self.path
        return res


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ----------------------------------------------------------------------------------------------- the lab
class ChironLab:
    """The environment. ``robot``: a ``Robot`` (Link tree or ``Robot.from_mjcf``); ``terrain``: a ``Terrain``.

    ``timestep`` [s] physics step; ``control_dt`` [s] controller period (a multiple of the timestep);
    ``log_dt`` [s] log period (a multiple of the timestep); ``log_geoms`` also logs every geom's pose (for
    ``viz``); ``course_extent`` (x0, x1, y0, y1) [m] and ``heightfield_cell`` [m] the height field; the other
    keywords are ``SimOptions`` fields (or pass ``options=SimOptions(...)``, which then wins).
    """

    def __init__(self, robot: Robot, terrain: Terrain | None = None, *, timestep=0.001, control_dt=0.001,
                 log_dt=0.01, log_geoms=False, course_extent=(-1.0, 3.0, -1.0, 1.0), heightfield_cell=0.005,
                 integrator="implicitfast", cone="pyramidal", impratio=1.0, condim=3, contact_solref=None,
                 contact_solimp=None, iterations=100, tolerance=1e-8, noslip_iterations=0, self_collision=False,
                 flat_as_plane=True, gravity=(0.0, 0.0, -9.81), options: SimOptions | None = None):
        import mujoco  # noqa: F401  (fail early with a clear message when MuJoCo is missing)

        if options is None:
            options = SimOptions(timestep=timestep, gravity=tuple(gravity), integrator=integrator, cone=cone,
                                 impratio=impratio, iterations=iterations, tolerance=tolerance,
                                 noslip_iterations=noslip_iterations, condim=condim, contact_solref=contact_solref,
                                 contact_solimp=contact_solimp, self_collision=self_collision,
                                 course_extent=tuple(course_extent), heightfield_cell=heightfield_cell,
                                 flat_as_plane=flat_as_plane)
        self.options = options
        self.robot = robot
        self.terrain = terrain if terrain is not None else Flat()
        self.log_geoms = bool(log_geoms)
        h = options.timestep
        self._n_ctrl = _ratio(control_dt, h, "control_dt")
        self._n_log = _ratio(log_dt, h, "log_dt")
        self.control_dt = self._n_ctrl * h
        self.log_dt = self._n_log * h
        self._disturbances: list[Disturbance] = []
        self._transient: list[Disturbance] = []
        self._build()
        self.reset()

    @classmethod
    def from_mjcf(cls, xml: str, meta: RobotMeta, terrain: Terrain | None = None, *, assets=None, name=None,
                  **kwargs) -> "ChironLab":
        """A lab for a third-party MJCF model described by ``meta`` (see ``Robot.from_mjcf``)."""
        return cls(Robot.from_mjcf(xml, meta, name=name, assets=assets), terrain, **kwargs)

    # ---- model construction
    def _build(self):
        import mujoco

        opt, robot, terrain = self.options, self.robot, self.terrain
        self.meta = robot.meta()
        tinfo = terrain_geometry(terrain, opt)
        if robot.is_mjcf:
            robot.validate()
            spec = robot._build_spec(terrain, opt)
            m = spec.compile()
        else:
            xml = robot.to_mjcf(terrain, opt, heightfield="deferred")
            m = mujoco.MjModel.from_xml_string(xml)
        if tinfo["type"] == "hfield":
            hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, TERRAIN_NAME)
            adr, n = m.hfield_adr[hid], m.hfield_nrow[hid] * m.hfield_ncol[hid]
            m.hfield_data[adr:adr + n] = tinfo["data"].ravel()
        self._tinfo = tinfo
        self.model = m
        self.data = mujoco.MjData(m)
        self._index()
        self._nominal()

    def set_terrain(self, terrain: Terrain) -> None:
        """Replace the terrain (rebuilds the MuJoCo model; disturbances are kept) and reset."""
        self.terrain = terrain
        self._build()
        self.reset()

    @property
    def xml(self) -> str:
        """The complete MJCF of the robot on its terrain (height field inline)."""
        return self.robot.to_mjcf(self.terrain, self.options)

    def _index(self):
        import mujoco

        m, meta = self.model, self.meta
        J = mujoco.mjtJoint
        name = lambda kind, i: mujoco.mj_id2name(m, kind, i)  # noqa: E731
        free = [j for j in range(m.njnt) if m.jnt_type[j] == J.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError(f"Chiron needs exactly one free joint (the robot's root); found {len(free)}")
        if any(m.jnt_type[j] == J.mjJNT_BALL for j in range(m.njnt)):
            raise ValueError("ball joints are not supported; use hinges")
        self._root_jnt = free[0]
        self._root_body = int(m.jnt_bodyid[free[0]])
        self._root_qadr = int(m.jnt_qposadr[free[0]])
        self._root_dadr = int(m.jnt_dofadr[free[0]])
        jids = [j for j in range(m.njnt) if int(m.jnt_type[j]) in (int(J.mjJNT_HINGE), int(J.mjJNT_SLIDE))]
        self.joint_names = [name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in jids]
        if any(n is None for n in self.joint_names):
            raise ValueError("every hinge/slide joint needs a name")
        self._joint_index = {n: i for i, n in enumerate(self.joint_names)}
        self._jid = np.array(jids, dtype=int)
        self._jq = m.jnt_qposadr[self._jid].astype(int)
        self._jd = m.jnt_dofadr[self._jid].astype(int)
        # actuated joints, in actuator (= ctrl) order
        servo_names = set(meta.servos)
        unknown = servo_names - set(self.joint_names)
        if unknown:
            raise ValueError(f"servos on unknown joints: {sorted(unknown)}")
        act = []
        for a in range(m.nu):
            an = name(mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
            if not an.startswith("servo:"):
                raise ValueError(f"unexpected actuator {an!r}; Chiron adds its own servos")
            act.append(an[len("servo:"):])
        if set(act) != servo_names or len(act) != len(servo_names):
            raise ValueError("actuators do not match the servo list")
        self.actuated_joints = act
        self._act_index = {n: i for i, n in enumerate(act)}
        self._act_j = np.array([self._joint_index[n] for n in act], dtype=int)
        self._aq = self._jq[self._act_j]
        self._ad = self._jd[self._act_j]
        servos = [meta.servos[n] for n in act]
        arr = lambda f: np.array([getattr(s, f) for s in servos], dtype=float)  # noqa: E731
        self._kp, self._kd = arr("kp"), arr("kd")
        self._stall, self._w0 = arr("stall_torque"), arr("no_load_speed")
        self._inv_w0 = 1.0 / self._w0 if len(servos) else np.zeros(0)
        self._rated, self._istall, self._volt = arr("rated_torque"), arr("stall_current"), arr("voltage")
        nA = len(act)
        self._q, self._qd = np.zeros(nA), np.zeros(nA)
        self._tau, self._work = np.zeros(nA), np.zeros(nA)
        self._qt, self._qdt, self._ff = np.zeros(nA), np.zeros(nA), np.zeros(nA)
        # servo integration (see _simulate): b = ∂τ/∂q̇ [N·m·s/rad] per actuator, carried by the actuator's
        # velocity bias (actuator_biasprm[:, 2], a view into the model) so implicitfast integrates it implicitly
        self._lim, self._slope = np.zeros(nA), np.zeros(nA)
        self._clip, self._line, self._brake, self._keep = (np.zeros(nA, dtype=bool) for _ in range(4))
        self._neg_kd, self._neg_line = -self._kd, -self._stall * self._inv_w0
        self._bias_vel = m.actuator_biasprm[:, 2]
        self._bias_vel[:] = self._neg_kd
        self._bias_is_kd = True
        # bodies
        self.bodies = list(meta.logged_bodies)
        self._bid = np.array([self._body_id(b) for b in self.bodies], dtype=int)
        self.body_groups = [meta.body_groups.get(b, b) for b in self.bodies]
        self._body_index = {b: i for i, b in enumerate(self.bodies)}
        # terrain and geoms
        self._terrain_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, TERRAIN_NAME)
        # feet
        self.feet = [f.name for f in meta.feet]
        self._foot_gid = np.array([self._geom_id(f.geom) for f in meta.feet], dtype=int)
        self._foot_of_geom = np.full(m.ngeom, -1, dtype=int)
        self._foot_of_geom[self._foot_gid] = np.arange(len(self.feet))
        for f in meta.feet:
            if f.body not in self._body_index:
                raise ValueError(f"foot {f.name}: body {f.body!r} is not a logged body")
        self._foot_body = np.array([self._body_index[f.body] for f in meta.feet], dtype=int)
        self.foot_groups = [self.body_groups[i] for i in self._foot_body]
        self.foot_joints = [list(f.joints) for f in meta.feet]
        self._foot_dofs = [np.array([self._jd[self._joint_index[j]] for j in f.joints], dtype=int) for f in meta.feet]
        self._nleg = max((len(d) for d in self._foot_dofs), default=0)
        # leg-joint table for the vectorised foot Jacobians (padding: -1); every leg joint must move its foot
        self._leg_jid = np.full((len(meta.feet), self._nleg), -1, dtype=int)
        for i, f in enumerate(meta.feet):
            chain = set()
            b = int(m.geom_bodyid[self._foot_gid[i]])
            while b > 0:
                chain.add(b)
                b = int(m.body_parentid[b])
            for c, jn in enumerate(f.joints):
                jid = int(self._jid[self._joint_index[jn]])
                if int(m.jnt_bodyid[jid]) not in chain:
                    raise ValueError(f"foot {f.name}: joint {jn!r} does not move the foot geom {f.geom!r}")
                self._leg_jid[i, c] = jid
        self._leg_valid = self._leg_jid >= 0
        self._leg_slide = np.zeros_like(self._leg_valid)
        if self._leg_jid.size:
            self._leg_slide = self._leg_valid & (m.jnt_type[np.maximum(self._leg_jid, 0)] == int(J.mjJNT_SLIDE))
        self._foot_mu = m.geom_friction[self._foot_gid, 0].copy() if len(self.feet) else np.zeros(0)
        # belly geoms -> logged body index
        self._belly_of_geom = np.full(m.ngeom, -1, dtype=int)
        if meta.belly_geoms is not None:
            for g, b in meta.belly_geoms.items():
                if b in self._body_index:
                    self._belly_of_geom[self._geom_id(g)] = self._body_index[b]
        else:
            for g in range(m.ngeom):
                b = int(m.geom_bodyid[g])
                if (m.geom_contype[g] or m.geom_conaffinity[g]) and g != self._terrain_gid and \
                        self._foot_of_geom[g] < 0 and b in self._bid:
                    self._belly_of_geom[g] = int(np.nonzero(self._bid == b)[0][0])
        # joint metadata
        legs = meta.leg_of_joint()
        foot_idx = {f: i for i, f in enumerate(self.feet)}
        self.joint_tags = [meta.joint_tags.get(n, "other") for n in self.joint_names]
        self._joint_active = np.array([n in servo_names for n in self.joint_names])
        self._joint_leg = np.array([foot_idx.get(legs.get(n), -1) for n in self.joint_names], dtype=int)
        rng = m.jnt_range[self._jid].copy()
        rng[~m.jnt_limited[self._jid].astype(bool)] = np.nan
        self._q_range = rng
        # geoms for rendering
        self._viz_gid = np.array([g for g in range(m.ngeom) if g != self._terrain_gid and
                                  not (name(mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("pm:") and
                                  m.geom_bodyid[g] != 0], dtype=int)
        self._jacp = np.zeros((3, m.nv))
        self._pyr4 = np.arange(4)
        self.total_mass = float(m.body_subtreemass[self._root_body])
        self._stamp = 0
        self._k = 0

    def _body_id(self, b):
        import mujoco

        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
        if i < 0:
            raise ValueError(f"no body {b!r}")
        return i

    def _geom_id(self, g):
        import mujoco

        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if i < 0:
            raise ValueError(f"no geom {g!r}")
        return i

    def _nominal(self):
        """Standing pose: joint angles, base height (lowest foot on z = 0), COM offset, hip height."""
        import mujoco

        m, meta = self.model, self.meta
        d = mujoco.MjData(m)
        q0 = np.zeros(len(self.joint_names))
        for n, v in meta.nominal_qpos.items():
            if n not in self._joint_index:
                raise ValueError(f"nominal_qpos names unknown joint {n!r}")
            q0[self._joint_index[n]] = v
        self.nominal_q = q0
        d.qpos[self._root_qadr:self._root_qadr + 7] = [0, 0, 0, 1, 0, 0, 0]
        d.qpos[self._jq] = q0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        gids = self._foot_gid if len(self._foot_gid) else np.array(
            [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g] != 0])
        low = min(_geom_lowest_z(m, d, g) for g in gids) if len(gids) else 0.0
        self.nominal_base_height = float(meta.nominal_base_height) if meta.nominal_base_height is not None else -low
        self._com_offset = d.subtree_com[self._root_body].copy()
        self._foot_xy0 = d.geom_xpos[self._foot_gid, :2].copy() if len(self.feet) else np.zeros((0, 2))
        if meta.nominal_hip_height is not None:
            self.nominal_hip_height = float(meta.nominal_hip_height)
        elif len(meta.feet):
            hips = [d.xanchor[self._jid[self._joint_index[f.joints[0]]], 2] for f in meta.feet if f.joints]
            self.nominal_hip_height = float(np.mean(hips) + self.nominal_base_height) if hips else \
                float(self._com_offset[2] + self.nominal_base_height)
        else:
            self.nominal_hip_height = float(self._com_offset[2] + self.nominal_base_height)

    # ---- state
    @property
    def timestep(self) -> float:
        return self.options.timestep

    @property
    def time(self) -> float:
        """Walking time [s] (0 at the end of ``run``'s settle phase; after ``reset``, time since reset)."""
        return self._k * self.options.timestep

    def nominal_command(self) -> Command:
        """Hold the standing pose."""
        return Command(q_target=self.nominal_q[self._act_j].copy())

    def reset(self, seed=None, base_pos=None, base_yaw=0.0, qpos=None, *, base_quat=None) -> Observation:
        """Put the robot in its standing pose at rest and return an Observation.

        ``base_pos``: None = the standing COM above (0, 0) with the root at its nominal height above the highest
        terrain point under the feet; (x, y) = the root there at that height; (x, y, z) = the root exactly there.
        ``base_yaw`` [rad] about z, or ``base_quat`` (w, x, y, z) for any orientation. ``qpos``: joint → angle
        overrides of the nominal pose. ``seed`` is recorded (the lab itself draws no random numbers).
        One-shot impulses (``apply_impulse``) are cleared; scheduled disturbances are kept.
        """
        import mujoco

        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if base_quat is not None:
            quat = np.asarray(base_quat, dtype=float)
            quat = quat / np.linalg.norm(quat)
        else:
            quat = np.array([math.cos(base_yaw / 2), 0.0, 0.0, math.sin(base_yaw / 2)])
        R = _quat_to_mat(quat)
        q = self.nominal_q.copy()
        for n, v in (qpos or {}).items():
            q[self._joint_index[n]] = v
        d.qpos[self._jq] = q
        if base_pos is None or len(base_pos) == 2:
            if base_pos is None:
                xy = -(R @ self._com_offset)[:2]
            else:
                xy = np.asarray(base_pos, dtype=float)
            if len(self._foot_xy0):
                pts = xy + (R[:2, :2] @ self._foot_xy0.T).T
                ground = float(np.max(self.terrain.height(pts[:, 0], pts[:, 1])))
            else:
                ground = float(self.terrain.height(xy[0], xy[1]))
            pos = np.array([xy[0], xy[1], ground + self.nominal_base_height])
        else:
            pos = np.asarray(base_pos, dtype=float)
        d.qpos[self._root_qadr:self._root_qadr + 3] = pos
        d.qpos[self._root_qadr + 3:self._root_qadr + 7] = quat
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self._seed = seed
        self._k = 0
        self._stamp += 1
        self._qt[:] = q[self._act_j]
        self._qdt[:] = 0.0
        self._ff[:] = 0.0
        self._tau[:] = 0.0
        self._bias_vel[:] = self._neg_kd                              # the compiled model's values
        self._bias_is_kd = True
        self._leg_phase = None
        self._leg_stance = None
        self._transient = []
        self.data.xfrc_applied[:] = 0.0
        return Observation(self)

    def observe(self, sync: bool = False) -> Observation:
        """The current Observation. ``sync=True`` first recomputes MuJoCo's derived quantities for the current
        state (mj_forward; it does not change the trajectory), so every field is at the same instant."""
        if sync:
            import mujoco

            mujoco.mj_forward(self.model, self.data)
            self._stamp += 1
        return Observation(self)

    # ---- commands and disturbances
    def _set_command(self, cmd):
        if cmd is None:
            return
        if not isinstance(cmd, Command):
            cmd = Command(q_target=cmd)
        self._fill(self._qt, cmd.q_target, keep=True)
        self._fill(self._qdt, cmd.qd_target, keep=False)
        self._fill(self._ff, cmd.tau_ff, keep=False)
        self._leg_phase = None if cmd.leg_phase is None else np.asarray(cmd.leg_phase, dtype=float)
        self._leg_stance = None if cmd.leg_stance is None else np.asarray(cmd.leg_stance, dtype=bool)

    def _fill(self, buf, value, keep):
        if value is None:
            if not keep:
                buf[:] = 0.0
            return
        if isinstance(value, dict):
            if not keep:
                buf[:] = 0.0
            idx = self._act_index
            for n, v in value.items():
                try:
                    buf[idx[n]] = v
                except KeyError:
                    raise KeyError(f"{n!r} is not an actuated joint") from None
            return
        arr = np.asarray(value, dtype=float)
        if arr.shape != buf.shape:
            raise ValueError(f"command arrays need {buf.size} values (lab.actuated_joints)")
        buf[:] = arr

    def add_disturbance(self, disturbance: Disturbance) -> None:
        """Schedule a disturbance (walking time). It stays scheduled across resets; ``clear_disturbances``."""
        self._body_id(disturbance.body)
        self._disturbances.append(disturbance)

    def clear_disturbances(self) -> None:
        self._disturbances = []
        self._transient = []
        self.data.xfrc_applied[:] = 0.0

    @property
    def disturbances(self) -> list:
        return list(self._disturbances)

    def apply_impulse(self, body: str, impulse, duration: float | None = None) -> Disturbance:
        """Deliver ``impulse`` [N·s] (a 3-vector, world) to ``body``'s COM starting at the next physics step,
        over ``duration`` [s] (default: one physics step). One-shot: cleared by ``reset``."""
        J = np.asarray(impulse, dtype=float)
        mag = float(np.linalg.norm(J))
        if mag == 0:
            raise ValueError("impulse must be non-zero")
        dur = self.timestep if duration is None else float(duration)
        dist = Disturbance(body=body, t_start=self.time, duration=dur, impulse=mag, direction=tuple(J / mag))
        self._body_id(body)
        self._transient.append(dist)
        return dist

    def _apply_disturbances(self, k):
        xf = self.data.xfrc_applied
        h = self.timestep
        xf[:, :3] = 0.0
        for dist in self._disturbances + self._transient:
            k0 = int(round(dist.t_start / h))
            n = max(1, int(round(dist.duration / h)))
            if k0 <= k < k0 + n:
                mag = dist.impulse / (n * h) if dist.impulse is not None else dist.force
                xf[self._body_id(dist.body), :3] += mag * dist.unit

    # ---- stepping
    def step(self, command=None) -> Observation:
        """Apply ``command`` (Command, dict or array; None = keep the last) for one control step."""
        self._set_command(command)
        self._simulate(self._n_ctrl)
        return Observation(self)

    def _simulate(self, nsteps, controller=None, recorder=None) -> bool:
        """The hot loop. Returns True when ``recorder`` stopped it (an outcome was reached)."""
        import mujoco

        m, d = self.model, self.data
        mj_step = mujoco.mj_step
        take, subtract, multiply, add = np.take, np.subtract, np.multiply, np.add
        absolute, maximum, minimum, negative = np.abs, np.maximum, np.minimum, np.negative
        qpos, qvel, ctrl = d.qpos, d.qvel, d.ctrl
        aq, ad = self._aq, self._ad
        q, qd, tau, work = self._q, self._qd, self._tau, self._work
        qt, qdt, ff = self._qt, self._qdt, self._ff
        kp, kd, stall, inv_w0 = self._kp, self._kd, self._stall, self._inv_w0
        w0, lim, slope, clip, line = self._w0, self._lim, self._slope, self._clip, self._line
        brake, keep = self._brake, self._keep
        neg_kd, neg_line, bias_vel, dinv, h = self._neg_kd, self._neg_line, self._bias_vel, d.qLDiagInv, m.opt.timestep
        greater, less, copyto = np.greater, np.less, np.copyto
        n_ctrl, n_log = self._n_ctrl, self._n_log
        has_act = len(aq) > 0
        for _ in range(nsteps):
            k = self._k
            if controller is not None and k % n_ctrl == 0:
                cmd = controller(Observation(self))
                if cmd is not None:
                    self._set_command(cmd)
            rec = recorder is not None and k >= 0 and k % n_log == 0
            if rec:
                recorder.pre(k)
            if has_act:
                take(qpos, aq, out=q)
                take(qvel, ad, out=qd)
                subtract(qt, q, out=tau)
                tau *= kp
                subtract(qdt, qd, out=work)
                work *= kd
                tau += work
                tau += ff                                             # τ_pd
                absolute(qd, out=lim)
                lim *= inv_w0
                subtract(1.0, lim, out=lim)
                maximum(lim, 0.0, out=lim)
                lim *= stall                                          # τ_max(q̇) = τ_stall·max(0, 1 − |q̇|/ω₀)
                absolute(tau, out=work)
                greater(work, lim, out=clip)                          # clipped joints
                minimum(tau, lim, out=tau)
                negative(lim, out=work)
                maximum(tau, work, out=tau)                           # τ = clip(τ_pd, ±τ_max)
                if clip.any():                       # saturated: b = ∂τ/∂q̇ of the regime (servo.implicit_slope)
                    copyto(slope, neg_kd)                             # unclipped: −kd
                    copyto(slope, 0.0, where=clip)                    # clipped at/beyond ω₀ (τ ≡ 0): 0
                    absolute(qd, out=lim)
                    less(lim, w0, out=line)
                    line &= clip                                      # clipped on the torque–speed line
                    multiply(tau, qd, out=work)
                    less(work, 0.0, out=brake)
                    brake &= line                                     # ... and braking
                    if brake.any():                                   # explicit, unless it stops within the step
                        take(dinv, ad, out=work)                      # 1/D_jj [1/(kg·m²)], M = LᵀDL (last step)
                        work *= tau
                        absolute(work, out=work)
                        work *= h                                     # |Δq̇| the torque alone gives in a step
                        less(work, lim, out=keep)
                        keep &= brake                                 # braking that does not stop: explicit
                        line ^= keep
                    copyto(slope, neg_line, where=line)               # pulling on the line: −τ_stall/ω₀
                    bias_vel[:] = slope
                    self._bias_is_kd = False
                    multiply(slope, qd, out=work)
                    subtract(tau, work, out=ctrl)                     # applied: ctrl + b·q̇ = τ exactly
                else:
                    if not self._bias_is_kd:
                        bias_vel[:] = neg_kd
                        self._bias_is_kd = True
                    multiply(kd, qd, out=work)
                    add(tau, work, out=ctrl)
            if self._disturbances or self._transient:                 # may be added by a controller
                self._apply_disturbances(k)
            mj_step(m, d)
            self._k = k + 1
            self._stamp += 1
            if rec and recorder.post(k):
                return True
        return False

    # ---- derived quantities (Observation fields and the log)
    def _compute(self, group: str) -> dict:
        import mujoco

        m, d = self.model, self.data
        if group == "joints":
            q = d.qpos[self._jq]
            qd = d.qvel[self._jd]
            tau = np.zeros(len(self.joint_names))
            tau[self._act_j] = self._tau
            return {"q": q, "qd": qd, "tau": tau, "q_act": q[self._act_j], "qd_act": qd[self._act_j]}
        if group == "bodies":
            return dict(zip(("body_pos", "body_quat", "body_angvel", "body_linvel"), self._body_state()))
        if group == "base":
            a = self._root_qadr
            return {"base_pos": d.qpos[a:a + 3].copy(), "base_quat": d.qpos[a + 3:a + 7].copy()}
        if group == "com":
            mujoco.mj_subtreeVel(m, d)
            r = self._root_body
            return {"com": d.subtree_com[r].copy(), "com_vel": d.subtree_linvel[r].copy(),
                    "ang_mom": d.subtree_angmom[r].copy()}
        if group == "foot_pos":
            return {"foot_pos": d.geom_xpos[self._foot_gid].copy()}
        if group == "contacts":
            f, n, fn, p, inc, belly = self._contacts()
            return {"foot_force": f, "foot_normal": n, "foot_normal_force": fn, "foot_contact_pos": p,
                    "foot_in_contact": inc, "belly_contact": belly}
        if group == "normal_force":
            return {"foot_normal_force": self._foot_normal_forces()}
        if group == "foot_jac":
            jac = self._foot_jacobians()
            return {"foot_jac": [jac[i, :, :len(self._foot_dofs[i])].copy() for i in range(len(self.feet))]}
        raise KeyError(group)

    def _body_state(self):
        d = self.data
        b = self._bid
        pos = d.xipos[b].copy()
        quat = d.xquat[b].copy()
        R = d.xmat[b].reshape(-1, 3, 3)
        w = d.cvel[b, :3]
        v = d.cvel[b, 3:]
        r = pos - d.subtree_com[self.model.body_rootid[b]]
        lin = v + _cross(w, r)
        angb = np.einsum("bji,bj->bi", R, w)
        return pos, quat, angb, lin

    def _foot_jacobians(self):
        """(F, n, 3, ...) -> array (F, 3, n): d(foot centre)/d(q) of each leg's joints (zero padding).
        Hinge column: axis × (p − anchor); slide column: axis (MuJoCo's world axes and anchors)."""
        d = self.data
        F, n = self._leg_jid.shape
        if F == 0 or n == 0:
            return np.zeros((F, 3, n))
        jid = np.maximum(self._leg_jid, 0)
        axis = d.xaxis[jid]                                    # (F, n, 3)
        arm = d.geom_xpos[self._foot_gid][:, None, :] - d.xanchor[jid]
        cols = np.where(self._leg_slide[..., None], axis, _cross(axis, arm))
        cols[~self._leg_valid] = 0.0
        return np.transpose(cols, (0, 2, 1))

    def _foot_jacobian(self, i):
        """MuJoCo's own Jacobian of foot ``i`` (mj_jac) for its leg joints — the reference for ``_foot_jacobians``."""
        import mujoco

        m, d = self.model, self.data
        g = self._foot_gid[i]
        jacp = self._jacp
        mujoco.mj_jac(m, d, jacp, None, d.geom_xpos[g], int(m.geom_bodyid[g]))
        return jacp[:, self._foot_dofs[i]].copy()

    def _foot_normal_forces(self):
        """Total normal ground force on each foot [N] — a lean path for controllers that read it every step
        (pyramidal or elliptic cones with condim 3; otherwise the full contact evaluation)."""
        import mujoco

        m, d = self.model, self.data
        F = len(self.feet)
        if d.ncon == 0 or F == 0:
            return np.zeros(F)
        con = d.contact
        g1, g2 = con.geom1, con.geom2
        terr = self._terrain_gid
        fi = self._foot_of_geom[np.where(g1 == terr, g2, g1)]
        adr = con.efc_address
        sel = (fi >= 0) & ((g1 == terr) | (g2 == terr)) & (adr >= 0)
        if not sel.any():
            return np.zeros(F)
        if not np.all(con.dim[sel] == 3):
            return self._contacts()[2]
        a = adr[sel]
        if m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
            fn = d.efc_force[a[:, None] + self._pyr4].sum(axis=1)
        else:
            fn = d.efc_force[a]
        return np.bincount(fi[sel], weights=fn, minlength=F)

    def _contacts(self):
        """Per foot: total force on it (world), unit normal into it, total normal force, contact point,
        in-contact flag; per logged body: belly contact. Only contacts with the terrain count."""
        import mujoco

        m, d = self.model, self.data
        F, B = len(self.feet), len(self.bodies)
        force, fn, cnt = np.zeros((F, 3)), np.zeros(F), np.zeros(F)
        nsum, psum, nraw, praw = np.zeros((F, 3)), np.zeros((F, 3)), np.zeros((F, 3)), np.zeros((F, 3))
        belly = np.zeros(B, dtype=bool)
        ncon = d.ncon
        if ncon:
            con = d.contact
            g1, g2 = con.geom1, con.geom2
            terr = self._terrain_gid
            with_t = (g1 == terr) | (g2 == terr)
            other = np.where(g1 == terr, g2, g1)
            adr = con.efc_address
            active = with_t & (adr >= 0)
            bi = self._belly_of_geom[other]
            hit = active & (bi >= 0)
            if hit.any():
                belly[bi[hit]] = True
            fi = self._foot_of_geom[other]
            sel = np.nonzero(active & (fi >= 0))[0]
            if sel.size:
                frame = con.frame[sel].reshape(-1, 3, 3)
                dim = con.dim[sel]
                a = adr[sel]
                local = np.zeros((sel.size, 3))
                if m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL and np.all(dim == 3):
                    p = d.efc_force[a[:, None] + np.arange(4)]
                    mu = con.friction[sel, :2]
                    local[:, 0] = p.sum(axis=1)
                    local[:, 1] = (p[:, 0] - p[:, 1]) * mu[:, 0]
                    local[:, 2] = (p[:, 2] - p[:, 3]) * mu[:, 1]
                elif m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and np.all(dim == 3):
                    local[:] = d.efc_force[a[:, None] + np.arange(3)]
                else:
                    buf = np.zeros(6)
                    for r, c in enumerate(sel):
                        mujoco.mj_contactForce(m, d, int(c), buf)
                        local[r] = buf[:3]
                world = np.einsum("nij,ni->nj", frame, local)
                # MuJoCo: the contact frame's normal points from geom1 to geom2 and the force acts on geom2.
                sign = np.where(g2[sel] == other[sel], 1.0, -1.0)[:, None]
                normal = frame[:, 0, :] * sign
                w = local[:, 0:1]
                cp = con.pos[sel]
                cols = np.hstack((world * sign, w, normal * w, cp * w, normal, cp, np.ones_like(w)))
                onehot = (fi[sel][None, :] == np.arange(F)[:, None]).astype(float)
                acc = onehot @ cols                                    # (F, 17) sums per foot
                force, fn = acc[:, 0:3], acc[:, 3]
                nsum, psum, nraw, praw, cnt = acc[:, 4:7], acc[:, 7:10], acc[:, 10:13], acc[:, 13:16], acc[:, 16]
        inc = cnt > 0
        normal = np.full((F, 3), np.nan)
        cpos = np.full((F, 3), np.nan)
        loaded = fn > 0
        if loaded.any():
            nn = nsum[loaded]
            normal[loaded] = nn / np.linalg.norm(nn, axis=1, keepdims=True)
            cpos[loaded] = psum[loaded] / fn[loaded, None]
        light = inc & ~loaded
        if light.any():
            nn = nraw[light]
            normal[light] = nn / np.linalg.norm(nn, axis=1, keepdims=True)
            cpos[light] = praw[light] / cnt[light, None]
        return force, normal, fn, cpos, inc, belly

    # ---- runs
    def run(self, controller, duration=None, rules: FailureRules | None = None, settle=0.5, seed=None, log=True,
            *, reset=True, base_pos=None, base_yaw=0.0, base_quat=None, qpos=None, info: dict | None = None
            ) -> Episode:
        """Run one trial and return its Episode.

        ``controller``: an object with ``reset(lab, seed)`` and ``__call__(obs) -> Command`` (or any callable
        obs → Command). Optional: ``settle_command(obs)`` for the settle phase (default: hold the standing pose),
        ``name`` for the log. The robot is reset (``seed``, ``base_pos``, ``base_yaw``, ``base_quat``, ``qpos``),
        ``controller.reset`` is called, the robot settles for ``settle`` s, then walks until an outcome of
        ``rules`` (or for ``duration`` s; default the rules' timeout). ``log=False`` keeps only the light log
        (time, bodies, COM, terrain height). ``info`` adds bookkeeping to the log (``treatment``, ``controller``,
        ``terrain`` (merged into the terrain spec, e.g. a normalised level), ``v_target``, ...).
        """
        wall0 = time.perf_counter()
        info = dict(info or {})
        h = self.timestep
        if reset:
            self.reset(seed=seed, base_pos=base_pos, base_yaw=base_yaw, base_quat=base_quat, qpos=qpos)
        else:
            self._seed = seed
        if callable(getattr(controller, "reset", None)):
            controller.reset(self, seed)
        # settle: hold the standing pose (or the controller's settle_command) with walking time < 0
        n_settle = int(round(settle / h))
        self._k = -n_settle
        self._set_command(self.nominal_command())
        settle_fn = getattr(controller, "settle_command", None)
        self._simulate(n_settle, controller=settle_fn if callable(settle_fn) else None)
        self._k = 0
        # walk
        if duration is None:
            if rules is None:
                raise ValueError("give a duration or FailureRules")
            duration = rules.timeout_s()
        if rules is not None:
            duration = min(float(duration), rules.timeout_s())
        n_samples = int(math.floor(duration / self.log_dt + 1e-9)) + 1
        nsteps = (n_samples - 1) * self._n_log + 1
        recorder = _Recorder(self, n_samples, full=log, rules=rules)
        stopped = self._simulate(nsteps, controller=controller, recorder=recorder)
        n = recorder.n
        outcome = recorder.outcome
        if outcome is None:
            t_end = float(recorder.t[n - 1]) if n else 0.0
            if rules is not None:
                to = rules.timeout_s()
                reason = "timeout" if t_end >= to - 1e-9 else "stopped"
                outcome = {"success": False, "reason": reason, "t_end": t_end,
                           "detail": f"no outcome by t = {t_end:.2f} s"}
            else:
                outcome = {"success": None, "reason": "completed", "t_end": t_end,
                           "detail": f"ran {t_end:.2f} s without failure rules"}
        com_x = recorder.com[:n, 0] if n else np.zeros(1)
        outcome["distance_m"] = float(com_x[-1] - com_x[0]) if n else 0.0
        outcome["x_end"] = float(com_x[-1]) if n else float("nan")
        wall = time.perf_counter() - wall0
        sim_t = (n_settle + (recorder.last_k + 1 if n else 0)) * h
        log_dict = recorder.build(controller, rules, seed, info)
        meta = {"chiron_log_format": LOG_FORMAT_VERSION, "wall_time_s": wall, "simulated_s": sim_t,
                "realtime_factor": sim_t / wall if wall > 0 else None, "timestep": h, "control_dt": self.control_dt,
                "log_dt": self.log_dt, "settle_s": n_settle * h, "stopped_by_rules": bool(stopped),
                "options": _plain(asdict(self.options)), "rules": _plain(asdict(rules)) if rules else None,
                "mujoco_version": _mujoco_version(), "servo_integration": SERVO_INTEGRATION,
                "robot": self.robot.name, "total_mass_kg": self.total_mass}
        return Episode(log_dict, outcome, meta)

    def summary(self) -> dict:
        return {"robot": self.robot.name, "total_mass_kg": self.total_mass, "joints": len(self.joint_names),
                "actuated": len(self.actuated_joints), "feet": list(self.feet), "bodies": list(self.bodies),
                "nominal_base_height_m": self.nominal_base_height, "nominal_hip_height_m": self.nominal_hip_height,
                "terrain": self.terrain.spec(), "timestep": self.timestep, "control_dt": self.control_dt,
                "log_dt": self.log_dt, "nq": self.model.nq, "nv": self.model.nv}


# ----------------------------------------------------------------------------------------------- recorder
class _Recorder:
    """Preallocated log buffers, filled at log steps (pre: state at t; post: derived quantities at t)."""

    def __init__(self, lab: ChironLab, T: int, full: bool, rules: FailureRules | None):
        self.lab, self.T, self.full, self.rules = lab, T, full, rules
        B, F, J, nl = len(lab.bodies), len(lab.feet), len(lab.joint_names), lab._nleg
        self.n = 0
        self.last_k = -1
        self.outcome = None
        self.t = np.zeros(T)
        self.com = np.zeros((T, 3))
        self.com_vel = np.zeros((T, 3))
        self.ang_mom = np.zeros((T, 3))
        self.body_pos = np.zeros((T, B, 3))
        self.body_quat = np.zeros((T, B, 4))
        self.ground = np.zeros(T)
        if full:
            self.body_angvel = np.zeros((T, B, 3))
            self.body_linvel = np.zeros((T, B, 3))
            self.q = np.zeros((T, J))
            self.qd = np.zeros((T, J))
            self.tau = np.zeros((T, J))
            self.foot_pos = np.zeros((T, F, 3))
            self.foot_force = np.zeros((T, F, 3))
            self.foot_normal = np.zeros((T, F, 3))
            self.foot_cpos = np.zeros((T, F, 3))
            self.foot_jac = np.zeros((T, F, 3, nl))
            self.belly = np.zeros((T, B), dtype=bool)
            self.leg_phase = None
            self.leg_stance = None
            if lab.log_geoms:
                G = len(lab._viz_gid)
                self.geom_pos = np.zeros((T, G, 3), dtype=np.float32)
                self.geom_mat = np.zeros((T, G, 9), dtype=np.float32)
        # rule state
        self.low_since = None
        self.rules_hip = None
        if rules is not None:
            hip = rules.nominal_hip_height
            self.rules_hip = hip if hip is not None else lab.nominal_hip_height
            self.n_window = int(round(rules.stall_window / lab.log_dt)) if rules.v_target else 0
            self.cos_tilt = math.cos(math.radians(rules.max_tilt_deg))

    def pre(self, k):
        i = self.n
        lab = self.lab
        d = lab.data
        self.t[i] = k * lab.timestep
        if self.full:
            np.take(d.qpos, lab._jq, out=self.q[i])
            np.take(d.qvel, lab._jd, out=self.qd[i])

    def post(self, k) -> bool:
        import mujoco

        lab = self.lab
        m, d = lab.model, lab.data
        i = self.n
        self.last_k = k
        b = lab._bid
        r = lab._root_body
        mujoco.mj_subtreeVel(m, d)
        self.com[i] = d.subtree_com[r]
        self.com_vel[i] = d.subtree_linvel[r]
        self.ang_mom[i] = d.subtree_angmom[r]
        if self.rules is not None:
            self.ground[i] = lab.terrain.height(self.com[i, 0], self.com[i, 1])
        if self.full:
            pos, quat, angb, lin = lab._body_state()
            self.body_pos[i], self.body_quat[i] = pos, quat
            self.body_angvel[i], self.body_linvel[i] = angb, lin
            self.tau[i, lab._act_j] = lab._tau
            F = len(lab.feet)
            if F:
                self.foot_pos[i] = d.geom_xpos[lab._foot_gid]
                f, nrm, _, cp, _, belly = lab._contacts()
                self.foot_force[i], self.foot_normal[i], self.foot_cpos[i] = f, nrm, cp
                self.belly[i] = belly
                self.foot_jac[i] = lab._foot_jacobians()
            elif len(lab.bodies):
                self.belly[i] = lab._contacts()[5]
            if lab._leg_phase is not None and len(lab._leg_phase) == F:
                if self.leg_phase is None:
                    self.leg_phase = np.full((self.T, F), np.nan)
                self.leg_phase[i] = lab._leg_phase
            if lab._leg_stance is not None and len(lab._leg_stance) == F:
                if self.leg_stance is None:
                    self.leg_stance = np.zeros((self.T, F), dtype=bool)
                self.leg_stance[i] = lab._leg_stance
            if lab.log_geoms:
                self.geom_pos[i] = d.geom_xpos[lab._viz_gid]
                self.geom_mat[i] = d.geom_xmat[lab._viz_gid]
        else:
            self.body_pos[i] = d.xipos[b]
            self.body_quat[i] = d.xquat[b]
        self.n = i + 1
        if self.rules is not None:
            self.outcome = self._check(i)
            return self.outcome is not None
        return False

    def _check(self, i):
        rules, t = self.rules, float(self.t[i])
        com = self.com[i]
        qb = self.body_quat[i]
        if len(qb):
            cos_tilt = 1.0 - 2.0 * (qb[:, 1] ** 2 + qb[:, 2] ** 2)  # R[2,2]: body z · world z
            worst = int(np.argmin(cos_tilt))
            if cos_tilt[worst] < self.cos_tilt:
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(cos_tilt[worst])))))
                return self._out("fall", t, f"{self.lab.bodies[worst]} tilted {tilt:.1f} deg "
                                            f"(> {rules.max_tilt_deg:g})")
        height = float(com[2] - self.ground[i])
        if height < rules.min_height_fraction * self.rules_hip:
            if self.low_since is None:
                self.low_since = t
            if t - self.low_since >= rules.low_height_time - 1e-9:
                return self._out("fall", t, f"COM {height * 1000:.0f} mm above ground for "
                                            f"{rules.low_height_time:g} s (< {rules.min_height_fraction:g} x "
                                            f"{self.rules_hip * 1000:.0f} mm)")
        else:
            self.low_since = None
        if abs(com[1]) > rules.lateral_limit:
            return self._out("off_course", t, f"|COM y| = {abs(com[1]):.3f} m > {rules.lateral_limit:g} m")
        if com[0] >= rules.course_m:
            return self._out("success", t, f"COM passed x = {rules.course_m:g} m", success=True)
        if rules.v_target and self.n_window > 0:
            j = i - self.n_window
            if j >= 0 and self.t[j] >= rules.stall_grace - 1e-9:
                adv = float(com[0] - self.com[j, 0])
                need = rules.stall_fraction * rules.v_target * rules.stall_window
                if adv < need:
                    return self._out("stall", t, f"COM advanced {adv * 1000:.0f} mm in {rules.stall_window:g} s "
                                                 f"(< {need * 1000:.0f} mm)")
        if t >= rules.timeout_s() - 1e-9:
            return self._out("timeout", t, f"no outcome by {rules.timeout_s():.2f} s")
        return None

    @staticmethod
    def _out(reason, t, detail, success=False):
        return {"success": success, "reason": reason, "t_end": t, "detail": detail}

    def build(self, controller, rules, seed, info) -> dict:
        lab, n = self.lab, self.n
        m = lab.model
        terrain_spec = dict(lab.terrain.spec())
        if isinstance(info.get("terrain"), dict):
            terrain_spec.update(info["terrain"])
        ctrl_name = info.get("controller") or getattr(controller, "name", None) or type(controller).__name__
        v_target = info.get("v_target", rules.v_target if rules is not None and rules.v_target else float("nan"))
        log = {
            "t": self.t[:n].copy(),
            "bodies": list(lab.bodies),
            "body_group": list(lab.body_groups),
            "body_pos": self.body_pos[:n].copy(),
            "body_quat": self.body_quat[:n].copy(),
            "com": self.com[:n].copy(),
            "com_vel": self.com_vel[:n].copy(),
            "ang_mom": self.ang_mom[:n].copy(),
            "total_mass": lab.total_mass,
            "gravity": np.array(m.opt.gravity, dtype=float),
            "terrain_height_under_com": (self.ground[:n].copy() if self.rules is not None else
                                         np.asarray(lab.terrain.height(self.com[:n, 0], self.com[:n, 1]),
                                                    dtype=float).reshape(n)),
            "v_target": float(v_target),
            "course_m": float(rules.course_m) if rules is not None else float(info.get("course_m", float("nan"))),
            "nominal_hip_height": float(self.rules_hip if self.rules_hip is not None else lab.nominal_hip_height),
            "disturbances": [{"t_start": float(dd.t_start), "duration": float(dd.duration),
                              "impulse_Ns": dd.impulse_Ns, "body": dd.body,
                              "direction": [float(x) for x in dd.unit]}
                             for dd in lab._disturbances + [x for x in lab._transient if x.t_start >= 0]],
            "robot": lab.robot.name,
            "treatment": info.get("treatment", ""),
            "controller": str(ctrl_name),
            "terrain": terrain_spec,
            "seed": -1 if seed is None else int(seed),
            "feet": list(lab.feet),
            "foot_body": lab._foot_body.copy(),
            "foot_group": list(lab.foot_groups),
            "foot_mu": lab._foot_mu.copy(),
            "foot_joints": [list(j) for j in lab.foot_joints],
            "joints": list(lab.joint_names),
            "joint_kind": list(lab.joint_tags),
            "joint_active": lab._joint_active.copy(),
            "joint_leg": lab._joint_leg.copy(),
            "q_range": lab._q_range.copy(),
            "actuated_joints": list(lab.actuated_joints),
        }
        J = len(lab.joint_names)
        for key, src in (("tau_stall", lab._stall), ("tau_rated", lab._rated), ("qd_noload", lab._w0),
                         ("i_stall", lab._istall), ("voltage", lab._volt)):
            arr = np.full(J, np.nan)
            arr[lab._act_j] = src
            log[key] = arr
        for k, v in info.items():
            if k not in ("terrain", "controller", "treatment", "v_target") and k not in log:
                log[k] = v
        if self.full:
            log.update({
                "body_angvel": self.body_angvel[:n].copy(),
                "body_linvel": self.body_linvel[:n].copy(),
                "q": self.q[:n].copy(), "qd": self.qd[:n].copy(), "tau": self.tau[:n].copy(),
                "foot_pos": self.foot_pos[:n].copy(), "foot_force": self.foot_force[:n].copy(),
                "foot_normal": self.foot_normal[:n].copy(), "foot_contact_pos": self.foot_cpos[:n].copy(),
                "foot_jac": self.foot_jac[:n].copy(), "belly_contact": self.belly[:n].copy(),
            })
            if self.leg_phase is not None:
                log["leg_phase"] = self.leg_phase[:n].copy()
            if self.leg_stance is not None:
                log["leg_stance_cmd"] = self.leg_stance[:n].copy()
            if lab.log_geoms:
                log["geom_pose"] = self._geom_static(n)
        return log

    def _geom_static(self, n):
        import mujoco

        lab = self.lab
        m = lab.model
        g = lab._viz_gid
        types = [mujoco.mjtGeom(int(t)).name.replace("mjGEOM_", "").lower() for t in m.geom_type[g]]
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or f"geom{int(i)}" for i in g]
        out = {"names": names, "type": types, "size": m.geom_size[g].copy(), "rgba": m.geom_rgba[g].copy(),
               "rbound": m.geom_rbound[g].copy(),
               "collides": ((m.geom_contype[g] | m.geom_conaffinity[g]) > 0),
               "foot": lab._foot_of_geom[g] >= 0,
               "pos": self.geom_pos[:n].copy(), "mat": self.geom_mat[:n].copy()}
        ti = lab._tinfo
        x0, x1, y0, y1 = lab.options.course_extent
        Z, ext = lab.terrain.heightfield((x0, x1), (y0, y1), max(0.02, lab.options.heightfield_cell))
        out["terrain_z"] = Z.astype(np.float32)
        out["terrain_extent"] = np.array(ext, dtype=float)
        out["terrain_plane"] = np.array(ti["type"] == "plane")
        return out


# ----------------------------------------------------------------------------------------------- helpers
def _cross(a, b):
    """Row-wise cross product (faster than np.cross for small arrays)."""
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def _ratio(dt, h, what) -> int:
    n = dt / h
    k = int(round(n))
    if k < 1 or abs(n - k) > 1e-6 * max(1.0, n):
        raise ValueError(f"{what} = {dt} must be a positive multiple of the timestep {h}")
    return k


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _geom_lowest_z(m, d, g) -> float:
    """Lowest world z of a geom (exact for sphere/capsule/box/ellipsoid/cylinder)."""
    import mujoco

    t = m.geom_type[g]
    p = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    s = m.geom_size[g]
    G = mujoco.mjtGeom
    if t == G.mjGEOM_SPHERE:
        return float(p[2] - s[0])
    if t == G.mjGEOM_CAPSULE:
        return float(p[2] - abs(R[2, 2]) * s[1] - s[0])
    if t == G.mjGEOM_CYLINDER:
        return float(p[2] - abs(R[2, 2]) * s[1] - s[0] * math.sqrt(max(0.0, 1 - R[2, 2] ** 2)))
    if t == G.mjGEOM_BOX:
        return float(p[2] - np.sum(np.abs(R[2, :]) * s[:3]))
    if t == G.mjGEOM_ELLIPSOID:
        return float(p[2] - math.sqrt(np.sum((R[2, :] * s[:3]) ** 2)))
    return float(p[2] - m.geom_rbound[g])


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _mujoco_version() -> str:
    try:
        import mujoco

        return mujoco.__version__
    except Exception:  # pragma: no cover
        return "unknown"
