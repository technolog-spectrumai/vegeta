"""Glove teleoperation of Onager Manus's two pincers in MuJoCo (``docs/bionic.md``, P2): the teleop bench and its
four tasks, an operator model, and the latency sweep.

**The bench.** Manus's two arms — the same geometry, actuators, gains, jaw model and IK (``onager_manus_robot``),
with straight jaws instead of the hooked tips (``BENCH_ARM``: hooks cannot close on a block lying on a table) — on a
fixed base instead of the 458 kg rover (whose chassis would cost most of the simulation and adds nothing to the
pincers), welded to the world at reset; a work table in front, its top at ``TABLE_TOP``.

**The loop** (``vegeta.bionic.teleop``): the operator's two hands -> two simulated gloves (calibrated) -> the
downlink -> the hand estimate -> ``PincerRetarget`` (position mode; the clutch and the mode from the operator's
interface, mocked) -> jaw commands; the arms move by the interface's commands (mocked: the task's waypoints), sent down
the same link. Up: the jaws' contact force -> the gloves' vibration motors, and what the operator sees (the cameras)
-> the operator model, both delayed by the uplink.

**The operator model** acts only on what it sees (``Seen``, delayed): it moves an arm to the object, closes the hand
when the jaws look in place, squeezes once it sees (or feels) contact, lifts when the grip looks firm, and so on.
Latency therefore costs time on every task and success on the timing-critical ones.

**The tasks** (``TASKS``): ``pick_place`` (a 50 mm block from the table into a tray), ``handover`` (left pincer to
right), ``rope`` (pinch a Ø10 mm rope lying on the table and lift it), ``catch`` (grip a block swinging on a string at
the end of its swing).

    import manus_teleop as mt
    r = mt.run_task("pick_place", latency_s=0.1, seed=0, record_to="demo_pick")
    df = mt.latency_sweep()                                   # success and time against latency, every task
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vegeta import chiron as ch
from vegeta.chiron import Command, Geom, Link, PointMass, Robot
from vegeta.bionic import Recorder, TeleopSide
from vegeta.bionic.hand import interpolate, key_pose
from vegeta.bionic.teleop import DelayLine, OperatorUI

import onager_manus_robot as omr

SHOULDER_HEIGHT = 1.10                  # m: the shoulders above the floor
TABLE_TOP = 0.55                        # m
TABLE = dict(x=(1.15, 1.95), y=(-0.75, 0.75))
BLOCK = dict(size=0.05, mass=0.30, mu=0.8)
ROPE = dict(radius=0.005, length=0.40, mass=0.04, mu=0.9)
TRAY = dict(inner=0.16, wall=0.01, height=0.06)
PENDULUM = dict(length=0.55, angle_deg=25.0, mass=0.20)
TASKS = ("pick_place", "handover", "rope", "catch")
JOG_SPEED, JOG_TURN = 0.15, math.radians(60.0)    # the interface's jog limits for the arms (mocked operator UI)
LAB_OPTIONS = dict(timestep=0.0005, control_dt=0.002, log_dt=0.02, integrator="implicitfast", cone="elliptic",
                   impratio=10.0, noslip_iterations=3)


# ----------------------------------------------------------------------------------------------- the bench
BENCH_ARM = {"hook_length": 0.0}         # straight fingers: hooked tips (for logs) cannot close on a block on a table


def bench(jaw_mu: float = 0.9) -> Robot:
    """Manus's arms on a fixed base (the hull frame's origin at the base's origin, so ``omr.arm_ik`` applies), with
    straight jaws (``BENCH_ARM``)."""
    p = omr.design_params(BENCH_ARM)
    cad = omr.cad_numbers(p)
    arms, masses = [], []
    for side in omr.SIDES:
        base, yaw_mass, _, _ = omr._arm(side, p, cad, jaw_mu)
        _add_roll(base, side, omr.arm_geometry(p))
        arms.append(base)
        masses.append(yaw_mass)
    g = omr.arm_geometry(p)
    root = Link("bench", geoms=[Geom("bench_block", "box", (0.25, 0.45, 0.08), pos=(g["x"], 0.0, g["z"] - g["ped_h"] - 0.08),
                                     mass=200.0, role="body", rgba=(0.35, 0.36, 0.40, 1.0)),
                                Geom("bench_column", "box", (0.12, 0.12, (SHOULDER_HEIGHT - 0.40) / 2),
                                     pos=(g["x"], 0.0, g["z"] - g["ped_h"] - 0.16 - (SHOULDER_HEIGHT - 0.40) / 2),
                                     role="visual", rgba=(0.35, 0.36, 0.40, 1.0))],
                masses=masses, children=arms, log=True)
    robot = Robot("Manus teleop bench", root, notes="Onager Manus's two arms on a fixed bench (manus_teleop)",
                  sources={"arms": "onager_manus_robot._arm", "assumptions": "onager_manus_robot.ASSUMPTIONS"})
    robot.nominal_qpos = {k: v for k, v in omr.nominal_qpos(p).items() if k.split("_", 1)[0] in omr.SIDES}
    robot.validate()
    robot.params, robot.arm = p, g
    return robot


def _add_roll(base: Link, side: str, g: dict) -> None:
    """A wrist roll between the palm and the jaws (about the pincer's axis), driven like the wrist (harmonic 60 Nm).
    Manus's arms have none: both pincers' jaws then open in their arm's vertical plane, and in a hand-over the jaws of
    one always cross the other's — the bench adds it, and a two-handed Manus arm would need it."""
    palm = base.children[0].children[0].children[0]
    jaws = palm.children
    for j in jaws:
        j.pos = (0.0, 0.0, 0.0)
    roll = Link(f"{side}_roll_link", pos=(g["Lp"], 0.0, 0.0),
                joints=[ch.Joint(f"{side}_roll", axis=(1, 0, 0), range=(-math.pi, math.pi), tag="roll", servo=omr._servo("wrist"))],
                geoms=[Geom(f"{side}_roll_hub", "cylinder", (g["jaw_d"] * 0.5, 0.012), pos=(0.012, 0.0, 0.0),
                            quat=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0), mass=0.6, role="link",
                            rgba=(0.30, 0.31, 0.27, 1.0))],
                children=jaws)
    palm.children = [roll]


def base_height(robot) -> float:
    return SHOULDER_HEIGHT - robot.arm["z"]


# ----------------------------------------------------------------------------------------------- the scenes
def _box(name, half, pos, rgba, mass=None, friction=(0.8, 0.005, 0.0001)):
    return Geom(name, "box", tuple(half), pos=tuple(pos), mass=mass, friction=friction, rgba=rgba)


def table_prop() -> ch.Prop:
    (x0, x1), (y0, y1) = TABLE["x"], TABLE["y"]
    return ch.Prop(Link("table", pos=((x0 + x1) / 2, (y0 + y1) / 2, TABLE_TOP),
                        geoms=[_box("table_top", ((x1 - x0) / 2, (y1 - y0) / 2, 0.05), (0, 0, -0.05), (0.6, 0.5, 0.4, 1.0))]))


def block_prop(name, xy, z=None, rgba=(0.85, 0.2, 0.2, 1.0)) -> ch.Prop:
    h = BLOCK["size"] / 2
    return ch.Prop(Link(name, pos=(xy[0], xy[1], (TABLE_TOP if z is None else z) + h + 0.001),
                        geoms=[_box(f"{name}_g", (h, h, h), (0, 0, 0), rgba, BLOCK["mass"], (BLOCK["mu"], 0.01, 0.0001))]),
                   free=True, condim=4)


def tray_prop(xy) -> ch.Prop:
    a, t, h = TRAY["inner"] / 2, TRAY["wall"], TRAY["height"]
    g = [_box("tray_floor", (a + t, a + t, t / 2), (0, 0, t / 2), (0.2, 0.5, 0.8, 1.0))]
    for k, (cx, cy, hx, hy) in enumerate(((a + t / 2, 0, t / 2, a + t), (-a - t / 2, 0, t / 2, a + t),
                                          (0, a + t / 2, a + t, t / 2), (0, -a - t / 2, a + t, t / 2))):
        g.append(_box(f"tray_wall{k}", (hx, hy, h / 2), (cx, cy, h / 2), (0.2, 0.5, 0.8, 1.0)))
    return ch.Prop(Link("tray", pos=(xy[0], xy[1], TABLE_TOP), geoms=g))


def rope_prop(xy) -> ch.Prop:
    r, L = ROPE["radius"], ROPE["length"]
    return ch.Prop(Link("rope", pos=(xy[0], xy[1], TABLE_TOP + r + 0.001),
                        geoms=[Geom("rope_g", "capsule", (r,), fromto=(0, -L / 2, 0, 0, L / 2, 0), mass=ROPE["mass"],   # across the jaws
                                    friction=(ROPE["mu"], 0.01, 0.0001), rgba=(0.9, 0.75, 0.2, 1.0))]),
                   free=True, condim=4, solref=(0.002, 1.0))                     # a stiff contact: a thin, light body


def pendulum_prop(pivot) -> ch.Prop:
    """A block on a string: a hinge about x at ``pivot`` (it swings in y-z), released from ``angle_deg``."""
    L, h = PENDULUM["length"], BLOCK["size"] / 2
    string = Geom("string_g", "capsule", (0.002,), fromto=(0, 0, 0, 0, 0, -L + h), mass=0.005, role="visual",
                  rgba=(0.9, 0.9, 0.9, 1.0))
    bob = _box("swing_g", (h, h, h), (0, 0, -L), (0.2, 0.75, 0.3, 1.0), PENDULUM["mass"], (BLOCK["mu"], 0.01, 0.0001))
    return ch.Prop(Link("pendulum", pos=tuple(pivot),
                        joints=[ch.Joint("swing", axis=(1, 0, 0), damping=0.001)], geoms=[string, bob]))


@dataclass
class Task:
    name: str
    props: list
    targets: dict = field(default_factory=dict)            # named points [m, world]


def task_scene(name: str) -> Task:
    yl, yr = omr.arm_geometry()["y"], -omr.arm_geometry()["y"]
    x = 1.42
    if name == "pick_place":
        return Task(name, [table_prop(), block_prop("block", (x, yl)), tray_prop((x, yl - 0.30))],
                    {"pick": (x, yl), "place": (x, yl - 0.30)})
    if name == "handover":
        return Task(name, [table_prop(), block_prop("block", (x, yl))], {"pick": (x, yl), "meet": (1.55, 0.0)})
    if name == "rope":
        return Task(name, [table_prop(), rope_prop((x, yl))], {"pick": (x, yl)})
    if name == "catch":
        # the swing's far end straight ahead of the left shoulder: the pincer points along +x into it (no yaw)
        a = math.radians(PENDULUM["angle_deg"])
        pivot = (x + 0.02, yl + PENDULUM["length"] * math.sin(a), TABLE_TOP + 0.75)
        return Task(name, [table_prop(), pendulum_prop(pivot)], {"pivot": pivot})
    raise KeyError(name)


# ----------------------------------------------------------------------------------------------- sensing
class Sensing:
    """Per step: each pincer's contact with the task's objects (normal force summed over both jaws) and the poses
    of the objects; both feed the uplink (what the operator feels and sees)."""

    def __init__(self, lab, objects):
        import mujoco
        self.mujoco = mujoco
        m = lab.model
        self.jaw_geoms = {s: {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in omr.jaw_parts(s)} for s in omr.SIDES}
        self.jaw_each = {s: [{mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (j, j + "_hook")}
                             for j in omr.jaw_geoms(s)] for s in omr.SIDES}
        self.obj_bodies = {o: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o) for o in objects}
        self.obj_geoms = {o: {i for i in range(m.ngeom) if m.geom_bodyid[i] == b} for o, b in self.obj_bodies.items()}
        bob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "swing_g")
        self.bob = bob if bob >= 0 else None
        self.lab = lab

    def contacts(self) -> dict:
        """{side: (force on the objects [N], both jaws touching, which object)}"""
        m, d = self.lab.model, self.lab.data
        out = {s: [0.0, [False, False], None] for s in omr.SIDES}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for s in omr.SIDES:
                for o, gs in self.obj_geoms.items():
                    if (c.geom1 in self.jaw_geoms[s] and c.geom2 in gs) or (c.geom2 in self.jaw_geoms[s] and c.geom1 in gs):
                        self.mujoco.mj_contactForce(m, d, i, f6)
                        out[s][0] += abs(f6[0])
                        out[s][2] = o
                        for k, jg in enumerate(self.jaw_each[s]):
                            if c.geom1 in jg or c.geom2 in jg:
                                out[s][1][k] = True
        return {s: (v[0], all(v[1]), v[2]) for s, v in out.items()}

    def objects(self) -> dict:
        d = self.lab.data
        out = {o: d.xpos[b].copy() for o, b in self.obj_bodies.items()}
        if self.bob is not None:
            out["bob"] = d.geom_xpos[self.bob].copy()
        return out


# ----------------------------------------------------------------------------------------------- the robot side
class BenchController:
    """The robot: arm joint targets from the interface's (delayed) arm commands through IK; the jaws from the gloves'
    pincer commands (opening -> jaw angles; grip -> a torque past the contact); gravity feed-forward."""

    def __init__(self, session):
        self.s = session

    def reset(self, lab, seed=None):
        import mujoco
        self.lab = lab
        self.idx = {n: i for i, n in enumerate(lab.joint_names)}
        self.dof = {n: int(lab.model.jnt_dofadr[mujoco.mj_name2id(lab.model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in lab.joint_names}
        self.kp_jaw = omr.ARM_GAINS["jaw"][0]
        self.g = omr.arm_geometry()
        self.arm_q = {s: None for s in omr.SIDES}
        self.jog = {s: None for s in omr.SIDES}                     # where the interface's jog has brought each pin
        self.roll_cmd = {}
        self.s.on_reset(lab, self)

    def __call__(self, obs):
        t = float(obs.t)
        q_t, qd_t, ff = {}, {}, {}
        bias = self.lab.data.qfrc_bias
        cmds = self.s.step(t, obs)
        for side in omr.SIDES:
            names = omr.arm_joints(side)
            arm = cmds["arm"][side]
            if arm is not None:
                arm = self._jogged(side, obs, arm)
                ik = omr.arm_ik(arm[0], arm[1], side)
                if ik is not None:
                    self.arm_q[side] = ik
            if self.arm_q[side] is None:
                self.arm_q[side] = tuple(float(obs.q[self.idx[n]]) for n in names[:4])
            for n, v in zip(names[:4], self.arm_q[side]):
                q_t[n], qd_t[n], ff[n] = float(v), 0.0, float(bias[self.dof[n]])
            # the roll jogs at JOG_TURN: a step to the servo runs it at its no-load speed into a limit cycle
            r0 = self.roll_cmd.get(side, 0.0)
            r1 = float(cmds["roll"][side])
            dr = JOG_TURN * self.lab.control_dt
            self.roll_cmd[side] = r1 if abs(r1 - r0) <= dr else r0 + math.copysign(dr, r1 - r0)
            q_t[f"{side}_roll"], qd_t[f"{side}_roll"] = self.roll_cmd[side], 0.0
            pc = cmds["pincer"][side]
            half = min(pc.opening / 2.0, 0.95 * self.g["Lj"])
            q_open = math.asin(half / self.g["Lj"])
            for n in names[4:]:
                ff[n] = float(bias[self.dof[n]])
                qd_t[n] = 0.0
                if pc.grip > 0.0 and cmds["contact"][side]:
                    q_t[n] = float(obs.q[self.idx[n]]) - pc.grip * 0.7 * self.g["Lj"] / self.kp_jaw
                else:
                    q_t[n] = q_open
        return Command(q_target=q_t, qd_target=qd_t, tau_ff=ff)

    def _jogged(self, side, obs, arm):
        """The interface jogs a pin towards its commanded pose at ``JOG_SPEED`` [m/s] and ``JOG_TURN`` [rad/s], from
        where the pin is when the first command arrives."""
        target, el = np.asarray(arm[0], float), float(arm[1])
        if self.jog[side] is None:
            q = [float(obs.q[self.idx[n]]) for n in omr.arm_joints(side)[:4]]
            p0, el0 = omr.arm_fk(q, side)
            self.jog[side] = [np.asarray(p0, float), float(el0)]
        p, e = self.jog[side]
        dt = self.lab.control_dt
        d = target - p
        L = float(np.linalg.norm(d))
        step = JOG_SPEED * dt
        p = target.copy() if L <= step else p + d / L * step
        e = el if abs(el - e) <= JOG_TURN * dt else e + math.copysign(JOG_TURN * dt, el - e)
        self.jog[side] = [p, e]
        return p, e


# ----------------------------------------------------------------------------------------------- the session
class Session:
    """Two gloves and the interface, the operator model, the sensing, the recording; one task."""

    def __init__(self, task: Task, *, latency_s: float = 0.02, jitter_s: float = 0.0, loss: float = 0.0, seed: int = 0,
                 record: bool = False):
        down, up = latency_s / 2, latency_s / 2
        self.task, self.latency = task, latency_s
        self.sides = {s: TeleopSide(s, seed=seed * 10 + k, down_latency_s=down, up_latency_s=up, jitter_s=jitter_s, loss=loss)
                      for k, s in enumerate(omr.SIDES)}
        self.ui = OperatorUI()                                       # clutch and mode: the interface (mocked, none here)
        self.arm_link = DelayLine(down)                              # the interface's arm commands go down too
        self.seen = DelayLine(up)                                    # the cameras: object and pincer states, delayed
        self.operator = OPERATORS[task.name](self)
        self.rec = Recorder(meta={"task": task.name, "latency_s": latency_s, "jitter_s": jitter_s, "loss": loss,
                                  "seed": seed, "robot": "Manus teleop bench (labs/land/designs/manus_teleop.py)",
                                  "mapping": "position (bionic.retarget.PincerMap defaults)"}) if record else None
        if self.rec:
            self.rec.camera("front", {"pos": [3.0, 0.0, 1.6], "lookat": [1.4, 0.0, 0.6], "fovy": 50, "size": [360, 640], "fps": 25})
        self.events = []
        self.contact_now = {s: (0.0, False, None) for s in omr.SIDES}

    def on_reset(self, lab, ctrl):
        self.lab, self.ctrl = lab, ctrl
        self.sensing = Sensing(lab, [p.name for p in self.task.props if p.free or p.name == "pendulum"])

    def bench_frame(self):
        """The bench's actual orientation and position (world): the hull frame the arms' IK and FK work in."""
        d = self.lab.data
        b = self.lab._body_id("bench")
        return d.xmat[b].reshape(3, 3).copy(), d.xpos[b].copy()

    def to_bench(self, p_world):
        Rb, pb = self.bench_frame()
        return Rb.T @ (np.asarray(p_world, float) - pb)

    def step(self, t, obs):
        c = self.sensing.contacts()
        self.contact_now = c
        objs = self.sensing.objects()
        pins = {}
        Rb, pb = self.bench_frame()
        for side in omr.SIDES:
            q = [float(obs.q[self.ctrl.idx[n]]) for n in omr.arm_joints(side)]
            p_hull, el = omr.arm_fk(q, side)
            pins[side] = (pb + Rb @ np.asarray(p_hull), el, q[4])
        self.seen.put(t, {"objects": objs, "pins": pins, "t": t})
        seen = self.seen.get(t)
        self.ui.apply(t, self.sides)
        hands, arm_cmd = self.operator(t, seen, {s: self.sides[s].seen_contact(t) for s in omr.SIDES})
        self.arm_link.put(t, arm_cmd)
        arm_now = self.arm_link.get(t) or {s: None for s in omr.SIDES}
        out = {"arm": {}, "pincer": {}, "contact": {}, "roll": {}}
        for side in omr.SIDES:
            ts = self.sides[side]
            ts.operator_hand(t, hands[side])
            force, both, _ = c[side]
            touching = force > 1.0
            pc, est = ts.robot_step(t, touching)
            ts.feedback(t, touching, force)
            out["pincer"][side], out["contact"][side] = pc, touching
            a = arm_now.get(side)
            out["arm"][side] = None if a is None else (self.to_bench(a[0]), a[1])
            out["roll"][side] = 0.0 if a is None or len(a) < 3 else a[2]
            if self.rec and int(round(t / self.lab.control_dt)) % 5 == 0:
                self.rec.add(f"hand_{side}", t, joints_true=hands[side].vector(),
                             joints_est=(est.vector() if est is not None else np.full(20, np.nan)))
                self.rec.add(f"robot_command_{side}", t, opening=pc.opening, grip=pc.grip, clutched=float(pc.clutched),
                             arm_target=(np.r_[a[0], a[1]] if a is not None else np.full(4, np.nan)))
                self.rec.add(f"contacts_{side}", t, force=force, both_jaws=float(both), vibration_g=ts.vibration_g)
        if self.rec and int(round(t / self.lab.control_dt)) % 5 == 0:
            self.rec.add("robot_state", t, q=np.asarray(obs.q, float))
            self.rec.add("objects", t, **{o: p for o, p in objs.items()})
            if int(round(t / self.lab.control_dt)) % 20 == 0:
                self.rec.frame("front", t)
        return out


# ----------------------------------------------------------------------------------------------- the operator
def _ease(t0, T, t):
    return min(max((t - t0) / T, 0.0), 1.0)


class Operator:
    """A scripted operator who acts on what it sees: ``plan`` is a list of steps, each a function
    (t, seen, felt) -> True when done; a step sets ``self.target`` (per side: (pin position, elevation) or None) and the
    hands (``self.hand_goal``, reached over ``hand_s``)."""

    hand_s = 0.35
    reach_tol = 0.012
    approach_h = 0.10

    def __init__(self, session):
        self.s = session
        self.k = 0
        self.t_step = 0.0
        self.target = {s: None for s in omr.SIDES}
        self.roll = {s: 0.0 for s in omr.SIDES}                       # the wrist roll the interface commands
        self.hand = {s: key_pose("flat") for s in omr.SIDES}
        self.hand_from = dict(self.hand)
        self.hand_goal = dict(self.hand)
        self.hand_t0 = {s: 0.0 for s in omr.SIDES}
        self.done_t = None
        self.log = []
        self.plan = self.steps()

    # helpers
    def set_hand(self, side, pose, t):
        self.hand_from[side], self.hand_goal[side], self.hand_t0[side] = self.hand[side], pose, t

    def move(self, side, p, elevation=-math.pi / 2):
        self.target[side] = (tuple(p), elevation, self.roll[side])

    def set_roll(self, side, angle):
        def step(t, seen, felt):
            self.roll[side] = angle
            if self.target[side] is not None:
                self.target[side] = (self.target[side][0], self.target[side][1], angle)
            return t - self.t_step > abs(angle) / JOG_TURN + 0.5            # the roll's jog, and a moment to settle
        return step

    def at(self, seen, side, p, tol=None):
        if seen is None:
            return False
        return float(np.linalg.norm(seen["pins"][side][0] - np.asarray(p))) < (tol or self.reach_tol)

    def __call__(self, t, seen, felt):
        if self.k < len(self.plan):
            if self.plan[self.k](t, seen, felt):
                self.log.append((t, self.k))
                self.k += 1
                self.t_step = t
        elif self.done_t is None:
            self.done_t = t
        for s in omr.SIDES:
            self.hand[s] = interpolate(self.hand_from[s], self.hand_goal[s], _ease(self.hand_t0[s], self.hand_s, t))
        return dict(self.hand), {s: self.target[s] for s in omr.SIDES}

    def steps(self):
        return []

    # the pieces of a plan
    def go(self, side, p_fn, elevation=-math.pi / 2, tol=None):
        def step(t, seen, felt):
            p = p_fn(seen)
            self.move(side, p, elevation)
            return self.at(seen, side, p, tol)
        return step

    def hand_to(self, side, name, u=1.0, wait=None):
        def step(t, seen, felt):
            if self.hand_goal[side] is not self._pose(name, u):
                self.set_hand(side, self._pose(name, u), t)
            return t - self.hand_t0[side] >= (wait if wait is not None else self.hand_s)
        return step

    def until_felt(self, side, force=5.0, timeout=3.0):
        def step(t, seen, felt):
            return felt[side][0] and felt[side][1] > force or t - self.t_step > timeout
        return step

    def wait(self, T):
        return lambda t, seen, felt: t - self.t_step >= T

    def freeze_hand(self, side):
        """Stop the hand where it is (the operator felt contact): the grip is the squeeze made in the feedback's delay."""
        def step(t, seen, felt):
            self.set_hand(side, self.hand[side], t)
            return True
        return step

    _poses = {}

    def _pose(self, name, u):
        key = (name, round(u, 3))
        if key not in self._poses:
            self._poses[key] = interpolate(key_pose("flat"), key_pose(name), u)
        return self._poses[key]


def grasp_pin_z(object_centre_z: float, into: float = 0.02) -> float:
    """The jaw pin's height for a vertical pincer that holds an object ``into`` m from its jaw tips (the tips just
    above whatever it stands on)."""
    return object_centre_z + omr.arm_geometry(omr.design_params(BENCH_ARM))["Lj"] - into


def _obj(seen, name):
    return seen["objects"][name] if seen else None


class PickPlace(Operator):
    def steps(self):
        L = "L"
        tgt = self.s.task.targets
        grasp_z = grasp_pin_z(TABLE_TOP + BLOCK["size"] / 2)
        return [self.hand_to(L, "pinch", 0.25),                                  # open the jaws wide
                self.go(L, lambda s: (tgt["pick"][0], tgt["pick"][1], grasp_z + self.approach_h), tol=0.02),
                self.go(L, lambda s: (_obj(s, "block")[0], _obj(s, "block")[1], grasp_z), tol=0.01),
                self.hand_to(L, "pinch", 0.75),                                  # close onto the block
                self.until_felt(L),
                self.hand_to(L, "pinch", 1.0),                                   # squeeze
                self.wait(0.3),
                self.go(L, lambda s: (tgt["pick"][0], tgt["pick"][1], grasp_z + 0.15), tol=0.02),
                self.go(L, lambda s: (tgt["place"][0], tgt["place"][1], grasp_z + 0.15), tol=0.02),
                self.go(L, lambda s: (tgt["place"][0], tgt["place"][1], grasp_z + 0.04), tol=0.015),
                self.hand_to(L, "pinch", 0.25),                                  # let go
                self.wait(0.5),
                self.go(L, lambda s: (tgt["place"][0], tgt["place"][1], grasp_z + 0.2), tol=0.03)]

    def success(self, lab, sensing) -> bool:
        b = sensing.objects()["block"]
        p = self.s.task.targets["place"]
        a = TRAY["inner"] / 2
        return abs(b[0] - p[0]) < a and abs(b[1] - p[1]) < a and b[2] < TABLE_TOP + TRAY["height"]


class Handover(Operator):
    """Left picks the block and lifts it to the meeting point; there it rolls the block so that its free faces face
    the right arm; the right pincer, rolled 90 deg (its jaws opening horizontally), comes in level along its own line
    of approach and grips those faces; the left lets go and withdraws. Needs the bench's wrist roll (``_add_roll``)."""

    def steps(self):
        L, R = "L", "R"
        tgt = self.s.task.targets
        grasp_z = grasp_pin_z(TABLE_TOP + BLOCK["size"] / 2)
        lift = 0.30
        meet = np.array([tgt["meet"][0], tgt["meet"][1]])
        g = omr.arm_geometry()
        shoulder = {sd: np.array([g["x"], omr.SIDES[sd] * g["y"]]) for sd in omr.SIDES}
        u = {sd: (meet - shoulder[sd]) / np.linalg.norm(meet - shoulder[sd]) for sd in omr.SIDES}
        th = {sd: math.atan2(u[sd][1], u[sd][0]) for sd in omr.SIDES}
        roll_L = th["R"] - th["L"]                        # the left jaws' faces turned to face the right arm's line

        def r_pin(s, back):
            b = _obj(s, "block")
            return (b[0] - back * u["R"][0], b[1] - back * u["R"][1], b[2])

        return [self.hand_to(L, "pinch", 0.25),
                self.go(L, lambda s: (tgt["pick"][0], tgt["pick"][1], grasp_z + self.approach_h), tol=0.02),
                self.go(L, lambda s: (_obj(s, "block")[0], _obj(s, "block")[1], grasp_z), tol=0.01),
                self.hand_to(L, "pinch", 0.75), self.until_felt(L), self.hand_to(L, "pinch", 1.0), self.wait(0.3),
                self.go(L, lambda s: (tgt["pick"][0], tgt["pick"][1], grasp_z + lift), tol=0.02),
                self.go(L, lambda s: (meet[0], meet[1], grasp_z + lift), tol=0.015),
                self.set_roll(L, roll_L),
                self.hand_to(R, "pinch", 0.25), self.set_roll(R, math.pi / 2),
                self.go(R, lambda s: r_pin(s, 0.32), elevation=0.0, tol=0.02),
                self.go(R, lambda s: r_pin(s, 0.19), elevation=0.0, tol=0.012),
                self.hand_to(R, "pinch", 0.75), self.until_felt(R), self.hand_to(R, "pinch", 1.0), self.wait(0.3),
                self.hand_to(L, "pinch", 0.25), self.wait(0.5),
                self.go(L, lambda s: (meet[0], meet[1], grasp_z + lift + 0.15), tol=0.03),
                self.wait(1.0)]

    def success(self, lab, sensing) -> bool:
        c = sensing.contacts()
        b = sensing.objects()["block"]
        return c["R"][0] > 1.0 and c["L"][0] < 0.5 and b[2] > TABLE_TOP + 0.15


class Rope(Operator):
    def steps(self):
        L = "L"
        tgt = self.s.task.targets
        z = grasp_pin_z(TABLE_TOP + ROPE["radius"], into=0.0)        # tips at the rope's centre: the open jaws' corners dip lower
        return [self.hand_to(L, "pinch", 0.72),                                 # opened just enough for a 10 mm rope
                self.go(L, lambda s: (tgt["pick"][0], tgt["pick"][1], z + self.approach_h), elevation=-math.pi / 2, tol=0.02),
                self.go(L, lambda s: (_obj(s, "rope")[0], tgt["pick"][1], z), elevation=-math.pi / 2, tol=0.012),
                self.hand_to(L, "pinch", 0.92), self.until_felt(L, force=2.0),
                self.hand_to(L, "pinch", 0.935), self.wait(0.3),                 # a gentle squeeze: a wedge on a round rope
                self.go(L, lambda s: (tgt["pick"][0], tgt["pick"][1], z + 0.25), tol=0.03),
                self.wait(1.0)]

    def success(self, lab, sensing) -> bool:
        return sensing.objects()["rope"][2] > TABLE_TOP + 0.15 and sensing.contacts()["L"][0] > 0.5


class Catch(Operator):
    """The pincer waits level with its jaws open at the end of the block's swing (the block is released once it is
    there); the operator closes as the block comes in — early, by ``lead_m``, as a person anticipates a swing."""

    def steps(self):
        L = "L"
        pv = np.asarray(self.s.task.targets["pivot"])
        a = math.radians(PENDULUM["angle_deg"])
        Lp = PENDULUM["length"]
        end = pv + np.array([0.0, -Lp * math.sin(a), -Lp * math.cos(a)])          # the swing's far end (it starts at +a)
        # the block 190 mm out along the jaws, where the open wedge is wide enough; the pin at its arrival height
        pin = end - np.array([0.19, 0.0, 0.008])

        def release(t, seen, felt):
            self.s.lab.set_weld("hold_swing", False)                           # the scene lets the block go
            self.s.events.append((t, "swing released"))
            return True

        return [self.hand_to(L, "pinch", self.open_u),
                self.go(L, lambda s: tuple(pin), elevation=0.0, tol=0.01),
                self.wait(0.5), release,
                self.wait_swing(end),
                self.hand_to(L, "pinch", 1.0, wait=0.0),                         # close fast...
                self.until_felt(L, force=2.0, timeout=1.0), self.freeze_hand(L),     # ...and stop on the first buzz
                self.wait(1.5)]

    lead_m = 0.10          # the operator anticipates: closes while the block is still this far from the swing's end
    open_u = 0.40          # the jaws open enough that the block clears them on its way in, no wider

    def wait_swing(self, end):
        """Close when the seen block is within ``lead_m`` of the swing's end and coming towards it."""
        last = {}

        def step(t, seen, felt):
            if seen is None:
                return False
            bob = seen["objects"].get("bob")
            if bob is None:
                return False
            d = float(np.linalg.norm(bob - end))
            closing = "d" in last and d < last["d"]
            last["d"] = d
            return d < self.lead_m and closing and t - self.t_step > 0.2
        return step

    def success(self, lab, sensing) -> bool:
        c = sensing.contacts()["L"]
        return c[0] > 0.5 and c[1]


OPERATORS = {"pick_place": PickPlace, "handover": Handover, "rope": Rope, "catch": Catch}


# ----------------------------------------------------------------------------------------------- running
def make_lab(task: Task, robot=None, **kwargs):
    robot = robot or bench()
    opts = dict(LAB_OPTIONS)
    opts.update(kwargs)
    welds = [ch.Weld("bench_fixed", "bench", None, active=True, solref=(0.001, 1.0))]   # stiff: the arms' loads
    if task.name == "catch":                                                   # the block held at its release angle
        welds.append(ch.Weld("hold_swing", "pendulum", None, active=True))
    return ch.ChironLab(robot, ch.Flat(), props=task.props, welds=welds, **opts)


def _fix_bench(lab, body="bench", weld="bench_fixed"):
    """Weld ``body`` to the world where it is now (the weld's relative pose set from the current state)."""
    import mujoco
    m, d = lab.model, lab.data
    b = lab._body_id(body)
    e = lab._weld_id[weld]
    qinv = np.zeros(4)
    mujoco.mju_negQuat(qinv, d.xquat[b])
    R = d.xmat[b].reshape(3, 3)
    m.eq_data[e, 0:3] = 0.0
    m.eq_data[e, 3:6] = R.T @ (np.zeros(3) - d.xpos[b])
    m.eq_data[e, 6:10] = qinv
    lab.set_weld(weld, True)


def start_pose(task_name: str) -> dict:
    """The arms raised over the table, jaws closed (and the pendulum released from its angle)."""
    q = {}
    for side in omr.SIDES:
        ik = omr.arm_ik((1.25, omr.SIDES[side] * omr.arm_geometry()["y"], SHOULDER_HEIGHT - 0.25 - (SHOULDER_HEIGHT - omr.arm_geometry()["z"])),
                        -math.pi / 2, side)
        for n, v in zip(omr.arm_joints(side)[:4], ik):
            q[n] = v
        for n in omr.arm_joints(side)[4:]:
            q[n] = 0.0
    if task_name == "catch":
        q["swing"] = math.radians(PENDULUM["angle_deg"])
    return q


def run_task(name: str, *, latency_s: float = 0.02, jitter_s: float = 0.0, loss: float = 0.0, seed: int = 0,
             duration: float = 30.0, record_to=None) -> dict:
    """One task end to end: the bench, the scene, two gloves and the operator; the outcome, and the recording when
    ``record_to`` is a folder."""
    import mujoco
    task = task_scene(name)
    lab = make_lab(task)
    session = Session(task, latency_s=latency_s, jitter_s=jitter_s, loss=loss, seed=seed, record=record_to is not None)
    ctrl = BenchController(session)
    q0 = start_pose(name)
    orig_reset = session.on_reset

    def on_reset(lab_, ctrl_):
        m, d = lab_.model, lab_.data
        for n, v in q0.items():
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j >= 0:
                d.qpos[m.jnt_qposadr[j]] = v
                d.qvel[m.jnt_dofadr[j]] = 0.0
        mujoco.mj_forward(m, d)
        _fix_bench(lab_)
        if name == "catch":
            _fix_bench(lab_, "pendulum", "hold_swing")
        orig_reset(lab_, ctrl_)
    session.on_reset = on_reset
    base = (0.0, 0.0, base_height(lab.robot))
    ep = lab.run(ctrl, duration=duration, rules=None, settle=0.0, seed=seed, base_pos=base,
                 info={"controller": f"glove teleop: {name}", "treatment": f"latency {latency_s * 1000:.0f} ms"},
                 stop_when=lambda lab_: session.operator.done_t is not None and lab_.time > session.operator.done_t + 0.3)
    op = session.operator
    ok = bool(op.success(lab, session.sensing)) and op.done_t is not None
    out = {"task": name, "latency_ms": latency_s * 1000, "seed": seed, "success": ok,
           "time_s": float(op.done_t) if op.done_t is not None else float("nan"),
           "steps_done": op.k, "steps": len(op.plan), "sim_s": float(lab.time),
           "link": {s: session.sides[s].channel.link.stats() for s in omr.SIDES}}
    if record_to is not None:
        session.rec.meta["outcome"] = {k: v for k, v in out.items() if k != "link"}
        session.rec.meta["operator_steps"] = op.log
        out["recording"] = str(session.rec.save(record_to))
    return out


def latency_sweep(latencies_ms=(10, 50, 100, 200, 300, 500), seeds=(0, 1), tasks=TASKS, processes: int = 4):
    """Every task at every round-trip latency (split evenly between the downlink and the uplink) and seed: success
    and completion time — where grasping starts to fail sets the latency budget (``docs/bionic.md``)."""
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd
    jobs = [(t, l / 1000.0, s) for t in tasks for l in latencies_ms for s in seeds]
    with ProcessPoolExecutor(processes) as ex:
        rows = list(ex.map(_sweep_one, jobs))
    return pd.DataFrame(rows)


def _sweep_one(job):
    t, l, s = job
    r = run_task(t, latency_s=l, seed=s)
    r.pop("link", None)
    return r
