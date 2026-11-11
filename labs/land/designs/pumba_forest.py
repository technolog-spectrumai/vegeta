"""Pumba in the forest — a MuJoCo (Chiron) scenario: the light Pumba drives a forest trail between the trees over
a rough floor and two fallen sticks, finds a fallen log lying across the trail, picks it up with the main arm's
grab, swings it off the trail, sets it down beside it and drives on.

* **the forest** (``Scene``, ``terrain``, ``props``): rough forest floor (RMS 30 mm, 0.6 m correlation) with
  hummocks; trees as fixed trunks (cylinders, 0.1-0.3 m radius, 6-11 m tall) with a canopy (visual) placed off the
  trail (deterministic from ``seed``, never closer than ``clear`` to the trail's centre line); two branches
  bedded across the trail to drive over (ridges in the terrain, as Pekari's course log, drawn as capsules: a loose
  round stick wedges between the simulated track's rollers, which a real belt bridges); the log (a free capsule, 0.12 m, 1.6 m, ~14 kg green spruce)
  across the trail.
* **the mission** (``ForestMission``): pure pursuit along the trail on the tracks (skid steer: the belt speeds of
  the two sides, Pekari's roller model) and the arm moved in joint space between IK solutions (``pekari_arms.ik``
  in the hull frame, retaken from the hull's pose when each move starts, plus an alignment step on the measured
  joint angles): stop before the log -> unfold -> over the log, grab open -> down -> align -> close -> lift ->
  swing to the side -> down -> open -> fold -> drive on. No gravity feed-forward: the servos' gains hold the arm.
* **the grasp** (``GraspWeld``): the jaws close on the log in MuJoCo's contacts; once both press on it (> 20 N
  each) the log is welded to the grab's hand in the pose it is in, and released when the grab opens. MuJoCo's
  soft contacts let a round log squeeze out from between two jaws, so holding is modelled as a weld; the approach,
  the closing and the jaw contact are simulated.

    import pumba_forest as pfo
    lab, scene = pfo.make_lab()
    ep = pfo.run(lab, scene)
    pfo.summary(ep, scene)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vegeta import chiron as ch

import pekari_arms as pa
import pekari_family as pf
import pumba_robot as pu

__all__ = ["Scene", "terrain", "props", "make_lab", "ForestMission", "run", "summary"]


@dataclass
class Scene:
    """Where things are (world, m). The trail: ``waypoints`` from the start (0, 0) heading +x; the log across the
    trail at ``log_x`` (centre on y = 0, its axis along y), ``stop_gap`` the hull centre's distance short of it."""

    seed: int = 11
    extent: tuple = (-4.0, 22.0, -9.0, 9.0)
    rough_rms: float = 0.03
    rough_len: float = 0.6
    hummocks: tuple = ((4.5, -2.5, 0.25, 2.0), (9.0, 3.5, 0.3, 2.5), (14.0, -3.0, 0.2, 2.0), (17.5, 2.5, 0.25, 2.2))
    waypoints: tuple = ((3.5, 0.0), (5.8, 0.9), (7.6, 0.9), (9.3, 0.0), (21.0, 0.0))
    n_trees: int = 46
    n_shrubs: int = 70
    clear: float = 1.45                               # trunk surface to the trail's centre line [m]
    sticks: tuple = ((2.2, 0.04), (7.3, 0.05))         # (x along the trail, radius) of the branches across it
    log_x: float = 13.0
    log_r: float = 0.05
    log_len: float = 1.6
    log_density: float = 760.0                        # green spruce [kg/m³]
    stop_gap: float = 1.30
    v: float = 1.0
    trees: list = field(default_factory=list)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        x0, x1, y0, y1 = self.extent
        pts = [(0.0, 0.0)] + list(self.waypoints)
        trees = []
        tries = 0
        while len(trees) < self.n_trees and tries < 5000:
            tries += 1
            x, y = rng.uniform(x0 + 1, x1 - 1), rng.uniform(y0 + 1, y1 - 1)
            r = float(rng.uniform(0.10, 0.30))
            if _dist_to_polyline((x, y), pts) < self.clear + r:
                continue
            if any(math.hypot(x - tx, y - ty) < r + tr + 1.0 for tx, ty, tr, _ in trees):
                continue
            if math.hypot(x, y) < 3.0:                # room around the start
                continue
            trees.append((x, y, r, float(rng.uniform(6.0, 11.0))))
        self.trees = trees

    @property
    def log_mass(self) -> float:
        return math.pi * self.log_r ** 2 * self.log_len * self.log_density


def _dist_to_polyline(p, pts) -> float:
    best = math.inf
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy))
    return best


def _pursuit_point(p, pts, lookahead):
    """The point ``lookahead`` [m] along the polyline ``pts`` past the point on it nearest to ``p``."""
    best, s_best, s0 = math.inf, 0.0, 0.0
    segs = []
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (L * L)))
        d = math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)
        if d < best:
            best, s_best = d, s0 + t * L
        segs.append((ax, ay, dx, dy, L, s0))
        s0 += L
    s = min(s_best + lookahead, s0)
    for ax, ay, dx, dy, L, a0 in segs:
        if s <= a0 + L + 1e-9:
            t = (s - a0) / L
            return ax + t * dx, ay + t * dy
    return pts[-1]


def terrain(scene: Scene):
    """The forest floor: ``Rough`` plus Gaussian hummocks, as a ``chiron.Custom``."""
    rough = ch.Rough(scene.rough_rms, scene.rough_len, start=1.0, seed=scene.seed, extent=scene.extent, cell=0.025)

    def ground(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        z = rough.height(x, y)
        for hx, hy, hh, hr in scene.hummocks:
            z = z + hh * np.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (2 * (hr / 2) ** 2))
        for xs, rs in scene.sticks:                    # a branch bedded across the trail: a ridge in the ground
            # the branch (height ``rs``) with leaf litter banked against it: twice as wide as it is high
            ridge = rs * np.sqrt(np.clip(1.0 - ((x - xs) / (2 * rs)) ** 2, 0.0, None))
            z = z + ridge * np.clip((0.9 - np.abs(y)) / 0.15, 0.0, 1.0)
        return z
    return ch.Custom(ground, name="forest floor")


def props(scene: Scene, ground) -> list:
    """Trunks (fixed, colliding) with canopies (visual), the sticks (fixed) and the log (a free capsule)."""
    bark = (0.36, 0.26, 0.17, 1.0)
    out = []
    for k, (x, y, r, h) in enumerate(scene.trees):
        z0 = float(ground.height(np.array([x]), np.array([y]))[0]) - 0.05
        green = (0.16 + 0.05 * (k % 3), 0.36 + 0.06 * (k % 4), 0.14, 1.0)
        out.append(ch.Prop(ch.Link(f"tree{k}", pos=(x, y, z0), geoms=[
            ch.Geom(f"tree{k}_trunk", "cylinder", (r, h / 2), pos=(0, 0, h / 2), friction=(0.6, 0.005, 0.0001), rgba=bark),
            ch.Geom(f"tree{k}_crown", "ellipsoid", (1.2 + 3 * r, 1.2 + 3 * r, 0.32 * h), pos=(0, 0, 0.62 * h),
                    role="visual", rgba=green)]), log=False))
    # undergrowth (visual): shrubs and ferns off the trail, deterministic from the seed
    rng = np.random.default_rng(scene.seed + 1)
    pts = [(0.0, 0.0)] + list(scene.waypoints)
    x0, x1, y0, y1 = scene.extent
    n = 0
    while n < scene.n_shrubs:
        x, y = rng.uniform(x0 + 0.5, x1 - 0.5), rng.uniform(y0 + 0.5, y1 - 0.5)
        r = float(rng.uniform(0.25, 0.6))
        if _dist_to_polyline((x, y), pts) < 1.1 + r or math.hypot(x, y) < 2.0:
            continue
        z0 = float(ground.height(np.array([x]), np.array([y]))[0])
        g = (0.20 + 0.08 * rng.random(), 0.38 + 0.15 * rng.random(), 0.12 + 0.05 * rng.random(), 1.0)
        out.append(ch.Prop(ch.Link(f"shrub{n}", pos=(x, y, z0), geoms=[
            ch.Geom(f"shrub{n}_g", "ellipsoid", (r, r * 0.9, r * 0.55), pos=(0, 0, r * 0.3), role="visual", rgba=g)]),
            log=False))
        n += 1
    wood = (0.45, 0.33, 0.20, 1.0)
    for k, (xs, rs) in enumerate(scene.sticks):        # drawn on the ridge the terrain carries (``terrain``)
        z = float(ground.height(np.array([xs]), np.array([0.0]))[0]) - 0.4 * rs
        out.append(ch.Prop(ch.Link(f"stick{k}", pos=(xs, 0.0, z), geoms=[
            ch.Geom(f"stick{k}_g", "capsule", (rs,), fromto=(0, -0.9, 0, 0, 0.9, 0), role="visual", rgba=wood)]),
            log=False))
    zl = float(ground.height(np.array([scene.log_x]), np.array([0.0]))[0]) + scene.log_r + 0.01
    h = scene.log_len / 2
    out.append(ch.Prop(ch.Link("log", pos=(scene.log_x, 0.0, zl), geoms=[
        ch.Geom("log_g", "capsule", (scene.log_r,), fromto=(0, -h, 0, 0, h, 0), mass=scene.log_mass,
                friction=(0.8, 0.01, 0.03), rgba=(0.50, 0.36, 0.22, 1.0))]),
        free=True, condim=6))                     # rolling friction: bark in the litter does not roll away
    return out


class GraspWeld:
    """Scene hook: weld the log to ``{arm}_hand`` when the mission asks to hold (``mission.hold``) and both jaws
    press on the log (> ``f_min`` each), in the relative pose it has then; release when ``hold`` goes False."""

    def __init__(self, mission, arm: str = "main", f_min: float = 20.0):
        self.mission, self.arm, self.f_min = mission, arm, f_min
        self.on = False

    def reset(self, lab):
        self.on = False

    def __call__(self, lab):
        import mujoco

        want = bool(getattr(self.mission, "hold", False))
        if want and not self.on:
            fa = lab.contact_force([f"{self.arm}_grip_a"], ["log_g"])[1]
            fb = lab.contact_force([f"{self.arm}_grip_b"], ["log_g"])[1]
            if fa > self.f_min and fb > self.f_min:
                m, d = lab.model, lab.data
                e = lab._weld_id["grasp"]
                b1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "log")
                b2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{self.arm}_hand")
                q1inv = np.zeros(4)
                mujoco.mju_negQuat(q1inv, d.xquat[b1])
                relpos = np.zeros(3)
                mujoco.mju_rotVecQuat(relpos, d.xpos[b2] - d.xpos[b1], q1inv)
                relq = np.zeros(4)
                mujoco.mju_mulQuat(relq, q1inv, d.xquat[b2])
                m.eq_data[e, 0:3] = 0.0                          # anchor at body1's origin
                m.eq_data[e, 3:6] = relpos                       # body2's pose in body1's frame, now
                m.eq_data[e, 6:10] = relq
                lab.set_weld("grasp", True)
                self.on = True
                lab.log_event("grasp", f"log held (jaw forces {fa:.0f} N, {fb:.0f} N)")
        elif not want and self.on:
            lab.set_weld("grasp", False)
            self.on = False
            lab.log_event("grasp", "log released")


def make_lab(scene: Scene | None = None, *, log_dt: float = 0.04, roller_kd: float | None = 8.0, **kw):
    """The ChironLab: Pumba (deck empty: a trail run), the forest terrain and props, geoms logged for the movie.
    ``roller_kd``: the track rollers' velocity gain — 8 N m s/rad (3x the family's mass-scaled default) so the skid
    steer follows the winding trail on the rough floor (None: the default)."""
    scene = scene or Scene()
    ground = terrain(scene)
    robot = pu.pumba(payload_kg=0.0, **({} if roller_kd is None else {"roller_kd": roller_kd}))
    lab = pf.lab(pu.PUMBA, ground, robot, props=props(scene, ground), log_geoms=True, log_dt=log_dt,
                 course_extent=scene.extent, heightfield_cell=0.025,
                 welds=[ch.Weld("grasp", "log", "main_hand", active=False, solref=(0.01, 1.0))], **kw)
    return lab, scene


# ----------------------------------------------------------------------------------------------- the mission
def _rot(quat):
    w, x, y, z = quat
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class ForestMission:
    """Drive the trail, clear the log with the main arm, drive on (see the module). ``events``: [t, phase, note]."""

    name = "pumba forest trail"
    ARM = "main"

    def __init__(self, scene: Scene, *, accel: float = 0.8, k_heading: float = 2.0, lookahead: float = 1.6):
        self.scene = scene
        self.accel, self.k_heading, self.lookahead = accel, k_heading, lookahead
        self.pl = pu.PUMBA
        self.arm = self.pl.mount(self.ARM).arm
        sh = pf.shoulder(self.pl, self.ARM)
        self.shoulder_hull = (sh[0], sh[1], sh[2] - pf.z_hull(self.pl))

    # ---- ChironLab protocol
    def reset(self, lab, seed=None):
        self.lab = lab
        g = lab.robot.geometry
        self.B = g["B"]
        self.radius = {f"{side}_{name}": r for side in ("L", "R") for name, _, _, r, *_ in lab.robot.rollers}
        self.joints = {side: [f"{side}_{name}" for name, *_ in lab.robot.rollers] for side in ("L", "R")}
        self.v = 0.0
        self.t_prev = None
        self.events = []
        self.phase = "drive"
        self.wp = 0
        rollers = {j for side in self.joints.values() for j in side}
        act = set(lab.actuated_joints)
        self.q_arm = {k: v for k, v in lab.robot.nominal_qpos.items() if k in act and k not in rollers}
        self.move = None                                   # (t0, T, q_from, q_to)
        self.plan = []
        self.drive_to = None
        self.hold = False

    def settle_command(self, obs):
        return self._command(0.0, 0.0)

    def _command(self, vl, vr):
        qd = {}
        for side, v in (("L", vl), ("R", vr)):
            for j in self.joints[side]:
                qd[j] = v / self.radius[j]
        return ch.Command(q_target=dict(self.q_arm), qd_target=qd)

    def _log(self, t, phase, note=""):
        self.events.append([round(t, 2), phase, note])

    # ---- the arm: joint-space moves between IK solutions in the hull frame
    def _arm_q(self, target_hull, elev=-math.pi / 2, grip=None):
        q = pa.ik(self.arm, target_hull, elev, self.shoulder_hull)
        self._last_target = tuple(float(v) for v in target_hull)
        if q is None:
            raise RuntimeError(f"arm target {target_hull} out of reach")
        out = pa.qpos(self.arm, f"{self.ARM}_", q)
        g = self.q_arm[f"{self.ARM}_grip_a"] if grip is None else grip
        out[f"{self.ARM}_grip_a"] = out[f"{self.ARM}_grip_b"] = g
        return out

    def _hold(self, on: bool, q: dict) -> dict:
        self.hold = on
        return q

    def _here(self, world):
        """A world point in the hull frame now."""
        return tuple(self.R.T @ (np.asarray(world, dtype=float) - self.p))

    def _aligned_q(self, target_hull, grip=None, gain=1.0):
        """IK for ``target_hull`` corrected by the error the arm shows now: forward kinematics of the *measured*
        joint angles against the IK target that was commanded (the sag under gravity and the hull's settling)."""
        obs = self.obs
        meas = {j: obs.joint(f"{self.ARM}_{j}") for j in pa.JOINTS if j in self.arm.drives and j != "grip"}
        tcp = np.asarray(pa.fk(self.arm, meas, self.shoulder_hull)["tcp"])
        err = tcp - np.asarray(self._last_target)
        return self._arm_q(tuple(np.asarray(target_hull) - gain * err), grip=grip)

    def _pose_q(self, name, grip=0.0):
        return pa.qpos(self.arm, f"{self.ARM}_", pa.pose(self.arm, name), grip)

    def _start_moves(self, t, steps):
        """``steps``: [(label, q_to dict or callable -> dict, T)] run one after the other."""
        self.plan = list(steps)
        self._next_move(t)

    def _next_move(self, t):
        if not self.plan:
            self.move = None
            return False
        label, q_to, T = self.plan.pop(0)
        q_to = q_to() if callable(q_to) else q_to
        self.move = (t, T, {k: self.q_arm[k] for k in q_to}, q_to)
        self._log(t, "arm", label)
        return True

    def _arm_step(self, t):
        if self.move is None:
            return False
        t0, T, qa, qb = self.move
        s = _smooth((t - t0) / T)
        for k in qb:
            self.q_arm[k] = qa[k] + (qb[k] - qa[k]) * s
        if t - t0 >= T:
            return self._next_move(t) or None
        return True

    # ---- the log's pose from the scene's props log
    def _log_world(self, obs):
        names = [p.name for p in self.lab.props if p.log]
        return np.asarray(obs.prop_pos)[names.index("log")]

    def __call__(self, obs):
        t = float(obs.t)
        self.obs = obs
        self.R, self.p = _rot(np.asarray(obs.base_quat)), np.asarray(obs.base_pos, dtype=float)
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        R = _rot(np.asarray(obs.base_quat))
        p = np.asarray(obs.base_pos)
        psi = math.atan2(R[1, 0], R[0, 0])
        sc = self.scene

        def to_hull(w):
            return R.T @ (np.asarray(w, dtype=float) - p)

        v_target, w = 0.0, 0.0
        if self.phase in ("drive", "drive_on"):
            # pure pursuit: the point ``lookahead`` along the trail ahead of the nearest point on it
            gx, gy = _pursuit_point((p[0], p[1]), [(0.0, 0.0)] + list(sc.waypoints), self.lookahead)
            err = (math.atan2(gy - p[1], gx - p[0]) - psi + math.pi) % (2 * math.pi) - math.pi
            v_target = sc.v * max(0.3, math.cos(err))
            w = self.k_heading * err
            if self.phase == "drive":
                x_stop = sc.log_x - sc.stop_gap
                if p[0] >= x_stop - 1.5:                   # slow down to a creep for the stop point
                    v_target = min(v_target, max(0.3, 0.6 * (x_stop - p[0]) + 0.3))
                if p[0] >= x_stop:
                    self.phase, v_target, w = "stop", 0.0, 0.0
                    self._log(t, "stop", f"before the log at x = {p[0]:.2f} m")
            elif p[0] >= sc.waypoints[-1][0] - 0.5:
                self.phase = "done"
                self._log(t, "done", "end of the trail")
        if self.phase == "stop" and abs(self.v) < 0.02:
            L = self._log_world(obs)
            # grab the log where it passes nearest the arm (within 0.45 m of its middle): its axis is along world y
            sh_w = R @ np.asarray(self.shoulder_hull) + p
            L = L + np.array([0.0, float(np.clip(sh_w[1] - L[1], -0.45, 0.45)), 0.0])
            lh = to_hull(L)
            # targets near the log are taken in the hull frame *when each move starts* (``self._here``): the hull
            # settles and pitches forward once the arm reaches out
            grasp_w = L.copy()
            above = lambda: self._here(grasp_w + np.array([0.0, 0.0, 0.30]))           # noqa: E731
            at = lambda: self._here(grasp_w + np.array([0.0, 0.0, -0.045]))           # noqa: E731  tips at the log's bottom
            # off the trail: straight out to the arm's side (yaw ~90 deg), 1.05 m from the shoulder — the log turns
            # with the arm and lies along the trail, beside the left track
            shh = self.shoulder_hull
            side = (shh[0] + 0.05, shh[1] + 1.05, lh[2] + 0.35)
            gw = R @ np.asarray((side[0], side[1], 0.0)) + p
            ground = float(self.lab.terrain.height(np.array([gw[0]]), np.array([gw[1]]))[0])
            side_down = (side[0], side[1], to_hull((gw[0], gw[1], ground + self.scene.log_r))[2] + 0.04)
            g = self.arm.ranges["grip"]
            g_open, g_close = math.radians(g[0]), math.radians(g[1])
            self._start_moves(t, [
                ("unfold", lambda: self._arm_q((0.9, 0.0, 0.25), grip=g_open), 3.0),
                ("over the log, grab open", lambda: self._arm_q(above(), grip=g_open), 2.5),
                ("down to the log", lambda: self._arm_q(at(), grip=g_open), 2.0),
                ("align on the log", lambda: self._aligned_q(at(), grip=g_open), 1.0),
                ("align again", lambda: self._aligned_q(at(), grip=g_open), 0.8),
                ("close the grab", lambda: self._hold(True, {f"{self.ARM}_grip_a": g_close, f"{self.ARM}_grip_b": g_close}), 2.5),
                ("lift", lambda: {**self._arm_q((lh[0] - 0.1, lh[1], lh[2] + 0.55)), f"{self.ARM}_grip_a": g_close,
                                  f"{self.ARM}_grip_b": g_close}, 2.5),
                ("swing off the trail", lambda: {**self._arm_q(side), f"{self.ARM}_grip_a": g_close,
                                                 f"{self.ARM}_grip_b": g_close}, 4.0),
                ("set it down", lambda: {**self._arm_q(side_down), f"{self.ARM}_grip_a": g_close,
                                         f"{self.ARM}_grip_b": g_close}, 2.0),
                ("open", lambda: self._hold(False, self._arm_q(side_down, grip=g_open)), 1.2),
                ("up", lambda: self._arm_q(side, grip=g_open), 1.5),
                ("fold", lambda: self._pose_q("folded"), 3.5),
            ])
            self.phase = "arm"
            self._log(t, "arm", f"log at {np.round(L, 2).tolist()} (hull {np.round(lh, 2).tolist()})")
        if self.phase == "arm":
            if self._arm_step(t) is None and self.move is None:
                L = self._log_world(obs)
                self._log(t, "log", f"log now at {np.round(L, 2).tolist()}")
                self.phase = "drive_on"
        # speed ramp and skid steer
        self.v = min(v_target, self.v + self.accel * dt) if self.v < v_target else max(v_target, self.v - 1.5 * self.accel * dt)
        if self.phase in ("stop", "arm", "done"):
            w = 0.0
        return self._command(self.v - w * self.B / 2, self.v + w * self.B / 2)


def run(lab, scene: Scene, duration: float = 62.0, seed: int = 0):
    ctrl = ForestMission(scene)
    lab.clear_hooks()
    lab.add_hook(GraspWeld(ctrl))
    ep = lab.run(ctrl, duration=duration, rules=None, settle=0.8, seed=seed, info={"controller": ctrl.name})
    ep.log["events"] = list(ctrl.events)
    return ep


def summary(ep, scene: Scene) -> dict:
    """Where the log ended (off the trail?), how far Pumba went, its largest tilt, the events."""
    log = ep.log
    names = list(log["props"])
    L = np.asarray(log["prop_pos"])[:, names.index("log")]
    com = np.asarray(log["com"])
    off = float(abs(L[-1, 1]))
    return {"distance_m": float(com[-1, 0] - com[0, 0]), "log_start": L[0].round(2).tolist(),
            "log_end": L[-1].round(2).tolist(), "log_lift_m": float(L[:, 2].max() - L[0, 2]),
            "log_off_trail": off > 0.9, "events": log.get("events", [])}
