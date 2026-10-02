"""The Sikarian Lobster Nefri's jobs in ChironLab (notebook 24 §9, ``scenarios/sikarian_lobster.py``), underwater
(fresh water, MuJoCo's fluid drag and the ``lobster_robot.Water`` hook's buoyancy and thrust):

* **swim and land** (``swim_and_land``) — released 1 m above the bed, legs tucked, the Lobster swims forward on its
  tail thruster: the tail points the thrust up and forward *through the centre of gravity* (``thrust_line``), a
  depth loop sets its angle, the yaw joints hold the heading; then it eases the thrust, lowers its legs and lands;
* **cut and enter** (``cut_and_enter``) — a flooded Ø600 mm concrete outfall pipe lies bedded in the bottom with
  150 mm of silt inside; a Ø10 mm polypropylene rope is strung across its mouth from two stakes at the height of the Lobster's chest.
  It walks up (tripod gait), puts its right pincer's cutter notch on the rope and closes the worm drive: the rope
  parts only if both jaws squeeze it with the cutting force ``F_CUT`` (the Onager Manus's ``WireCutter`` hook, the
  contact forces MuJoCo computes); it stows the arm and walks 1 m into the pipe;
* **hold in a current** (``hold_in_current``) — a 0.5 m/s cross-current (MuJoCo's ``wind``): standing, the drag on the
  shell is about the friction its wet weight gives, so it slides; the tail then points the thrust **down through the
  CG** (the jet up): the extra normal force holds it, and it walks across the current.

``run(kind, ...)`` returns the Episode; ``timeseries`` the numbers; ``ROPE``, ``PIPE`` the inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vegeta import chiron as ch

import lobster_controller as lc
import lobster_robot as lr
from onager_manus_scenario import WireCutter

__all__ = ["ROPE", "CABLE", "WIRES", "PIPE", "F_CUT", "T_SHEAR", "Scene", "RopeRecoil", "terrain", "pipe_props", "rope_props", "make_lab",
           "swim_and_land", "cut_and_enter", "hold_in_current", "run", "timeseries"]

#: The rope (inputs, Nefri's job): Ø10 mm 3-strand polypropylene, the cutting force of a hardened blade against an
#: anvil (rope-cutter tests of the class: 0.4-0.6 kN for Ø10 PP; the upper value is the input), 50 ms through it.
ROPE = {"name": "Ø10 mm PP rope", "diameter_m": 0.010, "density_kg_m3": 910.0, "contact_solref": (0.001, 1.0),
        "contact_solimp": (0.99, 0.999, 0.001), "recoil_Nm_rad": 0.3, "f_cut_N": 600.0, "t_shear_s": 0.05,
        "rgba": (0.95, 0.75, 0.1, 1.0)}
#: The cable (inputs, Ornatus's job): Ø12 mm PVC power cable, 3×2.5 mm² copper (0.19 kg/m: it sinks); ratchet
#: cable-cutter class ~2.5 kN blade against anvil, 100 ms through it; stiff: the cut ends spring back further.
CABLE = {"name": "Ø12 mm PVC power cable 3×2.5 mm²", "diameter_m": 0.012, "density_kg_m3": 1680.0, "contact_solref": (0.001, 1.0),
         "contact_solimp": (0.99, 0.999, 0.001), "recoil_Nm_rad": 1.0, "f_cut_N": 2500.0, "t_shear_s": 0.10,
         "rgba": (0.12, 0.12, 0.14, 1.0)}
WIRES = {"nefri": ROPE, "ornatus": CABLE}
F_CUT = ROPE["f_cut_N"]
T_SHEAR = ROPE["t_shear_s"]
#: The pipe (inputs): a Ø600 mm (inner) concrete outfall, 50 mm wall, 3 m long, bedded 150 mm into the bottom with
#: 150 mm of silt inside, flush with the bed (the silt floor is 0.52 m wide: the Lobster's feet span 0.43 m).
PIPE = {"inner_d": 0.60, "wall": 0.05, "length": 3.0, "staves": 28, "mu": 0.6, "silt": 0.15,
        "alpha": 0.35}                             # drawn translucent: the movies see the Lobster inside


@dataclass
class Scene:
    """Where things are (world, m): the pipe's mouth at ``x_mouth`` on the centre line, the rope (Nefri) or cable
    (Ornatus: ``variant``) ``rope_gap`` in front of it at ``z_rope`` (default: the variant's chest height), stakes at
    ±``stake_y``; the bed's roughness."""

    variant: str = "nefri"
    x_mouth: float = 2.0
    rope_gap: float = 0.15
    z_rope: float | None = None
    stake_y: float = 0.45
    rough_rms: float = 0.003
    seed: int = 3
    ROPE_WELD: str = "rope knot"
    WIRE_WELD: str = "rope knot"                 # the names WireCutter reads
    WIRE_GEOMS: tuple = ("rope_a_g", "rope_b_g")

    def __post_init__(self):
        if self.z_rope is None:                       # Nefri's 0.16 m, scaled with the variant's standing height
            self.z_rope = 0.16 * _standing_height(self.v.params) / _standing_height(lr.NEFRI)

    @property
    def v(self) -> lr.Variant:
        return lr.variant_of(self.variant)

    @property
    def wire(self) -> dict:
        """The rope or cable across the mouth (``WIRES``)."""
        return WIRES[self.v.name]

    @property
    def x_rope(self) -> float:
        return self.x_mouth - self.rope_gap

    def cut_y(self) -> float:
        """The rope is cut where the right claw meets it (the shoulder's y)."""
        return -lr.geometry(self.v.params)["claw"][1]


def terrain(scene: Scene):
    if scene.rough_rms <= 0:
        return ch.Flat()
    return ch.Rough(scene.rough_rms, 0.3, start=0.5, seed=scene.seed, extent=(-2.0, 8.0, -2.0, 6.0), cell=0.02)


def pipe_props(scene: Scene) -> list:
    """The pipe as one fixed prop: ``staves`` boxes around the ring, its silt floor flush with the bed (the bed's
    height field continues inside it; staves below the bed are left out)."""
    P = PIPE
    r_in, t, L, n = P["inner_d"] / 2, P["wall"], P["length"], P["staves"]
    rc = r_in + t / 2
    zc = r_in - P["silt"]                          # the axis: the inner bottom ``silt`` below the bed
    half_w = math.pi * rc / n * 1.08
    geoms = []
    for k in range(n):
        a = 2 * math.pi * k / n                    # 0 at the bottom
        y, z = rc * math.sin(a), zc - rc * math.cos(a)
        if z + t / 2 < 0.005:                       # under the bed and the silt
            continue
        q = (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)          # the stave's thickness along the radius
        geoms.append(ch.Geom(f"pipe_{k}", "box", (L / 2, half_w, t / 2), pos=(L / 2, y, z), quat=q,
                             friction=(P["mu"], 0.005, 0.0001), rgba=(0.55, 0.56, 0.52, P["alpha"])))
    return [ch.Prop(ch.Link("pipe", pos=(scene.x_mouth, 0.0, 0.0), geoms=geoms))]


def rope_props(scene: Scene) -> tuple:
    """Two stakes, the rope as two halves hinged at them (about the rope's line ... x, and z: it swings away when
    cut), held together by a weld where the right claw meets it."""
    W = scene.wire
    r = W["diameter_m"] / 2
    out = []
    for side, y0 in (("a", -scene.stake_y), ("b", scene.stake_y)):
        h = scene.z_rope + 0.08
        out.append(ch.Prop(ch.Link(f"stake_{side}", pos=(scene.x_rope, y0 + math.copysign(0.03, y0), 0.0),
                                   geoms=[ch.Geom(f"stake_{side}_g", "box", (0.02, 0.02, h / 2), pos=(0, 0, h / 2),
                                                  rgba=(0.35, 0.28, 0.2, 1.0))])))
    yc = scene.cut_y()
    for side, y0 in (("a", -scene.stake_y), ("b", scene.stake_y)):
        Lr = abs(yc - y0)
        m = W["density_kg_m3"] * math.pi * r * r * Lr
        # the rope is strung tight: cut, each half recoils and drops from its stake — RopeRecoil switches on a light
        # spring on the hinge (springref: hanging down) when the knot's weld is released
        down = -math.pi / 2 if y0 < 0 else math.pi / 2
        out.append(ch.Prop(ch.Link(f"rope_{side}", pos=(scene.x_rope, y0, scene.z_rope),
                                   joints=[ch.Joint(f"rope_{side}_hinge", axis=(1, 0, 0), damping=0.02, stiffness=0.0,
                                                    springref=down),
                                           ch.Joint(f"rope_{side}_swing", axis=(0, 0, 1), damping=0.002)],
                                   geoms=[ch.Geom(f"rope_{side}_g", "capsule", (r,), fromto=(0, 0, 0, 0, yc - y0, 0), mass=m,
                                                  friction=(0.4, 0.005, 0.0001), rgba=W["rgba"])]),
                           solref=W["contact_solref"], solimp=W["contact_solimp"]))
    return out, [ch.Weld(scene.ROPE_WELD, "rope_a", "rope_b")]


class RopeRecoil:
    """Scene hook: once the rope's knot (weld) is released, its two halves recoil and drop — the hinges' springs
    (``ROPE['recoil_Nm_rad']`` towards hanging down) are switched on; ``limp_after`` s later the halves stop touching
    the robot (still the bed): a cut rope lying on the bottom is limp and is walked over, which two rigid bars hinged
    at the stakes are not. Restored at reset."""

    def __init__(self, scene: Scene, limp_after: float = 2.0):
        self.scene, self.limp_after = scene, float(limp_after)

    def reset(self, lab):
        import mujoco

        m = lab.model
        self.jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"rope_{s}_hinge") for s in ("a", "b")]
        self.gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in self.scene.WIRE_GEOMS]
        if not hasattr(self, "aff0"):
            self.aff0 = [int(m.geom_conaffinity[g]) for g in self.gids]
        for j in self.jids:
            m.jnt_stiffness[j] = 0.0
        for g, a in zip(self.gids, self.aff0):
            m.geom_conaffinity[g] = a
        self.cut_at, self.limp = None, False

    def __call__(self, lab):
        if self.limp or lab.weld_active(self.scene.ROPE_WELD):
            return
        if self.cut_at is None:
            self.cut_at = lab.time
            for j in self.jids:
                lab.model.jnt_stiffness[j] = self.scene.wire["recoil_Nm_rad"]
        elif lab.time - self.cut_at >= self.limp_after:
            from vegeta.chiron.robot import TERRAIN_BIT

            for g in self.gids:
                lab.model.geom_conaffinity[g] = TERRAIN_BIT
            self.limp = True
            lab.log_event("rope", "the cut halves lie limp on the bed")


def make_lab(scene: Scene | None = None, kind: str = "cut_and_enter", robot=None, **kwargs):
    """The Lobster on the bed with the scenery of ``kind`` ('swim', 'cut_and_enter', 'current')."""
    scene = scene or Scene()
    props, welds = [], []
    if kind == "cut_and_enter":
        props += pipe_props(scene)
        rp, welds = rope_props(scene)
        props += rp
    lab = lr.lobster_lab(terrain(scene), robot=robot, variant=scene.variant, props=props, welds=welds, **kwargs)
    if kind == "cut_and_enter":
        lab.cutter = WireCutter(scene, jaws={"R": ("R_jaw_upper", "R_jaw_lower")}, f_cut=scene.wire["f_cut_N"], t_shear=scene.wire["t_shear_s"])
        lab.add_hook(lab.cutter)
        lab.add_hook(RopeRecoil(scene))
    lab.scene = scene
    return lab


# ----------------------------------------------------------------------------------------------- missions
Phase = lc.Phase


def _tuning(scene: Scene, **given) -> dict:
    """The variant's mission tuning (``lr.Variant.mission``) under the kwargs given explicitly (not None)."""
    ms = dict(scene.v.mission)
    ms.update({k: v for k, v in given.items() if v is not None})
    return ms


def swim_and_land(scene: Scene, *, rpm: float | None = None, depth: float | None = None, x_land: float = 3.0,
                  k_z: float | None = None) -> list:
    """Swim at ``depth`` above the bed (hull centre) to ``x_land``, then land. The single vectored thruster works as
    on an AUV: a depth loop sets the hull's pitch (nose up: the thrust lifts), the tail holds that pitch (``steer``)
    and the heading; the legs stay tucked. Defaults: the variant's ``swim_rpm``, ``swim_depth``, ``swim_k_z``."""
    ms = _tuning(scene, swim_rpm=rpm, swim_depth=depth, swim_k_z=k_z)
    rpm, depth, k_z = ms["swim_rpm"], ms["swim_depth"], ms["swim_k_z"]
    def tuck(m, obs):
        for leg in lr.LEGS:
            m.legs[leg] = [0.0, math.radians(50.0)]
        m.walk = None
        m.gait_phase = 0.0

    def heading(obs):
        yaw_err = lc.yaw_of(obs.base_quat) + 1.5 * float(obs.base_pos[1])
        return float(np.clip(1.0 * yaw_err, -0.5, 0.5))

    def swim(m, obs, tau):
        tuck(m, obs)
        m.rpm = rpm * min(1.0, tau / 0.3)
        z, vz = float(obs.base_pos[2]), float(obs.com_vel[2])
        pitch_ref = float(np.clip(k_z * (depth - z) - 0.8 * vz, math.radians(-20), math.radians(35)))
        m.steer(obs, pitch_ref, yaw=heading(obs))
        return float(obs.base_pos[0]) >= x_land

    def descend(m, obs, tau):
        tuck(m, obs)
        m.rpm = rpm * max(0.0, 1.0 - tau / 3.0)
        m.steer(obs, 0.0, yaw=heading(obs))
        return float(obs.base_pos[2]) < 0.30 or tau >= 6.0

    def legs_down(m, obs, tau):
        for leg in lr.LEGS:
            m.legs[leg] = [0.0, math.radians(50.0) * (1 - lc.smooth(tau / 1.5))]
        m.rpm = 0.0
        m.tail = np.zeros(4)
        return tau >= 1.5 and abs(float(obs.com_vel[2])) < 0.02 and float(obs.base_pos[2]) < 0.2

    return [Phase("tuck the legs", wait(0.3), tuck, 1.0), Phase("swim", swim, None, 40.0),
            Phase("ease off and sink", descend, None, 8.0), Phase("legs down, land", legs_down, None, 15.0),
            Phase("stand", wait(1.0), None, 2.0)]


def wait(T):
    return lambda m, obs, tau: tau >= T


def cut_and_enter(scene: Scene, *, v: float | None = None, notch_x: float | None = None, throat: float | None = None,
                  enter: float = 1.0) -> list:
    """Walk up to the rope, hold the right claw in a cutting pose — the cutter notch ``notch_x`` ahead of the hull
    centre at the rope's height, the jaws level and open — and creep forward until the rope sits in the notch; close
    (the rope parts only under the cutting force), open, back off so the cut ends drop, stow, walk ``enter`` m into
    the pipe. The claw stays fixed
    to the body: the body brings it to the rope, as a crab does. Defaults: the variant's ``walk_v``, ``notch_x``,
    ``throat`` (and ``jaw_open_deg``, ``claw_drop``)."""
    ms = _tuning(scene, walk_v=v, notch_x=notch_x, throat=throat)
    v, notch_x, throat, creep_v = ms["walk_v"], ms["notch_x"], ms["throat"], ms["creep_v"]
    jaw_open, claw_drop = math.radians(ms["jaw_open_deg"]), np.asarray(ms["claw_drop"], dtype=float)
    g = lr.geometry(scene.v.params)
    side = "R"
    z_hull = scene.z_rope - _standing_height(scene.v.params)         # the rope over the hull centre
    cut_pose = np.array([notch_x, scene.cut_y(), z_hull])
    x_cut = scene.x_rope - notch_x                                   # hull x with the notch on the rope

    def until_cut(m, obs, tau):
        return not m.lab.weld_active(scene.ROPE_WELD) and tau > 0.2

    st_stand, up_stand = lc.stand(0.6)
    ph = [Phase("walk to the rope", lc.walk_to(x_cut - 0.15, v=v), None, 60.0), Phase("stand", up_stand, st_stand, 2.0)]
    st, up = lc.claw_joint_move(side, lambda m, obs: lc.claw_ik(cut_pose, side, 0.0, m.g), 2.5)
    ph.append(Phase("R claw: cutting pose, jaws open", up, lc.both(lc.jaw(side, "angle", jaw_open), st), 4.0))
    # push the rope into the throat of the open jaws (the notch is 14 mm from the pin, the hand's face behind it):
    # creep 20 mm past the notch position, the arm's compliance takes the rest
    ph.append(Phase("creep onto the rope", lc.creep_to(x_cut + throat, v=creep_v, tol=0.006), None, 25.0))
    ph.append(Phase("cut", until_cut, lc.jaw(side, "close"), 4.0))
    ph.append(Phase("open", wait(0.6), lc.jaw(side, "angle", jaw_open), 1.0))
    # lower the open claw: the cut end resting on the lower jaw drops to the bed with it
    st, up = lc.claw_line(side, lambda m, obs: cut_pose, lambda m, obs: cut_pose + claw_drop, 1.5)
    ph.append(Phase("R claw: down, the ends drop", up, st, 3.0))
    # back off with the jaws open so the cut ends are clear of the claw before it stows
    ph.append(Phase("back off", lc.creep_to(x_cut - 0.12, v=0.04, tol=0.01), None, 15.0))
    ph.append(Phase("wait", wait(1.5), None, 2.0))
    st, up = lc.claw_joint_move(side, lc.stow_q, 2.0)
    ph.append(Phase("R claw: stow", up, lc.both(lc.jaw(side, "angle", 0.0), st), 3.0))
    ph.append(Phase("walk into the pipe", lc.walk_to(scene.x_mouth + enter, v=v), None, 60.0))
    ph.append(Phase("stand", up_stand, st_stand, 2.0))
    return ph


def _standing_height(p: dict | None = None) -> float:
    """The hull centre over the bed, standing (the legs at 0, the feet on the bed)."""
    import lobster

    p = lr.design_params() if p is None else p
    return (p["shell_bottom"] + p["shell_height"] / 2 - lobster.SikarianLobster.standing_clearance(p)) / 1000.0


def hold_in_current(scene: Scene, *, rpm: float | None = None, hold_s: float = 3.0, walk: float = 0.8,
                    press_deg: float | None = None, press_x: float | None = None, walk_rpm: float | None = None,
                    walk_deg: float | None = None, walk_x: float | None = None) -> list:
    """Stand in the cross-current (it slides), press down with the thrust (it holds), walk ``walk`` m across it. The
    tail curls up over the back and points the thrust ``press_deg`` below the hull's +x along a line through
    ``press_x`` (hull frame, just inside the rear feet): the support polygon carries the moment. Defaults: the
    variant's ``press_rpm``, ``press_deg``, ``press_x``, ``walk_rpm``, ``walk_deg``, ``walk_x``, ``current_walk_v``."""
    ms = _tuning(scene, press_rpm=rpm, press_deg=press_deg, press_x=press_x, walk_rpm=walk_rpm, walk_deg=walk_deg, walk_x=walk_x)
    rpm, press_deg, press_x = ms["press_rpm"], ms["press_deg"], ms["press_x"]
    walk_rpm, walk_deg, walk_x = ms["walk_rpm"], ms["walk_deg"], ms["walk_x"]
    point = (press_x, 0.0, 0.0)

    def mark(m, obs):
        m.memory.setdefault("y_marks", []).append(float(obs.base_pos[1]))

    def press(m, obs, tau):
        m.rpm = rpm * min(1.0, tau / 0.8)
        m.aim(obs, math.radians(press_deg), through=point)
        return tau >= hold_s

    def press_walk(update):
        """Walking, a tripod's triangle carries the robot: the thrust turns to ``walk_deg`` through ``walk_x`` (closer
        to the triangles' common centre; −45°: it pushes forward as well as down) at ``walk_rpm``."""
        def f(m, obs, tau):
            u = lc.smooth(tau / 1.5)
            m.rpm = rpm + (walk_rpm - rpm) * u
            m.aim(obs, math.radians(press_deg + (walk_deg - press_deg) * u), through=(press_x + (walk_x - press_x) * u, 0.0, 0.0))
            return update(m, obs, tau) if tau > 1.5 else False
        return f

    def ease_off(m, obs, tau):
        """The thrust fades and the tail straightens: in the current the robot slides again (it was only held)."""
        u = lc.smooth(tau / 1.5)
        m.rpm = rpm * (1 - u)
        m.tail = m.tail * (1 - u)
        return tau >= 2.0

    def curl(m, obs, tau):
        sol = lc.thrust_line(math.radians(press_deg), point, m.g)
        m.tail = np.array([0.0, sol[0], 0.0, sol[1]]) * lc.smooth(tau / 1.5)
        return tau >= 1.5
    st_stand, up_stand = lc.stand(0.6)
    return [Phase("stand in the current, no thrust", wait(hold_s), mark, hold_s + 1.0),
            Phase("curl the tail over the back", curl, mark, 2.0),
            Phase("press down with the thrust", press, mark, hold_s + 1.0),
            Phase("walk across the current, pressing", press_walk(lc.walk_to(walk, v=ms["current_walk_v"])), mark, 40.0),
            Phase("stand, pressing", press, lc.both(st_stand), hold_s + 1.0),
            Phase("thrust off, tail down", ease_off, None, 3.0)]


def run(kind: str, scene: Scene | None = None, lab=None, duration: float | None = None, **mission_kw):
    """One job: 'swim', 'cut_and_enter' or 'current'. Returns the Episode (``log['mission']``, ``log['events']``).
    ``duration`` defaults to the variant's (``mission["duration"][kind]``)."""
    scene = scene or Scene()
    duration = duration or scene.v.mission["duration"][kind]
    if kind == "swim":
        lab = lab or make_lab(scene, "swim")
        mission = lc.Mission(swim_and_land(scene, **mission_kw), name="swim and land")
        kw = dict(base_pos=(0.0, 0.0, 1.0))
    elif kind == "cut_and_enter":
        lab = lab or make_lab(scene, "cut_and_enter")
        mission = lc.Mission(cut_and_enter(scene, **mission_kw), name="cut the rope, enter the pipe")
        kw = {}
    elif kind == "current":
        lab = lab or make_lab(scene, "current", wind=(0.0, 0.5, 0.0), course_extent=(-2.0, 6.0, -2.0, 6.0))
        mission = lc.Mission(hold_in_current(scene, **mission_kw), name="hold in a current")
        kw = {}
    else:
        raise ValueError("kind must be 'swim', 'cut_and_enter' or 'current'")
    ep = lab.run(mission, duration=duration, rules=None, settle=0.0, seed=0, info={"controller": mission.name}, **kw)
    ep.log["mission"] = [list(x) for x in mission.log]
    ep.log["mission_finished"] = mission.finished
    if hasattr(lab, "cutter"):
        ep.log["cut_at"] = lab.cutter.cut_at
        ep.log["cutter_history"] = np.asarray(lab.cutter.history, dtype=float).reshape(-1, 3)
    return ep


def timeseries(ep) -> pd.DataFrame:
    """Per sample: hull position, speed, tilt, yaw; tail joints; the right jaw's torque."""
    log = ep.log
    names = list(log["joints"])
    q, tau = np.asarray(log["q"]), np.asarray(log["tau"])
    bq = np.asarray(log["body_quat"])[:, 0]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (bq[:, 1] ** 2 + bq[:, 2] ** 2), -1, 1)))
    pos = np.asarray(log["body_pos"])[:, 0]
    df = pd.DataFrame({"t": log["t"], "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
                       "vx": np.asarray(log["com_vel"])[:, 0], "vz": np.asarray(log["com_vel"])[:, 2], "tilt_deg": tilt,
                       "yaw_deg": np.degrees([lc.yaw_of(b) for b in bq])})
    for j in lr.TAIL_JOINTS:
        df[j + "_deg"] = np.degrees(q[:, names.index(j)])
    df["tau_R_jaw"] = tau[:, names.index("R_jaw_upper")]
    fz = np.asarray(log["foot_force"])[:, :, 2]
    df["foot_load_N"] = fz.sum(axis=1)
    return df
