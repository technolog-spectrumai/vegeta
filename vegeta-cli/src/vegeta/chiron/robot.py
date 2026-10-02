"""Robot descriptions: plain dataclasses that need no MuJoCo to build, and their MJCF.

A robot is a tree of ``Link`` s from a root that floats freely in the world. Each link carries its geometry
(``Geom``, with a role: ``foot``, ``body``, ``link`` or ``visual``), concentrated masses (``PointMass``) and the
joints that connect it to its parent (none = welded). A joint with a ``Servo`` is actuated (a PD position loop
clipped to the motor's torque–speed line, see ``servo.py``); one without is passive (spring, damper, limits).
``FootSpec`` names the foot geoms, their leg joints and the body each leg is mounted on.

Units: SI (m, kg, s, N, N·m, rad). Every physical number is an input: geometry and masses come from the design,
servo data from the actuator datasheet — put the source in ``Servo.source`` / ``Robot.sources``.

Third-party models (an MJCF file, e.g. from a model zoo) are wrapped with ``Robot.from_mjcf(xml, RobotMeta)``.

**Scenery.** ``Prop`` places things that are not the robot in the world — a post, a wire, a log to lift: a Link
tree, fixed to the world or free (a loose object), whose colliding geoms touch the robot, the terrain and the
other props. ``Weld`` is an equality constraint between two bodies (or a body and the world) that ChironLab can
switch off at run time — a wire that parts when it is cut, a load that is released.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterator
from xml.sax.saxutils import quoteattr

import numpy as np

from .terrain import Flat, Terrain

__all__ = ["Servo", "Joint", "Geom", "PointMass", "Link", "FootSpec", "Robot", "RobotMeta", "SimOptions", "Prop",
           "Weld", "TERRAIN_NAME", "GEOM_ROLES", "JOINT_TAGS"]

TERRAIN_NAME = "chiron_terrain"
GEOM_TYPES = ("box", "sphere", "capsule", "cylinder", "ellipsoid")
GEOM_ROLES = ("foot", "body", "link", "visual")
JOINT_KINDS = ("hinge", "slide")
#: Conventional joint tags (any string is accepted; metrics group joints by tag).
JOINT_TAGS = ("hip_yaw", "hip_pitch", "hip_roll", "knee", "ankle", "body_pitch", "body_yaw", "body_roll")
TERRAIN_BIT, ROBOT_BIT, PROP_BIT = 1, 2, 4
POINT_MASS_RADIUS = 1e-3  # m; a point mass is a 1 mm sphere (its own inertia is negligible)


def _f(v) -> str:
    return " ".join(f"{float(x):.10g}" for x in np.atleast_1d(v))


# ----------------------------------------------------------------------------------------------- building blocks
@dataclass(frozen=True)
class Servo:
    """A position servo: PD loop with gains ``kp`` [N·m/rad], ``kd`` [N·m·s/rad], output torque clipped to the
    DC-motor line ``|τ| ≤ stall_torque · (1 − |ω|/no_load_speed)`` (zero beyond the no-load speed).

    ``stall_torque`` [N·m], ``rated_torque`` (continuous/thermal) [N·m], ``no_load_speed`` [rad/s],
    ``stall_current`` [A] at ``voltage`` [V] — datasheet values (``source``). ``armature`` [kg·m²] is the
    reflected rotor inertia (rotor inertia × gear ratio²) added to the joint; 0 when unknown.
    """

    stall_torque: float
    rated_torque: float
    no_load_speed: float
    stall_current: float
    voltage: float
    kp: float
    kd: float
    armature: float = 0.0
    source: str = ""

    def __post_init__(self):
        for name in ("stall_torque", "no_load_speed"):
            if not getattr(self, name) > 0:
                raise ValueError(f"Servo.{name} must be positive")
        if self.kp < 0 or self.kd < 0 or self.armature < 0:
            raise ValueError("Servo gains and armature must be non-negative")

    @classmethod
    def from_rpm(cls, *, stall_torque, rated_torque, no_load_rpm, stall_current, voltage, kp, kd, armature=0.0,
                 source=""):
        """Same as the constructor with the no-load speed in rpm (datasheets quote rpm)."""
        return cls(stall_torque, rated_torque, no_load_rpm * 2 * math.pi / 60.0, stall_current, voltage, kp, kd,
                   armature, source)

    @classmethod
    def from_actuator(cls, act, *, kp, kd, armature=0.0):
        """From a catalogue entry with ``stall_Nm, rated_Nm, no_load_rpm, stall_A, voltage_V, source`` attributes
        (duck-typed, e.g. ``notebooks/designs/actuators.py``)."""
        return cls.from_rpm(stall_torque=act.stall_Nm, rated_torque=act.rated_Nm, no_load_rpm=act.no_load_rpm,
                            stall_current=act.stall_A, voltage=act.voltage_V, kp=kp, kd=kd, armature=armature,
                            source=f"{getattr(act, 'key', '')}: {getattr(act, 'source', '')}".strip(": "))

    @property
    def torque_constant(self) -> float:
        """k_t = τ_stall / I_stall [N·m/A] (a DC-motor estimate from the stall point)."""
        return self.stall_torque / self.stall_current

    @property
    def resistance(self) -> float:
        """R = V / I_stall [Ω] (a DC-motor estimate from the stall point)."""
        return self.voltage / self.stall_current

    def torque_limit(self, speed):
        """|τ| bound [N·m] at joint speed ``speed`` [rad/s]."""
        return self.stall_torque * np.maximum(0.0, 1.0 - np.abs(speed) / self.no_load_speed)


@dataclass
class Joint:
    """A joint between a link and its parent, in the link's frame.

    ``kind`` 'hinge' (axis of rotation, angles in rad) or 'slide'; ``pos`` the anchor [m]; ``range`` (lo, hi)
    limits (None = unlimited); ``stiffness`` [N·m/rad] about ``springref``, ``damping`` [N·m·s/rad],
    ``armature`` [kg·m²] and ``frictionloss`` [N·m] are passive properties. ``tag`` classifies the joint for
    metrics ('hip_yaw', 'hip_pitch', 'knee', 'body_pitch', ...); ``servo`` makes it actuated (None = passive);
    ``leg`` names the foot (FootSpec.name) whose leg it belongs to.
    """

    name: str
    kind: str = "hinge"
    axis: tuple = (0.0, 0.0, 1.0)
    pos: tuple = (0.0, 0.0, 0.0)
    range: tuple | None = None
    stiffness: float = 0.0
    damping: float = 0.0
    springref: float = 0.0
    armature: float = 0.0
    frictionloss: float = 0.0
    tag: str = ""
    servo: Servo | None = None
    leg: str | None = None

    def __post_init__(self):
        if self.kind not in JOINT_KINDS:
            raise ValueError(f"joint {self.name}: kind must be one of {JOINT_KINDS}")
        if self.range is not None and not self.range[0] < self.range[1]:
            raise ValueError(f"joint {self.name}: range must be (lo, hi) with lo < hi")

    @property
    def active(self) -> bool:
        return self.servo is not None


@dataclass
class Geom:
    """A shape on a link (MJCF conventions): ``size`` is (half-x, half-y, half-z) for a box, (r,) for a sphere,
    (r, half-length) for a capsule/cylinder along the local z axis — or (r,) with ``fromto`` (x1 y1 z1 x2 y2 z2)
    — and the three radii of an ellipsoid. ``pos``/``quat`` (w, x, y, z) place it in the link frame.

    ``mass`` [kg] (None = massless); ``friction`` (sliding, torsional, rolling) — the robot's value governs every
    terrain contact. ``role``: 'foot' (a foot pad named in a FootSpec), 'body' (a shell whose ground contact is a
    belly contact), 'link' (any other colliding part) or 'visual' (never collides).
    """

    name: str
    type: str
    size: tuple
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    fromto: tuple | None = None
    mass: float | None = None
    friction: tuple = (1.0, 0.005, 0.0001)
    role: str = "link"
    rgba: tuple | None = None

    def __post_init__(self):
        if self.type not in GEOM_TYPES:
            raise ValueError(f"geom {self.name}: type must be one of {GEOM_TYPES}")
        if self.role not in GEOM_ROLES:
            raise ValueError(f"geom {self.name}: role must be one of {GEOM_ROLES}")
        if self.mass is not None and self.mass < 0:
            raise ValueError(f"geom {self.name}: negative mass")


@dataclass
class PointMass:
    """A concentrated mass [kg] at ``pos`` in the link frame (a servo, a battery, a PCB)."""

    name: str
    mass: float
    pos: tuple = (0.0, 0.0, 0.0)


@dataclass
class Link:
    """A rigid body. ``pos``/``quat`` place its frame in the parent's frame (the root's in the world at build
    time; ChironLab.reset places the robot). ``joints`` connect it to the parent (empty = welded; the root always
    gets a free joint). ``log`` marks a body whose pose and velocity are logged; ``group`` names the unit it
    belongs to (e.g. 'segment 1'; default its own name). Frame convention for logged bodies: x forward, y left,
    z up."""

    name: str
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    geoms: list = field(default_factory=list)
    masses: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    children: list = field(default_factory=list)
    log: bool = False
    group: str | None = None

    def walk(self, parent: "Link | None" = None) -> Iterator[tuple["Link", "Link | None"]]:
        """Depth-first (link, parent) pairs, this link first."""
        yield self, parent
        for c in self.children:
            yield from c.walk(self)

    def own_mass(self) -> float:
        return sum(g.mass or 0.0 for g in self.geoms) + sum(p.mass for p in self.masses)


@dataclass
class Prop:
    """Scenery that is not the robot: a Link tree placed in the world by ``root.pos`` / ``root.quat``.

    ``free`` True gives the root a free joint (a loose object: a log, a box); False fixes it to the world (a
    post, a wall), while its links' own Joints (hinges, slides; passive — no servos) still move them (a gate, a
    wire hinged at a post). Its colliding geoms touch the robot, the terrain and the other props. ``priority``
    above the robot's 1 makes the prop's ``solref`` / ``solimp`` (None = MuJoCo's defaults) and its geoms'
    friction govern its contacts with the robot; ``condim`` its contact dimensionality (None = the lab's). A
    ``solref`` with negative values is MuJoCo's direct (−stiffness [N/m], −damping [N·s/m]) — a stiff contact
    for a thin object a gripper squeezes. ``log``: the poses of its links are logged (``prop_pos``,
    ``prop_quat``). Names (links, joints, geoms) must not clash with the robot's.
    """

    root: Link
    free: bool = False
    priority: int = 2
    solref: tuple | None = None
    solimp: tuple | None = None
    condim: int | None = None
    log: bool = True

    @property
    def name(self) -> str:
        return self.root.name

    def links(self) -> list:
        return [lk for lk, _ in self.root.walk()]

    def total_mass(self) -> float:
        return sum(lk.own_mass() for lk in self.links())


@dataclass
class Weld:
    """An equality constraint holding ``body1`` to ``body2`` (None = the world) in their relative pose of the
    model's reference configuration. ``active`` at reset; ChironLab's ``set_weld(name, False)`` releases it at run
    time (a wire cut through, a load let go). ``solref`` None = MuJoCo's default stiffness."""

    name: str
    body1: str
    body2: str | None = None
    active: bool = True
    solref: tuple | None = None


@dataclass
class FootSpec:
    """One foot: ``geom`` (the pad, role 'foot'), ``joints`` (the leg's actuated joints, hip to foot) and
    ``body`` (the logged body the leg is mounted on)."""

    name: str
    geom: str
    joints: list
    body: str


@dataclass
class RobotMeta:
    """What Chiron needs to know about a model beyond its MJCF (also derived automatically for Link trees).

    ``logged_bodies`` (MJCF body names), ``body_groups`` (body → group), ``joint_tags`` (joint → tag),
    ``servos`` (actuated joint → Servo; every other hinge/slide joint is passive), ``joint_legs`` (joint →
    foot name; default from ``feet``), ``nominal_qpos`` (joint → angle at the standing pose),
    ``nominal_base_height`` (root-frame height above flat ground when standing; None = computed so the lowest
    foot just touches), ``nominal_hip_height`` (None = mean height of each leg's first joint at the standing
    pose), ``belly_geoms`` (geom → logged body; None = every colliding non-foot geom of a logged body),
    ``remove_geoms`` (geoms to drop from a third-party file, e.g. its floor).
    """

    feet: list = field(default_factory=list)
    logged_bodies: list = field(default_factory=list)
    body_groups: dict = field(default_factory=dict)
    joint_tags: dict = field(default_factory=dict)
    servos: dict = field(default_factory=dict)
    joint_legs: dict = field(default_factory=dict)
    nominal_qpos: dict = field(default_factory=dict)
    nominal_base_height: float | None = None
    nominal_hip_height: float | None = None
    belly_geoms: dict | None = None
    remove_geoms: tuple = ()
    name: str = ""
    notes: str = ""
    sources: dict = field(default_factory=dict)

    def leg_of_joint(self) -> dict:
        legs = {j: f.name for f in self.feet for j in f.joints}
        legs.update(self.joint_legs)
        return legs


@dataclass
class SimOptions:
    """MuJoCo settings written into the MJCF (defaults: MuJoCo's own unless stated).

    ``integrator`` 'implicitfast' (default here: the servos' damping term is integrated implicitly, which keeps
    light legs stable at 1 ms), 'Euler' or 'implicit'; ``cone`` 'pyramidal'/'elliptic'; ``contact_solref``/
    ``contact_solimp`` None = MuJoCo defaults; ``condim`` 3 (sliding friction) for every robot geom;
    ``self_collision`` False: robot geoms collide only with the terrain. The terrain is a MuJoCo height field
    sampled every ``heightfield_cell`` metres over ``course_extent`` (x0, x1, y0, y1); a flat terrain is a plane
    when ``flat_as_plane``. ``heightfield_base`` is the height-field slab thickness below its lowest point.
    """

    timestep: float = 0.001
    gravity: tuple = (0.0, 0.0, -9.81)
    integrator: str = "implicitfast"
    cone: str = "pyramidal"
    impratio: float = 1.0
    iterations: int = 100
    tolerance: float = 1e-8
    noslip_iterations: int = 0
    condim: int = 3
    contact_solref: tuple | None = None
    contact_solimp: tuple | None = None
    self_collision: bool = False
    course_extent: tuple = (-1.0, 3.0, -1.0, 1.0)
    heightfield_cell: float = 0.005
    heightfield_base: float = 0.1
    flat_as_plane: bool = True
    memory: str = "64M"
    terrain_rgba: tuple = (0.55, 0.5, 0.45, 1.0)


# ----------------------------------------------------------------------------------------------- the robot
@dataclass
class Robot:
    """A floating-base robot: a ``root`` Link tree (or a third-party MJCF, see ``from_mjcf``).

    ``feet`` (FootSpec list), ``nominal_qpos`` (joint → standing angle [rad]; unnamed joints stand at 0),
    ``nominal_base_height`` (root frame height when standing on flat ground [m]; None = computed so the lowest
    foot just touches the ground), ``nominal_hip_height`` (None = computed: mean height of the legs' first
    joints when standing), ``notes`` and ``sources`` (where every number comes from).
    """

    name: str
    root: Link | None = None
    feet: list = field(default_factory=list)
    nominal_qpos: dict = field(default_factory=dict)
    nominal_base_height: float | None = None
    notes: str = ""
    sources: dict = field(default_factory=dict)
    nominal_hip_height: float | None = None
    mjcf: str | None = None
    meta_override: RobotMeta | None = None
    assets: dict | None = None

    # ---- construction from MJCF
    @classmethod
    def from_mjcf(cls, xml: str, meta: RobotMeta, *, name: str | None = None, assets: dict | None = None) -> "Robot":
        """Wrap a third-party MJCF model (its root body must have a free joint). Its own actuators are removed
        and every joint named in ``meta.servos`` gets a Chiron servo; ``assets`` (filename → bytes) resolves
        meshes referenced by the file."""
        if not isinstance(meta, RobotMeta):
            raise TypeError("meta must be a RobotMeta")
        return cls(name=name or meta.name or "mjcf robot", root=None, feet=list(meta.feet),
                   nominal_qpos=dict(meta.nominal_qpos), nominal_base_height=meta.nominal_base_height,
                   notes=meta.notes, sources=dict(meta.sources), nominal_hip_height=meta.nominal_hip_height,
                   mjcf=xml, meta_override=meta, assets=assets)

    @property
    def is_mjcf(self) -> bool:
        return self.mjcf is not None

    # ---- tree queries
    def links(self) -> list:
        if self.root is None:
            return []
        return [lk for lk, _ in self.root.walk()]

    def joints(self) -> list:
        return [j for lk in self.links() for j in lk.joints]

    def geoms(self) -> list:
        return [g for lk in self.links() for g in lk.geoms]

    def actuated_joints(self) -> list:
        if self.is_mjcf:
            return list(self.meta_override.servos)
        return [j.name for j in self.joints() if j.servo is not None]

    def total_mass(self) -> float:
        """Total mass [kg] (geoms with a mass + point masses; for MJCF robots: the compiled model's)."""
        if self.is_mjcf:
            m = self._compile_plain()
            free = [j for j in range(m.njnt) if int(m.jnt_type[j]) == 0]      # mjJNT_FREE: the robot's root
            if len(free) != 1:
                raise ValueError(f"the MJCF needs exactly one free joint (the robot's root); found {len(free)}")
            return float(m.body_subtreemass[m.jnt_bodyid[free[0]]])
        return float(sum(lk.own_mass() for lk in self.links()))

    def meta(self) -> RobotMeta:
        """The RobotMeta ChironLab uses: given for MJCF robots, derived from the Link tree otherwise."""
        if self.is_mjcf:
            return self.meta_override
        links = self.links()
        logged = [lk.name for lk in links if lk.log]
        groups = {lk.name: (lk.group or lk.name) for lk in links if lk.log}
        tags, servos, legs = {}, {}, {}
        for j in self.joints():
            if j.tag:
                tags[j.name] = j.tag
            if j.servo is not None:
                servos[j.name] = j.servo
            if j.leg is not None:
                legs[j.name] = j.leg
        belly = {}
        for lk, owner in _logged_owner(self.root):
            for g in lk.geoms:
                if g.role == "body" and owner is not None:
                    belly[g.name] = owner
        return RobotMeta(feet=list(self.feet), logged_bodies=logged, body_groups=groups, joint_tags=tags,
                         servos=servos, joint_legs=legs, nominal_qpos=dict(self.nominal_qpos),
                         nominal_base_height=self.nominal_base_height, nominal_hip_height=self.nominal_hip_height,
                         belly_geoms=belly, name=self.name, notes=self.notes, sources=dict(self.sources))

    def validate(self) -> None:
        """Raise ValueError on a malformed description (duplicate names, unknown feet/joints, massless moving
        links, a root with joints)."""
        if self.is_mjcf:
            m = self._compile_plain()
            import mujoco

            names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)}
            gnames = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)}
            bnames = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
            meta = self.meta_override
            for j in list(meta.servos) + list(meta.joint_tags) + list(meta.nominal_qpos):
                if j not in names:
                    raise ValueError(f"RobotMeta names joint {j!r}, which is not in the MJCF")
            for f in meta.feet:
                if f.geom not in gnames:
                    raise ValueError(f"foot {f.name}: geom {f.geom!r} not in the MJCF")
                for j in f.joints:
                    if j not in names:
                        raise ValueError(f"foot {f.name}: joint {j!r} not in the MJCF")
            for b in meta.logged_bodies:
                if b not in bnames:
                    raise ValueError(f"logged body {b!r} not in the MJCF")
            return
        if self.root is None:
            raise ValueError("robot has no root link")
        if self.root.joints:
            raise ValueError("the root link gets a free joint; give it no joints")
        seen = {}
        for kind, items in (("link", self.links()), ("joint", self.joints()), ("geom", self.geoms())):
            for it in items:
                if it.name in seen.get(kind, set()):
                    raise ValueError(f"duplicate {kind} name {it.name!r}")
                seen.setdefault(kind, set()).add(it.name)
        pm = [p.name for lk in self.links() for p in lk.masses]
        if len(set(pm)) != len(pm):
            raise ValueError("duplicate point-mass names")
        geoms = {g.name: g for g in self.geoms()}
        joints = {j.name: j for j in self.joints()}
        logged = {lk.name for lk in self.links() if lk.log}
        foot_names = set()
        for f in self.feet:
            if f.name in foot_names:
                raise ValueError(f"duplicate foot {f.name!r}")
            foot_names.add(f.name)
            if f.geom not in geoms:
                raise ValueError(f"foot {f.name}: unknown geom {f.geom!r}")
            if geoms[f.geom].role != "foot":
                raise ValueError(f"foot {f.name}: geom {f.geom!r} must have role 'foot'")
            for j in f.joints:
                if j not in joints:
                    raise ValueError(f"foot {f.name}: unknown joint {j!r}")
            if f.body not in logged:
                raise ValueError(f"foot {f.name}: body {f.body!r} is not a logged link (Link.log=True)")
        for j in self.joints():
            if j.leg is not None and j.leg not in foot_names:
                raise ValueError(f"joint {j.name}: leg {j.leg!r} is not a foot name")
        for j in self.nominal_qpos:
            if j not in joints:
                raise ValueError(f"nominal_qpos names unknown joint {j!r}")
        for lk, _ in self.root.walk():
            if lk.joints and _subtree_mass(lk) <= 0:
                raise ValueError(f"link {lk.name} moves but it and its subtree have no mass")
        if self.root.own_mass() <= 0 and _subtree_mass(self.root) <= 0:
            raise ValueError("robot has no mass")

    # ---- MJCF
    def to_mjcf(self, terrain: Terrain | None = None, options: SimOptions | None = None, *,
                heightfield: str = "inline", props=(), welds=()) -> str:
        """The MJCF model of this robot on ``terrain`` (default flat) with ``options``, plus scenery (``props``,
        ``welds``; Link-tree robots only).

        ``heightfield='inline'`` writes the sampled elevations into the file (self-contained);
        ``'deferred'`` leaves them out — ChironLab then fills ``model.hfield_data`` after compiling (faster).
        """
        terrain = terrain if terrain is not None else Flat()
        opt = options or SimOptions()
        props, welds = list(props), list(welds)
        if self.is_mjcf:
            if props or welds:
                raise ValueError("props and welds need a Link-tree robot (not Robot.from_mjcf)")
            return self._mjcf_from_spec(terrain, opt, heightfield)
        self.validate()
        _validate_scenery(self, props, welds)
        meta_servo = {j.name: j.servo for j in self.joints() if j.servo is not None}
        out = [f'<mujoco model={quoteattr(self.name)}>',
               '  <compiler angle="radian" autolimits="true" inertiafromgeom="true"/>',
               "  " + _option_xml(opt),
               f'  <size memory="{opt.memory}"/>']
        tinfo = terrain_geometry(terrain, opt)
        if tinfo["type"] == "hfield":
            out.append("  <asset>")
            out.append("    " + _hfield_xml(tinfo, heightfield))
            out.append("  </asset>")
        out.append("  <worldbody>")
        out.append("    " + _terrain_geom_xml(tinfo, opt))
        coll = _collision_bits(opt)
        out += _link_xml(self.root, opt, coll, indent=4, is_root=True)
        for pr in props:
            out += _prop_xml(pr, opt)
        out.append("  </worldbody>")
        if welds:
            out.append("  <equality>")
            for w in welds:
                a = [f"name={quoteattr(w.name)}", f"body1={quoteattr(w.body1)}"]
                if w.body2 is not None:
                    a.append(f"body2={quoteattr(w.body2)}")
                a.append(f'active="{str(bool(w.active)).lower()}"')
                if w.solref is not None:
                    a.append(f'solref="{_f(w.solref)}"')
                out.append("    <weld " + " ".join(a) + "/>")
            out.append("  </equality>")
        if meta_servo:
            out.append("  <actuator>")
            for jn, s in meta_servo.items():
                out.append(f'    <general name={quoteattr("servo:" + jn)} joint={quoteattr(jn)} gainprm="1" '
                           f'biastype="affine" biasprm="0 0 {-s.kd:.10g}" ctrllimited="false"/>')
            out.append("  </actuator>")
        out.append("</mujoco>")
        return "\n".join(out) + "\n"

    def _compile_plain(self):
        import mujoco

        spec = mujoco.MjSpec.from_string(self.mjcf, assets=self.assets or {})
        return spec.compile()

    def _build_spec(self, terrain: Terrain, opt: SimOptions):
        """MjSpec of a third-party model edited for Chiron (actuators, collision bits, options, terrain)."""
        import mujoco

        meta = self.meta_override
        spec = mujoco.MjSpec.from_string(self.mjcf, assets=self.assets or {})
        for a in list(spec.actuators):
            spec.delete(a)
        for gname in meta.remove_geoms:
            g = spec.geom(gname)
            if g is None:
                raise ValueError(f"remove_geoms: no geom {gname!r}")
            spec.delete(g)
        o = spec.option
        o.timestep = opt.timestep
        o.gravity = list(opt.gravity)
        o.integrator = {"euler": mujoco.mjtIntegrator.mjINT_EULER, "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                        "rk4": mujoco.mjtIntegrator.mjINT_RK4}[opt.integrator.lower()]
        o.cone = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[opt.cone]
        o.impratio = opt.impratio
        o.iterations = opt.iterations
        o.tolerance = opt.tolerance
        o.noslip_iterations = opt.noslip_iterations
        ctype, caff = _collision_bits(opt)
        for g in spec.geoms:
            if g.parent.name == spec.worldbody.name:      # static scenery keeps its own collision settings
                continue
            if g.contype or g.conaffinity:
                g.contype, g.conaffinity, g.priority = ctype, caff, 1
                g.condim = opt.condim
                if opt.contact_solref is not None:
                    g.solref = list(opt.contact_solref)
                if opt.contact_solimp is not None:
                    g.solimp = list(opt.contact_solimp)
        tinfo = terrain_geometry(terrain, opt)
        tg = spec.worldbody.add_geom()
        tg.name = TERRAIN_NAME
        tg.contype, tg.conaffinity, tg.priority = TERRAIN_BIT, ROBOT_BIT, 0
        tg.rgba = list(opt.terrain_rgba)
        if tinfo["type"] == "plane":
            tg.type = mujoco.mjtGeom.mjGEOM_PLANE
            tg.size = [0.0, 0.0, 1.0]
        else:
            hf = spec.add_hfield()
            hf.name = TERRAIN_NAME
            hf.nrow, hf.ncol = tinfo["nrow"], tinfo["ncol"]
            hf.size = list(tinfo["size"])
            hf.userdata = tinfo["data"].ravel().tolist()
            tg.type = mujoco.mjtGeom.mjGEOM_HFIELD
            tg.hfieldname = TERRAIN_NAME
            tg.pos = list(tinfo["pos"])
        order = [j.name for j in spec.joints]
        missing = [jn for jn in meta.servos if jn not in order]
        if missing:
            raise ValueError(f"RobotMeta.servos: no joints {missing} in the MJCF")
        for jn in [n for n in order if n in meta.servos]:      # actuators in MuJoCo joint order
            s = meta.servos[jn]
            j = spec.joint(jn)
            j.armature = j.armature + s.armature
            a = spec.add_actuator()
            a.name = "servo:" + jn
            a.target = jn
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.gainprm[0] = 1.0
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[2] = -s.kd
            a.ctrllimited = 0
        return spec

    def _mjcf_from_spec(self, terrain, opt, heightfield):
        # MjSpec writes the height field inline; ``heightfield`` only matters for Link-tree robots.
        spec = self._build_spec(terrain, opt)
        spec.compile()
        return spec.to_xml()

    # ---- summary
    def summary(self) -> dict:
        """Plain-dict description: masses, joints (tag, active, servo), feet, logged bodies."""
        meta = self.meta()
        if self.is_mjcf:
            return {"name": self.name, "source": "mjcf", "total_mass_kg": self.total_mass(),
                    "n_actuated": len(meta.servos), "feet": [asdict(f) for f in meta.feet],
                    "logged_bodies": list(meta.logged_bodies), "joint_tags": dict(meta.joint_tags),
                    "servos": {k: asdict(v) for k, v in meta.servos.items()}, "notes": self.notes,
                    "sources": dict(self.sources)}
        joints = []
        for lk in self.links():
            for j in lk.joints:
                joints.append({"name": j.name, "link": lk.name, "kind": j.kind, "tag": j.tag, "active": j.active,
                               "range": list(j.range) if j.range else None, "stiffness": j.stiffness,
                               "damping": j.damping, "leg": j.leg,
                               "servo": asdict(j.servo) if j.servo else None})
        return {"name": self.name, "source": "chiron", "total_mass_kg": self.total_mass(),
                "n_links": len(self.links()), "n_joints": len(joints),
                "n_actuated": sum(1 for j in joints if j["active"]),
                "n_passive": sum(1 for j in joints if not j["active"]),
                "link_mass_kg": {lk.name: lk.own_mass() for lk in self.links()},
                "logged_bodies": list(meta.logged_bodies), "body_groups": dict(meta.body_groups),
                "feet": [asdict(f) for f in self.feet], "joints": joints,
                "nominal_qpos": dict(self.nominal_qpos), "nominal_base_height": self.nominal_base_height,
                "notes": self.notes, "sources": dict(self.sources)}


# ----------------------------------------------------------------------------------------------- MJCF helpers
def _subtree_mass(link: Link) -> float:
    return link.own_mass() + sum(_subtree_mass(c) for c in link.children)


def _logged_owner(root: Link):
    """(link, nearest logged ancestor-or-self name) pairs."""
    def rec(lk, owner):
        own = lk.name if lk.log else owner
        yield lk, own
        for c in lk.children:
            yield from rec(c, own)
    yield from rec(root, None)


def _validate_scenery(robot: "Robot", props: list, welds: list) -> None:
    """Names unique across robot and props; prop joints passive; welds name known bodies."""
    taken = {"link": {lk.name for lk in robot.links()}, "joint": {j.name for j in robot.joints()},
             "geom": {g.name for g in robot.geoms()}}
    bodies = set(taken["link"])
    for pr in props:
        if not isinstance(pr, Prop):
            raise TypeError("props must be chiron.Prop")
        if pr.free and pr.root.joints:
            raise ValueError(f"prop {pr.name}: a free prop's root gets a free joint; give it no joints")
        for lk in pr.links():
            for kind, items in (("link", [lk]), ("joint", lk.joints), ("geom", lk.geoms)):
                for it in items:
                    if it.name in taken[kind]:
                        raise ValueError(f"prop {pr.name}: duplicate {kind} name {it.name!r}")
                    taken[kind].add(it.name)
            for j in lk.joints:
                if j.servo is not None:
                    raise ValueError(f"prop {pr.name}: joint {j.name} has a servo; props are passive")
            if lk.joints and _subtree_mass(lk) <= 0:
                raise ValueError(f"prop {pr.name}: link {lk.name} moves but has no mass")
        if pr.free and pr.total_mass() <= 0:
            raise ValueError(f"prop {pr.name}: a free prop needs mass")
        bodies |= {lk.name for lk in pr.links()}
    names = set()
    for w in welds:
        if not isinstance(w, Weld):
            raise TypeError("welds must be chiron.Weld")
        if w.name in names:
            raise ValueError(f"duplicate weld {w.name!r}")
        names.add(w.name)
        for b in (w.body1, w.body2):
            if b is not None and b not in bodies:
                raise ValueError(f"weld {w.name}: no body {b!r}")


def _prop_xml(pr: "Prop", opt: SimOptions) -> list:
    """A prop's bodies: its own contact settings, colliding with terrain, robot and other props."""
    coll = (PROP_BIT, TERRAIN_BIT | ROBOT_BIT | PROP_BIT)
    extra = [f'priority="{int(pr.priority)}"', f'condim="{int(pr.condim if pr.condim is not None else opt.condim)}"']
    if pr.solref is not None:
        extra.append(f'solref="{_f(pr.solref)}"')
    if pr.solimp is not None:
        extra.append(f'solimp="{_f(pr.solimp)}"')

    def body(link: Link, indent: int, root: bool) -> list:
        pad = " " * indent
        lines = [f'{pad}<body name={quoteattr(link.name)} pos="{_f(link.pos)}" quat="{_f(link.quat)}">']
        if root and pr.free:
            lines.append(f'{pad}  <freejoint name={quoteattr(link.name + ":free")}/>')
        for j in link.joints:
            lines.append(pad + "  " + _joint_xml(j))
        for g in link.geoms:
            a = [f"name={quoteattr(g.name)}", f'type="{g.type}"', f'size="{_f(g.size)}"']
            a.append(f'fromto="{_f(g.fromto)}"' if g.fromto is not None else f'pos="{_f(g.pos)}" quat="{_f(g.quat)}"')
            a.append(f'mass="{(g.mass or 0.0):.10g}"')
            if g.role == "visual":
                a.append('contype="0" conaffinity="0" group="1"')
            else:
                a.append(f'contype="{coll[0]}" conaffinity="{coll[1]}" friction="{_f(g.friction)}" ' + " ".join(extra))
            if g.rgba is not None:
                a.append(f'rgba="{_f(g.rgba)}"')
            lines.append(pad + "  <geom " + " ".join(a) + "/>")
        for pm in link.masses:
            lines.append(f'{pad}  <geom name={quoteattr("pm:" + pm.name)} type="sphere" size="{POINT_MASS_RADIUS}" '
                         f'pos="{_f(pm.pos)}" mass="{pm.mass:.10g}" contype="0" conaffinity="0" group="4" '
                         f'rgba="0 0 0 0"/>')
        for c in link.children:
            lines += body(c, indent + 2, False)
        lines.append(f"{pad}</body>")
        return lines

    return body(pr.root, 4, True)


def _collision_bits(opt: SimOptions):
    """(contype, conaffinity) for colliding robot geoms: terrain only, or terrain and robot."""
    return ROBOT_BIT, (TERRAIN_BIT | ROBOT_BIT) if opt.self_collision else TERRAIN_BIT


def _option_xml(opt: SimOptions) -> str:
    names = {"euler": "Euler", "implicit": "implicit", "implicitfast": "implicitfast", "rk4": "RK4"}
    integ = names[opt.integrator.lower()]
    return (f'<option timestep="{opt.timestep:.10g}" gravity="{_f(opt.gravity)}" integrator="{integ}" '
            f'cone="{opt.cone}" impratio="{opt.impratio:.10g}" iterations="{int(opt.iterations)}" '
            f'tolerance="{opt.tolerance:.10g}" noslip_iterations="{int(opt.noslip_iterations)}"/>')


def terrain_geometry(terrain: Terrain, opt: SimOptions) -> dict:
    """How a terrain becomes MuJoCo geometry: a plane, or a height field with normalised data (row 0 = y0)."""
    if terrain.is_flat and opt.flat_as_plane:
        return {"type": "plane"}
    x0, x1, y0, y1 = opt.course_extent
    Z, ext = terrain.heightfield((x0, x1), (y0, y1), opt.heightfield_cell)
    zmin, zmax = float(Z.min()), float(Z.max())
    span = zmax - zmin
    data = (Z - zmin) / span if span > 0 else np.zeros_like(Z)
    size = ((x1 - x0) / 2, (y1 - y0) / 2, span if span > 0 else 1e-6, opt.heightfield_base)
    return {"type": "hfield", "nrow": Z.shape[0], "ncol": Z.shape[1], "size": size, "data": data,
            "pos": ((x0 + x1) / 2, (y0 + y1) / 2, zmin), "zmin": zmin, "zmax": zmax, "extent": ext}


def _hfield_xml(tinfo: dict, heightfield: str) -> str:
    head = (f'<hfield name="{TERRAIN_NAME}" nrow="{tinfo["nrow"]}" ncol="{tinfo["ncol"]}" '
            f'size="{_f(tinfo["size"])}"')
    if heightfield == "deferred":
        return head + "/>"
    if heightfield != "inline":
        raise ValueError("heightfield must be 'inline' or 'deferred'")
    # MJCF's elevation attribute is image-ordered: its first row is y max.
    rows = tinfo["data"][::-1]
    body = " ".join(f"{v:.6g}" for v in rows.ravel())
    return head + f' elevation="{body}"/>'


def _terrain_geom_xml(tinfo: dict, opt: SimOptions) -> str:
    common = (f'name="{TERRAIN_NAME}" contype="{TERRAIN_BIT}" conaffinity="{ROBOT_BIT}" priority="0" '
              f'rgba="{_f(opt.terrain_rgba)}"')
    if tinfo["type"] == "plane":
        return f'<geom {common} type="plane" size="0 0 1"/>'
    return f'<geom {common} type="hfield" hfield="{TERRAIN_NAME}" pos="{_f(tinfo["pos"])}"/>'


def _geom_xml(g: Geom, opt: SimOptions, coll) -> str:
    a = [f"name={quoteattr(g.name)}", f'type="{g.type}"', f'size="{_f(g.size)}"']
    if g.fromto is not None:
        a.append(f'fromto="{_f(g.fromto)}"')
    else:
        a.append(f'pos="{_f(g.pos)}" quat="{_f(g.quat)}"')
    a.append(f'mass="{(g.mass or 0.0):.10g}"')
    if g.role == "visual":
        a.append('contype="0" conaffinity="0" group="1"')
    else:
        a.append(f'contype="{coll[0]}" conaffinity="{coll[1]}" priority="1" condim="{opt.condim}" '
                 f'friction="{_f(g.friction)}"')
        if opt.contact_solref is not None:
            a.append(f'solref="{_f(opt.contact_solref)}"')
        if opt.contact_solimp is not None:
            a.append(f'solimp="{_f(opt.contact_solimp)}"')
    if g.rgba is not None:
        a.append(f'rgba="{_f(g.rgba)}"')
    return "<geom " + " ".join(a) + "/>"


def _joint_xml(j: Joint) -> str:
    arm = j.armature + (j.servo.armature if j.servo is not None else 0.0)
    a = [f"name={quoteattr(j.name)}", f'type="{j.kind}"', f'axis="{_f(j.axis)}"', f'pos="{_f(j.pos)}"']
    if j.range is not None:
        a.append(f'range="{_f(j.range)}" limited="true"')
    else:
        a.append('limited="false"')
    for key, val in (("stiffness", j.stiffness), ("damping", j.damping), ("springref", j.springref),
                     ("armature", arm), ("frictionloss", j.frictionloss)):
        if val:
            a.append(f'{key}="{val:.10g}"')
    return "<joint " + " ".join(a) + "/>"


def _link_xml(link: Link, opt: SimOptions, coll, indent: int, is_root: bool = False) -> list:
    pad = " " * indent
    lines = [f'{pad}<body name={quoteattr(link.name)} pos="{_f(link.pos)}" quat="{_f(link.quat)}">']
    if is_root:
        lines.append(f'{pad}  <freejoint name={quoteattr(link.name + ":free")}/>')
    for j in link.joints:
        lines.append(pad + "  " + _joint_xml(j))
    for g in link.geoms:
        lines.append(pad + "  " + _geom_xml(g, opt, coll))
    for p in link.masses:
        lines.append(f'{pad}  <geom name={quoteattr("pm:" + p.name)} type="sphere" size="{POINT_MASS_RADIUS}" '
                     f'pos="{_f(p.pos)}" mass="{p.mass:.10g}" contype="0" conaffinity="0" group="4" '
                     f'rgba="0 0 0 0"/>')
    for c in link.children:
        lines += _link_xml(c, opt, coll, indent + 2)
    lines.append(f"{pad}</body>")
    return lines

