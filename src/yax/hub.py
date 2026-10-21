"""The hub: a machine as plain data (``Machine``, ``Link``, ``Shape``, ``Joint``) and its USD form (docs/converter.md).

Readers build a ``Machine`` and turn it into a USD stage (``to_stage``); writers read a stage back (``from_stage``)
and write their format — every conversion passes through USD. Frames: SI units, Z up; poses are in the machine frame
at q = 0 (``pos`` [m], ``quat`` (w, x, y, z)); a link's centre of mass and inertia are in its own frame, shapes are
placed in their link's frame. USD's mass and joint-frame attributes are single precision, so Yax also writes double
precision copies (``yax:*``) and reads them first: a round trip through USD is exact.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

KINDS = ("box", "sphere", "ellipsoid", "cylinder", "capsule", "mesh")
JOINTS = ("revolute", "continuous", "prismatic", "fixed", "spherical")


# --------------------------------------------------------------------------------------------------------- the data
@dataclass
class Shape:
    """A shape on a link. ``size`` by kind: box (sx, sy, sz) full lengths; sphere (r,); ellipsoid (a, b, c);
    cylinder and capsule (r, length) with ``axis`` (the capsule's length is its cylinder part); mesh: ``points``
    (N×3, link frame after ``pos``/``quat``) and ``faces`` (M×3 indices), ``uri`` where it came from."""
    name: str
    kind: str
    size: tuple = ()
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    axis: str = "z"
    role: str = "collision"                     # or "visual"
    rgba: tuple | None = None
    points: object = None
    faces: object = None
    uri: str | None = None


@dataclass
class Link:
    """A rigid body: pose in the machine frame at q = 0, total mass with its centre of mass and inertia (about the
    centre of mass, link axes, kg m²), shapes, point masses (``(name, kg, pos)``, informative: already in ``mass``)."""
    name: str
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    mass: float | None = None
    com: tuple = (0.0, 0.0, 0.0)
    inertia: object = None
    shapes: list = field(default_factory=list)
    point_masses: list = field(default_factory=list)
    data: dict = field(default_factory=dict)


@dataclass
class Joint:
    """``parent`` (None = the world) → ``child``; ``pos`` and ``axis`` in the machine frame at q = 0; limits in rad or
    m (None = unlimited); ``drive`` {stiffness, damping, max_force}; ``data`` e.g. effort and velocity limits."""
    name: str
    kind: str
    parent: str | None
    child: str
    pos: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (1.0, 0.0, 0.0)
    lower: float | None = None
    upper: float | None = None
    drive: dict | None = None
    data: dict = field(default_factory=dict)


@dataclass
class Machine:
    name: str
    links: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    data: dict = field(default_factory=dict)
    producer: dict = field(default_factory=dict)

    def link(self, name: str) -> Link:
        return next(lk for lk in self.links if lk.name == name)

    def parent_joint(self, child: str) -> Joint | None:
        return next((j for j in self.joints if j.child == child), None)

    def roots(self) -> list:
        children = {j.child for j in self.joints if j.parent is not None}
        return [lk for lk in self.links if lk.name not in children]

    def total_mass(self) -> float:
        return float(sum(lk.mass or 0.0 for lk in self.links))


@dataclass
class Report:
    """What a conversion could not carry into the target format."""
    source: str
    target: str
    dropped: list = field(default_factory=list)

    def drop(self, what: str):
        if what not in self.dropped:
            self.dropped.append(what)

    def __str__(self):
        head = f"{self.source} → {self.target}"
        return head + (": nothing dropped" if not self.dropped else ":\n  dropped: " + "\n  dropped: ".join(self.dropped))


def report_data(m: Machine, rep: Report, keep=()):
    """Report the format-specific data a writer cannot carry (``keep``: the keys it writes back)."""
    for k, v in m.data.items():
        if k not in keep:
            what = ", ".join(f"{len(x)} {n}" for n, x in v.items() if isinstance(x, list)) if isinstance(v, dict) else ""
            rep.drop(f"machine data {k!r}{f' ({what})' if what else ''}: the target format has no place for it")
    for lk in m.links:
        for k, v in lk.data.items():
            if k not in keep and k != "free_joint":
                what = ", ".join(f"{len(x)} {n}" for n, x in v.items()) if isinstance(v, dict) else k
                rep.drop(f"link {lk.name}: {k} data ({what})")


# --------------------------------------------------------------------------------------------------------- the maths
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def qnorm(q):
    n = math.sqrt(sum(c * c for c in q))
    q = tuple(c / n for c in q)
    return q if q[0] >= 0 else tuple(-c for c in q)


def R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_from_R(M) -> tuple:
    M = np.asarray(M, float)
    t = np.trace(M)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s)
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        q = ((M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s)
    elif M[1, 1] > M[2, 2]:
        s = math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        q = ((M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        q = ((M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s)
    return qnorm(q)


def rpy_to_quat(r, p, y) -> tuple:
    """URDF/SDF roll-pitch-yaw (fixed axes x, y, z: R = Rz(y) Ry(p) Rx(r))."""
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return qnorm((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy))


def quat_to_rpy(q) -> tuple:
    M = R(q)
    p = math.asin(max(-1.0, min(1.0, -M[2, 0])))
    if abs(M[2, 0]) < 1 - 1e-12:
        return math.atan2(M[2, 1], M[2, 2]), p, math.atan2(M[1, 0], M[0, 0])
    return math.atan2(-M[1, 2], M[1, 1]), p, 0.0


def to_frame(pos, quat, frame_pos, frame_quat):
    """A pose given in the machine frame, expressed in the frame (frame_pos, frame_quat)."""
    Ri = R(frame_quat).T
    return tuple(Ri @ (np.asarray(pos, float) - np.asarray(frame_pos, float))), qnorm(qmul(qconj(frame_quat), quat))


def from_frame(pos, quat, frame_pos, frame_quat):
    """A pose given in the frame (frame_pos, frame_quat), expressed in the machine frame."""
    return tuple(np.asarray(frame_pos, float) + R(frame_quat) @ np.asarray(pos, float)), qnorm(qmul(frame_quat, quat))


def x_to(axis) -> tuple:
    """The rotation taking +X to the unit vector ``axis`` (the joint frame of a USD joint)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x = np.array([1.0, 0.0, 0.0])
    c = float(x @ a)
    if c > 1 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    if c < -1 + 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    v = np.cross(x, a)
    return qnorm((1.0 + c, *v))


def principal(I) -> tuple:
    """(principal moments, principal axes quat) of an inertia tensor — USD's MassAPI form."""
    w, V = np.linalg.eigh(np.asarray(I, float))
    if np.linalg.det(V) < 0:
        V[:, 2] = -V[:, 2]
    return tuple(w), quat_from_R(V)


def compose_inertia(moments, axes_quat) -> np.ndarray:
    Rm = R(axes_quat)
    return Rm @ np.diag(moments) @ Rm.T


def shape_inertia(s: Shape, m: float) -> np.ndarray:
    """A solid shape's own inertia about its centre (MuJoCo's formulas), in the shape's frame."""
    k = {"x": 0, "y": 1, "z": 2}[s.axis]
    if s.kind == "box":
        x, y, z = (v * v for v in s.size)
        return m / 12 * np.diag([y + z, x + z, x + y])
    if s.kind == "sphere":
        return 0.4 * m * s.size[0] ** 2 * np.eye(3)
    if s.kind == "ellipsoid":
        a, b, c = (v * v for v in s.size)
        return m / 5 * np.diag([b + c, a + c, a + b])
    if s.kind == "cylinder":
        r2, h2 = s.size[0] ** 2, s.size[1] ** 2
        d = np.full(3, m * (3 * r2 + h2) / 12)
        d[k] = 0.5 * m * r2
        return np.diag(d)
    if s.kind == "capsule":
        r, h = s.size
        ms = m * 4 * r / (4 * r + 3 * h)
        mc, isph = m - ms, 2 * ms * r * r / 5
        d = np.full(3, mc * (3 * r * r + h * h) / 12 + isph + ms * h * (3 * r + 2 * h) / 8)
        d[k] = mc * r * r / 2 + isph
        return np.diag(d)
    raise ValueError(f"no analytic inertia for {s.kind}")


# ----------------------------------------------------------------------------------------------- Machine ↔ USD stage
def _ident(name: str) -> str:
    from pxr import Tf
    return Tf.MakeValidIdentifier(name) or "_"


def _set_pose(xf, pos, quat, scale=None):
    from pxr import Gf, UsdGeom
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:]))))
    if scale is not None:
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, scale)))


def _attr(prim, name, typ, value):
    prim.CreateAttribute(name, typ, custom=True).Set(value)


def to_stage(m: Machine, path: str | None = None):
    """The machine as a USD stage (in memory, or a new file at ``path``)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    stage = Usd.Stage.CreateNew(str(path)) if path else Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/" + _ident(m.name))
    rp = root.GetPrim()
    stage.SetDefaultPrim(rp)
    rp.SetCustomDataByKey("yax", 2)
    rp.SetCustomDataByKey("name", m.name)
    if m.producer:
        _attr(rp, "yax:producer", Sdf.ValueTypeNames.String, json.dumps(m.producer, sort_keys=True))
    if m.data:
        _attr(rp, "yax:data", Sdf.ValueTypeNames.String, json.dumps(m.data, sort_keys=True))
    UsdPhysics.ArticulationRootAPI.Apply(rp)
    paths = {}
    for lk in m.links:
        p = root.GetPath().AppendChild(_ident(lk.name))
        paths[lk.name] = p
        xf = UsdGeom.Xform.Define(stage, p)
        _set_pose(xf, lk.pos, lk.quat)
        prim = xf.GetPrim()
        _attr(prim, "yax:name", Sdf.ValueTypeNames.String, lk.name)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if lk.mass is not None:
            mapi = UsdPhysics.MassAPI.Apply(prim)
            mapi.CreateMassAttr(float(lk.mass))
            _attr(prim, "yax:mass", Sdf.ValueTypeNames.Double, float(lk.mass))
            mapi.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, lk.com)))
            _attr(prim, "yax:com", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*map(float, lk.com)))
            if lk.inertia is not None:
                mom, ax = principal(lk.inertia)
                mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, mom)))
                mapi.CreatePrincipalAxesAttr(Gf.Quatf(float(ax[0]), Gf.Vec3f(*map(float, ax[1:]))))
                _attr(prim, "yax:inertia", Sdf.ValueTypeNames.DoubleArray, Vt.DoubleArray(list(np.ravel(lk.inertia).astype(float))))
        if lk.data:
            _attr(prim, "yax:data", Sdf.ValueTypeNames.String, json.dumps(lk.data, sort_keys=True))
        used = set()
        for s in lk.shapes:                                                    # visual and collision may share a name
            ident, n = _ident(s.name), 2
            while ident in used:
                ident, n = f"{_ident(s.name)}_{n}", n + 1
            used.add(ident)
            _shape_to_usd(stage, p.AppendChild(ident), s)
        for name, kg, pos in lk.point_masses:
            pm = UsdGeom.Xform.Define(stage, p.AppendChild(_ident("pm_" + name)))
            _set_pose(pm, pos, (1.0, 0.0, 0.0, 0.0))
            _attr(pm.GetPrim(), "yax:kind", Sdf.ValueTypeNames.Token, "point_mass")
            _attr(pm.GetPrim(), "yax:name", Sdf.ValueTypeNames.String, name)
            _attr(pm.GetPrim(), "yax:mass", Sdf.ValueTypeNames.Double, float(kg))
    if m.joints:
        jroot = root.GetPath().AppendChild("joints")
        UsdGeom.Scope.Define(stage, jroot)
        for j in m.joints:
            _joint_to_usd(stage, jroot.AppendChild(_ident(j.name)), j, m, paths)
    if path:
        stage.GetRootLayer().Save()
    return stage


def _shape_to_usd(stage, p, s: Shape):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

    scale = None
    if s.kind == "box":
        g = UsdGeom.Cube.Define(stage, p)
        g.CreateSizeAttr(1.0)
        scale = s.size
    elif s.kind == "sphere":
        g = UsdGeom.Sphere.Define(stage, p)
        g.CreateRadiusAttr(float(s.size[0]))
    elif s.kind == "ellipsoid":
        g = UsdGeom.Sphere.Define(stage, p)
        g.CreateRadiusAttr(1.0)
        scale = s.size
    elif s.kind in ("cylinder", "capsule"):
        g = (UsdGeom.Cylinder if s.kind == "cylinder" else UsdGeom.Capsule).Define(stage, p)
        g.CreateRadiusAttr(float(s.size[0]))
        g.CreateHeightAttr(float(s.size[1]))
        g.CreateAxisAttr(s.axis.upper())
    elif s.kind == "mesh":
        g = UsdGeom.Mesh.Define(stage, p)
        pts = np.asarray(s.points, float)
        f = np.asarray(s.faces, int)
        g.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in pts]))
        g.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
        g.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in f.ravel()]))
        if s.uri:
            _attr(g.GetPrim(), "yax:uri", Sdf.ValueTypeNames.String, s.uri)
    else:
        raise ValueError(f"unknown shape kind {s.kind!r}")
    _set_pose(g, s.pos, s.quat, scale)
    prim = g.GetPrim()
    _attr(prim, "yax:name", Sdf.ValueTypeNames.String, s.name)
    _attr(prim, "yax:role", Sdf.ValueTypeNames.Token, s.role)
    if s.role == "collision":
        UsdPhysics.CollisionAPI.Apply(prim)
    if s.rgba is not None:
        g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, s.rgba[:3]))]))
        if len(s.rgba) > 3:
            g.CreateDisplayOpacityAttr(Vt.FloatArray([float(s.rgba[3])]))


def _joint_to_usd(stage, p, j: Joint, m: Machine, paths):
    from pxr import Gf, Sdf, UsdPhysics

    cls = {"revolute": UsdPhysics.RevoluteJoint, "continuous": UsdPhysics.RevoluteJoint,
           "prismatic": UsdPhysics.PrismaticJoint, "fixed": UsdPhysics.FixedJoint,
           "spherical": UsdPhysics.SphericalJoint}[j.kind]
    uj = cls.Define(stage, p)
    jq = x_to(j.axis)
    for idx, body in ((0, j.parent), (1, j.child)):
        if body is None:
            lp, lq = tuple(j.pos), jq                                      # the world frame = the machine frame
        else:
            lk = m.link(body)
            lp, lq = to_frame(j.pos, jq, lk.pos, lk.quat)
            (uj.CreateBody0Rel if idx == 0 else uj.CreateBody1Rel)().SetTargets([paths[body]])
        (uj.CreateLocalPos0Attr if idx == 0 else uj.CreateLocalPos1Attr)(Gf.Vec3f(*map(float, lp)))
        (uj.CreateLocalRot0Attr if idx == 0 else uj.CreateLocalRot1Attr)(Gf.Quatf(float(lq[0]), Gf.Vec3f(*map(float, lq[1:]))))
    prim = uj.GetPrim()
    _attr(prim, "yax:name", Sdf.ValueTypeNames.String, j.name)
    _attr(prim, "yax:kind", Sdf.ValueTypeNames.Token, j.kind)
    _attr(prim, "yax:pos", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*map(float, j.pos)))
    _attr(prim, "yax:axis", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*map(float, j.axis)))
    if j.kind in ("revolute", "prismatic", "continuous"):
        uj.CreateAxisAttr("X")
        if j.kind != "continuous" and j.lower is not None and j.upper is not None:
            k = 180 / math.pi if j.kind == "revolute" else 1.0
            uj.CreateLowerLimitAttr(float(j.lower * k))
            uj.CreateUpperLimitAttr(float(j.upper * k))
            _attr(prim, "yax:limits", Sdf.ValueTypeNames.Double2, (float(j.lower), float(j.upper)))
    if j.drive:
        d = UsdPhysics.DriveAPI.Apply(prim, "angular" if j.kind in ("revolute", "continuous") else "linear")
        for key, create in (("stiffness", d.CreateStiffnessAttr), ("damping", d.CreateDampingAttr), ("max_force", d.CreateMaxForceAttr)):
            if key in j.drive:
                create(float(j.drive[key]))
        _attr(prim, "yax:drive", Sdf.ValueTypeNames.String, json.dumps(j.drive, sort_keys=True))
    if j.data:
        _attr(prim, "yax:data", Sdf.ValueTypeNames.String, json.dumps(j.data, sort_keys=True))


def _get(prim, name, default=None):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.HasAuthoredValue() else default


def _decompose(M):
    from pxr import Gf
    t = Gf.Transform(M)
    q = t.GetRotation().GetQuat()
    return tuple(t.GetTranslation()), qnorm((q.GetReal(), *q.GetImaginary())), tuple(t.GetScale())


def from_stage(stage) -> Machine:
    """The machine of a USD stage written by Yax (or any stage with UsdPhysics rigid bodies and joints)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    root = stage.GetDefaultPrim() or next(iter(stage.GetPseudoRoot().GetChildren()))
    k = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    m = Machine(name=root.GetCustomDataByKey("name") or root.GetName())
    if _get(root, "yax:producer"):
        m.producer = json.loads(_get(root, "yax:producer"))
    if _get(root, "yax:data"):
        m.data = json.loads(_get(root, "yax:data"))
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    names = {}
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        W = cache.GetLocalToWorldTransform(prim)
        pos, quat, _ = _decompose(W)
        lk = Link(name=_get(prim, "yax:name") or prim.GetName(), pos=tuple(v * k for v in pos), quat=quat)
        names[prim.GetPath()] = lk.name
        if prim.HasAPI(UsdPhysics.MassAPI):
            mapi = UsdPhysics.MassAPI(prim)
            lk.mass = float(_get(prim, "yax:mass", mapi.GetMassAttr().Get()))
            com = _get(prim, "yax:com") or mapi.GetCenterOfMassAttr().Get()
            lk.com = tuple(float(v) * (1 if _get(prim, "yax:com") else k) for v in com)
            Ia = _get(prim, "yax:inertia")
            if Ia is not None:
                lk.inertia = np.asarray(list(Ia), float).reshape(3, 3)
            elif mapi.GetDiagonalInertiaAttr().HasAuthoredValue():
                ax = mapi.GetPrincipalAxesAttr().Get() or Gf.Quatf(1)
                lk.inertia = compose_inertia(tuple(mapi.GetDiagonalInertiaAttr().Get()), (ax.GetReal(), *ax.GetImaginary()))
        if _get(prim, "yax:data"):
            lk.data = json.loads(_get(prim, "yax:data"))
        Winv = W.GetInverse()
        for child in Usd.PrimRange(prim):
            if child == prim:
                continue
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                break
            if _get(child, "yax:kind") == "point_mass":
                p, _, _ = _decompose(cache.GetLocalToWorldTransform(child) * Winv)
                lk.point_masses.append((_get(child, "yax:name") or child.GetName(), float(_get(child, "yax:mass")), tuple(v * k for v in p)))
                continue
            s = _shape_from_usd(child, cache.GetLocalToWorldTransform(child) * Winv, k)
            if s is not None:
                lk.shapes.append(s)
        m.links.append(lk)
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        uj = UsdPhysics.Joint(prim)
        t0, t1 = uj.GetBody0Rel().GetTargets(), uj.GetBody1Rel().GetTargets()
        kind = _get(prim, "yax:kind") or ("revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else "prismatic"
                                         if prim.IsA(UsdPhysics.PrismaticJoint) else "spherical"
                                         if prim.IsA(UsdPhysics.SphericalJoint) else "fixed")
        j = Joint(name=_get(prim, "yax:name") or prim.GetName(), kind=kind, parent=names.get(t0[0]) if t0 else None,
                  child=names.get(t1[0]) if t1 else None)
        if _get(prim, "yax:pos") is not None:
            j.pos, j.axis = tuple(_get(prim, "yax:pos")), tuple(_get(prim, "yax:axis"))
        else:                                                              # a foreign stage: from body1's frame
            body = stage.GetPrimAtPath(t1[0])
            W = cache.GetLocalToWorldTransform(body)
            lp, lq = uj.GetLocalPos1Attr().Get(), uj.GetLocalRot1Attr().Get()
            bp, bq, _ = _decompose(W)
            jp, jq = from_frame(tuple(v * k for v in lp), qnorm((lq.GetReal(), *lq.GetImaginary())), tuple(v * k for v in bp), bq)
            token = (prim.GetAttribute("physics:axis").Get() or "X") if kind != "fixed" else "X"
            j.pos, j.axis = jp, tuple(R(jq) @ np.eye(3)["XYZ".index(token)])
        lim = _get(prim, "yax:limits")
        if lim is not None:
            j.lower, j.upper = float(lim[0]), float(lim[1])
        elif kind in ("revolute", "prismatic"):
            lo, hi = _get(prim, "physics:lowerLimit"), _get(prim, "physics:upperLimit")
            if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi):
                f = math.pi / 180 if kind == "revolute" else k
                j.lower, j.upper = lo * f, hi * f
        if _get(prim, "yax:drive"):
            j.drive = json.loads(_get(prim, "yax:drive"))
        if _get(prim, "yax:data"):
            j.data = json.loads(_get(prim, "yax:data"))
        m.joints.append(j)
    return m


def _shape_from_usd(prim, M, k) -> Shape | None:
    from pxr import UsdGeom, UsdPhysics

    pos, quat, scale = _decompose(M)
    pos = tuple(v * k for v in pos)
    name = _get(prim, "yax:name") or prim.GetName()
    role = _get(prim, "yax:role") or ("collision" if prim.HasAPI(UsdPhysics.CollisionAPI) else "visual")
    rgba = None
    if prim.IsA(UsdGeom.Gprim):
        c = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        if c:
            o = UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Get()
            rgba = (*map(float, c[0]), float(o[0]) if o else 1.0)
    kw = dict(name=name, pos=pos, quat=quat, role=role, rgba=rgba)
    if prim.IsA(UsdGeom.Cube):
        e = UsdGeom.Cube(prim).GetSizeAttr().Get() * k
        return Shape(kind="box", size=tuple(e * s for s in scale), **kw)
    if prim.IsA(UsdGeom.Sphere):
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get() * k
        sc = tuple(r * s for s in scale)
        if max(sc) - min(sc) < 1e-12 * max(sc):
            return Shape(kind="sphere", size=(sc[0],), **kw)
        return Shape(kind="ellipsoid", size=sc, **kw)
    if prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
        g = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(prim)
        axis = (g.GetAxisAttr().Get() or "Z").lower()
        return Shape(kind="cylinder" if prim.IsA(UsdGeom.Cylinder) else "capsule",
                     size=(g.GetRadiusAttr().Get() * k * scale[0], g.GetHeightAttr().Get() * k * scale[0]), axis=axis, **kw)
    if prim.IsA(UsdGeom.Mesh):
        g = UsdGeom.Mesh(prim)
        pts = np.asarray(g.GetPointsAttr().Get(), float) * k * np.asarray(scale)
        counts, idx = list(g.GetFaceVertexCountsAttr().Get()), list(g.GetFaceVertexIndicesAttr().Get())
        faces, at = [], 0
        for c in counts:                                                    # fan-triangulate polygons
            faces += [(idx[at], idx[at + i], idx[at + i + 1]) for i in range(1, c - 1)]
            at += c
        return Shape(kind="mesh", points=pts, faces=np.asarray(faces, int), uri=_get(prim, "yax:uri"), **kw)
    return None


def tessellate(s: Shape, n: int = 24) -> tuple:
    """Triangles of a shape in its link's frame: (points N×3, faces M×3)."""
    if s.kind == "mesh":
        pts, faces = np.asarray(s.points, float), np.asarray(s.faces, int)
    elif s.kind == "box":
        h = np.asarray(s.size, float) / 2
        pts = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * h
        faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                          [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    elif s.kind in ("sphere", "ellipsoid", "capsule"):
        r = s.size[0]
        abc = (r, r, r) if s.kind != "ellipsoid" else s.size
        half = s.size[1] / 2 if s.kind == "capsule" else 0.0
        rings, pts = [], [[0, 0, -abc[2] - half]]
        for i in range(1, n):
            th = math.pi * i / n - math.pi / 2
            z = abc[2] * math.sin(th) + (half if th > 0 else -half if th < 0 else 0)
            ring = [[abc[0] * math.cos(th) * math.cos(2 * math.pi * k / n), abc[1] * math.cos(th) * math.sin(2 * math.pi * k / n), z]
                    for k in range(n)]
            if s.kind == "capsule" and abs(th) < 1e-12:                       # the equator: both ends of the cylinder
                rings.append(len(pts)); pts += [[x, y, -half] for x, y, _ in ring]
                rings.append(len(pts)); pts += [[x, y, half] for x, y, _ in ring]
                continue
            rings.append(len(pts)); pts += ring
        pts.append([0, 0, abc[2] + half])
        faces = [[0, rings[0] + (k + 1) % n, rings[0] + k] for k in range(n)]
        for a, b in zip(rings, rings[1:]):
            for k in range(n):
                k1 = (k + 1) % n
                faces += [[a + k, a + k1, b + k1], [a + k, b + k1, b + k]]
        top = len(pts) - 1
        faces += [[top, rings[-1] + k, rings[-1] + (k + 1) % n] for k in range(n)]
        pts, faces = np.asarray(pts, float), np.asarray(faces, int)
        if s.kind == "capsule":
            pts = pts @ _AXIS_FROM_Z[s.axis].T
    elif s.kind == "cylinder":
        r, h = s.size
        ang = [2 * math.pi * k / n for k in range(n)]
        pts = [[0, 0, -h / 2], [0, 0, h / 2]] + [[r * math.cos(a), r * math.sin(a), -h / 2] for a in ang] + \
              [[r * math.cos(a), r * math.sin(a), h / 2] for a in ang]
        faces = []
        for k in range(n):
            k1 = (k + 1) % n
            b0, b1, t0, t1 = 2 + k, 2 + k1, 2 + n + k, 2 + n + k1
            faces += [[0, b1, b0], [1, t0, t1], [b0, b1, t1], [b0, t1, t0]]
        pts, faces = np.asarray(pts, float) @ _AXIS_FROM_Z[s.axis].T, np.asarray(faces, int)
    else:
        raise ValueError(s.kind)
    return (R(s.quat) @ pts.T).T + np.asarray(s.pos, float), faces


_AXIS_FROM_Z = {"z": np.eye(3), "x": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),
                "y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)}
