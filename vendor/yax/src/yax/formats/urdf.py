"""URDF (ROS). Plain XML: links with inertial, visual and collision; joints with origin, axis and limits. A URDF
child link's frame is its joint's frame, so the writer re-expresses a link whose own frame sits elsewhere."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .. import hub, stl
from ..hub import Joint, Link, Machine, Report, Shape


def _floats(text, n=None, default=0.0):
    v = [float(x) for x in (text or "").split()] if text else []
    return v if n is None else (v + [default] * n)[:n]


def _origin(el) -> tuple:
    o = el.find("origin") if el is not None else None
    if o is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    return tuple(_floats(o.get("xyz"), 3)), hub.rpy_to_quat(*_floats(o.get("rpy"), 3))


def _resolve(uri: str, base: Path) -> Path:
    for prefix in ("package://", "file://", "model://"):
        if uri.startswith(prefix):
            uri = uri[len(prefix):]
            if prefix == "package://":                                         # package://pkg/rest: try beside the file
                rest = uri.split("/", 1)[1] if "/" in uri else uri
                for cand in (base / rest, base / uri, base.parent / uri):
                    if cand.exists():
                        return cand
    p = Path(uri)
    return p if p.is_absolute() else base / p


def _geometry(g, base: Path, rep: Report, label: str) -> dict | None:
    if g is None:
        return None
    for child in g:
        t = child.tag
        if t == "box":
            return {"kind": "box", "size": tuple(_floats(child.get("size"), 3))}
        if t == "sphere":
            return {"kind": "sphere", "size": (float(child.get("radius")),)}
        if t == "cylinder":
            return {"kind": "cylinder", "size": (float(child.get("radius")), float(child.get("length")))}
        if t == "capsule":                                                      # the ROS 2 extension
            return {"kind": "capsule", "size": (float(child.get("radius")), float(child.get("length")))}
        if t == "mesh":
            uri = child.get("filename")
            scale = np.asarray(_floats(child.get("scale"), 3, 1.0), float) if child.get("scale") else np.ones(3)
            path = _resolve(uri, base)
            if path.suffix.lower() == ".stl" and path.exists():
                tris = np.asarray(stl.read(path), float) * scale
                pts = tris.reshape(-1, 3)
                return {"kind": "mesh", "points": pts, "faces": np.arange(len(pts)).reshape(-1, 3), "uri": uri}
            rep.drop(f"{label}: mesh {uri} (only STL meshes are read; kept as its URI)")
            return {"kind": "mesh", "points": np.zeros((0, 3)), "faces": np.zeros((0, 3), int), "uri": uri}
    return None


def read(path) -> tuple[Machine, Report]:
    path = Path(path)
    root = ET.parse(path).getroot()
    rep = Report(str(path), "usd")
    materials = {m.get("name"): _floats(m.find("color").get("rgba"), 4, 1.0) for m in root.findall("material")
                 if m.find("color") is not None}
    links = {}
    for el in root.findall("link"):
        lk = Link(name=el.get("name"))
        ine = el.find("inertial")
        if ine is not None:
            p, q = _origin(ine)
            mass = ine.find("mass")
            i = ine.find("inertia")
            lk.mass = float(mass.get("value")) if mass is not None else 0.0
            lk.com = p
            if i is not None:
                g = {k: float(i.get(k, 0.0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
                I = np.array([[g["ixx"], g["ixy"], g["ixz"]], [g["ixy"], g["iyy"], g["iyz"]], [g["ixz"], g["iyz"], g["izz"]]])
                Rq = hub.R(q)
                lk.inertia = Rq @ I @ Rq.T
        for role in ("collision", "visual"):
            for k, s in enumerate(el.findall(role)):
                geo = _geometry(s.find("geometry"), path.parent, rep, f"link {lk.name} {role} {k}")
                if geo is None:
                    continue
                p, q = _origin(s)
                rgba = None
                mat = s.find("material")
                if mat is not None:
                    c = mat.find("color")
                    rgba = tuple(_floats(c.get("rgba"), 4, 1.0)) if c is not None else tuple(materials.get(mat.get("name"), ())) or None
                lk.shapes.append(Shape(name=s.get("name") or f"{role}{k}", pos=p, quat=q, role=role, rgba=rgba, **geo))
        links[lk.name] = lk
    joints, parent_of = [], {}
    for el in root.findall("joint"):
        t = el.get("type")
        parent, child = el.find("parent").get("link"), el.find("child").get("link")
        p, q = _origin(el)
        axis = tuple(_floats(el.find("axis").get("xyz"), 3)) if el.find("axis") is not None else (1.0, 0.0, 0.0)
        lim = el.find("limit")
        kind = {"revolute": "revolute", "continuous": "continuous", "prismatic": "prismatic", "fixed": "fixed",
                "floating": None, "planar": None}[t]
        if kind is None:
            if t == "planar":
                rep.drop(f"joint {el.get('name')}: planar (no USD/MJCF equivalent; the child floats)")
            continue
        j = Joint(name=el.get("name"), kind=kind, parent=parent, child=child, pos=p, axis=axis)
        j.data["_frame"] = (p, q)                                               # in the parent's frame, resolved below
        if lim is not None and kind in ("revolute", "prismatic"):
            j.lower, j.upper = float(lim.get("lower", 0.0)), float(lim.get("upper", 0.0))
        if lim is not None:
            for k in ("effort", "velocity"):
                if lim.get(k) is not None:
                    j.data[k] = float(lim.get(k))
        dyn = el.find("dynamics")
        if dyn is not None and dyn.get("damping"):
            j.data["damping"] = float(dyn.get("damping"))
        joints.append(j)
        parent_of[child] = j
    # poses: the root(s) at the origin, every child frame = its joint frame
    placed = {}

    def place(name):
        if name in placed:
            return placed[name]
        j = parent_of.get(name)
        if j is None or j.parent is None or j.parent == "world":
            pose = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
            if j is not None:
                pose = hub.from_frame(*j.data["_frame"], (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        else:
            pp, pq = place(j.parent)
            pose = hub.from_frame(*j.data["_frame"], pp, pq)
        placed[name] = pose
        return pose

    m = Machine(name=root.get("name") or path.stem)
    world = "world" in links and not links["world"].shapes and not links["world"].mass
    for name, lk in links.items():
        if world and name == "world":
            continue
        lk.pos, lk.quat = place(name)
        m.links.append(lk)
    for j in joints:
        cp, cq = place(j.child)
        frame = j.data.pop("_frame")
        j.pos = cp
        j.axis = tuple(hub.R(cq) @ np.asarray(j.axis, float))
        if j.parent == "world" and world:
            j.parent = None
        m.joints.append(j)
    return m, rep


def _f(v) -> str:
    return " ".join(f"{float(x):.17g}" for x in np.ravel(v))


def write(m: Machine, path, rep: Report | None = None) -> Report:
    """Write ``m`` as URDF at ``path`` (meshes, ellipsoids and — unless ``capsule`` is supported — capsules as STL in
    ``<stem>_meshes/``)."""
    path = Path(path)
    rep = rep or Report("usd", str(path))
    mesh_dir = path.parent / f"{path.stem}_meshes"
    hub.report_data(m, rep, keep=())
    frames = {}
    for lk in m.links:                                                          # URDF: a child frame is its joint frame
        j = m.parent_joint(lk.name)
        frames[lk.name] = (tuple(j.pos), lk.quat) if j is not None else (lk.pos, lk.quat)
    out = [f'<robot name={quoteattr(m.name)}>']
    if any(j.parent is None for j in m.joints):
        out.append('  <link name="world"/>')
    for lk in m.links:
        fp, fq = frames[lk.name]
        out.append(f"  <link name={quoteattr(lk.name)}>")
        if lk.mass:
            cp, _ = hub.to_frame(hub.from_frame(lk.com, (1, 0, 0, 0), lk.pos, lk.quat)[0], lk.quat, fp, fq)
            I = np.asarray(lk.inertia if lk.inertia is not None else np.zeros((3, 3)), float)
            out += [f"    <inertial>", f'      <origin xyz="{_f(cp)}" rpy="0 0 0"/>', f'      <mass value="{lk.mass:.17g}"/>',
                    f'      <inertia ixx="{I[0, 0]:.17g}" ixy="{I[0, 1]:.17g}" ixz="{I[0, 2]:.17g}" iyy="{I[1, 1]:.17g}" '
                    f'iyz="{I[1, 2]:.17g}" izz="{I[2, 2]:.17g}"/>', "    </inertial>"]
        for s in lk.shapes:
            sp, sq = hub.to_frame(*hub.from_frame(s.pos, s.quat, lk.pos, lk.quat), fp, fq)
            if s.kind in ("cylinder", "capsule") and s.axis != "z":
                sq = hub.qnorm(hub.qmul(sq, hub.quat_from_R(hub._AXIS_FROM_Z[s.axis])))
            if s.kind == "box":
                geo = f'<box size="{_f(s.size)}"/>'
            elif s.kind == "sphere":
                geo = f'<sphere radius="{s.size[0]:.17g}"/>'
            elif s.kind == "cylinder":
                geo = f'<cylinder radius="{s.size[0]:.17g}" length="{s.size[1]:.17g}"/>'
            else:                                                               # mesh, ellipsoid, capsule → STL
                if s.kind != "mesh":
                    rep.drop(f"{lk.name}/{s.name}: a {s.kind} is not a URDF primitive — written as an STL mesh")
                pts, faces = hub.tessellate(Shape(s.name, s.kind, s.size, axis=s.axis, points=s.points, faces=s.faces))
                mesh_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{hub._ident(lk.name)}__{hub._ident(s.name)}.stl"
                stl.write(mesh_dir / fname, [tuple(map(tuple, pts[f])) for f in faces])
                geo = f'<mesh filename={quoteattr(f"{mesh_dir.name}/{fname}")}/>'
                if s.kind in ("cylinder", "capsule") and s.axis != "z":          # the STL already carries the axis
                    sq = hub.to_frame(*hub.from_frame(s.pos, s.quat, lk.pos, lk.quat), fp, fq)[1]
            rpy = hub.quat_to_rpy(sq)
            out.append(f"    <{s.role} name={quoteattr(s.name)}>")
            out.append(f'      <origin xyz="{_f(sp)}" rpy="{_f(rpy)}"/>')
            out.append(f"      <geometry>{geo}</geometry>")
            if s.rgba is not None and s.role == "visual":
                rgba = tuple(s.rgba) + ((1.0,) if len(s.rgba) == 3 else ())
                out.append(f'      <material name={quoteattr(lk.name + "_" + s.name)}><color rgba="{_f(rgba)}"/></material>')
            out.append(f"    </{s.role}>")
        for name, kg, pos in lk.point_masses:
            rep.drop(f"point mass {name} as a separate element (its mass is in {lk.name}'s inertial)")
        out.append("  </link>")
    for j in m.joints:
        child_frame = frames[j.child]
        parent_frame = frames[j.parent] if j.parent is not None else ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        op, oq = hub.to_frame(*child_frame, *parent_frame)
        t = {"revolute": "revolute", "continuous": "continuous", "prismatic": "prismatic", "fixed": "fixed",
             "spherical": None}[j.kind]
        if t is None:
            rep.drop(f"joint {j.name}: spherical (no URDF joint) — written as fixed")
            t = "fixed"
        out.append(f"  <joint name={quoteattr(j.name)} type=\"{t}\">")
        out.append(f'    <parent link={quoteattr(j.parent or "world")}/>')
        out.append(f"    <child link={quoteattr(j.child)}/>")
        out.append(f'    <origin xyz="{_f(op)}" rpy="{_f(hub.quat_to_rpy(oq))}"/>')
        if t in ("revolute", "continuous", "prismatic"):
            out.append(f'    <axis xyz="{_f(hub.R(child_frame[1]).T @ np.asarray(j.axis, float))}"/>')
        if t in ("revolute", "prismatic"):
            lo, hi = (j.lower, j.upper) if j.lower is not None else (-math.pi, math.pi)
            eff = j.data.get("effort", (j.drive or {}).get("max_force", 0.0))
            out.append(f'    <limit lower="{lo:.17g}" upper="{hi:.17g}" effort="{eff:.17g}" velocity="{j.data.get("velocity", 0.0):.17g}"/>')
        if "damping" in j.data:
            out.append(f'    <dynamics damping="{j.data["damping"]:.17g}"/>')
        if j.drive:
            rep.drop(f"joint {j.name}: its drive's stiffness/damping (URDF has no actuators; max force kept as the effort limit)")
        out.append("  </joint>")
    roots = [lk.name for lk in m.roots() if m.parent_joint(lk.name) is None]
    for r in roots:
        lk = m.link(r)
        if np.linalg.norm(lk.pos) > 1e-12 or abs(abs(lk.quat[0]) - 1) > 1e-12:
            rep.drop(f"root link {r}'s placement in the world (URDF puts the root at the origin; the robot is unchanged)")
    if len(roots) > 1:
        rep.drop(f"several root links {roots}: URDF needs one tree (written as is)")
    out.append("</robot>")
    path.write_text("\n".join(out) + "\n")
    return rep
