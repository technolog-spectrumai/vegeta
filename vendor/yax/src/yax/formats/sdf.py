"""SDF (Gazebo). Plain XML; written as SDF 1.9 (capsules, ellipsoids, ball joints). Links are posed in the model
frame, joints relative to their child link; axes in the joint frame (SDF ≥ 1.7; older files: the model frame)."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .. import hub, propulsion, stl
from ..hub import Joint, Link, Machine, Report, Shape
from .urdf import _resolve


def _vals(el, tag, n, default=0.0):
    t = el.find(tag) if el is not None else None
    v = [float(x) for x in t.text.split()] if t is not None and t.text else []
    return (v + [default] * n)[:n]


def _pose(el) -> tuple:
    p = el.find("pose") if el is not None else None
    if p is None or not (p.text or "").strip():
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), None
    v = ([float(x) for x in p.text.split()] + [0.0] * 6)[:6]
    if p.get("degrees") == "true":
        v[3:] = [math.radians(a) for a in v[3:]]
    return tuple(v[:3]), hub.rpy_to_quat(*v[3:]), p.get("relative_to")


def _geometry(g, base: Path, rep: Report, label: str) -> dict | None:
    if g is None:
        return None
    for c in g:
        if c.tag == "box":
            return {"kind": "box", "size": tuple(_vals(c, "size", 3))}
        if c.tag == "sphere":
            return {"kind": "sphere", "size": (_vals(c, "radius", 1)[0],)}
        if c.tag == "cylinder":
            return {"kind": "cylinder", "size": (_vals(c, "radius", 1)[0], _vals(c, "length", 1)[0])}
        if c.tag == "capsule":
            return {"kind": "capsule", "size": (_vals(c, "radius", 1)[0], _vals(c, "length", 1)[0])}
        if c.tag == "ellipsoid":
            return {"kind": "ellipsoid", "size": tuple(_vals(c, "radii", 3))}
        if c.tag == "mesh":
            uri = c.find("uri").text.strip()
            scale = np.asarray(_vals(c, "scale", 3, 1.0)) if c.find("scale") is not None else np.ones(3)
            path = _resolve(uri, base)
            if path.suffix.lower() == ".stl" and path.exists():
                pts = (np.asarray(stl.read(path), float) * scale).reshape(-1, 3)
                return {"kind": "mesh", "points": pts, "faces": np.arange(len(pts)).reshape(-1, 3), "uri": uri}
            rep.drop(f"{label}: mesh {uri} (only STL meshes are read; kept as its URI)")
            return {"kind": "mesh", "points": np.zeros((0, 3)), "faces": np.zeros((0, 3), int), "uri": uri}
        rep.drop(f"{label}: geometry {c.tag}")
    return None


def read(path) -> tuple[Machine, Report]:
    path = Path(path)
    root = ET.parse(path).getroot()
    rep = Report(str(path), "usd")
    version = tuple(int(x) for x in (root.get("version") or "1.9").split(".")[:2])
    model = root.find("model")
    if model is None:
        model = root.find("world/model")
        rep.drop("the world around the model (only the first model is read)")
    if model is None:
        raise ValueError(f"{path}: no <model>")
    if model.find("model") is not None or model.find("include") is not None:
        rep.drop("nested models and includes (not read)")
    mp, mq, _ = _pose(model)
    m = Machine(name=model.get("name") or path.stem)
    raw = {}
    for el in model.findall("link"):
        raw[el.get("name")] = el
    poses = {}

    def link_pose(name, depth=0):
        if name in poses:
            return poses[name]
        p, q, rel = _pose(raw[name])
        if rel and rel not in ("__model__", name) and rel in raw and depth < 50:
            rp, rq = link_pose(rel, depth + 1)
        else:
            if rel and rel not in ("__model__",):
                rep.drop(f"link {name}: pose relative_to {rel!r} (read as the model frame)")
            rp, rq = mp, mq
        poses[name] = hub.from_frame(p, q, rp, rq)
        return poses[name]

    for name, el in raw.items():
        lk = Link(name=name)
        kept = {t: [ET.tostring(c, encoding="unicode").strip() for c in el.findall(t)] for t in ("sensor", "plugin")}
        kept = {t: v for t, v in kept.items() if v}
        if kept:
            lk.data["sdf"] = kept
        lk.pos, lk.quat = link_pose(name)
        ine = el.find("inertial")
        if ine is not None:
            p, q, _ = _pose(ine)
            lk.mass = _vals(ine, "mass", 1)[0]
            i = ine.find("inertia")
            if i is not None:
                g = {k: _vals(i, k, 1)[0] for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
                I = np.array([[g["ixx"], g["ixy"], g["ixz"]], [g["ixy"], g["iyy"], g["iyz"]], [g["ixz"], g["iyz"], g["izz"]]])
                Rq = hub.R(q)
                lk.inertia = Rq @ I @ Rq.T
            lk.com = p
        for role in ("collision", "visual"):
            for k, s in enumerate(el.findall(role)):
                geo = _geometry(s.find("geometry"), path.parent, rep, f"link {name} {role} {s.get('name')}")
                if geo is None:
                    continue
                p, q, _ = _pose(s)
                rgba = None
                mat = s.find("material")
                if mat is not None and mat.find("diffuse") is not None:
                    rgba = tuple(_vals(mat, "diffuse", 4, 1.0))
                lk.shapes.append(Shape(name=s.get("name") or f"{role}{k}", pos=p, quat=q, role=role, rgba=rgba, **geo))
        m.links.append(lk)
    for el in model.findall("joint"):
        t = el.get("type")
        kind = {"revolute": "revolute", "continuous": "continuous", "prismatic": "prismatic", "fixed": "fixed",
                "ball": "spherical"}.get(t)
        parent, child = el.find("parent").text.strip(), el.find("child").text.strip()
        if kind is None:
            rep.drop(f"joint {el.get('name')}: type {t}")
            continue
        cp, cq = poses[child]
        p, q, rel = _pose(el)
        jp, jq = hub.from_frame(p, q, cp, cq)                                     # relative to the child by default
        ax = el.find("axis")
        axis = (1.0, 0.0, 0.0)
        if ax is not None:
            v = np.asarray(_vals(ax, "xyz", 3), float)
            x = ax.find("xyz")
            exp = x.get("expressed_in") if x is not None else None
            if version < (1, 7) or exp == "__model__":
                axis = tuple(hub.R(mq) @ v)
            elif exp and exp in poses:
                axis = tuple(hub.R(poses[exp][1]) @ v)
            else:
                axis = tuple(hub.R(jq) @ v)
        j = Joint(name=el.get("name"), kind=kind, parent=None if parent == "world" else parent, child=child, pos=jp, axis=axis)
        lim = ax.find("limit") if ax is not None else None
        if lim is not None and kind in ("revolute", "prismatic"):
            lo, hi = _vals(lim, "lower", 1, -1e16)[0], _vals(lim, "upper", 1, 1e16)[0]
            if abs(lo) < 1e15 and abs(hi) < 1e15:
                j.lower, j.upper = lo, hi
            for k in ("effort", "velocity"):
                if lim.find(k) is not None:
                    j.data[k] = _vals(lim, k, 1)[0]
        dyn = ax.find("dynamics") if ax is not None else None
        if dyn is not None and dyn.find("damping") is not None:
            j.data["damping"] = _vals(dyn, "damping", 1)[0]
        m.joints.append(j)
    known = {"link", "joint", "pose", "static", "plugin"}
    plugins = [ET.tostring(c, encoding="unicode").strip() for c in model.findall("plugin")]
    extra = [ET.tostring(c, encoding="unicode").strip() for c in model if c.tag not in known]
    if plugins or extra:
        m.data["sdf"] = {k: v for k, v in (("plugins", plugins), ("extra", extra)) if v}
    if model.find("static") is not None and (model.find("static").text or "").strip() == "true":
        for lk in m.roots():
            if m.parent_joint(lk.name) is None:
                m.joints.append(Joint(name=f"{lk.name}_static", kind="fixed", parent=None, child=lk.name, pos=lk.pos))
    return m, rep


def _f(v) -> str:
    return " ".join(f"{float(x):.17g}" for x in np.ravel(v))


def _pose_xml(p, q, pad) -> str:
    return f"{pad}<pose>{_f(p)} {_f(hub.quat_to_rpy(q))}</pose>"


def write(m: Machine, path, rep: Report | None = None) -> Report:
    """Write ``m`` as SDF 1.9 at ``path`` (meshes as STL in ``<stem>_meshes/``)."""
    path = Path(path)
    rep = rep or Report("usd", str(path))
    mesh_dir = path.parent / f"{path.stem}_meshes"
    hub.report_data(m, rep, keep=("sdf", propulsion.KEY))
    propulsion.report_reduction(m, rep, "SDF")
    out = ['<?xml version="1.0"?>', '<sdf version="1.9">', f"  <model name={quoteattr(m.name)}>"]
    for lk in m.links:
        out.append(f"    <link name={quoteattr(lk.name)}>")
        out.append(_pose_xml(lk.pos, lk.quat, "      "))
        if lk.mass:
            out += ["      <inertial>", _pose_xml(lk.com, (1, 0, 0, 0), "        "), f"        <mass>{lk.mass:.17g}</mass>"]
            if lk.inertia is None:                                             # none in the source: not invented
                rep.drop(f"link {lk.name}: no inertia in the source — none written (SDF's own default applies)")
            else:
                I = np.asarray(lk.inertia, float)
                out.append("        <inertia>" + "".join(f"<{k}>{v:.17g}</{k}>" for k, v in (("ixx", I[0, 0]), ("ixy", I[0, 1]), ("ixz", I[0, 2]),
                                                                                       ("iyy", I[1, 1]), ("iyz", I[1, 2]), ("izz", I[2, 2]))) + "</inertia>")
            out.append("      </inertial>")
        for s in lk.shapes:
            q = s.quat
            if s.kind in ("cylinder", "capsule") and s.axis != "z":
                q = hub.qnorm(hub.qmul(s.quat, hub.quat_from_R(hub._AXIS_FROM_Z[s.axis])))
            if s.kind == "box":
                geo = f"<box><size>{_f(s.size)}</size></box>"
            elif s.kind == "sphere":
                geo = f"<sphere><radius>{s.size[0]:.17g}</radius></sphere>"
            elif s.kind in ("cylinder", "capsule"):
                geo = f"<{s.kind}><radius>{s.size[0]:.17g}</radius><length>{s.size[1]:.17g}</length></{s.kind}>"
            elif s.kind == "ellipsoid":
                geo = f"<ellipsoid><radii>{_f(s.size)}</radii></ellipsoid>"
            else:
                mesh_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{hub._ident(lk.name)}__{hub._ident(s.name)}.stl"
                pts, faces = np.asarray(s.points, float), np.asarray(s.faces, int)
                stl.write(mesh_dir / fname, [tuple(map(tuple, pts[f])) for f in faces])
                geo = f"<mesh><uri>{mesh_dir.name}/{fname}</uri></mesh>"
            out.append(f"      <{s.role} name={quoteattr(s.name)}>")
            out.append(_pose_xml(s.pos, q, "        "))
            out.append(f"        <geometry>{geo}</geometry>")
            if s.rgba is not None and s.role == "visual":
                rgba = tuple(s.rgba) + ((1.0,) if len(s.rgba) == 3 else ())
                out.append(f"        <material><diffuse>{_f(rgba)}</diffuse><ambient>{_f(rgba)}</ambient></material>")
            out.append(f"      </{s.role}>")
        for name, kg, pos in lk.point_masses:
            rep.drop(f"point mass {name} as a separate element (its mass is in {lk.name}'s inertial)")
        for t in ("sensor", "plugin"):
            for raw_el in lk.data.get("sdf", {}).get(t, []):
                out += ["      " + line for line in raw_el.splitlines()]
        out.append("    </link>")
    for j in m.joints:
        child = m.link(j.child)
        t = {"revolute": "revolute", "continuous": "continuous", "prismatic": "prismatic", "fixed": "fixed", "spherical": "ball"}[j.kind]
        jp = hub.R(child.quat).T @ (np.asarray(j.pos, float) - np.asarray(child.pos, float))
        out.append(f"    <joint name={quoteattr(j.name)} type=\"{t}\">")
        out.append(f"      <parent>{j.parent or 'world'}</parent>")
        out.append(f"      <child>{j.child}</child>")
        out.append(_pose_xml(jp, (1, 0, 0, 0), "      "))
        if t in ("revolute", "continuous", "prismatic"):
            out.append("      <axis>")
            out.append(f"        <xyz>{_f(hub.R(child.quat).T @ np.asarray(j.axis, float))}</xyz>")
            if t != "continuous" and j.lower is not None:
                lim = f"<lower>{j.lower:.17g}</lower><upper>{j.upper:.17g}</upper>"
                for k in ("effort", "velocity"):
                    if k in j.data:
                        lim += f"<{k}>{j.data[k]:.17g}</{k}>"
                out.append(f"        <limit>{lim}</limit>")
            if "damping" in j.data:
                out.append(f"        <dynamics><damping>{j.data['damping']:.17g}</damping></dynamics>")
            out.append("      </axis>")
        if j.drive:
            rep.drop(f"joint {j.name}: its drive (SDF has no actuator element; a Gazebo controller plugin carries it)")
        out.append("    </joint>")
    for raw_el in m.data.get("sdf", {}).get("extra", []) + m.data.get("sdf", {}).get("plugins", []):
        out += ["    " + line for line in raw_el.splitlines()]
    out += ["  </model>", "</sdf>"]
    path.write_text("\n".join(out) + "\n")
    return rep
