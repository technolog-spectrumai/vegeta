"""MJCF (MuJoCo). Read by compiling with MuJoCo (defaults, classes, ``fromto``, meshes and inertia-from-geoms are
resolved by MuJoCo itself); written as plain MJCF with explicit inertials and checked by compiling."""
from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .. import hub, stl
from ..hub import Joint, Link, Machine, Report, Shape

_GEOM = {2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def read(path, *, xml: str | None = None) -> tuple[Machine, Report]:
    import mujoco

    path = Path(path)
    model = mujoco.MjModel.from_xml_string(xml) if xml is not None else mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    rep = Report(str(path), "usd")
    name = model.names[:model.names.index(b"\0")].decode() or path.stem
    mach = Machine(name=name or path.stem, data={"mjcf": {"timestep": float(model.opt.timestep),
                                                          "gravity": [float(v) for v in model.opt.gravity]}})
    nm = lambda kind, i, fallback: mujoco.mj_id2name(model, kind, i) or fallback   # noqa: E731
    bname = {b: nm(mujoco.mjtObj.mjOBJ_BODY, b, f"body{b}") for b in range(1, model.nbody)}
    for b in range(1, model.nbody):
        lk = Link(name=bname[b], pos=tuple(data.xpos[b]), quat=hub.qnorm(tuple(data.xquat[b])), mass=float(model.body_mass[b]),
                  com=tuple(model.body_ipos[b]), inertia=hub.compose_inertia(tuple(model.body_inertia[b]), tuple(model.body_iquat[b])))
        mach.links.append(lk)
    for g in range(model.ngeom):
        b, t = int(model.geom_bodyid[g]), int(model.geom_type[g])
        gname = nm(mujoco.mjtObj.mjOBJ_GEOM, g, f"geom{g}")
        if b == 0 or t not in _GEOM:
            rep.drop(f"geom {gname}: {'a world geom (terrain/scenery)' if b == 0 else 'type ' + str(t)}")
            continue
        sz, kind = model.geom_size[g], _GEOM[t]
        size = {"sphere": (sz[0],), "capsule": (sz[0], 2 * sz[1]), "cylinder": (sz[0], 2 * sz[1]),
                "ellipsoid": tuple(sz[:3]), "box": tuple(2 * sz[:3]), "mesh": ()}[kind]
        s = Shape(name=gname, kind=kind, size=tuple(map(float, size)), pos=tuple(model.geom_pos[g]),
                  quat=hub.qnorm(tuple(model.geom_quat[g])), role="visual" if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0 else "collision",
                  rgba=tuple(map(float, model.geom_rgba[g])))
        if kind == "mesh":
            mid = int(model.geom_dataid[g])
            va, vn, fa, fn = model.mesh_vertadr[mid], model.mesh_vertnum[mid], model.mesh_faceadr[mid], model.mesh_facenum[mid]
            s.points, s.faces = model.mesh_vert[va:va + vn].astype(float), model.mesh_face[fa:fa + fn].astype(int)
            s.uri = nm(mujoco.mjtObj.mjOBJ_MESH, mid, None)
        mach.link(bname[b]).shapes.append(s)
    act = {}
    for a in range(model.nu):
        if model.actuator_trntype[a] == 0:                                     # a joint transmission
            j = int(model.actuator_trnid[a, 0])
            act[j] = {"stiffness": float(model.actuator_gainprm[a, 0]), "damping": float(-model.actuator_biasprm[a, 2])}
            if model.actuator_forcelimited[a]:
                act[j]["max_force"] = float(max(abs(model.actuator_forcerange[a, 0]), abs(model.actuator_forcerange[a, 1])))
    for b in range(1, model.nbody):
        parent = int(model.body_parentid[b])
        pname = bname.get(parent)
        joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b]
        lk = mach.link(bname[b])
        if not joints:
            mach.joints.append(Joint(name=f"{bname[b]}_weld", kind="fixed", parent=pname, child=bname[b], pos=lk.pos))
            continue
        chain = [pname]
        for i, j in enumerate(joints):
            jt = int(model.jnt_type[j])
            jname = nm(mujoco.mjtObj.mjOBJ_JOINT, j, f"joint{j}")
            if jt == 0:                                                         # free: the root floats
                lk.data["free_joint"] = jname
                continue
            child = bname[b] if i == len(joints) - 1 else f"{bname[b]}__{jname}"
            if child != bname[b]:                                               # several dofs on one body: massless links
                mach.links.append(Link(name=child, pos=lk.pos, quat=lk.quat, mass=0.0, inertia=np.zeros((3, 3))))
            Rb = hub.R(lk.quat)
            pos = tuple(np.asarray(lk.pos) + Rb @ model.jnt_pos[j])
            axis = tuple(Rb @ model.jnt_axis[j])
            kind = {1: "spherical", 2: "prismatic", 3: "revolute"}[jt]
            limited = bool(model.jnt_limited[j])
            if kind == "revolute" and not limited:
                kind = "continuous"
            J = Joint(name=jname, kind=kind, parent=chain[-1], child=child, pos=pos, axis=axis,
                      lower=float(model.jnt_range[j, 0]) if limited and kind != "spherical" else None,
                      upper=float(model.jnt_range[j, 1]) if limited and kind != "spherical" else None, drive=act.get(j))
            if model.dof_damping[model.jnt_dofadr[j]]:
                J.data["damping"] = float(model.dof_damping[model.jnt_dofadr[j]])
            if kind == "spherical" and limited:
                rep.drop(f"joint {jname}: a ball joint's cone limit")
            mach.joints.append(J)
            chain.append(child)
    return mach, rep


def _f(v) -> str:
    return " ".join(f"{float(x):.17g}" for x in np.ravel(v))


def write(m: Machine, path, rep: Report | None = None) -> Report:
    """Write ``m`` as MJCF at ``path`` (meshes as STL in ``<stem>_meshes/``); check it compiles."""
    import mujoco

    path = Path(path)
    rep = rep or Report("usd", str(path))
    mesh_dir = path.parent / f"{path.stem}_meshes"
    assets, children = [], {}
    for j in m.joints:
        children.setdefault(j.parent, []).append(j)
    jointed = {j.child for j in m.joints}
    geom_names = set()

    def unique(name, owner):                                                   # MJCF geom names are model-wide
        cand = name if name not in geom_names else f"{owner}_{name}"
        n = 2
        while cand in geom_names:
            cand, n = f"{owner}_{name}_{n}", n + 1
        geom_names.add(cand)
        return cand

    massless = any((lk.mass or 0) <= 0 for lk in m.links if lk.name in jointed and m.parent_joint(lk.name).kind != "fixed")
    hub.report_data(m, rep, keep=("mjcf",))

    def body(lk: Link, parent: Link | None, joint: Joint | None, pad: str) -> list:
        if parent is None:
            pos, quat = lk.pos, lk.quat
        else:
            pos, quat = hub.to_frame(lk.pos, lk.quat, parent.pos, parent.quat)
        out = [f'{pad}<body name={quoteattr(lk.name)} pos="{_f(pos)}" quat="{_f(quat)}">']
        if joint is None:
            out.append(f'{pad}  <freejoint name={quoteattr(lk.data.get("free_joint", lk.name + "_free"))}/>')
        elif joint.kind != "fixed":
            Rl = hub.R(lk.quat).T
            jp = Rl @ (np.asarray(joint.pos) - np.asarray(lk.pos))
            a = [f"name={quoteattr(joint.name)}", f'type="{ {"revolute": "hinge", "continuous": "hinge", "prismatic": "slide", "spherical": "ball"}[joint.kind]}"',
                 f'pos="{_f(jp)}"']
            if joint.kind != "spherical":
                a.append(f'axis="{_f(Rl @ np.asarray(joint.axis, float))}"')
            if joint.lower is not None and joint.upper is not None and joint.kind in ("revolute", "prismatic"):
                a.append(f'limited="true" range="{_f((joint.lower, joint.upper))}"')
            if "damping" in joint.data:
                a.append(f'damping="{joint.data["damping"]:.17g}"')
            out.append(f'{pad}  <joint {" ".join(a)}/>')
        if lk.mass:
            I = np.asarray(lk.inertia if lk.inertia is not None else np.zeros((3, 3)), float)
            out.append(f'{pad}  <inertial pos="{_f(lk.com)}" mass="{lk.mass:.17g}" '
                       f'fullinertia="{_f((I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))}"/>')
        for s in lk.shapes:
            a = [f"name={quoteattr(unique(s.name, lk.name))}"]
            if s.kind == "mesh":
                mesh_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{hub._ident(lk.name)}__{hub._ident(s.name)}.stl"
                pts, faces = np.asarray(s.points, float), np.asarray(s.faces, int)
                stl.write(mesh_dir / fname, [tuple(map(tuple, pts[f])) for f in faces])
                assets.append(f'    <mesh name={quoteattr(fname[:-4])} file={quoteattr(str(Path(mesh_dir.name) / fname))}/>')
                a += ['type="mesh"', f'mesh={quoteattr(fname[:-4])}']
                quat = s.quat
            else:
                typ = s.kind
                sz = (s.size[:1] if typ == "sphere" else s.size if typ == "ellipsoid" else np.asarray(s.size) / 2
                      if typ == "box" else (s.size[0], s.size[1] / 2))
                quat = s.quat
                if typ in ("cylinder", "capsule") and s.axis != "z":
                    quat = hub.qnorm(hub.qmul(s.quat, hub.quat_from_R(hub._AXIS_FROM_Z[s.axis])))
                a += [f'type="{typ}"', f'size="{_f(sz)}"']
            a += [f'pos="{_f(s.pos)}"', f'quat="{_f(quat)}"', 'mass="0"']
            if s.role == "visual":
                a.append('contype="0" conaffinity="0" group="1"')
            if s.rgba is not None:
                a.append(f'rgba="{_f(s.rgba if len(s.rgba) == 4 else (*s.rgba, 1.0))}"')
            out.append(f'{pad}  <geom {" ".join(a)}/>')
        for name, kg, pos in lk.point_masses:
            rep.drop(f"point mass {name} as a separate element (its mass is in {lk.name}'s inertial)")
        for j in children.get(lk.name, []):
            out += body(m.link(j.child), lk, j, pad + "  ")
        out.append(f"{pad}</body>")
        return out

    lines = []
    for j in children.get(None, []):                                          # welded or jointed to the world
        lines += body(m.link(j.child), None, j, "    ")
    for lk in m.roots():
        if m.parent_joint(lk.name) is None:
            lines += body(lk, None, None, "    ")
    actuators = []
    for j in m.joints:
        if j.drive and j.kind in ("revolute", "continuous", "prismatic"):
            a = [f'joint={quoteattr(j.name)}', f'kp="{j.drive.get("stiffness", 0.0):.17g}"']
            if j.drive.get("damping"):
                a.append(f'kv="{j.drive["damping"]:.17g}"')
            if j.drive.get("max_force"):
                a.append(f'forcelimited="true" forcerange="{-j.drive["max_force"]:.17g} {j.drive["max_force"]:.17g}"')
            actuators.append(f'    <position name={quoteattr(j.name + "_drive")} {" ".join(a)}/>')
    opt = m.data.get("mjcf", {})
    head = [f'<mujoco model={quoteattr(m.name)}>',
            '  <compiler angle="radian"' + (' boundmass="1e-9" boundinertia="1e-12"' if massless else "") + "/>"]
    if opt:
        head.append(f'  <option timestep="{opt.get("timestep", 0.002):.17g}" gravity="{_f(opt.get("gravity", (0, 0, -9.81)))}"/>')
    if massless:
        rep.drop("massless jointed links (several dofs on one body): given MuJoCo's minimum mass 1e-9 kg")
    xml = head + (["  <asset>", *assets, "  </asset>"] if assets else []) + ["  <worldbody>", *lines, "  </worldbody>"]
    if actuators:
        xml += ["  <actuator>", *actuators, "  </actuator>"]
    xml.append("</mujoco>")
    path.write_text("\n".join(xml) + "\n")
    mujoco.MjModel.from_xml_path(str(path))                                    # it must compile
    return rep
