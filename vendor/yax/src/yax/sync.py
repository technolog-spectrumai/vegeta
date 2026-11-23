"""Physics sync: take the physical numbers of a design into a model another team owns — and nothing else.

Two teams, two feedback loops: the designers' file (any standard format, USD best — it also carries part data such as
rotor constants) and the simulation team's model (an SDF with its sensors, controllers, topics and gains). ``sync``
rewrites only what the map names — link masses, centres and inertias, link poses, joint axes, single values inside
plugins — and leaves every other element, attribute and comment as it was. The report lists every value it changed
(old → new); with ``check=True`` nothing is written and a non-empty report means the two have drifted apart.

The map (JSON)::

    {"design_root": "frame", "target_root": "base_link",
     "links":  {"frame": "base_link", "rotor_fl": "rotor_0"},          # mass, centre, inertia (+ pose with "poses")
     "joints": {"prop_fl": "rotor_0_joint"},                            # axis (+ position with "poses")
     "take":   ["mass", "poses", "axes"],
     "values": [{"target": "plugin[jointName='rotor_0_joint']/motorConstant", "from": "data.ogma.rotors.0.kF"},
                {"target": "link:rotor_0:pose", "from": "data.ogma.rotors.0.pos"}]}

``target`` is an ElementTree path under ``<model>`` (one element) or ``link:<name>:pose``; ``from`` is a dotted path
into the design machine (``data.…`` part data, ``link.<name>.mass``). Positions are taken relative to the design's root
link and placed relative to the target's root link, whose own pose is never touched.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import hub
from .convert import read
from .hub import Report


def load_map(path) -> dict:
    return json.loads(Path(path).read_text())


def _lookup(m, dotted: str):
    head, *rest = dotted.split(".")
    if head == "data":
        obj = m.data
    elif head == "link":
        obj = m.link(rest.pop(0)).__dict__
    else:
        raise KeyError(f"'from' must start with data. or link.: {dotted}")
    for k in rest:
        obj = obj[int(k)] if isinstance(obj, (list, tuple)) else obj[k]
    return obj


def _fmt(v) -> str:
    return " ".join(repr(float(x) + 0.0) for x in np.ravel(v))      # shortest exact form; + 0.0 turns -0.0 into 0.0


def _vals(text):
    return [float(x) for x in (text or "").split()]


def _differs(old, new, tol) -> bool:
    a, b = np.ravel(old).astype(float), np.ravel(new).astype(float)
    return a.shape != b.shape or not np.allclose(a, b, rtol=tol, atol=tol * 1e-3)


class _Editor:
    def __init__(self, rep: Report, check: bool, tol: float):
        self.rep, self.check, self.tol, self.changed = rep, check, tol, 0

    def set(self, parent, tag, new, label):
        """Set ``parent/tag`` text to ``new`` (creating it), reporting a real change."""
        el = parent.find(tag)
        old = _vals(el.text) if el is not None and el.text else None
        if old is None and not np.any(np.asarray(new, float)):
            return                                             # absent = zero in SDF: nothing to write
        if old is not None and not _differs(old, new, self.tol):
            return
        self.rep.changes.append(f"{label}: {_fmt(old) if old is not None else '—'} → {_fmt(new)}")
        self.changed += 1
        if not self.check:
            if el is None:
                el = ET.SubElement(parent, tag)
            el.text = _fmt(new)


def _sdf_pose(el):
    p = el.find("pose")
    v = (_vals(p.text) + [0.0] * 6)[:6] if p is not None and p.text else [0.0] * 6
    return tuple(v[:3]), hub.rpy_to_quat(*v[3:])


def sync(design, target, *, mapping, check: bool = False, out=None, tol: float = 1e-9) -> Report:
    """Take the mapped physical fields of ``design`` into the SDF ``target`` (written to ``out``, default in place)."""
    mp = load_map(mapping) if isinstance(mapping, (str, Path)) else mapping
    target = Path(target)
    if target.suffix.lower() != ".sdf":
        raise ValueError("sync writes SDF targets (the simulation models); convert other formats with yax.convert")
    d, _ = read(design)
    rep = Report(str(design), str(target))
    ed = _Editor(rep, check, tol)
    text = target.read_text()
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.ElementTree(ET.fromstring(text, parser=parser))
    model = tree.getroot().find("model")
    take = set(mp.get("take", ["mass"]))
    links = {el.get("name"): el for el in model.findall("link")}
    joints = {el.get("name"): el for el in model.findall("joint")}
    droot = d.link(mp["design_root"]) if mp.get("design_root") else d.roots()[0]
    troot_el = links[mp.get("target_root", "base_link")]
    trp, trq = _sdf_pose(troot_el)

    placed = {}                                        # link poses this run sets (joints are placed against them)

    def pose_of(name):
        return placed.get(name) or _sdf_pose(links[name])

    def to_target(pos_world_design, quat=(1.0, 0.0, 0.0, 0.0)):
        rel_p, rel_q = hub.to_frame(pos_world_design, quat, droot.pos, droot.quat)
        return hub.from_frame(rel_p, rel_q, trp, trq)

    for dname, tname in mp.get("links", {}).items():
        lk, el = d.link(dname), links.get(tname)
        if el is None:
            rep.drop(f"map: target link {tname!r} not in the model")
            continue
        if "poses" in take and el is not troot_el:
            p, q = to_target(lk.pos, lk.quat)
            placed[tname] = (p, q)
            ed.set(el, "pose", [*p, *hub.quat_to_rpy(q)], f"link {tname} pose")
        if "mass" in take and lk.mass is not None:
            ine = el.find("inertial")
            if ine is None:
                ine = ET.SubElement(el, "inertial")
            ed.set(ine, "mass", [lk.mass], f"link {tname} mass")
            ed.set(ine, "pose", [*lk.com, 0.0, 0.0, 0.0], f"link {tname} centre of mass")
            I = np.asarray(lk.inertia if lk.inertia is not None else np.zeros((3, 3)), float)
            iel = ine.find("inertia")
            if iel is None:
                iel = ET.SubElement(ine, "inertia")
            for k, v in (("ixx", I[0, 0]), ("ixy", I[0, 1]), ("ixz", I[0, 2]), ("iyy", I[1, 1]), ("iyz", I[1, 2]), ("izz", I[2, 2])):
                ed.set(iel, k, [v], f"link {tname} {k}")
    for dname, tname in mp.get("joints", {}).items():
        j, el = next((x for x in d.joints if x.name == dname), None), joints.get(tname)
        if j is None or el is None:
            rep.drop(f"map: joint {dname!r} → {tname!r} not found")
            continue
        cp, cq = pose_of(el.find("child").text.strip())
        if "axes" in take and j.kind in ("revolute", "continuous", "prismatic"):
            Rd = hub.R(droot.quat).T @ np.asarray(j.axis, float)               # design root frame
            axis = hub.R(cq).T @ (hub.R(trq) @ Rd)                              # target child (joint) frame
            ax = el.find("axis")
            if ax is None:
                ax = ET.SubElement(el, "axis")
            ed.set(ax, "xyz", axis, f"joint {tname} axis")
        if "poses" in take:
            jp, _ = to_target(j.pos)
            ed.set(el, "pose", [*(hub.R(cq).T @ (np.asarray(jp) - np.asarray(cp))), 0.0, 0.0, 0.0], f"joint {tname} position")
    for v in mp.get("values", []):
        try:
            val = _lookup(d, v["from"])
        except (KeyError, IndexError, ValueError, StopIteration) as e:
            rep.drop(f"map: {v['from']} not in the design ({type(e).__name__})")
            continue
        t = v["target"]
        if t.startswith("link:") and t.endswith(":pose"):
            name = t.split(":")[1]
            el = links.get(name)
            if el is None:
                rep.drop(f"map: target link {name!r} not in the model")
                continue
            p, _ = to_target(val)
            q = pose_of(name)[1]
            placed[name] = (p, q)
            ed.set(el, "pose", [*p, *hub.quat_to_rpy(q)], f"link {name} pose")
            continue
        parent_path, _, tag = t.rpartition("/")
        parent = model.find(parent_path) if parent_path else model
        if parent is None:
            rep.drop(f"map: target {parent_path!r} not in the model")
            continue
        ed.set(parent, tag, val, t)
    if not check and ed.changed:                       # the prolog (declaration, header comments) kept as it was
        prolog = text[:text.index("<" + tree.getroot().tag)]
        Path(out or target).write_text(prolog + ET.tostring(tree.getroot(), encoding="unicode") + "\n")
    return rep
