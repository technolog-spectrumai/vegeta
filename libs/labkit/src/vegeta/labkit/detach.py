"""Assemblies and detachable components as Yax part data (``vendor/yax/docs/engineering.md``, the ``00detach`` block)
from what a design states: the machine's joint tree gives the parent–child relations and attachment positions (every
link is a component of its parent's), the design says which components come off, through which connection, with which
separation limits. Nothing here decides that a part is detachable, sizes a limit or predicts a separation."""
from __future__ import annotations

import numpy as np


def component(link: str, *, parent: str | None, kind: str | None = None, detachable: bool | None = None, connection: str | None = None,
              attachment: dict | None = None, separation: dict | None = None, material: str | None = None, source: str = "") -> dict:
    out = {"link": link, "parent": parent}
    for k, v in (("kind", kind), ("detachable", detachable), ("connection", connection), ("attachment", attachment),
                 ("separation", separation), ("material", material)):
        if v is not None:
            out[k] = v
    if source:
        out["source"] = source
    return out


def from_joint_tree(m, *, name: str, source: str, kinds: dict | None = None, detachable: dict | None = None) -> dict:
    """The assembly of a machine from its joint tree: one component per link, its parent the joint's parent link, its
    attachment the joint's position in the parent's frame (axis along the joint's). ``kinds`` names a link's kind;
    ``detachable`` maps a link name to ``{"connection": name, "separation": {...}?, "source": ...}`` for the components
    the design means to come off — only those are marked detachable."""
    from yax import hub
    kinds, detachable = kinds or {}, detachable or {}
    parent_of = {j.child: j for j in m.joints if j.parent is not None}
    comps = {}
    root = None
    for lk in m.links:
        j = parent_of.get(lk.name)
        at = None
        if j is not None:
            plk = m.link(j.parent)
            pos, _ = hub.to_frame(j.pos, (1.0, 0.0, 0.0, 0.0), plk.pos, plk.quat)
            at = {"frame": "parent", "pos": [float(x) for x in pos], "source": f"{source}: the joint {j.name}"}
            if j.kind != "fixed":
                at["axis"] = [float(x) for x in (hub.R(plk.quat).T @ np.asarray(j.axis, float))]
        else:
            root = root or lk.name
        d = detachable.get(lk.name)
        comps[lk.name] = component(lk.name, parent=j.parent if j is not None else None, kind=kinds.get(lk.name), attachment=at,
                                   detachable=True if d else None, connection=d.get("connection") if d else None,
                                   separation=d.get("separation") if d else None, material=lk.data.get("material"),
                                   source=d.get("source", source) if d else source)
    out = {"name": name, "components": comps, "source": source}
    if root:
        out["root"] = root
    return out
