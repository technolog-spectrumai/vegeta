"""STL: one mesh. Read as one link with one mesh shape; written as every shape's triangles in the machine frame
(lengths in ``units``, mm by default, as CAD and slicers expect)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import hub, stl
from ..hub import Link, Machine, Report, Shape

TO_M = {"mm": 1e-3, "m": 1.0}


def read(path, *, units: str = "mm") -> tuple[Machine, Report]:
    path = Path(path)
    pts = np.asarray(stl.read(path), float).reshape(-1, 3) * TO_M[units]
    lk = Link(name=path.stem, shapes=[Shape(name="mesh", kind="mesh", points=pts, faces=np.arange(len(pts)).reshape(-1, 3), uri=path.name)])
    rep = Report(str(path), "usd")
    rep.drop("mass properties (an STL has none)")
    return Machine(name=path.stem, links=[lk]), rep


def write(m: Machine, path, rep: Report | None = None, *, units: str = "mm") -> Report:
    path = Path(path)
    rep = rep or Report("usd", str(path))
    tris = []
    for lk in m.links:
        for s in lk.shapes:
            pts, faces = hub.tessellate(s)
            if not len(faces):
                continue
            W = (hub.R(lk.quat) @ pts.T).T + np.asarray(lk.pos)
            tris += [tuple(map(tuple, W[f] / TO_M[units])) for f in faces]
    if any(lk.mass for lk in m.links):
        rep.drop("mass properties (an STL has none)")
    if m.joints:
        rep.drop("joints (one mesh at q = 0)")
    stl.write(path, tris)
    return rep
