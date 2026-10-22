"""Yax 1 packages (``yax.json`` + ``body.stl``): read only — one link with the package's mass properties and mesh."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import package, units as yunits
from ..hub import Link, Machine, Report, Shape


def read(path) -> tuple[Machine, Report]:
    d = package.load(path)
    k = yunits.LENGTH[d.length_unit]
    pts = np.asarray(d.mesh(), float).reshape(-1, 3) * k
    lk = Link(name=d.name, shapes=[Shape(name="body", kind="mesh", points=pts, faces=np.arange(len(pts)).reshape(-1, 3), uri="body.stl")])
    rep = Report(str(path), "usd")
    if d.mass:
        lk.mass, lk.com, lk.inertia = float(d.mass["kg"]), tuple(np.asarray(d.mass["com"], float) * k), np.asarray(d.mass["inertia"], float)
    else:
        rep.drop("mass properties (the package has none)")
    if d.frame != {"up": "+z", "forward": "+x"}:
        rep.drop(f"its frame {d.frame} (read as Z up, X forward)")
    m = Machine(name=d.name, links=[lk], producer=d.manifest.get("producer", {}))
    for p in d.parts:
        rep.drop(f"part {p['name']} (Yax 1 parts are not read)")
    return m, rep
