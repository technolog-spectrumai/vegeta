"""STEP (CAD). Read with CadQuery/OpenCascade: one link, a tessellated mesh per solid and — given a mass or a density —
the exact mass properties (adaptive integration). Written as solids placed in the machine frame: primitives exactly,
meshes as faceted solids. STEP carries geometry, not mass or joints (both reported). CAD units are mm by default."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import hub
from ..hub import Link, Machine, Report, Shape

TO_M = {"mm": 1e-3, "m": 1.0}


def _gprops(shape, precision=1e-9):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p, precision, True)
    c, M = p.CentreOfMass(), p.MatrixOfInertia()
    return p.Mass(), np.array([c.X(), c.Y(), c.Z()]), np.array([[M.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])


def read(path, *, units: str = "mm", mass_kg: float | None = None, density_kg_m3: float | None = None,
         tolerance: float = 0.05) -> tuple[Machine, Report]:
    import cadquery as cq

    path = Path(path)
    rep = Report(str(path), "usd")
    wp = cq.importers.importStep(str(path))
    k = TO_M[units]
    solids = [s for v in wp.vals() for s in (v.Solids() if hasattr(v, "Solids") else [v])]
    lk = Link(name=path.stem)
    for i, s in enumerate(solids):
        verts, tris = s.tessellate(tolerance, 0.2)
        pts = np.array([[v.x, v.y, v.z] for v in verts], float) * k
        lk.shapes.append(Shape(name=f"solid{i}", kind="mesh", points=pts, faces=np.asarray(tris, int), uri=path.name))
    if mass_kg is not None or density_kg_m3 is not None:
        comp = cq.Compound.makeCompound(solids).wrapped
        vol, com, I1 = _gprops(comp)
        kg = mass_kg if mass_kg is not None else density_kg_m3 * vol * k ** 3
        lk.mass, lk.com = float(kg), tuple(com * k)
        lk.inertia = I1 * (kg / vol) * k ** 2
    else:
        rep.drop("mass properties (STEP has none: pass mass_kg or density_kg_m3)")
    return Machine(name=path.stem, links=[lk]), rep


def _solid(s: Shape, k: float):
    """A shape as an OCP solid in its link frame (CAD units)."""
    import cadquery as cq
    plane = {"x": "YZ", "y": "XZ", "z": "XY"}
    if s.kind == "box":
        w = cq.Workplane("XY").box(*(v / k for v in s.size))
    elif s.kind == "sphere":
        w = cq.Workplane("XY").sphere(s.size[0] / k)
    elif s.kind in ("cylinder", "capsule"):
        r, h = s.size[0] / k, s.size[1] / k
        w = cq.Workplane(plane[s.axis]).circle(r).extrude(h / 2, both=True)
        if s.kind == "capsule":
            n = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[s.axis]
            for sgn in (1, -1):
                w = w.union(cq.Workplane("XY").sphere(r).translate(tuple(sgn * h / 2 * c for c in n)))
    elif s.kind == "ellipsoid":
        from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
        from OCP.gp import gp_GTrsf, gp_Mat
        g = gp_GTrsf()
        g.SetVectorialPart(gp_Mat(s.size[0] / k, 0, 0, 0, s.size[1] / k, 0, 0, 0, s.size[2] / k))
        return BRepBuilderAPI_GTransform(cq.Solid.makeSphere(1.0, angleDegrees1=-90, angleDegrees2=90).wrapped, g, True).Shape(), True
    else:
        return _faceted(np.asarray(s.points, float) / k, np.asarray(s.faces, int)), False
    return w.val().wrapped, True


def _faceted(pts, faces):
    """A mesh as a sewn faceted solid (a shell when it does not close)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt
    sew = BRepBuilderAPI_Sewing(1e-6)
    for f in faces:
        poly = BRepBuilderAPI_MakePolygon(*[gp_Pnt(*map(float, pts[i])) for i in f], True)
        if poly.IsDone():
            face = BRepBuilderAPI_MakeFace(poly.Wire(), True)
            if face.IsDone():
                sew.Add(face.Face())
    sew.Perform()
    shape = sew.SewedShape()
    try:
        return BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shape)).Solid()
    except Exception:
        return shape


def _placed(shape, pos, quat, k):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Quaternion, gp_Trsf, gp_Vec
    t = gp_Trsf()
    t.SetRotation(gp_Quaternion(float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])))
    t.SetTranslationPart(gp_Vec(*(float(v) / k for v in pos)))
    return BRepBuilderAPI_Transform(shape, t, True).Shape()


def write(m: Machine, path, rep: Report | None = None, *, units: str = "mm") -> Report:
    import cadquery as cq

    path = Path(path)
    rep = rep or Report("usd", str(path))
    k = TO_M[units]
    shapes = []
    for lk in m.links:
        for s in lk.shapes:
            if s.kind == "mesh" and (s.points is None or len(s.points) == 0):
                rep.drop(f"{lk.name}/{s.name}: an empty mesh ({s.uri})")
                continue
            solid, _ = _solid(s, k)
            pos, quat = hub.from_frame(s.pos, s.quat, lk.pos, lk.quat)
            shapes.append(_placed(solid, pos, quat, k))
    if any(lk.mass for lk in m.links):
        rep.drop("mass properties (STEP has none)")
    if m.joints:
        rep.drop("joints (STEP carries geometry only; the parts are placed at q = 0)")
    if m.data:
        rep.drop("machine data (" + ", ".join(sorted(m.data)) + ")")
    comp = cq.Compound.makeCompound([cq.Shape.cast(s) for s in shapes])
    cq.exporters.export(comp, str(path), exportType="STEP")
    return rep
