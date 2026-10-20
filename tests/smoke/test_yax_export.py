"""Vegeta -> Yax: a cube with exact mass properties, a real component whose OpenCascade values agree with Yax's
own integration of the exported STL, and the package conventions consumers rely on (mm, Z up, X forward)."""
import math

import pytest

yax = pytest.importorskip("yax")


def close(a, b, rel):
    if isinstance(a, list):
        return all(close(x, y, rel) for x, y in zip(a, b))
    return math.isclose(a, b, rel_tol=rel, abs_tol=rel * 1e-3)


def test_cube_is_exact(tmp_path):
    import cadquery as cq
    from vegeta.dedalus import Geometry, export_yax
    g = Geometry.from_cadquery(cq.Workplane().box(10, 10, 10, centered=False), name="cube")
    d = yax.load(export_yax(g, tmp_path / "cube.yax", density_kg_m3=2700.0, step=True).dir)
    i = 0.0027 * 0.01 ** 2 / 6
    assert d.length_unit == "mm" and d.frame == {"up": "+z", "forward": "+x"}
    assert close(d.mass["kg"], 0.0027, 1e-12) and close(d.mass["com"], [5.0, 5.0, 5.0], 1e-12)
    assert close(d.mass["inertia"], [[i, 0, 0], [0, i, 0], [0, 0, i]], 1e-9)
    assert sorted(d.manifest["files"]) == ["body.step", "body.stl"] and len(d.mesh()) == 12
    assert d.manifest["producer"]["tool"] == "vegeta.dedalus" and yax.check(d.dir) == []


def test_component_round_trip(tmp_path, golden):
    from components.quad_frame import QuadFrame
    from vegeta.dedalus import export_yax
    g = QuadFrame().generate(wheelbase=180.0)
    d = export_yax(g, tmp_path / "frame.yax", name="quad_frame", mass_kg=0.05, stl_tolerance=0.05)
    mesh = yax.describe(d.mesh(), "mm", mass_kg=0.05)            # what a consumer without CAD would compute
    assert close(mesh["geometry"]["volume"], d.geometry["volume"], 2e-3)        # tessellation of round holes
    assert all(abs(a - b) < 0.05 for a, b in zip(mesh["mass"]["com"], d.mass["com"]))       # mm
    for k in range(3):
        assert close(mesh["mass"]["inertia"][k][k], d.mass["inertia"][k][k], 5e-3)
    assert d.manifest["producer"]["parameters"]["wheelbase"] == 180.0
    assert d.manifest["identity"] == yax.identity(d.manifest["producer"])
    golden("yax.quad_frame_180", {"volume": d.geometry["volume"], "com": d.mass["com"], "inertia": d.mass["inertia"],
                                   "bbox": d.geometry["bbox"]}, rel=1e-7, abs=1e-9)
