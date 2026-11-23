"""The machine's looks: a CAD design rides in the export as the root link's visual mesh (``vegeta.labkit.visual``,
``machines.register(..., visual=...)``). Mock robot, a CadQuery box as the design; no lab design is built."""
import sys
from pathlib import Path

import numpy as np
import pytest

yax = pytest.importorskip("yax")
pytest.importorskip("mujoco")
cq = pytest.importorskip("cadquery")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export  # noqa: E402
from vegeta.labkit import visual as lv  # noqa: E402


def _robot():
    from vegeta.chiron import Geom, Joint, Link, Robot
    wheel = Link("wheel", pos=(0.08, 0.06, -0.02), joints=[Joint("axle", axis=(0, 1, 0), range=None)],
                 geoms=[Geom("tyre", "cylinder", (0.03, 0.01), quat=(0.7071068, -0.7071068, 0.0, 0.0), mass=0.05)])
    root = Link("body", geoms=[Geom("hull", "box", (0.1, 0.05, 0.02), mass=1.0)], children=[wheel])
    return Robot("mockcar", root=root)


def _design():
    """A 200 × 140 × 70 mm box with a bored hole (mm, the designs' unit), off the origin like a CAD part."""
    return cq.Workplane("XY").box(200, 140, 70).faces(">Z").workplane().hole(30).translate((500, 0, 35))


@pytest.fixture
def registered():
    from vegeta.labkit import machines
    machines.register("mockcar", _robot, visual=_design)
    machines.register("mockbare", _robot)
    yield
    machines._REGISTRY.pop("mockcar", None)
    machines._REGISTRY.pop("mockbare", None)


def test_the_mesh_is_light_and_keeps_its_outline():
    pts, faces = lv.tessellate(cq.Workplane("XY").sphere(50))
    p, f = lv.cluster(pts, faces, 0.01)
    assert len(f) < len(faces) and f.max() < len(p)
    assert np.allclose(np.ptp(p, axis=0), np.ptp(pts, axis=0), atol=0.01)
    small, sf = lv.mesh(lambda: cq.Workplane("XY").sphere(50), max_triangles=200)
    assert len(sf) <= 200 and np.allclose(np.ptp(small, axis=0), 0.1, atol=0.012)


def test_the_design_rides_as_the_root_links_visual(registered, tmp_path):
    reports = export.export("mockcar", tmp_path / "mockcar.yx2")
    assert reports
    m, _ = yax.read(tmp_path / "mockcar.yx2")
    cad = [s for s in m.links[0].shapes if s.kind == "mesh"]
    assert len(cad) == 1 and cad[0].role == "visual" and cad[0].name == "cad"
    lo, hi = lv.extent(m)                                      # the primitives: hull and wheel at q = 0
    pts = np.asarray(cad[0].points) + np.asarray(m.links[0].pos)
    assert np.allclose(pts.min(axis=0)[2], lo[2], atol=1e-6)   # on the same ground
    assert np.allclose((pts.min(axis=0) + pts.max(axis=0))[:2] / 2, (lo + hi)[:2] / 2, atol=1e-6)
    assert np.allclose(np.ptp(pts, axis=0), (0.2, 0.14, 0.07), atol=1e-3)                    # metres
    assert "00geom" in m.tags["table"]["link:body/visual:cad"]
    bare, _ = export.machine("mockbare")
    assert not [s for s in bare.links[0].shapes if s.kind == "mesh"]
    off, _ = export.machine("mockcar", visual=False)
    assert not [s for s in off.links[0].shapes if s.kind == "mesh"]
    assert lv.MAX_TRIANGLES >= 1000
