import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

from vegeta.dedalus import viz
from vegeta.dedalus.examples import Bracket, Cube


def test_sections_and_3d(tmp_path):
    cube = Cube().generate(size=10.0)
    polys = viz.section_polylines(cube, "z", 5.0)
    assert len(polys) == 1 and len(polys[0]) >= 4
    assert viz.plot_sections(Bracket().generate(), "z", n=2) is not None
    pl = viz.plot3d(cube, Bracket().generate())
    assert pl.screenshot(str(tmp_path / "x.png")).shape[0] > 0
    pl.close()
    assert viz.to_pyvista(cube).n_cells == 12
