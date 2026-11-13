from pathlib import Path

import pytest

from vegeta.mellonia import read_gcode

DATA = Path(__file__).parent / "data"


def test_moves_are_parsed():
    info = read_gcode(DATA / "tiny.gcode", moves=True)
    assert info.segments is not None and len(info.segments) == 4
    assert list(info.segment_layer) == [0, 1, 1, 2] and all(info.segment_extruding)
    assert info.segments[0].tolist() == [10, 10, 0, 20, 10, 0]


@pytest.mark.requires_prusaslicer
def test_toolpath_plots(tmp_path, cube_stl):
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    from vegeta.mellonia import Orientation, slice_stl, viz
    from vegeta.mellonia.examples import GENERIC_PLA_0_2MM

    r = slice_stl(cube_stl, GENERIC_PLA_0_2MM, Orientation(), tmp_path)
    poly = viz.toolpath_to_pyvista(r)
    assert poly.n_lines > 1000 and poly.cell_data["layer"].max() == r.metrics["layer_count"] - 1
    pl = viz.plot_toolpath(r)
    assert pl.screenshot(str(tmp_path / "t.png")).shape[0] > 0
    pl.close()
    assert viz.plot_layer(r, 10) is not None and viz.plot_layer_grid(r, n=4) is not None
