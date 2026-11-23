import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True
pytestmark = [pytest.mark.requires_gmsh, pytest.mark.requires_ccx]


@pytest.fixture
def solved(beam_step, tmp_path):
    from vegeta.talos import FixedSupport, Force, Material, MeshSettings, StructuralModel, SurfacesOnPlane

    m = StructuralModel(beam_step, "mm-N-MPa", Material("s", 210000, 0.3),
                        regions=[SurfacesOnPlane("fixed", "x", 0.0), SurfacesOnPlane("tip", "x", 200.0)],
                        supports=[FixedSupport("fixed")], loads=[Force("tip", fz=-100.0)],
                        mesh_settings=MeshSettings(10.0))
    m.mesh(tmp_path)
    return m, m.solve(tmp_path)


def test_mesh_and_results_grids(solved, tmp_path):
    from vegeta.talos import viz

    model, res = solved
    grid = viz.mesh_to_pyvista(res.artifacts["mesh"])
    assert grid.n_cells == res.metrics["n_elements"] and "region:fixed" in grid.point_data
    g = viz.results_to_pyvista(res)
    assert g.point_data["von_mises"].max() == pytest.approx(res.metrics["max_von_mises"], rel=1e-6)
    assert g.point_data["|U|"].max() == pytest.approx(res.metrics["max_displacement"], rel=1e-6)
    vtu = viz.export_vtu(res, tmp_path / "beam.vtu")
    assert vtu.stat().st_size > 0 and pv.read(str(vtu)).n_points == g.n_points


def test_plots_render(solved, tmp_path):
    from vegeta.talos import viz

    model, res = solved
    for pl in (viz.plot_mesh(res.artifacts["mesh"], clip="y"), viz.plot_problem(model, res.artifacts["mesh"]),
               viz.plot_results(res, clip="y")):
        img = pl.screenshot(str(tmp_path / "x.png"))
        pl.close()
        assert img.shape[0] > 0
    fig = viz.plot_section(res, normal="y")
    assert fig is not None


def test_animate_writes_video(solved, tmp_path):
    from vegeta.talos import viz

    cv2 = pytest.importorskip("cv2")
    model, res = solved
    out = viz.animate(res, tmp_path / "beam.mp4", rpm=30.0, axis="x", seconds=0.5, fps=8, size=(320, 240))
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4 and cap.read()[0]
    out2 = viz.animate(res, tmp_path / "orbit.mp4", seconds=0.5, fps=4, size=(320, 240), load_ramp=False)
    assert out2.stat().st_size > 0
