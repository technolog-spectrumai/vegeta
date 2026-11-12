"""animate_particles with a water surface: water below coloured by speed, particles above in one flat colour."""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
cv2 = pytest.importorskip("cv2")
from vegeta.aeromant import viz  # noqa: E402
from vegeta.aeromant.stl import Surface, write_stl_ascii  # noqa: E402


def test_surface_split_renders(tmp_path, monkeypatch):
    grid = pv.ImageData(dimensions=(40, 20, 20), spacing=(0.1, 0.1, 0.1), origin=(-2.0, -1.0, -1.0))
    grid.point_data["U"] = np.tile([1.5, 0.0, 0.0], (grid.n_points, 1)) + 0.1 * np.random.default_rng(0).normal(size=(grid.n_points, 3))
    mb = pv.MultiBlock({"internalMesh": grid})
    box = pv.Box(bounds=(-0.5, 0.5, -0.1, 0.1, -0.2, 0.2)).triangulate()
    case = tmp_path / "case"
    (case / "constant" / "triSurface").mkdir(parents=True)
    tris = box.faces.reshape(-1, 4)[:, 1:]
    write_stl_ascii(Surface(np.asarray(box.points)[tris]), case / "constant" / "triSurface" / "body.stl", "body")
    monkeypatch.setattr(viz, "read_results", lambda c: mb)
    monkeypatch.setattr(viz, "case_info", lambda c: {"body_bbox_m": [[-0.5, -0.1, -0.2], [0.5, 0.1, 0.2]]})
    shown = []
    real_add_text = pv.Plotter.add_text
    monkeypatch.setattr(pv.Plotter, "add_text", lambda self, text, **k: shown.append(text) or real_add_text(self, text, **k))
    out = viz.animate_particles(case, tmp_path / "boat.mp4", n=60, seconds=0.5, fps=6, size=(320, 240), surface_z=0.05,
                                above_label="mirror image (double body)")
    assert out.is_file() and out.stat().st_size > 1000
    assert any("mirror image (double body): grey" in t for t in shown)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
