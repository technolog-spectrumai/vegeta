"""``_common.solve_modes`` through the mocks: mesh here, or the mesh of a solved case copied."""
from __future__ import annotations

from vegeta import talos

from assemblies.workflows import _common
from test_life import _dummy


def test_solve_modes_meshes_or_copies_through_the_mocks(tmp_path, solvers, monkeypatch):
    m = _dummy(tmp_path, "modal")
    r = _common.solve_modes(m, tmp_path / "modal", n_modes=4)
    assert r.ok and r.metrics["n_modes"] == 4 and solvers.mesh.call_count == 1
    monkeypatch.setattr(talos.StructuralModel, "mesh_is_current", lambda self, d: str(d).endswith("solved"))
    (tmp_path / "solved").mkdir()
    (tmp_path / "solved" / "mesh.msh").write_text("x")
    r2 = _common.solve_modes(m, tmp_path / "modal2", n_modes=6, mesh_from=tmp_path / "solved")
    assert r2.ok and (tmp_path / "modal2" / "mesh.msh").is_file() and solvers.mesh.call_count == 1   # copied, not meshed
