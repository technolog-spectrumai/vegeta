"""StructuralModel.ensure and solve_models: a solved model is read back, never solved twice; a changed load solves
again on the same mesh; a changed mesh setting meshes again; a batch resumes."""
import pytest

from vegeta import talos
from vegeta.talos import FixedSupport, Force, Material, MeshSettings, StructuralModel, SurfacesOnPlane

STEEL = Material("steel", 210000.0, 0.3, density=7.85e-9, yield_strength=235.0)
pytestmark = [pytest.mark.requires_gmsh, pytest.mark.requires_ccx]


def _cantilever(step, fz=-100.0, size=8.0):
    return StructuralModel(step, "mm-N-MPa", STEEL,
                           regions=[SurfacesOnPlane("fixed", "x", 0.0), SurfacesOnPlane("tip", "x", 200.0)],
                           supports=[FixedSupport("fixed")], loads=[Force("tip", fz=fz)], mesh_settings=MeshSettings(size))


@pytest.fixture
def counted(monkeypatch):
    calls = {"mesh": 0, "solve": 0}
    mesh, solve = StructuralModel.mesh, StructuralModel.solve

    def m(self, *a, **k):
        calls["mesh"] += 1
        return mesh(self, *a, **k)

    def s(self, *a, **k):
        calls["solve"] += 1
        return solve(self, *a, **k)

    monkeypatch.setattr(StructuralModel, "mesh", m)
    monkeypatch.setattr(StructuralModel, "solve", s)
    return calls


def test_not_run_touches_nothing(beam_step, tmp_path, counted):
    r = _cantilever(beam_step).ensure(tmp_path / "w", run=False)
    assert not r.ok and "NOT RUN" in r.messages[0] and counted == {"mesh": 0, "solve": 0}
    assert not (tmp_path / "w").exists()


def test_solved_once_then_read_back(beam_step, tmp_path, counted):
    first = _cantilever(beam_step).ensure(tmp_path / "w")
    assert first.ok and first.metadata["reused"] is False and counted == {"mesh": 1, "solve": 1}
    again = _cantilever(beam_step).ensure(tmp_path / "w")
    assert again.ok and again.metadata["reused"] is True and counted == {"mesh": 1, "solve": 1}
    assert again.metrics["max_displacement"] == pytest.approx(first.metrics["max_displacement"])
    assert again.artifacts["frd"].is_file() and again.artifacts["mesh"].is_file()   # plots work on the read-back result
    assert talos.read_frd(again.artifacts["frd"]).displacement.shape[0] > 0


def test_changed_load_solves_again_on_the_same_mesh(beam_step, tmp_path, counted):
    _cantilever(beam_step).ensure(tmp_path / "w")
    r = _cantilever(beam_step, fz=-200.0).ensure(tmp_path / "w")
    assert r.ok and r.metadata["reused"] is False and counted == {"mesh": 1, "solve": 2}
    assert _cantilever(beam_step).key != _cantilever(beam_step, fz=-200.0).key


def test_changed_mesh_setting_meshes_again(beam_step, tmp_path, counted):
    _cantilever(beam_step).ensure(tmp_path / "w")
    r = _cantilever(beam_step, size=6.0).ensure(tmp_path / "w")
    assert r.ok and counted == {"mesh": 2, "solve": 2}


def test_batch_resumes(beam_step, tmp_path, counted):
    models = [_cantilever(beam_step, fz=-100.0 * (i + 1)) for i in range(3)]
    dirs = [tmp_path / f"m{i}" for i in range(3)]
    models[0].ensure(dirs[0])
    rs = talos.solve_models(models, dirs)
    assert all(r.ok for r in rs) and [r.metadata["reused"] for r in rs] == [True, False, False]
    assert counted == {"mesh": 3, "solve": 3}
    d = [r.metrics["max_displacement"] for r in rs]
    assert d[1] == pytest.approx(2 * d[0], rel=1e-3) and d[2] == pytest.approx(3 * d[0], rel=1e-3)
    again = talos.solve_models(models, dirs)
    assert all(r.metadata["reused"] for r in again) and counted == {"mesh": 3, "solve": 3}


def test_batch_without_running_and_input_checks(beam_step, tmp_path):
    rs = talos.solve_models([_cantilever(beam_step)], [tmp_path / "x"], run=False)
    assert not rs[0].ok and "NOT RUN" in rs[0].messages[0]
    with pytest.raises(ValueError, match="share a workdir"):
        talos.solve_models([_cantilever(beam_step), _cantilever(beam_step, fz=-1.0)], [tmp_path / "y", tmp_path / "y"])
    with pytest.raises(ValueError, match="one workdir per model"):
        talos.solve_models([_cantilever(beam_step)], [])
