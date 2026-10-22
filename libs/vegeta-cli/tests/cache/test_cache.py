"""vegeta.cache: simulation results cached next to the notebook. The entry exists -> loaded, nothing runs; it does not ->
the simulation runs and a successful result is saved (with copies of its files); nothing checks inputs or code.
No solver runs here: the private ``_solve``/``_mesh``/``_run`` are replaced by fakes that write files."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vegeta import aeromant, cache, talos

STEEL = talos.Material("steel", 210000.0, 0.3, density=7.85e-9, yield_strength=235.0)


@pytest.fixture(autouse=True)
def folder(tmp_path, monkeypatch):
    """Every test with its own cache folder, the module state reset afterwards."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VEGETA_CACHE_DIR", raising=False)
    monkeypatch.delenv("VEGETA_CACHE", raising=False)
    monkeypatch.delenv("JPY_SESSION_NAME", raising=False)
    for tool_cache in (talos.cache, aeromant.cache):
        monkeypatch.setattr(tool_cache, "verbose", False)
    return cache.notebook("08_quadcopter")              # sets VEGETA_CACHE_DIR; monkeypatch restores it afterwards


def _model(tmp_path, name="beam", fz=-100.0):
    step = tmp_path / "beam.step"
    step.write_text("not opened: the solver is faked\n")
    return talos.StructuralModel(step, "mm-N-MPa", STEEL, regions=[talos.SurfacesOnPlane("fixed", "x", 0.0),
                                 talos.SurfacesOnPlane("tip", "x", 200.0)], supports=[talos.FixedSupport("fixed")],
                                 loads=[talos.Force("tip", fz=fz)], mesh_settings=talos.MeshSettings(8.0), name=name)


@pytest.fixture
def fake_talos(monkeypatch):
    """_mesh / _solve / _solve_modes write small files and count their calls."""
    calls = {"mesh": 0, "solve": 0, "modes": 0}

    def _files(workdir, stem, kind, metrics):
        w = Path(workdir)
        w.mkdir(parents=True, exist_ok=True)
        (w / "mesh.msh").write_text("mesh")
        (w / f"{stem}.frd").write_text(f"frd of {w.name}")
        (w / f"{stem}.dat").write_text("dat")
        (w / "case.inp").write_text("inp (not kept)")
        r = talos.Result(kind=kind, metrics=metrics)
        r.artifacts.update(mesh=w / "mesh.msh", frd=w / f"{stem}.frd", dat=w / f"{stem}.dat", inp=w / "case.inp")
        return r

    def mesh(self, workdir, progress=False):
        calls["mesh"] += 1
        w = Path(workdir)
        w.mkdir(parents=True, exist_ok=True)
        (w / "mesh.msh").write_text("mesh")
        (w / "mesh_summary.json").write_text(json.dumps({"status": "success"}))
        r = talos.Result(kind="talos.mesh", metrics={"n_nodes": 10})
        r.artifacts.update(mesh=w / "mesh.msh", mesh_summary=w / "mesh_summary.json")
        return r

    def solve(self, workdir, **kw):
        calls["solve"] += 1
        return _files(workdir, "model", "talos.solve", {"max_von_mises": 12.5, "load": self.loads[0].fz})

    def modes(self, workdir, n_modes=10, **kw):
        calls["modes"] += 1
        return _files(workdir, "modes", "talos.modes", {"frequencies_hz": [10.0, 20.0][:n_modes]})

    monkeypatch.setattr(talos.StructuralModel, "_mesh", mesh)
    monkeypatch.setattr(talos.StructuralModel, "_solve", solve)
    monkeypatch.setattr(talos.StructuralModel, "_solve_modes", modes)
    return calls


# ------------------------------------------------------------------------------------------------ the folder
def test_folder_next_to_the_notebook(tmp_path, monkeypatch):
    assert cache.notebook("08_quadcopter") == tmp_path / "08_quadcopter.cache"
    assert cache.notebook("sub/09b_durability.ipynb") == tmp_path / "sub" / "09b_durability.cache"
    assert cache.path("frame/thrust") == tmp_path / "sub" / "09b_durability.cache" / "frame" / "thrust.json"
    monkeypatch.setenv("VEGETA_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert cache.directory() == tmp_path / "elsewhere"
    with pytest.raises(ValueError):
        cache.path("../outside")


def test_detected_notebook_and_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("VEGETA_CACHE_DIR")
    monkeypatch.setenv("JPY_SESSION_NAME", str(tmp_path / "nb" / "12_boat.ipynb"))
    assert cache.directory() == tmp_path / "nb" / "12_boat.cache"
    monkeypatch.delenv("JPY_SESSION_NAME")
    with pytest.warns(UserWarning, match="no notebook set"):
        monkeypatch.setattr(talos.cache, "_warned_fallback", False)
        assert cache.directory() == tmp_path / "vegeta.cache"


def test_the_tools_carry_identical_copies_and_share_the_settings(tmp_path, monkeypatch):
    src = Path(talos.cache.__file__).read_text()
    assert Path(aeromant.cache.__file__).read_text() == src            # the tools import nothing from each other
    talos.cache.notebook("13_submarine")
    assert aeromant.cache.directory() == tmp_path / "13_submarine.cache"
    cache.disable()
    assert not talos.cache.enabled() and not aeromant.cache.enabled()
    cache.enable()
    assert aeromant.cache.enabled()


# ------------------------------------------------------------------------------------------------ talos
def test_solve_once_then_loaded_with_its_files(tmp_path, fake_talos, folder):
    m = _model(tmp_path)
    r1 = m.solve(tmp_path / "runs" / "thrust", cache="frame_thrust")
    assert fake_talos["solve"] == 1 and "cached" not in r1.metadata
    entry = folder / "frame_thrust.json"
    assert entry.is_file() and (folder / "frame_thrust.files" / "model.frd").is_file()
    assert not (folder / "frame_thrust.files" / "case.inp").exists()                 # only the files later cells read
    r2 = m.solve(tmp_path / "runs" / "thrust", cache="frame_thrust")
    assert fake_talos["solve"] == 1 and r2.metadata["cached"] == str(entry)
    assert r2.metrics == r1.metrics and r2.ok
    assert r2.artifacts["frd"] == folder / "frame_thrust.files" / "model.frd" and r2.artifacts["frd"].read_text() == "frd of thrust"
    assert r2.artifacts["inp"] == (tmp_path / "runs" / "thrust" / "case.inp").resolve()   # not copied: original path


def test_no_input_check_delete_to_run_again(tmp_path, fake_talos, folder):
    _model(tmp_path, fz=-100.0).solve(tmp_path / "w", cache="beam")
    changed = _model(tmp_path, fz=-999.0).solve(tmp_path / "w", cache="beam")        # the user's call: still loaded
    assert fake_talos["solve"] == 1 and changed.metrics["load"] == -100.0
    assert cache.clear("beam") == ["beam"] and not (folder / "beam.files").exists()
    again = _model(tmp_path, fz=-999.0).solve(tmp_path / "w", cache="beam")
    assert fake_talos["solve"] == 2 and again.metrics["load"] == -999.0


def test_without_cache_or_disabled_nothing_is_read_or_written(tmp_path, fake_talos, folder, monkeypatch):
    m = _model(tmp_path)
    m.solve(tmp_path / "w")
    m.solve(tmp_path / "w")
    assert fake_talos["solve"] == 2 and not folder.exists()
    monkeypatch.setenv("VEGETA_CACHE", "off")
    m.solve(tmp_path / "w", cache="beam")
    m.solve(tmp_path / "w", cache="beam")
    assert fake_talos["solve"] == 4 and not folder.exists()


def test_a_failed_result_is_not_saved(tmp_path, monkeypatch, folder):
    calls = []
    monkeypatch.setattr(talos.StructuralModel, "_solve",
                        lambda self, w, **kw: calls.append(1) or talos.Result(kind="talos.solve").fail("ccx died"))
    m = _model(tmp_path)
    assert not m.solve(tmp_path / "w", cache="beam").ok
    assert not m.solve(tmp_path / "w", cache="beam").ok and len(calls) == 2 and not cache.exists("beam")


def test_a_broken_entry_says_delete_it(tmp_path, fake_talos, folder):
    folder.mkdir(parents=True)
    (folder / "beam.json").write_text("{not json")
    with pytest.raises(ValueError, match="delete it"):
        _model(tmp_path).solve(tmp_path / "w", cache="beam")


def test_mesh_entry_is_restored_into_the_next_workdir(tmp_path, fake_talos, folder):
    m = _model(tmp_path)
    m.mesh(tmp_path / "runs" / "mesh", cache="frame_mesh")
    r = m.mesh(tmp_path / "other" / "mesh", cache="frame_mesh")
    assert fake_talos["mesh"] == 1 and (tmp_path / "other" / "mesh" / "mesh.msh").read_text() == "mesh"
    assert r.artifacts["mesh"] == tmp_path / "other" / "mesh" / "mesh.msh"


def test_solve_modes_and_ensure(tmp_path, fake_talos, folder):
    m = _model(tmp_path)
    a = m.solve_modes(tmp_path / "modal", n_modes=2, cache="frame_modes")
    b = m.solve_modes(tmp_path / "modal", n_modes=2, cache="frame_modes")
    assert fake_talos["modes"] == 1 and b.metrics["frequencies_hz"] == a.metrics["frequencies_hz"]
    assert b.artifacts["frd"].name == "modes.frd" and b.artifacts["frd"].is_file()
    e1 = m.ensure(tmp_path / "e", cache="beam_ensure")
    e2 = m.ensure(tmp_path / "e", run=False, cache="beam_ensure")             # loaded even with run=False
    assert e1.ok and e2.ok and e2.metadata["cached"] and fake_talos["solve"] == 1


def test_solve_models_batch(tmp_path, fake_talos, folder):
    models = [_model(tmp_path, name="a"), _model(tmp_path, name="b", fz=-5.0)]
    dirs = [tmp_path / "a", tmp_path / "b"]
    r1 = talos.solve_models(models, dirs, cache=True)
    r2 = talos.solve_models(models, dirs, cache=True)
    assert [r.ok for r in r1] == [True, True] and fake_talos["solve"] == 2
    assert all(r.metadata.get("cached") for r in r2)
    assert sorted(cache.entries()) == [f"{tmp_path.name}/a", f"{tmp_path.name}/b"]      # <parent>/<workdir>
    talos.solve_models(models, dirs, cache=[f"{tmp_path.name}/a", None])     # b without the cache: solved again
    assert fake_talos["solve"] == 3
    with pytest.raises(ValueError, match="share"):
        talos.solve_models(models, dirs, cache=["x", "x"])
    with pytest.raises(ValueError, match="one entry name"):
        talos.solve_models(models, dirs, cache=["x"])


# ------------------------------------------------------------------------------------------------ aeromant
def _case(tmp_path, name="c"):
    stl = tmp_path / "body.stl"
    stl.write_text("solid x\nendsolid x\n")
    return aeromant.CFDCase("laminar_external", stl, dict(velocity=1.0, kinematic_viscosity=0.01, density=1.0,
                            reference_area=1.0, reference_length=1.0, center_of_rotation=(0, 0, 0)),
                            workdir=tmp_path / name, geometry_units="m", environment=aeromant.OpenFOAMEnvironment())


@pytest.fixture
def fake_cfd(monkeypatch):
    calls = []

    def run(self, steps=None, **kw):
        calls.append(self.workdir.name)
        w = self.workdir
        w.mkdir(parents=True, exist_ok=True)
        (w / "coefficient.dat").write_text("# Time Cd\n100 0.42\n")
        (w / "log.solver").write_text("solver log")
        r = aeromant.Result(kind="aeromant.run", metrics={"Cd": 0.42})
        r.artifacts.update(force_coefficients=w / "coefficient.dat", log_solver=w / "log.solver", case=w)
        return r

    monkeypatch.setattr(aeromant.CFDCase, "_run", run)
    monkeypatch.setattr(aeromant.CFDCase, "_ensure", lambda self, **kw: run(self))
    return calls


def test_cfd_run_once_then_loaded(tmp_path, fake_cfd, folder):
    c = _case(tmp_path)
    c.run(cache="canopy_15ms")
    r = c.run(cache="canopy_15ms")
    assert fake_cfd == ["c"] and r.metrics["Cd"] == 0.42 and r.metadata["cached"]
    assert r.artifacts["force_coefficients"] == folder / "canopy_15ms.files" / "coefficient.dat"
    assert r.artifacts["case"] == (tmp_path / "c").resolve()                  # the case directory is not copied
    with pytest.raises(ValueError, match="whole run"):
        c.run(["blockMesh"], cache="canopy_15ms")


def test_run_cases_batch(tmp_path, fake_cfd, folder):
    cases = [_case(tmp_path, "c1"), _case(tmp_path, "c2")]
    aeromant.run_cases(cases, cache=True)
    out = aeromant.run_cases(cases, cache=True)
    assert sorted(fake_cfd) == ["c1", "c2"] and all(r.metadata.get("cached") for r in out)
    assert sorted(cache.entries()) == [f"{tmp_path.name}/c1", f"{tmp_path.name}/c2"]


# ------------------------------------------------------------------------------------------------ what can go wrong
def test_clear_deletes_only_entries_and_refuses_other_folders(tmp_path, fake_talos, folder, monkeypatch):
    m = _model(tmp_path)
    m.mesh(tmp_path / "w", cache="frame")
    m.solve(tmp_path / "w", cache="frame/thrust")                             # grouped under frame/
    assert sorted(cache.entries()) == ["frame", "frame/thrust"]               # copied files are not entries
    (folder / "notes.txt").write_text("mine")
    assert sorted(cache.clear()) == ["frame", "frame/thrust"]
    assert (folder / "notes.txt").read_text() == "mine" and sorted(p.name for p in folder.iterdir()) == ["notes.txt"]
    (tmp_path / "08.ipynb").write_text("{}")
    monkeypatch.setenv("VEGETA_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="does not end in .cache"):
        cache.clear()
    assert (tmp_path / "08.ipynb").is_file()


def test_an_entry_of_another_kind_is_refused(tmp_path, fake_talos, folder):
    m = _model(tmp_path)
    m.mesh(tmp_path / "w", cache="frame")
    with pytest.raises(ValueError, match="talos.mesh"):
        m.solve(tmp_path / "w", cache="frame")


def test_missing_copied_files_say_delete_it(tmp_path, fake_talos, folder):
    import shutil
    m = _model(tmp_path)
    m.solve(tmp_path / "w", cache="beam")
    shutil.rmtree(folder / "beam.files")
    with pytest.raises(ValueError, match="copied files are missing"):
        m.solve(tmp_path / "w", cache="beam")


def test_a_hit_says_reused_and_batch_names_are_made_safe(tmp_path, fake_talos, folder):
    m = _model(tmp_path)
    m.solve(tmp_path / "w", cache="beam")
    assert m.solve(tmp_path / "w", cache="beam").metadata["reused"] is True
    assert cache.entry_names(True, ["runs/Frame arm", "runs/a:b"]) == ["runs/Frame_arm", "runs/a_b"]
    with pytest.raises(ValueError):
        cache.entry_names(["ok", "../x"], [0, 1])                             # checked before anything runs
