"""vegeta.mock: the notebook DEBUG switch. With the mocks installed, FEA and CFD return at once in the real Result
types (ensure / solve_models / run_cases work through them), result plots are placeholders, the cache is off, and
uninstall puts everything back. No solver runs."""
from __future__ import annotations

from pathlib import Path

import pytest

from vegeta import aeromant, mock, talos
from vegeta.talos import viz as tviz

STEEL = talos.Material("steel", 210000.0, 0.3, density=7.85e-9, yield_strength=235.0)


@pytest.fixture
def mocked(monkeypatch):
    monkeypatch.delenv("VEGETA_DEBUG", raising=False)
    monkeypatch.setenv("VEGETA_CACHE", "on")
    real = talos.StructuralModel.solve
    mock.install()
    yield real
    mock.uninstall()
    assert talos.StructuralModel.solve is real


def _model(tmp_path, name="beam"):
    step = tmp_path / "beam.step"
    step.write_text("never opened\n")
    return talos.StructuralModel(step, "mm-N-MPa", STEEL, regions=[talos.SurfacesOnPlane("fixed", "x", 0.0),
                                 talos.SurfacesOnPlane("tip", "x", 200.0)], supports=[talos.FixedSupport("fixed")],
                                 loads=[talos.Force("tip", fz=-100.0)], mesh_settings=talos.MeshSettings(8.0),
                                 masses=[talos.PointMass("tip", 1e-4)], name=name)


def test_switch(monkeypatch):
    monkeypatch.delenv("VEGETA_DEBUG", raising=False)
    assert mock.debug() is False and mock.debug(True) is True
    monkeypatch.setenv("VEGETA_DEBUG", "1")
    assert mock.debug(False) is True
    monkeypatch.setenv("VEGETA_DEBUG", "0")
    assert mock.debug(True) is False


def test_fea_through_the_mocks(tmp_path, mocked):
    import os
    assert os.environ["VEGETA_CACHE"] == "off"
    m = _model(tmp_path)
    assert m.mesh(tmp_path / "w").ok and m.mesh_is_current(tmp_path / "w")      # a mesh summary the real code accepts
    r = m.solve(tmp_path / "w")
    assert r.ok and r.metrics["max_von_mises"] == 12.0 and r.metrics["anything_else"] == 1.0
    assert m.solved(tmp_path / "w") is not None                                  # ensure reads it back
    e = m.ensure(tmp_path / "e")
    rs = talos.solve_models([m, _model(tmp_path, "b")], [tmp_path / "a", tmp_path / "b"])
    assert e.ok and all(x.ok for x in rs)
    modes = m.solve_modes(tmp_path / "w", n_modes=8)
    assert len(modes.metrics["frequencies_hz"]) == 8 and modes.metrics["point_mass_total"] == pytest.approx(1e-4)
    fr = talos.read_frd(r.artifacts["frd"])
    assert fr.von_mises.shape == (mock.N_NODES,)
    p = tviz.plot_results(r)
    p.add_text("anything")
    assert tviz.show(p) is None


def test_fatigue_through_the_mocks(tmp_path, mocked):
    from vegeta import chronos
    m = _model(tmp_path)
    m.mesh(tmp_path / "w")
    unit = {"lift": (m.solve(tmp_path / "w"), 1.0)}
    spec = chronos.build_spectrum(chronos.Mission("m", (chronos.Segment("s", 60, {"lift": 1.2}),)), None).to_dict()
    f = talos.assess_fatigue(unit, spec, talos.FatigueCurve("x", 50.0, -0.1))
    assert f.result.ok and f.hotspot == mock.N_NODES - 1


def test_cfd_through_the_mocks(tmp_path, mocked):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_sphere", Path(__file__).parents[1] / "aeromant" / "_sphere.py")
    sphere = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sphere)
    stl = sphere.write_stl_ascii(sphere.icosphere(), tmp_path / "sphere.stl")
    case = aeromant.CFDCase("laminar_external", stl, dict(velocity=7.0, kinematic_viscosity=0.01, density=1.0,
                            reference_area=1.0, reference_length=1.0, center_of_rotation=(0, 0, 0)),
                            workdir=tmp_path / "c", geometry_units="m", environment=aeromant.OpenFOAMEnvironment())
    r = case.run()
    assert r.ok and r.metrics["Cd"] == 0.03 and case.ensure().ok
    assert all(x.ok for x in aeromant.run_cases([case]))
    from vegeta.aeromant import movie
    u, valid = movie.openfoam_sampler(case)([[0, 1, 0], [1, 0, 1], [0, 0.05, 0], [0, 0.05 / 2 ** 0.5, 0.05 / 2 ** 0.5]])
    assert u[0, 0] == pytest.approx(7.0) and u[1, 0] == pytest.approx(7.0) and valid.all()   # the free stream
    assert u[2, 0] < 7.0 and u[2, 0] != pytest.approx(u[3, 0])           # a wake near the axis, not the same all round


def test_slicing_and_suction_through_the_mocks(tmp_path, mocked):
    from vegeta import mellonia
    from vegeta.mellonia import Orientation, examples
    stl = tmp_path / "part.stl"
    stl.write_text("solid x\nendsolid x\n")
    settings = examples.GENERIC_PLA_0_2MM
    ok = mellonia.slice_stl(stl, settings, Orientation(), tmp_path / "ok")
    assert ok.ok and ok.metrics["layer_count"] >= 20
    typo = mellonia.slice_stl(stl, settings.replace(print={"layer_heigth": 0.3}), Orientation(), tmp_path / "typo")
    assert not typo.ok and "layer_heigth" in typo.messages[0]
    m = mock._template_metrics(mock.Metrics(), {"flow_rate": 0.2})
    assert m["fan_static_pressure_Pa"] > 0 and m["air_power_W"] == pytest.approx(0.2 * m["fan_static_pressure_Pa"])
