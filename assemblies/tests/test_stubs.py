"""The mock layer itself (``stubs.py``): the solver run functions are replaced in every test, keep their signatures,
record their calls, answer with the stubs, and give NOT RUN for ``run=False``."""
from __future__ import annotations

import pytest
from vegeta import aeromant, talos

import assemblies.workflows._common as common
import assemblies.workflows._scene as scene
import stubs
from assemblies.vida import Assembly


class _Model:
    name = "plate"


class _Case:
    class template:
        name = "rans_ksst_external"

    def __init__(self, workdir, **parameters):
        self.workdir, self.user_parameters = workdir, parameters


def test_the_run_functions_are_mocks(solvers):
    assert talos.solve_models is solvers.solve_models and common.talos.solve_models is solvers.solve_models
    assert aeromant.run_cases is solvers.run_cases
    from assemblies.workflows import onager, walkers
    assert onager.run_scene is solvers.run_scene and walkers.run_scene is solvers.run_scene
    with pytest.raises(TypeError):                                 # autospec: the real signature is enforced
        talos.solve_models([_Model()], ["w"], no_such_option=1)


def test_fea_mock_answers_with_the_stub_and_records(tmp_path, solvers):
    rs = talos.solve_models([_Model()], [tmp_path / "plate"], threads=2)
    assert rs[0].ok and rs[0].metrics["safety_factor_yield"] == stubs.FEA_METRICS["safety_factor_yield"]
    assert (tmp_path / "plate" / "STUB").is_file()
    assert solvers.solve_models.call_count == 1 and solvers.solve_models.call_args.kwargs["threads"] == 2
    assert solvers.fea == ["plate"]
    off = talos.solve_models([_Model()], [tmp_path / "again"], run=False)
    assert not off[0].ok and solvers.fea == ["plate"]                       # run=False: NOT RUN, nothing solved


def test_cfd_mock_answers_with_the_stub_and_records(tmp_path, solvers):
    r = aeromant.run_cases([_Case(tmp_path / "c")])[0]
    assert r.ok and r.metrics["Cd"] == stubs.CFD_METRICS["Cd"] and solvers.cfd == ["rans_ksst_external"]
    p = aeromant.run_cases([_Case(tmp_path / "p", outlet_pressure=2e5, inlet_total_pressure=1e5)])[0]
    assert p.metrics["pressure_ratio_tt"] == pytest.approx(2.1)
    assert not aeromant.run_cases([_Case(tmp_path / "n")], run=False)[0].ok


def test_a_test_can_swap_a_stub(tmp_path, solvers):
    solvers.solve_models.side_effect = lambda models, workdirs, **kw: [talos.Result(kind="talos.results").fail("diverged")
                                                                       for _ in models]
    assert not talos.solve_models([_Model()], [tmp_path / "x"])[0].ok


def test_scene_mock(solvers):
    node = Assembly("scene", "episode")
    scene_fn = __import__("assemblies.workflows.onager", fromlist=["run_scene"]).run_scene
    scene_fn(node, lambda: pytest.fail("the real episode must not run"), run=True)
    assert node.results["stub"] and solvers.sim == ["scene"]
    off = Assembly("scene2", "episode")
    scene_fn(off, lambda: None, run=False)
    assert off.status() == "NOT RUN"


@pytest.mark.real_solvers
def test_real_solvers_marker_leaves_the_real_functions(solvers):
    assert solvers is None
    assert not isinstance(talos.solve_models, type(stubs.mock.MagicMock())) and scene.run_scene.__module__ == scene.__name__


def _plate(tmp_path, masses=True):
    """A real StructuralModel (nothing is meshed or solved: the mocks answer)."""
    step = tmp_path / "plate.step"
    step.write_text("not a real STEP: the mocks never open it\n")
    mat = talos.Material("PLA", 3500.0, 0.35, density=1.24e-9, yield_strength=50.0)
    regions = [talos.SurfacesOnPlane("root", "x", 0.0), talos.SurfacesOnPlane("tip", "x", 100.0)]
    return talos.StructuralModel(step, "mm-N-MPa", mat, regions, [talos.FixedSupport("root")],
                                 [talos.Force("tip", fz=1.0)], talos.MeshSettings(5.0), name="plate",
                                 masses=[talos.PointMass("tip", 1e-4)] if masses else ())


def test_mesh_and_modes_mocks_bind_self_and_record(tmp_path, solvers):
    m = _plate(tmp_path)
    assert m.mesh(tmp_path / "mesh").ok and (tmp_path / "mesh" / "mesh.msh").is_file()
    r = m.solve_modes(tmp_path / "modal", n_modes=6)
    assert r.metrics["frequencies_hz"] == stubs.MODES_HZ[:6] and r.metrics["point_mass_total"] == pytest.approx(1e-4)
    assert solvers.mesh.call_args.args[0] is m and solvers.solve_modes.call_args.args[0] is m
    with pytest.raises(TypeError):                                 # autospec: the real signature is enforced
        m.solve_modes(tmp_path / "x", 6, no_such_option=True)
    assert stubs.check_calls(solvers)["modal"] == 1


def test_fatigue_and_frd_mocks(tmp_path, solvers):
    from vegeta import chronos
    unit = {"lift": (talos.solve_models([_plate(tmp_path)], [tmp_path / "lift"])[0], 1.0)}
    mission = chronos.Mission("m", (chronos.Segment("cruise", 60, {"lift": 1.0}),
                                    chronos.Segment("gusts", 1, {"lift": 1.6}, repeat=10)))
    spec = chronos.build_spectrum(mission, None).to_dict()
    f = talos.assess_fatigue(unit, spec, talos.FatigueCurve("PLA", 50.0, -0.1), workdir=tmp_path / "fat")
    assert isinstance(f, talos.FatigueResult) and f.result.ok and f.hotspot == stubs.N_NODES - 1
    assert f.result.metrics["damage_per_pass"] > 0 and (tmp_path / "fat").is_dir()
    vm = talos.read_frd(tmp_path / "lift" / "model.frd").von_mises
    assert vm[-1] == pytest.approx(12.0)
    assert stubs.check_calls(solvers)["fatigue"] == 1
    other = chronos.Mission("x", (chronos.Segment("climb", 10, {"thrust": 2.0}),))     # a pattern without a unit case
    assert not talos.assess_fatigue(unit, chronos.build_spectrum(other, None).to_dict(), talos.FatigueCurve("PLA", 50.0, -0.1)).result.ok
    with pytest.raises(AssertionError, match="every spectrum pattern has a unit case"):
        stubs.check_calls(solvers)
