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
