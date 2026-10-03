"""Mocks and stubs for the long-running solvers, so every assembly can be tested end to end in seconds.

Two pieces, used together:

- **Mocks** replace the solver *run functions* every workflow goes through: ``talos.solve_models`` (CalculiX),
  ``aeromant.run_cases`` (OpenFOAM) and ``run_scene`` (the MuJoCo episodes of the ground workflows), plus the OpenFOAM
  installation lookup. They are ``unittest.mock`` objects made with ``create_autospec`` from the real functions, so a
  call with a wrong signature fails, and every call is recorded: ``solvers.solve_models.call_args_list`` holds the
  real ``StructuralModel`` objects a workflow built, ``solvers.run_cases`` the real ``CFDCase`` objects.
- **Stubs** are what the mocks answer: ``fea_result(model, workdir)``, ``cfd_result(case)``, ``scene_result(node)`` —
  fixed, plausible numbers in the same ``Result`` types and metric names the real solvers give. A test can swap one
  (``solvers.solve_models.side_effect = ...``) to give a failed case, a low safety factor, an odd force.

``run=False`` still gives NOT RUN, as the real functions do. ``tests/conftest.py`` installs all of it for every test
(the ``solvers`` fixture, autouse); a test that must run the real tools is marked ``@pytest.mark.real_solvers`` (those
are also ``slow``: ``./assembly_tests.sh slow``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from vegeta import aeromant, talos

# ------------------------------------------------------------------------------------------------ the stubs
FEA_METRICS = {"max_von_mises": 12.0, "safety_factor_yield": 4.0, "safety_factor_temperature": 3.0, "max_displacement": 0.2,
               "n_elements": 1000, "frequencies_hz": [45.0, 110.0, 190.0, 320.0]}
CFD_METRICS = {"drag_force_N": 5.0, "lift_force_N": 20.0, "Cd": 0.03, "Cl": 0.4, "thrust_N": 5.0, "torque_Nm": 0.1,
               "power_W": 60.0, "efficiency": 0.6, "figure_of_merit": 0.6, "mach_number": 0.3, "converged": True,
               "mesh_cells": 1000, "jet_excess_T_at_1m_K": 60.0,
               "plume": {"distance_m": [0.1, 0.5], "excess_temperature_K": [600.0, 120.0], "exhaust_fraction": [1.0, 0.2]}}


def not_run(kind: str):
    return (talos if kind == "talos" else aeromant).Result(kind=f"{kind}.run").fail("NOT RUN")


def fea_result(model, workdir) -> talos.Result:
    """A solved structural model: the workdir is made (with a STUB marker) and the usual metrics returned."""
    Path(workdir).mkdir(parents=True, exist_ok=True)
    (Path(workdir) / "STUB").write_text("solved by assemblies/tests/stubs.py\n")
    return talos.Result(kind="talos.results", metrics=dict(FEA_METRICS), metadata={"stub": True})


def speedline_point(parameters: dict) -> dict:
    """A compressor point that falls along a speed line: less flow and more pressure ratio at higher back pressure."""
    pr = parameters["outlet_pressure"] / parameters["inlet_total_pressure"]
    flow = max(0.05, 0.40 - 0.08 * pr)
    return {"mass_flow_kg_s": flow, "corrected_mass_flow_kg_s": flow, "pressure_ratio_tt": 1.05 * pr, "efficiency_tt": 0.72,
            "efficiency_from_torque": 0.70, "work_coefficient": 0.6, "shaft_power_W": 1000.0 * flow * 1.05 * pr * 40.0}


def cfd_result(case) -> aeromant.Result:
    """A solved CFD case: the workdir is made and the forces, coefficients and rotor numbers returned (a compressor
    case also gets its speed-line point)."""
    Path(case.workdir).mkdir(parents=True, exist_ok=True)
    metrics = dict(CFD_METRICS)
    p = case.user_parameters
    if "outlet_pressure" in p and "inlet_total_pressure" in p:
        metrics.update(speedline_point(p))
    return aeromant.Result(kind="aeromant.results", metrics=metrics, metadata={"stub": True, "reused": False})


def scene_result(node) -> dict:
    """A simulated episode's results (no MuJoCo)."""
    return {"stub": True, "events": [], "phases": {}}


# ------------------------------------------------------------------------------------------------ the mocks' behaviour
def _solve_models(models, workdirs, *, threads=1, run=True, executable="ccx", timeout=None, progress=False, cancel=None):
    return [fea_result(m, w) if run else not_run("talos") for m, w in zip(models, workdirs)]


def _run_cases(cases, *, jobs=1, processors=1, run=True, progress=False, cancel=None, timeout=None):
    return [cfd_result(c) if run else not_run("aeromant") for c in cases]


def _run_scene(node, episode, *, run):
    if node.results:
        return node
    if not run:
        return node.not_run("run_sim=False")
    return node.record(**scene_result(node))


@dataclass
class Solvers:
    """The installed mocks. ``fea``/``cfd``/``sim`` list what was actually solved (calls with ``run=True``)."""
    solve_models: mock.MagicMock
    run_cases: mock.MagicMock
    run_scene: mock.MagicMock

    @property
    def fea(self) -> list[str]:
        return [m.name for c in self.solve_models.call_args_list if c.kwargs.get("run", True) for m in c.args[0]]

    @property
    def cfd(self) -> list[str]:
        return [k.template.name for c in self.run_cases.call_args_list if c.kwargs.get("run", True) for k in c.args[0]]

    @property
    def sim(self) -> list[str]:
        return [c.args[0].name for c in self.run_scene.call_args_list if c.kwargs.get("run") and not c.args[0].meta.get("reused")]


def install(monkeypatch) -> Solvers:
    """Put the mocks in place (undone by ``monkeypatch`` after the test)."""
    import assemblies.workflows._scene as scene
    from assemblies.components import impeller

    solvers = Solvers(solve_models=mock.create_autospec(talos.solve_models, side_effect=_solve_models),
                      run_cases=mock.create_autospec(aeromant.run_cases, side_effect=_run_cases),
                      run_scene=mock.create_autospec(scene.run_scene, side_effect=_run_scene))
    monkeypatch.setattr(talos, "solve_models", solvers.solve_models)
    monkeypatch.setattr(aeromant, "run_cases", solvers.run_cases)
    real_env, real_scene = impeller.environment, scene.run_scene
    no_openfoam = lambda run=True: aeromant.OpenFOAMEnvironment()  # noqa: E731
    for name, mod in list(sys.modules.items()):
        if not name.startswith("assemblies."):
            continue
        if getattr(mod, "environment", None) is real_env:
            monkeypatch.setattr(mod, "environment", no_openfoam)
        if getattr(mod, "run_scene", None) is real_scene:
            monkeypatch.setattr(mod, "run_scene", solvers.run_scene)
    return solvers
