"""Mocks and stubs for the long-running solvers, so every assembly can be tested end to end in seconds.

Two pieces, used together:

- **Mocks** replace the solver *run functions* every workflow goes through: ``talos.solve_models`` (CalculiX),
  ``StructuralModel.mesh`` (Gmsh) and ``StructuralModel.solve_modes`` (CalculiX, modal), ``talos.assess_fatigue`` and
  ``talos.read_frd`` (they read solved ``.frd`` files), ``aeromant.run_cases`` (OpenFOAM) and ``run_scene`` (the MuJoCo
  episodes of the ground workflows), plus the OpenFOAM installation lookup. They are ``unittest.mock`` objects made
  with ``create_autospec`` from the real functions, so a call with a wrong signature fails, and every call is
  recorded: ``solvers.solve_models.call_args_list`` holds the real ``StructuralModel`` objects a workflow built,
  ``solvers.run_cases`` the real ``CFDCase`` objects.
- **Stubs** are what the mocks answer: ``fea_result``, ``mesh_result``, ``modes_result``, ``fatigue_result``,
  ``frd_fields``, ``cfd_result``, ``scene_result`` — fixed, plausible numbers in the same ``Result`` types and metric
  names the real solvers give. A test can swap one (``solvers.solve_models.side_effect = ...``) to give a failed case,
  a low safety factor, an odd force.

``run=False`` still gives NOT RUN, as the real functions do. ``tests/conftest.py`` installs all of it for every test
(the ``solvers`` fixture, autouse); a test that must run the real tools is marked ``@pytest.mark.real_solvers`` (those
are also ``slow``: ``./assembly_tests.sh slow``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import numpy as np
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
    w = Path(workdir)
    return talos.Result(kind="talos.results", metrics=dict(FEA_METRICS), metadata={"stub": True},
                        artifacts={"mesh": w / "mesh.msh", "frd": w / "model.frd", "summary": w / "summary.json"})


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


MODES_HZ = [45.0, 110.0, 190.0, 320.0, 480.0, 650.0, 870.0, 1100.0, 1400.0, 1750.0]
N_NODES = 10


def mesh_result(model, workdir) -> talos.Result:
    """A meshed model: the workdir with a ``mesh.msh`` marker (so a copy of the directory per case works as usual)."""
    Path(workdir).mkdir(parents=True, exist_ok=True)
    (Path(workdir) / "mesh.msh").write_text("meshed by assemblies/tests/stubs.py\n")
    return talos.Result(kind="talos.mesh", metrics={"n_nodes": N_NODES, "n_elements": 1000}, metadata={"stub": True},
                        artifacts={"mesh": Path(workdir) / "mesh.msh"})


def modes_result(model, workdir, n_modes) -> talos.Result:
    """A modal solve: the first ``n_modes`` of ``MODES_HZ``, the point masses' total."""
    Path(workdir).mkdir(parents=True, exist_ok=True)
    f = MODES_HZ[:n_modes]
    return talos.Result(kind="talos.modes", metadata={"stub": True},
                        metrics={"n_nodes": N_NODES, "n_elements": 1000, "n_modes": len(f), "frequencies_hz": list(f),
                                 "first_frequency_hz": f[0], "point_mass_total": float(sum(m.mass for m in model.masses))},
                        artifacts={"frd": Path(workdir) / "modes.frd"})


def frd_fields(path) -> talos.FieldResults:
    """An ``.frd`` read: ``N_NODES`` nodes along x, a uniaxial stress rising to 12 MPa at the last node."""
    ids = np.arange(1, N_NODES + 1)
    coords = np.column_stack([np.linspace(0.0, 90.0, N_NODES), np.zeros(N_NODES), np.zeros(N_NODES)])
    stress = np.zeros((N_NODES, 6))
    stress[:, 0] = np.linspace(1.2, 12.0, N_NODES)
    return talos.FieldResults(ids, coords, {"STRESS": stress, "DISP": np.zeros((N_NODES, 3))})


def fatigue_result(unit_cases, spectrum, curve, workdir=None) -> talos.FatigueResult:
    """A fatigue assessment in the real type: damage grows with the spectrum (sum of amplitude x cycles), so missions
    differ; the hotspot is the last node; a pattern without a unit case fails, as the real one does."""
    blocks = spectrum["blocks"]
    missing = sorted({b["pattern"] for b in blocks} - set(unit_cases))
    fr = frd_fields(None)
    if missing:
        r = talos.Result(kind="talos.fatigue").fail(f"no unit case for pattern(s) {missing}")
        return talos.FatigueResult(fr.node_ids, fr.coords, np.zeros(N_NODES), {}, r)
    load = sum(abs(b["amplitude"]) * b["cycles"] for b in blocks) * 1e-9 + 1e-12
    damage = load * np.linspace(0.1, 1.0, N_NODES)
    hot = int(np.argmax(damage))
    hours = spectrum["duration_s"] / 3600.0
    d = float(damage[hot])
    r = talos.Result(kind="talos.fatigue", metadata={"stub": True},
                     metrics={"damage_per_pass": d, "passes_to_failure": 1.0 / d, "hours_to_failure": hours / d,
                              "spectrum_duration_h": hours, "blocks": len(blocks), "hotspot_node": int(fr.node_ids[hot]),
                              "hotspot_location": fr.coords[hot].tolist(), "nodes_above_1pct": int((damage > 0.01 * d).sum())})
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    total = sum(abs(b["amplitude"]) * b["cycles"] for b in blocks) or 1.0
    contributions = {b.get("source") or b["pattern"]: d * abs(b["amplitude"]) * b["cycles"] / total for b in blocks}
    return talos.FatigueResult(fr.node_ids, fr.coords, damage, contributions, r)


def scene_result(node) -> dict:
    """A simulated episode's results (no MuJoCo)."""
    return {"stub": True, "events": [], "phases": {}}


# ------------------------------------------------------------------------------------------------ the mocks' behaviour
def _solve_models(models, workdirs, *, threads=1, run=True, executable="ccx", timeout=None, progress=False, cancel=None):
    return [fea_result(m, w) if run else not_run("talos") for m, w in zip(models, workdirs)]


def _run_cases(cases, *, jobs=1, processors=1, run=True, progress=False, cancel=None, timeout=None):
    return [cfd_result(c) if run else not_run("aeromant") for c in cases]


def _mesh(self, workdir, progress=False):
    return mesh_result(self, workdir)


def _solve_modes(self, workdir, n_modes=10, *, executable="ccx", threads=1, timeout=None, progress=False, cancel=None):
    return modes_result(self, workdir, n_modes)


def _assess_fatigue(unit_cases, spectrum, curve, *, workdir=None):
    return fatigue_result(unit_cases, spectrum, curve, workdir)


def _read_frd(path):
    return frd_fields(path)


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
    mesh: mock.MagicMock = field(default=None)            # StructuralModel.mesh(self, workdir, progress)
    solve_modes: mock.MagicMock = field(default=None)     # StructuralModel.solve_modes(self, workdir, n_modes, ...)
    assess_fatigue: mock.MagicMock = field(default=None)  # talos.assess_fatigue(unit_cases, spectrum, curve, workdir=)
    read_frd: mock.MagicMock = field(default=None)        # talos.read_frd(path)

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

    real_mesh, real_modes = talos.StructuralModel.mesh, talos.StructuralModel.solve_modes
    solvers = Solvers(solve_models=mock.create_autospec(talos.solve_models, side_effect=_solve_models),
                      run_cases=mock.create_autospec(aeromant.run_cases, side_effect=_run_cases),
                      run_scene=mock.create_autospec(scene.run_scene, side_effect=_run_scene),
                      mesh=mock.create_autospec(real_mesh, side_effect=_mesh),
                      solve_modes=mock.create_autospec(real_modes, side_effect=_solve_modes),
                      assess_fatigue=mock.create_autospec(talos.assess_fatigue, side_effect=_assess_fatigue),
                      read_frd=mock.create_autospec(talos.read_frd, side_effect=_read_frd))
    monkeypatch.setattr(talos, "solve_models", solvers.solve_models)
    monkeypatch.setattr(aeromant, "run_cases", solvers.run_cases)
    monkeypatch.setattr(talos, "assess_fatigue", solvers.assess_fatigue)
    monkeypatch.setattr(talos, "read_frd", solvers.read_frd)
    # methods: a real function on the class hands ``self`` to the autospecced mock (a mock on a class does not bind)
    monkeypatch.setattr(talos.StructuralModel, "mesh",
                        lambda self, workdir, progress=False: solvers.mesh(self, workdir, progress=progress))
    monkeypatch.setattr(talos.StructuralModel, "solve_modes",
                        lambda self, workdir, n_modes=10, **kw: solvers.solve_modes(self, workdir, n_modes, **kw))
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


# ------------------------------------------------------------------------------------------------ API compatibility
def check_calls(solvers: Solvers, *, progress=None) -> dict:
    """What a workflow handed the mocked solvers must be what the real ones take: real ``StructuralModel`` /
    ``CFDCase`` objects whose ``key`` computes (their configuration is complete), one distinct workdir per FEA model,
    a callable episode per scene. With ``progress`` given, every call must pass the workflow's ``progress`` through
    (a long calculation shows its bar when asked, and none in tests). Returns the counts checked."""
    n = {"fea_calls": 0, "models": 0, "cfd_calls": 0, "cases": 0, "scenes": 0}
    for c in solvers.solve_models.call_args_list:
        models, workdirs = c.args[0], c.args[1]
        assert len(models) == len(workdirs), "one workdir per model"
        assert len({Path(w).resolve() for w in workdirs}) == len(workdirs), "two FEA models share a workdir"
        for m in models:
            assert isinstance(m, talos.StructuralModel), f"not a StructuralModel: {m!r}"
            assert isinstance(m.key, str) and m.key
        if progress is not None:
            assert c.kwargs.get("progress") is progress, f"solve_models called with progress={c.kwargs.get('progress')!r}"
        n["fea_calls"] += 1
        n["models"] += len(models)
    for c in solvers.run_cases.call_args_list:
        for case in c.args[0]:
            assert isinstance(case, aeromant.CFDCase), f"not a CFDCase: {case!r}"
            assert case.template.name and isinstance(case.key, str) and case.key
        if progress is not None:
            assert c.kwargs.get("progress") is progress, f"run_cases called with progress={c.kwargs.get('progress')!r}"
        n["cfd_calls"] += 1
        n["cases"] += len(c.args[0])
    for c in solvers.run_scene.call_args_list:
        assert callable(c.args[1]) and "run" in c.kwargs
        n["scenes"] += 1
    n.update(meshes=0, modal=0, fatigue=0)
    for c in solvers.mesh.call_args_list:
        assert isinstance(c.args[0], talos.StructuralModel)
        n["meshes"] += 1
    for c in solvers.solve_modes.call_args_list:
        m = c.args[0]
        assert isinstance(m, talos.StructuralModel) and m.material.density, "a modal model needs a density"
        assert int(c.args[2]) > 0
        n["modal"] += 1
    for c in solvers.assess_fatigue.call_args_list:
        unit_cases, spectrum, curve = c.args[:3]
        assert isinstance(curve, talos.FatigueCurve) and isinstance(spectrum, dict) and spectrum["blocks"]
        for name, (case, load) in unit_cases.items():
            assert isinstance(case, talos.Result) and case.ok and load != 0, f"unit case {name}: (ok Result, load != 0)"
            assert "frd" in case.artifacts, f"unit case {name}: assess_fatigue reads its .frd"
        assert {b["pattern"] for b in spectrum["blocks"]} <= set(unit_cases), "every spectrum pattern has a unit case"
        n["fatigue"] += 1
    return n
