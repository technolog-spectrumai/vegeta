"""The data notebooks 28 and 29 compute that the microjet and AGUYA workflows now record on their roots (audit group
jet: jet-1 the drag build-up, jet-2 the altitude / hot-day table, jet-4 the three catalogue classes). Each lifted
function is checked against the notebook cell itself, read out of the .ipynb and run as it is over the same inputs
(plots go to a recording stand-in for matplotlib); then the new keys are looked for in a workflow run. The CAD and
solver steps of those runs are replaced by no-ops: what is checked is the root's recording, in seconds."""
from __future__ import annotations

import ast
import json
import math
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from vegeta.boreas import microjet as mj

pytest.importorskip("cadquery")
from assemblies import DATA, results, vida                                         # noqa: E402
from assemblies.components import aguya as ag, aguya_flight as F, cycle            # noqa: E402
from assemblies.workflows import aguya as wa, microjet as wm                       # noqa: E402

NB = Path(__file__).resolve().parents[2] / "notebooks"
NB28, NB29 = "28_microjet.ipynb", "29_aguya.ipynb"
ENGINE_CLASS = "140 N class"
CLASS_COLOR = {"100 N class": "#1f77b4", "140 N class": "#2ca02c", "200 N class": "#9467bd"}     # notebook 28 cell 2


def cell(notebook: str, index: int, names=None, **globals_) -> dict:
    """Run notebook ``notebook`` cell ``index`` over ``globals_``: the whole cell, or only its function definitions and
    assignments to ``names``; returns the namespace (as ``test_life.cell``)."""
    src = "".join(json.loads((NB / notebook).read_text())["cells"][index]["source"])
    tree = ast.parse(src)
    ns = {"math": math, "np": np, "pd": pd, "mj": mj, **globals_}
    if names is None:
        body = tree.body
    else:
        def assigned(node):
            ts = node.targets if isinstance(node, ast.Assign) else []
            out = set()
            for t in ts:
                out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
            return out
        body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
                or (isinstance(n, ast.Assign) and assigned(n) and assigned(n) <= set(names))]
    exec(compile(ast.Module(body, []), f"{notebook}:{index}", "exec"), ns)
    return ns


def recording_plt(n_axes: int):
    """A stand-in for ``matplotlib.pyplot``: ``plt.subplots`` gives a figure and ``n_axes`` mock axes that record what
    the cell plots (``axes[i].plot.call_args_list``)."""
    axes = [mock.MagicMock(name=f"ax{i}") for i in range(n_axes)]
    plt = mock.MagicMock(name="plt")
    plt.subplots.return_value = (mock.MagicMock(name="fig"), axes)
    return plt, axes


@pytest.fixture(scope="module")
def engines():
    return {k: mj.from_catalogue(k) for k in mj.CATALOGUE}


# ------------------------------------------------------------------------------------------------ jet-2 (notebook 28)
CELL10 = {"density [kg/m³]": "density_kg_m3", "static thrust [N]": "static_thrust_N",
          "thrust at 150 m/s [N]": "thrust_150_m_s_N", "fuel at 150 m/s [g/min]": "fuel_150_m_s_g_min", "EGT [K]": "egt_K"}


@pytest.mark.parametrize("engine_class", ["140 N class", "200 N class"])
def test_atmosphere_cases_are_cell_28_10(engines, engine_class):
    e = engines[engine_class]
    ns = cell(NB28, 10, ENGINE=e)
    got = wm.atmosphere_cases(e)
    assert list(got) == list(ns["rows"]) == list(wm.ATMOSPHERES)               # the notebook's four atmospheres, in order
    for name, row in ns["rows"].items():
        assert ns["cases"][name] == mj.isa(*wm.ATMOSPHERES[name])
        assert {CELL10[k]: v for k, v in row.items()} == pytest.approx({k: got[name][k] for k in CELL10.values()}, rel=1e-12)
        assert (got[name]["altitude_m"], got[name]["delta_isa_K"]) == wm.ATMOSPHERES[name]
    if engine_class == "140 N class":                                           # the audit's numbers
        assert got["ISA sea level"]["static_thrust_N"] == pytest.approx(142.0, abs=0.05)
        assert got["1500 m, ISA+20 (≈ 25 °C)"]["thrust_150_m_s_N"] == pytest.approx(92.0, abs=0.05)


# ------------------------------------------------------------------------------------------------ jet-4 (notebook 28)
CELL4 = {"thrust [N]": "thrust_N", "fuel [g/min]": "fuel_g_min", "air flow [kg/s]": "mass_flow_kg_s",
         "pressure ratio": "pressure_ratio", "TIT [K]": "tit_K", "EGT [K]": "egt_K", "jet speed [m/s]": "jet_velocity_m_s",
         "impeller tip [m/s]": "tip_speed_m_s", "tip Mach": "tip_mach", "burner efficiency (fitted)": "burner_efficiency",
         "TSFC [kg/(N h)]": "tsfc_kg_N_h", "nozzle choked": "nozzle_choked", "mass + system [kg]": "mass_kg"}


def test_catalogue_classes_are_cell_28_4():
    ns = cell(NB28, 4)                                                          # the whole cell: ENGINES, cycle_row, table
    got = wm.catalogue_classes(ns["ENGINES"])
    assert list(got) == list(mj.CATALOGUE) and ns["ENGINE_CLASS"] == wm.DEFAULT_ENGINE
    for k, e in ns["ENGINES"].items():
        row = ns["cycle_row"](e)
        assert set(row) == set(CELL4) and set(got[k]) == set(CELL4.values())
        assert {CELL4[c]: v for c, v in row.items()} == pytest.approx(got[k], rel=1e-12)
    own = wm.catalogue_classes({k: cycle.from_catalogue(k) for k in mj.CATALOGUE})   # what run() hands it: the same
    assert all(own[k] == pytest.approx(got[k], rel=1e-12) for k in mj.CATALOGUE)


def test_catalogue_full_throttle_is_cell_28_8(engines):
    plt, ax = recording_plt(3)
    ns = cell(NB28, 8, plt=plt, ENGINES=engines, ENGINE=engines[ENGINE_CLASS], ENGINE_CLASS=ENGINE_CLASS,
              CLASS_COLOR=CLASS_COLOR)
    np.testing.assert_array_equal(ns["V"], cycle.V_MAP)                         # the export's airspeed axis
    got = wm.catalogue_full_throttle(engines)
    assert list(got) == list(mj.CATALOGUE)
    plotted = {(i, c.kwargs["label"]): c.args for i in range(3) for c in ax[i].plot.call_args_list}
    for k in mj.CATALOGUE:
        np.testing.assert_array_equal(got[k]["V_m_s"], plotted[(0, k)][0])
        np.testing.assert_allclose(got[k]["thrust_N"], plotted[(0, k)][1], rtol=1e-12)
        np.testing.assert_allclose(got[k]["tsfc_kg_N_h"], plotted[(1, k)][1], rtol=1e-12)
        e = engines[k]
        np.testing.assert_allclose(got[k]["fuel_g_min"], [mj.solve(e, e.rpm_max, v).fuel_flow_g_min for v in ns["V"]],
                                   rtol=1e-12)
    eff = got[ENGINE_CLASS]                                                     # the third plot: the chosen class
    np.testing.assert_allclose(eff["overall_efficiency"] * 100, plotted[(2, "overall (thrust power / fuel)")][1], rtol=1e-12)
    np.testing.assert_allclose(eff["propulsive_efficiency"] * 100, plotted[(2, "propulsive 2V/(V+Vj)")][1], rtol=1e-12)


# ------------------------------------------------------------------------------------------------ jet-1 (notebook 29)
def test_drag_buildup_is_cell_29_8(engines):
    engine = engines[ENGINE_CLASS]
    unit, P0 = F.jet_unit(engine), ag.for_engine(engine)
    plt, _ = recording_plt(2)
    shown = []
    ns = cell(NB29, 8, plt=plt, display=shown.append, P0=P0, UNIT=unit, BUDGET=dict(F.DRY_MASS_KG), MISSION=wa.MISSION,
              ag=ag, F=F, RHO=1.225, G=9.80665, A0=340.3, AGUYA_COLOR="#2ca02c")
    got = wa.drag_buildup(P0)
    b = ns["b"]
    assert got["parts_m2"] == pytest.approx(b["parts_m2"], rel=1e-12) and list(got["parts_m2"]) == list(b["parts_m2"])
    table = shown[0]["Cd·A [cm²]"]                                              # the table the cell displays
    pd.testing.assert_series_equal(pd.Series(got["parts_cm2"]).round(2), table, check_names=False)
    assert got["cd_area_buildup_m2"] == pytest.approx(b["cd_area_m2"], rel=1e-12)
    assert got["cd0_buildup"] == pytest.approx(b["cd0"], rel=1e-12)
    assert got["cd0"] == pytest.approx(ns["AF0"].cd0, rel=1e-12)                 # with the allowance: the airframe's cd0
    assert got["cd0"] == pytest.approx(ns["SPEEDS"]["Cd0"], rel=1e-12)
    assert sum(got["parts_m2"].values()) + got["interference_m2"] == pytest.approx(got["cd_area_buildup_m2"], rel=1e-12)
    assert got["interference_m2"] == pytest.approx(0.10 * sum(got["parts_m2"].values()), rel=1e-12)
    assert got["cd_area_m2"] == pytest.approx(got["cd_area_buildup_m2"] + got["extra_cd_area_m2"], rel=1e-12)
    assert (got["speed_m_s"], got["extra_cd_area_m2"]) == (150.0, 0.0008)


# ------------------------------------------------------------------------------------------------ the workflows
@pytest.fixture
def no_cad(monkeypatch):
    """The workflows' CAD and solver steps as no-ops (the root's recording is what these tests look at)."""
    from assemblies.components import aguya_jet, aguya_wing, impeller, turbojet_parts, wheels
    monkeypatch.setattr(turbojet_parts, "make", lambda node, out: {})
    for mod, name in ((impeller, "speedline"), (wheels, "solve"), (aguya_jet, "solve"), (aguya_wing, "solve")):
        monkeypatch.setattr(mod, name, lambda node, *a, **kw: node)


def test_microjet_records_the_new_keys(tmp_path, no_cad, solvers):
    shutil.copy(DATA / "microjet.vida", tmp_path / "x.vida")
    kw = dict(fidelity="full", out=tmp_path / "out", vida_path=tmp_path / "x.vida", export_path=tmp_path / "microjet.json",
              progress=False)
    for _ in range(2):
        root = wm.run(**kw)
        assert root.child("parts").status() == "reused"
        engine = mj.Microjet(**root.results["engine"])
        assert root.results["design_point"] == cycle.design_point(engine)                  # the old keys as they were
        assert root.results["atmosphere_cases"] == wm.atmosphere_cases(engine)
        assert root.results["catalogue_classes"] == wm.catalogue_classes({k: mj.from_catalogue(k) for k in mj.CATALOGUE})
        assert root.results["catalogue_classes"][wm.DEFAULT_ENGINE] == root.results["catalogue_design_point"]
    assert not solvers.fea and not solvers.cfd
    r = results.load(tmp_path / "x_results.json")
    assert set(r[""]) >= {"engine", "calibration", "design_point", "catalogue_design_point", "map", "metal_temperatures_K",
                          "atmosphere_cases", "catalogue_classes", "catalogue_full_throttle"}
    t = r.table("", "atmosphere_cases")
    assert list(t.index) == list(wm.ATMOSPHERES) and "static_thrust_N" in t.columns
    assert list(r.table("", "catalogue_classes").index) == list(mj.CATALOGUE)
    curve = pd.DataFrame(r[""]["catalogue_full_throttle"]["100 N class"])
    assert len(curve) == len(cycle.V_MAP) and curve["thrust_N"].iloc[0] == pytest.approx(100.0, rel=1e-3)
    saved = vida.load(tmp_path / "x.vida")
    np.testing.assert_allclose(saved.results["catalogue_full_throttle"]["200 N class"]["thrust_N"],
                               root.results["catalogue_full_throttle"]["200 N class"]["thrust_N"])


def test_aguya_records_the_drag_buildup(tmp_path, no_cad, solvers):
    shutil.copy(DATA / "aguya.vida", tmp_path / "x.vida")
    kw = dict(fidelity="full", out=tmp_path / "out", vida_path=tmp_path / "x.vida", engine_vida=DATA / "microjet.vida",
              merlin_vida=DATA / "merlin.vida", propulsors_vida=DATA / "propulsors.vida",
              export_path=tmp_path / "aguya.json", progress=False)
    for _ in range(2):
        root = wa.run(**kw)
        air = root.child("airframe")
        assert air.status() == "reused"                                         # the airframe is not recorded on
        d = root.results["drag_buildup"]
        assert d["cd0"] == pytest.approx(air.results["speeds"]["cd0"], rel=1e-12)
        assert d["cd0"] == pytest.approx(air.results["airframe"]["cd0"], rel=1e-12)
        assert d["parts_m2"] == pytest.approx(wa.drag_buildup(air.results["geometry"])["parts_m2"], rel=1e-12)
        assert d["start_point"] == wa.drag_buildup(air.params["start_geometry"])
        assert "summary" in root.results
    assert not solvers.fea and not solvers.cfd
    r = results.load(tmp_path / "x_results.json")
    d = r[""]["drag_buildup"]
    assert set(d["parts_cm2"]) == {"wing", "fuselage", "nacelle", "pylon", "vtail"}
    assert d["parts_cm2"]["wing"] == pytest.approx(26.03, abs=0.01)                # the audit's numbers (design geometry)
    assert d["start_point"]["parts_cm2"]["fuselage"] == pytest.approx(12.98, abs=0.01)
