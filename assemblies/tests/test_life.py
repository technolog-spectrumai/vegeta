"""The life lifts against the notebook cells they came from: the cell code is read out of the .ipynb and run as it is
(notebook 08 Part 3, 09b), over the same inputs; whatever reads solver files goes to the mocks (``stubs.py``)."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from vegeta import boreas, chronos, talos

from assemblies.components import fixed_wing_life as fl, life as lf, propeller as pr, quad_life as ql

NB = Path(__file__).resolve().parents[2] / "notebooks"
Q, F = "08_quadcopter.ipynb", "09b_fixed_wing_durability.ipynb"


def cell(notebook: str, index: int, names=None, **globals_) -> dict:
    """Run notebook ``notebook`` cell ``index`` over ``globals_``: the whole cell, or only its function definitions and
    assignments to ``names``; returns the namespace."""
    src = "".join(json.loads((NB / notebook).read_text())["cells"][index]["source"])
    tree = ast.parse(src)
    ns = {"math": math, "np": np, "pd": pd, "chronos": chronos, "talos": talos, "boreas": boreas, **globals_}
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


# ------------------------------------------------------------------------------------------------ the quadcopter
@pytest.fixture(scope="module")
def quad():
    from assemblies.workflows import quadcopter as wq
    prop, sec, system, pts = wq.drive_points(0.48)
    return SimpleNamespace(prop=prop, pts=pts, motor=pr.motor(wq.DRIVE["motor"]), PROP=ql.prop_points(prop, pts), wq=wq)


def test_quad_points_and_masses_are_cell_80(quad):
    ns = cell(Q, 80, ["PROP", "MOTOR_PROP_MASS_T", "STACK_MASS_T"], prop=quad.prop, motor=quad.motor, MOTORS=4,
              hover=quad.pts["hover"], cruise=quad.pts["cruise"], punch=quad.pts["full"],
              parts=pd.DataFrame({"mass_g": {**quad.wq.PARTS_G, "frame (printed, from CAD volume)": 61.5}}))
    assert ns["PROP"] == quad.PROP
    assert ql.masses_t(sum(quad.wq.PARTS_G.values()), quad.motor.mass_kg, quad.prop.mass_kg) == pytest.approx(
        (ns["MOTOR_PROP_MASS_T"], ns["STACK_MASS_T"]), rel=1e-12)


def test_quad_regions_are_cell_82():
    from assemblies.components.quad_frame import QuadFrame
    p = QuadFrame().resolve()
    assert ql.frame_regions(p) == cell(Q, 82, ["frame_regions"])["frame_regions"](p)


def test_quad_missions_are_cell_93(quad):
    ns = cell(Q, 93, ["unb", "SPOOL", "HOVER", "CRUISE", "missions"], PROP=quad.PROP, MAX_THRUST_N=8.0)
    assert ql.missions(quad.PROP, 8.0) == ns["missions"]


def test_quad_margins_are_cell_88(quad):
    st = lf.structure([140.0, 210.0, 380.0, 520.0])
    src = "".join(json.loads((NB / Q).read_text())["cells"][88]["source"])
    expr = ast.parse(src).body[-1].value                                    # the cell's table
    ns = {"structure": st, "lines": {k: v["rpm"] / 60 for k, v in quad.PROP.items()}, "pd": pd}
    table = eval(compile(ast.Expression(expr), "c88", "eval"), ns)
    assert pd.DataFrame(lf.margins(st, ns["lines"])).round(2).to_dict() == table.to_dict()      # the cell shows .round(2)


@pytest.fixture
def unit_and_spectra(tmp_path, quad, solvers):
    """Unit cases and spectra over the mocks (the solved results stubbed), as cells 90 and 95 leave them."""
    unit = {k: (talos.solve_models([_dummy(tmp_path, k)], [tmp_path / k])[0], load)
            for k, load in (("thrust", 8.0), ("unbalance", 1.0), ("landing", 40.0))}
    missions = ql.missions(quad.PROP, 8.0)
    spectra = lf.spectra(missions, lf.structure([140.0, 210.0, 380.0, 520.0]))
    return unit, missions, spectra


def _dummy(tmp_path, name):
    step = tmp_path / "x.step"
    step.write_text("never opened: the mocks answer\n")
    return talos.StructuralModel(step, "mm-N-MPa", talos.Material("m", 1000.0, 0.3, density=1e-9), [talos.SurfacesOnPlane("a", "x", 0)],
                                 [talos.FixedSupport("a")], [talos.Force("a", fz=1.0)], talos.MeshSettings(5.0), name=name)


def test_quad_fatigue_recheck_and_life_are_cells_97_to_104(unit_and_spectra, tmp_path):
    unit, missions, spectra = unit_and_spectra
    ns97 = cell(Q, 97, unit_cases=unit, spectra=spectra, missions=missions, RUNS=tmp_path,
                tqdm=lambda x, **k: x)
    fat, table = lf.fatigue(unit, spectra, missions, ql.CURVE, workdir=tmp_path)
    assert ql.CURVE == ns97["CURVE"] and table == ns97["life"].to_dict(orient="index")
    worst = lf.worst(table)
    assert worst == ns97["life"]["damage_per_mission"].astype(float).idxmax()                          # cell 98
    ns101 = cell(Q, 101, fatigue=fat, worst=worst, unit_results={k: r for k, (r, _) in unit.items()},
                 UNIT={k: (None, load) for k, (_, load) in unit.items()}, spectra=spectra,
                 PETG_CF=SimpleNamespace(yield_strength=45.0))
    rows = lf.static_recheck(fat[worst], unit, spectra, 45.0)
    assert rows == pd.DataFrame(ns101["rows"]).T.to_dict(orient="index")
    damage = {k: table[k]["damage_per_mission"] for k in table}
    hours = {k: m.duration_h for k, m in missions.items()}
    ns103 = cell(Q, 103, ["damage", "hours", "usage", "sim"], fatigue=fat, missions=missions)
    assert ns103["usage"] == ql.USAGE and lf.usage_life(damage, hours, ql.USAGE, 40000).hours_to_failure == ns103["sim"].hours_to_failure
    ns104 = cell(Q, 104, spectra=spectra, unit_cases=unit, CURVE=ql.CURVE, hours=hours, damage=damage)
    ours = {k: talos.assess_fatigue(unit, lf.balanced(sp).to_dict(), ql.CURVE).result.metrics["damage_per_pass"] for k, sp in spectra.items()}
    assert ours == ns104["balanced"] and ns104["mixes"] == ql.MIXES
    assert lf.life_hours(ours, hours, ql.MIXES["cruise-only"]) == ns104["life_hours"](ours, ql.MIXES["cruise-only"])


# ------------------------------------------------------------------------------------------------ the fixed wing
@pytest.fixture(scope="module")
def plane():
    from assemblies.workflows import fixed_wing as wf
    prop, sec, battery, pts, perf = wf.drive(0.98, 0.17, 6.0, 0.40, 0.070)
    return SimpleNamespace(prop=prop, pts=pts, perf=perf, PROP=fl.prop_points(prop, pts), motor=pr.motor(wf.DRIVE["motor"]), wf=wf)


def test_wing_prop_and_motor_are_cell_4():
    ns = cell(F, 4, ["d", "p", "prop", "motor"])
    spec = pr.get("9x6 electric")
    assert spec.model() == ns["prop"] and pr.motor("2212-920KV") == ns["motor"]


def test_wing_points_masses_and_lift_are_cell_6(plane):
    pt = lambda k: SimpleNamespace(rpm=plane.pts[k].rpm, thrust=plane.pts[k].thrust)                 # noqa: E731
    ns = cell(F, 6, ["PROP", "NACELLE_MASS_T", "W", "LIFT_UNIT_MPA", "AUW_KG", "WING_AREA_M2"], prop=plane.prop,
              motor=plane.motor, cruise=pt("cruise"), climb=pt("climb"), one_engine=pt("engine_out"), AUW_kg=0.98, S=0.17)
    assert ns["PROP"] == plane.PROP
    assert fl.nacelle_mass_t(plane.motor.mass_kg, plane.prop.mass_kg) == ns["NACELLE_MASS_T"]
    assert fl.lift_unit_mpa(0.98, 0.17) == ns["LIFT_UNIT_MPA"]


def test_wing_missions_are_cell_17(plane):
    ns = cell(F, 17, ["unb", "TC", "TCL", "missions"], PROP=plane.PROP)
    assert fl.missions(plane.PROP) == ns["missions"]


def test_wing_rate_and_nacelle_amplitude_are_cells_24_25(plane, tmp_path, solvers):
    damage, hours = {"survey": 1e-7, "patrol": 3e-7, "windy_hops": 5e-7}, {"survey": 0.75, "patrol": 0.5, "windy_hops": 0.58}
    ns24 = cell(F, 24, ["usage", "rate"], damage=damage, hours=hours)
    assert ns24["usage"] == fl.USAGE and lf.damage_rate(damage, hours, fl.USAGE) == ns24["rate"]
    vib = talos.solve_models([_dummy(tmp_path, "vib_left")], [tmp_path / "vib"])[0]
    st = lf.structure([60.0, 95.0, 150.0])
    ns25 = cell(F, 25, ["nacelle_amplitude_mm"], UNIT={"vib_left": (None, 1.0)}, PROP=plane.PROP)
    for s_ in (st, None):
        for k, v in plane.PROP.items():
            assert lf.nacelle_amplitude_mm(s_, vib, 1.0, v["rpm"], v["unbalance_N"]) == ns25["nacelle_amplitude_mm"](s_, {"vib_left": (vib, 1.0)}, k)


@pytest.fixture(scope="module")
def fuselage():
    from assemblies.components.fixed_wing import FixedWing
    return FixedWing().resolve(part="fuselage", thickness=fl.PREFERRED_THICKNESS)


def test_fuselage_regions_and_masses_are_cells_30_32_36(fuselage, tmp_path):
    pf = fuselage
    ns30 = cell(F, 30, ["r_f", "tt", "x_nose", "x_cyl_end", "xt", "tc", "NOSE_KG"], pf=pf,
                parts=pd.DataFrame({"mass_g": __import__("assemblies.workflows.fixed_wing", fromlist=["PARTS_G"]).PARTS_G}))
    ns32 = cell(F, 32, ["x_te", "FREGIONS", "FMASSES", "FMESH"], **{k: ns30[k] for k in ("r_f", "tt", "x_nose", "x_cyl_end", "xt", "tc", "NOSE_KG")},
                pf=pf)
    step = tmp_path / "f.step"
    step.write_text("never opened\n")
    base, unit = fl.fuselage_models(step, pf, ns30["NOSE_KG"])
    assert fl.nose_kg(__import__("assemblies.workflows.fixed_wing", fromlist=["PARTS_G"]).PARTS_G) == ns30["NOSE_KG"]
    assert list(base.regions) == ns32["FREGIONS"] and list(base.masses) == ns32["FMASSES"] and base.mesh_settings == ns32["FMESH"]
    assert [u[1] for u in unit.values()] == [1.0, 5.0, 3.0] and list(unit) == ["inertia", "tail_lift", "fin_side"]


def test_fuselage_mission_is_cell_38(fuselage, plane):
    W, S, rho, v, endurance = 0.98 * 9.81, 0.17, 1.2, 14.0, plane.perf["endurance_min"]
    ns = cell(F, 38, ["S_T", "S_F", "A_W", "A_T", "A_F", "V", "dn_gust", "tail_gust", "fin_gust", "TAIL_TRIM", "T_FLIGHT",
                      "U_L", "U_M", "GUST_S", "N_LIGHT", "N_SIDE", "N_MOD", "LEGS", "T_LEGS", "long_mission"],
              pf=fuselage, RHO=rho, V_CRUISE=v, S=S, W=W, endurance_min=endurance)
    mission, gusts = fl.long_mission(fuselage, W, S, rho, v, endurance)
    assert mission == ns["long_mission"] and gusts["tail_trim_N"] == ns["TAIL_TRIM"] and gusts["dn_moderate"] == ns["dn_gust"](5.0)
