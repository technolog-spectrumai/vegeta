"""The ground step: the lifted road-wheel and leg snippets against the notebook cells they came from (the cell code is
read out of the .ipynb and run as it is), and the Onager, walkers and rover trees with the simulations and FEA off."""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from assemblies.components import leg, road_wheel as rw

NB = Path(__file__).resolve().parents[2] / "notebooks"
G = 9.81


def notebook_defs(notebook: str, cell: int, names, **globals_) -> dict:
    """The functions ``names`` exactly as notebook ``notebook`` cell ``cell`` defines them, over ``globals_``."""
    src = "".join(json.loads((NB / notebook).read_text())["cells"][cell]["source"])
    tree = ast.parse(src)
    ns = {"math": math, "np": np, **globals_}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), f"{notebook}:{cell}", "exec"), ns)
    missing = set(names) - set(ns)
    assert not missing, f"{notebook} cell {cell} has no {missing}"
    return ns


def _road(nb, length=12.0):
    x, z = nb["iso8608_profile"](length, 0.005, 4096e-6, 3)
    return x, nb["add_drop"](x, nb["add_rocks"](x, z, 0.04, 0.12, 2.5, seed=13), 8.0, 0.12)


def test_rover_road_and_quarter_car_are_notebook_11s():
    k = dict(K_SUSP=520.0, C_SUSP=2 * 0.3 * math.sqrt(520.0 * 0.45), K_TYRE=6000.0, M_SPRUNG_CORNER=0.45, M_WHEEL=0.07,
             M_TOTAL=2.08, G=G)
    nb = notebook_defs("11_rover_mechanics.ipynb", 8, ("iso8608_profile", "add_rocks", "add_drop", "quarter_car"), **k)
    x, z = _road(nb)
    x2, z2 = rw.iso8608_profile(12.0, 0.005, 4096e-6, 3)
    assert np.array_equal(x, x2)
    assert np.array_equal(z, rw.add_drop(x2, rw.add_rocks(x2, z2, 0.04, 0.12, 2.5, seed=13), 8.0, 0.12))
    t, zr, Fs, Ft, As, travel = nb["quarter_car"](x, z, 0.8)
    r = rw.quarter_car(x, z, 0.8, k_susp=k["K_SUSP"], c_susp=k["C_SUSP"], k_tyre=k["K_TYRE"], m_sprung=k["M_SPRUNG_CORNER"],
                       m_unsprung=k["M_WHEEL"], static_corner_N=k["M_TOTAL"] * G / 4)
    for a, b in zip((t, zr, Fs, Ft, As, travel), (r["t"], r["zr"], r["Fs"], r["Ft"], r["As"], r["travel"])):
        assert np.array_equal(a, b)


def test_onager_quarter_cars_are_notebooks_20_and_21s():
    k = dict(K_SUSP=31000.0, C_SUSP=900.0, K_TYRE=150e3, M_SPRUNG_CORNER=85.0, M_UNSPRUNG=17.0, M_TOTAL=408.0, G=G)
    nb20 = notebook_defs("20_onager_sentinel.ipynb", 17, ("iso8608_profile", "add_rocks", "add_drop", "quarter_car"), **k)
    x, z = _road(nb20)
    out = nb20["quarter_car"](x, z, 2.5)
    r = rw.quarter_car(x, z, 2.5, k_susp=k["K_SUSP"], c_susp=k["C_SUSP"], k_tyre=k["K_TYRE"], m_sprung=k["M_SPRUNG_CORNER"],
                       m_unsprung=k["M_UNSPRUNG"], static_corner_N=k["M_TOTAL"] * G / 4)
    for a, b in zip(out, (r["t"], r["zr"], r["Fs"], r["Ft"], r["As"], r["travel"], r["zs"])):
        assert np.array_equal(a, b)

    nb21 = notebook_defs("21_onager_atlas.ipynb", 18, ("iso8608_profile", "quarter_car"), **k)
    x, z = nb21["iso8608_profile"](10.0, 0.005, 256e-6, 5)
    assert np.array_equal(z, rw.iso8608_profile(10.0, 0.005, 256e-6, 5)[1])
    t, As, Ft = nb21["quarter_car"](x, z, 2.0, 160.0)
    r = rw.quarter_car(x, z, 2.0, k_susp=k["K_SUSP"], c_susp=k["C_SUSP"], k_tyre=k["K_TYRE"], m_sprung=160.0, m_unsprung=17.0, dt=2e-4)
    assert np.array_equal(t, r["t"]) and np.array_equal(As, r["As"]) and np.array_equal(Ft, r["Ft"])


def test_leg_hand_checks_are_the_notebooks():
    nb16 = notebook_defs("16_robot_dog.ipynb", 16, ("pin_bending", "pin_shear"))
    nb20 = notebook_defs("20_onager_sentinel.ipynb", 23, ("pin_bending", "pin_shear"))
    for F, lever, d in ((850.0, 6.5, 8.0), (14e3, 40.0, 30.0)):
        assert leg.pin_bending(F, lever, d) == nb16["pin_bending"](F, lever, d) == nb20["pin_bending"](F, lever, d)
        assert leg.pin_shear(F, d) == nb16["pin_shear"](F, d) == nb20["pin_shear"](F, d)
        assert leg.pin_shear(F, d, 1) == nb20["pin_shear"](F, d, 1)
    peak16 = notebook_defs("16_robot_dog.ipynb", 8, ("foot_peak",))["foot_peak"]
    for W, beta in ((128.2, 0.75), (177.3, 0.3)):
        assert leg.foot_peak(W, beta) == peak16(W, beta)

    from assemblies.components import gait, onager_robot as orb
    geo = orb.geometry()
    nb = notebook_defs("20_onager_sentinel.ipynb", 14, ("leg_torques", "wheel_peak"), gait=gait, L1=geo["L1"], L2=geo["L2"],
                       R_WHEEL=geo["r_wheel"], H_AXLE=geo["h_axle"])
    for x_axle, fx, fz in ((geo["axle_x"], 0.0, 1000.0), (geo["axle_x"] + 0.2, -300.0, 1800.0), (geo["axle_x"] - 0.25, 400.0, 900.0)):
        mine = leg.two_link_torques(x_axle, fx, fz, L1=geo["L1"], L2=geo["L2"], h_axle=geo["h_axle"], r_wheel=geo["r_wheel"])
        assert mine == nb["leg_torques"](x_axle, fx, fz)
    assert leg.foot_peak(4000.0, 0.6) == nb["wheel_peak"](4000.0, 0.6)


# ------------------------------------------------------------------------------------------------- the trees
pytest.importorskip("cadquery")

from assemblies.workflows import onager, rover, walkers  # noqa: E402


def test_onager_tree_masses_and_reuse(tmp_path):
    kw = dict(fidelity="smoke", run_sim=False, run_fea=False, out=tmp_path / "o", vida_path=tmp_path / "o.vida", progress=False)
    root = onager.run(**kw)
    for v, (module, _) in onager.VARIANTS.items():
        b = root.child(f"{v}/body")
        assert b.results["mass_kg"] == pytest.approx(sum(module.mass_budget().values()), rel=2e-3)
        assert b.results["cad_drift_max"] < 0.01
        assert root.child(f"{v}/scene").status() == "NOT RUN"
    assert root.child("sentinel/leg_fea").status() == "NOT RUN"
    again = onager.run(**kw)
    assert all(again.child(f"{v}/body").status() == "reused" for v in onager.VARIANTS)


def test_walkers_tree_and_dog_gaits(tmp_path):
    kw = dict(run_sim=False, vida_path=tmp_path / "w.vida")
    root = walkers.run(**kw)
    dog, cleo, aph = (root.child(f"{m}/body").results for m in ("dog", "cleopatra", "apheloria"))
    assert dog["mass_kg"] == pytest.approx(dog["robot_mass_kg"]) and dog["servos"] == 12
    assert cleo["mass_kg"] == pytest.approx(cleo["robot_mass_kg"]) and cleo["servos"] == 36
    assert aph["mass_kg"] == pytest.approx(33.116, abs=1e-3) and aph["servos"] == 104
    assert root.child("persephone/body").results["volume_mm3"] > 0
    g = root.child("dog/gaits").results["gaits"]
    assert g["trot"]["peak_foot_N"] == pytest.approx(math.pi / 2 * dog["mass_kg"] * G / 2)
    assert g["walk"]["peak_foot_loaded_N"] == pytest.approx(math.pi / 2 * (dog["mass_kg"] + 5.0) * G / 3)
    assert root.child("apheloria/pack").status() == "NOT RUN"
    again = walkers.run(**kw)
    assert all(again.child(f"{m}/body").status() == "reused" for m in walkers.MACHINES)


def test_rover_tree_without_fea(tmp_path):
    root = rover.run(fidelity="smoke", run_fea=False, out=tmp_path / "r", vida_path=tmp_path / "r.vida", export=False, progress=False)
    T = root.child("terrains").results
    assert T["peak_z_loaded"] >= T["summary"]["rocky field"]["tyre_force_max_N"] > 0
    assert root.child("arm_fea").status() == "NOT RUN" and root.child("chassis_fea").status() == "NOT RUN"
    assert set(root.results["pins"]) == set(root.child("arm_fea").params["cases"])
    assert all(math.isfinite(v["pin_SF"]) and v["pin_SF"] > 0 for v in root.results["pins"].values())


@pytest.mark.slow
@pytest.mark.requires_ccx
def test_ground_fea_for_real(tmp_path):
    if shutil.which("ccx") is None:
        pytest.skip("CalculiX not on PATH")
    o = onager.run(fidelity="smoke", variants=("sentinel",), run_sim=False, out=tmp_path / "o", vida_path=tmp_path / "o.vida",
                   progress=False)
    assert o.child("sentinel/leg_fea").results["complete"]
    r = rover.run(fidelity="smoke", out=tmp_path / "r", vida_path=tmp_path / "r.vida", export=False, progress=False)
    assert r.child("arm_fea").results["complete"] and r.child("chassis_fea").results["complete"]


@pytest.mark.slow
def test_apheloria_packs_in_mujoco(tmp_path):
    pytest.importorskip("mujoco")
    root = walkers.run(machines=("apheloria",), vida_path=tmp_path / "a.vida")
    pack = root.child("apheloria/pack").results
    assert all(abs(q - pack["target_body_joint_deg"]) < 5 for q in pack["final_body_joint_deg"])
