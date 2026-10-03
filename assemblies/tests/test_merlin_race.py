"""(Copied from notebooks/designs/tests with only the imports changed: the promoted copy must pass the same tests.)
The propulsor library (notebooks 25, 25b, 25c) and MERLIN's race on it (notebook 27): the mass models at notebook 26's units, a
library that survives its JSON round trip, the installation applied to a propeller map, multi-stage fans, the scaled fan in
CAD, the out-and-back race and the study.

Run: cd /home/user/vegeta && .venv/bin/python -m pytest -q notebooks/designs/tests/test_merlin_race.py   (~40 s)
"""
import numpy as np
import pytest

from assemblies.components import merlin_flight as mf
from assemblies.components import merlin_race as mr
from assemblies.components import propulsor_maps as pm

V = np.linspace(0.0, 90.0, 10)
SMALL_EDF = {"diameter_mm": (90.0,), "pitch_ratio": (1.78,), "exit_area_ratio": (0.9,), "duct_loss": (pm.CATALOGUE_DUCT_LOSS,),
             "stages": (1, 2)}
SMALL_PROP = {"diameter_in": (10.0,), "pitch_ratio": (1.0,), "blades": (2,)}


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    d = tmp_path_factory.mktemp("maps")
    files = [pm.save(pm.build_library("edf", SMALL_EDF, processes=2, edf_bound=None), d / "edf.json"),
             pm.save(pm.build_library("propellers", SMALL_PROP, processes=2), d / "props.json")]
    return pm.load(files)


def test_library_round_trip(lib):
    assert len(lib["by_id"]) == 3 and not [e for e in lib["entries"] if "error" in e]
    assert sorted(lib["kinds"]) == ["edf", "propellers"] and lib["meta"]["edf"]["kind"] == "edf"
    e = lib["by_id"]["edf-90-p1.78-e0.90-catalogue-s1"]
    assert e["thrust"].shape == (len(pm.V_TABLE), len(e["rpm"])) and np.all(np.diff(e["rpm"]) > 0)
    assert pm.find(lib, kind="edf", stages=2)[0]["stages"] == 2


def test_mass_models_reproduce_notebook_26(lib):
    e1, e2 = (lib["by_id"][f"edf-90-p1.78-e0.90-catalogue-s{n}"] for n in (1, 2))
    p = lib["by_id"]["prop-10x10-b2"]
    assert pm.unit_mass_kg(e1, 1900.0) == pytest.approx(mf.UNIT_MASS_KG["edf"], abs=1e-9)
    assert pm.unit_mass_kg(p, 1900.0) == pytest.approx(mf.UNIT_MASS_KG["pusher"], abs=1e-9)
    assert pm.unit_mass_kg(e2, 1900.0) == pytest.approx(pm.unit_mass_kg(e1, 1900.0) + 0.045)
    assert pm.unit_mass_kg(e1, 3500.0) > pm.unit_mass_kg(e1, 1900.0)


def test_installation_applies_to_the_map(lib):
    p = lib["by_id"]["prop-10x10-b2"]
    free = mf.Unit("free", p["V"], p["rpm"], p["thrust"], p["power"], 0.0, 1900.0)
    for layout in ("tractor", "pusher"):
        w, t = p["installation"][layout]["w"], p["installation"][layout]["t"]
        assert 0 < w < 0.1 and 0 < t < 0.1
        u = pm.unit(p, 1900.0, layout=layout)
        j = len(p["rpm"]) // 2
        assert u.thrust[8, j] == pytest.approx(np.interp(p["V"][8] * (1 - w), p["V"], p["thrust"][:, j]) * (1 - t), rel=1e-6)
    assert pm.unit(p, 1900.0, layout="pusher").full(20.0)[0] < free.full(20.0)[0] * 1.02
    with pytest.raises(ValueError):
        pm.unit(p, 1900.0)


def test_a_second_stage_does_not_help_a_power_limited_fan(lib):
    t = [mf.race(pm.merlin_airframe("edf", 1900.0, e), pm.unit(e, 1900.0), 10000.0)["time_to_fire_s"]
         for e in (lib["by_id"][f"edf-90-p1.78-e0.90-catalogue-s{n}"] for n in (1, 2))]
    assert t[1] > t[0]


@pytest.mark.parametrize("d", [70.0, 120.0])
def test_scaled_fan_builds(d):
    pytest.importorskip("cadquery")
    from assemblies.components import ducted_fan as df
    from assemblies.components import merlin as m
    p = m.Merlin().resolve(propulsion="edf", edf_diameter=d, edf_pitch=1.78 * d, edf_exit_area_ratio=0.8)
    g = m.Merlin().generate(**p)
    assert g.shape.isValid() and len(g.shape.Solids()) == 1
    hp = m.edf_housing_params(p)
    assert hp["hub_diameter"] / hp["diameter"] == pytest.approx(40.0 / 90.0)
    assert df.housing_geometry(hp)["exit_area_ratio"] == pytest.approx(0.8, rel=0.01)


def test_round_trip(lib):
    for e, layout in mr.candidates(lib):
        u = pm.unit(e, 1900.0, layout=None if layout == "edf" else layout)
        af = pm.merlin_airframe(layout, 1900.0, e)
        r5, r10 = mr.round_trip(af, u, 5000.0), mr.round_trip(af, u, 10000.0)
        assert r5["reachable"] and r10["reachable"] and r5["time_s"] < r10["time_s"]
        assert r5["energy_wh"] <= r5["budget_wh"] * (1 + 1e-6) and r5["dash_speed"] <= r5["top_speed"]
        reach = mf.race(af, u, 5000.0)["time_to_fire_s"]
        assert 1.6 * reach < r5["time_s"] < 2.6 * reach                    # about twice the reach, plus the turn


def test_study_and_winners(lib):
    table = mr.study(lib, powers=(1900.0, 3500.0))
    assert len(table) == 2 * 4 and set(table["kind"]) == {"edf", "tractor", "pusher"}
    w = mr.winners(table, "5 km and back [s]")
    assert set(w.index.get_level_values(0)) == {"edf", "tractor", "pusher"}
    edf = table[table["kind"] == "edf"].set_index(["id", "power [W]"])["5 km and back [s]"]
    assert edf[("edf-90-p1.78-e0.90-catalogue-s1", 3500.0)] < edf[("edf-90-p1.78-e0.90-catalogue-s1", 1900.0)]


def test_design_export_drives_the_airframe(lib):
    e = lib["by_id"]["prop-10x10-b2"]
    base = pm.merlin_airframe("pusher", 1900.0, e)
    design = {"merlin_params": {"thickness": 0.15}, "common_mass_kg": {"everything": 2.0}, "extra_cd_area_m2": 0.0015,
              "cd0_correction": {"pusher": 0.004}, "cl_max": 1.3, "oswald": 0.75}
    af = pm.merlin_airframe("pusher", 1900.0, e, design=design)
    assert af.mass_kg == pytest.approx(2.0 + pm.unit_mass_kg(e, 1900.0)) and af.cl_max == 1.3 and af.oswald == 0.75
    assert af.cd0 > base.cd0 + 0.004 - 1e-9                         # the correction, and a thicker wing's form factor
    assert mf.load_design("/nonexistent/merlin_design.json") is None
