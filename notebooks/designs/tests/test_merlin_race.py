"""MERLIN's propulsor race (notebook 27): the mass models at notebook 26's units, the scaled fan in CAD, the out-and-back
race, and one fan and one propeller through the study.

Run: cd /home/user/vegeta && .venv/bin/python -m pytest -q notebooks/designs/tests/test_merlin_race.py   (~30 s)
"""
from dataclasses import replace

import numpy as np
import pytest

import merlin_flight as mf
import merlin_race as mr

EDF90 = mr.Config("edf", 90.0, 160.0 / 90.0, 0.9, mr.CATALOGUE_DUCT_LOSS)
PUSHER10 = mr.Config("pusher", 10.0, 1.0)
V = np.linspace(0.0, 90.0, 10)


@pytest.fixture(scope="module")
def units():
    return {c: mr.build_unit(c, V=V, n_rpm=8) for c in (EDF90, PUSHER10)}


def test_mass_models_reproduce_notebook_26():
    assert mr.unit_mass_kg(EDF90, 1900.0) == pytest.approx(mf.UNIT_MASS_KG["edf"], abs=1e-9)
    assert mr.unit_mass_kg(PUSHER10, 1900.0) == pytest.approx(mf.UNIT_MASS_KG["pusher"], abs=1e-9)
    assert mr.unit_mass_kg(EDF90, 3500.0) > mr.unit_mass_kg(EDF90, 1900.0)
    assert mr.unit_mass_kg(replace(EDF90, diameter=120.0), 1900.0) > mr.unit_mass_kg(EDF90, 1900.0)


@pytest.mark.parametrize("d", [70.0, 120.0])
def test_scaled_fan_builds(d):
    pytest.importorskip("cadquery")
    import ducted_fan as df
    import merlin as m
    p = m.Merlin().resolve(edf_diameter=d, edf_pitch=1.78 * d, edf_exit_area_ratio=0.8)
    g = m.Merlin().generate(**p)
    assert g.shape.isValid() and len(g.shape.Solids()) == 1
    hp = m.edf_housing_params(p)
    assert hp["hub_diameter"] / hp["diameter"] == pytest.approx(40.0 / 90.0)
    assert df.housing_geometry(hp)["exit_area_ratio"] == pytest.approx(0.8, rel=0.01)


def test_round_trip(units):
    for cfg, (u, _) in units.items():
        unit = replace(u, max_electrical_w=1900.0, mass_kg=mr.unit_mass_kg(cfg, 1900.0))
        af = mr.airframe(cfg, 1900.0)
        r5, r10 = mr.round_trip(af, unit, 5000.0), mr.round_trip(af, unit, 10000.0)
        assert r5["reachable"] and r10["reachable"] and r5["time_s"] < r10["time_s"]
        assert r5["energy_wh"] <= r5["budget_wh"] * (1 + 1e-6) and r5["dash_speed"] <= r5["top_speed"]
        reach = mf.race(af, unit, 5000.0)["time_to_fire_s"]
        assert 1.6 * reach < r5["time_s"] < 2.6 * reach                    # about twice the reach, plus the turn


def test_more_power_is_faster_but_heavier(units):
    u, _ = units[EDF90]
    t = []
    for P in (1900.0, 3500.0):
        unit = replace(u, max_electrical_w=P, mass_kg=mr.unit_mass_kg(EDF90, P))
        t.append(mr.round_trip(mr.airframe(EDF90, P), unit, 5000.0)["time_s"])
    assert t[1] < t[0]


def test_study_table(units):
    table = mr.study(units, powers=(1900.0,))
    assert len(table) == 2 and set(table["kind"]) == {"edf", "pusher"}
    w = mr.winners(table, "5 km and back [s]", by=("kind",))
    assert set(w.index) == {"edf", "pusher"}
